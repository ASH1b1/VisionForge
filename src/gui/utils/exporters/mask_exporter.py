"""Mask 图像导出器 — 将 polygon 渲染为 PNG mask 图像"""
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import cv2
import numpy as np
from .base_exporter import BaseExporter


def _imwrite_safe(path: str, img: np.ndarray) -> bool:
    """安全写入 PNG，兼容中文路径，返回是否成功。"""
    try:
        success, buf = cv2.imencode(".png", img)
        if not success:
            print(f"[MaskExporter] cv2.imencode failed for: {path}")
            return False
        buf.tofile(path)
        return True
    except Exception as e:
        print(f"[MaskExporter] imwrite error for {path}: {e}")
        return False


class MaskExporter(BaseExporter):
    """Mask 图像导出器

    支持输出模式：
    - "color"  : 彩色实例 mask（单张 PNG）
    - "binary" : 每实例独立二值 PNG
    - "class"  : 语义分割 class mask（灰度图）
    - "overlay" : 原图 + mask 轮廓叠加
    """

    def __init__(self, output_dir: str | Path):
        super().__init__(output_dir)

    # ── 公开入口 ───────────────────────────────────────────────────────

    def export_single(
        self,
        annotations: List[Dict],
        image_path: Path,
        output_dir: Path,
        modes: Optional[List[str]] = None,
        overlay_alpha: float = 0.4,
        overlay_line_width: int = 2,
    ) -> Dict[str, Path]:
        """单帧导出。

        Args:
            annotations: 标注字典列表，每项含 class_id/bbox/polygon/visible
            image_path: 原始图像路径（用于 overlay 和获取尺寸）
            output_dir: 输出目录
            modes: 输出模式列表，默认 ["color"]
            overlay_alpha: overlay 混合透明度
            overlay_line_width: overlay 轮廓线宽度

        Returns:
            {"mode": output_path, ...} 导出路径映射（仅含写入成功的项）

        Raises:
            ValueError: 无法读取图片尺寸时
        """
        modes = modes or ["color"]
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        img_w, img_h = self._get_image_size(image_path)
        stem = image_path.stem
        results: Dict[str, Path] = {}

        from ..segmentation_utils import (
            polygon_to_mask,
            render_instance_mask_image,
            render_class_mask_image,
        )

        visible_anns = [a for a in annotations if a.get("visible", True)]

        for mode in modes:
            if mode == "color":
                img = render_instance_mask_image(visible_anns, img_w, img_h)
                out_path = output_dir / f"{stem}_mask_color.png"
                if _imwrite_safe(str(out_path), cv2.cvtColor(img, cv2.COLOR_RGB2BGR)):
                    results["color"] = out_path

            elif mode == "binary":
                binary_dir = output_dir / f"{stem}_masks"
                binary_dir.mkdir(parents=True, exist_ok=True)
                wrote_any = False
                for i, ann in enumerate(visible_anns):
                    polygon = ann.get("polygon")
                    if not polygon:
                        continue
                    binary = polygon_to_mask(polygon, img_w, img_h)
                    if binary is not None:
                        class_name = self.get_class_name(ann.get("class_id", 0))
                        safe_name = (
                            str(class_name)
                            .replace("\\", "_")
                            .replace("/", "_")
                            .replace("..", "_")
                            .replace(":", "_")
                            .replace("*", "_")
                            .replace("?", "_")
                            .replace('"', "_")
                            .replace("<", "_")
                            .replace(">", "_")
                            .replace("|", "_")
                        )
                        out_path = binary_dir / f"{stem}_{safe_name}_{i}.png"
                        if _imwrite_safe(str(out_path), binary):
                            wrote_any = True
                if wrote_any:
                    results["binary"] = binary_dir

            elif mode == "class":
                img = render_class_mask_image(visible_anns, img_w, img_h)
                out_path = output_dir / f"{stem}_mask_class.png"
                if _imwrite_safe(str(out_path), img):
                    results["class"] = out_path

            elif mode == "overlay":
                out_path = output_dir / f"{stem}_mask_overlay.png"
                if self._render_overlay(
                    image_path, visible_anns, img_w, img_h,
                    overlay_alpha, overlay_line_width, out_path,
                ):
                    results["overlay"] = out_path

        return results

    def export_batch(
        self,
        data: List[Tuple[Path, List[Dict]]],
        output_dir: Path,
        modes: Optional[List[str]] = None,
        split_map: Optional[Dict[int, str]] = None,
        overlay_alpha: float = 0.4,
        overlay_line_width: int = 2,
    ) -> Tuple[int, int, Dict[str, int]]:
        """批量导出。

        Args:
            data: [(img_path, annotations), ...]
            output_dir: 输出根目录
            modes: 输出模式
            split_map: {frame_index: "train"/"val"/"test"}
            overlay_alpha / overlay_line_width: overlay 参数

        Returns:
            (成功数, 总数, split_counts)
        """
        output_dir = Path(output_dir)
        modes = modes or ["color"]
        split_map = split_map or {}

        success = 0
        split_counts: Dict[str, int] = {"train": 0, "val": 0, "test": 0}

        if not split_map:
            for img_path, anns in data:
                try:
                    results = self.export_single(
                        anns, img_path, output_dir, modes,
                        overlay_alpha, overlay_line_width,
                    )
                except Exception as e:
                    print(f"[MaskExporter] Skipping {img_path}: {e}")
                    continue
                if results:
                    success += 1
                    split_counts["train"] += 1
            return success, len(data), split_counts

        for idx, (img_path, anns) in enumerate(data):
            split = split_map.get(idx, "train")
            split_dir = output_dir / split
            try:
                results = self.export_single(
                    anns, img_path, split_dir, modes,
                    overlay_alpha, overlay_line_width,
                )
            except Exception as e:
                print(f"[MaskExporter] Skipping {img_path}: {e}")
                continue
            if results:
                success += 1
                split_counts[split] = split_counts.get(split, 0) + 1

        return success, len(data), split_counts

    # ── BaseExporter abstract methods ─────────────────────────────────

    def export(
        self,
        annotations: List[Dict],
        image_path: Path,
        output_path: Path,
    ) -> bool:
        """导出单帧 mask（color 模式），兼容 BaseExporter 接口。"""
        try:
            results = self.export_single(
                annotations, image_path, output_path, modes=["color"]
            )
            return bool(results)
        except Exception as e:
            print(f"[MaskExporter] Export error: {e}")
            return False

    # ── 内部辅助 ───────────────────────────────────────────────────────

    @classmethod
    def _get_image_size(cls, image_path: Path) -> Tuple[int, int]:
        """获取图片尺寸；失败时抛出 ValueError，绝不伪造 1920×1080。"""
        try:
            return cls.read_image_size(image_path)
        except Exception as e:
            raise ValueError(f"Cannot read image size: {image_path}") from e

    def _render_overlay(
        self,
        image_path: Path,
        annotations: List[Dict],
        img_w: int,
        img_h: int,
        alpha: float,
        line_width: int,
        output_path: Path,
    ) -> bool:
        """渲染原图 + mask 轮廓叠加。返回写入是否成功。"""
        from ..segmentation_utils import polygon_to_mask

        try:
            img_array = np.fromfile(str(image_path), dtype=np.uint8)
            base = cv2.imdecode(img_array, cv2.IMREAD_COLOR)
            if base is None:
                base = cv2.imread(str(image_path))
        except Exception:
            base = None

        if base is None:
            print(f"[MaskExporter] Cannot read image for overlay: {image_path}")
            return False

        base = cv2.resize(base, (img_w, img_h))

        overlay = base.copy()
        for ann in annotations:
            polygon = ann.get("polygon")
            if not polygon:
                continue
            binary = polygon_to_mask(polygon, img_w, img_h)
            if binary is not None:
                contours, _ = cv2.findContours(
                    binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
                )
                class_id = ann.get("class_id", 0)
                color = self._get_bgr_color(class_id)
                cv2.drawContours(overlay, contours, -1, color, line_width)

        result = cv2.addWeighted(base, 1.0 - alpha, overlay, alpha, 0)
        return _imwrite_safe(str(output_path), result)

    @staticmethod
    def _get_bgr_color(class_id: int) -> Tuple[int, int, int]:
        """class_id → BGR 颜色（OpenCV 格式）"""
        COLORS = [
            (0, 0, 255), (0, 255, 0), (255, 0, 0),
            (0, 255, 255), (255, 0, 255), (255, 255, 0),
            (0, 128, 255), (255, 0, 128), (128, 0, 255),
            (255, 128, 0),
        ]
        cid = class_id if isinstance(class_id, int) else 0
        return COLORS[cid % len(COLORS)]
