"""YOLO TXT 格式导出器"""
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Tuple

from .base_exporter import BaseExporter
from src.gui.utils.yolo_label_format import format_yolo_floats, normalize_task

_TASK_KINDS = {
    "detect": ("bbox", "polygon"),
    "segment": ("polygon",),
    "pose": ("pose",),
    "obb": ("obb",),
}


def annotation_kind(ann: dict) -> str:
    kind = ann.get("kind")
    if kind in ("bbox", "polygon", "obb", "pose"):
        return kind
    if ann.get("obb"):
        return "obb"
    if ann.get("keypoints"):
        return "pose"
    polygon = ann.get("polygon")
    if polygon and len(polygon) >= 3:
        return "polygon"
    return "bbox"


class YOLOExporter(BaseExporter):
    """YOLO TXT 格式导出器

    输出格式:
        class_id center_x center_y width height
        (所有坐标归一化到 0-1)
    """

    def __init__(self, output_dir: str | Path):
        super().__init__(output_dir)

    def export(
        self,
        annotations: List[Dict],
        image_path: Path,
        output_path: Path
    ) -> bool:
        """导出单个标注为 YOLO 检测 TXT。"""
        try:
            self.export_yolo_task(annotations, output_path, "detect")
            return True
        except Exception as e:
            print(f"[YOLOExporter] Export error: {e}")
            return False

    def format_yolo_task(
        self,
        annotations: List[Dict],
        task: str,
        kpt_shape: tuple[int, int] | None = None,
    ) -> Tuple[str, Dict]:
        task_norm = normalize_task(task) or task
        allowed = _TASK_KINDS.get(task_norm, ())
        skipped: dict[str, int] = defaultdict(int)
        lines: List[str] = []
        written = 0
        for ann in annotations:
            if not ann.get("visible", True):
                continue
            kind = annotation_kind(ann)
            if kind not in allowed:
                skipped[kind] += 1
                continue
            line = self._format_one(ann, task_norm, kpt_shape)
            if line is None:
                skipped[kind] += 1
                continue
            lines.append(line)
            written += 1
        text = "\n".join(lines) + ("\n" if lines else "")
        return text, {"written": written, "skipped": dict(skipped)}

    def export_yolo_task(
        self,
        annotations: List[Dict],
        output_path: Path,
        task: str,
        kpt_shape: tuple[int, int] | None = None,
    ) -> Dict:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        text, report = self.format_yolo_task(annotations, task, kpt_shape)
        with open(output_path, "w", encoding="utf-8") as f:
            f.write(text)
        return report

    @staticmethod
    def _format_one(
        ann: Dict,
        task: str,
        kpt_shape: tuple[int, int] | None,
    ) -> str | None:
        class_id = ann["class_id"]
        if task == "detect":
            cx, cy, w, h = ann["bbox"]
            return f"{class_id} {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}"
        if task == "segment":
            polygon = ann.get("polygon") or []
            if len(polygon) < 3:
                return None
            coords = format_yolo_floats(coord for point in polygon for coord in point)
            return f"{class_id} {coords}"
        if task == "obb":
            obb = ann.get("obb") or ()
            if len(obb) != 4:
                return None
            coords = format_yolo_floats(coord for point in obb for coord in point)
            return f"{class_id} {coords}"
        if task == "pose":
            keypoints = ann.get("keypoints") or []
            bbox = ann["bbox"]
            if kpt_shape:
                k, d = kpt_shape
                if len(keypoints) != k:
                    return None
            else:
                d = 3
                if not keypoints:
                    return None
            parts = [
                str(class_id),
                f"{bbox[0]:.6f}",
                f"{bbox[1]:.6f}",
                f"{bbox[2]:.6f}",
                f"{bbox[3]:.6f}",
            ]
            for point in keypoints:
                parts.append(f"{float(point[0]):.6f}")
                parts.append(f"{float(point[1]):.6f}")
                if d == 3:
                    v = int(point[2]) if len(point) >= 3 else 2
                    parts.append(str(v))
            return " ".join(parts)
        return None

    def export_batch(
        self,
        data: List[Tuple[Path, List[Dict]]],
        output_dir: Path
    ) -> Tuple[int, int]:
        """批量导出 YOLO 格式"""
        output_dir = Path(output_dir)
        label_dir = output_dir / 'labels'
        label_dir.mkdir(parents=True, exist_ok=True)

        success = 0
        total = len(data)

        for image_path, annotations in data:
            label_path = label_dir / f"{image_path.stem}.txt"
            if self.export(annotations, image_path, label_path):
                success += 1

        self._generate_classes_file(output_dir)
        return success, total

    def export_current_frame(
        self,
        annotations: List[Dict],
        image_path: Path,
        output_dir: Path
    ) -> Path:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        label_path = output_dir / f"{image_path.stem}.txt"
        if not self.export(annotations, image_path, label_path):
            raise OSError(f"无法导出 YOLO 标注: {label_path}")

        self._generate_classes_file(output_dir)
        return label_path

    def export_yolo_seg(
        self,
        annotations: List[Dict],
        image_path: Path,
        output_path: Path,
    ) -> int:
        """导出单张图像的 YOLO 分割格式标签。不兼容 kind 会被跳过。"""
        output_path = Path(output_path)
        if output_path.is_dir():
            output_path = output_path / f"{image_path.stem}.txt"
        try:
            report = self.export_yolo_task(annotations, output_path, "segment")
        except OSError as e:
            raise OSError(f"无法写入 YOLO 分割标签: {output_path}") from e
        return report["written"]

    def _generate_classes_file(self, output_dir: Path):
        """生成 classes.txt 文件"""
        classes_path = output_dir / 'classes.txt'
        class_names = self.get_all_class_names()

        with open(classes_path, 'w', encoding='utf-8') as f:
            for name in class_names:
                f.write(f"{name}\n")
