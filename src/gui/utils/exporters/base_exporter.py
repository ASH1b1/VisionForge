"""导出器基类"""
from abc import ABC, abstractmethod
from collections import Counter, defaultdict
from pathlib import Path
from typing import List, Dict, Tuple


class BaseExporter(ABC):
    """标注导出器基类"""

    def __init__(self, output_dir: str | Path | None = None):
        self.output_dir = Path(output_dir) if output_dir is not None else None
        self._class_names: Dict[int, str] = {}

    def set_classes(self, class_names: Dict[int, str]):
        """设置类别名称映射"""
        self._class_names = class_names

    def get_class_name(self, class_id: int) -> str:
        """获取类别名称"""
        return self._class_names.get(class_id, f"class_{class_id}")

    def get_all_class_names(self) -> List[str]:
        """获取所有类别名称（按ID排序）"""
        if not self._class_names:
            return []
        return [self.get_class_name(cid) for cid in sorted(self._class_names.keys())]

    @staticmethod
    def unique_image_names(image_paths: List[Path]) -> List[str]:
        """Return deterministic destination names without basename collisions."""
        counts = Counter(path.name.casefold() for path in image_paths)
        occurrences = defaultdict(int)
        used = set()
        result = []
        for path in image_paths:
            key = path.name.casefold()
            if counts[key] == 1 and key not in used:
                candidate = path.name
            else:
                occurrences[key] += 1
                candidate = f"{path.stem}__{occurrences[key]}{path.suffix}"
                while candidate.casefold() in used:
                    occurrences[key] += 1
                    candidate = f"{path.stem}__{occurrences[key]}{path.suffix}"
            used.add(candidate.casefold())
            result.append(candidate)
        return result

    @abstractmethod
    def export(
        self,
        annotations: List[Dict],
        image_path: Path,
        output_path: Path
    ) -> bool:
        """
        导出单个标注

        Args:
            annotations: 标注列表，每个元素包含 bbox, class_id 等
            image_path: 图片路径
            output_path: 输出文件路径

        Returns:
            是否导出成功
        """
        pass

    @abstractmethod
    def export_batch(
        self,
        data: List[Tuple[Path, List[Dict]]],
        output_dir: Path
    ) -> Tuple[int, int]:
        """
        批量导出标注

        Args:
            data: (image_path, annotations) 元组列表
            output_dir: 输出目录

        Returns:
            (成功数, 总数)
        """
        pass

    def create_output_structure(self, output_dir: Path, copy_images: bool = True) -> Dict[str, Path]:
        """
        创建输出目录结构

        Returns:
            包含各目录路径的字典
        """
        structure = {
            'root': output_dir,
            'images': output_dir / 'images',
            'annotations': output_dir / 'annotations',
        }

        for key, path in structure.items():
            if key != 'root':
                path.mkdir(parents=True, exist_ok=True)

        if copy_images:
            structure['images_copy'] = output_dir / 'images'
            structure['images_copy'].mkdir(parents=True, exist_ok=True)

        return structure

    def yolo_to_corners(self, bbox: Tuple[float, float, float, float]) -> Tuple[float, float, float, float]:
        """
        将 YOLO 格式 (center_x, center_y, width, height) 转换为角点坐标 (x1, y1, x2, y2)
        """
        cx, cy, w, h = bbox
        return (cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2)

    def corners_to_yolo(self, corners: Tuple[float, float, float, float]) -> Tuple[float, float, float, float]:
        """
        将角点坐标 (x1, y1, x2, y2) 转换为 YOLO 格式 (center_x, center_y, width, height)
        """
        x1, y1, x2, y2 = corners
        cx = (x1 + x2) / 2
        cy = (y1 + y2) / 2
        w = x2 - x1
        h = y2 - y1
        return (cx, cy, w, h)

    @staticmethod
    def read_image_size(image_path: Path) -> Tuple[int, int]:
        """Read (width, height) from an image file. Raises on failure — never invents dimensions."""
        from PIL import Image

        with Image.open(image_path) as img:
            return img.size
