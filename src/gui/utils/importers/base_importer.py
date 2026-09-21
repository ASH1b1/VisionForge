# src/gui/utils/importers/base_importer.py
"""导入器基类"""
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Dict, Tuple, Optional
from abc import ABC, abstractmethod
import sys
if not getattr(sys, "frozen", False):
    sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent.parent))
from src.gui.utils.exporters.base_exporter import BaseExporter


@dataclass
class ImportResult:
    """导入结果汇总"""
    success: bool = True
    total_images: int = 0
    total_annotations: int = 0
    matched_pairs: int = 0          # 成功匹配的图片-标注对
    unmatched_images: int = 0        # 有图片无标注
    unmatched_labels: int = 0        # 有标注无图片
    parse_errors: List[str] = field(default_factory=list)
    class_names: Dict[int, str] = field(default_factory=dict)
    split_map: Dict[int, str] = field(default_factory=dict)
    warnings: List[str] = field(default_factory=list)
    detected_task: Optional[str] = None
    kpt_shape: Optional[tuple] = None
    kpt_names: Optional[List[str]] = None
    kpt_shape_source: Optional[str] = None  # yaml | inferred
    needs_task_choice: bool = False
    skipped_incompatible: int = 0

    def summary(self) -> str:
        """生成人类可读的汇总信息"""
        lines = [
            f"图片总数: {self.total_images}",
            f"标注总数: {self.total_annotations}",
            f"成功匹配: {self.matched_pairs} 对",
        ]
        if self.detected_task:
            lines.append(f"判定任务: {self.detected_task}")
        if self.kpt_shape:
            source = f"（{self.kpt_shape_source}）" if self.kpt_shape_source else ""
            lines.append(f"kpt_shape: {list(self.kpt_shape)}{source}")
        if self.needs_task_choice:
            lines.append("无法从 data.yaml 判定格式，请手动选择检测/分割/姿态/OBB")
        if self.skipped_incompatible:
            lines.append(f"因关键点数冲突跳过: {self.skipped_incompatible} 条")
        if self.unmatched_images:
            lines.append(f"无标注图片: {self.unmatched_images} 张")
        if self.unmatched_labels:
            lines.append(f"无匹配标注: {self.unmatched_labels} 个")
        if self.class_names:
            names = [self.class_names[k] for k in sorted(self.class_names)]
            lines.append(f"类别 ({len(names)}): {names}")
        if self.split_map:
            from collections import Counter
            counts = Counter(self.split_map.values())
            parts = [f"{k} {v}" for k, v in sorted(counts.items())]
            lines.append(f"划分: {' / '.join(parts)}")
        if self.parse_errors:
            lines.append(f"\n解析错误 ({len(self.parse_errors)}):")
            for err in self.parse_errors[:5]:
                lines.append(f"  - {err}")
            if len(self.parse_errors) > 5:
                lines.append(f"  ... 及其他 {len(self.parse_errors) - 5} 个错误")
        if self.warnings:
            lines.append(f"警告 ({len(self.warnings)}):")
            for warn in self.warnings[:5]:
                lines.append(f"  - {warn}")
        return "\n".join(lines)


class BaseImporter(BaseExporter, ABC):
    """标注导入器基类，继承 BaseExporter 以复用坐标转换方法

    BaseExporter.__init__ 的 output_dir 已改为可选参数（默认 None），
    导入器调用 super().__init__() 正常。
    """

    def __init__(self):
        super().__init__()  # output_dir=None
        self._class_names: Dict[int, str] = {}
        self._next_ann_id: int = 1       # 导入标注的全局 ID 计数器

    def set_classes(self, class_names: Dict[int, str]):
        self._class_names = class_names

    def _register_class(self, class_name: str, class_names: Dict[int, str],
                        name_to_id: Dict[str, int]) -> int:
        """注册类名，返回内部 class_id。

        - 若类名已存在，返回已有 ID
        - 若类名不存在，分配新 ID（从已有最大 ID+1 开始，min 1）
        """
        key = class_name.strip().casefold()
        if key in name_to_id:
            return name_to_id[key]

        if class_names:
            new_id = max(class_names.keys()) + 1
        else:
            new_id = 1

        class_names[new_id] = class_name.strip()
        name_to_id[key] = new_id
        return new_id

    def _voc_to_yolo_normalized(self, xmin: int, ymin: int, xmax: int, ymax: int,
                                 img_w: int, img_h: int) -> Tuple[float, float, float, float]:
        """VOC 像素坐标 → 归一化 YOLO 格式 (cx, cy, w, h)"""
        w = (xmax - xmin) / img_w
        h = (ymax - ymin) / img_h
        cx = ((xmin + xmax) / 2) / img_w
        cy = ((ymin + ymax) / 2) / img_h
        return (cx, cy, w, h)

    def _coco_to_yolo_normalized(self, bbox: List[float],
                                  img_w: int, img_h: int) -> Tuple[float, float, float, float]:
        """COCO bbox [x, y, w, h] 像素坐标 → 归一化 YOLO 格式"""
        x, y, w, h = bbox
        cx = (x + w / 2) / img_w
        cy = (y + h / 2) / img_h
        nw = w / img_w
        nh = h / img_h
        return (cx, cy, nw, nh)

    def _normalize_polygon(self, polygon: List[float],
                            img_w: int, img_h: int) -> List[Tuple[float, float]]:
        """像素坐标 polygon（扁平列表 [x1,y1,x2,y2,...]）→ 归一化 [(x,y),...]"""
        result = []
        for i in range(0, len(polygon), 2):
            if i + 1 < len(polygon):
                x = polygon[i] / img_w
                y = polygon[i + 1] / img_h
                result.append((max(0.0, min(x, 1.0)), max(0.0, min(y, 1.0))))
        return result

    def _make_annotation(self, class_id: int, bbox: Tuple[float, float, float, float],
                         polygon: Optional[List[Tuple[float, float]]] = None,
                         *,
                         kind: Optional[str] = None,
                         obb: Optional[tuple] = None,
                         keypoints: Optional[List[Tuple[float, float, int]]] = None) -> dict:
        """创建标注字典（含全局唯一 id，供 _load_frame_annotations 使用）"""
        if kind is None:
            if obb:
                kind = "obb"
            elif keypoints:
                kind = "pose"
            elif polygon and len(polygon) >= 3:
                kind = "polygon"
            else:
                kind = "bbox"
        ann = {
            'id': self._next_ann_id,
            'class_id': class_id,
            'bbox': bbox,
            'polygon': polygon,
            'confidence': 1.0,
            'visible': True,
            'kind': kind,
            'obb': obb,
            'keypoints': keypoints,
        }
        self._next_ann_id += 1
        return ann

    @abstractmethod
    def import_dataset(self, image_dir: Path, label_dir: Path) -> Tuple[
        List[Path],                      # image_files (排序后)
        Dict[int, List[dict]],           # frame_annotations: frame_index → [annotation_dict, ...]
        Dict[int, str],                  # class_names: class_id → class_name
        Dict[int, str],                  # split_map: frame_index → "train"/"val"/"test"
        ImportResult                     # 导入结果汇总
    ]:
        """导入数据集

        Args:
            image_dir: 图片根目录
            label_dir: 标注根目录

        Returns:
            (image_files, frame_annotations, class_names, split_map, result)
        """
        pass
