"""标注数据处理工具模块 - 统一坐标转换、验证和NMS处理"""

from typing import List, Tuple, Optional, Dict, Any
from dataclasses import dataclass
import logging

logger = logging.getLogger(__name__)

# ============================================================================
# 数据结构定义
# ============================================================================

@dataclass
class YOLOAnnotation:
    """YOLO格式标注数据"""
    class_name: str
    cx: float  # 归一化中心点X (0-1)
    cy: float  # 归一化中心点Y (0-1)
    w: float   # 归一化宽度 (0-1)
    h: float   # 归一化高度 (0-1)
    score: float = 1.0  # 置信度

    def to_tuple(self) -> Tuple[str, Tuple[float, float, float, float], float]:
        """转换为旧代码兼容的元组格式"""
        return (self.class_name, (self.cx, self.cy, self.w, self.h), self.score)


@dataclass
class DetectionBox:
    """检测框原始数据"""
    x1: float  # 左上角X
    y1: float  # 左上角Y
    x2: float  # 右下角X
    y2: float  # 右下角Y
    label: Any
    score: float

    @property
    def width(self) -> float:
        return self.x2 - self.x1

    @property
    def height(self) -> float:
        return self.y2 - self.y1


# ============================================================================
# 坐标转换函数
# ============================================================================

def is_normalized_coords(x1: float, y1: float, x2: float, y2: float,
                          epsilon: float = 1e-4) -> bool:
    """
    检测坐标是否为归一化格式

    Args:
        x1, y1, x2, y2: 框坐标
        epsilon: 判定阈值，小于此值视为归一化

    Returns:
        True: 归一化坐标 (0-1范围)
        False: 像素坐标
    """
    has_negative = any(v < -epsilon for v in (x1, y1, x2, y2))
    max_val = max(abs(x1), abs(y1), abs(x2), abs(y2))

    # Bug 11: 负值坐标（即使绝对值小）也视为像素坐标
    if has_negative:
        return False

    # 空框/零框视为归一化但由调用方进一步过滤
    if max_val < epsilon:
        return True

    # 最大值在 [epsilon, 1.0] 范围内 → 归一化坐标
    if max_val <= 1.0:
        return True

    # 最大值轻微超出 1.0（浮点误差容差）→ 仍视为归一化
    if max_val <= 1.0 + epsilon:
        return True

    # 最大值 > 1.0 + epsilon → 像素坐标
    return False


def xyxy_to_yolo(x1: float, y1: float, x2: float, y2: float,
                 img_width: int, img_height: int,
                 is_normalized: bool = False) -> Tuple[float, float, float, float]:
    """
    将 XYXY 格式转换为 YOLO 中心点格式

    Args:
        x1, y1, x2, y2: XYXY格式坐标
        img_width, img_height: 图片尺寸
        is_normalized: 输入坐标是否为归一化格式

    Returns:
        (cx, cy, w, h) YOLO格式（归一化0-1）
    """
    if is_normalized:
        # 输入已经是归一化坐标
        cx = (x1 + x2) / 2
        cy = (y1 + y2) / 2
        w = x2 - x1
        h = y2 - y1
    else:
        # 输入是像素坐标，需要归一化
        cx = ((x1 + x2) / 2) / img_width
        cy = ((y1 + y2) / 2) / img_height
        w = (x2 - x1) / img_width
        h = (y2 - y1) / img_height

    return (cx, cy, w, h)


def yolo_to_xyxy(cx: float, cy: float, w: float, h: float,
                 img_width: int, img_height: int) -> Tuple[float, float, float, float]:
    """
    将 YOLO 中心点格式转换为 XYXY 像素坐标

    Args:
        cx, cy, w, h: YOLO格式（归一化0-1）
        img_width, img_height: 图片尺寸

    Returns:
        (x1, y1, x2, y2) XYXY像素坐标
    """
    x1 = (cx - w / 2) * img_width
    y1 = (cy - h / 2) * img_height
    x2 = (cx + w / 2) * img_width
    y2 = (cy + h / 2) * img_height

    return (x1, y1, x2, y2)


# ============================================================================
# YOLO标注验证函数
# ============================================================================

class YOLOAnnotationValidator:
    """YOLO格式标注验证器"""

    # 默认阈值
    DEFAULT_MAX_AREA = 0.90      # 最大面积比例
    DEFAULT_MIN_AREA = 0.00001   # 最小面积比例
    DEFAULT_MAX_ASPECT = 20.0    # 最大宽高比
    DEFAULT_MIN_ASPECT = 0.05    # 最小宽高比

    def __init__(
        self,
        max_area: float = DEFAULT_MAX_AREA,
        min_area: float = DEFAULT_MIN_AREA,
        max_aspect: float = DEFAULT_MAX_ASPECT,
        min_aspect: float = DEFAULT_MIN_ASPECT
    ):
        self.max_area = max_area
        self.min_area = min_area
        self.max_aspect = max_aspect
        self.min_aspect = min_aspect

    def validate(self, annotation: YOLOAnnotation) -> Tuple[bool, str]:
        """
        验证标注框的合理性

        Returns:
            (is_valid, reason) - 是否有效及原因
        """
        cx, cy, w, h = annotation.cx, annotation.cy, annotation.w, annotation.h
        class_name = annotation.class_name

        # 边界检查
        if not (0 <= cx <= 1 and 0 <= cy <= 1 and 0 < w <= 1 and 0 < h <= 1):
            return False, f"坐标越界: cx={cx:.3f}, cy={cy:.3f}, w={w:.3f}, h={h:.3f}"

        # 面积检查 - 最大
        area = w * h
        if area > self.max_area:
            return False, f"面积过大({area:.1%})"

        # 面积检查 - 最小
        if area < self.min_area:
            return False, f"面积过小({area:.4%})"

        # 宽高比检查
        aspect = w / h if h > 0 else float('inf')
        if aspect > self.max_aspect or aspect < self.min_aspect:
            return False, f"宽高比异常({aspect:.1f})"

        return True, "通过"

    def filter_annotations(self, annotations: List[YOLOAnnotation]) -> List[YOLOAnnotation]:
        """过滤无效标注"""
        valid = []
        for ann in annotations:
            is_valid, reason = self.validate(ann)
            if is_valid:
                valid.append(ann)
            else:
                logger.debug(f"[Filter] {ann.class_name}: {reason}")
        return valid


# ============================================================================
# NMS (非极大值抑制)
# ============================================================================

def compute_iou_yolo(bbox1: Tuple[float, float, float, float],
                     bbox2: Tuple[float, float, float, float]) -> float:
    """
    计算两个YOLO格式框的IoU

    Args:
        bbox1, bbox2: (cx, cy, w, h) YOLO格式

    Returns:
        IoU 值 (0-1)
    """
    cx1, cy1, w1, h1 = bbox1
    cx2, cy2, w2, h2 = bbox2

    # 转换为角点坐标
    x1_min = cx1 - w1 / 2
    y1_min = cy1 - h1 / 2
    x1_max = cx1 + w1 / 2
    y1_max = cy1 + h1 / 2

    x2_min = cx2 - w2 / 2
    y2_min = cy2 - h2 / 2
    x2_max = cx2 + w2 / 2
    y2_max = cy2 + h2 / 2

    # 计算交集
    inter_x_min = max(x1_min, x2_min)
    inter_y_min = max(y1_min, y2_min)
    inter_x_max = min(x1_max, x2_max)
    inter_y_max = min(y1_max, y2_max)

    inter_area = max(0, inter_x_max - inter_x_min) * max(0, inter_y_max - inter_y_min)

    # 计算并集（YOLO格式已经是归一化面积）
    area1 = w1 * h1
    area2 = w2 * h2
    union_area = area1 + area2 - inter_area

    return inter_area / union_area if union_area > 0 else 0


def apply_nms(annotations: List[YOLOAnnotation],
              iou_threshold: float = 0.5) -> List[YOLOAnnotation]:
    """
    对标注应用非极大值抑制

    Args:
        annotations: 标注列表
        iou_threshold: IoU阈值，超过此值则过滤

    Returns:
        过滤后的标注列表
    """
    if not annotations:
        return annotations

    # 按类别分组
    from collections import defaultdict
    class_groups: Dict[str, List[YOLOAnnotation]] = defaultdict(list)
    for ann in annotations:
        class_groups[ann.class_name].append(ann)

    result = []

    for class_name, group in class_groups.items():
        if len(group) == 1:
            result.extend(group)
            continue

        # 按置信度降序排列
        group = sorted(group, key=lambda x: (x.score, x.w * x.h), reverse=True)

        keep: List[YOLOAnnotation] = []
        while group:
            current = group.pop(0)
            keep.append(current)

            # 移除IoU超过阈值的框
            remaining = []
            current_tuple = (current.cx, current.cy, current.w, current.h)

            for ann in group:
                ann_tuple = (ann.cx, ann.cy, ann.w, ann.h)
                iou = compute_iou_yolo(current_tuple, ann_tuple)
                if iou < iou_threshold:
                    remaining.append(ann)

            group = remaining

        result.extend(keep)

    return result


# ============================================================================
# 标注处理流程
# ============================================================================

class AnnotationProcessor:
    """
    统一的标注处理流程

    处理流程:
    1. 检测框 → YOLO格式转换
    2. 标签映射
    3. NMS过滤
    4. 质量验证
    """

    def __init__(
        self,
        class_names: List[str],
        img_width: int,
        img_height: int,
        nms_threshold: float = 0.5,
        validator: Optional[YOLOAnnotationValidator] = None
    ):
        self.class_names = [name.strip().lower() for name in class_names]
        self.img_width = img_width
        self.img_height = img_height
        self.nms_threshold = nms_threshold
        self.validator = validator or YOLOAnnotationValidator()

        # 构建类别映射
        self.class_mapping = {name: idx for idx, name in enumerate(self.class_names)}

    def _map_label(self, raw_label: Any) -> Optional[str]:
        """映射标签到类别名称"""
        if isinstance(raw_label, str):
            label_lower = raw_label.lower().strip()
            # 精确匹配
            if label_lower in self.class_mapping:
                return self.class_names[self.class_mapping[label_lower]]
            # 模糊匹配
            for class_name in self.class_names:
                if class_name in label_lower or label_lower in class_name:
                    return class_name
            return label_lower  # 返回原始标签
        elif isinstance(raw_label, int):
            # 数字索引
            if 0 <= raw_label < len(self.class_names):
                return self.class_names[raw_label]
            return f"class_{raw_label}"
        else:
            return str(raw_label)

    def process_detection(
        self,
        boxes: List[Tuple[float, float, float, float]],
        labels: List[Any],
        scores: List[float]
    ) -> List[YOLOAnnotation]:
        """
        处理检测结果

        Args:
            boxes: 检测框列表 [(x1, y1, x2, y2), ...]
            labels: 标签列表
            scores: 置信度列表

        Returns:
            处理后的YOLO标注列表
        """
        raw_annotations: List[YOLOAnnotation] = []

        for box, raw_label, score in zip(boxes, labels, scores):
            x1, y1, x2, y2 = box

            # 检测是否为归一化坐标
            is_norm = is_normalized_coords(x1, y1, x2, y2)

            # 转换为YOLO格式
            cx, cy, w, h = xyxy_to_yolo(
                x1, y1, x2, y2,
                self.img_width, self.img_height,
                is_normalized=is_norm
            )

            # 映射标签
            class_name = self._map_label(raw_label)
            if class_name is None:
                continue  # 跳过无法映射的标签

            raw_annotations.append(YOLOAnnotation(
                class_name=class_name,
                cx=cx, cy=cy, w=w, h=h,
                score=score
            ))

        # 应用NMS
        if len(raw_annotations) > 1 and self.nms_threshold < 1.0:
            raw_annotations = apply_nms(raw_annotations, self.nms_threshold)

        # 质量验证过滤
        filtered = self.validator.filter_annotations(raw_annotations)

        return filtered

    def to_legacy_format(
        self,
        annotations: List[YOLOAnnotation]
    ) -> List[Tuple[str, Tuple[float, float, float, float], float]]:
        """
        转换为旧代码兼容的元组格式

        Args:
            annotations: YOLOAnnotation列表

        Returns:
            [(class_name, (cx, cy, w, h), score), ...]
        """
        return [ann.to_tuple() for ann in annotations]


# ============================================================================
# 便捷函数
# ============================================================================

def process_raw_detections(
    boxes: List[Tuple[float, float, float, float]],
    labels: List[Any],
    scores: List[float],
    class_names: List[str],
    img_width: int,
    img_height: int,
    nms_threshold: float = 0.5
) -> List[Tuple[str, Tuple[float, float, float, float], float]]:
    """
    便捷函数：处理原始检测结果为YOLO格式标注

    这是对外的主要接口函数。

    Args:
        boxes: 检测框列表
        labels: 标签列表
        scores: 置信度列表
        class_names: 类别名称列表
        img_width, img_height: 图片尺寸
        nms_threshold: NMS阈值

    Returns:
        [(class_name, (cx, cy, w, h), score), ...]
    """
    processor = AnnotationProcessor(
        class_names=class_names,
        img_width=img_width,
        img_height=img_height,
        nms_threshold=nms_threshold
    )

    annotations = processor.process_detection(boxes, labels, scores)
    return processor.to_legacy_format(annotations)
