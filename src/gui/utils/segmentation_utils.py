import cv2
import numpy as np


def _get_ann_attr(ann, name: str, default=None):
    """从标注对象（dict 或 object）获取属性值。"""
    if isinstance(ann, dict):
        return ann.get(name, default)
    return getattr(ann, name, default)


def polygon_to_mask(polygon: list, width: int, height: int) -> "np.ndarray | None":
    """将归一化 polygon 渲染为二值 mask 图像（像素坐标）。

    Args:
        polygon: 归一化顶点列表 [(x, y), ...]，坐标范围 [0, 1]
        width: 图像宽度（像素）
        height: 图像高度（像素）

    Returns:
        shape=(height, width) 的 uint8 二值 mask，或 None（输入无效时）
    """
    if not polygon or len(polygon) < 3:
        return None

    points = []
    for x, y in polygon:
        px = int(round(x * width))
        py = int(round(y * height))
        px = max(0, min(px, width - 1))
        py = max(0, min(py, height - 1))
        points.append([px, py])

    mask = np.zeros((height, width), dtype=np.uint8)
    cv2.fillPoly(mask, [np.array(points, dtype=np.int32)], 255)
    return mask


def render_instance_mask_image(
    annotations: list,
    width: int,
    height: int,
    class_colors: "dict[int, tuple[int, int, int]] | None" = None,
) -> np.ndarray:
    """渲染彩色实例分割 mask。

    Args:
        annotations: Annotation 对象或 dict 列表
                     （需有 polygon, class_id, visible 属性/key）
        width, height: 画布尺寸
        class_colors: class_id → (R, G, B) 颜色映射，为 None 时自动分配

    Returns:
        shape=(height, width, 3) 的 RGB mask 图像（uint8）
    """
    if class_colors is None:
        class_colors = {}

    DEFAULT_COLORS = [
        (255, 0, 0), (0, 255, 0), (0, 0, 255),
        (255, 255, 0), (255, 0, 255), (0, 255, 255),
        (255, 128, 0), (128, 0, 255), (255, 0, 128),
        (0, 128, 255),
    ]

    def _color_for(cid: int) -> tuple[int, int, int]:
        if cid in class_colors:
            return class_colors[cid]
        return DEFAULT_COLORS[cid % len(DEFAULT_COLORS)]

    mask = np.zeros((height, width, 3), dtype=np.uint8)

    for ann in annotations:
        if not _get_ann_attr(ann, "visible", True):
            continue
        polygon = _get_ann_attr(ann, "polygon")
        if not polygon:
            continue
        binary = polygon_to_mask(polygon, width, height)
        if binary is not None:
            color = _color_for(_get_ann_attr(ann, "class_id", 0))
            mask[binary > 0] = color

    return mask


def render_class_mask_image(
    annotations: list, width: int, height: int
) -> np.ndarray:
    """渲染语义分割 class mask（每个像素值为 class_id + 1，0 为背景）。

    支持 Annotation 对象和 dict 两种输入格式。

    Returns:
        shape=(height, width) 的 uint8 灰度 mask
    """
    mask = np.zeros((height, width), dtype=np.uint8)

    for ann in annotations:
        if not _get_ann_attr(ann, "visible", True):
            continue
        polygon = _get_ann_attr(ann, "polygon")
        if not polygon:
            continue
        binary = polygon_to_mask(polygon, width, height)
        if binary is not None:
            class_id = _get_ann_attr(ann, "class_id", 0)
            # 加 1 避免 class_id=0 与背景混淆；限制在 [1, 255]
            value = min(class_id + 1, 255)
            mask[binary > 0] = value

    return mask


def normalize_polygon(points, width, height):
    if width <= 0 or height <= 0:
        return []

    normalized = []
    for x, y in points:
        normalized.append((
            min(max(float(x) / width, 0.0), 1.0),
            min(max(float(y) / height, 0.0), 1.0),
        ))
    return normalized


def xyxy_pixels_to_yolo(box, width, height):
    if width <= 0 or height <= 0 or box is None or len(box) != 4:
        return None

    x1, y1, x2, y2 = [float(value) for value in box]
    x1 = min(max(x1, 0.0), float(width))
    y1 = min(max(y1, 0.0), float(height))
    x2 = min(max(x2, 0.0), float(width))
    y2 = min(max(y2, 0.0), float(height))

    if x2 <= x1 or y2 <= y1:
        return None

    return (
        ((x1 + x2) / 2.0) / width,
        ((y1 + y2) / 2.0) / height,
        (x2 - x1) / width,
        (y2 - y1) / height,
    )


def mask_to_polygon(mask):
    if hasattr(mask, "detach") and hasattr(mask, "cpu"):
        mask = mask.detach().cpu().numpy()
    mask_array = np.asarray(mask)
    if mask_array.ndim > 2:
        mask_array = np.squeeze(mask_array)
    mask_array = (mask_array > 0).astype(np.uint8)
    contours, _ = cv2.findContours(mask_array, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return []

    contour = max(contours, key=cv2.contourArea)
    epsilon = 0.002 * cv2.arcLength(contour, True)
    contour = cv2.approxPolyDP(contour, epsilon, True)
    return [(int(point[0][0]), int(point[0][1])) for point in contour]
