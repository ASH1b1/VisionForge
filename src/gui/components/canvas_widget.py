from PySide6.QtWidgets import QWidget, QVBoxLayout, QHBoxLayout, QLabel, QMessageBox
from PySide6.QtCore import Qt, Signal, QPointF, QRectF
from PySide6.QtGui import (
    QPainter, QImage, QPixmap, QPen, QColor, QBrush,
    QFont, QPainterPath, QMouseEvent
)
from typing import List, Optional, Tuple

from ..models.app_state import Tool
from ..utils.annotation_draw import (
    keypoint_draw_items,
    keypoint_radius_px,
    obb_corners,
    resolved_kind,
    should_draw_aabb,
)
from ..utils.class_utils import coerce_class_id
from ..utils.hit_testing import HIT_RADIUS_PX, edge_hit_insert, vertex_hit_index
from ..utils.polygon_edit import yolo_bbox_from_polygon


class CanvasWidget(QWidget):
    """中间画布 - 图像/视频显示，标注框绘制和交互"""

    # Signals
    frame_requested = Signal(int)  # frame_index
    zoom_changed = Signal(float)  # zoom_level
    annotation_selected = Signal(int)  # annotation_id
    annotation_modified = Signal(int, tuple)  # id, new_bbox
    annotation_created = Signal(int, tuple)  # class_id, bbox
    annotation_deleted = Signal(int)  # annotation_id
    polygon_created = Signal(int, tuple, list)  # class_id, bbox, polygon
    polygon_edited = Signal(int, list, tuple)  # ann_id, polygon, bbox
    sam_click_at = Signal(float, float, int)  # x_norm, y_norm, label 1=fg 0=bg
    sam_confirm_requested = Signal()

    # 默认颜色方案 (YOLO 风格)
    CLASS_COLORS = [
        (255, 0, 0),    # 红色
        (0, 255, 0),    # 绿色
        (0, 0, 255),    # 蓝色
        (255, 255, 0),  # 黄色
        (255, 0, 255),  # 品红
        (0, 255, 255),  # 青色
        (255, 128, 0),  # 橙色
        (128, 0, 255),  # 紫色
        (255, 0, 128),  # 粉红
        (0, 128, 255),  # 天蓝
    ]

    def __init__(self, parent=None):
        super().__init__(parent)
        self._image = None
        self._annotations = []
        self._class_names = {}

        # 视图变换
        self._zoom = 1.0
        self._offset = QPointF(0, 0)
        self._image_rect = QRectF()

        # 交互状态
        self._tool = Tool.SELECT
        self._selected_class_id = None
        self._hovered_annotation_id = None
        self._selected_ids: set = set()  # 多选集合 (ids from ProjectDocument.selection_changed)
        self._dragging = False
        self._creating = False
        self._create_start_pos = QPointF()
        self._create_current_pos = QPointF()
        self._panning = False
        self._last_pan_pos = QPointF()

        # Polygon 绘制状态
        self._polygon_vertices: List[Tuple[float, float]] = []  # 归一化顶点
        self._polygon_preview_pos: Optional[QPointF] = None     # 鼠标预览位置

        # 选择工具下拖顶点
        self._dragging_vertex: Optional[Tuple[int, int]] = None  # (ann_id, index)
        self._vertex_drag_polygon: Optional[List[Tuple[float, float]]] = None
        self._vertex_drag_origin: Optional[List[Tuple[float, float]]] = None
        self._highlighted_vertex: Optional[Tuple[int, int]] = None
        self._sam_points: List[Tuple[float, float, int]] = []
        self._sam_preview_polygon: Optional[List[Tuple[float, float]]] = None
        self._sam_bound_id: Optional[int] = None

        # UI 设置
        self.setMouseTracking(True)
        self.setFocusPolicy(Qt.StrongFocus)
        self.setMinimumSize(400, 300)
        self.setStyleSheet("background-color: #1E1E1E;")

        # 回调函数 (外部设置)
        self._get_annotations_callback = None
        self._get_class_names_callback = None

        # Detect-mode overlay (YOLO-normalized AutoAnnotationResult list)
        self._detection_overlays = []
        self._show_annotations = True
        self._show_detections = False
        self._detection_score_filter = 0.0

    def set_callbacks(self, get_annotations, get_class_names):
        """设置获取标注和类别名的回调函数"""
        self._get_annotations_callback = get_annotations
        self._get_class_names_callback = get_class_names

    def set_detection_overlays(self, results: list | None, score_filter: float = 0.0):
        """Set transient Detect-mode boxes (YOLO cx,cy,w,h)."""
        self._detection_overlays = list(results or [])
        self._detection_score_filter = float(score_filter)
        self.update()

    def set_overlay_visibility(self, *, show_annotations: bool = True, show_detections: bool = False):
        """Toggle annotation vs detection overlay layers."""
        self._show_annotations = show_annotations
        self._show_detections = show_detections
        self.update()

    def _refresh_annotations(self):
        """刷新标注数据"""
        if self._get_annotations_callback:
            self._annotations = self._get_annotations_callback()
        if self._get_class_names_callback:
            self._class_names = self._get_class_names_callback()
        self.update()

    def set_selected_ids(self, ids: set) -> None:
        """同步多选 ID 集合到 Canvas（从 ProjectDocument.selection_changed 信号）"""
        self._selected_ids = ids
        if self._highlighted_vertex and self._highlighted_vertex[0] not in ids:
            self._highlighted_vertex = None
        self.update()

    def set_image(self, image: QImage):
        """设置显示的图像"""
        self._image = image
        self._fit_image_to_view()
        self.update()

    def set_tool(self, tool: Tool):
        """设置当前工具"""
        self._tool = tool
        self._reset_interaction_state()
        if tool == Tool.SAM_CLICK:
            self.setContextMenuPolicy(Qt.PreventContextMenu)
        else:
            self.setContextMenuPolicy(Qt.DefaultContextMenu)
        self.update()

    def set_sam_session_overlay(
        self,
        points: list | None = None,
        polygon: list | None = None,
        bound_id: int | None = None,
    ) -> None:
        self._sam_points = list(points or [])
        self._sam_preview_polygon = list(polygon) if polygon else None
        self._sam_bound_id = bound_id
        self.update()

    def _emit_sam_click(self, pos: QPointF, label: int) -> None:
        if self._image is None or not self._image_rect.contains(pos):
            return
        x_norm, y_norm = self._canvas_to_image_coords(pos)
        self.sam_click_at.emit(x_norm, y_norm, int(label))

    def _draw_sam_click_overlay(self, painter: QPainter) -> None:
        painter.save()
        if self._sam_preview_polygon and len(self._sam_preview_polygon) >= 3:
            path = self._polygon_to_canvas_path(self._sam_preview_polygon)
            if not path.isEmpty():
                painter.fillPath(path, QColor(59, 130, 246, 50))
                pen = QPen(QColor("#60A5FA"), 2, Qt.DashLine)
                painter.setPen(pen)
                painter.setBrush(Qt.NoBrush)
                painter.drawPath(path)
        painter.setPen(Qt.NoPen)
        for x, y, label in self._sam_points:
            pt = self._image_to_canvas_coords(x, y)
            color = QColor("#22C55E") if int(label) == 1 else QColor("#EF4444")
            painter.setBrush(QBrush(color))
            painter.drawEllipse(pt, 4, 4)
        painter.restore()

    def set_selected_class_id(self, class_id: int):
        """设置选中的类别 ID"""
        self._selected_class_id = class_id

    def set_zoom(self, zoom: float):
        """设置缩放级别"""
        self._zoom = max(0.1, min(zoom, 10.0))
        self._fit_image_to_view()
        self.update()
        self.zoom_changed.emit(self._zoom)

    def _fit_image_to_view(self):
        """将图像适配到视图中心"""
        if self._image is None:
            return

        # 计算图像在当前缩放下的尺寸
        img_width = self._image.width() * self._zoom
        img_height = self._image.height() * self._zoom

        # 居中显示
        widget_width = self.width()
        widget_height = self.height()

        self._offset.setX(max(0, (widget_width - img_width) / 2))
        self._offset.setY(max(0, (widget_height - img_height) / 2))

        # 更新图像矩形
        self._image_rect = QRectF(
            self._offset.x(),
            self._offset.y(),
            img_width,
            img_height
        )

    def resizeEvent(self, event):
        """窗口大小改变时重新计算位置"""
        super().resizeEvent(event)
        self._fit_image_to_view()

    def paintEvent(self, event):
        """绘制事件"""
        painter = QPainter(self)
        try:
            painter.setRenderHint(QPainter.Antialiasing)

            # 1. 绘制背景
            painter.fillRect(self.rect(), QColor("#1E1E1E"))

            if self._image is None:
                # 无图像时显示提示文本
                painter.setPen(QColor("#888888"))
                painter.setFont(QFont("Arial", 14))
                painter.drawText(self.rect(), Qt.AlignCenter, "No image loaded")
                return

            # 2. 绘制图像
            self._draw_image(painter)

            # 3. 绘制标注框（Annotate 模式）
            if self._show_annotations:
                self._refresh_annotations()
                self._draw_annotations(painter)

            # 3b. Detect 预览叠加层（不写入工程）
            if self._show_detections:
                self._draw_detection_overlays(painter)

            # 4. 绘制正在创建的框
            if self._creating and self._show_annotations:
                self._draw_creating_bbox(painter)

            # 5. 绘制正在创建的 polygon 预览
            if self._show_annotations and self._tool == Tool.POLYGON and self._polygon_vertices:
                self._draw_polygon_preview(painter)

            if self._show_annotations and self._tool == Tool.SAM_CLICK:
                self._draw_sam_click_overlay(painter)
        finally:
            painter.end()

    def _draw_image(self, painter: QPainter):
        """绘制图像"""
        # 使用目标矩形绘制缩放后的图像
        source_rect = QRectF(0, 0, self._image.width(), self._image.height())
        painter.drawImage(self._image_rect, self._image, source_rect)

    def _draw_annotations(self, painter: QPainter):
        """绘制所有标注框 - 统一样式：边框+左下角标签"""
        for ann in self._annotations:
            if not ann.visible:
                continue

            polygon = self._display_polygon(ann)
            bbox = (
                yolo_bbox_from_polygon(polygon)
                if polygon and self._dragging_vertex and self._dragging_vertex[0] == ann.id
                else ann.bbox
            )

            # 转换到画布坐标
            canvas_rect = self._image_to_canvas_rect(bbox)
            if canvas_rect.isEmpty():
                continue

            # 获取颜色
            color = self._get_class_color(ann.class_id)

            # 是否选中 (多选优先，fallback 到悬停)
            is_selected = ann.id in self._selected_ids or ann.id == self._hovered_annotation_id
            kind = resolved_kind(ann)

            # 绘制边框 - 选中时边框加粗加亮
            pen_width = 3 if is_selected else 2
            pen = QPen(QColor(*color), pen_width)
            if is_selected:
                bright_color = tuple(min(c + 80, 255) for c in color)
                pen = QPen(QColor(*bright_color), pen_width)
            painter.setPen(pen)
            painter.setBrush(Qt.NoBrush)
            if should_draw_aabb(kind):
                painter.drawRect(canvas_rect)

            if kind == "obb":
                corners = obb_corners(ann)
                if corners:
                    path = self._polygon_to_canvas_path(corners)
                    if not path.isEmpty():
                        fill_color = QColor(*color, 60)
                        painter.fillPath(path, fill_color)
                        painter.drawPath(path)
            elif polygon:
                if not (
                    self._tool == Tool.SAM_CLICK
                    and self._sam_bound_id == ann.id
                    and self._sam_preview_polygon
                ):
                    path = self._polygon_to_canvas_path(polygon)
                    if not path.isEmpty():
                        fill_color = QColor(*color, 60)
                        painter.fillPath(path, fill_color)
                        painter.drawPath(path)

            if kind == "pose":
                self._draw_keypoints(painter, getattr(ann, "keypoints", None), color)

            # 绘制标签背景和文本 - 左下角
            self._draw_annotation_label(painter, canvas_rect, ann, color)

            if (
                self._tool == Tool.SELECT
                and is_selected
                and len(self._selected_ids) == 1
                and polygon
                and len(polygon) >= 3
            ):
                highlight = None
                if self._highlighted_vertex and self._highlighted_vertex[0] == ann.id:
                    highlight = self._highlighted_vertex[1]
                self._draw_vertex_handles(painter, polygon, color, highlight)

    def _draw_detection_overlays(self, painter: QPainter):
        """Draw Detect-mode preview boxes (dashed, score in label)."""
        detect_colors = [
            (125, 255, 166),
            (80, 220, 255),
            (255, 200, 80),
            (200, 140, 255),
            (255, 120, 120),
        ]
        for idx, item in enumerate(self._detection_overlays):
            score = float(getattr(item, "score", 0.0))
            if score < self._detection_score_filter:
                continue
            bbox = getattr(item, "bbox", None)
            if not isinstance(bbox, tuple) or len(bbox) != 4:
                continue
            canvas_rect = self._image_to_canvas_rect(bbox)
            if canvas_rect.isEmpty():
                continue
            color = detect_colors[idx % len(detect_colors)]
            pen = QPen(QColor(*color), 2, Qt.DashLine)
            painter.setPen(pen)
            painter.setBrush(Qt.NoBrush)
            painter.drawRect(canvas_rect)

            class_name = getattr(item, "class_name", "?")
            label_text = f"{class_name} {score:.2f}"
            font = QFont("Arial", 10)
            font.setBold(True)
            painter.setFont(font)
            fm = painter.fontMetrics()
            tw = fm.horizontalAdvance(label_text) + 6
            th = fm.height() + 4
            label_rect = QRectF(canvas_rect.left(), max(0, canvas_rect.top() - th), tw, th)
            painter.setBrush(QBrush(QColor(*color, 210)))
            painter.setPen(Qt.NoPen)
            painter.drawRect(label_rect)
            painter.setPen(QColor(10, 20, 14))
            painter.drawText(label_rect.adjusted(3, 0, 0, 0), Qt.AlignVCenter | Qt.AlignLeft, label_text)

    def _draw_keypoints(self, painter: QPainter, keypoints, color: Tuple[int, int, int]) -> None:
        items = keypoint_draw_items(keypoints)
        if not items:
            return
        radius = keypoint_radius_px(self._zoom)
        painter.save()
        for x, y, style in items:
            pt = self._image_to_canvas_coords(x, y)
            painter.setPen(QPen(QColor(*color), 1))
            if style == "hollow":
                painter.setBrush(Qt.NoBrush)
            else:
                painter.setBrush(QBrush(QColor(*color)))
            painter.drawEllipse(pt, radius, radius)
        painter.restore()

    def _polygon_to_canvas_points(self, polygon: list[tuple[float, float]]) -> list[QPointF]:
        return [self._image_to_canvas_coords(x, y) for x, y in polygon]

    def _polygon_to_canvas_path(self, polygon: list[tuple[float, float]]) -> QPainterPath:
        path = QPainterPath()
        polygon_points = self._polygon_to_canvas_points(polygon)
        if not polygon_points:
            return path

        path.moveTo(polygon_points[0])
        for point in polygon_points[1:]:
            path.lineTo(point)
        path.closeSubpath()
        return path

    def _draw_annotation_label(self, painter: QPainter, canvas_rect: QRectF, ann, color: Tuple[int, int, int]):
        """绘制标注标签 - 左下角样式"""
        try:
            class_name = self._class_names.get(ann.class_id, f"Class {ann.class_id}")
            label_text = f"{class_name}"
            if ann.confidence < 1.0:
                label_text += f" {ann.confidence:.2f}"

            # 字体设置
            font = QFont("Arial", 10)
            font.setBold(True)
            painter.setFont(font)
            fm = painter.fontMetrics()
            text_width = fm.horizontalAdvance(label_text)
            text_height = fm.height()
            
            # 标签尺寸
            label_padding = 2
            label_height = text_height + label_padding * 2
            label_width = text_width + label_padding * 2

            # 标签位置 - 左下角 (紧贴边框内侧)
            label_x = canvas_rect.left()
            label_y = canvas_rect.bottom() - label_height

            # 确保标签在框内
            if label_x + label_width > canvas_rect.right():
                label_x = canvas_rect.right() - label_width
            if label_y < canvas_rect.top():
                label_y = canvas_rect.top()

            # 绘制背景
            bg_rect = QRectF(label_x, label_y, label_width, label_height)
            bg_color = QColor(*color, 220)
            painter.setBrush(QBrush(bg_color))
            painter.setPen(Qt.NoPen)
            painter.drawRect(bg_rect)

            # 绘制文本
            painter.setPen(QColor(255, 255, 255))
            text_rect = QRectF(label_x + label_padding, label_y + label_padding, 
                             text_width, text_height)
            painter.drawText(text_rect, Qt.AlignLeft | Qt.AlignVCenter, label_text)
        except (TypeError, ValueError, AttributeError):
            # Silently skip drawing if data is invalid
            pass

    def _draw_creating_bbox(self, painter: QPainter):
        """绘制正在创建的标注框"""
        if not self._creating:
            return

        # 计算矩形
        start = self._create_start_pos
        current = self._create_current_pos

        rect = QRectF(start, current).normalized()
        if rect.width() < 5 or rect.height() < 5:
            return

        # 绘制虚线框
        color = self._get_class_color(self._selected_class_id)
        pen = QPen(QColor(*color), 2, Qt.DashLine)
        painter.setPen(pen)
        painter.setBrush(Qt.NoBrush)
        painter.drawRect(rect)

        # 绘制尺寸提示
        painter.setPen(QColor(255, 255, 255))
        painter.setFont(QFont("Arial", 9))
        size_text = f"{int(rect.width())}x{int(rect.height())}"
        painter.drawText(int(rect.x()) + 5, int(rect.y()) - 5, size_text)

    def _draw_polygon_preview(self, painter: QPainter):
        """绘制正在创建的 polygon 预览"""
        painter.save()
        pen = QPen(QColor("#3B82F6"), 2)
        painter.setPen(pen)

        # 已放置的顶点间连线
        path = self._build_polygon_preview_path(self._polygon_vertices)
        painter.drawPath(path)

        # 预览虚线到鼠标
        if self._polygon_preview_pos and self._polygon_vertices:
            last_vertex = self._polygon_vertices[-1]
            last_canvas = self._image_to_canvas_coords(last_vertex[0], last_vertex[1])
            preview_pen = QPen(QColor("#93C5FD"), 2, Qt.DashLine)
            painter.setPen(preview_pen)
            painter.drawLine(last_canvas, self._polygon_preview_pos)

        # 顶点标记（小圆点）
        painter.setPen(Qt.NoPen)
        painter.setBrush(QBrush(QColor("#3B82F6")))
        for v in self._polygon_vertices:
            pt = self._image_to_canvas_coords(v[0], v[1])
            painter.drawEllipse(pt, 3, 3)

        painter.restore()

    def _draw_vertex_handles(
        self,
        painter: QPainter,
        polygon: list[tuple[float, float]],
        color: Tuple[int, int, int],
        highlight_index: Optional[int],
    ) -> None:
        painter.save()
        points = self._polygon_to_canvas_points(polygon)
        for i, pt in enumerate(points):
            radius = 5.0 if i == highlight_index else 4.0
            painter.setPen(QPen(QColor(*color), 1))
            if i == highlight_index:
                painter.setBrush(QBrush(QColor(*color)))
            else:
                painter.setBrush(QBrush(QColor(255, 255, 255)))
            painter.drawEllipse(pt, radius, radius)
        painter.restore()

    def _get_class_color(self, class_id: int) -> Tuple[int, int, int]:
        """获取类别颜色"""
        resolved_class_id = coerce_class_id(class_id)
        if resolved_class_id is None:
            return self.CLASS_COLORS[0]

        idx = resolved_class_id % len(self.CLASS_COLORS)
        return self.CLASS_COLORS[idx]

    # ========== 坐标转换 ==========

    def _canvas_to_image_coords(self, canvas_pos: QPointF) -> Tuple[float, float]:
        """画布坐标 -> 图像归一化坐标"""
        if self._image is None:
            return 0.0, 0.0

        # 减去偏移
        x = (canvas_pos.x() - self._offset.x()) / self._zoom
        y = (canvas_pos.y() - self._offset.y()) / self._zoom

        # 归一化到 [0, 1]
        norm_x = x / self._image.width()
        norm_y = y / self._image.height()

        # 限制范围
        norm_x = max(0.0, min(1.0, norm_x))
        norm_y = max(0.0, min(1.0, norm_y))

        return norm_x, norm_y

    def _image_to_canvas_coords(self, norm_x: float, norm_y: float) -> QPointF:
        """图像归一化坐标 -> 画布坐标"""
        if self._image is None:
            return QPointF()

        # 从归一化坐标转换到像素坐标
        pixel_x = norm_x * self._image.width() * self._zoom
        pixel_y = norm_y * self._image.height() * self._zoom

        # 加上偏移
        return QPointF(
            pixel_x + self._offset.x(),
            pixel_y + self._offset.y()
        )

    def _image_to_canvas_rect(self, bbox: tuple) -> QRectF:
        """YOLO bbox -> 画布矩形

        bbox 格式: (x_center, y_center, width, height) 归一化坐标
        """
        try:
            if not bbox or len(bbox) != 4:
                return QRectF()

            x_center, y_center, w, h = bbox

            # 计算角点 (归一化)
            x1 = x_center - w / 2
            y1 = y_center - h / 2
            x2 = x_center + w / 2
            y2 = y_center + h / 2

            # 转换为画布坐标
            top_left = self._image_to_canvas_coords(x1, y1)
            bottom_right = self._image_to_canvas_coords(x2, y2)

            return QRectF(top_left, bottom_right).normalized()
        except (TypeError, ValueError, AttributeError):
            return QRectF()

    def _canvas_to_yolo_bbox(self, start_pos: QPointF, end_pos: QPointF) -> Tuple[float, float, float, float]:
        """画布矩形 -> YOLO bbox

        返回: (x_center, y_center, width, height) 归一化坐标
        """
        # 转换起点和终点到图像归一化坐标
        x1, y1 = self._canvas_to_image_coords(start_pos)
        x2, y2 = self._canvas_to_image_coords(end_pos)

        # 确保顺序正确
        x_min, x_max = min(x1, x2), max(x1, x2)
        y_min, y_max = min(y1, y2), max(y1, y2)

        # 计算中心点和尺寸
        x_center = (x_min + x_max) / 2
        y_center = (y_min + y_max) / 2
        width = x_max - x_min
        height = y_max - y_min

        return (x_center, y_center, width, height)

    # ========== 鼠标事件 ==========

    def mousePressEvent(self, event: QMouseEvent):
        """鼠标按下事件"""
        if self._image is None:
            return

        self.setFocus(Qt.MouseFocusReason)
        pos = event.position()

        # 中键 - 平移
        if event.button() == Qt.MiddleButton:
            self._panning = True
            self._last_pan_pos = pos
            self.setCursor(Qt.ClosedHandCursor)
            return

        # 左键 - 根据工具执行不同操作
        if event.button() == Qt.LeftButton:
            if self._tool == Tool.SELECT:
                self._handle_select_press(pos)
            elif self._tool == Tool.RECTANGLE:
                self._handle_rectangle_press(pos)
            elif self._tool == Tool.ERASER:
                self._handle_eraser_press(pos)
            elif self._tool == Tool.POLYGON:
                self._handle_polygon_press(pos)
            elif self._tool == Tool.SAM_CLICK:
                label = 0 if event.modifiers() & Qt.AltModifier else 1
                self._emit_sam_click(pos, label)

        elif event.button() == Qt.RightButton and self._tool == Tool.POLYGON:
            if len(self._polygon_vertices) >= 3:
                self._finish_polygon()
        elif event.button() == Qt.RightButton and self._tool == Tool.SAM_CLICK:
            self._emit_sam_click(pos, 0)

    def mouseMoveEvent(self, event: QMouseEvent):
        """鼠标移动事件"""
        if self._image is None:
            return

        pos = event.position()

        if self._dragging_vertex is not None:
            self._update_vertex_drag(pos)
            return

        # 平移模式
        if self._panning:
            delta = pos - self._last_pan_pos
            self._offset += delta
            self._image_rect.translate(delta)
            self._last_pan_pos = pos
            self.update()
            return

        # 更新悬停状态 (非拖拽/创建时)
        if not self._dragging and not self._creating:
            self._update_hover_annotation(pos)
            self.update()

        # 创建新框
        if self._creating and self._tool == Tool.RECTANGLE:
            self._create_current_pos = pos
            self.update()

        # Polygon 预览
        elif self._tool == Tool.POLYGON and self._polygon_vertices:
            self._polygon_preview_pos = pos
            self.update()

    def mouseReleaseEvent(self, event: QMouseEvent):
        """鼠标释放事件"""
        if self._image is None:
            return

        pos = event.position()

        # 结束平移
        if self._panning and event.button() == Qt.MiddleButton:
            self._panning = False
            self.setCursor(Qt.ArrowCursor)
            return

        if self._dragging_vertex is not None and event.button() == Qt.LeftButton:
            self._commit_vertex_drag()
            return

        # 结束创建
        if self._creating and event.button() == Qt.LeftButton:
            self._handle_rectangle_release(pos)
            return

    def wheelEvent(self, event):
        """滚轮缩放"""
        if self._image is None:
            return

        # 计算新的缩放级别
        delta = event.angleDelta().y()
        zoom_factor = 1.1 if delta > 0 else 0.9
        new_zoom = self._zoom * zoom_factor
        new_zoom = max(0.1, min(new_zoom, 10.0))

        # 以鼠标位置为中心缩放
        mouse_pos = event.position()
        mouse_in_image_before = (
            (mouse_pos.x() - self._offset.x()) / self._zoom,
            (mouse_pos.y() - self._offset.y()) / self._zoom
        )

        self._zoom = new_zoom

        # 调整偏移使鼠标位置保持不变
        self._offset.setX(
            mouse_pos.x() - mouse_in_image_before[0] * self._zoom
        )
        self._offset.setY(
            mouse_pos.y() - mouse_in_image_before[1] * self._zoom
        )

        # 更新图像矩形
        self._fit_image_to_view()

        self.zoom_changed.emit(self._zoom)
        self.update()

    def mouseDoubleClickEvent(self, event: QMouseEvent):
        if self._image is None or event.button() != Qt.LeftButton:
            return
        if self._tool != Tool.SELECT:
            return
        self._cancel_vertex_drag()
        self._try_insert_vertex(event.position())

    # ========== 工具处理方法 ==========

    def _handle_select_press(self, pos: QPointF):
        """处理选择工具的点击"""
        if self._try_begin_vertex_drag(pos):
            return
        ann_id = self._find_annotation_at(pos)
        if ann_id is not None:
            self.annotation_selected.emit(ann_id)
            self._hovered_annotation_id = ann_id
        else:
            self.annotation_selected.emit(-1)  # 取消选择
            self._hovered_annotation_id = None
            self._highlighted_vertex = None
        self.update()

    def _handle_rectangle_press(self, pos: QPointF):
        """处理矩形工具的点击 - 开始创建"""
        # 检查是否在图像区域内
        if not self._image_rect.contains(pos):
            return

        self._creating = True
        self._create_start_pos = pos
        self._create_current_pos = pos

    def _handle_rectangle_release(self, pos: QPointF):
        """处理矩形工具的释放 - 完成创建"""
        if not self._creating:
            return

        self._creating = False

        # 计算最终 bbox
        bbox = self._canvas_to_yolo_bbox(self._create_start_pos, pos)

        # 检查最小尺寸
        if bbox[2] < 0.01 or bbox[3] < 0.01:
            return

        # 检查是否已选择类别
        if self._selected_class_id is None:
            QMessageBox.warning(self, "提示", "请先在左侧类别栏中添加类别")
            return

        # 发送创建信号
        self.annotation_created.emit(self._selected_class_id, bbox)
        self.update()

    def _handle_eraser_press(self, pos: QPointF):
        """处理橡皮擦工具的点击 - 删除标注"""
        ann_id = self._find_annotation_at(pos)
        if ann_id is not None:
            self.annotation_deleted.emit(ann_id)
            self._hovered_annotation_id = None
            self._refresh_annotations()
            self.update()

    def _handle_polygon_press(self, pos: QPointF):
        """处理多边形工具的点击 - 添加顶点"""
        if not self._image_rect.contains(pos):
            return

        # 如果点击落在已有标注上，先选中（不开始绘制）
        ann_id = self._find_annotation_at(pos)
        if ann_id is not None and not self._polygon_vertices:
            self.annotation_selected.emit(ann_id)
            return

        # 转换坐标
        norm_x, norm_y = self._canvas_to_image_coords(pos)
        self._polygon_vertices.append((norm_x, norm_y))
        self.update()

    def _finish_polygon(self):
        """闭合 polygon 并创建标注"""
        if len(self._polygon_vertices) < 3:
            self._cancel_polygon()
            return

        vertices = list(self._polygon_vertices)
        bbox = yolo_bbox_from_polygon(vertices)

        # 检查是否已选择类别
        if self._selected_class_id is None:
            QMessageBox.warning(self, "提示", "请先在左侧类别栏中添加类别")
            self._cancel_polygon()
            return

        # 发送信号
        self.polygon_created.emit(self._selected_class_id, bbox, vertices)

        # 清理
        self._polygon_vertices.clear()
        self._polygon_preview_pos = None
        self.update()

    def _cancel_polygon(self):
        """取消 polygon 绘制"""
        self._polygon_vertices.clear()
        self._polygon_preview_pos = None
        self.update()

    def _build_polygon_preview_path(self, vertices: list) -> QPainterPath:
        """构建 polygon 预览路径（画布坐标）"""
        path = QPainterPath()
        if not vertices:
            return path
        canvas_points = self._polygon_to_canvas_points(vertices)
        path.moveTo(canvas_points[0])
        for pt in canvas_points[1:]:
            path.lineTo(pt)
        return path

    def _sync_annotations(self) -> list:
        if self._get_annotations_callback:
            self._annotations = self._get_annotations_callback()
        return self._annotations

    def _display_polygon(self, ann) -> list | None:
        if (
            self._dragging_vertex
            and self._dragging_vertex[0] == ann.id
            and self._vertex_drag_polygon is not None
        ):
            return self._vertex_drag_polygon
        return ann.polygon

    def _selected_polygon_ann(self):
        if self._tool != Tool.SELECT or len(self._selected_ids) != 1:
            return None
        self._sync_annotations()
        ann_id = next(iter(self._selected_ids))
        for ann in self._annotations:
            if ann.id == ann_id and ann.visible and ann.polygon and len(ann.polygon) >= 3:
                return ann
        return None

    def _try_begin_vertex_drag(self, pos: QPointF) -> bool:
        ann = self._selected_polygon_ann()
        if ann is None:
            return False
        verts = self._polygon_to_canvas_points(ann.polygon)
        index = vertex_hit_index(verts, pos, HIT_RADIUS_PX)
        if index is None:
            return False
        self._dragging = True
        self._dragging_vertex = (ann.id, index)
        self._vertex_drag_origin = [(float(x), float(y)) for x, y in ann.polygon]
        self._vertex_drag_polygon = list(self._vertex_drag_origin)
        self._highlighted_vertex = (ann.id, index)
        self.setCursor(Qt.SizeAllCursor)
        self.update()
        return True

    def _update_vertex_drag(self, pos: QPointF) -> None:
        if self._dragging_vertex is None or self._vertex_drag_polygon is None:
            return
        _, index = self._dragging_vertex
        nx, ny = self._canvas_to_image_coords(pos)
        polygon = list(self._vertex_drag_polygon)
        polygon[index] = (nx, ny)
        self._vertex_drag_polygon = polygon
        self.update()

    def _commit_vertex_drag(self) -> None:
        if self._dragging_vertex is None or self._vertex_drag_polygon is None:
            self._cancel_vertex_drag()
            return
        ann_id, index = self._dragging_vertex
        polygon = [(float(x), float(y)) for x, y in self._vertex_drag_polygon]
        origin = self._vertex_drag_origin or []
        self._dragging = False
        self._dragging_vertex = None
        self._vertex_drag_polygon = None
        self._vertex_drag_origin = None
        self._highlighted_vertex = (ann_id, index)
        if polygon != origin:
            self.polygon_edited.emit(ann_id, polygon, yolo_bbox_from_polygon(polygon))
        self.update()

    def _cancel_vertex_drag(self) -> None:
        self._dragging = False
        self._dragging_vertex = None
        self._vertex_drag_polygon = None
        self._vertex_drag_origin = None
        self.update()

    def _try_insert_vertex(self, pos: QPointF) -> bool:
        ann = self._selected_polygon_ann()
        if ann is None:
            return False
        verts = self._polygon_to_canvas_points(ann.polygon)
        if vertex_hit_index(verts, pos, HIT_RADIUS_PX) is not None:
            return False
        hit = edge_hit_insert(verts, pos, HIT_RADIUS_PX)
        if hit is None:
            return False
        index, canvas_pt = hit
        nx, ny = self._canvas_to_image_coords(canvas_pt)
        polygon = [(float(x), float(y)) for x, y in ann.polygon]
        polygon.insert(index + 1, (nx, ny))
        self._highlighted_vertex = (ann.id, index + 1)
        self.polygon_edited.emit(ann.id, polygon, yolo_bbox_from_polygon(polygon))
        self.update()
        return True

    def try_delete_highlighted_vertex(self) -> bool:
        """Delete the highlighted vertex. True if the shortcut was consumed."""
        if self._tool != Tool.SELECT or self._highlighted_vertex is None:
            return False
        ann = self._selected_polygon_ann()
        if ann is None or ann.id != self._highlighted_vertex[0]:
            return False
        if len(ann.polygon) <= 3:
            return True
        index = self._highlighted_vertex[1]
        if index < 0 or index >= len(ann.polygon):
            return True
        polygon = [(float(x), float(y)) for i, (x, y) in enumerate(ann.polygon) if i != index]
        new_index = min(index, len(polygon) - 1)
        self._highlighted_vertex = (ann.id, new_index)
        self.polygon_edited.emit(ann.id, polygon, yolo_bbox_from_polygon(polygon))
        self.update()
        return True

    # ========== 辅助方法 ==========

    def _find_annotation_at(self, pos: QPointF) -> Optional[int]:
        """查找指定画布坐标位置的标注 ID"""
        for ann in reversed(self._annotations):  # 从上层开始检查
            if not ann.visible:
                continue

            canvas_rect = self._image_to_canvas_rect(ann.bbox)

            # 检查是否在框内
            if canvas_rect.contains(pos):
                return ann.id

            # 检查是否在边框附近 (扩大检测范围)
            expanded = canvas_rect.adjusted(-5, -5, 5, 5)
            if expanded.contains(pos):
                return ann.id

        return None

    def _update_hover_annotation(self, pos: QPointF):
        """更新悬停的标注"""
        ann = self._selected_polygon_ann()
        if ann is not None:
            verts = self._polygon_to_canvas_points(ann.polygon)
            index = vertex_hit_index(verts, pos, HIT_RADIUS_PX)
            if index is not None:
                self._highlighted_vertex = (ann.id, index)
                self._hovered_annotation_id = ann.id
                self.setCursor(Qt.SizeAllCursor)
                return
        if self._highlighted_vertex is not None and (
            ann is None or self._highlighted_vertex[0] != (ann.id if ann else None)
        ):
            self._highlighted_vertex = None

        ann_id = self._find_annotation_at(pos)
        if ann_id != self._hovered_annotation_id:
            self._hovered_annotation_id = ann_id
            # 更新鼠标样式
            if ann_id is not None:
                if self._tool == Tool.SELECT:
                    self.setCursor(Qt.PointingHandCursor)
                elif self._tool == Tool.ERASER:
                    self.setCursor(Qt.CrossCursor)
                else:
                    self.setCursor(Qt.ArrowCursor)
            else:
                self.setCursor(Qt.ArrowCursor)

    def _reset_interaction_state(self):
        """重置交互状态"""
        self._dragging = False
        self._creating = False
        self._panning = False
        self._hovered_annotation_id = None
        self._dragging_vertex = None
        self._vertex_drag_polygon = None
        self._vertex_drag_origin = None
        self._highlighted_vertex = None
        self._sam_points = []
        self._sam_preview_polygon = None
        self._sam_bound_id = None
        self._polygon_vertices.clear()
        self._polygon_preview_pos = None
        self.setCursor(Qt.ArrowCursor)

    def keyPressEvent(self, event):
        """键盘事件"""
        if self._tool == Tool.SAM_CLICK and event.key() in (Qt.Key_Return, Qt.Key_Enter):
            self.sam_confirm_requested.emit()
            event.accept()
            return
        if self._tool == Tool.POLYGON:
            if event.key() == Qt.Key_Escape:
                self._cancel_polygon()
            elif event.key() == Qt.Key_Backspace and self._polygon_vertices:
                self._polygon_vertices.pop()
                self.update()
            elif event.key() in (Qt.Key_Return, Qt.Key_Enter):
                if len(self._polygon_vertices) >= 3:
                    self._finish_polygon()
        super().keyPressEvent(event)
