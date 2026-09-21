from __future__ import annotations

from dataclasses import dataclass

from PySide6.QtGui import QImage

from src.gui.components.canvas_widget import CanvasWidget, Tool
from src.gui.controllers.annotation_controller import AnnotationController
from src.gui.models.app_state import AppStateManager
from src.gui.models.project_document import ProjectDocument
from src.gui.utils.polygon_edit import yolo_bbox_from_polygon


@dataclass
class FakeAnn:
    id: int
    class_id: int = 0
    confidence: float = 1.0
    bbox: tuple = (0.4, 0.4, 0.4, 0.4)
    polygon: list | None = None
    visible: bool = True


SQUARE = [(0.2, 0.2), (0.6, 0.2), (0.6, 0.6), (0.2, 0.6)]


def _make_canvas(qtbot, anns: list) -> CanvasWidget:
    canvas = CanvasWidget()
    qtbot.addWidget(canvas)
    image = QImage(200, 200, QImage.Format_RGB32)
    image.fill(0)
    canvas.resize(400, 300)
    canvas.set_image(image)
    canvas.set_tool(Tool.SELECT)
    canvas.set_callbacks(lambda: anns, lambda: {0: "a"})
    return canvas


def test_select_press_on_vertex_starts_preview_not_write(qtbot):
    ann = FakeAnn(id=1, polygon=list(SQUARE))
    canvas = _make_canvas(qtbot, [ann])
    canvas.set_selected_ids({1})
    emitted: list = []
    canvas.polygon_edited.connect(lambda *args: emitted.append(args))

    press = canvas._image_to_canvas_coords(0.2, 0.2)
    assert canvas._try_begin_vertex_drag(press)
    assert canvas._dragging_vertex == (1, 0)
    canvas._update_vertex_drag(canvas._image_to_canvas_coords(0.25, 0.25))
    assert canvas._vertex_drag_polygon[0] == (0.25, 0.25)
    assert ann.polygon[0] == (0.2, 0.2)
    assert emitted == []


def test_commit_vertex_drag_emits_polygon_and_bbox(qtbot):
    ann = FakeAnn(id=1, polygon=list(SQUARE))
    canvas = _make_canvas(qtbot, [ann])
    canvas.set_selected_ids({1})
    emitted: list = []
    canvas.polygon_edited.connect(lambda *args: emitted.append(args))

    canvas._try_begin_vertex_drag(canvas._image_to_canvas_coords(0.2, 0.2))
    canvas._update_vertex_drag(canvas._image_to_canvas_coords(0.1, 0.2))
    canvas._commit_vertex_drag()

    assert len(emitted) == 1
    ann_id, polygon, bbox = emitted[0]
    assert ann_id == 1
    assert polygon[0] == (0.1, 0.2)
    expected = yolo_bbox_from_polygon(polygon)
    assert bbox == expected
    assert canvas._dragging_vertex is None


def test_bbox_only_annotation_does_not_start_vertex_drag(qtbot):
    ann = FakeAnn(id=2, polygon=None)
    canvas = _make_canvas(qtbot, [ann])
    canvas.set_selected_ids({2})
    press = canvas._image_to_canvas_coords(0.4, 0.4)
    assert canvas._try_begin_vertex_drag(press) is False
    assert canvas._dragging_vertex is None


def test_insert_vertex_on_edge_emits_longer_polygon(qtbot):
    ann = FakeAnn(id=1, polygon=list(SQUARE))
    canvas = _make_canvas(qtbot, [ann])
    canvas.set_selected_ids({1})
    emitted: list = []
    canvas.polygon_edited.connect(lambda *args: emitted.append(args))

    assert canvas._try_insert_vertex(canvas._image_to_canvas_coords(0.4, 0.2))
    assert len(emitted) == 1
    _, polygon, _ = emitted[0]
    assert len(polygon) == 5
    assert abs(polygon[1][0] - 0.4) < 1e-6
    assert abs(polygon[1][1] - 0.2) < 1e-6


def test_delete_vertex_keeps_at_least_three(qtbot):
    ann = FakeAnn(id=1, polygon=list(SQUARE))
    canvas = _make_canvas(qtbot, [ann])
    canvas.set_selected_ids({1})
    emitted: list = []
    canvas.polygon_edited.connect(lambda *args: emitted.append(args))
    canvas._highlighted_vertex = (1, 0)

    assert canvas.try_delete_highlighted_vertex() is True
    assert len(emitted[0][1]) == 3

    ann.polygon = list(emitted[0][1])
    canvas._highlighted_vertex = (1, 0)
    assert canvas.try_delete_highlighted_vertex() is True
    assert len(emitted) == 1


def test_edit_polygon_is_one_undo_step(qtbot):
    project = ProjectDocument()
    ctrl = AnnotationController(project, AppStateManager())
    origin = list(SQUARE)
    ann = ctrl.add_annotation(0, yolo_bbox_from_polygon(origin), polygon=origin)
    moved = [(0.1, 0.2), (0.6, 0.2), (0.6, 0.6), (0.2, 0.6)]
    ctrl.edit_polygon(ann.id, moved, yolo_bbox_from_polygon(moved))
    current = project.current_annotations[0]
    assert current.polygon[0] == (0.1, 0.2)
    ctrl.undo()
    current = project.current_annotations[0]
    assert current.polygon[0] == (0.2, 0.2)
    assert current.bbox == yolo_bbox_from_polygon(origin)


def test_update_annotation_history_deepcopies_polygon(qtbot):
    project = ProjectDocument()
    origin = list(SQUARE)
    ann = project.add_annotation(0, yolo_bbox_from_polygon(origin), polygon=origin)
    moved = [(0.1, 0.2), (0.6, 0.2), (0.6, 0.6), (0.2, 0.6)]
    project.update_annotation(ann.id, polygon=moved, bbox=yolo_bbox_from_polygon(moved))
    moved[0] = (0.9, 0.9)
    project.undo()
    current = project.current_annotations[0]
    assert current.polygon[0] == (0.2, 0.2)


def test_shortcut_manager_does_not_rebind_undo():
    from pathlib import Path

    text = Path("src/gui/managers/shortcut_manager.py").read_text(encoding="utf-8")
    assert 'self._add_shortcut("Ctrl+Z"' not in text
    assert 'self._add_shortcut("Ctrl+Y"' not in text


def test_sam_click_return_and_enter_emit_confirm(qtbot):
    from PySide6.QtCore import Qt

    canvas = _make_canvas(qtbot, [])
    canvas.set_tool(Tool.SAM_CLICK)
    received = []
    canvas.sam_confirm_requested.connect(lambda: received.append(True))
    qtbot.keyClick(canvas, Qt.Key_Return)
    qtbot.keyClick(canvas, Qt.Key_Enter)
    assert received == [True, True]
