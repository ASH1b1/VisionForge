from __future__ import annotations

from pathlib import Path

from src.gui.controllers.annotation_controller import AnnotationController
from src.gui.controllers.sam_click_controller import (
    MSG_BLOCKED_KIND,
    MSG_CLICK_MISS,
    MSG_HINT_ACTIVE,
    MSG_NO_CLASS,
    MSG_NO_PREDICTOR,
    MSG_SAM_NOT_LOADED,
    SamClickController,
)
from src.gui.models.app_state import AppStateManager, Tool
from src.gui.models.project_document import ProjectDocument
from src.gui.utils.polygon_edit import yolo_bbox_from_polygon


SQUARE = [(0.2, 0.2), (0.6, 0.2), (0.6, 0.6), (0.2, 0.6)]
FAKE_POLY = [(0.25, 0.25), (0.55, 0.25), (0.55, 0.55), (0.25, 0.55)]


class FakeCanvas:
    def __init__(self):
        self.overlay = {"points": [], "polygon": None, "bound_id": None}

    def set_sam_session_overlay(self, points=None, polygon=None, bound_id=None):
        self.overlay = {
            "points": list(points or []),
            "polygon": list(polygon) if polygon else None,
            "bound_id": bound_id,
        }


class FakeSam:
    def __init__(self, predictor=True):
        self._predictor = predictor

    def has_interactive_predictor(self):
        return self._predictor

    def is_loaded(self):
        return True


class FakeModelCtrl:
    def __init__(self, *, loaded=True, predictor=True):
        self._loaded = loaded
        self._predictor = predictor
        self.leases = 0
        self.max_leases = 0
        self._sam_loading = False
        self.load_calls = 0

    def is_sam3_loaded(self):
        return self._loaded

    def acquire_inference_lease(self):
        self.leases += 1
        self.max_leases = max(self.max_leases, self.leases)

    def release_inference_lease(self):
        self.leases = max(0, self.leases - 1)

    def get_sam_model(self):
        if not self._loaded:
            return None
        return FakeSam(self._predictor)

    def load_sam3(self, parent_widget=None, *, skip_vram_confirm=False):
        self.load_calls += 1
        return True


def _make_predict(calls):
    def predict(**kwargs):
        calls.append(kwargs)
        return {
            "polygon": list(FAKE_POLY),
            "low_res_logits": "logits",
            "generation": kwargs.get("generation", 0),
        }

    return predict


def _harness(
    *,
    predict=None,
    loaded=True,
    predictor=True,
    class_id=0,
    ask_load=lambda: False,
):
    project = ProjectDocument()
    project.class_names = {0: "person"}
    app_state = AppStateManager()
    ann_ctrl = AnnotationController(project, app_state)
    canvas = FakeCanvas()
    calls: list = []
    statuses: list = []
    model_ctrl = FakeModelCtrl(loaded=loaded, predictor=predictor)
    ctrl = SamClickController(
        project,
        app_state,
        ann_ctrl,
        model_ctrl,
        canvas,
        predict_fn=predict or _make_predict(calls),
        image_provider=lambda: object(),
        image_size_provider=lambda: (100, 100),
        selected_class_provider=lambda: class_id,
        ask_load_sam=ask_load,
        frame_key_provider=lambda: (0, "img.png", 100, 100),
    )
    ctrl.status_message.connect(statuses.append)
    return ctrl, project, ann_ctrl, canvas, calls, statuses, model_ctrl


def test_bind_without_polygon_sends_box_and_does_not_commit():
    ctrl, project, ann_ctrl, canvas, calls, _, model_ctrl = _harness()
    ann = ann_ctrl.add_annotation(0, (0.4, 0.4, 0.4, 0.4))
    project.set_selection({ann.id})
    ctrl.on_tool_changed(Tool.SAM_CLICK)

    assert len(calls) == 1
    assert calls[0]["box_xyxy"] is not None
    assert calls[0]["point_coords"] is None
    assert calls[0]["mask_input"] is None
    assert canvas.overlay["bound_id"] == ann.id
    assert canvas.overlay["polygon"] == FAKE_POLY
    assert project.current_annotations[0].polygon is None
    assert model_ctrl.max_leases >= 1
    assert model_ctrl.leases == 0


def test_bind_with_polygon_does_not_predict_until_first_point():
    ctrl, project, ann_ctrl, canvas, calls, _, _ = _harness()
    ann = ann_ctrl.add_annotation(0, yolo_bbox_from_polygon(SQUARE), polygon=list(SQUARE))
    project.set_selection({ann.id})
    ctrl.on_tool_changed(Tool.SAM_CLICK)

    assert calls == []
    assert canvas.overlay["polygon"] == SQUARE

    ctrl.handle_click(0.3, 0.3, 1)
    assert len(calls) == 1
    assert calls[0]["mask_input"] is not None
    assert calls[0]["point_coords"] is not None
    assert canvas.overlay["points"][0][:2] == (0.3, 0.3)


def test_enter_refine_writes_same_id_and_undo_restores():
    ctrl, project, ann_ctrl, _, _, _, _ = _harness()
    origin_bbox = (0.4, 0.4, 0.4, 0.4)
    ann = ann_ctrl.add_annotation(0, origin_bbox)
    project.set_selection({ann.id})
    ctrl.on_tool_changed(Tool.SAM_CLICK)
    ctrl.handle_click(0.4, 0.4, 1)
    ctrl.commit()

    current = project.current_annotations[0]
    assert current.id == ann.id
    assert current.polygon == FAKE_POLY
    assert current.bbox == yolo_bbox_from_polygon(FAKE_POLY)

    ann_ctrl.undo()
    restored = project.current_annotations[0]
    assert restored.polygon is None
    assert restored.bbox == origin_bbox


def test_enter_create_adds_annotation_and_rejects_without_class():
    ctrl, project, _, _, calls, statuses, _ = _harness(class_id=None)
    ctrl.on_tool_changed(Tool.SAM_CLICK)
    ctrl.handle_click(0.5, 0.5, 1)
    assert calls == []
    assert MSG_NO_CLASS in "".join(statuses)
    assert project.current_annotations == []

    ctrl2, project2, _, _, calls2, _, _ = _harness(class_id=0)
    ctrl2.on_tool_changed(Tool.SAM_CLICK)
    ctrl2.handle_click(0.5, 0.5, 1)
    assert len(calls2) == 1
    assert calls2[0]["box_xyxy"] is None
    ctrl2.commit()
    assert len(project2.current_annotations) == 1
    created = project2.current_annotations[0]
    assert created.polygon == FAKE_POLY
    assert created.bbox == yolo_bbox_from_polygon(FAKE_POLY)


def test_discard_on_tool_change_does_not_write():
    ctrl, project, ann_ctrl, canvas, _, _, _ = _harness()
    ann = ann_ctrl.add_annotation(0, (0.4, 0.4, 0.4, 0.4))
    project.set_selection({ann.id})
    ctrl.on_tool_changed(Tool.SAM_CLICK)
    ctrl.handle_click(0.4, 0.4, 1)
    ctrl.on_tool_changed(Tool.SELECT)
    assert project.current_annotations[0].polygon is None
    assert canvas.overlay["polygon"] is None
    assert canvas.overlay["points"] == []


def test_stale_generation_does_not_overwrite_preview():
    ctrl, project, ann_ctrl, canvas, _, _, _ = _harness()
    ann = ann_ctrl.add_annotation(0, (0.4, 0.4, 0.4, 0.4))
    project.set_selection({ann.id})
    ctrl.on_tool_changed(Tool.SAM_CLICK)
    current_poly = list(canvas.overlay["polygon"])
    stale = {
        "polygon": [(0.1, 0.1), (0.2, 0.1), (0.2, 0.2)],
        "low_res_logits": None,
        "generation": 0,
    }
    ctrl.apply_predict_result(stale)
    assert canvas.overlay["polygon"] == current_poly


def test_no_predictor_uses_chinese_path():
    ctrl, _, ann_ctrl, _, _, statuses, _ = _harness(predictor=False)
    ann = ann_ctrl.add_annotation(0, (0.4, 0.4, 0.4, 0.4))
    ann_ctrl._project.set_selection({ann.id})
    ctrl.on_tool_changed(Tool.SAM_CLICK)
    assert any(MSG_NO_PREDICTOR in s for s in statuses)


def test_click_without_sam_prompts_load_message():
    ctrl, _, _, _, calls, statuses, model_ctrl = _harness(loaded=False, ask_load=lambda: False)
    ctrl.on_tool_changed(Tool.SAM_CLICK)
    ctrl.handle_click(0.5, 0.5, 1)
    assert calls == []
    assert model_ctrl.load_calls == 0
    assert any(MSG_SAM_NOT_LOADED in s for s in statuses)


def test_wait_create_right_click_on_box_binds_without_background_point():
    ctrl, project, ann_ctrl, canvas, calls, _, _ = _harness()
    ann = ann_ctrl.add_annotation(0, (0.4, 0.4, 0.4, 0.4))
    ctrl.on_tool_changed(Tool.SAM_CLICK)
    ctrl.handle_click(0.4, 0.4, 0)
    assert canvas.overlay["bound_id"] == ann.id
    assert canvas.overlay["points"] == []
    assert len(calls) == 1
    assert calls[0]["point_coords"] is None


def test_obb_kind_is_not_bound():
    ctrl, project, ann_ctrl, canvas, calls, statuses, _ = _harness()
    ann = ann_ctrl.add_annotation(0, (0.4, 0.4, 0.4, 0.4))
    ann.kind = "obb"
    project.set_selection({ann.id})
    ctrl.on_tool_changed(Tool.SAM_CLICK)
    assert calls == []
    ctrl.handle_click(0.4, 0.4, 1)
    assert calls == []
    assert canvas.overlay["bound_id"] is None
    assert project.current_annotations[0].polygon is None


def test_pose_kind_is_not_bound_or_created():
    ctrl, project, ann_ctrl, canvas, calls, statuses, _ = _harness()
    ann = ann_ctrl.add_annotation(0, (0.4, 0.4, 0.4, 0.4))
    ann.kind = "pose"
    project.set_selection({ann.id})
    ctrl.on_tool_changed(Tool.SAM_CLICK)
    assert calls == []
    ctrl.handle_click(0.4, 0.4, 1)
    assert calls == []
    assert canvas.overlay["bound_id"] is None
    assert any(MSG_BLOCKED_KIND in s for s in statuses)
    assert len(project.current_annotations) == 1


def test_create_first_click_sends_pick_click_xy():
    ctrl, project, _, _, calls, _, _ = _harness()
    ctrl.on_tool_changed(Tool.SAM_CLICK)
    ctrl.handle_click(0.5, 0.5, 1)
    assert project.current_annotations == []
    assert len(calls) == 1
    assert calls[0]["box_xyxy"] is None
    assert calls[0]["pick_click_xy"] == (50.0, 50.0)
    assert calls[0]["point_coords"] == [[50.0, 50.0]]
    assert calls[0]["point_labels"] == [1]


def test_refine_path_does_not_send_pick_click_xy():
    ctrl, project, ann_ctrl, _, calls, _, _ = _harness()
    ann = ann_ctrl.add_annotation(0, (0.4, 0.4, 0.4, 0.4))
    project.set_selection({ann.id})
    ctrl.on_tool_changed(Tool.SAM_CLICK)
    assert calls[0]["pick_click_xy"] is None
    ctrl.handle_click(0.4, 0.4, 1)
    assert calls[-1]["pick_click_xy"] is None
    assert calls[-1]["box_xyxy"] is not None


def test_click_miss_does_not_write_and_does_not_keep_failed_point():
    def predict(**kwargs):
        return {
            "polygon": [(0.0, 0.0), (1.0, 0.0), (1.0, 1.0)],
            "low_res_logits": None,
            "generation": kwargs.get("generation", 0),
            "click_miss": True,
        }

    ctrl, project, _, canvas, _, statuses, _ = _harness(predict=predict)
    flags: list = []
    ctrl.commit_available.connect(flags.append)
    ctrl.on_tool_changed(Tool.SAM_CLICK)
    ctrl.handle_click(0.5, 0.5, 1)
    assert canvas.overlay["polygon"] is None
    assert canvas.overlay["points"] == []
    assert project.current_annotations == []
    assert any(MSG_CLICK_MISS in s for s in statuses)
    ctrl.commit()
    assert project.current_annotations == []
    assert flags[-1] is False


def test_commit_available_true_only_with_uncommitted_preview():
    ctrl, _, _, _, _, statuses, _ = _harness()
    flags: list = []
    ctrl.commit_available.connect(flags.append)
    ctrl.on_tool_changed(Tool.SAM_CLICK)
    ctrl.handle_click(0.5, 0.5, 1)
    assert flags[-1] is True
    assert any(MSG_HINT_ACTIVE in s for s in statuses)
    ctrl.commit()
    assert flags[-1] is False


def test_after_commit_click_outside_starts_new_create():
    ctrl, project, _, canvas, calls, _, _ = _harness()
    ctrl.on_tool_changed(Tool.SAM_CLICK)
    ctrl.handle_click(0.4, 0.4, 1)
    ctrl.commit()
    assert len(project.current_annotations) == 1
    first_id = project.current_annotations[0].id
    assert canvas.overlay["bound_id"] == first_id
    n_calls = len(calls)
    ctrl.handle_click(0.9, 0.9, 1)
    assert canvas.overlay["bound_id"] is None
    assert len(calls) == n_calls + 1
    assert calls[-1]["box_xyxy"] is None
    assert calls[-1]["pick_click_xy"] == (90.0, 90.0)
    assert project.current_annotations[0].id == first_id
    ctrl.commit()
    assert len(project.current_annotations) == 2


def test_wiring_files_include_sam_click():
    shortcut = Path("src/gui/managers/shortcut_manager.py").read_text(encoding="utf-8")
    assert "from ..models.app_state import Tool" in shortcut
    assert 'self._add_shortcut("A"' in shortcut
    assert 'self._add_shortcut("Return"' in shortcut
    assert "sam_confirm_requested" in Path("src/gui/components/canvas_widget.py").read_text(encoding="utf-8")
    main = Path("src/gui/main_window.py").read_text(encoding="utf-8")
    assert "SamClickController" in main
    assert "SAM 点选" in main
    assert "确认分割" in main
    assert "commit_available" in main
    assert "sam_confirm_requested" in main
    explorer = Path("src/gui/components/explorer_panel.py").read_text(encoding="utf-8")
    assert "sam_click" in explorer
    assert "<b>A</b>" in main or "SAM 点选" in main
