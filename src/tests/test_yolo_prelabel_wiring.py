from __future__ import annotations

from PySide6.QtGui import QImage

from src.gui.controllers.detect_controller import detections_from_raw
from src.gui.dialogs.detect_dialog import DetectDialog
from src.gui.models.detector_protocol import UNMATCHED_SKIP


class _FakeYolo:
    def is_loaded(self) -> bool:
        return True

    def get_names(self):
        return {0: "person"}

    def class_names_list(self):
        return ["person"]


class _FakeCtrl:
    def get_loaded_detector(self):
        return _FakeYolo()

    def get_active_detector(self):
        return "onnx_yolo"


def test_detections_from_raw_maps_closed_set_names_and_skips():
    results = detections_from_raw(
        boxes=[(10.0, 10.0, 50.0, 50.0), (60.0, 10.0, 90.0, 40.0)],
        labels=[0, 1],
        scores=[0.9, 0.8],
        prompt="",
        img_width=100,
        img_height=100,
        nms_threshold=1.0,
        class_names=["person", "car"],
        project_class_names={0: "Person"},
        unmatched=UNMATCHED_SKIP,
    )
    assert len(results) == 1
    assert results[0].class_name == "Person"


def test_detect_dialog_accepts_yolo_without_text_prompt(qapp):
    image = QImage(8, 8, QImage.Format.Format_RGB32)
    dialog = DetectDialog(
        image=image,
        model_ctrl=_FakeCtrl(),
        project_class_names={0: "person"},
    )
    assert dialog._active_detector() is not None
    assert dialog._requires_text_prompt() is False
    assert dialog.get_unmatched_policy() == "create"
