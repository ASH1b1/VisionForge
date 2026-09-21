from __future__ import annotations

from src.gui.models.detector_protocol import (
    DETECTOR_GROUNDING_DINO,
    DETECTOR_IDS,
    DETECTOR_ONNX_YOLO,
    UNMATCHED_CREATE,
    UNMATCHED_SKIP,
    filter_detections_by_project_classes,
    names_to_class_list,
    resolve_detector_handle,
)
from src.gui.models.grounding_dino_model import DetectionResult
from src.utils.annotation_processor import process_raw_detections


class FakeDetector:
    def __init__(self, detector_id: str = "fake"):
        self.detector_id = detector_id
        self._loaded = False
        self.unload_calls = 0
        self.names = {0: "person", 1: "car"}

    def is_loaded(self) -> bool:
        return self._loaded

    def load(self) -> None:
        self._loaded = True

    def load_sync(self) -> bool:
        self._loaded = True
        return True

    def unload(self) -> None:
        self._loaded = False
        self.unload_calls += 1

    def infer(self, image, **kwargs):
        return DetectionResult(
            boxes=[(10.0, 10.0, 50.0, 50.0), (60.0, 10.0, 90.0, 40.0)],
            labels=[0, 1],
            scores=[0.91, 0.88],
            image_size=(100, 100),
        )

    def get_names(self):
        return dict(self.names)


def test_detector_ids_exclude_locate_anything():
    assert DETECTOR_GROUNDING_DINO == "grounding_dino"
    assert DETECTOR_ONNX_YOLO == "onnx_yolo"
    assert DETECTOR_IDS == (
        DETECTOR_GROUNDING_DINO,
        DETECTOR_ONNX_YOLO,
    )
    assert "locate_anything" not in DETECTOR_IDS


def test_fake_detector_infer_goes_through_processor_and_class_mapping():
    fake = FakeDetector()
    result = fake.infer(None)
    class_names = names_to_class_list(fake.get_names())
    processed = process_raw_detections(
        boxes=result.boxes,
        labels=result.labels,
        scores=result.scores,
        class_names=class_names,
        img_width=100,
        img_height=100,
        nms_threshold=1.0,
    )
    mapped = filter_detections_by_project_classes(
        processed,
        {0: "Person"},
        unmatched=UNMATCHED_CREATE,
    )
    names = [row[0] for row in mapped]
    assert "Person" in names
    assert "car" in names


def test_unmatched_skip_drops_unknown_classes():
    processed = [
        ("person", (0.3, 0.3, 0.2, 0.2), 0.9),
        ("cat", (0.7, 0.7, 0.1, 0.1), 0.8),
    ]
    mapped = filter_detections_by_project_classes(
        processed,
        {0: "person"},
        unmatched=UNMATCHED_SKIP,
    )
    assert mapped == [("person", (0.3, 0.3, 0.2, 0.2), 0.9)]


def test_resolve_detector_handle_prefers_loaded_model_ctrl():
    yolo = FakeDetector("yolo")
    yolo._loaded = True
    grounding = FakeDetector("gdino")
    grounding._loaded = True

    class Ctrl:
        def get_loaded_detector(self):
            return yolo

    handle = resolve_detector_handle(
        grounding=grounding,
        yolo=None,
        model_ctrl=Ctrl(),
    )
    assert handle is yolo


def test_model_controller_load_unloads_other_detectors():
    from src.gui.controllers.model_controller import ModelController

    ctrl = ModelController()
    gdino = FakeDetector("gdino")
    gdino._loaded = True
    yolo = FakeDetector("yolo")
    ctrl.register_detector(DETECTOR_GROUNDING_DINO, gdino)
    ctrl.register_detector(DETECTOR_ONNX_YOLO, yolo)

    assert ctrl.get_active_detector() == DETECTOR_GROUNDING_DINO
    assert ctrl.load_detector(DETECTOR_ONNX_YOLO, sync=True) is True
    assert gdino.unload_calls == 1
    assert not gdino.is_loaded()
    assert yolo.is_loaded()
    assert ctrl.get_active_detector() == DETECTOR_ONNX_YOLO
    assert ctrl.get_loaded_detector() is yolo
