from __future__ import annotations

import builtins
import sys
import types

import numpy as np

from src.gui.models.detector_protocol import ONNX_RUNTIME_MISSING_MESSAGE
from src.gui.models.grounding_dino_model import DetectionResult
from src.gui.models.onnx_yolo_model import OnnxYoloModel
from src.utils.annotation_processor import process_raw_detections


def test_module_imports_without_onnxruntime(monkeypatch):
    real_import = builtins.__import__

    def fake_import(name, *args, **kwargs):
        if name == "onnxruntime" or name.startswith("onnxruntime."):
            raise ImportError("mocked missing onnxruntime")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", fake_import)
    assert OnnxYoloModel.is_available() is False


def test_module_import_does_not_load_onnxruntime():
    sys.modules.pop("onnxruntime", None)
    import importlib

    import src.gui.models.onnx_yolo_model as mod

    importlib.reload(mod)
    assert "onnxruntime" not in sys.modules


def test_load_sync_fails_without_onnxruntime(monkeypatch):
    real_import = builtins.__import__

    def fake_import(name, *args, **kwargs):
        if name == "onnxruntime" or name.startswith("onnxruntime."):
            raise ImportError("mocked missing onnxruntime")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", fake_import)
    model = OnnxYoloModel("best.onnx")
    assert model.load_sync() is False
    assert model.is_loaded() is False
    assert model.get_last_error() == ONNX_RUNTIME_MISSING_MESSAGE


class _FakeInput:
    name = "images"
    shape = [1, 3, 640, 640]


class _FakeSession:
    def __init__(self, path, providers=None):
        self.path = path
        self.providers = providers

    def get_inputs(self):
        return [_FakeInput()]

    def get_modelmeta(self):
        meta = types.SimpleNamespace(
            custom_metadata_map={
                "names": "{0: 'a', 1: 'person'}",
            }
        )
        return meta

    def run(self, _output_names, _feeds):
        pred = np.zeros((1, 6, 1), dtype=np.float32)
        pred[0, :, 0] = [320, 320, 64, 64, 0.1, 0.9]
        return [pred]


def test_infer_with_fake_session(tmp_path, monkeypatch):
    onnx_path = tmp_path / "best.onnx"
    onnx_path.write_bytes(b"")

    fake_ort = types.ModuleType("onnxruntime")
    fake_ort.InferenceSession = _FakeSession
    monkeypatch.setitem(sys.modules, "onnxruntime", fake_ort)

    model = OnnxYoloModel(str(onnx_path))
    assert model.load_sync() is True
    assert model.is_loaded()
    assert model.get_names()[1] == "person"

    image = np.zeros((640, 640, 3), dtype=np.uint8)
    result = model.infer(
        image=image,
        text_prompt="",
        box_threshold=0.25,
        text_threshold=0.25,
    )
    assert isinstance(result, DetectionResult)
    assert result.labels == [1]
    assert abs(result.scores[0] - 0.9) < 1e-5
    x1, y1, x2, y2 = result.boxes[0]
    assert abs(x1 - 288) < 1.0 and abs(y2 - 352) < 1.0
    assert result.image_size == (640, 640)

    processed = process_raw_detections(
        boxes=result.boxes,
        labels=result.labels,
        scores=result.scores,
        class_names=model.class_names_list(),
        img_width=640,
        img_height=640,
        nms_threshold=1.0,
    )
    assert processed[0][0] == "person"
