"""ONNX YOLO detector wrapper. Optional onnxruntime — do not import it at module top."""
from __future__ import annotations

import ast
import json
import logging
import threading
from pathlib import Path
from typing import Any, Optional

import cv2
import numpy as np
from PySide6.QtCore import QObject, Signal

from .detector_protocol import ONNX_RUNTIME_MISSING_MESSAGE, names_to_class_list
from .grounding_dino_model import DetectionResult
from .onnx_yolo_decode import (
    class_names_from_onnx_path,
    decode_ultralytics_detect,
    fallback_class_names,
    letterbox_rgb,
    parse_names_metadata,
)

logger = logging.getLogger(__name__)


def _positive_int(value: Any) -> int | None:
    try:
        number = int(value)
    except (TypeError, ValueError):
        return None
    return number if number > 0 else None


def _hw_from_input_shape(shape: Any) -> tuple[int, int] | None:
    if shape is None:
        return None
    dims = list(shape)
    if len(dims) < 2:
        return None
    height = _positive_int(dims[-2])
    width = _positive_int(dims[-1])
    if height is None or width is None:
        return None
    return height, width


def _hw_from_imgsz(raw: Any) -> tuple[int, int] | None:
    if raw is None:
        return None
    if isinstance(raw, (list, tuple)):
        if len(raw) >= 2:
            height = _positive_int(raw[0])
            width = _positive_int(raw[1])
            if height and width:
                return height, width
        if len(raw) == 1:
            size = _positive_int(raw[0])
            if size:
                return size, size
    size = _positive_int(raw)
    if size:
        return size, size
    text = str(raw).strip()
    if not text:
        return None
    data: Any = None
    try:
        data = json.loads(text)
    except Exception:
        try:
            data = ast.literal_eval(text)
        except Exception:
            data = None
    if data is not None and data != raw:
        return _hw_from_imgsz(data)
    cleaned = text.strip("[]() ").replace("x", ",").replace("X", ",")
    parts = [p for p in cleaned.split(",") if p]
    if len(parts) >= 2:
        height = _positive_int(parts[0])
        width = _positive_int(parts[1])
        if height and width:
            return height, width
    size = _positive_int(cleaned)
    if size:
        return size, size
    return None


def _nc_from_output_shape(shape: Any) -> int | None:
    if shape is None:
        return None
    dims = [int(d) for d in list(shape) if _positive_int(d) is not None]
    while len(dims) > 2 and dims[0] == 1:
        dims = dims[1:]
    if len(dims) < 2:
        return None
    feat = min(dims[0], dims[1])
    if feat >= 5:
        return feat - 4
    return None


class OnnxYoloModel(QObject):
    """Closed-set YOLO detection ONNX. Pose/OBB weights are out of scope."""

    supports_detection = True
    supports_segmentation = False

    loading_progress = Signal(str)
    loading_finished = Signal(bool)
    inference_completed = Signal(object)

    def __init__(self, weights_path: str = "", parent=None):
        super().__init__(parent)
        self.weights_path = weights_path
        self.model = None
        self._loaded = False
        self._lock = threading.RLock()
        self._load_id = 0
        self._names: dict[int, str] = {}
        self._last_error: Optional[str] = None
        self._input_name = "images"
        self._input_hw: tuple[int, int] = (640, 640)

    @staticmethod
    def is_available() -> bool:
        try:
            import onnxruntime  # noqa: F401
        except ImportError:
            return False
        return True

    def get_last_error(self) -> Optional[str]:
        return self._last_error

    def get_names(self) -> dict[int, str]:
        return dict(self._names)

    def class_names_list(self) -> list[str]:
        return names_to_class_list(self._names)

    def is_loaded(self) -> bool:
        return self._loaded and self.model is not None

    def load(self) -> None:
        with self._lock:
            self._load_id += 1
            load_id = self._load_id

        def _load() -> None:
            ok = self.load_sync(expected_load_id=load_id)
            self.loading_finished.emit(ok)

        threading.Thread(target=_load, daemon=True).start()

    def load_sync(self, expected_load_id: int | None = None) -> bool:
        self.loading_progress.emit("正在加载 YOLO ONNX…")
        try:
            import onnxruntime as ort
        except ImportError:
            self._last_error = ONNX_RUNTIME_MISSING_MESSAGE
            self._loaded = False
            self.model = None
            return False
        try:
            session = self._create_session(ort, self.weights_path)
            input0 = session.get_inputs()[0]
            input_name = getattr(input0, "name", None) or "images"
            metadata: dict[str, Any] = {}
            getter = getattr(session, "get_modelmeta", None)
            if callable(getter):
                meta = getter()
                metadata = dict(getattr(meta, "custom_metadata_map", None) or {})
            input_hw = _hw_from_input_shape(getattr(input0, "shape", None))
            if input_hw is None:
                input_hw = _hw_from_imgsz(metadata.get("imgsz"))
            if input_hw is None:
                input_hw = (640, 640)
            names = parse_names_metadata(metadata.get("names"))
            if not names:
                names = class_names_from_onnx_path(self.weights_path)
            if not names:
                out_shape = None
                get_outputs = getattr(session, "get_outputs", None)
                if callable(get_outputs):
                    out_list = get_outputs()
                    if out_list:
                        out_shape = getattr(out_list[0], "shape", None)
                nc = _nc_from_output_shape(out_shape)
                if nc:
                    names = fallback_class_names(nc)
            with self._lock:
                if expected_load_id is not None and expected_load_id != self._load_id:
                    return False
                self.model = session
                self._input_name = input_name
                self._input_hw = input_hw
                self._names = names
                self._loaded = True
                self._last_error = None
            return True
        except Exception as exc:
            logger.error("ONNX YOLO load failed: %s", exc, exc_info=True)
            self._last_error = str(exc)
            with self._lock:
                self.model = None
                self._loaded = False
            return False

    @staticmethod
    def _create_session(ort, path: str):
        providers = ["CUDAExecutionProvider", "CPUExecutionProvider"]
        try:
            return ort.InferenceSession(path, providers=providers)
        except Exception:
            return ort.InferenceSession(path, providers=["CPUExecutionProvider"])

    def unload(self) -> None:
        with self._lock:
            self._load_id += 1
            self.model = None
            self._loaded = False
            self._names = {}
        try:
            self.loading_finished.emit(False)
        except RuntimeError:
            pass

    def infer(
        self,
        image,
        text_prompt: str = "",
        box_threshold: float = 0.25,
        text_threshold: float = 0.25,
        use_amp: bool = True,
        **kwargs,
    ) -> Optional[DetectionResult]:
        del text_prompt, text_threshold, use_amp, kwargs
        with self._lock:
            if not self._loaded or self.model is None:
                logger.warning("YOLO ONNX model not loaded, inference skipped")
                return None
            session = self.model
            input_name = self._input_name
            input_hw = self._input_hw
        try:
            rgb = self._to_rgb_uint8(image)
            if rgb is None:
                logger.warning("YOLO ONNX could not convert image")
                return None
            orig_h, orig_w = rgb.shape[:2]
            letterboxed, scale, pad = letterbox_rgb(rgb, input_hw)
            blob = letterboxed.astype(np.float32) / 255.0
            blob = np.transpose(blob, (2, 0, 1))[None, ...]
            outputs = session.run(None, {input_name: blob})
            boxes, labels, scores = decode_ultralytics_detect(
                outputs[0],
                conf_threshold=float(box_threshold),
                scale=scale,
                pad=pad,
                orig_wh=(orig_w, orig_h),
            )
            return DetectionResult(
                boxes=boxes,
                labels=labels,
                scores=scores,
                image_size=(orig_w, orig_h),
            )
        except Exception as exc:
            logger.error("ONNX YOLO inference failed: %s", exc, exc_info=True)
            return None

    @staticmethod
    def _to_rgb_uint8(image) -> Optional[np.ndarray]:
        try:
            from PIL import Image as PILImage
        except ImportError:
            PILImage = None
        if isinstance(image, (str, Path)):
            path = str(image)
            data = np.fromfile(path, dtype=np.uint8)
            bgr = cv2.imdecode(data, cv2.IMREAD_COLOR) if data.size else None
            if bgr is None:
                bgr = cv2.imread(path)
            if bgr is None:
                return None
            return np.ascontiguousarray(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB))
        if PILImage is not None and isinstance(image, PILImage.Image):
            return np.ascontiguousarray(np.array(image.convert("RGB")))
        if isinstance(image, np.ndarray):
            arr = image
            if arr.ndim == 2:
                arr = np.stack([arr, arr, arr], axis=-1)
            elif arr.ndim == 3 and arr.shape[2] == 4:
                arr = arr[:, :, :3]
            elif arr.ndim == 3 and arr.shape[2] == 3:
                arr = cv2.cvtColor(arr, cv2.COLOR_BGR2RGB)
            else:
                return None
            if arr.dtype != np.uint8:
                arr = np.clip(arr, 0, 255).astype(np.uint8)
            return np.ascontiguousarray(arr)
        return None
