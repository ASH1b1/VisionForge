from __future__ import annotations

import importlib
import logging
import os
import sys
import threading
from pathlib import Path
from typing import Any

from PySide6.QtCore import QObject, Signal

# torch 延迟导入（加速启动）
_torch = None
logger = logging.getLogger(__name__)


def _lazy_import_torch():
    global _torch
    if _torch is None:
        import torch as _t
        _torch = _t
    return _torch

from ..utils.segmentation_utils import mask_to_polygon, normalize_polygon, xyxy_pixels_to_yolo
from ..utils.sam_click_mask import select_mask_containing_click


def _get_project_root() -> Path:
    """获取项目根目录，兼容 PyInstaller 打包（sys._MEIPASS）和源码运行（__file__）。"""
    if getattr(sys, "frozen", False):
        # PyInstaller 打包后，所有资源存放在 sys._MEIPASS
        return Path(sys._MEIPASS)
    # 源码运行：sam3_model.py 位于 src/gui/models/sam3_model.py
    return Path(__file__).resolve().parents[3]


_project_root = _get_project_root()

DEFAULT_SAM3_CHECKPOINT = _project_root / "models" / "sam3" / "sam3" / "sam3.pt"
DEFAULT_SAM3_SOURCE_ROOT = _project_root / "sam3_src"
DEFAULT_CONFIDENCE_THRESHOLD = 0.5


class SAM3Model(QObject):
    supports_detection = False
    supports_segmentation = True

    loading_progress = Signal(str)
    loading_finished = Signal(bool)

    def __init__(self, checkpoint_path=None, source_root=None, parent=None):
        super().__init__(parent)
        self._checkpoint_path = Path(checkpoint_path) if checkpoint_path is not None else DEFAULT_SAM3_CHECKPOINT
        self._source_root = Path(source_root) if source_root is not None else DEFAULT_SAM3_SOURCE_ROOT
        self._confidence_threshold = DEFAULT_CONFIDENCE_THRESHOLD
        self._lock = threading.RLock()
        self._loaded = False
        self._runtime = None
        self._processor = None
        self._device = None
        self._load_error = None
        self._last_error = None
        self._last_alignment_info = None
        # Invalidate in-flight async load after unload
        self._load_id = 0
        self._interactive_state = None
        self._interactive_key = None

    def load(self):
        """Start SAM3 load on a background thread (non-blocking)."""
        with self._lock:
            self._load_id += 1
            load_id = self._load_id

        def _load():
            try:
                self.loading_progress.emit("检测设备...")
                self.loading_progress.emit("加载 SAM3 模型（这可能需要几分钟）...")
                success, message = self.load_sync(expected_load_id=load_id)
                with self._lock:
                    stale = load_id != self._load_id
                if stale:
                    # Always notify waiters — never leave QEventLoop hanging
                    self._last_error = self._last_error or "SAM3 load cancelled"
                    self.loading_finished.emit(False)
                    return
                if success:
                    self.loading_progress.emit(message or "SAM3 加载完成")
                    self.loading_finished.emit(True)
                else:
                    self.loading_progress.emit(f"加载失败: {message}")
                    self.loading_finished.emit(False)
            except Exception as exc:
                logger.error("SAM3 加载失败: %s", exc, exc_info=True)
                self._last_error = str(exc)
                self.loading_progress.emit(f"加载失败: {exc}")
                # Always emit so ensure/_wait_sam3_load cannot hang
                self.loading_finished.emit(False)

        threading.Thread(target=_load, daemon=True).start()

    def load_sync(self, expected_load_id: int | None = None):
        """Load SAM3 weights.

        If ``expected_load_id`` is set and no longer matches ``_load_id`` after
        heavy work (unload / superseded load), discard orphan weights and return
        ``(False, "cancelled")`` without committing GPU state.
        """
        with self._lock:
            self._runtime = None
            self._processor = None
            self._loaded = False
            self._load_error = None
            self._last_error = None

            if not self._checkpoint_path.exists():
                self._load_error = f"SAM3 checkpoint not found: {self._checkpoint_path}"
                return False, self._load_error

            if not (self._source_root / "sam3").exists():
                self._load_error = f"SAM3 source root not found: {self._source_root}"
                return False, self._load_error

            if expected_load_id is not None and expected_load_id != self._load_id:
                return False, "cancelled"

        # Heavy work outside lock so is_loaded()/unload can proceed (async UI path)
        runtime = None
        processor = None
        try:
            torch = _lazy_import_torch()
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            with self._lock:
                if expected_load_id is not None and expected_load_id != self._load_id:
                    return False, "cancelled"
                self._device = device
            runtime = self._build_runtime()
            processor = self._processor
            if processor is None:
                processor = runtime

            with self._lock:
                if expected_load_id is not None and expected_load_id != self._load_id:
                    # Unloaded / superseded — drop orphan weights
                    self._runtime = None
                    self._processor = None
                    self._loaded = False
                    self._device = None
                    self._load_error = "cancelled"
                    self._last_error = "SAM3 load cancelled"
                    return False, "cancelled"
                self._runtime = runtime
                self._processor = processor
                self._loaded = self._runtime is not None and self._processor is not None
                message = f"SAM3 loaded on {self._device}"
                return True, message
        except Exception as exc:
            with self._lock:
                self._runtime = None
                self._processor = None
                self._loaded = False
                self._load_error = str(exc)
                self._last_error = str(exc)
                return False, self._load_error

    def unload(self):
        """Unload model and invalidate in-flight async loads.

        Emits ``loading_finished(False)`` so nested wait loops always quit.
        """
        with self._lock:
            self._load_id += 1  # invalidate in-flight async load
            self._runtime = None
            self._processor = None
            self._loaded = False
            self._device = None
            self._last_error = self._last_error or "SAM3 unloaded"
            self._interactive_state = None
            self._interactive_key = None
        # Notify waiters (QueuedConnection → UI thread) even if background
        # thread would otherwise suppress finished due to stale load_id.
        self.loading_finished.emit(False)

    def is_loaded(self):
        with self._lock:
            return self._loaded and self._runtime is not None and self._processor is not None

    def get_last_error(self):
        return self._last_error or self._load_error

    def get_confidence_threshold(self):
        return self._confidence_threshold

    def set_confidence_threshold(self, threshold: float, reload_processor: bool = False):
        need_reload = False
        with self._lock:
            self._confidence_threshold = float(threshold)
            processor = self._processor
            if processor is None:
                return
            # Mutate processor under the same lock as infer_image/align_detections
            if hasattr(processor, "set_confidence_threshold"):
                processor.set_confidence_threshold(self._confidence_threshold)
                return
            need_reload = bool(reload_processor)
        if need_reload:
            self.load_sync()

    def get_last_alignment_info(self):
        return self._last_alignment_info or {}

    def _ensure_source_path(self):
        sam3_package_dir = self._source_root / "sam3"
        if not sam3_package_dir.exists():
            raise FileNotFoundError(f"SAM3 source root not found: {self._source_root}")

        source_root_str = str(self._source_root)
        if source_root_str not in sys.path:
            sys.path.insert(0, source_root_str)

    def _import_builder(self):
        self._ensure_source_path()
        builder_module = importlib.import_module("sam3.model_builder")
        processor_module = importlib.import_module("sam3.model.sam3_image_processor")
        return builder_module.build_sam3_image_model, processor_module.Sam3Processor

    def _checkpoint_has_tracker_weights(self) -> bool:
        """True if checkpoint contains tracker.* keys needed for predict_inst."""
        torch = _lazy_import_torch()
        path = self._checkpoint_path
        if path is None or not Path(path).exists():
            return False
        try:
            with open(path, "rb") as f:
                ckpt = torch.load(f, map_location="cpu", weights_only=True)
            if isinstance(ckpt, dict) and "model" in ckpt and isinstance(ckpt["model"], dict):
                ckpt = ckpt["model"]
            if not isinstance(ckpt, dict):
                return False
            return any("tracker" in str(k) for k in ckpt.keys())
        except Exception:
            return False

    def _build_runtime(self):
        build_sam3_image_model, Sam3Processor = self._import_builder()
        # 显式传入 BPE 路径，避免 pkg_resources.resource_filename 在 PyInstaller 中失败
        _bpe_path = str(self._source_root / "sam3" / "assets" / "bpe_simple_vocab_16e6.txt.gz")
        if not os.path.exists(_bpe_path):
            _bpe_path = None  # 回退到 pkg_resources
        # SAM3 内部 _setup_device_and_mode 使用 device == "cuda" 字符串比较，
        # 但 self._device 是 torch.device 对象，比较会失败导致模型留在 CPU。
        # 这里传入字符串形式，并在构造后显式 .to(device) 确保所有参数迁移。
        common_kwargs = dict(
            bpe_path=_bpe_path,
            checkpoint_path=str(self._checkpoint_path),
            device=str(self._device),
            eval_mode=True,
            load_from_HF=False,
        )
        # Fail closed: only enable box prompts when checkpoint has tracker weights
        want_inst = self._checkpoint_has_tracker_weights()
        try:
            runtime = build_sam3_image_model(
                **common_kwargs,
                enable_inst_interactivity=want_inst,
            )
        except Exception as exc:
            self._last_error = f"inst_interactivity unavailable, fallback: {exc}"
            runtime = build_sam3_image_model(
                **common_kwargs,
                enable_inst_interactivity=False,
            )
        # Guard against predictor present but unusable after partial load
        if (
            want_inst
            and getattr(runtime, "inst_interactive_predictor", None) is not None
            and not self._checkpoint_has_tracker_weights()
        ):
            runtime.inst_interactive_predictor = None
        runtime = runtime.to(self._device)
        self._processor = Sam3Processor(
            runtime,
            device=str(self._device),
            confidence_threshold=self._confidence_threshold,
        )
        return runtime

    def infer_image(self, image, prompt: str):
        with self._lock:
            if not self.is_loaded():
                return None
            processor = self._processor

        try:
            state = processor.set_image(image)
            return processor.set_text_prompt(state=state, prompt=prompt)
        except Exception as exc:
            self._last_error = str(exc)
            return None

    def segment(self, image, boxes):
        width, height = self._resolve_image_size(image)

        if not width or not height:
            return []

        results = []
        for box in boxes:
            if hasattr(box, "shape"):
                polygon = mask_to_polygon(box)
                results.append(normalize_polygon(polygon, width, height))
            else:
                results.append([])
        return results

    def _has_inst_predictor(self) -> bool:
        runtime = self._runtime
        return (
            runtime is not None
            and getattr(runtime, "inst_interactive_predictor", None) is not None
            and hasattr(runtime, "predict_inst")
        )

    def has_interactive_predictor(self) -> bool:
        with self._lock:
            return self.is_loaded() and self._has_inst_predictor()

    def clear_interactive_cache(self) -> None:
        with self._lock:
            self._interactive_state = None
            self._interactive_key = None

    @staticmethod
    def _as_mask_input(mask_input):
        import numpy as np

        if mask_input is None:
            return None
        arr = np.asarray(mask_input)
        if arr.size == 0:
            return None
        if arr.ndim == 3:
            arr = arr[0]
        if arr.ndim != 2:
            return None
        return arr[None, ...].astype(np.float32)

    def predict_interactive(
        self,
        image,
        *,
        box_xyxy=None,
        point_coords=None,
        point_labels=None,
        mask_input=None,
        generation: int = 0,
        frame_key=None,
        pick_click_xy=None,
    ) -> dict:
        """Run one interactive predict_inst. Reuses set_image for the same frame_key."""
        import numpy as np

        with self._lock:
            if not self.is_loaded() or self._processor is None or self._runtime is None:
                raise RuntimeError("SAM3 未加载")
            if not self._has_inst_predictor():
                raise RuntimeError("当前 SAM3 权重不支持点选修正（缺少 interactive predictor）")

            width, height = self._resolve_image_size(image)
            if not width or not height:
                raise RuntimeError("invalid image size")

            key = frame_key if frame_key is not None else (id(image), int(width), int(height))
            if self._interactive_key != key or self._interactive_state is None:
                self._interactive_state = self._processor.set_image(image)
                self._interactive_key = key

            kwargs: dict[str, Any] = {"multimask_output": bool(pick_click_xy is not None)}
            if box_xyxy is not None:
                kwargs["box"] = np.asarray(box_xyxy, dtype=np.float32)
            if point_coords is not None and point_labels is not None:
                coords = np.asarray(point_coords, dtype=np.float32)
                labels = np.asarray(point_labels, dtype=np.int32)
                if coords.size:
                    kwargs["point_coords"] = coords
                    kwargs["point_labels"] = labels
                    kwargs["normalize_coords"] = False
            mask = self._as_mask_input(mask_input)
            if mask is not None:
                kwargs["mask_input"] = mask

            masks, ious, low_res = self._runtime.predict_inst(
                self._interactive_state,
                **kwargs,
            )
            click_miss = False
            if pick_click_xy is not None:
                best, idx = select_mask_containing_click(masks, pick_click_xy)
                if best is None:
                    click_miss = True
                    polygon = []
                    logits = None
                else:
                    polygon = normalize_polygon(mask_to_polygon(best), width, height) or []
                    logits = None
                    if low_res is not None:
                        arr = np.asarray(low_res)
                        if arr.ndim >= 3 and idx is not None and arr.shape[0] > idx:
                            logits = arr[idx]
                        else:
                            logits = arr
            else:
                best = self._select_best_mask(masks, ious)
                polygon = []
                if best is not None:
                    polygon = normalize_polygon(mask_to_polygon(best), width, height) or []
                logits = None
                if low_res is not None:
                    logits = np.asarray(low_res)
            return {
                "polygon": polygon,
                "low_res_logits": logits,
                "generation": generation,
                "click_miss": click_miss,
            }

    def align_detections(self, image, prompt: str, detections, iou_threshold: float = 0.5):
        """Align DINO boxes to polygons via SAM3.

        Preferred path: set_image once + predict_inst(box) per detection.
        Fallback: legacy text grounding + IoU match when inst predictor is unavailable.
        """
        if not detections:
            self._last_alignment_info = {
                "sam_box_count": 0,
                "sam_mask_count": 0,
                "matched_count": 0,
                "alignment_iou_threshold": iou_threshold,
                "alignment_mode": "empty",
                "error": None,
            }
            return []

        with self._lock:
            if not self.is_loaded():
                self._last_alignment_info = {
                    "sam_box_count": 0,
                    "sam_mask_count": 0,
                    "matched_count": 0,
                    "alignment_iou_threshold": iou_threshold,
                    "alignment_mode": "unloaded",
                    "error": self.get_last_error(),
                }
                return [None for _ in detections]
            use_box = self._has_inst_predictor()

        if use_box:
            try:
                return self._align_via_box_prompts(image, detections, iou_threshold)
            except Exception as exc:
                self._last_error = str(exc)
                # Fall through to text path

        return self._align_via_text_grounding(image, prompt, detections, iou_threshold)

    def _align_via_box_prompts(self, image, detections, iou_threshold: float = 0.5):
        import numpy as np

        # Hold lock for the whole box-align path so unload cannot drop runtime mid-call
        with self._lock:
            processor = self._processor
            runtime = self._runtime
            if processor is None or runtime is None or not self._has_inst_predictor():
                raise RuntimeError("SAM3 box predictor unavailable")

            state = processor.set_image(image)
            width, height = self._resolve_image_size(image)
            if not width or not height:
                self._last_alignment_info = {
                    "sam_box_count": 0,
                    "sam_mask_count": 0,
                    "matched_count": 0,
                    "alignment_iou_threshold": iou_threshold,
                    "alignment_mode": "box_prompt",
                    "error": "invalid image size",
                }
                return [None for _ in detections]

            matched_polygons = []
            for detection in detections:
                xyxy = self._yolo_bbox_to_xyxy(detection, width, height)
                box = np.asarray(xyxy, dtype=np.float32)
                masks, ious, _ = runtime.predict_inst(
                    state,
                    box=box,
                    multimask_output=False,
                )
                mask = self._select_best_mask(masks, ious)
                if mask is None:
                    matched_polygons.append(None)
                    continue
                polygon = mask_to_polygon(mask)
                normalized = normalize_polygon(polygon, width, height)
                matched_polygons.append(normalized or None)

            matched_count = sum(1 for polygon in matched_polygons if polygon)
            self._last_alignment_info = {
                "sam_box_count": len(detections),
                "sam_mask_count": matched_count,
                "matched_count": matched_count,
                "alignment_iou_threshold": iou_threshold,
                "alignment_mode": "box_prompt",
                "error": None,
            }
            return matched_polygons

    @staticmethod
    def _select_best_mask(masks, ious):
        import numpy as np

        if masks is None:
            return None
        arr = np.asarray(masks)
        if arr.size == 0:
            return None
        if arr.ndim == 2:
            return arr
        if arr.ndim == 3:
            if ious is not None and len(ious) == arr.shape[0]:
                best_idx = int(np.argmax(np.asarray(ious)))
            else:
                best_idx = 0
            return arr[best_idx]
        return None

    def _align_via_text_grounding(self, image, prompt: str, detections, iou_threshold: float = 0.5):
        inference = self.infer_image(image=image, prompt=prompt)
        if not inference:
            self._last_alignment_info = {
                "sam_box_count": 0,
                "sam_mask_count": 0,
                "matched_count": 0,
                "alignment_iou_threshold": iou_threshold,
                "alignment_mode": "text_grounding",
                "error": self.get_last_error(),
            }
            return [None for _ in detections]

        width, height = self._resolve_image_size(image)
        if not width or not height:
            return [None for _ in detections]

        # 避免空 tensor 的布尔求值错误（"Boolean value of Tensor with no values is ambiguous"）
        sam_masks = list(inference.get("masks")) if inference.get("masks") is not None else []
        sam_boxes = list(inference.get("boxes")) if inference.get("boxes") is not None else []
        matched_polygons = []
        used_indices = set()

        for detection in detections:
            detection_box = self._yolo_bbox_to_xyxy(detection, width, height)
            best_index = None
            best_iou = 0.0

            for index, sam_box in enumerate(sam_boxes):
                if index in used_indices:
                    continue
                iou = self._compute_iou(detection_box, self._coerce_box(sam_box))
                if iou > best_iou:
                    best_iou = iou
                    best_index = index

            if best_index is None or best_iou < iou_threshold:
                matched_polygons.append(None)
                continue

            used_indices.add(best_index)
            polygon = mask_to_polygon(sam_masks[best_index]) if best_index < len(sam_masks) else []
            normalized = normalize_polygon(polygon, width, height)
            matched_polygons.append(normalized or None)

        self._last_alignment_info = {
            "sam_box_count": len(sam_boxes),
            "sam_mask_count": len(sam_masks),
            "matched_count": sum(1 for polygon in matched_polygons if polygon),
            "alignment_iou_threshold": iou_threshold,
            "alignment_mode": "text_grounding",
            "error": self.get_last_error(),
        }
        return matched_polygons

    def outputs_to_shapes(self, output, image):
        width, height = self._resolve_image_size(image)
        if not width or not height or not output:
            return []

        masks = list(output.get("masks")) if output.get("masks") is not None else []
        boxes = list(output.get("boxes")) if output.get("boxes") is not None else []
        scores = list(output.get("scores")) if output.get("scores") is not None else []
        shapes = []

        for index, box in enumerate(boxes):
            polygon = []
            if index < len(masks):
                polygon = normalize_polygon(mask_to_polygon(masks[index]), width, height)
            shapes.append({
                "polygon": polygon or None,
                "bbox": xyxy_pixels_to_yolo(self._coerce_box(box), width, height),
                "score": scores[index] if index < len(scores) else None,
                "xyxy": self._coerce_box(box),
            })

        return shapes

    def _resolve_image_size(self, image):
        # Prefer PIL-style size when available
        width = getattr(image, "width", None)
        height = getattr(image, "height", None)
        if callable(width):
            width = width()
        if callable(height):
            height = height()
        if width and height:
            return width, height

        # ndarray / torch.Tensor: use spatial dims (HWC or CHW → last two)
        if hasattr(image, "shape"):
            shape = tuple(image.shape)
            if len(shape) >= 2:
                h, w = int(shape[-2]), int(shape[-1])
                return w, h
        return None, None

    def _yolo_bbox_to_xyxy(self, bbox, width, height):
        cx, cy, box_width, box_height = bbox
        half_width = (box_width * width) / 2.0
        half_height = (box_height * height) / 2.0
        center_x = cx * width
        center_y = cy * height
        return (
            center_x - half_width,
            center_y - half_height,
            center_x + half_width,
            center_y + half_height,
        )

    def _coerce_box(self, box):
        torch = _lazy_import_torch()
        if torch.is_tensor(box):
            box = box.detach().cpu().tolist()
        return tuple(float(value) for value in box)

    def _compute_iou(self, left, right):
        if left is None or right is None:
            return 0.0

        left_x1, left_y1, left_x2, left_y2 = left
        right_x1, right_y1, right_x2, right_y2 = right
        inter_x1 = max(left_x1, right_x1)
        inter_y1 = max(left_y1, right_y1)
        inter_x2 = min(left_x2, right_x2)
        inter_y2 = min(left_y2, right_y2)

        inter_width = max(0.0, inter_x2 - inter_x1)
        inter_height = max(0.0, inter_y2 - inter_y1)
        intersection = inter_width * inter_height
        if intersection <= 0:
            return 0.0

        left_area = max(0.0, left_x2 - left_x1) * max(0.0, left_y2 - left_y1)
        right_area = max(0.0, right_x2 - right_x1) * max(0.0, right_y2 - right_y1)
        union = left_area + right_area - intersection
        if union <= 0:
            return 0.0
        return intersection / union
