"""ModelController — unified interface for AI model operations."""
from __future__ import annotations
import logging
import threading
from typing import Any, Dict, List, Optional
from PySide6.QtCore import QEventLoop, QObject, QTimer
from PySide6.QtWidgets import QMessageBox, QWidget
from ..models.detector_protocol import (
    DETECTOR_GROUNDING_DINO,
    DETECTOR_IDS,
    DETECTOR_ONNX_YOLO,
)
from ..models.grounding_dino_model import GroundingDINOModel
from ..models.sam3_model import SAM3Model

logger = logging.getLogger(__name__)

# Conservative SAM3 estimate: ~3.5GB weights + ~1.0GB activation headroom
_SAM3_VRAM_ESTIMATE_GB = 4.5
# Max time ensure_sam3_loaded will wait for async load (ms)
_SAM3_LOAD_WAIT_TIMEOUT_MS = 600_000

INFERENCE_BUSY_MESSAGE = "推理进行中，无法卸载或切换模型。请等待当前标注完成后再试。"


def _lazy_import_torch():
    import torch
    return torch


class ModelController(QObject):
    """Coordinates model loading, inference, and status queries.
    
    Detection engine mutual exclusion:
      - Only one detector id may be loaded (grounding_dino / onnx_yolo).
        Loading one unloads the others.

    VRAM confirmation:
      - ensure_sam3_loaded() shows allocated/free + estimate; OOM unloads SAM.
    """

    def __init__(self, parent: QObject | None = None):
        super().__init__(parent)
        self._grounding: Optional[GroundingDINOModel] = None
        self._sam: Optional[SAM3Model] = None
        self._onnx_yolo: Optional[Any] = None
        # Loading state flags for mutual exclusion (Bug 2 fix)
        self._grounding_loading: bool = False
        self._onnx_yolo_loading: bool = False
        self._sam_loading: bool = False
        self._last_sam_error: Optional[str] = None
        # Nested QEventLoops waiting on SAM3 load — quit on unload/timeout
        self._sam_wait_loops: List[QEventLoop] = []
        # Inference leases: workers hold while running so unload/switch cannot drop weights
        self._inference_leases: int = 0
        # RLock: load may call unload for mutual exclusion under the same thread
        self._inference_lock = threading.RLock()

    # ── Inference leases ─────────────────────────────────────────────

    def acquire_inference_lease(self) -> None:
        with self._inference_lock:
            self._inference_leases += 1

    def release_inference_lease(self) -> None:
        with self._inference_lock:
            self._inference_leases = max(0, self._inference_leases - 1)

    @property
    def is_inference_busy(self) -> bool:
        with self._inference_lock:
            return self._inference_leases > 0

    def _check_busy_locked(self, action: str) -> bool:
        """Return True when action must be refused. Caller must hold `_inference_lock`."""
        if self._inference_leases <= 0:
            return False
        logger.warning(
            "Refusing %s: inference busy (leases=%s)",
            action,
            self._inference_leases,
        )
        return True

    def _refuse_if_inference_busy(self, action: str) -> bool:
        """Return True when action must be refused because inference is in progress."""
        with self._inference_lock:
            return self._check_busy_locked(action)

    # ── Detection engine registry ────────────────────────────────────

    def _handle_attr(self, detector_id: str) -> str:
        return {
            DETECTOR_GROUNDING_DINO: "_grounding",
            DETECTOR_ONNX_YOLO: "_onnx_yolo",
        }[detector_id]

    def _loading_attr(self, detector_id: str) -> str:
        return {
            DETECTOR_GROUNDING_DINO: "_grounding_loading",
            DETECTOR_ONNX_YOLO: "_onnx_yolo_loading",
        }[detector_id]

    def _get_handle(self, detector_id: str):
        return getattr(self, self._handle_attr(detector_id))

    def _set_handle(self, detector_id: str, handle) -> None:
        setattr(self, self._handle_attr(detector_id), handle)

    def _is_loading(self, detector_id: str) -> bool:
        return bool(getattr(self, self._loading_attr(detector_id)))

    def _set_loading(self, detector_id: str, value: bool) -> None:
        setattr(self, self._loading_attr(detector_id), value)

    def register_detector(self, detector_id: str, handle) -> None:
        if detector_id not in DETECTOR_IDS:
            raise ValueError(f"Unknown detector id: {detector_id}")
        self._set_handle(detector_id, handle)

    def get_loaded_detector(self):
        detector_id = self.get_active_detector()
        if detector_id is None:
            return None
        return self._get_handle(detector_id)

    def unload_detector(self, detector_id: str) -> bool:
        if detector_id not in DETECTOR_IDS:
            raise ValueError(f"Unknown detector id: {detector_id}")
        with self._inference_lock:
            if self._check_busy_locked(f"unload_{detector_id}"):
                return False
            self._set_loading(detector_id, False)
            handle = self._get_handle(detector_id)
            self._set_handle(detector_id, None)
            if handle is not None:
                finisher = {
                    DETECTOR_GROUNDING_DINO: self._on_grounding_load_finished,
                    DETECTOR_ONNX_YOLO: self._on_onnx_yolo_load_finished,
                }[detector_id]
                try:
                    handle.loading_finished.disconnect(finisher)
                except (TypeError, RuntimeError, AttributeError):
                    pass
                handle.unload()
        return True

    def _unload_other_detectors(self, keep_id: str) -> bool:
        for detector_id in DETECTOR_IDS:
            if detector_id == keep_id:
                continue
            handle = self._get_handle(detector_id)
            if handle is None and not self._is_loading(detector_id):
                continue
            loaded = handle is not None and handle.is_loaded()
            if not loaded and not self._is_loading(detector_id):
                continue
            if not self.unload_detector(detector_id):
                return False
        return True

    def load_detector(self, detector_id: str, *, sync: bool = False) -> bool:
        """Load a registered detector and unload the other detector slots."""
        if detector_id not in DETECTOR_IDS:
            raise ValueError(f"Unknown detector id: {detector_id}")
        if self._refuse_if_inference_busy(f"load_{detector_id}"):
            return False
        if not self._unload_other_detectors(detector_id):
            return False
        handle = self._get_handle(detector_id)
        if handle is None:
            return False
        if handle.is_loaded():
            return True
        self._set_loading(detector_id, True)
        try:
            if sync and hasattr(handle, "load_sync"):
                return bool(handle.load_sync())
            handle.load()
            return True
        finally:
            if sync:
                self._set_loading(detector_id, False)

    def _on_onnx_yolo_load_finished(self, success: bool) -> None:
        self._onnx_yolo_loading = False

    def load_onnx_yolo(self, weights_path: str, parent_widget: QWidget | None = None) -> bool:
        """Start ONNX YOLO load. Returns False if blocked or unavailable."""
        from ..models.onnx_yolo_model import OnnxYoloModel

        if self._refuse_if_inference_busy("load_onnx_yolo"):
            return False
        if not OnnxYoloModel.is_available():
            return False
        if not self._unload_other_detectors(DETECTOR_ONNX_YOLO):
            return False
        existing = self._onnx_yolo
        if existing is not None and getattr(existing, "weights_path", None) == weights_path:
            model = existing
            if model.is_loaded():
                return True
        else:
            if existing is not None:
                try:
                    existing.loading_finished.disconnect(self._on_onnx_yolo_load_finished)
                except (TypeError, RuntimeError):
                    pass
                existing.unload()
            model = OnnxYoloModel(weights_path)
            self._onnx_yolo = model
        if model.is_loaded():
            return True
        if self._onnx_yolo_loading:
            return True
        self._onnx_yolo_loading = True
        try:
            model.loading_finished.disconnect(self._on_onnx_yolo_load_finished)
        except (TypeError, RuntimeError):
            pass
        model.loading_finished.connect(self._on_onnx_yolo_load_finished)
        model.load()
        return True

    def unload_onnx_yolo(self) -> bool:
        return self.unload_detector(DETECTOR_ONNX_YOLO)

    def is_onnx_yolo_loaded(self) -> bool:
        handle = self._onnx_yolo
        loaded = handle is not None and handle.is_loaded()
        if loaded:
            self._onnx_yolo_loading = False
        return loaded

    def get_onnx_yolo_model(self):
        return self._onnx_yolo

    def set_onnx_yolo_model(self, model) -> None:
        self._onnx_yolo = model

    # ── Detection engine management ──────────────────────────────────

    def load_grounding_dino(self, project_root: str = "") -> bool:
        """Start GroundingDINO load. Returns False if blocked by inference lease."""
        if self._refuse_if_inference_busy("load_grounding_dino"):
            return False
        if not self._unload_other_detectors(DETECTOR_GROUNDING_DINO):
            return False

        if self._grounding is None:
            self._grounding = GroundingDINOModel(project_root)
        if not self._grounding.is_loaded() and not self._grounding_loading:
            self._grounding_loading = True
            try:
                self._grounding.loading_finished.disconnect(self._on_grounding_load_finished)
            except (TypeError, RuntimeError):
                pass
            self._grounding.loading_finished.connect(self._on_grounding_load_finished)
            self._grounding.load()
        return True

    def _on_grounding_load_finished(self, success: bool) -> None:
        """Clear loading flag on success or failure so retries are not blocked."""
        self._grounding_loading = False

    def unload_grounding_dino(self) -> bool:
        """Unload GroundingDINO. Returns False if blocked by inference lease."""
        return self.unload_detector(DETECTOR_GROUNDING_DINO)

    def is_grounding_loaded(self) -> bool:
        loaded = self._grounding is not None and self._grounding.is_loaded()
        if loaded:
            self._grounding_loading = False
        return loaded

    def get_active_detector(self) -> str | None:
        """Return loaded detector id, or None."""
        for detector_id in DETECTOR_IDS:
            handle = self._get_handle(detector_id)
            if handle is not None and handle.is_loaded():
                self._set_loading(detector_id, False)
                return detector_id
        return None

    # ── SAM3 management ──────────────────────────────────────────────

    def load_sam3(
        self,
        parent_widget: QWidget | None = None,
        *,
        skip_vram_confirm: bool = False,
    ) -> bool:
        """Start SAM3 load asynchronously (non-blocking).

        Returns True if already loaded or async load was started.
        Returns False if blocked by inference lease.
        Connect to model.loading_finished / loading_progress for completion UI.
        """
        del skip_vram_confirm  # kept for call-site compatibility
        if self._refuse_if_inference_busy("load_sam3"):
            return False

        if self._sam is None:
            self._sam = SAM3Model()
        if self._sam.is_loaded():
            self._sam_loading = False
            self._last_sam_error = None
            return True
        if self._sam_loading:
            return True

        self._sam_loading = True
        self._last_sam_error = None
        try:
            self._sam.loading_finished.disconnect(self._on_sam3_load_finished)
        except (TypeError, RuntimeError):
            pass
        self._sam.loading_finished.connect(self._on_sam3_load_finished)
        self._sam.load()
        return True

    def load_sam3_sync(
        self,
        parent_widget: QWidget | None = None,
        *,
        skip_vram_confirm: bool = False,
    ) -> bool:
        """Synchronous SAM3 load (tests / scripts). Prefer async load_sam3 in UI."""
        del skip_vram_confirm  # kept for call-site compatibility
        if self._refuse_if_inference_busy("load_sam3_sync"):
            return False

        if self._sam is None:
            self._sam = SAM3Model()
        if self._sam.is_loaded():
            return True

        self._sam_loading = True
        try:
            success, message = self._sam.load_sync()
        except Exception as exc:
            self._last_sam_error = str(exc)
            self.unload_sam3()
            logger.error("SAM3 sync load failed: %s", exc, exc_info=True)
            return False
        finally:
            self._sam_loading = False

        if not success:
            self._last_sam_error = message or (self._sam.get_last_error() if self._sam else None)
            return False
        self._last_sam_error = None
        return True

    def _on_sam3_load_finished(self, success: bool) -> None:
        self._sam_loading = False
        if success:
            self._last_sam_error = None
        elif self._sam is not None:
            self._last_sam_error = self._sam.get_last_error()

    def unload_sam3(self) -> bool:
        """Unload SAM3. Returns False if blocked by inference lease.

        Holds `_inference_lock` through model.unload() so a worker cannot
        acquire a lease and infer while weights are being dropped.
        """
        with self._inference_lock:
            if self._check_busy_locked("unload_sam3"):
                return False
            self._sam_loading = False
            # Quit nested wait loops first so ensure_sam3_loaded cannot hang
            for loop in list(self._sam_wait_loops):
                loop.quit()
            model = self._sam
            self._sam = None
            if model is not None and not self._last_sam_error:
                self._last_sam_error = "SAM3 unloaded during load"
            if model is not None:
                try:
                    model.loading_finished.disconnect(self._on_sam3_load_finished)
                except (TypeError, RuntimeError):
                    pass
                # unload() emits loading_finished(False) for any remaining waiters
                model.unload()
        return True

    def is_sam3_loaded(self) -> bool:
        loaded = self._sam is not None and self._sam.is_loaded()
        if loaded:
            self._sam_loading = False
        return loaded

    def get_last_sam_error(self) -> Optional[str]:
        return self._last_sam_error

    def _wait_sam3_load(self, timeout_ms: int | None = None) -> bool:
        """Wait for in-flight async SAM3 load via Qt event loop (UI stays responsive).

        Always returns: finished signal, unload_sam3(), or timeout.
        """
        from PySide6.QtCore import Qt

        if self.is_sam3_loaded():
            return True
        if self._sam is None:
            return False

        if timeout_ms is None:
            timeout_ms = _SAM3_LOAD_WAIT_TIMEOUT_MS

        loop = QEventLoop()
        result: Dict[str, Any] = {"ok": False, "timed_out": False}
        self._sam_wait_loops.append(loop)

        def _on_finished(success: bool) -> None:
            result["ok"] = bool(success)
            if not success and self._sam is not None:
                err = self._sam.get_last_error()
                if err:
                    self._last_sam_error = err
            loop.quit()

        def _on_timeout() -> None:
            result["timed_out"] = True
            result["ok"] = False
            self._last_sam_error = "SAM3 load timed out"
            self._sam_loading = False
            # Invalidate in-flight load so background cannot commit (stale-id path)
            if self._sam is not None:
                try:
                    self._sam.unload()
                except Exception as exc:
                    logger.warning("SAM3 cancel-on-timeout failed: %s", exc)
            loop.quit()

        timer = QTimer()
        timer.setSingleShot(True)
        timer.timeout.connect(_on_timeout)

        sam = self._sam
        try:
            sam.loading_finished.disconnect(_on_finished)
        except (TypeError, RuntimeError):
            pass
        # Queued: worker-thread emit must quit the loop on the UI thread
        sam.loading_finished.connect(
            _on_finished, Qt.ConnectionType.QueuedConnection
        )

        try:
            # Race: finished between is_loaded check and connect
            if self.is_sam3_loaded():
                return True

            if not self._sam_loading and not self.is_sam3_loaded():
                # Load already finished (failure) or never started — no signal coming
                return False

            # Model dropped by concurrent unload
            if self._sam is None:
                return False

            timer.start(max(1, int(timeout_ms)))
            loop.exec()
        finally:
            timer.stop()
            if loop in self._sam_wait_loops:
                self._sam_wait_loops.remove(loop)
            try:
                sam.loading_finished.disconnect(_on_finished)
            except (TypeError, RuntimeError):
                pass

        if result["timed_out"]:
            return False
        # Unload clears _sam / may emit finished(False)
        if self._sam is None:
            return False
        return bool(result["ok"]) and self.is_sam3_loaded()

    # ── VRAM helpers ─────────────────────────────────────────────────

    def get_vram_snapshot(self) -> Dict[str, Any]:
        """Return {device, allocated_gb, reserved_gb, free_gb, total_gb}."""
        torch = _lazy_import_torch()
        if not torch.cuda.is_available():
            return {
                "device": "cpu",
                "allocated_gb": 0.0,
                "reserved_gb": 0.0,
                "free_gb": None,
                "total_gb": None,
            }
        props = torch.cuda.get_device_properties(0)
        total = float(props.total_memory)
        reserved = float(torch.cuda.memory_reserved(0))
        allocated = float(torch.cuda.memory_allocated(0))
        free = max(0.0, total - reserved)
        gb = 1024.0 ** 3
        return {
            "device": torch.cuda.get_device_name(0),
            "allocated_gb": allocated / gb,
            "reserved_gb": reserved / gb,
            "free_gb": free / gb,
            "total_gb": total / gb,
        }

    def estimate_sam3_vram_gb(self) -> float:
        """Conservative estimate (~3.5 weights + ~1.0 activation headroom)."""
        return _SAM3_VRAM_ESTIMATE_GB

    def ensure_sam3_loaded(self, parent_widget: QWidget | None = None) -> bool:
        """Load SAM3 with VRAM prompt; keep resident on success; unload on OOM.

        If free VRAM is below the estimate, default answer is No (soft-block).
        Numbers are estimates — not a hard guarantee against OOM.
        """
        if self.is_sam3_loaded():
            return True

        snap = self.get_vram_snapshot()
        estimate = self.estimate_sam3_vram_gb()
        free = snap.get("free_gb")
        allocated = snap.get("allocated_gb") or 0.0
        total = snap.get("total_gb")
        device = snap.get("device") or "unknown"

        if free is None:
            free_txt = "N/A (CPU 或无法查询)"
            insufficient = False
        else:
            free_txt = f"{free:.1f} GB"
            insufficient = free < estimate

        total_txt = f"{total:.1f} GB" if total is not None else "N/A"
        msg = (
            f"即将加载 SAM3（估算约 {estimate:.1f} GB，含推理余量；仅为估算，非保证）。\n\n"
            f"设备: {device}\n"
            f"已用: {allocated:.1f} GB\n"
            f"剩余: {free_txt}\n"
            f"总量: {total_txt}\n\n"
        )
        if insufficient:
            msg += "剩余显存可能不足。建议先卸载其他模型或关闭占用 GPU 的程序。\n\n是否仍要尝试加载？"
            default_btn = QMessageBox.No
        else:
            msg += "是否继续加载？"
            default_btn = QMessageBox.Yes

        reply = QMessageBox.question(
            parent_widget,
            "显存确认",
            msg,
            QMessageBox.Yes | QMessageBox.No,
            default_btn,
        )
        if reply != QMessageBox.Yes:
            return False

        try:
            started = self.load_sam3(parent_widget, skip_vram_confirm=True)
        except Exception as exc:
            self._last_sam_error = str(exc)
            self.unload_sam3()
            QMessageBox.critical(
                parent_widget,
                "SAM3 加载失败",
                f"加载 SAM3 时出错（已尝试卸载）：\n{exc}",
            )
            return False

        if not started:
            return False

        if self.is_sam3_loaded():
            return True

        ok = self._wait_sam3_load()
        if not ok:
            err = self.get_last_sam_error() or "未知错误"
            # OOM-ish failures: ensure unloaded
            if self._sam is not None and not self.is_sam3_loaded():
                self.unload_sam3()
            elif "out of memory" in err.lower() or "oom" in err.lower():
                self.unload_sam3()
            QMessageBox.critical(
                parent_widget,
                "SAM3 加载失败",
                f"无法加载 SAM3：\n{err}",
            )
            return False
        return True

    # ── Status queries ───────────────────────────────────────────────

    def get_model_status(self) -> Dict[str, Any]:
        return {
            "grounding_dino": self.is_grounding_loaded(),
            "sam3": self.is_sam3_loaded(),
            "onnx_yolo": self.is_onnx_yolo_loaded(),
            "active_detector": self.get_active_detector(),
        }

    def get_grounding_model(self) -> Optional[GroundingDINOModel]:
        return self._grounding

    def get_sam_model(self) -> Optional[SAM3Model]:
        return self._sam

    def set_sam_model(self, model: SAM3Model) -> None:
        """Set the SAM3 model instance (used for stub creation before async load)."""
        self._sam = model

    def warmup_detector(self) -> None:
        """Warm active detector if loaded; failures are non-fatal."""
        handle = self.get_loaded_detector()
        if handle is None or not hasattr(handle, "warmup"):
            return
        try:
            handle.warmup()
        except Exception as exc:
            logger.warning("Detector warmup via controller failed: %s", exc)
