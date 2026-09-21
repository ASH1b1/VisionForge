"""Interactive SAM click-refine session (Annotate canvas)."""
from __future__ import annotations

from typing import Callable, Optional

from PySide6.QtCore import QObject, QThread, Signal
from PySide6.QtWidgets import QMessageBox

from ..models.app_state import Tool
from ..models.sam3_model import SAM3Model
from ..utils.hit_testing import smallest_bbox_hit
from src.gui.utils.annotation_draw import resolved_kind
from ..utils.polygon_edit import yolo_bbox_from_polygon
from ..utils.segmentation_utils import polygon_to_mask

MSG_LOAD_PROMPT = (
    "点选需要 SAM3。现在加载可能占用较多显存、需要一些时间。是否加载？"
)
MSG_SAM_LOADING = "正在加载 SAM3…"
MSG_SAM_NOT_LOADED = "请先加载 SAM3（菜单：模型 → 加载 SAM3）"
MSG_NO_PREDICTOR = (
    "当前 SAM3 权重不支持点选修正（缺少 interactive predictor）。"
    "仍可用矩形/多边形手画。"
)
MSG_NO_CLASS = "请先在左侧类别栏中添加类别"
MSG_BLOCKED_KIND = "旋转框/关键点标注不能用点选修正"
MSG_NO_IMAGE = "未加载图片"
MSG_CLICK_MISS = "未点中物体，请点在目标上"
MSG_HINT_ACTIVE = "左键点目标 · 右键排除 · Enter 或「确认分割」写入 · Esc 取消"
MSG_HINT_COMMITTED = "已写入。可按 V 拖顶点微调，或继续点选下一个目标"


class _SamClickWorker(QThread):
    result_ready = Signal(object)
    failed = Signal(int, str)

    def __init__(self, fn, kwargs, model_ctrl, parent=None):
        super().__init__(parent)
        self._fn = fn
        self._kwargs = kwargs
        self._model_ctrl = model_ctrl

    def run(self):
        ctrl = self._model_ctrl
        if ctrl is not None:
            ctrl.acquire_inference_lease()
        try:
            result = self._fn(**self._kwargs)
            self.result_ready.emit(result)
        except Exception as exc:
            self.failed.emit(int(self._kwargs.get("generation") or 0), str(exc))
        finally:
            if ctrl is not None:
                ctrl.release_inference_lease()


class SamClickController(QObject):
    status_message = Signal(str)
    commit_available = Signal(bool)

    def __init__(
        self,
        project,
        app_state,
        annotation_ctrl,
        model_ctrl,
        canvas=None,
        *,
        parent=None,
        predict_fn: Callable | None = None,
        image_provider: Callable | None = None,
        image_size_provider: Callable | None = None,
        selected_class_provider: Callable | None = None,
        ask_load_sam: Callable | None = None,
        frame_key_provider: Callable | None = None,
        parent_widget=None,
    ):
        super().__init__(parent)
        self._project = project
        self._app_state = app_state
        self._ann_ctrl = annotation_ctrl
        self._model_ctrl = model_ctrl
        self._canvas = canvas
        self._predict_fn = predict_fn
        self._image_provider = image_provider
        self._image_size_provider = image_size_provider
        self._selected_class_provider = selected_class_provider
        self._ask_load_sam = ask_load_sam
        self._frame_key_provider = frame_key_provider
        self._parent_widget = parent_widget
        self._on_sam_tool = False
        self._load_declined = False
        self._last_selected_id: Optional[int] = None
        self._workers: list[_SamClickWorker] = []
        self._generation = 0
        self._reset_session_fields()

    def _reset_session_fields(self) -> None:
        self._bound_id: Optional[int] = None
        self._points: list[tuple[float, float, int]] = []
        self._preview: list | None = None
        self._logits = None
        self._seed_box_xyxy = None
        self._seed_polygon: list | None = None
        self._had_polygon = False
        self._uncommitted = False
        self._predicting = False

    def is_session_active(self) -> bool:
        return self._on_sam_tool and (
            self._bound_id is not None or bool(self._points) or bool(self._preview)
        )

    def note_annotation_selected(self, ann_id: int) -> None:
        if ann_id >= 0:
            self._last_selected_id = int(ann_id)

    def on_tool_changed(self, tool) -> None:
        if tool != Tool.SAM_CLICK:
            if self._on_sam_tool:
                self.discard_session()
            self._on_sam_tool = False
            return
        self._on_sam_tool = True
        self._maybe_prompt_load()
        if not self._ensure_can_predict():
            self._collapse_selection()
            return
        self._enter_tool()

    def on_frame_changed(self, _index=None) -> None:
        self.discard_session()
        sam = self._sam()
        if sam is not None and hasattr(sam, "clear_interactive_cache"):
            sam.clear_interactive_cache()

    def on_mode_changed(self, mode_value: str) -> None:
        if str(mode_value).lower() == "detect":
            self.discard_session()

    def discard_session(self) -> None:
        self._reset_session_fields()
        self._generation += 1
        self._refresh_overlay()

    def deselect_to_create(self) -> None:
        self.discard_session()
        self._project.clear_selection()

    def handle_click(self, x_norm: float, y_norm: float, label: int) -> None:
        if not self._on_sam_tool:
            return
        if self._image_size() is None:
            self._emit_status(MSG_NO_IMAGE)
            return
        if not self._ensure_can_predict():
            return
        hit_id = smallest_bbox_hit(self._project.current_annotations, x_norm, y_norm)
        hit_ann = self._find_ann(hit_id) if hit_id is not None else None
        if hit_ann is not None and self._is_blocked(hit_ann):
            self._emit_status(MSG_BLOCKED_KIND)
            return
        if self._bound_id is None:
            if hit_ann is not None:
                self._bind_refine(hit_ann)
                return
            self._start_create(x_norm, y_norm, int(label))
            return
        if hit_ann is None:
            self._start_create(x_norm, y_norm, int(label))
            return
        if hit_ann.id != self._bound_id:
            self._bind_refine(hit_ann)
            return
        self._points.append((float(x_norm), float(y_norm), int(label)))
        self._refresh_overlay()
        self._request_predict()

    def undo_last_point(self) -> None:
        if not self._on_sam_tool or not self._points:
            return
        self._points.pop()
        if not self._points:
            self._logits = None
            if self._had_polygon and self._seed_polygon:
                self._preview = list(self._seed_polygon)
                self._refresh_overlay()
                return
            if self._seed_box_xyxy is not None:
                self._refresh_overlay()
                self._request_predict()
                return
            self._preview = None
            self._bound_id = None
            self._refresh_overlay()
            return
        self._refresh_overlay()
        self._request_predict()

    def commit(self) -> None:
        if not self._on_sam_tool or not self._can_commit():
            return
        bbox = yolo_bbox_from_polygon(self._preview)
        polygon = list(self._preview)
        if self._bound_id is not None and self._find_ann(self._bound_id) is not None:
            self._ann_ctrl.edit_polygon(self._bound_id, polygon, bbox)
            ann_id = self._bound_id
        else:
            class_id = self._class_id()
            if class_id is None:
                self._emit_status(MSG_NO_CLASS)
                return
            ann = self._ann_ctrl.add_annotation(int(class_id), bbox, polygon=polygon)
            ann_id = ann.id
        self._project.set_selection({ann_id})
        self._bound_id = ann_id
        self._points = []
        self._logits = None
        self._seed_box_xyxy = None
        self._had_polygon = True
        self._seed_polygon = list(polygon)
        self._preview = polygon
        self._uncommitted = False
        self._predicting = False
        self._refresh_overlay()
        self._emit_status(MSG_HINT_COMMITTED)

    def apply_predict_result(self, result) -> None:
        if not result:
            return
        if int(result.get("generation", -1)) != self._generation:
            return
        self._predicting = False
        if result.get("click_miss"):
            self._points = []
            self._preview = None
            self._logits = None
            self._uncommitted = False
            self._refresh_overlay()
            self._emit_status(MSG_CLICK_MISS)
            return
        poly = result.get("polygon") or []
        if len(poly) >= 3:
            self._preview = list(poly)
            self._uncommitted = True
        self._logits = result.get("low_res_logits")
        self._refresh_overlay()
        if self._uncommitted:
            self._emit_status(MSG_HINT_ACTIVE)

    def _enter_tool(self) -> None:
        self.discard_session()
        self._on_sam_tool = True
        self._collapse_selection()
        ids = self._project.selected_ids
        if len(ids) != 1:
            return
        ann = self._find_ann(next(iter(ids)))
        if ann is None:
            return
        if self._is_blocked(ann):
            self._emit_status(MSG_BLOCKED_KIND)
            return
        self._bind_refine(ann)

    def _bind_refine(self, ann) -> None:
        self._generation += 1
        self._bound_id = ann.id
        self._points = []
        self._logits = None
        self._had_polygon = bool(ann.polygon) and len(ann.polygon) >= 3
        self._seed_polygon = list(ann.polygon) if self._had_polygon else None
        self._preview = list(ann.polygon) if self._had_polygon else None
        size = self._image_size()
        if size and ann.bbox:
            self._seed_box_xyxy = SAM3Model._yolo_bbox_to_xyxy(ann, ann.bbox, size[0], size[1])
        else:
            self._seed_box_xyxy = None
        self._project.set_selection({ann.id})
        self._refresh_overlay()
        if not self._had_polygon:
            self._request_predict()

    def _start_create(self, x_norm: float, y_norm: float, label: int) -> None:
        if self._class_id() is None:
            self._emit_status(MSG_NO_CLASS)
            return
        self._generation += 1
        self._bound_id = None
        self._had_polygon = False
        self._seed_polygon = None
        self._seed_box_xyxy = None
        self._logits = None
        self._preview = None
        self._points = [(float(x_norm), float(y_norm), int(label))]
        self._refresh_overlay()
        self._request_predict()

    def _request_predict(self) -> None:
        if not self._ensure_can_predict():
            return
        self._predicting = True
        self.commit_available.emit(False)
        self._generation += 1
        kwargs = self._build_predict_kwargs(self._generation)
        self._run_predict(kwargs)

    def _build_predict_kwargs(self, generation: int) -> dict:
        size = self._image_size() or (1, 1)
        width, height = size
        coords = None
        labels = None
        if self._points:
            coords = [[p[0] * width, p[1] * height] for p in self._points]
            labels = [int(p[2]) for p in self._points]
        mask = self._logits
        if mask is None and self._had_polygon and self._seed_polygon and self._points:
            mask = self._mask_from_polygon(self._seed_polygon)
        image = None
        if self._image_provider is not None:
            image = self._image_provider()
        elif self._canvas is not None:
            image = self._pil_from_canvas()
        pick_click_xy = None
        if (
            self._bound_id is None
            and len(self._points) == 1
            and int(self._points[0][2]) == 1
        ):
            pick_click_xy = (self._points[0][0] * width, self._points[0][1] * height)
        return {
            "image": image,
            "box_xyxy": self._seed_box_xyxy,
            "point_coords": coords,
            "point_labels": labels,
            "mask_input": mask,
            "generation": generation,
            "frame_key": self._frame_key(),
            "pick_click_xy": pick_click_xy,
        }

    def _run_predict(self, kwargs: dict) -> None:
        if self._predict_fn is not None:
            ctrl = self._model_ctrl
            if ctrl is not None:
                ctrl.acquire_inference_lease()
            try:
                result = self._predict_fn(**kwargs)
                self.apply_predict_result(result)
            except Exception as exc:
                self._on_predict_failed(int(kwargs.get("generation") or 0), str(exc))
            finally:
                if ctrl is not None:
                    ctrl.release_inference_lease()
            return
        worker = _SamClickWorker(self._model_predict, kwargs, self._model_ctrl)
        worker.result_ready.connect(self.apply_predict_result)
        worker.failed.connect(self._on_predict_failed)
        self._workers.append(worker)

        def _drop(w=worker):
            if w in self._workers:
                self._workers.remove(w)

        worker.finished.connect(_drop)
        worker.start()

    def _model_predict(self, **kwargs):
        sam = self._sam()
        if sam is None:
            raise RuntimeError("SAM3 未加载")
        image = kwargs.get("image")
        if image is None:
            raise RuntimeError(MSG_NO_IMAGE)
        return sam.predict_interactive(
            image,
            box_xyxy=kwargs.get("box_xyxy"),
            point_coords=kwargs.get("point_coords"),
            point_labels=kwargs.get("point_labels"),
            mask_input=kwargs.get("mask_input"),
            generation=int(kwargs.get("generation") or 0),
            frame_key=kwargs.get("frame_key"),
            pick_click_xy=kwargs.get("pick_click_xy"),
        )

    def _on_predict_failed(self, generation: int, message: str) -> None:
        if generation != self._generation:
            return
        self._predicting = False
        self._emit_status(message or MSG_NO_PREDICTOR)
        self._refresh_overlay()

    def _mask_from_polygon(self, polygon):
        size = self._image_size()
        if not size:
            return None
        width, height = size
        binary = polygon_to_mask(polygon, width, height)
        if binary is None:
            return None
        import cv2
        import numpy as np

        resized = cv2.resize(
            binary.astype(np.float32) / 255.0,
            (256, 256),
            interpolation=cv2.INTER_LINEAR,
        )
        return resized

    def _maybe_prompt_load(self) -> None:
        if self._predict_fn is not None:
            return
        if self._model_ctrl is None or self._model_ctrl.is_sam3_loaded():
            return
        if getattr(self._model_ctrl, "_sam_loading", False):
            self._emit_status(MSG_SAM_LOADING)
            return
        if self._load_declined:
            return
        if self._ask_load_sam is not None:
            ok = bool(self._ask_load_sam())
        else:
            reply = QMessageBox.question(
                self._parent_widget,
                "加载 SAM3",
                MSG_LOAD_PROMPT,
                QMessageBox.Yes | QMessageBox.No,
            )
            ok = reply == QMessageBox.Yes
        if not ok:
            self._load_declined = True
            return
        started = self._model_ctrl.load_sam3(self._parent_widget)
        if started:
            self._emit_status(MSG_SAM_LOADING)
            sam = self._sam()
            if sam is not None:
                try:
                    sam.loading_finished.connect(self._on_sam_loaded)
                except Exception:
                    pass

    def _on_sam_loaded(self, success: bool) -> None:
        if not self._on_sam_tool:
            return
        if success:
            self._enter_tool()
            return
        err = ""
        if self._model_ctrl is not None:
            err = self._model_ctrl.get_last_sam_error() or ""
        self._emit_status(err or MSG_SAM_NOT_LOADED)

    def _ensure_can_predict(self) -> bool:
        if self._model_ctrl is None:
            return self._predict_fn is not None
        if getattr(self._model_ctrl, "_sam_loading", False):
            self._emit_status(MSG_SAM_LOADING)
            return False
        if not self._model_ctrl.is_sam3_loaded():
            self._emit_status(MSG_SAM_NOT_LOADED)
            return False
        sam = self._sam()
        if sam is None or not sam.has_interactive_predictor():
            self._emit_status(MSG_NO_PREDICTOR)
            return False
        return True

    def _collapse_selection(self) -> None:
        ids = set(self._project.selected_ids)
        if len(ids) <= 1:
            return
        keep = self._last_selected_id if self._last_selected_id in ids else None
        if keep is None:
            keep = smallest_bbox_hit(
                [a for a in self._project.current_annotations if a.id in ids],
                0.5,
                0.5,
            )
            if keep is None:
                best_id = None
                best_area = None
                for ann in self._project.current_annotations:
                    if ann.id not in ids or not ann.bbox:
                        continue
                    area = abs(float(ann.bbox[2]) * float(ann.bbox[3]))
                    if best_area is None or area < best_area:
                        best_area = area
                        best_id = ann.id
                keep = best_id
        if keep is None:
            return
        self._project.set_selection({keep})
        ann = self._find_ann(keep)
        name = ""
        if ann is not None:
            name = self._project.class_names.get(ann.class_id, str(keep))
        self._emit_status(f"点选一次只能修一条，已选中 {name}/{keep}")

    def _class_id(self):
        if self._selected_class_provider is not None:
            return self._selected_class_provider()
        return getattr(self._app_state.state, "selected_class_id", None)

    def _image_size(self):
        if self._image_size_provider is not None:
            return self._image_size_provider()
        img = getattr(self._canvas, "_image", None) if self._canvas is not None else None
        if img is None:
            return None
        width, height = img.width(), img.height()
        if not width or not height:
            return None
        return int(width), int(height)

    def _pil_from_canvas(self):
        from ..dialogs.auto_annotate_dialog import _qimage_to_pil

        img = getattr(self._canvas, "_image", None) if self._canvas is not None else None
        return _qimage_to_pil(img) if img is not None else None

    def _frame_key(self):
        if self._frame_key_provider is not None:
            return self._frame_key_provider()
        path = ""
        frame = 0
        if self._app_state is not None:
            current = getattr(self._app_state.state, "current_file", None)
            if current is not None:
                path = str(current)
            frame = int(getattr(self._app_state.state, "current_frame", 0) or 0)
        size = self._image_size() or (0, 0)
        return (frame, path, size[0], size[1])

    def _sam(self):
        if self._model_ctrl is None:
            return None
        return self._model_ctrl.get_sam_model()

    def _find_ann(self, ann_id):
        if ann_id is None:
            return None
        for ann in self._project.current_annotations:
            if ann.id == ann_id:
                return ann
        return None

    @staticmethod
    def _is_blocked(ann) -> bool:
        return resolved_kind(ann) in ("obb", "pose")

    def _refresh_overlay(self) -> None:
        canvas = self._canvas
        if canvas is not None and hasattr(canvas, "set_sam_session_overlay"):
            canvas.set_sam_session_overlay(
                points=self._points,
                polygon=self._preview,
                bound_id=self._bound_id,
            )
        self.commit_available.emit(self._can_commit())

    def _can_commit(self) -> bool:
        return (
            self._on_sam_tool
            and self._uncommitted
            and not self._predicting
            and bool(self._preview)
            and len(self._preview) >= 3
        )

    def _emit_status(self, text: str) -> None:
        if text:
            self.status_message.emit(text)
