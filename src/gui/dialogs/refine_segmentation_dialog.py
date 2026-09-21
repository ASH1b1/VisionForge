"""补分割：对已有检测框用 SAM3 生成多边形。"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
from PIL import Image as PILImage
from PySide6.QtCore import QThread, Signal
from PySide6.QtGui import QCloseEvent
from PySide6.QtWidgets import (
    QDialog,
    QDialogButtonBox,
    QHBoxLayout,
    QLabel,
    QMessageBox,
    QProgressBar,
    QPushButton,
    QTextEdit,
    QVBoxLayout,
    QWidget,
)

logger = logging.getLogger(__name__)


def _load_pil_image(image_path: Path) -> Optional[PILImage.Image]:
    """Load RGB PIL image; supports Chinese paths via np.fromfile."""
    try:
        path_str = str(image_path)
        if not image_path.exists():
            return None
        arr = np.fromfile(path_str, dtype=np.uint8)
        import cv2

        bgr = cv2.imdecode(arr, cv2.IMREAD_COLOR)
        if bgr is None:
            return PILImage.open(path_str).convert("RGB")
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        return PILImage.fromarray(rgb)
    except Exception as exc:
        logger.error("Failed to load image %s: %s", image_path, exc)
        return None


def _prompt_from_class_names(class_names: Dict[int, str], class_ids: Sequence[int]) -> str:
    names = []
    for cid in class_ids:
        name = class_names.get(cid)
        if name and name not in names:
            names.append(name)
    return ", ".join(names) if names else "object"


class RefineSegmentationWorker(QThread):
    """Run SAM3 align_detections for one image's YOLO bboxes."""

    progress = Signal(str)
    finished = Signal(list)  # list[(ann_id, polygon|None)]
    error = Signal(str)

    def __init__(
        self,
        sam_model,
        image: PILImage.Image,
        pairs: List[Tuple[int, tuple]],  # (ann_id, yolo_bbox)
        prompt: str = "object",
        iou_threshold: float = 0.1,
        parent=None,
        model_ctrl=None,
    ):
        super().__init__(parent)
        self._sam_model = sam_model
        self._image = image
        self._pairs = pairs
        self._prompt = prompt
        self._iou_threshold = iou_threshold
        self._model_ctrl = model_ctrl
        self._cancelled = False

    def cancel(self) -> None:
        self._cancelled = True

    def run(self) -> None:
        ctrl = self._model_ctrl
        if ctrl is None:
            self.error.emit("缺少 ModelController，无法持有推理租约")
            return
        ctrl.acquire_inference_lease()
        try:
            if self._cancelled:
                return
            if not self._pairs:
                self.finished.emit([])
                return
            if self._sam_model is None or not self._sam_model.is_loaded():
                self.error.emit("SAM3 未加载")
                return

            self.progress.emit(f"正在补分割 ({len(self._pairs)} 个框)...")
            ann_ids = [p[0] for p in self._pairs]
            bboxes = [p[1] for p in self._pairs]
            polygons = self._sam_model.align_detections(
                self._image,
                self._prompt,
                bboxes,
                iou_threshold=self._iou_threshold,
            )
            if self._cancelled:
                return
            results = []
            for ann_id, poly in zip(ann_ids, polygons):
                results.append((ann_id, poly))
            matched = sum(1 for _, p in results if p)
            self.progress.emit(f"完成: {matched}/{len(results)} 个多边形")
            self.finished.emit(results)
        except Exception as exc:
            import traceback

            self.error.emit(f"{exc}\n\n{traceback.format_exc()}")
        finally:
            ctrl.release_inference_lease()


class BatchRefineSegmentationWorker(QThread):
    """Batch: for each frame with boxes, run SAM align and emit per-frame results."""

    progress = Signal(str)
    frame_progress = Signal(int, int)  # current, total
    frame_finished = Signal(int, list)  # frame_idx, [(ann_id, polygon)]
    batch_finished = Signal(int, int)  # frames_done, polygons_updated
    error = Signal(str)

    def __init__(
        self,
        sam_model,
        jobs: List[dict],
        # each job: {frame_idx, image_path|pil_image, pairs, prompt}
        iou_threshold: float = 0.1,
        parent=None,
        model_ctrl=None,
    ):
        super().__init__(parent)
        self._sam_model = sam_model
        self._jobs = jobs
        self._iou_threshold = iou_threshold
        self._model_ctrl = model_ctrl
        self._cancelled = False

    def cancel(self) -> None:
        self._cancelled = True

    def run(self) -> None:
        ctrl = self._model_ctrl
        if ctrl is None:
            self.error.emit("缺少 ModelController，无法持有推理租约")
            return
        ctrl.acquire_inference_lease()
        try:
            frames_done = 0
            polys_updated = 0
            total = len(self._jobs)
            for i, job in enumerate(self._jobs):
                if self._cancelled:
                    break
                self.frame_progress.emit(i + 1, total)
                frame_idx = job["frame_idx"]
                pairs = job["pairs"]
                prompt = job.get("prompt") or "object"
                image = job.get("pil_image")
                if image is None:
                    path = job.get("image_path")
                    self.progress.emit(f"读取图像 [{frame_idx}]...")
                    image = _load_pil_image(Path(path)) if path else None
                if image is None:
                    self.progress.emit(f"跳过帧 {frame_idx}: 无法读图")
                    continue
                if not pairs:
                    continue
                if self._sam_model is None or not self._sam_model.is_loaded():
                    self.error.emit("SAM3 未加载")
                    self.batch_finished.emit(frames_done, polys_updated)
                    return

                self.progress.emit(f"补分割帧 {frame_idx + 1}/{total} ({len(pairs)} 框)...")
                ann_ids = [p[0] for p in pairs]
                bboxes = [p[1] for p in pairs]
                polygons = self._sam_model.align_detections(
                    image,
                    prompt,
                    bboxes,
                    iou_threshold=self._iou_threshold,
                )
                if self._cancelled:
                    break
                results = list(zip(ann_ids, polygons))
                polys_updated += sum(1 for _, p in results if p)
                frames_done += 1
                self.frame_finished.emit(frame_idx, results)

            self.batch_finished.emit(frames_done, polys_updated)
        except Exception as exc:
            import traceback

            self.error.emit(f"{exc}\n\n{traceback.format_exc()}")
            # Always finish so dialog UI can recover (Ok / Cancel)
            self.batch_finished.emit(0, 0)
        finally:
            ctrl.release_inference_lease()


class RefineSegmentationDialog(QDialog):
    """批量补分割对话框：对已有框的图像跑 SAM3。"""

    def __init__(
        self,
        model_ctrl,
        jobs: List[dict],
        parent: QWidget | None = None,
    ):
        super().__init__(parent)
        self._model_ctrl = model_ctrl
        self._jobs = jobs
        self._worker: Optional[BatchRefineSegmentationWorker] = None
        self._close_when_worker_done = False
        self._results: Dict[int, List[Tuple[int, list]]] = {}
        self.setWindowTitle("批量补分割")
        self.setMinimumWidth(480)
        self._setup_ui()

    def _setup_ui(self) -> None:
        layout = QVBoxLayout(self)
        n_frames = len(self._jobs)
        n_boxes = sum(len(j.get("pairs") or []) for j in self._jobs)
        self._info = QLabel(
            f"将对 {n_frames} 张已有检测框的图片补全多边形（共 {n_boxes} 个框）。\n"
            "SAM3 将按需加载并在会话内常驻。"
        )
        self._info.setWordWrap(True)
        layout.addWidget(self._info)

        self._progress_label = QLabel("")
        layout.addWidget(self._progress_label)
        self._progress_bar = QProgressBar()
        self._progress_bar.hide()
        layout.addWidget(self._progress_bar)

        self._log = QTextEdit()
        self._log.setReadOnly(True)
        self._log.setMaximumHeight(160)
        layout.addWidget(self._log)

        btn_row = QHBoxLayout()
        self._run_btn = QPushButton("开始补分割")
        self._run_btn.clicked.connect(self._on_run)
        btn_row.addWidget(self._run_btn)
        self._buttons = QDialogButtonBox(QDialogButtonBox.Cancel | QDialogButtonBox.Ok)
        self._buttons.button(QDialogButtonBox.Ok).setEnabled(False)
        self._buttons.accepted.connect(self.accept)
        self._buttons.rejected.connect(self._on_cancel)
        btn_row.addWidget(self._buttons)
        layout.addLayout(btn_row)

    def get_results(self) -> Dict[int, List[Tuple[int, list]]]:
        return self._results

    def _on_run(self) -> None:
        if not self._jobs:
            QMessageBox.information(self, "提示", "没有可补分割的帧（需已有检测框）。")
            return
        if self._model_ctrl is None:
            QMessageBox.warning(self, "警告", "缺少 ModelController")
            return
        if not self._model_ctrl.ensure_sam3_loaded(self):
            QMessageBox.warning(self, "已中止", "SAM3 未加载，无法补分割。")
            return

        sam = self._model_ctrl.get_sam_model()
        self._run_btn.setEnabled(False)
        self._progress_bar.show()
        self._progress_bar.setRange(0, len(self._jobs))
        self._progress_bar.setValue(0)
        self._worker = BatchRefineSegmentationWorker(
            sam, self._jobs, model_ctrl=self._model_ctrl
        )
        self._worker.progress.connect(self._on_progress)
        self._worker.frame_progress.connect(
            lambda cur, total: self._progress_bar.setValue(cur)
        )
        self._worker.frame_finished.connect(self._on_frame_finished)
        self._worker.batch_finished.connect(self._on_batch_finished)
        self._worker.error.connect(self._on_error)
        # Inherited QThread.finished (not batch_finished) for deferred close.
        self._worker.finished.connect(self._on_worker_thread_finished)
        self._worker.start()

    def _on_progress(self, msg: str) -> None:
        self._progress_label.setText(msg)
        self._log.append(msg)

    def _on_frame_finished(self, frame_idx: int, results: list) -> None:
        self._results[frame_idx] = results

    def _on_batch_finished(self, frames_done: int, polys: int) -> None:
        if self._close_when_worker_done:
            return
        self._run_btn.setEnabled(True)
        self._progress_bar.hide()
        self._progress_label.setText(f"完成: {frames_done} 帧, {polys} 个多边形")
        self._buttons.button(QDialogButtonBox.Ok).setEnabled(True)

    def _on_error(self, err: str) -> None:
        if self._close_when_worker_done:
            return
        self._run_btn.setEnabled(True)
        self._progress_bar.hide()
        self._log.append(err)
        # Allow accepting partial results if any frames succeeded
        if self._results:
            self._buttons.button(QDialogButtonBox.Ok).setEnabled(True)
            self._progress_label.setText("部分完成（出错后可应用已完成帧）")
        else:
            self._progress_label.setText("失败")
        QMessageBox.critical(self, "补分割失败", err)

    def _on_cancel(self) -> None:
        if self._worker and self._worker.isRunning():
            self._worker.cancel()
            self._progress_label.setText("正在停止…")
            self._run_btn.setEnabled(False)
            self._buttons.button(QDialogButtonBox.Cancel).setEnabled(False)
            self._close_when_worker_done = True
            return
        self.reject()

    def _on_worker_thread_finished(self) -> None:
        """Worker thread fully stopped — finish deferred close if requested."""
        if not self._close_when_worker_done:
            return
        self._close_when_worker_done = False
        self.reject()

    def closeEvent(self, event: QCloseEvent) -> None:
        """Ignore close while worker runs; reject after QThread.finished."""
        if self._worker and self._worker.isRunning():
            self._worker.cancel()
            self._progress_label.setText("正在停止…")
            self._run_btn.setEnabled(False)
            self._buttons.button(QDialogButtonBox.Cancel).setEnabled(False)
            self._close_when_worker_done = True
            event.ignore()
            return
        event.accept()
