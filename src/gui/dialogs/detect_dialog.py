"""Detect dialog — run detection without committing annotations."""
from __future__ import annotations

from typing import List, Optional

from PySide6.QtCore import Qt, QThread, Signal
from PySide6.QtGui import QImage
from PySide6.QtWidgets import (
    QDialog,
    QVBoxLayout,
    QHBoxLayout,
    QFormLayout,
    QLineEdit,
    QDoubleSpinBox,
    QPushButton,
    QProgressBar,
    QLabel,
    QGroupBox,
    QMessageBox,
    QComboBox,
)

from ..controllers.detect_controller import detections_from_raw
from ..models.auto_annotation_result import AutoAnnotationResult
from ..models.detector_protocol import (
    UNMATCHED_CREATE,
    UNMATCHED_SKIP,
    resolve_detector_handle,
)
from .auto_annotate_dialog import _qimage_to_pil


class DetectWorker(QThread):
    """Background detect inference (bbox only, no SAM3)."""

    progress = Signal(str)
    finished_ok = Signal(list)  # List[AutoAnnotationResult]
    failed = Signal(str)

    def __init__(
        self,
        image,
        prompt: str,
        box_threshold: float,
        text_threshold: float,
        nms_threshold: float,
        detector_model,
        model_ctrl,
        img_width: int,
        img_height: int,
        class_names=None,
        project_class_names=None,
        unmatched: str = UNMATCHED_CREATE,
        parent=None,
    ):
        super().__init__(parent)
        self._image = image
        self._prompt = prompt
        self._box_threshold = box_threshold
        self._text_threshold = text_threshold
        self._nms_threshold = nms_threshold
        self._detector_model = detector_model
        self._model_ctrl = model_ctrl
        self._img_width = img_width
        self._img_height = img_height
        self._class_names = class_names
        self._project_class_names = project_class_names
        self._unmatched = unmatched
        self._cancelled = False

    def cancel(self) -> None:
        self._cancelled = True

    def run(self) -> None:
        ctrl = self._model_ctrl
        if ctrl is None:
            self.failed.emit("缺少 ModelController")
            return
        ctrl.acquire_inference_lease()
        try:
            self.progress.emit("正在推理…")
            # Models accept path / ndarray / PIL — not raw QImage (same as auto-annotate).
            if isinstance(self._image, QImage):
                pil_image = _qimage_to_pil(self._image)
                if pil_image is None:
                    self.failed.emit("无法将当前图像转换为可推理格式")
                    return
            else:
                pil_image = self._image
            result = self._detector_model.infer(
                image=pil_image,
                text_prompt=self._prompt,
                box_threshold=self._box_threshold,
                text_threshold=self._text_threshold,
            )
            if self._cancelled:
                return
            if result is None:
                self.failed.emit("模型返回空结果")
                return
            if not result.boxes:
                self.progress.emit("未检测到目标")
                self.finished_ok.emit([])
                return
            results = detections_from_raw(
                boxes=result.boxes,
                labels=result.labels,
                scores=result.scores,
                prompt=self._prompt,
                img_width=self._img_width,
                img_height=self._img_height,
                nms_threshold=self._nms_threshold,
                class_names=self._class_names,
                project_class_names=self._project_class_names,
                unmatched=self._unmatched,
            )
            self.progress.emit(f"完成，共 {len(results)} 个目标")
            self.finished_ok.emit(results)
        except Exception as e:
            import traceback

            self.failed.emit(f"{e}\n{traceback.format_exc()}")
        finally:
            ctrl.release_inference_lease()


class DetectDialog(QDialog):
    """Configure and run Detect-mode inference on the current image/frame."""

    def __init__(
        self,
        image: QImage,
        grounding_model=None,
        model_ctrl=None,
        parent=None,
        default_prompt: str = "object",
        project_class_names=None,
    ):
        super().__init__(parent)
        self.setWindowTitle("Detect — 检测预览")
        self.resize(480, 400)
        self._image = image
        self._grounding = grounding_model
        self._model_ctrl = model_ctrl
        self._project_class_names = dict(project_class_names or {})
        self._worker: Optional[DetectWorker] = None
        self._results: List[AutoAnnotationResult] = []
        self._prompt = default_prompt

        layout = QVBoxLayout(self)

        form_box = QGroupBox("参数")
        form = QFormLayout(form_box)
        self._prompt_edit = QLineEdit(default_prompt)
        self._prompt_edit.setPlaceholderText("例如: plant, weed")
        form.addRow("文本提示", self._prompt_edit)

        self._box_spin = QDoubleSpinBox()
        self._box_spin.setRange(0.01, 1.0)
        self._box_spin.setSingleStep(0.05)
        self._box_spin.setValue(0.35)
        form.addRow("Box 阈值", self._box_spin)

        self._text_spin = QDoubleSpinBox()
        self._text_spin.setRange(0.01, 1.0)
        self._text_spin.setSingleStep(0.05)
        self._text_spin.setValue(0.25)
        form.addRow("Text 阈值", self._text_spin)

        self._nms_spin = QDoubleSpinBox()
        self._nms_spin.setRange(0.01, 1.0)
        self._nms_spin.setSingleStep(0.05)
        self._nms_spin.setValue(0.5)
        form.addRow("NMS 阈值", self._nms_spin)

        self._unmatched_combo = QComboBox()
        self._unmatched_combo.addItem("创建新类", UNMATCHED_CREATE)
        self._unmatched_combo.addItem("跳过未知类", UNMATCHED_SKIP)
        form.addRow("未知类别", self._unmatched_combo)

        layout.addWidget(form_box)

        self._model_hint = QLabel()
        self._model_hint.setWordWrap(True)
        if not self._requires_text_prompt():
            self._prompt_edit.setPlaceholderText("YOLO ONNX 使用权重内类别，无需文本提示")
            self._model_hint.setText("当前为自定义 YOLO ONNX：按模型类别出框，结果写入 Detect 预览，不改工程文件。")
            self._model_hint.setStyleSheet("color: #FF9800; font-style: italic;")
        layout.addWidget(self._model_hint)

        self._progress = QProgressBar()
        self._progress.setRange(0, 0)
        self._progress.setVisible(False)
        layout.addWidget(self._progress)

        self._status = QLabel("点击「开始检测」在画布上预览结果（不会写入标注）。")
        self._status.setWordWrap(True)
        layout.addWidget(self._status)

        btn_row = QHBoxLayout()
        self._run_btn = QPushButton("开始检测")
        self._run_btn.clicked.connect(self._start)
        self._cancel_btn = QPushButton("取消任务")
        self._cancel_btn.setEnabled(False)
        self._cancel_btn.clicked.connect(self._cancel_worker)
        self._accept_btn = QPushButton("应用到预览")
        self._accept_btn.setEnabled(False)
        self._accept_btn.clicked.connect(self.accept)
        close_btn = QPushButton("关闭")
        close_btn.clicked.connect(self.reject)
        btn_row.addWidget(self._run_btn)
        btn_row.addWidget(self._cancel_btn)
        btn_row.addStretch()
        btn_row.addWidget(self._accept_btn)
        btn_row.addWidget(close_btn)
        layout.addLayout(btn_row)

    def get_results(self) -> List[AutoAnnotationResult]:
        return list(self._results)

    def get_prompt(self) -> str:
        return self._prompt_edit.text().strip()

    def get_box_threshold(self) -> float:
        return float(self._box_spin.value())

    def get_text_threshold(self) -> float:
        return float(self._text_spin.value())

    def get_nms_threshold(self) -> float:
        return float(self._nms_spin.value())

    def get_unmatched_policy(self) -> str:
        data = self._unmatched_combo.currentData()
        return data if data in (UNMATCHED_CREATE, UNMATCHED_SKIP) else UNMATCHED_CREATE

    def _requires_text_prompt(self) -> bool:
        detector = self._active_detector()
        return not (detector is not None and hasattr(detector, "class_names_list"))

    def _closed_set_class_names(self):
        detector = self._active_detector()
        getter = getattr(detector, "class_names_list", None) if detector is not None else None
        if callable(getter):
            return list(getter())
        return None

    def _active_detector(self):
        return resolve_detector_handle(
            grounding=self._grounding,
            model_ctrl=self._model_ctrl,
        )

    def _start(self) -> None:
        detector = self._active_detector()
        if detector is None:
            QMessageBox.warning(self, "警告", "请先加载检测模型")
            return
        prompt = self.get_prompt()
        if self._requires_text_prompt() and not prompt:
            QMessageBox.warning(self, "警告", "请输入文本提示")
            return

        self._results = []
        self._accept_btn.setEnabled(False)
        self._progress.setVisible(True)
        self._run_btn.setEnabled(False)
        self._cancel_btn.setEnabled(True)
        self._status.setText("检测中…")

        self._worker = DetectWorker(
            image=self._image,
            prompt=prompt,
            box_threshold=self.get_box_threshold(),
            text_threshold=self.get_text_threshold(),
            nms_threshold=self.get_nms_threshold(),
            detector_model=detector,
            model_ctrl=self._model_ctrl,
            img_width=self._image.width(),
            img_height=self._image.height(),
            class_names=self._closed_set_class_names(),
            project_class_names=self._project_class_names,
            unmatched=self.get_unmatched_policy(),
            parent=self,
        )
        self._worker.progress.connect(self._status.setText)
        self._worker.finished_ok.connect(self._on_ok)
        self._worker.failed.connect(self._on_fail)
        self._worker.finished.connect(self._on_thread_done)
        self._worker.start()

    def _cancel_worker(self) -> None:
        if self._worker is not None:
            self._worker.cancel()
            self._status.setText("正在取消…")

    def _on_ok(self, results: list) -> None:
        self._results = list(results)
        self._accept_btn.setEnabled(True)
        self._status.setText(f"检测完成：{len(results)} 个目标。点击「应用到预览」。")

    def _on_fail(self, message: str) -> None:
        QMessageBox.critical(self, "检测失败", message)
        self._status.setText("检测失败")

    def _on_thread_done(self) -> None:
        self._progress.setVisible(False)
        self._run_btn.setEnabled(True)
        self._cancel_btn.setEnabled(False)
