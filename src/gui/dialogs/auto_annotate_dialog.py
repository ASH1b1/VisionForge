"""批量自动标注对话框 - 使用 GroundingDINO 模型处理多张图片"""
from PySide6.QtWidgets import (
    QDialog, QVBoxLayout, QHBoxLayout, QLabel, QLineEdit,
    QPushButton, QSpinBox, QCheckBox, QProgressBar, QTextEdit, QMessageBox, QRadioButton, QButtonGroup,
    QGroupBox, QFrame, QComboBox
)
from PySide6.QtCore import Qt, Signal, QThread, QObject
from PySide6.QtGui import QImage
from pathlib import Path
from typing import List, Tuple, Optional, TYPE_CHECKING, Dict
import cv2
import numpy as np
import random
import logging

# 导入提示词增强器
from ..utils.prompt_enhancer import PromptEnhancer
from ..utils.split_utils import allocate_split_counts

# 导入统一标注处理模块（兼容 src.gui.* 运行与 tests 的 gui.* 导入）
try:
    from ...utils.annotation_processor import (
        process_raw_detections,
        YOLOAnnotationValidator,
        is_normalized_coords,
        xyxy_to_yolo
    )
except ImportError:
    from utils.annotation_processor import (
        process_raw_detections,
        YOLOAnnotationValidator,
        is_normalized_coords,
        xyxy_to_yolo
    )
from ..models.auto_annotation_result import AutoAnnotationResult
from ..models.detector_protocol import (
    UNMATCHED_CREATE,
    UNMATCHED_SKIP,
    filter_detections_by_project_classes,
    resolve_detector_handle,
)

# 配置日志
logger = logging.getLogger(__name__)

if TYPE_CHECKING:
    from ..models.grounding_dino_model import GroundingDINOModel
    from ..models.sam3_model import SAM3Model


def _apply_class_policy(annotations, project_class_names, unmatched):
    if not project_class_names:
        return list(annotations)
    return filter_detections_by_project_classes(
        annotations,
        project_class_names,
        unmatched=unmatched,
    )


def _class_names_from_prompt(prompt: str) -> List[str]:
    from ..models.grounding_dino_model import GroundingDINOModel

    return [name.lower() for name in GroundingDINOModel._parse_prompt_class_names(prompt)]


def _qimage_to_pil(qimage: QImage):
    """Convert QImage to RGB PIL Image without PNG encode/decode."""
    from PIL import Image

    if qimage is None or qimage.isNull():
        return None

    image = qimage
    if image.format() != QImage.Format.Format_RGBA8888:
        image = image.convertToFormat(QImage.Format.Format_RGBA8888)

    width = image.width()
    height = image.height()
    bytes_per_line = image.bytesPerLine()
    ptr = image.constBits()
    # PySide6 returns memoryview / sip.array; copy into contiguous buffer
    buf = bytes(ptr) if not isinstance(ptr, (bytes, bytearray)) else ptr
    arr = np.frombuffer(buf, dtype=np.uint8).reshape((height, bytes_per_line))
    arr = arr[:, : width * 4].reshape((height, width, 4))
    rgb = arr[:, :, :3].copy()
    return Image.fromarray(rgb, mode="RGB")


class SingleAnnotationWorker(QThread):
    """后台线程执行单张图片自动标注（重构版，使用统一处理模块）"""

    progress = Signal(str)
    # Must NOT be named `finished` — that shadows QThread.finished used for lifecycle.
    annotation_finished = Signal(list)
    error = Signal(str)

    def __init__(
        self,

        grounding_model: Optional['GroundingDINOModel'] = None,
        sam_model: Optional['SAM3Model'] = None,
        image: Optional[QImage] = None,
        prompt: str = "",
        box_threshold: float = 0.25,
        text_threshold: float = 0.25,
        enable_segmentation: bool = False,
        class_names: List[str] = None,
        nms_threshold: float = 0.5,
        iou_threshold: float = 0.5,
        sam_alignment_iou_threshold: float | None = None,
        model: Optional['GroundingDINOModel'] = None,
        use_amp: bool = True,
        model_ctrl=None,
        project_class_names=None,
        unmatched: str = UNMATCHED_CREATE,
    ):
        super().__init__()
        if grounding_model is None:
            grounding_model = model
        self._grounding_model = grounding_model
        self._sam_model = sam_model
        self._image = image
        self._prompt = prompt
        self._box_threshold = box_threshold
        self._text_threshold = text_threshold
        self._enable_segmentation = enable_segmentation
        self._class_names = class_names or []
        self._nms_threshold = nms_threshold
        self._iou_threshold = iou_threshold
        self._sam_alignment_iou_threshold = sam_alignment_iou_threshold if sam_alignment_iou_threshold is not None else iou_threshold
        self._use_amp = use_amp
        self._model_ctrl = model_ctrl
        self._project_class_names = dict(project_class_names or {})
        self._unmatched = unmatched
        self._is_cancelled = False

    @property
    def _detector_model(self):
        handle = resolve_detector_handle(
            grounding=self._grounding_model,
            model_ctrl=self._model_ctrl,
        )
        if handle is None:
            raise RuntimeError("No detection model loaded: load a detector first")
        return handle


    def _is_valid_annotation(self, x_center: float, y_center: float, width: float, height: float, class_name: str) -> bool:
        if not (0 <= x_center <= 1 and 0 <= y_center <= 1 and 0 < width <= 1 and 0 < height <= 1):
            return False

        area = width * height
        if area > 0.90:
            return False
        if area < 0.0001:
            return False

        aspect_ratio = width / height if height > 0 else float("inf")
        if aspect_ratio > 10.0 or aspect_ratio < 0.1:
            return False

        if area >= 0.5:
            center_distance = abs(x_center - 0.5) + abs(y_center - 0.5)
            if center_distance > 0.4:
                return False

        return True

    def cancel(self):
        """请求取消任务（若检测器支持 request_cancel 则调用）"""
        self._is_cancelled = True
        try:
            det = self._detector_model
        except Exception:
            det = None
        if det is not None and hasattr(det, "request_cancel"):
            det.request_cancel()

    def run(self):
        """
        执行自动标注任务（重构版）
        
        使用统一的 process_raw_detections 处理模块处理检测结果
        """
        ctrl = self._model_ctrl
        if ctrl is None:
            self.error.emit("缺少 ModelController，无法持有推理租约")
            return
        ctrl.acquire_inference_lease()
        try:
            if self._is_cancelled:
                return

            self.progress.emit("准备图像...")

            # Convert QImage to PIL Image without PNG round-trip
            from PIL import Image

            pil_image = _qimage_to_pil(self._image)
            if pil_image is None:
                raise RuntimeError("Failed to convert QImage to PIL Image")

            if self._is_cancelled:
                return

            # 获取图片尺寸
            img_width, img_height = pil_image.size
            self.progress.emit(f"图片尺寸: {img_width}x{img_height}")

            self.progress.emit("正在推理...")
            result = self._detector_model.infer(
                image=pil_image,
                text_prompt=self._prompt,
                box_threshold=self._box_threshold,
                text_threshold=self._text_threshold,
                use_amp=self._use_amp,
            )

            if self._is_cancelled:
                return

            if result is None:
                self.progress.emit("推理失败: 模型返回 None")
                self.annotation_finished.emit([])
                return

            if not result.boxes:
                self.progress.emit("未检测到目标")
                self.annotation_finished.emit([])
                return

            self.progress.emit(f"模型返回 {len(result.boxes)} 个检测框")

            # 使用统一的处理模块处理检测结果
            # process_raw_detections 内部完成：
            # 1. 坐标转换（像素→YOLO格式）
            # 2. 标签映射
            # 3. NMS过滤
            # 4. 质量验证
            final_annotations = _apply_class_policy(
                process_raw_detections(
                    boxes=result.boxes,
                    labels=result.labels,
                    scores=result.scores,
                    class_names=self._class_names,
                    img_width=img_width,
                    img_height=img_height,
                    nms_threshold=self._nms_threshold
                ),
                self._project_class_names,
                self._unmatched,
            )

            results = [
                AutoAnnotationResult(class_name=class_name, bbox=bbox, score=score)
                for class_name, bbox, score in final_annotations
            ]

            if self._enable_segmentation and self._sam_model is not None and self._sam_model.is_loaded():
                polygons = self._sam_model.align_detections(
                    pil_image,
                    self._prompt,
                    [item.bbox for item in results],
                    iou_threshold=self._sam_alignment_iou_threshold,
                )
                self._emit_sam_diagnostics(len(results))
                for item, polygon in zip(results, polygons):
                    item.polygon = polygon

            self.progress.emit(f"完成! 共 {len(results)} 个有效标注")
            self.annotation_finished.emit(results)

        except Exception as e:
            import traceback
            error_detail = f"{str(e)}\n\n{traceback.format_exc()}"
            self.error.emit(error_detail)
        finally:
            ctrl.release_inference_lease()

    def _run_for_test(self):
        from PIL import Image

        if isinstance(self._image, QImage):
            pil_image = Image.new("RGB", (self._image.width(), self._image.height()))
            img_width, img_height = pil_image.size
        else:
            pil_image = self._image
            img_width, img_height = pil_image.size

        result = self._detector_model.infer(
            image=pil_image,
            text_prompt=self._prompt,
            box_threshold=self._box_threshold,
            text_threshold=self._text_threshold,
            use_amp=self._use_amp,
        )
        final_annotations = _apply_class_policy(
            process_raw_detections(
                boxes=result.boxes,
                labels=result.labels,
                scores=result.scores,
                class_names=self._class_names,
                img_width=img_width,
                img_height=img_height,
                nms_threshold=self._nms_threshold,
            ),
            self._project_class_names,
            self._unmatched,
        )
        results = [
            AutoAnnotationResult(class_name=class_name, bbox=bbox, score=score)
            for class_name, bbox, score in final_annotations
        ]
        if self._enable_segmentation and self._sam_model is not None and self._sam_model.is_loaded():
            polygons = self._sam_model.align_detections(
                pil_image,
                self._prompt,
                [item.bbox for item in results],
                iou_threshold=self._sam_alignment_iou_threshold,
            )
            for item, polygon in zip(results, polygons):
                item.polygon = polygon
        return results

    def _emit_sam_diagnostics(self, detection_count: int):
        if self._sam_model is None or not hasattr(self._sam_model, "get_last_alignment_info"):
            return
        info = self._sam_model.get_last_alignment_info()
        if not info:
            return

        self.progress.emit(
            "SAM3诊断: detection=%s, sam_boxes=%s, sam_masks=%s, matched=%s, align_iou=%.2f, mode=%s" % (
                detection_count,
                info.get("sam_box_count", 0),
                info.get("sam_mask_count", 0),
                info.get("matched_count", 0),
                info.get("alignment_iou_threshold", 0.0),
                info.get("alignment_mode", "unknown"),
            )
        )
        if info.get("error"):
            self.progress.emit(f"SAM3错误: {info['error']}")


# 包装批量结果以解决 PySide6 Signal(dict) 传递问题
class BatchResultsWrapper(QObject):
    def __init__(self, results: Dict[int, List]):
        super().__init__()
        self.results = results


class BatchAnnotationWorker(QThread):
    """
    批量标注多张图片（重构版）
    
    使用统一的 process_raw_detections 处理模块处理检测结果
    """

    progress = Signal(str)  # Status message
    batch_progress = Signal(int, int)  # current, total
    image_finished = Signal(int, list)  # frame_index, annotations
    batch_finished = Signal(object)  # BatchResultsWrapper
    error = Signal(str)

    def __init__(
        self,
        grounding_model: 'GroundingDINOModel',
        sam_model: Optional['SAM3Model'],
        image_files: List[Path],
        prompt: str,
        box_threshold: float,
        text_threshold: float,
        enable_segmentation: bool = False,
        class_names: List[str] = None,
        nms_threshold: float = 0.5,
        iou_threshold: float = 0.5,
        sam_alignment_iou_threshold: float | None = None,
        use_amp: bool = True,
        model_ctrl=None,
        project_class_names=None,
        unmatched: str = UNMATCHED_CREATE,
    ):
        super().__init__()
        self._grounding_model = grounding_model
        self._sam_model = sam_model
        self._enable_segmentation = enable_segmentation
        self._image_files = image_files
        self._prompt = prompt
        self._box_threshold = box_threshold
        self._text_threshold = text_threshold
        self._class_names = class_names or []
        self._nms_threshold = nms_threshold
        self._iou_threshold = iou_threshold
        self._sam_alignment_iou_threshold = sam_alignment_iou_threshold if sam_alignment_iou_threshold is not None else iou_threshold
        self._use_amp = use_amp
        self._model_ctrl = model_ctrl
        self._project_class_names = dict(project_class_names or {})
        self._unmatched = unmatched
        self._is_cancelled = False

    @property
    def _detector_model(self):
        handle = resolve_detector_handle(
            grounding=self._grounding_model,
            model_ctrl=self._model_ctrl,
        )
        if handle is None:
            raise RuntimeError("No detection model loaded: load a detector first")
        return handle


    def cancel(self):
        """请求取消任务（若检测器支持 request_cancel 则调用）"""
        self._is_cancelled = True
        try:
            det = self._detector_model
        except Exception:
            det = None
        if det is not None and hasattr(det, "request_cancel"):
            det.request_cancel()

    @staticmethod
    def _decode_image(img_path: Path):
        """Decode image path to RGB PIL; None on failure."""
        from PIL import Image

        img_path_str = str(img_path)
        try:
            img_array = np.fromfile(img_path_str, dtype=np.uint8)
            img_array = cv2.imdecode(img_array, cv2.IMREAD_COLOR)
            if img_array is None:
                img_array = cv2.imread(img_path_str)
        except Exception:
            img_array = cv2.imread(img_path_str)
        if img_array is None:
            return None
        img_array = cv2.cvtColor(img_array, cv2.COLOR_BGR2RGB)
        return Image.fromarray(img_array).convert("RGB")

    def run(self):
        """批量标注：预读下一张与当前推理重叠（仍逐张串行 GPU 推理）。"""
        def _file_log(msg):
            return

        ctrl = self._model_ctrl
        if ctrl is None:
            self.error.emit("缺少 ModelController，无法持有推理租约")
            return
        ctrl.acquire_inference_lease()
        try:
            from concurrent.futures import ThreadPoolExecutor

            total = len(self._image_files)
            results: Dict[int, List] = {}
            self.progress.emit(f"开始处理 {total} 张图片...")

            with ThreadPoolExecutor(max_workers=1) as io_pool:
                next_future = None
                if total > 0:
                    next_future = io_pool.submit(self._decode_image, self._image_files[0])

                for idx, img_path in enumerate(self._image_files):
                    if self._is_cancelled:
                        if next_future is not None:
                            next_future.cancel()
                        self.progress.emit("批处理已取消")
                        return

                    self.batch_progress.emit(idx + 1, total)
                    self.progress.emit(f"[{idx + 1}/{total}] 处理: {img_path.name}")

                    prefetch = None
                    if idx + 1 < total and not self._is_cancelled:
                        prefetch = io_pool.submit(self._decode_image, self._image_files[idx + 1])

                    pil_image = None
                    if next_future is not None:
                        try:
                            pil_image = next_future.result()
                        except Exception:
                            pil_image = None
                    next_future = prefetch

                    if pil_image is None:
                        self.progress.emit(f"无法读取图片，已跳过: {img_path.name}")
                        continue

                    img_width, img_height = pil_image.size
                    _file_log(f"[{idx+1}/{total}] 调用 infer(), image={img_width}x{img_height}")
                    result = self._detector_model.infer(
                        image=pil_image,
                        text_prompt=self._prompt,
                        box_threshold=self._box_threshold,
                        text_threshold=self._text_threshold,
                        use_amp=self._use_amp,
                    )

                    if result is None:
                        self.progress.emit(f"推理失败，已保留原标注: {img_path.name}")
                        continue

                    if not result.boxes:
                        results[idx] = []
                        self.image_finished.emit(idx, [])
                        continue

                    final_annotations = _apply_class_policy(
                        process_raw_detections(
                            boxes=result.boxes,
                            labels=result.labels,
                            scores=result.scores,
                            class_names=self._class_names,
                            img_width=img_width,
                            img_height=img_height,
                            nms_threshold=self._nms_threshold
                        ),
                        self._project_class_names,
                        self._unmatched,
                    )
                    processed_results = [
                        AutoAnnotationResult(class_name=class_name, bbox=bbox, score=score)
                        for class_name, bbox, score in final_annotations
                    ]
                    if self._enable_segmentation and self._sam_model is not None and self._sam_model.is_loaded():
                        polygons = self._sam_model.align_detections(
                            pil_image,
                            self._prompt,
                            [item.bbox for item in processed_results],
                            iou_threshold=self._sam_alignment_iou_threshold,
                        )
                        for item, polygon in zip(processed_results, polygons):
                            item.polygon = polygon

                    results[idx] = processed_results
                    self.image_finished.emit(idx, processed_results)

            if self._is_cancelled:
                return

            self.batch_progress.emit(total, total)
            self.progress.emit(f"批处理完成! 共处理 {total} 张图片")
            self.batch_finished.emit(BatchResultsWrapper(results))

        except Exception as e:
            import traceback
            self.error.emit(f"{str(e)}\n\n{traceback.format_exc()}")
        finally:
            ctrl.release_inference_lease()


class AutoAnnotateDialog(QDialog):
    """自动标注对话框 - 支持单张和批量模式"""

    def __init__(self, grounding_model: 'GroundingDINOModel', sam_model: Optional['SAM3Model'] = None, model_ctrl=None, parent=None, batch_mode: bool = False, image_files: Optional[List[Path]] = None, project_class_names=None):
        super().__init__(parent)
        self._model = grounding_model
        self._sam_model = sam_model
        self._model_ctrl = model_ctrl
        self._project_class_names = dict(project_class_names or {})
        self._batch_mode = batch_mode
        self._image_files = image_files or []
        self._annotations: List[AutoAnnotationResult] = []
        self._batch_results: Dict[int, List] = {}
        self._split_result: Dict[int, str] = {}  # frame_index -> "train"/"val"/"test"
        self._worker: Optional[QThread] = None
        self._close_when_worker_done = False
        self._accept_when_worker_done = False
        self._setup_ui()

        # Set mode
        if self._batch_mode and self._image_files:
            self._set_batch_mode()
        else:
            self._set_single_mode()

    def _setup_ui(self):
        """设置UI"""
        self.setWindowTitle("自动标注图像")
        self.setMinimumWidth(500)

        layout = QVBoxLayout(self)

        # Mode selection
        mode_layout = QHBoxLayout()
        mode_layout.addWidget(QLabel("处理模式:"))
        self.mode_group = QButtonGroup(self)
        self.single_mode_btn = QRadioButton("单张图片")
        self.batch_mode_btn = QRadioButton("批量处理所有图片")
        self.single_mode_btn.setChecked(True)
        self.mode_group.addButton(self.single_mode_btn, 0)
        self.mode_group.addButton(self.batch_mode_btn, 1)
        self.single_mode_btn.toggled.connect(self._on_mode_changed)
        mode_layout.addWidget(self.single_mode_btn)
        mode_layout.addWidget(self.batch_mode_btn)
        layout.addLayout(mode_layout)

        # Prompt input
        layout.addWidget(QLabel("文本提示 (逗号分隔):"))
        self.prompt_input = QLineEdit()
        self.prompt_input.setPlaceholderText("例如: person, car, dog")
        self.prompt_input.setText("person, car, bicycle")
        layout.addWidget(self.prompt_input)

        # Thresholds
        thresh_layout = QHBoxLayout()
        thresh_layout.addWidget(QLabel("框阈值:"))
        self.box_threshold = QSpinBox()
        self.box_threshold.setRange(1, 100)
        self.box_threshold.setValue(25)
        self.box_threshold.setSuffix("%")
        self.box_threshold.setSingleStep(5)
        thresh_layout.addWidget(self.box_threshold)

        thresh_layout.addWidget(QLabel("文本阈值:"))
        self.text_threshold = QSpinBox()
        self.text_threshold.setRange(1, 100)
        self.text_threshold.setValue(25)
        self.text_threshold.setSuffix("%")
        self.text_threshold.setSingleStep(5)
        thresh_layout.addWidget(self.text_threshold)
        layout.addLayout(thresh_layout)

        # 模型类型提示
        self._model_hint_label = QLabel()
        detector = resolve_detector_handle(
            grounding=self._model,
            model_ctrl=self._model_ctrl,
        )
        if detector is not None and hasattr(detector, "class_names_list"):
            self._model_hint_label.setText("当前为自定义 YOLO ONNX，使用模型内类别，无需文本提示")
            self._model_hint_label.setStyleSheet("color: #FF9800; font-style: italic;")
            self.prompt_input.setPlaceholderText("YOLO ONNX 预标可留空；文本提示仅用于开放词汇检测器")
        layout.addWidget(self._model_hint_label)

        unmatched_row = QHBoxLayout()
        unmatched_row.addWidget(QLabel("未知类别:"))
        self._unmatched_combo = QComboBox()
        self._unmatched_combo.addItem("创建新类", UNMATCHED_CREATE)
        self._unmatched_combo.addItem("跳过未知类", UNMATCHED_SKIP)
        unmatched_row.addWidget(self._unmatched_combo)
        unmatched_row.addStretch()
        layout.addLayout(unmatched_row)

        # NMS and IOU thresholds
        nms_layout = QHBoxLayout()
        nms_layout.addWidget(QLabel("NMS 阈值:"))
        self.nms_threshold = QSpinBox()
        self.nms_threshold.setRange(10, 100)
        self.nms_threshold.setValue(50)  # 默认50%，行业标准阈值，过滤高度重叠框
        self.nms_threshold.setSuffix("%")
        self.nms_threshold.setSingleStep(5)
        self.nms_threshold.setToolTip("非极大值抑制阈值，100%禁用。IOU > 此值时过滤重叠框")
        nms_layout.addWidget(self.nms_threshold)

        nms_layout.addWidget(QLabel("IOU 阈值:"))
        self.iou_threshold = QSpinBox()
        self.iou_threshold.setRange(10, 95)
        self.iou_threshold.setValue(50)
        self.iou_threshold.setSuffix("%")
        self.iou_threshold.setSingleStep(5)
        self.iou_threshold.setToolTip("IoU 阈值，用于 NMS 判断两个框的重叠程度")
        nms_layout.addWidget(self.iou_threshold)
        layout.addLayout(nms_layout)

        sam_layout = QHBoxLayout()
        sam_layout.addWidget(QLabel("SAM3对齐IoU:"))
        self.sam_alignment_iou_threshold = QSpinBox()
        self.sam_alignment_iou_threshold.setRange(1, 95)
        self.sam_alignment_iou_threshold.setValue(10)
        self.sam_alignment_iou_threshold.setSuffix("%")
        self.sam_alignment_iou_threshold.setSingleStep(5)
        self.sam_alignment_iou_threshold.setToolTip("GroundingDINO框与SAM3框的匹配IoU阈值")
        sam_layout.addWidget(self.sam_alignment_iou_threshold)

        sam_layout.addWidget(QLabel("SAM3置信度:"))
        self.sam_confidence_threshold = QSpinBox()
        self.sam_confidence_threshold.setRange(1, 100)
        self.sam_confidence_threshold.setValue(25)
        self.sam_confidence_threshold.setSuffix("%")
        self.sam_confidence_threshold.setSingleStep(5)
        self.sam_confidence_threshold.setToolTip("SAM3内部mask过滤阈值，过高会导致0 mask")
        sam_layout.addWidget(self.sam_confidence_threshold)
        layout.addLayout(sam_layout)

        # ── Dataset Split Config ──────────────────────────────────────────────
        split_group = QGroupBox("数据集划分 (批量模式)")
        split_group.setCheckable(True)
        split_group.setChecked(True)
        self.split_group = split_group
        split_layout = QVBoxLayout(split_group)
        split_layout.setContentsMargins(6, 4, 6, 4)
        split_layout.setSpacing(4)

        # Preset buttons row
        preset_row = QHBoxLayout()
        preset_row.addWidget(QLabel("预设:"))
        for label, vals in [("8:1:1", (80, 10, 10)), ("8:2:0", (80, 20, 0)), ("7:2:1", (70, 20, 10)), ("全部Train", (100, 0, 0))]:
            btn = QPushButton(label)
            btn.setFixedHeight(22)
            btn.setStyleSheet("padding: 0 4px;")
            btn.clicked.connect(lambda _=False, v=vals: self._apply_split_preset(*v))
            preset_row.addWidget(btn)
        preset_row.addStretch()
        split_layout.addLayout(preset_row)

        # Ratio spinboxes row
        ratio_row = QHBoxLayout()
        ratio_row.addWidget(QLabel("Train:"))
        self.split_train = QSpinBox()
        self.split_train.setRange(0, 100)
        self.split_train.setValue(80)
        self.split_train.setSuffix("%")
        self.split_train.setFixedWidth(70)
        self.split_train.valueChanged.connect(self._on_split_changed)
        ratio_row.addWidget(self.split_train)

        ratio_row.addWidget(QLabel("Val:"))
        self.split_val = QSpinBox()
        self.split_val.setRange(0, 100)
        self.split_val.setValue(10)
        self.split_val.setSuffix("%")
        self.split_val.setFixedWidth(70)
        self.split_val.valueChanged.connect(self._on_split_changed)
        ratio_row.addWidget(self.split_val)

        ratio_row.addWidget(QLabel("Test:"))
        self.split_test = QSpinBox()
        self.split_test.setRange(0, 100)
        self.split_test.setValue(10)
        self.split_test.setSuffix("%")
        self.split_test.setFixedWidth(70)
        self.split_test.valueChanged.connect(self._on_split_changed)
        ratio_row.addWidget(self.split_test)

        # Sum indicator
        self.split_sum_label = QLabel("= 100% ✓")
        self.split_sum_label.setStyleSheet("color: #4CAF50;")
        ratio_row.addWidget(self.split_sum_label)
        ratio_row.addStretch()
        split_layout.addLayout(ratio_row)

        # Random seed checkbox
        seed_row = QHBoxLayout()
        self.split_fixed_seed = QCheckBox("固定随机种子 (可复现)")
        self.split_fixed_seed.setChecked(True)
        seed_row.addWidget(self.split_fixed_seed)
        seed_row.addStretch()
        split_layout.addLayout(seed_row)

        layout.addWidget(split_group)
        # ─────────────────────────────────────────────────────────────────────

        # GPU checkbox
        self.use_gpu = QCheckBox("使用 GPU (AMP)")
        self.use_gpu.setChecked(True)
        self.use_gpu.setToolTip(
            "启用 CUDA autocast（AMP）。取消后仍用 GPU，但关闭半精度运算；"
            "不重新加载权重。"
        )
        layout.addWidget(self.use_gpu)

        self._segmentation_checkbox = QCheckBox("同时生成多边形（更慢）")
        self._segmentation_checkbox.setChecked(False)
        # Allow checking even when SAM is not loaded yet; ModelController loads on demand.
        self._segmentation_checkbox.setEnabled(True)
        self._segmentation_checkbox.setToolTip(
            "将额外加载/调用 SAM3 生成多边形，显著增加耗时。\n"
            "推荐先只出框，再用工具菜单「补分割」。"
        )
        layout.addWidget(self._segmentation_checkbox)

        # Prompt enhancement checkbox
        enhance_layout = QHBoxLayout()
        self.prompt_enhance_check = QCheckBox("启用提示词增强")
        self.prompt_enhance_check.setChecked(False)
        self.prompt_enhance_check.setToolTip("使用同义词扩展增强检测效果，包含遥感领域专用词库")
        enhance_layout.addWidget(self.prompt_enhance_check)
        enhance_layout.addStretch()
        layout.addLayout(enhance_layout)

        # Initialize prompt enhancer
        self._prompt_enhancer = PromptEnhancer(enable_remote_sensing=True)

        # Batch info label
        self.batch_info_label = QLabel("")
        self.batch_info_label.setVisible(False)
        layout.addWidget(self.batch_info_label)

        # Progress
        self.progress_label = QLabel("")
        layout.addWidget(self.progress_label)

        self.progress_bar = QProgressBar()
        self.progress_bar.setRange(0, 0)
        self.progress_bar.hide()
        layout.addWidget(self.progress_bar)

        # Log
        layout.addWidget(QLabel("日志:"))
        self.log_text = QTextEdit()
        self.log_text.setMaximumHeight(100)
        self.log_text.setReadOnly(True)
        layout.addWidget(self.log_text)

        # Buttons
        btn_layout = QHBoxLayout()
        self.run_btn = QPushButton("▶ 运行")
        self.run_btn.clicked.connect(self._on_run)
        self.cancel_btn = QPushButton("取消")
        self.cancel_btn.clicked.connect(self._on_cancel)
        btn_layout.addStretch()
        btn_layout.addWidget(self.run_btn)
        btn_layout.addWidget(self.cancel_btn)
        layout.addLayout(btn_layout)

    def _set_single_mode(self):
        """设置单张模式"""
        self._batch_mode = False
        self.batch_info_label.setVisible(False)
        self.progress_bar.setRange(0, 0)
        # 单张模式下折叠 split 区域
        self.split_group.setChecked(False)
        self.split_group.setEnabled(False)

    def _set_batch_mode(self):
        """设置批量模式"""
        self._batch_mode = True
        count = len(self._image_files)
        self.batch_info_label.setText(f"将处理 {count} 张图片")
        self.batch_info_label.setVisible(True)
        self.batch_mode_btn.setChecked(True)
        self.progress_bar.setRange(0, count)
        self.split_group.setEnabled(True)

    # ── Split helpers ─────────────────────────────────────────────────────────

    def _apply_split_preset(self, train: int, val: int, test: int):
        """应用预设比例"""
        self.split_train.blockSignals(True)
        self.split_val.blockSignals(True)
        self.split_test.blockSignals(True)
        self.split_train.setValue(train)
        self.split_val.setValue(val)
        self.split_test.setValue(test)
        self.split_train.blockSignals(False)
        self.split_val.blockSignals(False)
        self.split_test.blockSignals(False)
        self._on_split_changed()

    def _on_split_changed(self):
        """划分比例变化时更新合计提示"""
        total = self.split_train.value() + self.split_val.value() + self.split_test.value()
        if total == 100:
            self.split_sum_label.setText("= 100% ✓")
            self.split_sum_label.setStyleSheet("color: #4CAF50;")
        else:
            self.split_sum_label.setText(f"= {total}%  ✗")
            self.split_sum_label.setStyleSheet("color: #f44336;")

    def _compute_split(self, image_files: List[Path]) -> Dict[int, str]:
        """计算 train/val/test 划分，返回 {index: split_name}"""
        n = len(image_files)
        if n == 0:
            return {}

        train_pct = self.split_train.value() / 100.0
        val_pct   = self.split_val.value()   / 100.0
        test_pct  = self.split_test.value()  / 100.0

        indices = list(range(n))
        seed = 42 if self.split_fixed_seed.isChecked() else None
        rng = random.Random(seed)
        rng.shuffle(indices)

        n_train, n_val, _n_test = allocate_split_counts(n, train_pct, val_pct, test_pct)
        train_end = n_train
        val_end   = n_train + n_val

        result: Dict[int, str] = {}
        for pos, idx in enumerate(indices):
            if pos < train_end:
                result[idx] = "train"
            elif pos < val_end:
                result[idx] = "val"
            else:
                result[idx] = "test"
        return result

    # ─────────────────────────────────────────────────────────────────────────

    def _on_mode_changed(self):
        """模式切换"""
        if self.batch_mode_btn.isChecked():
            if self._image_files:
                self._set_batch_mode()
            else:
                QMessageBox.warning(self, "警告", "没有可处理的图片列表")
                self.single_mode_btn.setChecked(True)
        else:
            self._set_single_mode()

    def _log(self, message: str) -> None:
        """添加日志"""
        self.log_text.append(message)

    def _apply_sam_settings(self) -> None:
        if self._sam_model is None or not self._sam_model.is_loaded():
            return
        if hasattr(self._sam_model, "set_confidence_threshold"):
            self._sam_model.set_confidence_threshold(self.sam_confidence_threshold.value() / 100.0)

    def _on_run(self) -> None:
        """运行自动标注"""
        if self._model_ctrl is None:
            QMessageBox.warning(
                self,
                "警告",
                "缺少 ModelController，无法启动自动标注（需持有推理租约）。",
            )
            return

        detector = resolve_detector_handle(
            grounding=self._model,
            model_ctrl=self._model_ctrl,
        )
        if detector is None:
            QMessageBox.warning(self, "警告", "请先加载 GroundingDINO 或自定义 YOLO")
            return

        prompt = self.prompt_input.text().strip()
        closed_set = hasattr(detector, "class_names_list")
        if not closed_set and not prompt:
            QMessageBox.warning(self, "警告", "请输入文本提示")
            return

        # Clear stale cancel from a previous Cancel click
        if hasattr(detector, "clear_cancel"):
            detector.clear_cancel()

        # Disable controls
        self.run_btn.setEnabled(False)
        self.prompt_input.setEnabled(False)
        self.single_mode_btn.setEnabled(False)
        self.batch_mode_btn.setEnabled(False)
        self.progress_bar.show()
        self._log(f"提示词: {prompt}")
        self._log(f"框阈值: {self.box_threshold.value()}%")
        self._log(f"文本阈值: {self.text_threshold.value()}%")
        self._log(f"NMS 阈值: {self.nms_threshold.value()}%")
        self._log(f"IOU 阈值: {self.iou_threshold.value()}%")
        self._log(f"SAM3 对齐IoU: {self.sam_alignment_iou_threshold.value()}%")
        self._log(f"SAM3 置信度: {self.sam_confidence_threshold.value()}%")

        box_thresh = self.box_threshold.value() / 100.0
        text_thresh = self.text_threshold.value() / 100.0
        # Both NMS and IOU controls describe NMS IoU; apply the stricter value
        nms_thresh = min(self.nms_threshold.value(), self.iou_threshold.value()) / 100.0
        iou_thresh = self.iou_threshold.value() / 100.0
        sam_align_iou_thresh = self.sam_alignment_iou_threshold.value() / 100.0
        use_amp = self.use_gpu.isChecked()
        self._apply_sam_settings()

        # 解析类别名称列表（与 infer() 一致：逗号/分号分隔，保留多词类名）
        if closed_set:
            class_names = list(detector.class_names_list())
            self._log(f"YOLO 类别列表: {class_names}")
        else:
            class_names = _class_names_from_prompt(prompt)
            self._log(f"类别列表: {class_names}")
        self._log(f"GPU AMP: {'开' if use_amp else '关'}")

        # 应用提示词增强（封闭集 YOLO 不改 names）
        if not closed_set and self.prompt_enhance_check.isChecked():
            original_prompt = prompt
            prompt = self._prompt_enhancer.enhance(prompt)
            class_names = self._prompt_enhancer.enhance_list(original_prompt)
            self._log(f"提示词增强后: {prompt}")
            self._log(f"增强后类别列表: {class_names}")

        want_seg = self.is_segmentation_enabled()
        if want_seg:
            if self._model_ctrl is None:
                QMessageBox.warning(
                    self,
                    "警告",
                    "无法按需加载 SAM3（缺少 ModelController）。请取消「同时生成多边形」或先从模型菜单加载 SAM3。",
                )
                self._restore_controls()
                return
            if not self._model_ctrl.ensure_sam3_loaded(self):
                QMessageBox.warning(
                    self,
                    "已中止",
                    "SAM3 未加载，已中止自动标注。\n可取消「同时生成多边形」后仅出框，或稍后再试。",
                )
                self._restore_controls()
                return
            # Refresh SAM handle after on-demand load
            self._sam_model = self._model_ctrl.get_sam_model()
            self._apply_sam_settings()

        if self._batch_mode and self._image_files:
            # Batch mode
            self._worker = BatchAnnotationWorker(
                self._model,
                self._sam_model,
                self._image_files,
                prompt,
                box_thresh,
                text_thresh,
                want_seg,
                class_names,
                nms_thresh,
                iou_thresh,
                sam_align_iou_thresh,
                use_amp=use_amp,
                model_ctrl=self._model_ctrl,
                project_class_names=self._project_class_names,
                unmatched=self.get_unmatched_policy(),
            )
            self._worker.batch_progress.connect(self._on_batch_progress)
            self._worker.batch_finished.connect(self._on_batch_finished)
        else:
            # Single image mode
            parent = self.parent()
            if parent and hasattr(parent, 'canvas') and parent.canvas._image:
                image = parent.canvas._image
            else:
                QMessageBox.warning(self, "警告", "未加载图片")
                self._restore_controls()
                return

            self._worker = SingleAnnotationWorker(
                self._model,
                self._sam_model,
                image,
                prompt,
                box_thresh,
                text_thresh,
                want_seg,
                class_names,
                nms_thresh,
                iou_thresh,
                sam_align_iou_thresh,
                use_amp=use_amp,
                model_ctrl=self._model_ctrl,
                project_class_names=self._project_class_names,
                unmatched=self.get_unmatched_policy(),
            )
            self._worker.annotation_finished.connect(self._on_single_finished)

        self._worker.progress.connect(self._on_progress)
        self._worker.error.connect(self._on_error)
        # Inherited QThread.finished (not custom result signal) for deferred close.
        self._worker.finished.connect(self._on_worker_thread_finished)
        self._worker.start()

    def _restore_controls(self):
        """恢复控件状态"""
        self.run_btn.setEnabled(True)
        self.prompt_input.setEnabled(True)
        self.single_mode_btn.setEnabled(True)
        self.batch_mode_btn.setEnabled(True)
        self.progress_bar.hide()

    def _on_cancel(self) -> None:
        """取消操作：协作打断 generate，线程结束后再关闭对话框。"""
        if self._worker and self._worker.isRunning():
            self._worker.cancel()
            self.progress_label.setText("正在停止…")
            self.run_btn.setEnabled(False)
            self.cancel_btn.setEnabled(False)
            self._close_when_worker_done = True
            return
        self.reject()

    def _on_worker_thread_finished(self) -> None:
        """Worker thread fully stopped — finish deferred close/accept if requested."""
        if self._close_when_worker_done:
            self._close_when_worker_done = False
            self._accept_when_worker_done = False
            self.reject()
            return
        if self._accept_when_worker_done:
            self._accept_when_worker_done = False
            self.accept()

    def _on_progress(self, message: str) -> None:
        """进度更新"""
        self.progress_label.setText(message)
        self._log(message)

    def _on_batch_progress(self, current: int, total: int) -> None:
        """批量进度更新"""
        self.progress_bar.setValue(current)
        self.progress_bar.setMaximum(total)
        self.progress_label.setText(f"处理中: {current}/{total}")

    def _defer_accept_until_worker_finished(self) -> None:
        """Accept only after QThread.finished — never timed wait(3000)."""
        if self._worker and self._worker.isRunning():
            self._accept_when_worker_done = True
            return
        self.accept()

    def _disconnect_worker_result_signals(self) -> None:
        """Disconnect result/progress/error; keep QThread.finished for deferred accept."""
        if not self._worker:
            return
        for signal_name in (
            "annotation_finished",
            "batch_finished",
            "progress",
            "batch_progress",
            "error",
        ):
            signal = getattr(self._worker, signal_name, None)
            if signal is None:
                continue
            try:
                signal.disconnect()
            except (RuntimeError, TypeError):
                pass

    def _on_single_finished(self, annotations: list) -> None:
        """单张模式完成"""
        if self._close_when_worker_done:
            return
        self._annotations = annotations
        self._log(f"找到 {len(annotations)} 个标注")

        self._restore_controls()
        self.progress_label.setText(f"完成! 找到 {len(annotations)} 个对象")
        self._disconnect_worker_result_signals()

        QMessageBox.information(
            self, "成功",
            f"找到 {len(annotations)} 个标注。\n点击确定添加到当前帧。"
        )
        self._defer_accept_until_worker_finished()

    def _on_batch_finished(self, wrapper: object) -> None:
        """批量模式完成"""
        if self._close_when_worker_done:
            return
        self._log(f"[Debug] _on_batch_finished called")
        
        # 从 wrapper 中提取结果
        if isinstance(wrapper, BatchResultsWrapper):
            results = wrapper.results
        elif isinstance(wrapper, dict):
            results = wrapper
        else:
            logger.error(f"Unexpected batch result type: {type(wrapper)}")
            results = {}
        self._log(f"[Debug] extracted results with {len(results)} entries")
        
        self._batch_results = results
        total_count = sum(len(anns) for anns in results.values())
        self._log(f"批处理完成! 共找到 {total_count} 个标注")

        # ── 计算数据集划分 ──────────────────────────────────────────────────
        if self.split_group.isChecked() and self.split_group.isEnabled() and self._image_files:
            total_split = self.split_train.value() + self.split_val.value() + self.split_test.value()
            if total_split == 100:
                self._split_result = self._compute_split(self._image_files)
                counts = {"train": 0, "val": 0, "test": 0}
                for s in self._split_result.values():
                    counts[s] = counts.get(s, 0) + 1
                self._log(
                    f"数据集划分: Train={counts['train']} | Val={counts['val']} | Test={counts['test']}"
                )
            else:
                self._log(f"划分比例合计 {total_split}% ≠ 100%，跳过划分")
                self._split_result = {}
        else:
            self._split_result = {}
        # ───────────────────────────────────────────────────────────────────

        self._restore_controls()
        self.progress_label.setText(f"完成! 共 {total_count} 个标注")
        self._disconnect_worker_result_signals()

        split_info = ""
        if self._split_result:
            counts = {"train": 0, "val": 0, "test": 0}
            for s in self._split_result.values():
                counts[s] = counts.get(s, 0) + 1
            split_info = (
                f"\n\n数据集划分:\n"
                f"  Train: {counts['train']} 张\n"
                f"  Val:   {counts['val']} 张\n"
                f"  Test:  {counts['test']} 张"
            )

        QMessageBox.information(
            self, "完成",
            f"批处理完成!\n处理了 {len(results)} 张图片，共找到 {total_count} 个标注。"
            f"{split_info}\n\n点击确定应用所有标注。"
        )
        self._defer_accept_until_worker_finished()

    def _on_error(self, error_msg: str) -> None:
        """错误"""
        if self._close_when_worker_done:
            return
        self._log(f"错误: {error_msg}")
        self._restore_controls()
        self.progress_label.setText("错误!")

        display_error = error_msg if len(error_msg) < 500 else error_msg[:500] + "..."
        QMessageBox.critical(self, "错误", f"标注失败:\n{display_error}")

    def closeEvent(self, event):
        """对话框关闭时：线程未结束则忽略关闭，待 finished 后再关。"""
        if self._worker and self._worker.isRunning():
            self._worker.cancel()
            self.progress_label.setText("正在停止…")
            self.run_btn.setEnabled(False)
            self.cancel_btn.setEnabled(False)
            self._close_when_worker_done = True
            event.ignore()
            return
        event.accept()

    def get_annotations(self) -> List[AutoAnnotationResult]:
        """获取单张模式标注结果"""
        return self._annotations

    def get_batch_results(self) -> Dict[int, List[AutoAnnotationResult]]:
        """获取批量模式标注结果"""
        return self._batch_results

    def get_split_result(self) -> Dict[int, str]:
        """获取数据集划分结果 {frame_index: 'train'/'val'/'test'}"""
        return self._split_result

    def is_batch_mode(self) -> bool:
        """是否为批量模式"""
        return self._batch_mode

    def get_prompt(self) -> str:
        """获取文本提示"""
        return self.prompt_input.text()

    def get_unmatched_policy(self) -> str:
        data = self._unmatched_combo.currentData()
        return data if data in (UNMATCHED_CREATE, UNMATCHED_SKIP) else UNMATCHED_CREATE

    def get_box_threshold(self) -> float:
        """获取框阈值"""
        return self.box_threshold.value() / 100.0

    def get_text_threshold(self) -> float:
        """获取文本阈值"""
        return self.text_threshold.value() / 100.0

    def get_nms_threshold(self) -> float:
        """获取NMS阈值"""
        return self.nms_threshold.value() / 100.0

    def get_iou_threshold(self) -> float:
        """获取IoU阈值"""
        return self.iou_threshold.value() / 100.0

    def is_segmentation_enabled(self):
        return self._segmentation_checkbox.isChecked()
