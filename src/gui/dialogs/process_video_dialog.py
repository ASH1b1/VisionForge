"""
视频处理对话框 - GroundingDINO 推理参数设置

批处理与自动标注共用 process_raw_detections（NMS + 质量过滤）。
"""
from PySide6.QtWidgets import (
    QDialog, QVBoxLayout, QHBoxLayout, QFormLayout,
    QLineEdit, QSpinBox, QDoubleSpinBox, QCheckBox,
    QPushButton, QProgressBar, QLabel, QGroupBox, QMessageBox
)
from PySide6.QtCore import Qt, QThread, Signal
from pathlib import Path
import cv2
import json
import random

try:
    from ...utils.annotation_processor import process_raw_detections
except ImportError:
    from utils.annotation_processor import process_raw_detections
from ..utils.class_utils import display_class_name, normalize_class_name
from ..models.auto_annotation_result import AutoAnnotationResult
from ..models.grounding_dino_model import GroundingDINOModel
class VideoProcessThread(QThread):
    """视频处理后台线程（完整后处理管线）"""
    progress = Signal(int, str)  # progress, message
    frame_processed = Signal(int, object)  # frame_idx, annotations payload
    finished = Signal(bool, str)  # success, message

    def __init__(
        self,
        video_path: str,
        output_dir: Path,
        text_prompt: str,
        sample_interval: int,
        box_threshold: float,
        text_threshold: float,
        use_gpu: bool,
        model,
        model_ctrl=None,
        nms_threshold: float = 0.5,
        export_json: bool = False,
    ):
        super().__init__()
        self.video_path = video_path
        self.output_dir = output_dir
        self.text_prompt = text_prompt
        self.sample_interval = sample_interval
        self.box_threshold = box_threshold
        self.text_threshold = text_threshold
        self.use_gpu = use_gpu
        self.model = model
        self._model_ctrl = model_ctrl
        self.nms_threshold = nms_threshold
        self.export_json = export_json
        self._is_running = True

    def stop(self):
        self._is_running = False

    def run(self):
        """执行视频处理"""
        if self._model_ctrl is None:
            self.finished.emit(False, "缺少 ModelController，无法持有推理租约")
            return
        ctrl = self._model_ctrl
        ctrl.acquire_inference_lease()
        try:
            cap = cv2.VideoCapture(self.video_path)
            if not cap.isOpened():
                self.finished.emit(False, "无法打开视频文件")
                return

            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

            classes = [
                display_class_name(c)
                for c in GroundingDINOModel._parse_prompt_class_names(self.text_prompt)
            ]
            classes = [c for c in classes if c]
            class_to_id = {
                normalize_class_name(name): i
                for i, name in enumerate(classes)
            }

            output_dir = self.output_dir
            for split in ["train", "val", "test"]:
                (output_dir / "images" / split).mkdir(parents=True, exist_ok=True)
                (output_dir / "labels" / split).mkdir(parents=True, exist_ok=True)

            self.progress.emit(0, f"开始处理视频 (共 {total_frames} 帧)")

            frame_idx = 0
            saved_count = 0
            annotations_dict = {}
            json_frames = []

            while self._is_running and cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break

                if frame_idx % self.sample_interval == 0:
                    pct = int(frame_idx / max(total_frames, 1) * 100)
                    self.progress.emit(pct, f"处理帧 {frame_idx}/{total_frames}")

                    result = self.model.infer(
                        image=frame,
                        text_prompt=self.text_prompt,
                        box_threshold=self.box_threshold,
                        text_threshold=self.text_threshold,
                    )

                    if result and result.boxes:
                        img_h, img_w = frame.shape[0], frame.shape[1]
                        processed = process_raw_detections(
                            boxes=result.boxes,
                            labels=result.labels,
                            scores=result.scores,
                            class_names=classes,
                            img_width=img_w,
                            img_height=img_h,
                            nms_threshold=self.nms_threshold,
                        )
                        auto_results = [
                            AutoAnnotationResult(class_name=n, bbox=b, score=s)
                            for n, b, s in processed
                        ]
                        annotations = self._legacy_from_processed(
                            processed, class_to_id
                        )

                        split = self._get_split(saved_count)
                        img_filename = f"frame_{frame_idx:06d}.jpg"
                        label_filename = f"frame_{frame_idx:06d}.txt"

                        img_path = output_dir / "images" / split / img_filename
                        label_path = output_dir / "labels" / split / label_filename

                        cv2.imwrite(str(img_path), frame)

                        with open(label_path, "w", encoding="utf-8") as f:
                            for ann in annotations:
                                f.write(
                                    f"{ann['class_id']} {ann['x_center']} "
                                    f"{ann['y_center']} {ann['width']} {ann['height']}\n"
                                )

                        payload = {
                            "image_path": str(img_path),
                            "label_path": str(label_path),
                            "split": split,
                            "annotations": annotations,
                            "auto_results": auto_results,
                            "image_width": img_w,
                            "image_height": img_h,
                        }
                        annotations_dict[frame_idx] = payload
                        if self.export_json:
                            json_frames.append(
                                {
                                    "frame_index": frame_idx,
                                    "image_path": str(img_path),
                                    "detections": [
                                        {
                                            "class_name": r.class_name,
                                            "score": float(r.score),
                                            "bbox_yolo": list(r.bbox),
                                        }
                                        for r in auto_results
                                    ],
                                }
                            )

                        saved_count += 1
                        self.frame_processed.emit(frame_idx, payload)

                frame_idx += 1

            cap.release()
            self._create_data_yaml(output_dir, classes)

            if self.export_json and json_frames:
                json_path = output_dir / "detections.json"
                with open(json_path, "w", encoding="utf-8") as f:
                    json.dump(
                        {
                            "video_path": self.video_path,
                            "prompt": self.text_prompt,
                            "nms_threshold": self.nms_threshold,
                            "frames": json_frames,
                        },
                        f,
                        ensure_ascii=False,
                        indent=2,
                    )

            if self._is_running:
                self.progress.emit(100, f"处理完成! 共保存 {saved_count} 帧")
                self.finished.emit(True, f"成功处理 {saved_count} 帧")
            else:
                self.finished.emit(False, "处理已取消")

        except Exception as e:
            self.finished.emit(False, f"处理错误: {str(e)}")
        finally:
            ctrl.release_inference_lease()

    @staticmethod
    def _legacy_from_processed(processed, class_to_id):
        """Convert process_raw_detections tuples → YOLO label dicts with scores."""
        annotations = []
        for class_name, bbox, score in processed:
            class_id = class_to_id.get(normalize_class_name(class_name))
            if class_id is None:
                # Fallback: accept unknown as new sequential id skip
                continue
            cx, cy, w, h = bbox
            annotations.append(
                {
                    "class_id": class_id,
                    "x_center": float(cx),
                    "y_center": float(cy),
                    "width": float(w),
                    "height": float(h),
                    "confidence": float(score),
                    "class_name": class_name,
                }
            )
        return annotations

    def _get_split(self, count):
        """自动划分数据集 (70% train, 20% val, 10% test)"""
        r = random.random()
        if r < 0.7:
            return "train"
        elif r < 0.9:
            return "val"
        else:
            return "test"

    def _create_data_yaml(self, output_dir, classes):
        """创建 YOLO data.yaml 配置文件"""
        content = f"""path: {output_dir.absolute()}
train: images/train
val: images/val
test: images/test

nc: {len(classes)}

names:
"""
        for i, name in enumerate(classes):
            content += f"  {i}: {name}\n"

        with open(output_dir / "data.yaml", "w", encoding="utf-8") as f:
            f.write(content)


class ProcessVideoDialog(QDialog):
    """视频处理对话框"""

    def __init__(self, video_path: str, model, parent=None, model_ctrl=None):
        super().__init__(parent)
        self.video_path = video_path
        self.model = model
        self._model_ctrl = model_ctrl
        self.output_dir = Path(video_path).parent / "output"
        self.process_thread = None
        self.annotations_dict = {}

        self._setup_ui()

    def _setup_ui(self):
        self.setWindowTitle("处理视频 - GroundingDINO")
        self.setMinimumWidth(500)
        self.setModal(True)

        layout = QVBoxLayout(self)

        # 视频信息
        info_group = QGroupBox("视频信息")
        info_layout = QFormLayout()
        self.video_label = QLabel(Path(self.video_path).name)
        info_layout.addRow("文件:", self.video_label)

        # 获取视频信息
        cap = cv2.VideoCapture(self.video_path)
        if cap.isOpened():
            frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            fps = cap.get(cv2.CAP_PROP_FPS)
            duration = frames / fps if fps > 0 else 0
            info_layout.addRow("总帧数:", QLabel(str(frames)))
            info_layout.addRow("帧率:", QLabel(f"{fps:.2f}"))
            info_layout.addRow("时长:", QLabel(f"{duration:.2f}s"))
            cap.release()

        info_group.setLayout(info_layout)
        layout.addWidget(info_group)

        # 参数设置
        param_group = QGroupBox("推理参数")
        param_layout = QFormLayout()

        # 类别提示
        self.class_prompt = QLineEdit("car, person, bicycle")
        self.class_prompt.setPlaceholderText("用逗号分隔的类别列表 (英文)")
        param_layout.addRow("类别提示:", self.class_prompt)

        # 采样间隔
        self.sample_interval = QSpinBox()
        self.sample_interval.setRange(1, 300)
        self.sample_interval.setValue(30)
        self.sample_interval.setSuffix(" 帧")
        param_layout.addRow("采样间隔:", self.sample_interval)

        # Box 阈值
        self.box_threshold = QDoubleSpinBox()
        self.box_threshold.setRange(0.0, 1.0)
        self.box_threshold.setValue(0.35)
        self.box_threshold.setSingleStep(0.05)
        self.box_threshold.setDecimals(2)
        param_layout.addRow("Box 阈值:", self.box_threshold)

        # Text 阈值
        self.text_threshold = QDoubleSpinBox()
        self.text_threshold.setRange(0.0, 1.0)
        self.text_threshold.setValue(0.25)
        self.text_threshold.setSingleStep(0.05)
        self.text_threshold.setDecimals(2)
        param_layout.addRow("Text 阈值:", self.text_threshold)

        # NMS（与自动标注 / Detect 共用后处理）
        self.nms_threshold = QDoubleSpinBox()
        self.nms_threshold.setRange(0.0, 1.0)
        self.nms_threshold.setValue(0.5)
        self.nms_threshold.setSingleStep(0.05)
        self.nms_threshold.setDecimals(2)
        param_layout.addRow("NMS 阈值:", self.nms_threshold)

        # GPU 加速
        self.gpu_checkbox = QCheckBox("使用 GPU 加速 (AMP)")
        self.gpu_checkbox.setChecked(True)
        param_layout.addRow("", self.gpu_checkbox)

        self.export_json_checkbox = QCheckBox("同时导出 detections.json（含置信度）")
        self.export_json_checkbox.setChecked(False)
        param_layout.addRow("", self.export_json_checkbox)

        param_group.setLayout(param_layout)
        layout.addWidget(param_group)

        # 输出目录
        output_layout = QHBoxLayout()
        output_layout.addWidget(QLabel("输出目录:"))
        self.output_label = QLabel(str(self.output_dir))
        self.output_label.setStyleSheet("color: gray;")
        output_layout.addWidget(self.output_label)
        layout.addLayout(output_layout)

        # 进度条
        self.progress_bar = QProgressBar()
        self.progress_bar.setVisible(False)
        layout.addWidget(self.progress_bar)

        self.status_label = QLabel()
        self.status_label.setVisible(False)
        layout.addWidget(self.status_label)

        # 按钮
        button_layout = QHBoxLayout()
        self.start_btn = QPushButton("开始处理")
        self.start_btn.clicked.connect(self._start_processing)
        self.cancel_btn = QPushButton("取消")
        self.cancel_btn.clicked.connect(self.reject)
        self.cancel_btn.setEnabled(False)
        button_layout.addWidget(self.start_btn)
        button_layout.addWidget(self.cancel_btn)
        layout.addLayout(button_layout)

    def _start_processing(self):
        """开始处理"""
        if not self.model.is_loaded():
            QMessageBox.warning(self, "错误", "模型未加载，请等待模型加载完成")
            return

        if self._model_ctrl is None:
            QMessageBox.warning(self, "错误", "缺少 ModelController，无法启动视频推理")
            return

        # 验证输入
        prompt = self.class_prompt.text().strip()
        if not prompt:
            QMessageBox.warning(self, "错误", "请输入类别提示")
            return

        # 更新 UI
        self.start_btn.setEnabled(False)
        self.cancel_btn.setEnabled(True)
        self.cancel_btn.setText("停止")
        self.cancel_btn.clicked.disconnect()
        self.cancel_btn.clicked.connect(self._stop_processing)
        self.progress_bar.setVisible(True)
        self.progress_bar.setValue(0)
        self.status_label.setVisible(True)

        # 创建输出目录
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # 启动处理线程
        self.process_thread = VideoProcessThread(
            video_path=self.video_path,
            output_dir=self.output_dir,
            text_prompt=prompt,
            sample_interval=self.sample_interval.value(),
            box_threshold=self.box_threshold.value(),
            text_threshold=self.text_threshold.value(),
            use_gpu=self.gpu_checkbox.isChecked(),
            model=self.model,
            model_ctrl=self._model_ctrl,
            nms_threshold=self.nms_threshold.value(),
            export_json=self.export_json_checkbox.isChecked(),
        )
        self.process_thread.progress.connect(self._on_progress)
        self.process_thread.frame_processed.connect(self._on_frame_processed)
        self.process_thread.finished.connect(self._on_finished)
        self.process_thread.start()

    def _stop_processing(self):
        """停止处理"""
        if self.process_thread and self.process_thread.isRunning():
            self.process_thread.stop()
            self.status_label.setText("正在停止...")

    def _on_progress(self, progress, message):
        """更新进度"""
        self.progress_bar.setValue(progress)
        self.status_label.setText(message)

    def _on_frame_processed(self, frame_idx, annotations):
        """帧处理完成"""
        self.annotations_dict[frame_idx] = annotations

    def _on_finished(self, success, message):
        """处理完成"""
        self.start_btn.setEnabled(True)
        self.cancel_btn.setEnabled(False)
        self.progress_bar.setVisible(False)
        self.status_label.setVisible(False)

        if success:
            QMessageBox.information(
                self, "完成",
                f"{message}\n\n输出目录: {self.output_dir}"
            )
            self.accept()
        else:
            QMessageBox.warning(self, "错误", message)
            self.reject()

    def get_results(self):
        """获取处理结果"""
        return {
            "output_dir": self.output_dir,
            "annotations": self.annotations_dict
        }
