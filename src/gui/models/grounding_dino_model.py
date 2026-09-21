"""GroundingDINO 模型包装 - 支持线程安全推理"""
from __future__ import annotations

import os
import sys as _sys

# 模型目录：备份路径（HuggingFace 缓存格式）
# PyInstaller 兼容：打包后资源在 sys._MEIPASS，源码运行用 __file__ 相对路径
if getattr(_sys, "frozen", False):
    _BACKUP_MODELS_ROOT = os.path.join(_sys._MEIPASS, "models", "grounding_dino")
else:
    _BACKUP_MODELS_ROOT = os.path.join(
        os.path.dirname(__file__), "..", "..", "..", "models", "grounding_dino"
    )
os.environ["HF_HOME"] = os.path.abspath(_BACKUP_MODELS_ROOT)

# 模型快照路径（直接路径加载，避免 HF Hub 缓存解析问题）
_SNAPSHOT_HASH = "12bdfa3120f3e7ec7b434d90674b3396eccf88eb"
_SNAPSHOT_PATH = os.path.join(
    _BACKUP_MODELS_ROOT, "models--IDEA-Research--grounding-dino-base",
    "snapshots", _SNAPSHOT_HASH
)

import logging
import threading
import traceback
from typing import List, Tuple, Optional, Union

import numpy as np
import cv2
from PIL import Image as PILImage
from PySide6.QtCore import QObject, Signal
from dataclasses import dataclass

# torch 和 transformers 延迟导入（加速启动）
_torch = None
_AutoProcessor = None
_AutoModelForZeroShotObjectDetection = None


def _lazy_import_torch():
    global _torch
    if _torch is None:
        import torch as _t
        _torch = _t
    return _torch


def _lazy_import_transformers():
    global _AutoProcessor, _AutoModelForZeroShotObjectDetection
    if _AutoProcessor is None:
        from transformers import AutoProcessor as AP, AutoModelForZeroShotObjectDetection as AM
        _AutoProcessor = AP
        _AutoModelForZeroShotObjectDetection = AM
    return _AutoProcessor, _AutoModelForZeroShotObjectDetection

# 设置日志
logger = logging.getLogger(__name__)

# 默认阈值常量
DEFAULT_BOX_THRESHOLD = 0.35
DEFAULT_TEXT_THRESHOLD = 0.25


@dataclass
class DetectionResult:
    """检测结果数据类"""
    boxes: List[Tuple[float, float, float, float]]  # [x1, y1, x2, y2]
    labels: List[int]
    scores: List[float]
    image_size: Tuple[int, int]  # (width, height)


class GroundingDINOModel(QObject):
    """GroundingDINO 模型包装 - 线程安全"""

    supports_detection = True
    supports_segmentation = False

    # Signals
    loading_progress = Signal(str)  # status message
    loading_finished = Signal(bool)  # success
    inference_completed = Signal(object)  # DetectionResult

    def __init__(self, model_path: str):
        super().__init__()
        self.model_path = model_path
        self.processor: Optional[AutoProcessor] = None
        self.model: Optional[AutoModelForZeroShotObjectDetection] = None
        self.device: Optional[torch.device] = None
        self._amp_dtype = None  # torch.dtype when CUDA half/bfloat16 load succeeded
        self._post_process_uses_box_threshold: Optional[bool] = None

        # 线程安全锁
        self._lock = threading.RLock()
        self._loaded = False
        # Invalidate in-flight async loads on unload / newer load()
        self._load_id = 0

    def load(self) -> None:
        """加载模型（后台线程）"""
        with self._lock:
            self._load_id += 1
            load_id = self._load_id

        def _load() -> None:
            processor = None
            model = None
            try:
                # 延迟导入 torch + transformers (加速启动)
                torch = _lazy_import_torch()

                self.loading_progress.emit("检测设备...")
                # Device detection outside lock (slow operation)
                device = torch.device(
                    "cuda" if torch.cuda.is_available() else "cpu"
                )

                with self._lock:
                    if load_id != self._load_id:
                        return
                    self.device = device

                self.loading_progress.emit(f"使用设备: {self.device}")

                # 延迟导入 transformers
                AutoProcessor, AutoModelForZeroShotObjectDetection = _lazy_import_transformers()

                # 直接从快照路径加载（绕过 HF Hub 缓存解析）
                # _SNAPSHOT_PATH 已包含完整的模型文件路径
                model_load_path = _SNAPSHOT_PATH if os.path.exists(_SNAPSHOT_PATH) else self.model_path

                self.loading_progress.emit("加载处理器...")
                processor = AutoProcessor.from_pretrained(model_load_path)

                with self._lock:
                    if load_id != self._load_id:
                        processor = None
                        return

                self.loading_progress.emit("加载模型...")
                model = None
                amp_dtype = None
                if device.type == "cuda":
                    # Prefer float16 AMP; fall back to bfloat16, then float32
                    for candidate in (torch.float16, torch.bfloat16):
                        try:
                            model = AutoModelForZeroShotObjectDetection.from_pretrained(
                                model_load_path, torch_dtype=candidate
                            ).to(device)
                            amp_dtype = candidate
                            break
                        except Exception as dtype_err:
                            logger.warning(
                                "Failed to load GroundingDINO with %s: %s",
                                candidate,
                                dtype_err,
                            )
                            model = None
                if model is None:
                    model = AutoModelForZeroShotObjectDetection.from_pretrained(
                        model_load_path
                    ).to(device)

                model.eval()

                # Final state update under lock; discard if unload/newer load invalidated us
                discarded = False
                with self._lock:
                    if load_id != self._load_id:
                        discarded = True
                    else:
                        self.processor = processor
                        self.model = model
                        self._amp_dtype = amp_dtype
                        self._post_process_uses_box_threshold = None
                        self._loaded = True
                        processor = None
                        model = None

                if discarded:
                    if model is not None:
                        del model
                    if processor is not None:
                        del processor
                    try:
                        if torch.cuda.is_available():
                            torch.cuda.empty_cache()
                    except Exception:
                        pass
                    return

                self.loading_finished.emit(True)
                self.loading_progress.emit("模型加载完成")
                logger.info("GroundingDINO model loaded successfully")
                try:
                    self.warmup()
                except Exception as warm_err:
                    logger.warning("GroundingDINO warmup skipped: %s", warm_err)

            except Exception as e:
                with self._lock:
                    cancelled = load_id != self._load_id
                if cancelled:
                    return

                logger.error(f"Model loading failed: {e}\n{traceback.format_exc()}")

                # Clean up partial resources
                try:
                    torch = _lazy_import_torch()
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                except Exception:
                    pass

                with self._lock:
                    if load_id != self._load_id:
                        return
                    self._loaded = False
                    self.model = None
                    self.processor = None

                self.loading_finished.emit(False)
                self.loading_progress.emit(f"加载失败: {str(e)}")

        thread = threading.Thread(target=_load, daemon=True)
        thread.start()

    def is_loaded(self) -> bool:
        """检查模型是否已加载（线程安全）"""
        with self._lock:
            return self._loaded and self.model is not None and self.processor is not None

    def warmup(self) -> None:
        """Run a tiny infer to amortize first-call CUDA/kernel setup. Failures are logged only."""
        from PIL import Image as PILImage

        if not self.is_loaded():
            return
        try:
            img = PILImage.new("RGB", (64, 64), color=(0, 0, 0))
            self.infer(
                image=img,
                text_prompt="object",
                box_threshold=0.5,
                text_threshold=0.5,
                use_amp=True,
            )
            logger.info("GroundingDINO warmup completed")
        except Exception as exc:
            logger.warning("GroundingDINO warmup failed: %s", exc)

    def _validate_boxes(
        self,
        boxes: List[Tuple[float, float, float, float]],
        image_size: Tuple[int, int]
    ) -> bool:
        """
        验证模型返回的框格式是否合理

        Args:
            boxes: 检测框列表，格式 [x1, y1, x2, y2] 或 [y1, x1, y2, x2]
            image_size: 图像尺寸 (width, height)

        Returns:
            True: 格式合理
            False: 格式异常，应该拒绝
        """
        if not boxes:
            logger.debug("No boxes to validate")
            return True

        img_w, img_h = image_size
        validated_count = 0

        # 检查前5个框的坐标范围
        # 允许 1 像素的浮点精度误差（模型后处理可能产生轻微越界）
        _EPSILON = 1.5  # 像素容差
        
        for i, box in enumerate(boxes[:5]):
            if len(box) != 4:
                logger.error(f"Box {i} has invalid length: {len(box)}, expected 4")
                return False

            x1, y1, x2, y2 = box

            # 检查是否为归一化坐标（所有值在[0,1]）
            if all(0 <= v <= 1.0 for v in box):
                logger.debug(f"Box {i}: normalized coordinates detected")
                validated_count += 1
                continue

            # 检查像素坐标格式（允许浮点精度误差）
            # 情况1: [x1, y1, x2, y2] 格式（期望格式）
            is_xyxy = (-_EPSILON <= x1 <= img_w + _EPSILON and 
                       -_EPSILON <= x2 <= img_w + _EPSILON and
                       -_EPSILON <= y1 <= img_h + _EPSILON and 
                       -_EPSILON <= y2 <= img_h + _EPSILON)

            # 情况2: [y1, x1, y2, x2] 格式（轴反转，异常）
            is_yxyx = (-_EPSILON <= x1 <= img_h + _EPSILON and 
                       -_EPSILON <= x2 <= img_h + _EPSILON and
                       -_EPSILON <= y1 <= img_w + _EPSILON and 
                       -_EPSILON <= y2 <= img_w + _EPSILON)

            if not is_xyxy and not is_yxyx:
                logger.error(
                    f"Box {i} has invalid coordinates: "
                    f"box={box}, img_size=({img_w}, {img_h})"
                )
                return False

            # 检测轴反转警告
            if is_yxyx and not is_xyxy:
                logger.warning(
                    f"Possible axis swap detected in box {i}: "
                    f"box=[{x1}, {y1}, {x2}, {y2}] looks like [y1, x1, y2, x2] "
                    f"instead of [x1, y1, x2, y2]. "
                    f"This may indicate target_size mismatch!"
                )

            validated_count += 1

        logger.debug(f"Validated {validated_count}/{min(5, len(boxes))} boxes")
        return True

    @staticmethod
    def _parse_prompt_class_names(text_prompt: str) -> list[str]:
        import re
        parts = re.split(r"[,;，；]+", text_prompt or "")
        return [p.strip() for p in parts if p.strip()]

    @staticmethod
    def _map_string_labels(boxes, labels, scores, class_names):
        class_mapping = {name.strip().lower(): idx for idx, name in enumerate(class_names)}
        out_boxes = []
        out_labels = []
        out_scores = []
        for box, label, score in zip(boxes, labels, scores):
            label_lower = label.lower().strip()
            if label_lower in class_mapping:
                out_boxes.append(box)
                out_labels.append(class_mapping[label_lower])
                out_scores.append(score)
            else:
                matched = False
                for class_name, class_idx in class_mapping.items():
                    if class_name in label_lower or label_lower in class_name:
                        out_boxes.append(box)
                        out_labels.append(class_idx)
                        out_scores.append(score)
                        matched = True
                        break
                if not matched:
                    continue
        return out_boxes, out_labels, out_scores

    def infer(
        self,
        image: Union[str, np.ndarray, PILImage.Image],
        text_prompt: str,
        box_threshold: float = DEFAULT_BOX_THRESHOLD,
        text_threshold: float = DEFAULT_TEXT_THRESHOLD,
        use_amp: bool = True,
    ) -> Optional[DetectionResult]:
        """
        执行推理（线程安全）

        Args:
            image: 输入图像（路径、numpy 数组或 PIL Image）
            text_prompt: 文本提示词（逗号分隔的类别）
            box_threshold: 框置信度阈值
            text_threshold: 文本置信度阈值
            use_amp: 在 CUDA 上启用 autocast（需模型已以半精度加载）

        Returns:
            DetectionResult 对象，失败时返回 None
        """
        with self._lock:
            if not self._loaded or self.model is None or self.processor is None:
                logger.warning("Model not loaded, inference skipped")
                return None

            device = self.device
            model = self.model
            processor = self.processor
            amp_dtype = self._amp_dtype
            uses_box_threshold = self._post_process_uses_box_threshold

        try:
            # 转换图像为 PIL Image
            pil_image = self._ensure_pil_image(image)
            if pil_image is None:
                logger.error("Failed to convert image to PIL format")
                return None

            # ============================================================
            # 修复后的推理与后处理流程（极简版，先跑通）
            # ============================================================

            torch = _lazy_import_torch()
            from typing import Tuple, List
            import inspect

            # ----------------------------------------------------------------
            # Step 1: 基础参数准备（保持不变）
            # ----------------------------------------------------------------
            image_width: int = pil_image.width
            image_height: int = pil_image.height
            original_size: Tuple[int, int] = (image_width, image_height)
            target_sizes: List[List[int]] = [[image_height, image_width]]

            # ----------------------------------------------------------------
            # Step 2: Prompt 解析与格式化
            # ----------------------------------------------------------------
            # 规则：
            # 1. 从 text_prompt 解析类别列表（逗号/分号分隔，保留多词类名）
            # 2. 每个类别末尾加「空格 + 句号」
            class_names_list = self._parse_prompt_class_names(text_prompt)

            # 格式化："person" -> "person ."
            formatted_prompt = " , ".join([f"{name.strip()} ." for name in class_names_list if name.strip()])

            # ----------------------------------------------------------------
            # Step 3: Processor 调用
            # ----------------------------------------------------------------
            # 只传 images 和 text，其他预处理参数用默认值
            inputs = processor(
                images=pil_image,
                text=formatted_prompt,  # 用格式化后的 Prompt
                return_tensors="pt"
            )
            if device is not None:
                inputs = inputs.to(device, non_blocking=(getattr(device, "type", None) == "cuda"))

            # ----------------------------------------------------------------
            # Step 4: 模型推理（可选 CUDA AMP）
            # ----------------------------------------------------------------
            # Standard AMP: autocast works with fp32 weights too; half-loaded
            # weights just set a preferred dtype when available.
            enable_amp = bool(use_amp and device is not None and device.type == "cuda")
            autocast_dtype = amp_dtype if amp_dtype is not None else torch.float16
            with torch.no_grad():
                if enable_amp:
                    with torch.autocast(device_type="cuda", dtype=autocast_dtype):
                        outputs = model(**inputs)
                else:
                    outputs = model(**inputs)

            # ----------------------------------------------------------------
            # Step 5: 后处理（兼容版本；signature 只解析一次）
            # ----------------------------------------------------------------
            if uses_box_threshold is None:
                sig_params = inspect.signature(
                    processor.post_process_grounded_object_detection
                ).parameters
                uses_box_threshold = "box_threshold" in sig_params
                with self._lock:
                    self._post_process_uses_box_threshold = uses_box_threshold

            if uses_box_threshold:
                # transformers 4.40.0
                results = processor.post_process_grounded_object_detection(
                    outputs,
                    inputs.input_ids,
                    box_threshold=box_threshold,
                    text_threshold=text_threshold,
                    target_sizes=target_sizes
                )[0]
            else:
                # transformers 5.0.0+
                results = processor.post_process_grounded_object_detection(
                    outputs,
                    inputs.input_ids,
                    threshold=box_threshold,
                    text_threshold=text_threshold,
                    target_sizes=target_sizes
                )[0]

            # ----------------------------------------------------------------
            # Step 6: 结果提取（保持不变）
            # ----------------------------------------------------------------
            if isinstance(results["boxes"], torch.Tensor):
                boxes = results["boxes"].detach().cpu().tolist()
                raw_labels = results["labels"]
                labels = raw_labels.detach().cpu().tolist() if isinstance(raw_labels, torch.Tensor) else raw_labels
                scores = results["scores"].detach().cpu().tolist() if isinstance(results["scores"], torch.Tensor) else results["scores"]
            else:
                boxes = results["boxes"]
                labels = results["labels"]
                scores = results["scores"]

            # ----------------------------------------------------------------
            # Step 7: 标签映射
            # ----------------------------------------------------------------
            if labels and isinstance(labels[0], str):
                boxes, labels, scores = self._map_string_labels(
                    boxes, labels, scores, class_names_list
                )

            # ----------------------------------------------------------------
            # Step 8: 框验证
            # ----------------------------------------------------------------
            # 先做简单的边界检查，保留有效框
            valid_boxes = []
            valid_labels = []
            valid_scores = []
            for box, label, score in zip(boxes, labels, scores):
                x1, y1, x2, y2 = box
                # 允许轻微超出边界（容差1.5像素）
                if (x1 >= -1.5 and y1 >= -1.5 and
                    x2 <= image_width + 1.5 and y2 <= image_height + 1.5 and
                    x2 > x1 and y2 > y1):
                    valid_boxes.append(box)
                    valid_labels.append(label)
                    valid_scores.append(score)

            boxes, labels, scores = valid_boxes, valid_labels, valid_scores

            # ----------------------------------------------------------------
            # Step 9: 返回结果
            # ----------------------------------------------------------------
            return DetectionResult(
                boxes=boxes,
                labels=labels,
                scores=scores,
                image_size=original_size
            )

        except Exception as e:
            logger.error(f"Inference error: {e}\n{traceback.format_exc()}")
            return None

    def _ensure_pil_image(
        self,
        image: Union[str, np.ndarray, PILImage.Image]
    ) -> Optional[PILImage.Image]:
        """确保输入是 PIL Image 格式"""
        if isinstance(image, str):
            # 从文件路径加载
            try:
                return PILImage.open(image).convert("RGB")
            except Exception as e:
                logger.error(f"Failed to load image from path: {e}")
                return None

        elif isinstance(image, np.ndarray):
            # 从 numpy 数组转换
            try:
                if image.ndim == 2:  # 灰度图
                    image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
                elif image.ndim == 3 and image.shape[2] == 4:  # RGBA
                    image = cv2.cvtColor(image, cv2.COLOR_RGBA2RGB)
                elif image.ndim == 3 and image.shape[2] == 1:  # 灰度（3通道）
                    image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
                return PILImage.fromarray(image).convert("RGB")
            except Exception as e:
                logger.error(f"Failed to convert numpy array: {e}")
                return None

        elif isinstance(image, PILImage.Image):
            # 已经是 PIL Image
            return image.convert("RGB")

        else:
            logger.error(f"Unsupported image type: {type(image)}")
            return None

    def unload(self) -> None:
        """卸载模型释放内存"""
        with self._lock:
            # Invalidate any in-flight async load so it will not re-assign model
            self._load_id += 1
            if self.model is not None:
                del self.model
                self.model = None
            if self.processor is not None:
                del self.processor
                self.processor = None
            self._loaded = False
            self._amp_dtype = None
            self._post_process_uses_box_threshold = None

            # 清理 GPU 缓存
            if self.device and self.device.type == "cuda":
                try:
                    torch = _lazy_import_torch()
                    torch.cuda.empty_cache()
                except Exception:
                    pass

        logger.info("Model unloaded")
