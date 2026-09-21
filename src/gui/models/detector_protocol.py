"""Detector protocol, ids, and class-name mapping helpers.

Must not import onnxruntime / torch. GUI start must succeed without YOLO extras.
"""
from __future__ import annotations

from typing import Any, Mapping, Protocol, Sequence

DETECTOR_GROUNDING_DINO = "grounding_dino"
DETECTOR_ONNX_YOLO = "onnx_yolo"
DETECTOR_IDS = (
    DETECTOR_GROUNDING_DINO,
    DETECTOR_ONNX_YOLO,
)

UNMATCHED_CREATE = "create"
UNMATCHED_SKIP = "skip"

ONNX_RUNTIME_MISSING_MESSAGE = (
    "未安装 onnxruntime，无法加载 YOLO ONNX。"
    "请在 conda 环境 visionforge 中执行：pip install onnxruntime"
    "（NVIDIA GPU 可用：pip install onnxruntime-gpu）"
)


class DetectorHandle(Protocol):
    def is_loaded(self) -> bool: ...
    def load(self) -> None: ...
    def load_sync(self) -> bool: ...
    def unload(self) -> None: ...
    def infer(self, image, **kwargs) -> Any: ...


def names_to_class_list(model_names: Mapping[int, str] | Sequence[str]) -> list[str]:
    """Indexable class-name list for process_raw_detections (label id → name)."""
    if isinstance(model_names, Mapping):
        if not model_names:
            return []
        max_id = max(int(k) for k in model_names)
        return [str(model_names.get(i, model_names.get(str(i), f"class_{i}"))) for i in range(max_id + 1)]
    return [str(name) for name in model_names]


def _canonical_project_name(model_name: str, project_class_names: Mapping[int, str]) -> str | None:
    needle = str(model_name).strip().lower()
    if not needle:
        return None
    for name in project_class_names.values():
        if str(name).strip().lower() == needle:
            return str(name)
    return None


def filter_detections_by_project_classes(
    detections: Sequence[tuple[str, tuple, float]],
    project_class_names: Mapping[int, str],
    unmatched: str = UNMATCHED_CREATE,
) -> list[tuple[str, tuple, float]]:
    """Remap detector class names onto the project; skip or keep unmatched."""
    policy = unmatched if unmatched in (UNMATCHED_CREATE, UNMATCHED_SKIP) else UNMATCHED_CREATE
    mapped: list[tuple[str, tuple, float]] = []
    for name, bbox, score in detections:
        matched = _canonical_project_name(name, project_class_names)
        if matched is not None:
            mapped.append((matched, bbox, score))
        elif policy == UNMATCHED_SKIP:
            continue
        else:
            mapped.append((str(name), bbox, score))
    return mapped


def resolve_detector_handle(
    *,
    grounding=None,
    yolo=None,
    model_ctrl=None,
):
    """Return the loaded detector without a GDINO-vs-YOLO if-chain.

    ModelController is the source of truth when present.
    """
    if model_ctrl is not None:
        getter = getattr(model_ctrl, "get_loaded_detector", None)
        if callable(getter):
            handle = getter()
            if handle is not None:
                return handle
    for handle in (yolo, grounding):
        if handle is not None and hasattr(handle, "is_loaded") and handle.is_loaded():
            return handle
    return None
