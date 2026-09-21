"""Decode Ultralytics-style YOLO detection ONNX tensors. No Qt, no onnxruntime."""
from __future__ import annotations

import ast
import json
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from ..utils.yolo_label_format import parse_data_yaml


def letterbox_rgb(
    image: np.ndarray,
    new_shape: tuple[int, int] = (640, 640),
    fill: tuple[int, int, int] = (114, 114, 114),
) -> tuple[np.ndarray, float, tuple[float, float]]:
    """返回 (padded_hwc_uint8, scale, (pad_w, pad_h))。new_shape 为 (h, w)。"""
    if image.ndim == 2:
        image = np.stack([image, image, image], axis=-1)
    elif image.ndim == 3 and image.shape[2] == 4:
        image = image[:, :, :3]
    height, width = image.shape[:2]
    out_h, out_w = int(new_shape[0]), int(new_shape[1])
    scale = min(out_h / float(height), out_w / float(width))
    unpad_w = int(round(width * scale))
    unpad_h = int(round(height * scale))
    resized = cv2.resize(image, (unpad_w, unpad_h), interpolation=cv2.INTER_LINEAR)
    pad_w = (out_w - unpad_w) / 2.0
    pad_h = (out_h - unpad_h) / 2.0
    top = int(round(pad_h - 0.1))
    bottom = int(round(pad_h + 0.1))
    left = int(round(pad_w - 0.1))
    right = int(round(pad_w + 0.1))
    padded = cv2.copyMakeBorder(
        resized, top, bottom, left, right, cv2.BORDER_CONSTANT, value=fill
    )
    return padded, float(scale), (float(pad_w), float(pad_h))


def scale_xyxy_to_original(
    xyxy: np.ndarray,
    scale: float,
    pad: tuple[float, float],
    orig_wh: tuple[int, int],
) -> np.ndarray:
    if xyxy.size == 0:
        return np.zeros((0, 4), dtype=np.float32)
    out = np.asarray(xyxy, dtype=np.float32).copy()
    pad_w, pad_h = pad
    out[:, [0, 2]] -= pad_w
    out[:, [1, 3]] -= pad_h
    denom = float(scale) if scale else 1.0
    out /= denom
    width, height = orig_wh
    out[:, [0, 2]] = np.clip(out[:, [0, 2]], 0, max(width - 1, 0))
    out[:, [1, 3]] = np.clip(out[:, [1, 3]], 0, max(height - 1, 0))
    return out


def arrange_detect_predictions(raw: np.ndarray) -> np.ndarray:
    """[1, 4+nc, N] 或 [N, 4+nc] → [N, 4+nc]。

    若形状为 [N,6] 且 32<=N<=1024（或转置同类），raise ValueError，文案要求去掉 nms=True。
    """
    arr = np.asarray(raw)
    while arr.ndim > 2 and arr.shape[0] == 1:
        arr = arr[0]
    if arr.ndim != 2:
        raise ValueError(f"YOLO ONNX 输出应为二维检测张量，实际形状 {tuple(np.asarray(raw).shape)}")
    rows, cols = int(arr.shape[0]), int(arr.shape[1])

    def _looks_like_end2end(num: int, feat: int) -> bool:
        return feat == 6 and 32 <= num <= 1024

    if _looks_like_end2end(rows, cols) or _looks_like_end2end(cols, rows):
        raise ValueError(
            "该 ONNX 像是带图内 NMS 的导出。请去掉 nms=True，在训练环境使用默认命令："
            " yolo export model=你的.pt format=onnx"
        )
    # Last dim should be 4+nc (>=5). [4+nc, N] including N=1 needs a transpose.
    if cols < 5 or (rows >= 5 and rows < cols):
        arr = arr.T
    return np.ascontiguousarray(arr)


def decode_ultralytics_detect(
    raw: np.ndarray,
    *,
    conf_threshold: float,
    scale: float,
    pad: tuple[float, float],
    orig_wh: tuple[int, int],
) -> tuple[list[tuple[float, float, float, float]], list[int], list[float]]:
    """xywh（letterbox 像素）→ 原图像素 xyxy；class score argmax；过滤 conf。"""
    pred = arrange_detect_predictions(raw)
    if pred.shape[1] < 5:
        raise ValueError(f"YOLO ONNX 检测输出通道不足: {pred.shape}")
    xywh = pred[:, :4].astype(np.float32, copy=False)
    class_scores = pred[:, 4:].astype(np.float32, copy=False)
    class_ids = np.argmax(class_scores, axis=1)
    conf = class_scores[np.arange(class_scores.shape[0]), class_ids]
    keep = conf >= float(conf_threshold)
    xywh = xywh[keep]
    class_ids = class_ids[keep]
    conf = conf[keep]
    xyxy = np.empty_like(xywh)
    xyxy[:, 0] = xywh[:, 0] - xywh[:, 2] / 2.0
    xyxy[:, 1] = xywh[:, 1] - xywh[:, 3] / 2.0
    xyxy[:, 2] = xywh[:, 0] + xywh[:, 2] / 2.0
    xyxy[:, 3] = xywh[:, 1] + xywh[:, 3] / 2.0
    xyxy = scale_xyxy_to_original(xyxy, scale, pad, orig_wh)
    boxes = [tuple(map(float, row)) for row in xyxy]
    labels = [int(v) for v in class_ids]
    scores = [float(v) for v in conf]
    return boxes, labels, scores


def parse_names_metadata(raw: object) -> dict[int, str]:
    if raw is None:
        return {}
    if isinstance(raw, dict):
        parsed: dict[int, str] = {}
        for key, value in raw.items():
            try:
                parsed[int(key)] = str(value)
            except (TypeError, ValueError):
                continue
        return parsed
    text = str(raw).strip()
    if not text:
        return {}
    data: Any = None
    try:
        data = json.loads(text)
    except (json.JSONDecodeError, TypeError):
        try:
            data = ast.literal_eval(text)
        except (SyntaxError, ValueError):
            return {}
    if not isinstance(data, dict):
        return {}
    return parse_names_metadata(data)


def class_names_from_onnx_path(onnx_path: str | Path) -> dict[int, str]:
    path = Path(onnx_path)
    for candidate in (path.parent / "data.yaml", path.with_suffix(".yaml")):
        if not candidate.is_file():
            continue
        names = dict(parse_data_yaml(candidate).names)
        if names:
            return names
    return {}


def fallback_class_names(num_classes: int) -> dict[int, str]:
    count = max(0, int(num_classes))
    return {i: f"class_{i}" for i in range(count)}
