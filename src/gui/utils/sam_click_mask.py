"""Pick a SAM mask that contains a click, preferring the smallest area."""
from __future__ import annotations

from typing import Optional

import numpy as np


def _as_n_hw(masks) -> np.ndarray:
    arr = np.asarray(masks)
    if arr.size == 0:
        return np.zeros((0, 0, 0), dtype=np.float32)
    if arr.ndim == 2:
        return arr[None, ...]
    if arr.ndim == 4:
        arr = arr[:, 0]
    if arr.ndim != 3:
        raise ValueError(f"unsupported mask ndim: {arr.ndim}")
    return arr.astype(np.float32, copy=False)


def select_mask_containing_click(
    masks,
    click_xy: tuple[float, float],
    *,
    threshold: float = 0.5,
) -> tuple[Optional[np.ndarray], Optional[int]]:
    batch = _as_n_hw(masks)
    if batch.shape[0] == 0 or batch.shape[1] == 0 or batch.shape[2] == 0:
        return None, None
    x, y = float(click_xy[0]), float(click_xy[1])
    col = int(round(x))
    row = int(round(y))
    height, width = int(batch.shape[1]), int(batch.shape[2])
    if col < 0 or row < 0 or col >= width or row >= height:
        return None, None
    hits: list[tuple[int, int]] = []
    for index, mask in enumerate(batch):
        if float(mask[row, col]) > float(threshold):
            area = int((mask > float(threshold)).sum())
            hits.append((area, index))
    if not hits:
        return None, None
    hits.sort(key=lambda item: (item[0], item[1]))
    best = hits[0][1]
    return batch[best], best
