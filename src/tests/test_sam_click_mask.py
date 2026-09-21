from __future__ import annotations

import numpy as np

from src.gui.utils.sam_click_mask import select_mask_containing_click


def test_picks_smallest_mask_that_contains_click():
    large = np.zeros((8, 8), dtype=np.float32)
    large[1:7, 1:7] = 1
    small = np.zeros((8, 8), dtype=np.float32)
    small[3:5, 3:5] = 1
    other = np.zeros((8, 8), dtype=np.float32)
    other[0:2, 6:8] = 1
    masks = np.stack([large, small, other])
    chosen, index = select_mask_containing_click(masks, (4.0, 4.0))
    assert index == 1
    assert chosen is not None
    assert int(chosen.sum()) == int(small.sum())


def test_different_clicks_select_different_masks():
    left = np.zeros((8, 8), dtype=np.float32)
    left[2:6, 0:4] = 1
    right = np.zeros((8, 8), dtype=np.float32)
    right[2:6, 4:8] = 1
    masks = np.stack([left, right])
    _, i_left = select_mask_containing_click(masks, (1.0, 4.0))
    _, i_right = select_mask_containing_click(masks, (6.0, 4.0))
    assert i_left == 0
    assert i_right == 1


def test_miss_returns_none_when_click_not_covered():
    blob = np.zeros((8, 8), dtype=np.float32)
    blob[0:2, 0:2] = 1
    chosen, index = select_mask_containing_click(np.stack([blob]), (7.0, 7.0))
    assert chosen is None
    assert index is None


def test_accepts_nchw_and_single_hw_masks():
    hw = np.zeros((4, 4), dtype=np.float32)
    hw[1:3, 1:3] = 1.0
    chosen, index = select_mask_containing_click(hw, (1.0, 1.0))
    assert index == 0
    assert chosen is not None

    n1hw = hw[None, None, ...]
    chosen2, index2 = select_mask_containing_click(n1hw, (1.0, 1.0))
    assert index2 == 0
    assert chosen2 is not None
