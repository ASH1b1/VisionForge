from __future__ import annotations

from src.gui.utils.replace_frame_warning import (
    POLYGON_REPLACE_NOTE,
    frames_have_polygons,
    replace_warning_message,
)


class _Ann:
    def __init__(self, polygon=None, kind=None):
        self.polygon = polygon
        self.kind = kind


def test_frames_have_polygons_true_when_any_poly():
    assert frames_have_polygons([[_Ann()], [_Ann(polygon=[(0, 0), (1, 0), (1, 1)])]])
    assert not frames_have_polygons([[_Ann()], [_Ann(polygon=[(0, 0), (1, 0)])]])


def test_frames_have_polygons_true_when_obb_or_pose():
    assert frames_have_polygons([[_Ann(kind="obb")]])
    assert frames_have_polygons([[_Ann(kind="pose")]])
    assert not frames_have_polygons([[_Ann(kind="bbox")]])


def test_replace_warning_appends_note_only_when_polygons():
    base = "将用 3 个 Detect 结果替换当前帧标注。"
    assert replace_warning_message(base, has_polygons=False) == base
    text = replace_warning_message(base, has_polygons=True)
    assert POLYGON_REPLACE_NOTE in text
    assert base in text
