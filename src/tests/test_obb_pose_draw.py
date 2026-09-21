from src.gui.utils.annotation_draw import (
    keypoint_draw_items,
    obb_corners,
    resolved_kind,
    should_draw_aabb,
)


class _Ann:
    def __init__(self, **kwargs):
        self.__dict__.update(kwargs)


def test_obb_does_not_draw_aabb():
    ann = _Ann(kind="obb", bbox=(0.5, 0.5, 0.8, 0.8), obb=((0.1, 0.1), (0.9, 0.1), (0.9, 0.9), (0.1, 0.9)))
    assert resolved_kind(ann) == "obb"
    assert should_draw_aabb("obb") is False
    assert obb_corners(ann)[2] == (0.9, 0.9)


def test_pose_draws_aabb_and_skips_v0():
    ann = _Ann(
        kind="pose",
        bbox=(0.5, 0.5, 0.2, 0.2),
        keypoints=[(0.1, 0.1, 2), (0.2, 0.2, 1), (0.3, 0.3, 0)],
    )
    assert should_draw_aabb(resolved_kind(ann)) is True
    items = keypoint_draw_items(ann.keypoints)
    assert items == [(0.1, 0.1, "filled"), (0.2, 0.2, "hollow")]


def test_bbox_and_polygon_kinds_unchanged():
    assert resolved_kind(_Ann(kind="bbox", polygon=None)) == "bbox"
    assert resolved_kind(_Ann(polygon=[(0, 0), (1, 0), (1, 1)])) == "polygon"
    assert should_draw_aabb("polygon") is True
    assert should_draw_aabb("bbox") is True
