from __future__ import annotations

from src.gui.utils.hit_testing import edge_hit_insert, smallest_bbox_hit


class _Ann:
    def __init__(self, id, bbox, visible=True):
        self.id = id
        self.bbox = bbox
        self.visible = visible


def test_smallest_area_nested_boxes():
    big = _Ann(1, (0.5, 0.5, 0.8, 0.8))
    small = _Ann(2, (0.5, 0.5, 0.2, 0.2))
    assert smallest_bbox_hit([big, small], 0.5, 0.5) == 2


def test_smallest_bbox_hit_miss_and_hidden():
    box = _Ann(1, (0.5, 0.5, 0.2, 0.2))
    hidden = _Ann(2, (0.5, 0.5, 0.1, 0.1), visible=False)
    assert smallest_bbox_hit([box], 0.01, 0.01) is None
    assert smallest_bbox_hit([box, hidden], 0.5, 0.5) == 1


def test_edge_hit_insert_prefers_later_closer_edge():
    verts = [(0.0, 0.0), (10.0, 0.0), (10.0, 10.0), (0.0, 10.0)]
    hit = edge_hit_insert(verts, (9.0, 5.0), radius_px=8)
    assert hit is not None
    index, point = hit
    assert index == 1
    assert abs(point.x() - 10.0) < 1e-6
    assert abs(point.y() - 5.0) < 1e-6
