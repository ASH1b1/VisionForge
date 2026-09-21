from __future__ import annotations

from src.gui.utils.hit_testing import edge_hit_insert, vertex_hit_index
from src.gui.utils.polygon_edit import yolo_bbox_from_polygon


def test_yolo_bbox_from_polygon_square():
    poly = [(0.2, 0.2), (0.6, 0.2), (0.6, 0.6), (0.2, 0.6)]
    cx, cy, w, h = yolo_bbox_from_polygon(poly)
    assert abs(cx - 0.4) < 1e-9
    assert abs(cy - 0.4) < 1e-9
    assert abs(w - 0.4) < 1e-9
    assert abs(h - 0.4) < 1e-9


def test_yolo_bbox_from_polygon_clips_and_has_min_size():
    poly = [(-0.2, 0.5), (1.2, 0.5), (0.5, 0.5000001)]
    cx, cy, w, h = yolo_bbox_from_polygon(poly)
    assert 0.0 <= cx <= 1.0
    assert 0.0 <= cy <= 1.0
    assert w >= 1e-6
    assert h >= 1e-6
    assert abs(w - 1.0) < 1e-9


def test_vertex_hit_index_nearest_within_radius():
    verts = [(0.0, 0.0), (10.0, 0.0), (10.0, 10.0)]
    assert vertex_hit_index(verts, (1.0, 1.0), radius_px=8) == 0
    assert vertex_hit_index(verts, (100.0, 100.0), radius_px=8) is None


def test_edge_hit_insert_on_top_edge():
    verts = [(0.0, 0.0), (10.0, 0.0), (10.0, 10.0), (0.0, 10.0)]
    hit = edge_hit_insert(verts, (5.0, 1.0), radius_px=8)
    assert hit is not None
    index, point = hit
    assert index == 0
    assert abs(point.x() - 5.0) < 1e-6
    assert abs(point.y() - 0.0) < 1e-6
