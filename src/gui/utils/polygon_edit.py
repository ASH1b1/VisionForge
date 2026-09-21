"""Polygon geometry helpers for vertex editing (normalized YOLO space)."""
from __future__ import annotations


def yolo_bbox_from_polygon(
    polygon: list[tuple[float, float]],
) -> tuple[float, float, float, float]:
    """Normalized polygon vertices → YOLO (cx, cy, w, h), clipped to [0, 1]."""
    if not polygon:
        return (0.0, 0.0, 1e-6, 1e-6)

    xs = [max(0.0, min(1.0, float(p[0]))) for p in polygon]
    ys = [max(0.0, min(1.0, float(p[1]))) for p in polygon]
    x1, x2 = min(xs), max(xs)
    y1, y2 = min(ys), max(ys)
    w = max(x2 - x1, 1e-6)
    h = max(y2 - y1, 1e-6)
    cx = (x1 + x2) / 2.0
    cy = (y1 + y2) / 2.0
    cx = max(w / 2.0, min(1.0 - w / 2.0, cx))
    cy = max(h / 2.0, min(1.0 - h / 2.0, cy))
    return (cx, cy, w, h)
