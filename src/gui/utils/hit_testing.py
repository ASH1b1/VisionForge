"""Canvas-space hit testing for polygon vertices and edges."""
from __future__ import annotations

from typing import Sequence

from PySide6.QtCore import QPointF

HIT_RADIUS_PX = 8.0


def _xy(point) -> tuple[float, float]:
    if isinstance(point, QPointF):
        return float(point.x()), float(point.y())
    return float(point[0]), float(point[1])


def vertex_hit_index(
    vertices_canvas: Sequence,
    pos,
    radius_px: float = HIT_RADIUS_PX,
) -> int | None:
    """Return the nearest vertex index within radius_px, else None."""
    if not vertices_canvas:
        return None
    px, py = _xy(pos)
    best_i: int | None = None
    best_d2 = float(radius_px) * float(radius_px)
    for i, vertex in enumerate(vertices_canvas):
        vx, vy = _xy(vertex)
        d2 = (vx - px) ** 2 + (vy - py) ** 2
        if d2 <= best_d2:
            best_d2 = d2
            best_i = i
    return best_i


def _closest_point_on_segment(
    px: float, py: float, ax: float, ay: float, bx: float, by: float
) -> tuple[float, float, float]:
    abx, aby = bx - ax, by - ay
    length2 = abx * abx + aby * aby
    if length2 <= 1e-12:
        dx, dy = px - ax, py - ay
        return ax, ay, dx * dx + dy * dy
    t = ((px - ax) * abx + (py - ay) * aby) / length2
    t = max(0.0, min(1.0, t))
    qx, qy = ax + t * abx, ay + t * aby
    dx, dy = px - qx, py - qy
    return qx, qy, dx * dx + dy * dy


def edge_hit_insert(
    vertices_canvas: Sequence,
    pos,
    radius_px: float = HIT_RADIUS_PX,
) -> tuple[int, QPointF] | None:
    """If pos is within radius of an edge, return (i, point) to insert after vertex i."""
    n = len(vertices_canvas)
    if n < 2:
        return None
    px, py = _xy(pos)
    best_i: int | None = None
    best_pt: QPointF | None = None
    best_d2 = float(radius_px) * float(radius_px)
    for i in range(n):
        ax, ay = _xy(vertices_canvas[i])
        bx, by = _xy(vertices_canvas[(i + 1) % n])
        qx, qy, d2 = _closest_point_on_segment(px, py, ax, ay, bx, by)
        if d2 <= best_d2:
            best_d2 = d2
            best_i = i
            best_pt = QPointF(qx, qy)
    if best_i is None or best_pt is None:
        return None
    return best_i, best_pt


def smallest_bbox_hit(annotations, x_norm: float, y_norm: float) -> int | None:
    """Hit visible YOLO bboxes; return the id with the smallest area, else None."""
    best_id: int | None = None
    best_area: float | None = None
    for ann in annotations:
        if not getattr(ann, "visible", True):
            continue
        bbox = getattr(ann, "bbox", None)
        if not bbox or len(bbox) != 4:
            continue
        cx, cy, w, h = (float(v) for v in bbox)
        x1, y1 = cx - w / 2.0, cy - h / 2.0
        x2, y2 = cx + w / 2.0, cy + h / 2.0
        if x1 <= x_norm <= x2 and y1 <= y_norm <= y2:
            area = abs(w * h)
            if best_area is None or area < best_area:
                best_area = area
                best_id = getattr(ann, "id", None)
    return best_id
