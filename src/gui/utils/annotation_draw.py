"""Pure helpers for drawing pose keypoints and OBB quads. No Qt."""
from __future__ import annotations

from typing import Any, Iterable


def resolved_kind(ann: Any) -> str:
    if isinstance(ann, dict):
        kind = ann.get("kind")
        polygon = ann.get("polygon")
        obb = ann.get("obb")
        keypoints = ann.get("keypoints")
    else:
        kind = getattr(ann, "kind", None)
        polygon = getattr(ann, "polygon", None)
        obb = getattr(ann, "obb", None)
        keypoints = getattr(ann, "keypoints", None)
    if kind in ("bbox", "polygon", "obb", "pose"):
        return kind
    if obb:
        return "obb"
    if keypoints:
        return "pose"
    if polygon and len(polygon) >= 3:
        return "polygon"
    return "bbox"


def should_draw_aabb(kind: str) -> bool:
    return kind != "obb"


def obb_corners(ann: Any) -> list[tuple[float, float]] | None:
    if isinstance(ann, dict):
        obb = ann.get("obb")
    else:
        obb = getattr(ann, "obb", None)
    if not obb or len(obb) != 4:
        return None
    return [(float(p[0]), float(p[1])) for p in obb]


def keypoint_draw_items(
    keypoints: Iterable | None,
) -> list[tuple[float, float, str]]:
    """Return (x, y, style) for visible keypoints. style is filled or hollow.

    v=0 skipped, v=1 hollow, v=2 filled.
    """
    items: list[tuple[float, float, str]] = []
    if not keypoints:
        return items
    for point in keypoints:
        if point is None or len(point) < 2:
            continue
        x, y = float(point[0]), float(point[1])
        v = int(point[2]) if len(point) >= 3 else 2
        if v == 0:
            continue
        style = "hollow" if v == 1 else "filled"
        items.append((x, y, style))
    return items


def keypoint_radius_px(zoom: float) -> float:
    zoom = max(float(zoom), 0.01)
    return max(3.0, min(8.0, 6.0 * zoom / max(zoom, 1.0)))
