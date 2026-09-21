"""Warnings when a whole-frame replace would wipe refined polygons or pose/obb."""
from __future__ import annotations

POLYGON_REPLACE_NOTE = (
    "将清空本帧已有标注（包括多边形、旋转框或关键点），"
    "再写入自动标注结果。无法自动只补漏框。是否继续？"
)


def annotation_has_polygon(ann) -> bool:
    if isinstance(ann, dict):
        poly = ann.get("polygon")
    else:
        poly = getattr(ann, "polygon", None)
    return bool(poly) and len(poly) >= 3


def annotation_is_protected(ann) -> bool:
    if annotation_has_polygon(ann):
        return True
    if isinstance(ann, dict):
        kind = ann.get("kind")
    else:
        kind = getattr(ann, "kind", None)
    return kind in ("obb", "pose")


def frames_have_polygons(frames_annotations) -> bool:
    for anns in frames_annotations:
        for ann in anns or []:
            if annotation_is_protected(ann):
                return True
    return False


def count_frames_with_polygons(frames_annotations) -> int:
    return sum(1 for anns in frames_annotations if frames_have_polygons([anns]))


def replace_warning_message(
    base: str,
    *,
    has_polygons: bool,
    polygon_frame_count: int = 0,
) -> str:
    if not has_polygons:
        return base
    extra = POLYGON_REPLACE_NOTE
    if polygon_frame_count > 1:
        extra = f"共 {polygon_frame_count} 帧含多边形。\n{extra}"
    base = (base or "").rstrip()
    if not base:
        return extra
    return f"{base}\n\n{extra}"
