"""Utilities for normalizing class labels and IDs."""

from typing import Any, Optional


_TRIM_CHARS = " \t\r\n.,;:|"


def display_class_name(value: Any) -> str:
    """Return a trimmed display name for a class label."""
    if value is None:
        return ""

    text = " ".join(str(value).strip().split())
    return text.strip(_TRIM_CHARS)


def normalize_class_name(value: Any) -> str:
    """Return a normalized key suitable for class-name comparisons."""
    return display_class_name(value).casefold()


def coerce_class_id(value: Any) -> Optional[int]:
    """Convert a raw class label to an integer ID when possible."""
    if isinstance(value, bool):
        return int(value)

    if isinstance(value, int):
        return value

    if isinstance(value, float):
        return int(value) if value.is_integer() else None

    text = display_class_name(value)
    if not text:
        return None

    try:
        return int(text)
    except ValueError:
        return None
