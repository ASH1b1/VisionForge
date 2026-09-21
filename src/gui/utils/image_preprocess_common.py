"""Shared helpers for image resize / split preprocessing tools."""
from __future__ import annotations

from pathlib import Path

# Align with annotation software image open support
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".tif", ".webp"}


def collect_images(input_path: Path, recursive: bool = False) -> list[Path]:
    """Collect image files from a file or directory."""
    if input_path.is_file():
        if input_path.suffix.lower() in IMAGE_EXTENSIONS:
            return [input_path]
        return []

    if not input_path.is_dir():
        return []

    if recursive:
        files = [
            f
            for f in input_path.rglob("*")
            if f.is_file() and f.suffix.lower() in IMAGE_EXTENSIONS
        ]
    else:
        files = [
            f
            for f in input_path.iterdir()
            if f.is_file() and f.suffix.lower() in IMAGE_EXTENSIONS
        ]
    return sorted(files)


def source_root_for(src_path: Path, input_path: Path) -> Path:
    """Directory used as root when preserving relative output paths."""
    if input_path.is_dir():
        return input_path
    return src_path.parent


def resolve_output_path(
    src_path: Path,
    input_path: Path,
    output_dir: Path | None,
    *,
    suffix: str = "",
    new_suffix: str | None = None,
    preserve_relative: bool = True,
) -> Path:
    """
    Build destination path.

    - If output_dir is None: write beside source with optional stem suffix.
    - If output_dir is set and preserve_relative: mirror relative path under output_dir.
    - new_suffix replaces the file extension (e.g. '.png'); None keeps original.
    """
    stem = src_path.stem + (suffix or "")
    ext = new_suffix if new_suffix is not None else src_path.suffix

    if output_dir is None:
        return src_path.with_name(stem + ext)

    if preserve_relative:
        root = source_root_for(src_path, input_path)
        try:
            rel_parent = src_path.parent.relative_to(root)
        except ValueError:
            rel_parent = Path()
        dest_dir = output_dir / rel_parent
    else:
        dest_dir = output_dir

    dest_dir.mkdir(parents=True, exist_ok=True)
    return dest_dir / (stem + ext)


def existing_targets(paths: list[Path]) -> list[Path]:
    """Return paths that already exist on disk."""
    return [p for p in paths if p.exists()]
