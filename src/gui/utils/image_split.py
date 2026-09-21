"""Image tile-split core — fixed-size tiles with overlap (edge snap-back)."""
from __future__ import annotations

from pathlib import Path

from PIL import Image, ImageOps

from .image_preprocess_common import resolve_output_path

OUTPUT_FORMATS = ("PNG", "JPG", "TIF", "WEBP", "BMP")


def format_to_ext(fmt: str) -> str:
    mapping = {
        "PNG": ".png",
        "JPG": ".jpg",
        "TIF": ".tif",
        "WEBP": ".webp",
        "BMP": ".bmp",
    }
    return mapping.get(fmt.upper(), ".png")


def tile_origins(length: int, tile: int, overlap: float) -> list[int]:
    """Compute start positions along one axis with edge snap-back to full tiles."""
    if tile >= length:
        return [0]
    step = max(1, int(round(tile * (1.0 - overlap))))
    positions: list[int] = []
    pos = 0
    while True:
        positions.append(pos)
        if pos + tile >= length:
            break
        nxt = pos + step
        if nxt + tile > length:
            nxt = length - tile
        if nxt <= pos:
            break
        pos = nxt
    return positions


def count_tiles(width: int, height: int, tile_w: int, tile_h: int, overlap: float) -> int:
    xs = tile_origins(width, tile_w, overlap)
    ys = tile_origins(height, tile_h, overlap)
    return len(xs) * len(ys)


def plan_split_outputs(
    images: list[Path],
    input_path: Path,
    output_dir: Path,
    tile_w: int,
    tile_h: int,
    overlap: float,
    fmt: str,
) -> tuple[list[Path], int]:
    """
    Plan all tile output paths and total tile count.
    Opens each image to read size (needed for accurate overwrite checks / progress).
    """
    ext = format_to_ext(fmt)
    planned: list[Path] = []
    total = 0
    for src in images:
        with Image.open(src) as im:
            im = ImageOps.exif_transpose(im) or im
            w, h = im.size
        xs = tile_origins(w, tile_w, overlap)
        ys = tile_origins(h, tile_h, overlap)
        total += len(xs) * len(ys)
        # Destination directory mirrors relative structure; tiles flatten under that dir
        base = resolve_output_path(
            src,
            input_path,
            output_dir,
            new_suffix=ext,
            preserve_relative=True,
        )
        dest_dir = base.parent
        name = src.stem
        for ri in range(len(ys)):
            for ci in range(len(xs)):
                planned.append(dest_dir / f"{name}_r{ri}_c{ci}{ext}")
    return planned, total


def save_tile(img: Image.Image, path: Path, fmt: str, quality: int = 95) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fmt_u = fmt.upper()
    if fmt_u == "JPG":
        img.convert("RGB").save(path, format="JPEG", quality=quality)
    elif fmt_u == "TIF":
        img.save(path, format="TIFF")
    elif fmt_u == "WEBP":
        out = img.convert("RGB") if img.mode in ("RGBA", "P") else img
        out.save(path, format="WEBP", quality=quality)
    elif fmt_u == "BMP":
        out = img.convert("RGB") if img.mode in ("RGBA", "P") else img
        out.save(path, format="BMP")
    else:
        img.save(path, format="PNG")


def split_one(
    src_path: Path,
    input_path: Path,
    output_dir: Path,
    tile_w: int,
    tile_h: int,
    overlap: float,
    fmt: str,
    *,
    quality: int = 95,
    should_cancel=None,
    on_tile=None,
) -> tuple[int, bool]:
    """
    Split one image into tiles.

    Returns (tiles_written, cancelled).
    should_cancel: callable () -> bool
    on_tile: callable (done_in_this_image: int, total_in_this_image: int)
    """
    ext = format_to_ext(fmt)
    with Image.open(src_path) as im:
        im = ImageOps.exif_transpose(im) or im
        img = im.copy()

    w, h = img.size
    xs = tile_origins(w, tile_w, overlap)
    ys = tile_origins(h, tile_h, overlap)
    total_here = len(xs) * len(ys)

    base = resolve_output_path(
        src_path,
        input_path,
        output_dir,
        new_suffix=ext,
        preserve_relative=True,
    )
    dest_dir = base.parent
    name = src_path.stem
    written = 0

    for ri, y in enumerate(ys):
        for ci, x in enumerate(xs):
            if should_cancel and should_cancel():
                return written, True
            # Clamp crop box to image (tile always full size via snap-back origins)
            tw = min(tile_w, w - x)
            th = min(tile_h, h - y)
            tile = img.crop((x, y, x + tw, y + th))
            out = dest_dir / f"{name}_r{ri}_c{ci}{ext}"
            save_tile(tile, out, fmt, quality=quality)
            written += 1
            if on_tile:
                on_tile(written, total_here)

    return written, False
