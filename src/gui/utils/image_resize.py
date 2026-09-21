"""Image resize core — fit / fill / exact modes."""
from __future__ import annotations

from pathlib import Path

from PIL import Image, ImageOps

from .image_preprocess_common import resolve_output_path


def resize_exact(img: Image.Image, target: tuple[int, int]) -> Image.Image:
    """Stretch image to exact target dimensions."""
    return img.resize(target, Image.LANCZOS)


def resize_fit(
    img: Image.Image, target: tuple[int, int], pad_color: tuple[int, ...]
) -> Image.Image:
    """Scale to fit within target, preserving aspect ratio. Pad remaining area."""
    tw, th = target
    sw, sh = img.size
    ratio = min(tw / sw, th / sh)
    new_w, new_h = int(sw * ratio), int(sh * ratio)
    resized = img.resize((new_w, new_h), Image.LANCZOS)

    if img.mode == "RGBA":
        canvas = Image.new("RGBA", (tw, th), pad_color + (255,))
    else:
        canvas = Image.new("RGB", (tw, th), pad_color)
    canvas.paste(resized, ((tw - new_w) // 2, (th - new_h) // 2))
    return canvas


def resize_fill(img: Image.Image, target: tuple[int, int]) -> Image.Image:
    """Scale to fill target, preserving aspect ratio. Center-crop overflow."""
    tw, th = target
    sw, sh = img.size
    ratio = max(tw / sw, th / sh)
    new_w, new_h = int(sw * ratio), int(sh * ratio)
    resized = img.resize((new_w, new_h), Image.LANCZOS)
    left = (new_w - tw) // 2
    top = (new_h - th) // 2
    return resized.crop((left, top, left + tw, top + th))


def plan_resize_outputs(
    images: list[Path],
    input_path: Path,
    output_dir: Path | None,
    suffix: str,
) -> list[Path]:
    """Compute destination paths for a resize batch (for overwrite checks)."""
    return [
        resolve_output_path(
            src,
            input_path,
            output_dir,
            suffix=suffix if output_dir is None else "",
            preserve_relative=True,
        )
        for src in images
    ]


def process_resize_one(
    src_path: Path,
    input_path: Path,
    target: tuple[int, int],
    mode: str,
    output_dir: Path | None,
    suffix: str,
    pad_color: tuple[int, ...],
    quality: int,
) -> Path:
    """
    Resize one image and save. Returns destination path.
    Raises on failure.
    """
    with Image.open(src_path) as img:
        img = ImageOps.exif_transpose(img) or img
        original = img.copy()

    if mode == "exact":
        result = resize_exact(original, target)
    elif mode == "fit":
        result = resize_fit(original, target, pad_color)
    else:
        result = resize_fill(original, target)

    dst = resolve_output_path(
        src_path,
        input_path,
        output_dir,
        suffix=suffix if output_dir is None else "",
        preserve_relative=True,
    )
    dst.parent.mkdir(parents=True, exist_ok=True)

    ext = dst.suffix.lower()
    save_kwargs: dict = {}
    out_img = result

    if ext in {".jpg", ".jpeg"}:
        if out_img.mode in ("RGBA", "P"):
            out_img = out_img.convert("RGB")
        save_kwargs["quality"] = quality
    elif ext == ".webp":
        save_kwargs["quality"] = quality

    out_img.save(dst, **save_kwargs)
    return dst
