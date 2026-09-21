"""Export Detect-mode results (JSON / CSV / overlay image) without ProjectDocument."""
from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Iterable, List, Optional, Sequence

import cv2
import numpy as np

from ..models.auto_annotation_result import AutoAnnotationResult
from ..models.detection_state import FrameDetections


def _yolo_to_xyxy(
    bbox: tuple[float, float, float, float],
    img_w: int,
    img_h: int,
) -> tuple[float, float, float, float]:
    cx, cy, w, h = bbox
    x1 = (cx - w / 2.0) * img_w
    y1 = (cy - h / 2.0) * img_h
    x2 = (cx + w / 2.0) * img_w
    y2 = (cy + h / 2.0) * img_h
    return x1, y1, x2, y2


def results_to_dicts(
    results: Sequence[AutoAnnotationResult],
    *,
    img_w: int = 0,
    img_h: int = 0,
    include_xyxy: bool = True,
) -> List[dict]:
    rows: List[dict] = []
    for item in results:
        row = {
            "class_name": item.class_name,
            "score": float(item.score),
            "bbox_yolo": list(item.bbox),
        }
        if include_xyxy and img_w > 0 and img_h > 0:
            x1, y1, x2, y2 = _yolo_to_xyxy(item.bbox, img_w, img_h)
            row["bbox_xyxy"] = [x1, y1, x2, y2]
        rows.append(row)
    return rows


def export_detections_json(
    frames: Iterable[FrameDetections],
    output_path: Path,
    *,
    score_filter: float = 0.0,
    tool_name: str = "VisionForge",
) -> Path:
    output_path = Path(output_path)
    payload = {
        "tool": tool_name,
        "score_filter": score_filter,
        "frames": [],
    }
    for frame in frames:
        visible = [r for r in frame.results if float(r.score) >= score_filter]
        payload["frames"].append(
            {
                "frame_key": frame.frame_key,
                "source_path": frame.source_path,
                "prompt": frame.prompt,
                "image_width": frame.image_width,
                "image_height": frame.image_height,
                "box_threshold": frame.box_threshold,
                "text_threshold": frame.text_threshold,
                "detections": results_to_dicts(
                    visible,
                    img_w=frame.image_width,
                    img_h=frame.image_height,
                ),
            }
        )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)
    return output_path


def export_detections_csv(
    frames: Iterable[FrameDetections],
    output_path: Path,
    *,
    score_filter: float = 0.0,
) -> Path:
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "frame_key",
        "source_path",
        "class_name",
        "score",
        "cx",
        "cy",
        "w",
        "h",
        "x1",
        "y1",
        "x2",
        "y2",
    ]
    with open(output_path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for frame in frames:
            for item in frame.results:
                if float(item.score) < score_filter:
                    continue
                cx, cy, w, h = item.bbox
                row = {
                    "frame_key": frame.frame_key,
                    "source_path": frame.source_path,
                    "class_name": item.class_name,
                    "score": float(item.score),
                    "cx": cx,
                    "cy": cy,
                    "w": w,
                    "h": h,
                    "x1": "",
                    "y1": "",
                    "x2": "",
                    "y2": "",
                }
                if frame.image_width > 0 and frame.image_height > 0:
                    x1, y1, x2, y2 = _yolo_to_xyxy(
                        item.bbox, frame.image_width, frame.image_height
                    )
                    row.update({"x1": x1, "y1": y1, "x2": x2, "y2": y2})
                writer.writerow(row)
    return output_path


def render_detection_overlay(
    image_bgr: np.ndarray,
    results: Sequence[AutoAnnotationResult],
    *,
    score_filter: float = 0.0,
    line_width: int = 2,
) -> np.ndarray:
    """Draw YOLO-normalized boxes onto a BGR image copy."""
    out = image_bgr.copy()
    h, w = out.shape[:2]
    colors = [
        (0, 200, 255),
        (80, 220, 100),
        (255, 160, 40),
        (200, 100, 255),
        (60, 180, 255),
    ]
    for idx, item in enumerate(results):
        if float(item.score) < score_filter:
            continue
        x1, y1, x2, y2 = _yolo_to_xyxy(item.bbox, w, h)
        color = colors[idx % len(colors)]
        p1 = (int(round(x1)), int(round(y1)))
        p2 = (int(round(x2)), int(round(y2)))
        cv2.rectangle(out, p1, p2, color, line_width)
        label = f"{item.class_name} {float(item.score):.2f}"
        cv2.putText(
            out,
            label,
            (p1[0], max(16, p1[1] - 6)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            color,
            1,
            cv2.LINE_AA,
        )
    return out


def export_overlay_image(
    image_path: Path,
    results: Sequence[AutoAnnotationResult],
    output_path: Path,
    *,
    score_filter: float = 0.0,
) -> Optional[Path]:
    image_path = Path(image_path)
    data = np.fromfile(str(image_path), dtype=np.uint8)
    image = cv2.imdecode(data, cv2.IMREAD_COLOR)
    if image is None:
        return None
    overlay = render_detection_overlay(image, results, score_filter=score_filter)
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    ok, buf = cv2.imencode(output_path.suffix or ".png", overlay)
    if not ok:
        return None
    buf.tofile(str(output_path))
    return output_path
