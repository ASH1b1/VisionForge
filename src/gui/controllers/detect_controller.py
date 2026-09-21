"""DetectController — non-destructive detection preview & export."""
from __future__ import annotations

import logging
from pathlib import Path
from typing import List, Optional, Sequence

from PySide6.QtCore import QObject
from PySide6.QtGui import QImage

try:
    from ...utils.annotation_processor import process_raw_detections
except ImportError:
    from utils.annotation_processor import process_raw_detections
from ..models.auto_annotation_result import AutoAnnotationResult
from ..models.detection_state import AppMode, DetectionState, FrameDetections
from ..models.detector_protocol import (
    UNMATCHED_CREATE,
    filter_detections_by_project_classes,
)
from ..models.grounding_dino_model import GroundingDINOModel
from ..utils.detect_exporter import (
    export_detections_csv,
    export_detections_json,
    export_overlay_image,
)

logger = logging.getLogger(__name__)


def parse_prompt_class_names(prompt: str) -> List[str]:
    """Parse prompt into class names (shared with GroundingDINO prompt style)."""
    return [
        c
        for c in GroundingDINOModel._parse_prompt_class_names(prompt)
        if c
    ]


def detections_from_raw(
    *,
    boxes,
    labels,
    scores,
    prompt: str,
    img_width: int,
    img_height: int,
    nms_threshold: float = 0.5,
    class_names: Optional[List[str]] = None,
    project_class_names: Optional[dict] = None,
    unmatched: str = UNMATCHED_CREATE,
) -> List[AutoAnnotationResult]:
    """Run the shared NMS / quality pipeline and return AutoAnnotationResult list."""
    names = list(class_names) if class_names else parse_prompt_class_names(prompt)
    if not names:
        names = [str(lbl) for lbl in labels]
    final = process_raw_detections(
        boxes=boxes,
        labels=labels,
        scores=scores,
        class_names=names,
        img_width=img_width,
        img_height=img_height,
        nms_threshold=nms_threshold,
    )
    if project_class_names is not None:
        final = filter_detections_by_project_classes(
            final,
            project_class_names,
            unmatched=unmatched,
        )
    return [
        AutoAnnotationResult(class_name=name, bbox=bbox, score=score)
        for name, bbox, score in final
    ]


class DetectController(QObject):
    """Owns DetectionState and bridges infer results → overlay / export / annotate."""

    def __init__(
        self,
        detection_state: DetectionState | None = None,
        parent: QObject | None = None,
    ):
        super().__init__(parent)
        self.state = detection_state or DetectionState(self)

    def set_mode(self, mode: AppMode) -> None:
        self.state.set_mode(mode)

    def enter_detect(self) -> None:
        self.set_mode(AppMode.DETECT)

    def enter_annotate(self) -> None:
        self.set_mode(AppMode.ANNOTATE)

    def store_results(
        self,
        *,
        frame_key: str,
        results: Sequence[AutoAnnotationResult],
        prompt: str,
        box_threshold: float,
        text_threshold: float,
        image_width: int,
        image_height: int,
        source_path: str = "",
    ) -> FrameDetections:
        frame = FrameDetections(
            frame_key=frame_key,
            results=list(results),
            prompt=prompt,
            box_threshold=box_threshold,
            text_threshold=text_threshold,
            image_width=image_width,
            image_height=image_height,
            source_path=source_path,
        )
        self.state.set_frame_detections(frame)
        return frame

    def clear(self, key: Optional[str] = None) -> None:
        self.state.clear(key)

    def export_json(self, path: Path, tool_name: str = "VisionForge") -> Path:
        return export_detections_json(
            self.state.all_frames().values(),
            path,
            score_filter=self.state.score_filter,
            tool_name=tool_name,
        )

    def export_csv(self, path: Path) -> Path:
        return export_detections_csv(
            self.state.all_frames().values(),
            path,
            score_filter=self.state.score_filter,
        )

    def export_active_overlay(self, image_path: Path, output_path: Path) -> Optional[Path]:
        frame = self.state.get_frame()
        if frame is None:
            return None
        return export_overlay_image(
            image_path,
            frame.results,
            output_path,
            score_filter=self.state.score_filter,
        )

    @staticmethod
    def image_size_from_qimage(image: QImage) -> tuple[int, int]:
        return int(image.width()), int(image.height())
