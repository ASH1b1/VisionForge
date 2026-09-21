"""Transient detection results for Detect mode (non-destructive overlay)."""
from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Optional

from PySide6.QtCore import QObject, Signal

from .auto_annotation_result import AutoAnnotationResult


class AppMode(str, Enum):
    ANNOTATE = "annotate"
    DETECT = "detect"


@dataclass
class FrameDetections:
    """Detection results for one image/frame key."""

    frame_key: str
    results: List[AutoAnnotationResult] = field(default_factory=list)
    prompt: str = ""
    box_threshold: float = 0.35
    text_threshold: float = 0.25
    image_width: int = 0
    image_height: int = 0
    source_path: str = ""


class DetectionState(QObject):
    """Holds Detect-mode overlays without writing to ProjectDocument."""

    detections_changed = Signal()
    score_filter_changed = Signal(float)
    mode_changed = Signal(str)  # AppMode value
    cleared = Signal()

    def __init__(self, parent: QObject | None = None):
        super().__init__(parent)
        self._mode = AppMode.ANNOTATE
        self._by_frame: Dict[str, FrameDetections] = {}
        self._active_key: Optional[str] = None
        self._score_filter: float = 0.0

    @property
    def mode(self) -> AppMode:
        return self._mode

    def set_mode(self, mode: AppMode) -> None:
        if mode is self._mode:
            return
        self._mode = mode
        self.mode_changed.emit(mode.value)

    @property
    def score_filter(self) -> float:
        return self._score_filter

    def set_score_filter(self, threshold: float) -> None:
        threshold = max(0.0, min(1.0, float(threshold)))
        if abs(threshold - self._score_filter) < 1e-9:
            return
        self._score_filter = threshold
        self.score_filter_changed.emit(threshold)

    @property
    def active_key(self) -> Optional[str]:
        return self._active_key

    def set_active_key(self, key: Optional[str]) -> None:
        if key == self._active_key:
            return
        self._active_key = key
        self.detections_changed.emit()

    def set_frame_detections(self, frame: FrameDetections) -> None:
        self._by_frame[frame.frame_key] = frame
        self._active_key = frame.frame_key
        self.detections_changed.emit()

    def get_frame(self, key: Optional[str] = None) -> Optional[FrameDetections]:
        key = key if key is not None else self._active_key
        if key is None:
            return None
        return self._by_frame.get(key)

    def visible_results(self, key: Optional[str] = None) -> List[AutoAnnotationResult]:
        frame = self.get_frame(key)
        if frame is None:
            return []
        return [r for r in frame.results if float(r.score) >= self._score_filter]

    def all_frames(self) -> Dict[str, FrameDetections]:
        return dict(self._by_frame)

    def clear(self, key: Optional[str] = None) -> None:
        if key is None:
            self._by_frame.clear()
            self._active_key = None
        else:
            self._by_frame.pop(key, None)
            if self._active_key == key:
                self._active_key = next(iter(self._by_frame), None)
        self.cleared.emit()
        self.detections_changed.emit()

    def has_any(self) -> bool:
        return any(bool(f.results) for f in self._by_frame.values())
