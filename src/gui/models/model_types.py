from dataclasses import dataclass
from typing import Optional


@dataclass
class SegmentationResult:
    bbox: tuple[float, float, float, float]
    polygon: Optional[list[tuple[float, float]]] = None
    mask = None
