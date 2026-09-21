from dataclasses import dataclass
from typing import Optional


@dataclass
class AutoAnnotationResult:
    class_name: str
    bbox: tuple[float, float, float, float]
    score: float
    polygon: Optional[list[tuple[float, float]]] = None
