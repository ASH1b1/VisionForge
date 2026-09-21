"""Session fixtures. Bootstrap env before any test imports torch / cv2."""
from __future__ import annotations

from src.runtime_env import bootstrap_runtime_env

bootstrap_runtime_env()
