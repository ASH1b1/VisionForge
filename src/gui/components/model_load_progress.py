"""Status-bar busy indicator for model weight loading."""
from __future__ import annotations

from PySide6.QtWidgets import QProgressBar


def make_model_load_progress_bar() -> QProgressBar:
    bar = QProgressBar()
    bar.setObjectName("modelLoadProgress")
    bar.setRange(0, 0)
    bar.setTextVisible(False)
    bar.setMaximumWidth(180)
    bar.setFixedHeight(14)
    bar.setVisible(False)
    return bar
