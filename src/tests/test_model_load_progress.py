from __future__ import annotations

from pathlib import Path

from PySide6.QtWidgets import QApplication

from src.gui.components.model_load_progress import make_model_load_progress_bar
from src.gui.models.detector_protocol import ONNX_RUNTIME_MISSING_MESSAGE

_SRC_ROOT = Path(__file__).resolve().parents[1]
_REPO_ROOT = Path(__file__).resolve().parents[2]


def _ensure_qapp() -> QApplication:
    app = QApplication.instance()
    if app is None:
        app = QApplication([])
    return app


def test_make_model_load_progress_bar_is_indeterminate_and_hidden():
    _ensure_qapp()
    bar = make_model_load_progress_bar()
    assert bar.minimum() == 0
    assert bar.maximum() == 0
    assert bar.isTextVisible() is False
    assert bar.isVisible() is False
    bar.setVisible(True)
    assert bar.isVisible() is True
    bar.setVisible(False)
    assert bar.isVisible() is False


def test_onnx_runtime_missing_message_names_visionforge_env():
    assert "visionforge" in ONNX_RUNTIME_MISSING_MESSAGE
    assert "groundingdino2" not in ONNX_RUNTIME_MISSING_MESSAGE
    assert "pip install onnxruntime" in ONNX_RUNTIME_MISSING_MESSAGE
    assert "ultralytics" not in ONNX_RUNTIME_MISSING_MESSAGE


def test_src_python_has_no_groundingdino2_env_name():
    hits = []
    skip_dir = _SRC_ROOT / "tests"
    for path in _SRC_ROOT.rglob("*.py"):
        if skip_dir in path.parents or path.parent == skip_dir:
            continue
        text = path.read_text(encoding="utf-8")
        if "groundingdino2" in text:
            hits.append(str(path.relative_to(_REPO_ROOT)))
    assert hits == []


def test_launch_bat_hard_cuts_to_visionforge():
    text = (_REPO_ROOT / "启动.bat").read_text(encoding="utf-8")
    assert "visionforge" in text
    assert "groundingdino2" not in text
