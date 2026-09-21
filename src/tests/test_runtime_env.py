from __future__ import annotations

import os
from pathlib import Path

import pytest

from src.runtime_env import (
    bootstrap_runtime_env,
    is_openmp_duplicate_error,
    openmp_user_message,
)

_TRACKED = (
    "KMP_DUPLICATE_LIB_OK",
    "TRANSFORMERS_OFFLINE",
    "HF_HUB_OFFLINE",
    "OMP_NUM_THREADS",
    "MKL_NUM_THREADS",
)


@pytest.fixture(autouse=True)
def _restore_env():
    snapshot = {key: os.environ.get(key) for key in _TRACKED}
    yield
    for key, value in snapshot.items():
        if value is None:
            os.environ.pop(key, None)
        else:
            os.environ[key] = value


def test_bootstrap_sets_kmp_and_offline():
    os.environ.pop("KMP_DUPLICATE_LIB_OK", None)
    os.environ.pop("TRANSFORMERS_OFFLINE", None)
    os.environ.pop("HF_HUB_OFFLINE", None)
    bootstrap_runtime_env()
    assert os.environ["KMP_DUPLICATE_LIB_OK"] == "TRUE"
    assert os.environ["TRANSFORMERS_OFFLINE"] == "1"
    assert os.environ["HF_HUB_OFFLINE"] == "1"


def test_bootstrap_does_not_override_existing():
    os.environ["KMP_DUPLICATE_LIB_OK"] = "FALSE"
    bootstrap_runtime_env()
    assert os.environ["KMP_DUPLICATE_LIB_OK"] == "FALSE"


def test_openmp_error_detector():
    assert is_openmp_duplicate_error("OMP: Error #15: Initializing libiomp5md.dll")
    assert not is_openmp_duplicate_error("some other error")


def test_openmp_user_message_is_chinese():
    text = openmp_user_message()
    assert "OpenMP" in text
    assert "不要手动删除 DLL" in text


def test_segmentation_utils_does_not_import_torch_at_module_level():
    text = Path("src/gui/utils/segmentation_utils.py").read_text(encoding="utf-8")
    for line in text.splitlines():
        s = line.strip()
        if s.startswith("#"):
            continue
        assert "import torch" not in s


_ALLOWED = {
    Path("src/runtime_env.py"),
    Path("启动.bat"),
}


def test_kmp_not_set_outside_shim():
    hits = []
    root = Path("src")
    for p in list(root.rglob("*.py")) + [Path("启动.bat")]:
        if p in _ALLOWED or p.name == "test_runtime_env.py":
            continue
        try:
            text = p.read_text(encoding="utf-8", errors="ignore")
        except OSError:
            continue
        if "KMP_DUPLICATE_LIB_OK" in text:
            hits.append(str(p))
    assert hits == [], hits
