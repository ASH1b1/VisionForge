"""Process-wide env bootstrap.

MUST run before import torch / cv2 / sam3 / transformers.
Do not set these variables in model files; setdefault cannot run after OpenMP loads.
"""
from __future__ import annotations

import os


def bootstrap_runtime_env() -> None:
    # Must run before import torch. Do not set these in model files.
    os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
    os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    os.environ.setdefault("MKL_NUM_THREADS", "1")


def is_openmp_duplicate_error(text: str) -> bool:
    t = (text or "").upper().replace(" ", "")
    return "OMP" in t and "ERROR#15" in t


def openmp_user_message() -> str:
    return (
        "检测到重复加载 OpenMP（OMP Error #15）。"
        "这是 PyTorch 与 SAM3 各带一份运行库导致的，不要手动删除 DLL。"
        "请从「启动.bat」或确保已调用 runtime_env 垫片后重试。"
    )
