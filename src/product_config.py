"""VisionForge product identity and feature flags.

This repository is the standalone VisionForge product.
"""
from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache


@dataclass(frozen=True)
class ProductConfig:
    app_name: str
    app_version: str
    organization: str
    window_title: str
    about_html: str
    stylesheet_relpath: str
    logo_filename: str
    feature_detect_mode: bool
    feature_mode_switcher: bool
    feature_product_branding: bool

    @property
    def export_tool_name(self) -> str:
        return self.app_name


_CONFIG = ProductConfig(
    app_name="VisionForge",
    app_version="3.0.0-dev",
    organization="VisionForge",
    window_title="VisionForge — Annotate & Detect",
    about_html="""
        <h2>VisionForge</h2>
        <p>自有产品版：标注数据集与 Detect 预览一体。</p>
        <p><b>功能:</b></p>
        <ul>
        <li>Annotate 模式：GroundingDINO 检测；SAM3 可按需安装</li>
        <li>Detect 模式：非破坏性检测预览与结果导出</li>
        <li>多格式数据集导入/导出</li>
        <li>持续迭代的新模型与工作流</li>
        </ul>
        <p><b>依赖:</b> PySide6, OpenCV, PyTorch, Transformers</p>
    """,
    stylesheet_relpath="src/gui/styles/product_shell.qss",
    logo_filename="logo.png",
    feature_detect_mode=True,
    feature_mode_switcher=True,
    feature_product_branding=True,
)


@lru_cache(maxsize=1)
def get_product_config() -> ProductConfig:
    """Return the process-wide VisionForge ProductConfig."""
    return _CONFIG


def reset_product_config_cache() -> None:
    get_product_config.cache_clear()
