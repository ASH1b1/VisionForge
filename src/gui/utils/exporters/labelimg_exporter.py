"""LabelImg 兼容格式导出器

LabelImg 使用 PASCAL VOC XML 格式，因此直接继承 VOCExporter。
本模块不实现 LabelMe JSON；勿将本导出器宣传为 LabelMe。
"""
from pathlib import Path
from .voc_exporter import VOCExporter


class LabelImgExporter(VOCExporter):
    """LabelImg 兼容格式导出器（PASCAL VOC XML）。"""

    def __init__(self, output_dir: str | Path):
        super().__init__(output_dir)


# Deprecated alias — historical misnomer (was never LabelMe JSON).
LabelmeExporter = LabelImgExporter
