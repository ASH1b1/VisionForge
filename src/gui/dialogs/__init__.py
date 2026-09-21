# src/gui/dialogs/__init__.py
from .process_video_dialog import ProcessVideoDialog
from .auto_annotate_dialog import AutoAnnotateDialog
from .replace_class_dialog import ReplaceClassDialog
from .refine_segmentation_dialog import RefineSegmentationDialog
from .image_resize_dialog import ImageResizeDialog
from .image_split_dialog import ImageSplitDialog

__all__ = [
    "ProcessVideoDialog",
    "AutoAnnotateDialog",
    "ReplaceClassDialog",
    "RefineSegmentationDialog",
    "ImageResizeDialog",
    "ImageSplitDialog",
]
