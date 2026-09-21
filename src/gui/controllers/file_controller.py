"""FileController — handles file I/O for the annotation tool."""
from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional, Set

import cv2
from PySide6.QtCore import QObject
from PySide6.QtWidgets import QFileDialog, QInputDialog, QMessageBox, QTreeWidgetItem

from ..models.project_document import ProjectDocument


class FileController(QObject):
    """Orchestrates file open / save / import operations.

    All mutations write into the shared ProjectDocument;
    this class does not hold its own state.
    """

    IMAGE_EXTENSIONS: Set[str] = {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".tif", ".webp"}

    def __init__(self, project: ProjectDocument, parent: QObject | None = None):
        super().__init__(parent)
        self._project = project

    # ========================================================================
    # Open
    # ========================================================================

    def open_video(self, parent_widget=None) -> Optional[Dict[str, Any]]:
        """Open a video file. Returns metadata dict or None if cancelled/failed."""
        path_str, _ = QFileDialog.getOpenFileName(
            parent_widget, "打开视频", "",
            "视频文件 (*.mp4 *.avi *.mov *.mkv *.flv *.wmv);;所有文件 (*)",
        )
        if not path_str:
            return None

        path = Path(path_str)
        cap = cv2.VideoCapture(str(path))
        if not cap.isOpened():
            QMessageBox.critical(parent_widget, "错误", f"无法打开视频: {path.name}")
            return None

        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps_value = fps if fps > 0 else 30.0

        self._project.replace_data(
            current_frame_index=0,
            video_capture=cap,
            video={
                "path": path,
                "fps": fps_value,
                "total_frames": total_frames,
                "width": width,
                "height": height,
            },
        )

        return {
            "path": path,
            "total_frames": total_frames,
            "fps": fps_value,
            "width": width,
            "height": height,
        }

    def open_image_dir(self, parent_widget=None) -> Optional[Dict[str, Any]]:
        """Open a directory of images. Returns metadata dict or None."""
        dir_path = QFileDialog.getExistingDirectory(
            parent_widget, "选择图片目录", "",
            QFileDialog.ShowDirsOnly | QFileDialog.DontResolveSymlinks,
        )
        if not dir_path:
            return None

        directory = Path(dir_path)
        image_files: List[Path] = []
        seen: Set[str] = set()

        for entry in sorted(directory.iterdir()):
            if not entry.is_file():
                continue
            ext = entry.suffix.lower()
            if ext not in self.IMAGE_EXTENSIONS:
                continue
            key = str(entry.resolve())
            if key in seen:
                continue
            seen.add(key)
            image_files.append(entry)

        if not image_files:
            QMessageBox.warning(
                parent_widget, "未找到图片",
                f"在 {directory.name} 中未找到支持的图片文件",
            )
            return None

        self._project.replace_data(
            image_files=sorted(image_files),
            current_frame_index=0,
        )

        return {
            "directory": directory,
            "image_files": self._project.image_files,
        }

    # ========================================================================
    # Save / Load project (.gsproj)
    # ========================================================================

    def save_project(self, parent_widget=None) -> bool:
        """Save current project state to .gsproj file."""
        if self._project.project_path is None:
            return self.save_project_as(parent_widget)
        try:
            self._project.save_to_file(self._project.project_path)
            return True
        except Exception as e:
            QMessageBox.critical(parent_widget, "保存失败", str(e))
            return False

    def save_project_as(self, parent_widget=None) -> bool:
        """Save project to a new .gsproj file."""
        path_str, _ = QFileDialog.getSaveFileName(
            parent_widget, "保存项目", "",
            "GS项目文件 (*.gsproj);;所有文件 (*)",
        )
        if not path_str:
            return False
        path = Path(path_str)
        if path.suffix != ".gsproj":
            path = path.with_suffix(".gsproj")
        try:
            self._project.save_to_file(path)
            return True
        except Exception as e:
            QMessageBox.critical(parent_widget, "保存失败", str(e))
            return False

    def open_project(self, parent_widget=None) -> bool:
        """Open a .gsproj project file and restore state."""
        path_str, _ = QFileDialog.getOpenFileName(
            parent_widget, "打开项目", "",
            "GS项目文件 (*.gsproj);;所有文件 (*)",
        )
        if not path_str:
            return False
        path = Path(path_str)
        if not path.exists():
            QMessageBox.critical(parent_widget, "错误", f"文件不存在: {path}")
            return False
        try:
            self._project.load_from_file(path)
            self._try_restore_video_capture()
            return True
        except Exception as e:
            QMessageBox.critical(parent_widget, "打开项目失败", str(e))
            return False

    def _try_restore_video_capture(self) -> bool:
        """Reopen VideoCapture from restored video metadata when possible."""
        video = self._project.video
        if not video or not video.get("path"):
            return False
        vpath = Path(video["path"])
        if not vpath.exists():
            return False
        cap = cv2.VideoCapture(str(vpath))
        if not cap.isOpened():
            cap.release()
            return False
        old = self._project.video_capture
        if old is not None and old is not cap:
            old.release()
        self._project.video_capture = cap
        return True

    # ========================================================================
    # Legacy save (YOLO TXT)
    # ========================================================================

    def save_current_frame_yolo(self, image_path: Path, parent_widget=None) -> bool:
        """Save annotations of current frame to YOLO TXT file."""
        label_dir = image_path.parent
        label_file = label_dir / f"{image_path.stem}.txt"
        try:
            with open(label_file, "w", encoding="utf-8") as f:
                for ann in self._project.visible_annotations:
                    x, y, w, h = ann.bbox
                    f.write(f"{ann.class_id} {x:.6f} {y:.6f} {w:.6f} {h:.6f}\n")
            return True
        except Exception as e:
            QMessageBox.critical(parent_widget, "保存失败", str(e))
            return False

    def save_all_yolo(self, output_dir: Path, parent_widget=None) -> int:
        """Save all frames to YOLO TXT files. Returns number of files written."""
        label_dir = output_dir / "labels"
        label_dir.mkdir(parents=True, exist_ok=True)
        count = 0
        for frame_idx, frame_data in self._project.frame_annotations.items():
            if self._project.image_files and 0 <= frame_idx < len(self._project.image_files):
                img_path = self._project.image_files[frame_idx]
                label_file = label_dir / f"{img_path.stem}.txt"
                with open(label_file, "w", encoding="utf-8") as f:
                    for ann in frame_data:
                        if ann.visible:
                            x, y, w, h = ann.bbox
                            f.write(f"{ann.class_id} {x:.6f} {y:.6f} {w:.6f} {h:.6f}\n")
                count += 1
        return count

    # ========================================================================
    # Import
    # ========================================================================

    YOLO_TASK_ITEMS = [
        ("自动（读 data.yaml）", "auto"),
        ("检测", "detect"),
        ("分割", "segment"),
        ("姿态", "pose"),
        ("OBB 旋转框", "obb"),
    ]

    def prompt_yolo_import_task(self, parent_widget=None, *, required: bool = False) -> Optional[str]:
        items = self.YOLO_TASK_ITEMS[1:] if required else self.YOLO_TASK_ITEMS
        labels = [label for label, _code in items]
        label, ok = QInputDialog.getItem(
            parent_widget,
            "YOLO 导入格式",
            "无法自动判定，请选择标签格式：" if required else "选择标签格式：",
            labels,
            0,
            False,
        )
        if not ok:
            return None
        mapping = dict(items)
        return mapping.get(label)

    def import_dataset(self, format_type: str, image_dir: Path, label_dir: Path,
                       parent_widget=None, task_choice: str = "auto") -> bool:
        """Import a YOLO/VOC/COCO dataset into ProjectDocument."""
        from ..utils.importers import YOLOImporter, VOCImporter, COCOImporter
        importers = {"yolo": YOLOImporter, "voc": VOCImporter, "coco": COCOImporter}
        importer_cls = importers.get(format_type)
        if importer_cls is None:
            QMessageBox.critical(parent_widget, "错误", f"不支持的格式: {format_type}")
            return False
        try:
            importer = importer_cls()
            kwargs = {}
            if format_type == "yolo":
                kwargs["task_choice"] = task_choice
            image_files, frame_annotations, class_names, split_map, result = (
                importer.import_dataset(image_dir, label_dir, **kwargs))
            if getattr(result, "needs_task_choice", False):
                chosen = self.prompt_yolo_import_task(parent_widget, required=True)
                if not chosen:
                    return False
                image_files, frame_annotations, class_names, split_map, result = (
                    importer.import_dataset(image_dir, label_dir, task_choice=chosen)
                )
            self._project.replace_data(
                image_files=list(image_files),
                frame_annotations={
                    int(k): list(v) for k, v in frame_annotations.items()
                },
                class_names=dict(class_names),
                split_map=dict(split_map),
                current_frame_index=0,
                kpt_shape=getattr(result, "kpt_shape", None),
                kpt_names=getattr(result, "kpt_names", None),
            )
            return True
        except Exception as e:
            QMessageBox.critical(parent_widget, "导入失败", str(e))
            return False

    # ========================================================================
    # Helpers
    # ========================================================================

    def populate_file_tree(self, directory: Path, files: list, tree_widget) -> None:
        """Populate a QTreeWidget with file list."""
        tree_widget.clear()
        root_item = tree_widget.invisibleRootItem()
        dir_item = QTreeWidgetItem([f"📁 {directory.name}"])
        dir_item.setData(0, 0, directory)
        root_item.addChild(dir_item)
        for idx, file_path in enumerate(files):
            file_item = QTreeWidgetItem([f"🖼️ {file_path.name}"])
            file_item.setData(0, 0, file_path)
            file_item.setData(0, 1, idx)
            dir_item.addChild(file_item)
        dir_item.setExpanded(True)
