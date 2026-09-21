"""ExportController — format-aware dataset export."""
from __future__ import annotations
import shutil
from collections import Counter
from pathlib import Path
from typing import Dict, Optional
import yaml
from PySide6.QtCore import QObject
from PySide6.QtWidgets import QDialog, QFileDialog, QMessageBox
from ..models.project_document import ProjectDocument
from src.gui.utils.yolo_label_format import format_export_skip_message


_YOLO_BATCH_TASKS = {
    "yolo": "detect",
    "yolo_seg": "segment",
    "yolo_pose": "pose",
    "yolo_obb": "obb",
}
_YOLO_TASK_LABELS = {
    "detect": "检测框",
    "segment": "多边形",
    "pose": "姿态",
    "obb": "旋转框",
}


def _annotation_export_dict(ann, class_id=None) -> dict:
    data = ann.to_dict() if hasattr(ann, "to_dict") else dict(ann)
    if class_id is not None:
        data["class_id"] = class_id
    data.setdefault("visible", getattr(ann, "visible", True))
    data.setdefault("kind", getattr(ann, "kind", None))
    data.setdefault("obb", getattr(ann, "obb", None))
    data.setdefault("keypoints", getattr(ann, "keypoints", None))
    return data


class ExportController(QObject):
    """Handles exporting annotations in YOLO/VOC/COCO/Mask formats."""

    def __init__(self, project: ProjectDocument, parent: QObject | None = None):
        super().__init__(parent)
        self._project = project

    def export_current_frame(self, format_type: str, image_path: Path,
                             parent_widget=None) -> Optional[Path]:
        output_dir = QFileDialog.getExistingDirectory(parent_widget, "选择导出目录", "")
        if not output_dir:
            return None
        output_path = Path(output_dir)
        annotations = [
            _annotation_export_dict(a)
            for a in self._project.current_annotations if a.visible
        ]
        try:
            exporter = self._get_exporter(format_type)
            exporter.set_classes(self._project.class_names)
            if format_type == "coco":
                exporter.kpt_shape = self._project.kpt_shape
                exporter.kpt_names = self._project.kpt_names
            return exporter.export_current_frame(annotations, image_path, output_path)
        except Exception as e:
            QMessageBox.critical(parent_widget, "导出失败", str(e))
            return None

    def export_batch(self, format_type: str, parent_widget=None) -> Optional[Dict[str, int]]:
        output_dir = QFileDialog.getExistingDirectory(parent_widget, "选择导出目录", "")
        if not output_dir:
            return None
        output_path = Path(output_dir)
        try:
            if format_type in _YOLO_BATCH_TASKS:
                return self._export_yolo_by_task(
                    _YOLO_BATCH_TASKS[format_type], output_path, parent_widget
                )
            elif format_type in ("voc", "coco"):
                return self._export_standard_batch(
                    format_type, output_path, parent_widget
                )
            else:
                QMessageBox.critical(parent_widget, "错误", f"不支持的导出格式: {format_type}")
                return None
        except Exception as e:
            QMessageBox.critical(parent_widget, "导出失败", str(e))
            return None

    def _export_standard_batch(
        self, format_type: str, output_path: Path, parent_widget=None
    ) -> Optional[Dict[str, int]]:
        """VOC/COCO batch export via existing exporters."""
        if not self._project.image_files:
            QMessageBox.warning(
                parent_widget,
                "无法导出",
                "没有可导出的图片文件。视频项目请先提取帧或打开图片目录后再导出。",
            )
            return {}
        data = []
        image_paths = []
        for idx, img_path in enumerate(self._project.image_files):
            frame_data = [
                self._project._coerce_annotation(a)
                for a in self._project.frame_annotations.get(idx, [])
            ]
            ann_dicts = [
                _annotation_export_dict(a)
                for a in frame_data
                if a.visible
            ]
            data.append((img_path, ann_dicts))
            image_paths.append(img_path)
        if not data:
            return {}
        split_map = self._get_or_configure_split(len(data), parent_widget)
        if split_map is None:
            return {}

        target_names = (
            ("Annotations", "JPEGImages", "classes.txt")
            if format_type == "voc"
            else ("annotations", "images")
        )
        existing = [
            name for name in target_names if (output_path / name).exists()
        ]
        if existing:
            reply = QMessageBox.question(
                parent_widget,
                "确认覆盖",
                f"导出目录已存在 {', '.join(existing)}，继续将删除其中内容。是否继续？",
                QMessageBox.Yes | QMessageBox.No,
                QMessageBox.No,
            )
            if reply != QMessageBox.Yes:
                return {}
            for name in existing:
                target = output_path / name
                if target.is_dir():
                    shutil.rmtree(target)
                else:
                    target.unlink()

        exporter = self._get_exporter(format_type)
        exporter.set_classes(self._project.class_names)
        if format_type == "coco":
            exporter.kpt_shape = self._project.kpt_shape
            exporter.kpt_names = self._project.kpt_names
        result = exporter.export_batch(
            data, output_path, image_paths if image_paths else None, split_map=split_map
        )
        if not (isinstance(result, tuple) and len(result) == 3):
            return {}
        success, total, split_counts = result
        if success == 0:
            QMessageBox.critical(
                parent_widget,
                "导出失败",
                f"没有成功导出任何图片（0/{total}）。"
                "请检查图片是否可读、输出目录是否可写。",
            )
            return None
        return split_counts

    def _export_yolo_dataset(self, output_path: Path, parent_widget=None) -> Optional[Dict[str, int]]:
        return self._export_yolo_by_task("detect", output_path, parent_widget)

    def _export_yolo_seg_dataset(
        self, output_path: Path, parent_widget=None
    ) -> Optional[Dict[str, int]]:
        return self._export_yolo_by_task("segment", output_path, parent_widget)

    def _export_yolo_by_task(
        self, task: str, output_path: Path, parent_widget=None
    ) -> Optional[Dict[str, int]]:
        from ..utils.exporters import YOLOExporter
        _IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".tif", ".webp"}
        files = self._project.image_files
        if not files:
            QMessageBox.warning(
                parent_widget,
                "无法导出",
                "没有可导出的图片文件。视频项目请先提取帧或打开图片目录后再导出。",
            )
            return {}
        splits = self._get_or_configure_split(len(files), parent_widget)
        if splits is None:
            return {}

        existing = [
            name for name in ("images", "labels")
            if (output_path / name).exists()
        ]
        if existing:
            reply = QMessageBox.question(
                parent_widget,
                "确认覆盖",
                f"导出目录已存在 {', '.join(existing)} 文件夹，继续将删除其中内容。是否继续？",
                QMessageBox.Yes | QMessageBox.No,
                QMessageBox.No,
            )
            if reply != QMessageBox.Yes:
                return {}
        for dir_name in ("images", "labels"):
            d = output_path / dir_name
            if d.exists():
                shutil.rmtree(d)
        for split in ("train", "val", "test"):
            (output_path / "images" / split).mkdir(parents=True, exist_ok=True)
            (output_path / "labels" / split).mkdir(parents=True, exist_ok=True)

        class_ids = set(self._project.class_names)
        for frame_data in self._project.frame_annotations.values():
            for ann in frame_data:
                normalized = self._project._coerce_annotation(ann)
                if normalized.visible:
                    class_ids.add(normalized.class_id)
        sorted_class_ids = sorted(class_ids)
        class_id_map = {
            source_id: export_id
            for export_id, source_id in enumerate(sorted_class_ids)
        }

        count = {"train": 0, "val": 0, "test": 0}
        skip_total: Counter = Counter()
        written = 0
        exporter = YOLOExporter(str(output_path))
        exporter.set_classes({
            class_id_map[source_id]: self._project.class_names.get(
                source_id, f"class_{source_id}"
            )
            for source_id in sorted_class_ids
        })
        export_names = exporter.unique_image_names(files)
        kpt_shape = self._project.kpt_shape
        for idx, img_path in enumerate(files):
            if not img_path.is_file() or img_path.suffix.lower() not in _IMAGE_EXTS:
                continue
            split = splits.get(idx, "train")
            export_name = export_names[idx]
            dest_img = output_path / "images" / split / export_name
            try:
                shutil.copy2(img_path, dest_img)
            except OSError as e:
                raise OSError(f"无法复制图片: {img_path}") from e
            frame_data = self._project.frame_annotations.get(idx, [])
            ann_dicts = []
            for raw_ann in frame_data:
                ann = self._project._coerce_annotation(raw_ann)
                if not ann.visible:
                    continue
                ann_dicts.append(
                    _annotation_export_dict(ann, class_id_map[ann.class_id])
                )
            label_path = (
                output_path / "labels" / split
                / f"{Path(export_name).stem}.txt"
            )
            report = exporter.export_yolo_task(
                ann_dicts, label_path, task, kpt_shape=kpt_shape
            )
            if not dest_img.is_file() or not label_path.is_file():
                raise OSError(f"无法写入 YOLO 导出: {export_name}")
            written += int(report.get("written", 0))
            skip_total.update(report.get("skipped") or {})
            count[split] = count.get(split, 0) + 1

        data_yaml = {
            "path": str(output_path.resolve()),
            "train": "images/train",
            "val": "images/val",
            "test": "images/test",
            "nc": len(sorted_class_ids),
            "names": [
                self._project.class_names.get(i, f"class_{i}")
                for i in sorted_class_ids
            ],
            "task": task,
        }
        if task == "pose" and kpt_shape:
            data_yaml["kpt_shape"] = list(kpt_shape)
            if self._project.kpt_names:
                data_yaml["kpt_names"] = list(self._project.kpt_names)
        with open(output_path / "data.yaml", "w", encoding="utf-8") as f:
            yaml.safe_dump(
                data_yaml, f, default_flow_style=False, sort_keys=False,
                allow_unicode=True,
            )
        if sum(v for k, v in count.items() if k in ("train", "val", "test")) == 0:
            QMessageBox.critical(
                parent_widget,
                "导出失败",
                "没有成功导出任何图片。请检查图片文件是否有效。",
            )
            return None
        count["written"] = written
        count["skipped"] = dict(skip_total)
        count["skip_message"] = format_export_skip_message(
            written, skip_total, _YOLO_TASK_LABELS.get(task, task)
        )
        return count

    def _get_or_configure_split(
        self, n: int, parent_widget=None
    ) -> Optional[Dict[int, str]]:
        """Always show split dialog so export UX is consistent.

        After accept, persist via set_split_map. Cancel → None.
        """
        from ..dialogs.split_config_dialog import SplitConfigDialog

        dialog = SplitConfigDialog(n, parent_widget)
        if dialog.exec() == QDialog.Accepted:
            split_map = dialog.compute_split(n)
            self._project.set_split_map(split_map)
            return split_map
        return None

    def configure_split(self, ratios: Dict[str, float]) -> None:
        import random
        from ..utils.split_utils import allocate_split_counts

        rng = random.Random(42)
        indices = list(range(len(self._project.image_files)))
        rng.shuffle(indices)
        n_train, n_val, _n_test = allocate_split_counts(
            len(indices),
            ratios.get("train", 0.8),
            ratios.get("val", 0.1),
            ratios.get("test", 0.1),
        )
        split_map: Dict[int, str] = {}
        for pos, idx in enumerate(indices):
            if pos < n_train:
                split_map[idx] = "train"
            elif pos < n_train + n_val:
                split_map[idx] = "val"
            else:
                split_map[idx] = "test"
        self._project.set_split_map(split_map)

    @staticmethod
    def _get_exporter(format_type: str):
        from ..utils.exporters import YOLOExporter, VOCExporter, COCOExporter, MaskExporter
        exporters = {"yolo": YOLOExporter, "voc": VOCExporter, "coco": COCOExporter, "mask": MaskExporter}
        if format_type not in exporters:
            raise ValueError(f"不支持的导出格式: {format_type}")
        return exporters[format_type]("")
