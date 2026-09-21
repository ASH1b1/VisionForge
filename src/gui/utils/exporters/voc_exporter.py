"""VOC XML 格式导出器 - 支持 train/val/test 划分"""
from pathlib import Path
from typing import List, Dict, Tuple
import shutil
import xml.etree.ElementTree as ET
from .base_exporter import BaseExporter


class VOCExporter(BaseExporter):
    """Pascal VOC XML 格式导出器

    输出格式: PASCAL VOC XML
    支持 train/val/test 划分目录结构:
        Annotations/train/*.xml
        Annotations/val/*.xml
        Annotations/test/*.xml
        JPEGImages/train/*.jpg
        JPEGImages/val/*.jpg
        JPEGImages/test/*.jpg
    """

    def __init__(self, output_dir: str | Path):
        super().__init__(output_dir)

    def export(
        self,
        annotations: List[Dict],
        image_path: Path,
        output_path: Path,
        image_size: Tuple[int, int] = None,
        *,
        voc_folder: str | None = None,
        voc_path: str | None = None,
    ) -> bool:
        """导出单个标注为 VOC XML 格式

        Args:
            voc_folder: `<folder>` 文本。批量导出应写成与 JPEGImages 布局一致的
                相对目录（如 `train`），扁平导出传 `""` 省略；`None` 时回退为
                `image_path.parent.name`（兼容单文件调用）。
            voc_path: 可选 `<path>`，指向导出后的图片路径，便于再导入匹配。
        """
        try:
            if image_size is None:
                try:
                    width, height = self.read_image_size(image_path)
                except Exception as e:
                    print(f"[VOCExporter] Cannot read image size for {image_path}: {e}")
                    return False
            else:
                width, height = image_size

            root = ET.Element("annotation")
            folder_text = (
                image_path.parent.name if voc_folder is None else voc_folder
            )
            if folder_text:
                ET.SubElement(root, "folder").text = folder_text
            ET.SubElement(root, "filename").text = image_path.name
            if voc_path:
                ET.SubElement(root, "path").text = str(voc_path)

            source = ET.SubElement(root, "source")
            ET.SubElement(source, "database").text = "GS Annotation"

            owner = ET.SubElement(root, "owner")
            from src.product_config import get_product_config
            ET.SubElement(owner, "name").text = get_product_config().export_tool_name

            size_el = ET.SubElement(root, "size")
            ET.SubElement(size_el, "width").text = str(width)
            ET.SubElement(size_el, "height").text = str(height)
            ET.SubElement(size_el, "depth").text = "3"

            ET.SubElement(root, "segmented").text = "0"

            for ann in annotations:
                if not ann.get('visible', True):
                    continue

                class_name = self.get_class_name(ann['class_id'])
                bbox = ann['bbox']

                x1, y1, x2, y2 = self.yolo_to_corners(bbox)

                xmin = int(x1 * width)
                ymin = int(y1 * height)
                xmax = int(x2 * width)
                ymax = int(y2 * height)

                xmin = max(0, min(xmin, width - 1))
                ymin = max(0, min(ymin, height - 1))
                xmax = max(xmin + 1, min(xmax, width))
                ymax = max(ymin + 1, min(ymax, height))

                obj = ET.SubElement(root, "object")
                ET.SubElement(obj, "name").text = class_name
                ET.SubElement(obj, "pose").text = "Unspecified"
                ET.SubElement(obj, "truncated").text = "0"
                ET.SubElement(obj, "difficult").text = "0"
                bndbox = ET.SubElement(obj, "bndbox")
                ET.SubElement(bndbox, "xmin").text = str(xmin)
                ET.SubElement(bndbox, "ymin").text = str(ymin)
                ET.SubElement(bndbox, "xmax").text = str(xmax)
                ET.SubElement(bndbox, "ymax").text = str(ymax)

            output_path = Path(output_path)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            ET.ElementTree(root).write(
                output_path, encoding="utf-8", xml_declaration=True
            )
            return True
        except Exception as e:
            print(f"[VOCExporter] Export error: {e}")
            return False

    def export_batch(
        self,
        data: List[Tuple[Path, List[Dict]]],
        output_dir: Path,
        image_paths: List[Path] = None,
        split_map: Dict[int, str] = None
    ) -> Tuple[int, int, Dict[str, int]]:
        """批量导出 VOC XML 格式

        Args:
            data: [(img_path, annotations), ...]
            output_dir: 输出根目录
            image_paths: 图片路径列表（用于复制图片）
            split_map: {frame_index: 'train'/'val'/'test'}，若有则按划分输出

        Returns:
            (成功数, 总数, 各 split 计数)
        """
        output_dir = Path(output_dir)
        split_map = split_map or {}

        if not split_map:
            success, split_counts = self._export_flat(data, output_dir, image_paths)
            return success, len(data), split_counts

        split_counts = {"train": 0, "val": 0, "test": 0}
        success = 0
        source_paths = image_paths or [item[0] for item in data]
        export_names = self.unique_image_names(source_paths)

        for idx, (image_path, annotations) in enumerate(data):
            split = split_map.get(idx, "train")
            export_name = export_names[idx]

            annot_dir = output_dir / "Annotations" / split
            annot_dir.mkdir(parents=True, exist_ok=True)
            xml_path = annot_dir / f"{Path(export_name).stem}.xml"

            src_img = (
                image_paths[idx]
                if image_paths and idx < len(image_paths)
                else image_path
            )
            try:
                image_size = self.read_image_size(src_img)
            except Exception as e:
                print(f"[VOCExporter] Skipping {src_img}: cannot read size ({e})")
                continue

            alias_path = image_path.with_name(export_name)
            img_dir = output_dir / "JPEGImages" / split
            img_dir.mkdir(parents=True, exist_ok=True)
            dest_img = img_dir / export_name
            if not self.export(
                annotations,
                alias_path,
                xml_path,
                image_size,
                voc_folder=split,
                voc_path=str(dest_img.resolve()),
            ):
                continue

            success += 1
            split_counts[split] += 1

            if image_paths and idx < len(image_paths):
                img_path = image_paths[idx]
                if not dest_img.exists():
                    shutil.copy2(img_path, dest_img)

        self._generate_classes_file(output_dir)
        return success, len(data), split_counts

    def _export_flat(
        self,
        data: List[Tuple[Path, List[Dict]]],
        output_dir: Path,
        image_paths: List[Path] = None
    ) -> Tuple[int, Dict[str, int]]:
        """无划分时：扁平导出到 Annotations/ 和 JPEGImages/"""
        annot_dir = output_dir / "Annotations"
        annot_dir.mkdir(parents=True, exist_ok=True)
        source_paths = image_paths or [item[0] for item in data]
        export_names = self.unique_image_names(source_paths)
        success = 0

        for idx, (image_path, annotations) in enumerate(data):
            export_name = export_names[idx]
            xml_path = annot_dir / f"{Path(export_name).stem}.xml"

            src_img = (
                image_paths[idx]
                if image_paths and idx < len(image_paths)
                else image_path
            )
            try:
                image_size = self.read_image_size(src_img)
            except Exception as e:
                print(f"[VOCExporter] Skipping {src_img}: cannot read size ({e})")
                continue

            img_dir = output_dir / "JPEGImages"
            img_dir.mkdir(parents=True, exist_ok=True)
            dest_img = img_dir / export_name
            if not self.export(
                annotations,
                image_path.with_name(export_name),
                xml_path,
                image_size,
                voc_folder="",
                voc_path=str(dest_img.resolve()),
            ):
                continue

            success += 1
            if image_paths and idx < len(image_paths):
                img_path = image_paths[idx]
                if not dest_img.exists():
                    shutil.copy2(img_path, dest_img)

        self._generate_classes_file(output_dir)
        return success, {"train": success}

    def export_current_frame(
        self,
        annotations: List[Dict],
        image_path: Path,
        output_dir: Path
    ) -> Path:
        """导出当前帧标注到指定目录（扁平结构）"""
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        annot_dir = output_dir / "Annotations"
        annot_dir.mkdir(parents=True, exist_ok=True)

        xml_path = annot_dir / f"{image_path.stem}.xml"
        img_dir = output_dir / "JPEGImages"
        img_dir.mkdir(parents=True, exist_ok=True)
        dest_img = img_dir / image_path.name
        if not self.export(
            annotations,
            image_path,
            xml_path,
            voc_folder="",
            voc_path=str(dest_img.resolve()),
        ):
            raise OSError(f"无法导出 VOC 标注: {image_path}")

        if not dest_img.exists():
            shutil.copy2(image_path, dest_img)

        self._generate_classes_file(output_dir)
        return xml_path

    def _generate_classes_file(self, output_dir: Path):
        """生成 classes.txt 文件"""
        classes_path = output_dir / "classes.txt"
        class_names = self.get_all_class_names()

        with open(classes_path, 'w', encoding='utf-8') as f:
            for name in class_names:
                f.write(f"{name}\n")
