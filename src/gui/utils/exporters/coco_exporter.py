"""COCO JSON 格式导出器 - 支持 train/val/test 划分"""
from pathlib import Path
from typing import List, Dict, Tuple
from datetime import datetime
import json
import shutil
from .base_exporter import BaseExporter
from src.product_config import get_product_config
from src.gui.utils.exporters.yolo_exporter import annotation_kind
from src.gui.utils.yolo_label_format import default_kpt_names


def normalize_coco_keypoints(
    raw: List[float], img_w: int, img_h: int
) -> list[tuple[float, float, int]]:
    points: list[tuple[float, float, int]] = []
    for i in range(0, len(raw) - 2, 3):
        x = float(raw[i]) / img_w if img_w else 0.0
        y = float(raw[i + 1]) / img_h if img_h else 0.0
        v = int(raw[i + 2])
        if v not in (0, 1, 2):
            v = 0
        points.append((x, y, v))
    return points


def denormalize_coco_keypoints(
    keypoints: List[tuple], img_w: int, img_h: int
) -> tuple[list[float], int]:
    flat: list[float] = []
    num = 0
    for point in keypoints:
        x, y = float(point[0]), float(point[1])
        v = int(point[2]) if len(point) >= 3 else 2
        if v not in (0, 1, 2):
            v = 0
        if v == 0:
            flat.extend([0, 0, 0])
        else:
            flat.extend([round(x * img_w, 2), round(y * img_h, 2), v])
            num += 1
    return flat, num


class COCOExporter(BaseExporter):
    """COCO JSON 格式导出器

    支持 train/val/test 划分目录结构:
        annotations/instances_train.json
        annotations/instances_val.json
        annotations/instances_test.json
        images/train/*.jpg
        images/val/*.jpg
        images/test/*.jpg
    """

    def __init__(self, output_dir: str | Path):
        super().__init__(output_dir)
        self.kpt_shape: tuple[int, int] | None = None
        self.kpt_names: list[str] | None = None
        self.skipped_kinds: dict[str, int] = {}

    def export(
        self,
        annotations: List[Dict],
        image_path: Path,
        output_path: Path,
        image_id: int = 1,
        annotation_id_start: int = 1
    ) -> Tuple[bool, int]:
        """导出单个标注为 COCO 格式（用于累积）"""
        ok, ann_count, _, _ = self._create_coco_json(
            [(image_path, annotations)],
            output_path,
            [image_path]
        )
        return ok, ann_count

    def _build_coco_structure(
        self,
        data: List[Tuple[Path, List[Dict]]],
        image_paths: List[Path] = None,
        export_names: List[str] = None,
    ) -> Tuple[List[Dict], List[Dict], Dict[int, int], List[int]]:
        """构建 COCO images + annotations 列表 + 类别映射。

        Images whose size cannot be read are skipped (no invented dimensions).
        Returns (images, annotations, category_id_map, included_source_indices).
        """
        categories = []
        category_id_map = {}
        seen_classes = set()

        images = []
        annotations_list = []
        included_indices: List[int] = []
        ann_id = 1
        skipped: dict[str, int] = {}

        pose_present = any(
            annotation_kind(ann) == "pose"
            for _image_path, annotations in data
            for ann in annotations
            if ann.get("visible", True)
        )
        kpt_names = list(self.kpt_names) if self.kpt_names else None
        if pose_present and not kpt_names:
            k = self.kpt_shape[0] if self.kpt_shape else 0
            if not k:
                for _image_path, annotations in data:
                    for ann in annotations:
                        kpts = ann.get("keypoints") or []
                        if kpts:
                            k = len(kpts)
                            break
                    if k:
                        break
            kpt_names = default_kpt_names(k) if k else None
        if kpt_names and not self.kpt_names:
            self.kpt_names = list(kpt_names)

        for idx, (image_path, annotations) in enumerate(data):
            src_path = (
                image_paths[idx]
                if image_paths and idx < len(image_paths)
                else image_path
            )
            try:
                w, h = self.read_image_size(src_path)
            except Exception as e:
                print(f"[COCOExporter] Skipping {src_path}: cannot read size ({e})")
                continue

            for ann in annotations:
                if not ann.get('visible', True):
                    continue
                class_id = ann['class_id']
                if class_id not in seen_classes:
                    seen_classes.add(class_id)
                    class_name = self.get_class_name(class_id)
                    cat_id = len(categories) + 1
                    category_id_map[class_id] = cat_id
                    cat = {
                        "id": cat_id,
                        "name": class_name,
                        "supercategory": "object",
                    }
                    if kpt_names:
                        cat["keypoints"] = list(kpt_names)
                    categories.append(cat)

            image_id = len(images) + 1
            images.append({
                "id": image_id,
                "file_name": (
                    export_names[idx]
                    if export_names and idx < len(export_names)
                    else Path(image_path).name
                ),
                "width": w,
                "height": h,
                "license": 1,
                "date_captured": ""
            })
            included_indices.append(idx)

            for ann in annotations:
                if not ann.get('visible', True):
                    continue
                kind = annotation_kind(ann)
                if kind == "obb":
                    skipped["obb"] = skipped.get("obb", 0) + 1
                    continue

                class_id = ann['class_id']
                bbox = ann['bbox']
                polygon = ann.get('polygon') or []

                x1, y1, x2, y2 = self.yolo_to_corners(bbox)
                x = x1 * w
                y = y1 * h
                box_w = (x2 - x1) * w
                box_h = (y2 - y1) * h

                area = box_w * box_h
                segmentation = []
                if polygon:
                    segmentation = [[
                        coord
                        for point in polygon
                        for coord in (round(point[0] * w, 2), round(point[1] * h, 2))
                    ]]
                entry = {
                    "id": ann_id,
                    "image_id": image_id,
                    "category_id": category_id_map.get(class_id, 1),
                    "bbox": [round(x, 2), round(y, 2), round(box_w, 2), round(box_h, 2)],
                    "area": round(area, 2),
                    "iscrowd": 0,
                    "segmentation": segmentation,
                }
                if kind == "pose":
                    kpts = ann.get("keypoints") or []
                    flat, num = denormalize_coco_keypoints(kpts, w, h)
                    entry["keypoints"] = flat
                    entry["num_keypoints"] = num
                annotations_list.append(entry)
                ann_id += 1

        self.skipped_kinds = skipped
        return images, annotations_list, category_id_map, included_indices

    def _create_coco_json(
        self,
        data: List[Tuple[Path, List[Dict]]],
        output_path: Path,
        image_paths: List[Path] = None,
        export_names: List[str] = None,
    ) -> Tuple[bool, int, int, List[int]]:
        """创建完整的 COCO JSON。

        Returns:
            (write_ok, annotation_count, included_image_count, included_source_indices)
        """
        try:
            info = {
                "year": datetime.now().year,
                "version": "1.0",
                "description": f"Exported from {get_product_config().export_tool_name}",
                "date_created": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            }
            licenses = [{"id": 1, "name": "Attribution License", "url": ""}]

            images, annotations_list, category_id_map, included = self._build_coco_structure(
                data, image_paths, export_names
            )

            categories = []
            id_to_name = {}
            for _image_path, annotations in data:
                for ann in annotations:
                    if not ann.get('visible', True):
                        continue
                    cid = ann['class_id']
                    if cid in category_id_map:
                        id_to_name[category_id_map[cid]] = self.get_class_name(cid)
            for cat_id in sorted(id_to_name):
                cat = {
                    "id": cat_id,
                    "name": id_to_name[cat_id],
                    "supercategory": "object",
                }
                if self.kpt_names:
                    cat["keypoints"] = list(self.kpt_names)
                elif self.kpt_shape:
                    cat["keypoints"] = default_kpt_names(self.kpt_shape[0])
                categories.append(cat)

            if data and not images:
                print("[COCOExporter] No images could be sized; refusing to invent dimensions")
                return False, 0, 0, []

            coco_json = {
                "info": info,
                "licenses": licenses,
                "categories": categories,
                "images": images,
                "annotations": annotations_list
            }

            output_path = Path(output_path)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(coco_json, f, indent=2, ensure_ascii=False)

            return True, len(annotations_list), len(included), included
        except Exception as e:
            print(f"[COCOExporter] Export error: {e}")
            return False, 0, 0, []

    def export_batch(
        self,
        data: List[Tuple[Path, List[Dict]]],
        output_dir: Path,
        image_paths: List[Path] = None,
        split_map: Dict[int, str] = None
    ) -> Tuple[int, int, Dict[str, int]]:
        """批量导出 COCO JSON 格式

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

        split_data: Dict[str, List[Tuple[Path, List[Dict]]]] = {
            "train": [], "val": [], "test": []
        }
        split_image_paths: Dict[str, List[Path]] = {
            "train": [], "val": [], "test": []
        }
        split_export_names: Dict[str, List[str]] = {
            "train": [], "val": [], "test": []
        }
        source_paths = image_paths or [item[0] for item in data]
        export_names = self.unique_image_names(source_paths)
        for idx, item in enumerate(data):
            split = split_map.get(idx, "train")
            split_data[split].append(item)
            if image_paths and idx < len(image_paths):
                split_image_paths[split].append(image_paths[idx])
            else:
                split_image_paths[split].append(item[0])
            split_export_names[split].append(export_names[idx])

        split_counts = {"train": 0, "val": 0, "test": 0}
        total_success = 0

        for split, items in split_data.items():
            if not items:
                continue

            annot_dir = output_dir / "annotations"
            annot_dir.mkdir(parents=True, exist_ok=True)
            json_path = annot_dir / f"instances_{split}.json"

            ok, _count, n_images, included = self._create_coco_json(
                items,
                json_path,
                split_image_paths[split],
                split_export_names[split],
            )
            if not ok:
                continue

            total_success += n_images
            split_counts[split] = n_images

            img_dir = output_dir / "images" / split
            img_dir.mkdir(parents=True, exist_ok=True)
            for local_idx in included:
                img_path = split_image_paths[split][local_idx]
                export_name = split_export_names[split][local_idx]
                dest = img_dir / export_name
                if not dest.exists():
                    shutil.copy2(img_path, dest)

        return total_success, len(data), split_counts

    def _export_flat(
        self,
        data: List[Tuple[Path, List[Dict]]],
        output_dir: Path,
        image_paths: List[Path] = None
    ) -> Tuple[int, Dict[str, int]]:
        """无划分时：扁平导出"""
        annot_dir = output_dir / "annotations"
        annot_dir.mkdir(parents=True, exist_ok=True)
        json_path = annot_dir / "instances.json"
        source_paths = image_paths or [item[0] for item in data]
        export_names = self.unique_image_names(source_paths)
        ok, _ann_count, n_images, included = self._create_coco_json(
            data, json_path, source_paths, export_names
        )
        if not ok:
            return 0, {"train": 0}

        if source_paths:
            img_dir = output_dir / "images"
            img_dir.mkdir(parents=True, exist_ok=True)
            for local_idx in included:
                img_path = source_paths[local_idx]
                export_name = export_names[local_idx]
                dest = img_dir / export_name
                if not dest.exists():
                    shutil.copy2(img_path, dest)

        return n_images, {"train": n_images}

    def export_current_frame(
        self,
        annotations: List[Dict],
        image_path: Path,
        output_dir: Path
    ) -> Path:
        """导出当前帧标注到指定目录（扁平结构）"""
        output_dir = Path(output_dir)
        annot_dir = output_dir / "annotations"
        annot_dir.mkdir(parents=True, exist_ok=True)
        img_dir = output_dir / "images"
        img_dir.mkdir(parents=True, exist_ok=True)

        json_path = annot_dir / f"instances_{image_path.stem}.json"
        ok, _ann_count, n_images, _ = self._create_coco_json(
            [(image_path, annotations)], json_path, [image_path]
        )
        if not ok or n_images == 0:
            raise OSError(f"无法导出 COCO 标注: {image_path}")

        dest_img = img_dir / image_path.name
        if not dest_img.exists():
            shutil.copy2(image_path, dest_img)

        return json_path

    def _generate_categories_file(self, output_dir: Path):
        """生成 categories.json 文件"""
        categories_path = output_dir / 'categories.json'
        class_names = self.get_all_class_names()

        categories = []
        for idx, name in enumerate(class_names):
            categories.append({
                "id": idx + 1,
                "name": name,
                "supercategory": "object"
            })

        with open(categories_path, 'w', encoding='utf-8') as f:
            json.dump({"categories": categories}, f, indent=2, ensure_ascii=False)
