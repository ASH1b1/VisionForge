# src/gui/utils/importers/coco_importer.py
"""COCO JSON 格式导入器"""
from pathlib import Path
from typing import List, Dict, Tuple
import json
from .base_importer import BaseImporter, ImportResult
from src.gui.utils.exporters.coco_exporter import normalize_coco_keypoints
from src.gui.utils.yolo_label_format import default_kpt_names


class COCOImporter(BaseImporter):
    """COCO JSON 格式导入器

    支持结构:
      annotations/instances_train.json
      annotations/instances_val.json
      annotations/instances_test.json
      images/{train,val,test}/
    """

    IMAGE_EXTS = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif', '.webp'}

    def import_dataset(self, image_dir: Path, label_dir: Path) -> Tuple[
        List[Path], Dict[int, List[dict]], Dict[int, str], Dict[int, str], ImportResult
    ]:
        self._next_ann_id = 1  # 每次导入重置 ID 计数器
        result = ImportResult()
        class_names: Dict[int, str] = {}
        name_to_id: Dict[str, int] = {}
        image_files: List[Path] = []
        frame_annotations: Dict[int, List[dict]] = {}
        split_map: Dict[int, str] = {}

        image_dir = Path(image_dir)
        label_dir = Path(label_dir)

        # Step 1: 收集 JSON 文件
        json_files = list(label_dir.rglob("instances_*.json"))
        if not json_files:
            json_files = list(label_dir.rglob("*.json"))

        if not json_files:
            result.parse_errors.append("未找到 COCO JSON 文件")
            return image_files, frame_annotations, class_names, split_map, result

        # Step 2: 建立图片索引（basename + 相对路径；同名冲突时去掉 basename 避免静默覆盖）
        filename_index: Dict[str, Path] = {}
        _basename_collisions = set()
        for ext in self.IMAGE_EXTS:
            for img_path in image_dir.rglob(f"*{ext}"):
                try:
                    rel = str(img_path.relative_to(image_dir)).replace("\\", "/").lower()
                    filename_index[rel] = img_path
                except ValueError:
                    pass
                key = img_path.name.lower()
                if key in _basename_collisions:
                    continue
                if key in filename_index and filename_index[key].resolve() != img_path.resolve():
                    _basename_collisions.add(key)
                    if filename_index[key].name.lower() == key:
                        del filename_index[key]
                else:
                    filename_index[key] = img_path

        # Step 3: 逐个 JSON 文件解析
        for json_path in sorted(json_files):
            split_name = self._infer_split(json_path)
            self._parse_coco_json(json_path, split_name,
                                  image_files, frame_annotations, split_map,
                                  class_names, name_to_id,
                                  filename_index, result)

        # Step 4: 排序图片（保留排序前索引映射）
        old_order = list(image_files)  # 排序前的顺序
        image_files.sort(key=lambda p: p.name.lower())

        # Step 5: 按排序后索引重建 frame_annotations
        img_to_new_idx = {str(img.resolve()): idx for idx, img in enumerate(image_files)}
        new_annotations: Dict[int, List[dict]] = {}
        new_split_map: Dict[int, str] = {}

        for old_idx, anns in frame_annotations.items():
            if old_idx >= len(old_order):
                continue
            img_path = old_order[old_idx]
            new_idx = img_to_new_idx[str(img_path.resolve())]
            new_annotations[new_idx] = anns
            if old_idx in split_map:
                new_split_map[new_idx] = split_map[old_idx]

        # Step 6: 填充 result
        result.total_images = len(image_files)
        result.total_annotations = sum(len(a) for a in new_annotations.values())
        result.matched_pairs = sum(1 for a in new_annotations.values() if a)
        result.unmatched_images = sum(1 for i in range(len(image_files)) if not new_annotations.get(i))
        result.class_names = class_names
        result.split_map = new_split_map

        return image_files, new_annotations, class_names, new_split_map, result

    def _infer_split(self, json_path: Path) -> str:
        """从 JSON 文件名推断 split 名称"""
        name = json_path.stem.lower()
        for split in ("train", "val", "test"):
            if split in name:
                return split
        return "train"

    def _parse_coco_json(self, json_path: Path, split_name: str,
                         image_files: List[Path],
                         frame_annotations: Dict[int, List[dict]],
                         split_map: Dict[int, str],
                         class_names: Dict[int, str],
                         name_to_id: Dict[str, int],
                         filename_index: Dict[str, Path],
                         result: ImportResult):
        """解析单个 COCO JSON 文件"""
        try:
            # Guard against huge JSON (local DoS)
            if json_path.stat().st_size > 512 * 1024 * 1024:
                result.parse_errors.append(
                    f"{json_path.name}: JSON 文件过大 (>{512}MB)，已跳过"
                )
                return
            with open(json_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
        except Exception as e:
            result.parse_errors.append(f"{json_path.name}: JSON 解析失败: {e}")
            return

        # 解析 categories
        cat_id_to_internal: Dict[int, int] = {}
        for cat in data.get('categories', []):
            cat_name = cat.get('name', f"class_{cat['id']}")
            internal_id = self._register_class(cat_name, class_names, name_to_id)
            cat_id_to_internal[cat['id']] = internal_id
            kpts = cat.get("keypoints")
            if result.kpt_names is None and isinstance(kpts, list) and kpts:
                result.kpt_names = [str(n) for n in kpts]
                result.kpt_shape = (len(result.kpt_names), 3)
                result.kpt_shape_source = "coco"

        # 解析 images
        images_info: Dict[int, dict] = {}
        for img in data.get('images', []):
            images_info[img['id']] = img

        # 匹配图片 + 分配帧索引
        coco_img_to_frame: Dict[int, int] = {}
        for img_id, img_info in images_info.items():
            filename = img_info.get('file_name', '')
            key = filename.replace("\\", "/").lower()
            img_path = filename_index.get(key)
            if img_path is None:
                img_path = filename_index.get(Path(filename).name.lower())
            if img_path is None:
                result.unmatched_labels += 1
                result.parse_errors.append(f"{json_path.name}: 未找到图片 '{filename}'")
                continue

            key = str(img_path.resolve())
            existing = [i for i, p in enumerate(image_files) if str(p.resolve()) == key]
            if existing:
                frame_idx = existing[0]
            else:
                frame_idx = len(image_files)
                image_files.append(img_path)
                frame_annotations[frame_idx] = []
                split_map[frame_idx] = split_name

            coco_img_to_frame[img_id] = frame_idx

        # 解析 annotations
        img_sizes: Dict[int, Tuple[int, int]] = {}
        for ann in data.get('annotations', []):
            image_id = ann.get('image_id')
            if image_id not in coco_img_to_frame:
                continue

            frame_idx = coco_img_to_frame[image_id]
            if 'category_id' not in ann:
                result.parse_errors.append(
                    f"{json_path.name}: annotation id={ann.get('id', '?')} "
                    f"缺少 category_id，已跳过"
                )
                continue
            category_id = ann['category_id']
            if category_id not in cat_id_to_internal:
                result.parse_errors.append(
                    f"{json_path.name}: annotation id={ann.get('id', '?')} "
                    f"未知 category_id={category_id}，已跳过"
                )
                continue
            internal_class_id = cat_id_to_internal[category_id]

            # 获取图片尺寸 — 缺少有效 width/height 时绝不伪造 1920×1080
            if image_id not in img_sizes:
                img_info = images_info.get(image_id, {})
                w = img_info.get("width")
                h = img_info.get("height")
                try:
                    w_i = int(w) if w is not None else 0
                    h_i = int(h) if h is not None else 0
                except (TypeError, ValueError):
                    w_i, h_i = 0, 0
                if w_i <= 0 or h_i <= 0:
                    img_sizes[image_id] = None
                    result.parse_errors.append(
                        f"{json_path.name}: image id={image_id} "
                        f"缺少有效 width/height，相关标注已跳过"
                    )
                else:
                    img_sizes[image_id] = (w_i, h_i)
            size = img_sizes.get(image_id)
            if size is None:
                continue
            img_w, img_h = size

            # 转换 bbox
            coco_bbox = ann.get('bbox', [0, 0, 0, 0])
            bbox = self._coco_to_yolo_normalized(coco_bbox, img_w, img_h)

            # 转换 segmentation：RLE → 报错；多 polygon → 每环一条标注
            seg = ann.get('segmentation')
            polygons: List = []
            if isinstance(seg, dict):
                result.parse_errors.append(
                    f"{json_path.name}: annotation id={ann.get('id', '?')} "
                    f"RLE segmentation unsupported，已跳过多边形（保留 bbox）"
                )
            elif isinstance(seg, list) and seg:
                if isinstance(seg[0], (int, float)):
                    # 扁平 [x1,y1,...] 视为单环
                    poly = self._normalize_polygon(seg, img_w, img_h)
                    if poly:
                        polygons.append(poly)
                else:
                    for ring in seg:
                        if not isinstance(ring, list):
                            result.parse_errors.append(
                                f"{json_path.name}: annotation id={ann.get('id', '?')} "
                                f"非法 segmentation 环，已跳过该环"
                            )
                            continue
                        poly = self._normalize_polygon(ring, img_w, img_h)
                        if poly:
                            polygons.append(poly)

            if polygons:
                # Policy: one annotation per polygon ring (app stores single polygon).
                if isinstance(ann.get("keypoints"), list) and len(ann.get("keypoints") or []) >= 2:
                    pose_ann = self._pose_or_fallback(
                        ann, internal_class_id, bbox, polygons[0], img_w, img_h, result, json_path
                    )
                    frame_annotations[frame_idx].append(pose_ann)
                    for i, polygon in enumerate(polygons[1:], start=1):
                        ring_bbox = self._bbox_from_polygon(polygon)
                        frame_annotations[frame_idx].append(
                            self._make_annotation(internal_class_id, ring_bbox, polygon)
                        )
                else:
                    for i, polygon in enumerate(polygons):
                        ring_bbox = bbox if i == 0 else self._bbox_from_polygon(polygon)
                        frame_annotations[frame_idx].append(
                            self._make_annotation(internal_class_id, ring_bbox, polygon)
                        )
            else:
                frame_annotations[frame_idx].append(
                    self._pose_or_fallback(
                        ann, internal_class_id, bbox, None, img_w, img_h, result, json_path
                    )
                )

    def _pose_or_fallback(
        self,
        coco_ann: dict,
        class_id: int,
        bbox,
        polygon,
        img_w: int,
        img_h: int,
        result: ImportResult,
        json_path: Path,
    ) -> dict:
        raw = coco_ann.get("keypoints")
        if not isinstance(raw, list) or len(raw) < 2:
            return self._make_annotation(class_id, bbox, polygon)
        if result.kpt_shape is None:
            if len(raw) % 3 == 0 and len(raw) >= 3:
                k = len(raw) // 3
                result.kpt_shape = (k, 3)
                result.kpt_shape_source = "inferred"
                if not result.kpt_names:
                    result.kpt_names = default_kpt_names(k)
            else:
                result.parse_errors.append(
                    f"{json_path.name}: annotation id={coco_ann.get('id', '?')} "
                    f"keypoints 长度无法推断 K，已降为框"
                )
                return self._make_annotation(class_id, bbox, polygon)
        k = result.kpt_shape[0]
        if len(raw) != 3 * k:
            result.warnings.append(
                f"{json_path.name}: annotation id={coco_ann.get('id', '?')} "
                f"keypoints 长度 {len(raw)} ≠ {3 * k}，已跳过关键点"
            )
            return self._make_annotation(class_id, bbox, polygon)
        keypoints = normalize_coco_keypoints(raw, img_w, img_h)
        return self._make_annotation(
            class_id, bbox, polygon, kind="pose", keypoints=keypoints
        )

    @staticmethod
    def _bbox_from_polygon(
        polygon: List[Tuple[float, float]],
    ) -> Tuple[float, float, float, float]:
        """Normalized polygon → YOLO (cx, cy, w, h)."""
        xs = [p[0] for p in polygon]
        ys = [p[1] for p in polygon]
        xmin, xmax = min(xs), max(xs)
        ymin, ymax = min(ys), max(ys)
        return (
            (xmin + xmax) / 2,
            (ymin + ymax) / 2,
            xmax - xmin,
            ymax - ymin,
        )

    def export(self, annotations, image_path, output_path):
        raise NotImplementedError

    def export_batch(self, data, output_dir):
        raise NotImplementedError
