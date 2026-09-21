# src/gui/utils/importers/voc_importer.py
"""VOC XML 格式导入器"""
from pathlib import Path
from typing import List, Dict, Tuple
import xml.etree.ElementTree as ET
from .base_importer import BaseImporter, ImportResult


class VOCImporter(BaseImporter):
    """Pascal VOC XML 格式导入器

    支持结构:
      - 扁平: Annotations/ + JPEGImages/
      - 划分: Annotations/{train,val,test}/ + JPEGImages/{train,val,test}/
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

        # Step 1: 收集所有 XML 标注文件
        xml_files = self._collect_xml_files(label_dir)

        if not xml_files:
            result.parse_errors.append("未找到 XML 标注文件")
            return image_files, frame_annotations, class_names, split_map, result

        # Step 2: 检测 split 目录结构
        split_patterns = self._detect_voc_splits(label_dir)

        # Step 3: 建立图片索引（resolve + 相对路径与 basename 分表，避免互相覆盖）
        resolved_index: Dict[str, Path] = {}
        rel_index: Dict[str, Path] = {}
        basename_index: Dict[str, Path] = {}
        basename_collisions: set = set()
        for ext in self.IMAGE_EXTS:
            for img_path in image_dir.rglob(f"*{ext}"):
                resolved_index[str(img_path.resolve())] = img_path
                try:
                    rel = str(img_path.relative_to(image_dir)).replace("\\", "/").lower()
                    rel_index[rel] = img_path
                except ValueError:
                    pass
                base = img_path.name.lower()
                if base in basename_collisions:
                    continue
                if base in basename_index and basename_index[base].resolve() != img_path.resolve():
                    basename_collisions.add(base)
                    del basename_index[base]
                else:
                    basename_index[base] = img_path

        # Step 4: 匹配 XML → Image
        matched_xmls = []
        for xml_path in xml_files:
            try:
                if xml_path.stat().st_size > 50 * 1024 * 1024:
                    result.parse_errors.append(
                        f"{xml_path.name}: XML 文件过大 (>50MB)，已跳过"
                    )
                    continue
                tree = ET.parse(str(xml_path))
                root = tree.getroot()
                filename_el = root.find('filename')
                if filename_el is None or not filename_el.text:
                    result.parse_errors.append(f"{xml_path.name}: 缺少 <filename>")
                    continue
                filename = filename_el.text.strip()
                img_path = self._resolve_image_for_xml(
                    root, filename, xml_path,
                    resolved_index, rel_index, basename_index, basename_collisions, result,
                )
                if img_path is None:
                    continue
                matched_xmls.append((xml_path, img_path))
            except Exception as e:
                result.parse_errors.append(f"{xml_path.name}: XML 解析失败: {e}")

        # Step 5: 按图片路径去重排序
        unique_images: Dict[str, Tuple[Path, List[Path]]] = {}
        for xml_path, img_path in matched_xmls:
            key = str(img_path.resolve())
            if key not in unique_images:
                unique_images[key] = (img_path, [])
            unique_images[key][1].append(xml_path)

        image_files = sorted([img for img, _ in unique_images.values()],
                             key=lambda p: p.name.lower())

        # Step 6: 解析标注
        total_anns = 0
        for frame_idx, img_path in enumerate(image_files):
            key = str(img_path.resolve())
            _, xml_paths = unique_images[key]
            anns = []
            for xml_path in xml_paths:
                parsed = self._parse_voc_xml(xml_path, class_names, name_to_id, result)
                anns.extend(parsed)
                total_anns += len(parsed)
            frame_annotations[frame_idx] = anns

            # 推断 split
            if split_patterns:
                for split_name, pattern in split_patterns.items():
                    if pattern in str(xml_paths[0].parent):
                        split_map[frame_idx] = split_name
                        break

        # Step 7: 填充 result
        result.total_images = len(image_files)
        result.total_annotations = total_anns
        result.matched_pairs = sum(1 for anns in frame_annotations.values() if anns)
        result.unmatched_images = sum(1 for anns in frame_annotations.values() if not anns)
        result.class_names = class_names
        result.split_map = split_map

        return image_files, frame_annotations, class_names, split_map, result

    def _resolve_image_for_xml(
        self,
        root: ET.Element,
        filename: str,
        xml_path: Path,
        resolved_index: Dict[str, Path],
        rel_index: Dict[str, Path],
        basename_index: Dict[str, Path],
        basename_collisions: set,
        result: ImportResult,
    ):
        """Match XML to an image without silent basename last-wins.

        Prefer <path>, then Annotations/{split} layout, then folder/relative keys;
        orphan source-folder names may unique-basename fallback; colliding
        basenames never silent last-wins.
        """
        # 1) Absolute / relative <path> element
        path_el = root.find('path')
        if path_el is not None and path_el.text and path_el.text.strip():
            raw = path_el.text.strip()
            candidate = Path(raw)
            if not candidate.is_absolute():
                candidate = xml_path.parent / candidate
            try:
                key = str(candidate.resolve())
            except OSError:
                key = str(candidate)
            if key in resolved_index:
                return resolved_index[key]
            try:
                rel = str(Path(raw).as_posix()).replace("\\", "/").lower()
                if rel in rel_index:
                    return rel_index[rel]
            except Exception:
                pass
            result.unmatched_labels += 1
            result.parse_errors.append(
                f"{xml_path.name}: <path> 未匹配到图片 '{raw}'"
            )
            return None

        fname_norm = filename.replace("\\", "/").strip()
        fname_base = Path(fname_norm).name

        # 2) filename already contains relative path segments
        if "/" in fname_norm:
            hit = rel_index.get(fname_norm.lower())
            if hit is not None:
                return hit
            result.unmatched_labels += 1
            result.parse_errors.append(
                f"{xml_path.name}: 未找到匹配图片 '{filename}'"
            )
            return None

        # 3) Infer split from XML layout: Annotations/{train,val,test}/*.xml
        #    (fixes export→import when <folder> still holds the source folder name)
        xml_parent = xml_path.parent.name.lower()
        if xml_parent in ("train", "val", "test"):
            split_key = f"{xml_parent}/{fname_base.lower()}"
            hit = rel_index.get(split_key)
            if hit is not None:
                return hit

        # 4) <folder> + basename → relative under image_dir
        folder_el = root.find('folder')
        folder = folder_el.text.strip() if folder_el is not None and folder_el.text else ""
        if folder:
            folder_l = folder.replace("\\", "/").lower().rstrip("/")
            suffix = f"{folder_l}/{fname_base.lower()}"
            hit = rel_index.get(suffix)
            if hit is not None:
                return hit
            for key, img in rel_index.items():
                if key == suffix or key.endswith("/" + suffix):
                    return img

            # Orphan LabelImg/source folder names (e.g. 测试数据集) are not part of
            # JPEGImages layout — allow unique-basename fallback. Keep C6 strict when
            # folder is a known split or actually exists under image_dir.
            folder_is_split = folder_l in ("train", "val", "test")
            folder_in_tree = any(
                key == folder_l or key.startswith(folder_l + "/")
                for key in rel_index
            )
            if not folder_is_split and not folder_in_tree:
                base = fname_base.lower()
                if base not in basename_collisions:
                    hit = basename_index.get(base)
                    if hit is not None:
                        return hit

            result.unmatched_labels += 1
            result.parse_errors.append(
                f"{xml_path.name}: <folder> '{folder}' 未匹配到图片 '{fname_base}'"
            )
            return None

        # 5) Unique basename only (no folder/path present)
        base = fname_base.lower()
        if base in basename_collisions:
            result.unmatched_labels += 1
            result.parse_errors.append(
                f"{xml_path.name}: 图片 basename '{fname_base}' 在多个目录中重复，"
                f"无法消歧（请在 XML 中提供 <folder> 或 <path>）"
            )
            return None

        hit = basename_index.get(base)
        if hit is None:
            result.unmatched_labels += 1
            result.parse_errors.append(
                f"{xml_path.name}: 未找到匹配图片 '{filename}'"
            )
            return None
        return hit

    def _collect_xml_files(self, label_dir: Path) -> List[Path]:
        """收集所有 XML 文件"""
        xml_files = []
        for xml_path in label_dir.rglob("*.xml"):
            xml_files.append(xml_path)
        return sorted(xml_files)

    def _detect_voc_splits(self, label_dir: Path) -> Dict[str, str]:
        """检测划分目录模式"""
        patterns = {}
        for split in ("train", "val", "test"):
            if (label_dir / split).is_dir():
                patterns[split] = f"{label_dir.name}/{split}"
        return patterns

    def _parse_voc_xml(self, xml_path: Path,
                       class_names: Dict[int, str],
                       name_to_id: Dict[str, int],
                       result: ImportResult) -> List[dict]:
        """解析单个 VOC XML 文件"""
        annotations = []
        try:
            # Guard against huge XML (entity-expansion / memory DoS)
            if xml_path.stat().st_size > 50 * 1024 * 1024:
                result.parse_errors.append(
                    f"{xml_path.name}: XML 文件过大 (>50MB)，已跳过"
                )
                return annotations
            tree = ET.parse(str(xml_path))
            root = tree.getroot()

            # 获取图片尺寸 — 缺少有效 <size> 时绝不伪造 1920×1080，整份 XML 跳过
            size_el = root.find('size')
            if size_el is None:
                result.parse_errors.append(
                    f"{xml_path.name}: 缺少 <size>，已跳过"
                )
                return annotations
            w_el = size_el.find('width')
            h_el = size_el.find('height')
            try:
                img_w = int(float(w_el.text)) if w_el is not None and w_el.text else 0
                img_h = int(float(h_el.text)) if h_el is not None and h_el.text else 0
            except (TypeError, ValueError):
                img_w, img_h = 0, 0
            if img_w <= 0 or img_h <= 0:
                result.parse_errors.append(
                    f"{xml_path.name}: <size> 缺少有效 width/height，已跳过"
                )
                return annotations

            # 解析每个 <object>
            for obj_el in root.findall('object'):
                name_el = obj_el.find('name')
                bndbox_el = obj_el.find('bndbox')

                if name_el is None or bndbox_el is None:
                    continue

                class_name = name_el.text.strip() if name_el.text else "unknown"
                class_id = self._register_class(class_name, class_names, name_to_id)

                xmin = int(float(bndbox_el.find('xmin').text))
                ymin = int(float(bndbox_el.find('ymin').text))
                xmax = int(float(bndbox_el.find('xmax').text))
                ymax = int(float(bndbox_el.find('ymax').text))

                bbox = self._voc_to_yolo_normalized(xmin, ymin, xmax, ymax, img_w, img_h)
                annotations.append(self._make_annotation(class_id, bbox))

        except Exception as e:
            result.parse_errors.append(f"{xml_path.name}: 解析失败: {e}")

        return annotations

    def export(self, annotations, image_path, output_path):
        raise NotImplementedError

    def export_batch(self, data, output_dir):
        raise NotImplementedError
