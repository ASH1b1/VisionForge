# src/gui/utils/importers/yolo_importer.py
"""YOLO TXT 格式导入器"""
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from .base_importer import BaseImporter, ImportResult
from src.gui.utils.yolo_label_format import (
    default_kpt_names,
    find_data_yaml,
    parse_data_yaml,
    parse_yolo_line,
    resolve_import_task,
)


class YOLOImporter(BaseImporter):
    """YOLO TXT 格式导入器

    支持结构:
      - 扁平: images/ + labels/
      - 划分: images/{train,val,test}/ + labels/{train,val,test}/
    """

    IMAGE_EXTS = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif', '.webp'}

    def import_dataset(self, image_dir: Path, label_dir: Path,
                       task_choice: str = "auto",
                       existing_kpt_shape: Optional[tuple] = None) -> Tuple[
        List[Path], Dict[int, List[dict]], Dict[int, str], Dict[int, str], ImportResult
    ]:
        self._next_ann_id = 1  # 每次导入重置 ID 计数器
        result = ImportResult()
        class_names: Dict[int, str] = {}
        name_to_id: Dict[str, int] = {}
        image_files: List[Path] = []
        frame_annotations: Dict[int, Tuple] = {}  # frame_index → (img_path, annotations)
        split_map: Dict[int, str] = {}

        image_dir = Path(image_dir)
        label_dir = Path(label_dir)

        # Step 1: 检测是否有 train/val/test 子目录
        splits = self._detect_splits(image_dir, label_dir)

        # Step 2: 若有 data.yaml，读取类别名 / task / kpt_shape
        yaml_meta = None
        data_yaml_path = find_data_yaml(image_dir, label_dir)
        if data_yaml_path is not None:
            yaml_meta = parse_data_yaml(data_yaml_path)
            class_names = dict(yaml_meta.names)
            name_to_id = {
                str(name).strip().casefold(): cid for cid, name in class_names.items()
            }

        resolved = resolve_import_task(yaml_meta, task_choice)
        if resolved.needs_user_choice:
            result.success = False
            result.needs_task_choice = True
            result.parse_errors.append(
                "无法从 data.yaml 判定任务，请选择检测 / 分割 / 姿态 / OBB"
            )
            return image_files, {}, class_names, {}, result

        task = resolved.task
        result.detected_task = task
        kpt_shape = existing_kpt_shape
        if yaml_meta and yaml_meta.kpt_shape:
            if existing_kpt_shape and existing_kpt_shape[0] != yaml_meta.kpt_shape[0]:
                result.warnings.append(
                    f"工程已有 kpt_shape={list(existing_kpt_shape)}，"
                    f"本次 yaml 为 {list(yaml_meta.kpt_shape)}，将跳过不匹配的姿态行"
                )
            elif existing_kpt_shape is None:
                kpt_shape = yaml_meta.kpt_shape
                result.kpt_shape_source = "yaml"
        if yaml_meta and yaml_meta.kpt_names and existing_kpt_shape is None:
            result.kpt_names = list(yaml_meta.kpt_names)
        self._import_task = task
        self._import_kpt_shape = kpt_shape
        self._kpt_shape_locked = existing_kpt_shape is not None or (
            yaml_meta is not None and yaml_meta.kpt_shape is not None
        )

        # Step 3: 收集图片和标注文件
        if splits:
            for split, (img_subdir, lbl_subdir) in splits.items():
                self._collect_split(img_subdir, lbl_subdir, split,
                                    image_files, frame_annotations, split_map)
        else:
            self._collect_flat(image_dir, label_dir, image_files, frame_annotations)

        # Step 4: 按文件名排序
        image_files.sort(key=lambda p: p.name.lower())

        # Step 5: 重建 frame_annotations，使用排序后的索引
        img_to_ann = {img_path: anns for img_path, anns in frame_annotations.values()}
        sorted_annotations: Dict[int, List[dict]] = {}
        sorted_split_map: Dict[int, str] = {}
        img_to_idx = {img: idx for idx, img in enumerate(image_files)}

        for old_idx, (img_path, _) in frame_annotations.items():
            new_idx = img_to_idx[img_path]
            sorted_annotations[new_idx] = img_to_ann.get(img_path, [])
            if old_idx in split_map:
                sorted_split_map[new_idx] = split_map[old_idx]

        # Step 6: 解析标注文件
        total_anns = 0
        for frame_idx, img_path in enumerate(image_files):
            anns = sorted_annotations.get(frame_idx, [])
            stem = img_path.stem
            split_name = sorted_split_map.get(frame_idx)
            label_paths = self._find_label_files(stem, label_dir, splits, split_name)
            for lp in label_paths:
                parsed = self._parse_yolo_label(lp, class_names, name_to_id, result)
                anns.extend(parsed)
                total_anns += len(parsed)
            sorted_annotations[frame_idx] = anns

        # Step 7: 填充 result
        result.total_images = len(image_files)
        result.total_annotations = total_anns
        result.matched_pairs = sum(1 for anns in sorted_annotations.values() if anns)
        result.unmatched_images = sum(1 for anns in sorted_annotations.values() if not anns)
        result.class_names = class_names
        result.split_map = sorted_split_map
        result.kpt_shape = self._import_kpt_shape
        if result.kpt_shape and not result.kpt_shape_source:
            result.kpt_shape_source = "inferred" if task == "pose" else result.kpt_shape_source
        if result.kpt_shape and not result.kpt_names:
            result.kpt_names = default_kpt_names(result.kpt_shape[0])

        return image_files, sorted_annotations, class_names, sorted_split_map, result

    def _detect_splits(self, image_dir: Path, label_dir: Path) -> Dict[str, Tuple[Path, Path]] | None:
        """检测 train/val/test 子目录结构

        Returns:
            {"train": (images/train, labels/train), ...} 或 None
        """
        splits = {}
        for split in ("train", "val", "test"):
            img_sub = image_dir / split
            lbl_sub = label_dir / split
            if not (img_sub.is_dir() and lbl_sub.is_dir()):
                continue
            # Same path (e.g. user selected images/ for both) is not a real split
            try:
                if img_sub.resolve() == lbl_sub.resolve():
                    continue
            except OSError:
                if img_sub == lbl_sub:
                    continue
            splits[split] = (img_sub, lbl_sub)
        return splits if splits else None

    def _collect_split(self, img_subdir: Path, lbl_subdir: Path, split_name: str,
                       image_files: List[Path],
                       frame_annotations: Dict[int, Tuple[Path, List[dict]]],
                       split_map: Dict[int, str]):
        """收集一个 split 的图片（仅按 resolved path 去重，保留 train/val 同 stem）。"""
        resolved_seen = {str(p.resolve()) for p in image_files}
        for ext in self.IMAGE_EXTS:
            for img_path in sorted(img_subdir.glob(f"*{ext}")):
                key = str(img_path.resolve())
                if key in resolved_seen:
                    continue
                resolved_seen.add(key)
                idx = len(image_files)
                image_files.append(img_path)
                frame_annotations[idx] = (img_path, [])
                split_map[idx] = split_name

    def _collect_flat(self, image_dir: Path, label_dir: Path,
                      image_files: List[Path],
                      frame_annotations: Dict[int, Tuple[Path, List[dict]]]):
        """扁平结构收集（仅按 resolved path 去重）。"""
        resolved_seen = {str(p.resolve()) for p in image_files}
        for ext in self.IMAGE_EXTS:
            for img_path in sorted(image_dir.glob(f"*{ext}")):
                key = str(img_path.resolve())
                if key in resolved_seen:
                    continue
                resolved_seen.add(key)
                idx = len(image_files)
                image_files.append(img_path)
                frame_annotations[idx] = (img_path, [])

    def _find_label_files(self, stem: str, label_dir: Path,
                          splits: Dict[str, Tuple[Path, Path]] | None,
                          split_name: str | None = None) -> List[Path]:
        """查找标注文件，若已知图片所属 split 则只查找该 split 目录。

        Prefer split-specific labels over flat labels to avoid double-parsing
        when both label_dir/{stem}.txt and labels/{split}/{stem}.txt exist.
        """
        found: List[Path] = []
        if splits:
            if split_name and split_name in splits:
                # 只查找图片所属 split 的标签（避免跨 split 重复加载）
                _, lbl_sub = splits[split_name]
                p = lbl_sub / f"{stem}.txt"
                if p.is_file():
                    return [p]
            else:
                # 兜底：遍历所有 split（兼容旧数据）
                for _, (_, lbl_sub) in splits.items():
                    p = lbl_sub / f"{stem}.txt"
                    if p.is_file():
                        found.append(p)
                if found:
                    return found
        flat = label_dir / f"{stem}.txt"
        if flat.is_file():
            found.append(flat)
        return found

    def _parse_yolo_label(self, label_path: Path,
                          class_names: Dict[int, str],
                          name_to_id: Dict[str, int],
                          result: ImportResult) -> List[dict]:
        """解析单个 YOLO label 文件"""
        annotations = []
        try:
            with open(label_path, 'r', encoding='utf-8') as f:
                for line_no, line in enumerate(f, 1):
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        ann = self._parse_yolo_line(line, class_names, name_to_id, result)
                        if ann:
                            annotations.append(ann)
                    except Exception as e:
                        result.parse_errors.append(
                            f"{label_path.name}:{line_no}: {e}"
                        )
        except Exception as e:
            result.parse_errors.append(f"{label_path.name}: 读取失败: {e}")
        return annotations

    def _parse_yolo_line(self, line: str,
                         class_names: Dict[int, str],
                         name_to_id: Dict[str, int],
                         result: ImportResult) -> dict | None:
        """解析单行 YOLO 标注"""
        parsed = parse_yolo_line(line, self._import_task, self._import_kpt_shape)
        if not parsed.ok:
            if self._import_task == "pose" and self._import_kpt_shape is not None:
                result.skipped_incompatible += 1
            raise ValueError(parsed.error or "无法解析")
        if parsed.warning:
            result.warnings.append(parsed.warning)
        if parsed.kind == "pose" and parsed.inferred_kpt_shape:
            inferred = parsed.inferred_kpt_shape
            if self._import_kpt_shape and inferred[0] != self._import_kpt_shape[0]:
                result.skipped_incompatible += 1
                raise ValueError(
                    f"关键点数冲突：工程 {list(self._import_kpt_shape)}，"
                    f"本行 {list(inferred)}"
                )
            if self._import_kpt_shape is None:
                self._import_kpt_shape = inferred
                result.kpt_shape_source = "inferred"
        elif parsed.kind == "pose" and self._import_kpt_shape:
            k = self._import_kpt_shape[0]
            if parsed.keypoints is not None and len(parsed.keypoints) != k:
                result.skipped_incompatible += 1
                raise ValueError(f"关键点数冲突：需要 {k}，实际 {len(parsed.keypoints)}")

        class_id = parsed.class_id if parsed.class_id is not None else 0
        if class_id in class_names:
            pass
        elif class_names:
            # keep original class id even if unnamed
            class_names.setdefault(class_id, f"class_{class_id}")
        else:
            class_names[class_id] = f"class_{class_id}"
            name_to_id[class_names[class_id].casefold()] = class_id

        return self._make_annotation(
            class_id,
            parsed.bbox or (0.5, 0.5, 0.0, 0.0),
            parsed.polygon,
            kind=parsed.kind,
            obb=parsed.obb,
            keypoints=parsed.keypoints,
        )

    # 实现基类的抽象方法（导出用，导入不需要，空实现）
    def export(self, annotations, image_path, output_path):
        raise NotImplementedError

    def export_batch(self, data, output_dir):
        raise NotImplementedError
