from __future__ import annotations

import json
from pathlib import Path

from src.gui.utils.exporters.coco_exporter import (
    denormalize_coco_keypoints,
    normalize_coco_keypoints,
)
from src.gui.utils.exporters.yolo_exporter import YOLOExporter
from src.gui.utils.importers.coco_importer import COCOImporter
from src.gui.utils.yolo_label_format import default_kpt_names, parse_yolo_line


def _touch_jpg(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(b"\xff\xd8\xff\xd9")


def _write_coco(tmp_path: Path, payload: dict) -> tuple[Path, Path]:
    images = tmp_path / "images"
    labels = tmp_path / "annotations"
    labels.mkdir(parents=True, exist_ok=True)
    _touch_jpg(images / "img.jpg")
    (labels / "instances_train.json").write_text(
        json.dumps(payload), encoding="utf-8"
    )
    return images, labels


def test_normalize_denormalize_keypoints():
    raw = [20.0, 40.0, 2, 0.0, 0.0, 0]
    kpts = normalize_coco_keypoints(raw, 100, 200)
    assert kpts[0] == (0.2, 0.2, 2)
    assert kpts[1][2] == 0
    flat, num = denormalize_coco_keypoints(kpts, 100, 200)
    assert num == 1
    assert flat[0] == 20.0
    assert flat[1] == 40.0
    assert flat[3:6] == [0, 0, 0]


def test_import_coco_keypoints_and_names(tmp_path: Path):
    payload = {
        "images": [{"id": 1, "file_name": "img.jpg", "width": 100, "height": 100}],
        "categories": [{
            "id": 1,
            "name": "person",
            "keypoints": ["nose", "left_eye"],
        }],
        "annotations": [{
            "id": 1,
            "image_id": 1,
            "category_id": 1,
            "bbox": [10, 10, 40, 40],
            "keypoints": [20, 20, 2, 30, 30, 1],
            "num_keypoints": 2,
        }],
    }
    images, labels = _write_coco(tmp_path, payload)
    _files, frames, _names, _splits, result = COCOImporter().import_dataset(images, labels)
    assert result.kpt_shape == (2, 3)
    assert result.kpt_names == ["nose", "left_eye"]
    ann = frames[0][0]
    assert ann["kind"] == "pose"
    assert ann["keypoints"][0][2] == 2
    assert abs(ann["keypoints"][0][0] - 0.2) < 1e-9


def test_import_seg_plus_keypoints_is_pose(tmp_path: Path):
    payload = {
        "images": [{"id": 1, "file_name": "img.jpg", "width": 100, "height": 100}],
        "categories": [{"id": 1, "name": "person", "keypoints": ["a", "b"]}],
        "annotations": [{
            "id": 1,
            "image_id": 1,
            "category_id": 1,
            "bbox": [0, 0, 50, 50],
            "segmentation": [[10, 10, 40, 10, 40, 40]],
            "keypoints": [12, 12, 2, 0, 0, 0],
        }],
    }
    images, labels = _write_coco(tmp_path, payload)
    _f, frames, _n, _s, _r = COCOImporter().import_dataset(images, labels)
    ann = frames[0][0]
    assert ann["kind"] == "pose"
    assert ann["polygon"] and len(ann["polygon"]) == 3


def test_wrong_keypoint_length_falls_back_to_bbox(tmp_path: Path):
    payload = {
        "images": [{"id": 1, "file_name": "img.jpg", "width": 100, "height": 100}],
        "categories": [{"id": 1, "name": "person", "keypoints": ["a", "b"]}],
        "annotations": [{
            "id": 1,
            "image_id": 1,
            "category_id": 1,
            "bbox": [10, 10, 20, 20],
            "keypoints": [1, 2, 2],
        }],
    }
    images, labels = _write_coco(tmp_path, payload)
    _f, frames, _n, _s, result = COCOImporter().import_dataset(images, labels)
    assert frames[0][0]["kind"] == "bbox"
    assert frames[0][0]["keypoints"] is None
    assert result.warnings or result.parse_errors


def test_coco_to_yolo_pose_export(tmp_path: Path):
    payload = {
        "images": [{"id": 1, "file_name": "img.jpg", "width": 100, "height": 100}],
        "categories": [{"id": 1, "name": "person"}],
        "annotations": [{
            "id": 1,
            "image_id": 1,
            "category_id": 1,
            "bbox": [10, 10, 40, 40],
            "keypoints": [20, 20, 2, 0, 0, 0, 30, 30, 1],
        }],
    }
    images, labels = _write_coco(tmp_path, payload)
    _f, frames, _n, _s, result = COCOImporter().import_dataset(images, labels)
    assert result.kpt_shape == (3, 3)
    text, report = YOLOExporter(tmp_path).format_yolo_task(frames[0], "pose", result.kpt_shape)
    assert report["written"] == 1
    parsed = parse_yolo_line(text.strip(), "pose", (3, 3))
    assert parsed.ok
    assert parsed.keypoints[0][2] == 2
    assert parsed.keypoints[1][2] == 0
    assert parsed.keypoints[2][2] == 1
    assert default_kpt_names(3) == ["kpt_1", "kpt_2", "kpt_3"]
