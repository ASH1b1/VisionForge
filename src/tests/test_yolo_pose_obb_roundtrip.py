from __future__ import annotations

from pathlib import Path

from src.gui.models.project_document import Annotation, ProjectDocument
from src.gui.utils.exporters.yolo_exporter import YOLOExporter
from src.gui.utils.importers.yolo_importer import YOLOImporter
from src.gui.utils.yolo_label_format import (
    format_export_skip_message,
    parse_yolo_line,
    quantize6,
)


def _touch_jpg(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(b"\xff\xd8\xff\xd9")


def _dataset(tmp_path: Path, lines: list[str], yaml_text: str | None = None) -> tuple[Path, Path]:
    images = tmp_path / "images"
    labels = tmp_path / "labels"
    _touch_jpg(images / "img.jpg")
    (labels / "img.txt").parent.mkdir(parents=True, exist_ok=True)
    (labels / "img.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")
    if yaml_text is not None:
        (tmp_path / "data.yaml").write_text(yaml_text, encoding="utf-8")
    return images, labels


def test_import_yaml_obb_is_not_polygon(tmp_path: Path):
    images, labels = _dataset(
        tmp_path,
        ["0 0.1 0.1 0.9 0.1 0.9 0.9 0.1 0.9"],
        "task: obb\nnames: [box]\n",
    )
    files, frames, names, _splits, result = YOLOImporter().import_dataset(images, labels)
    assert result.detected_task == "obb"
    assert len(files) == 1
    ann = frames[0][0]
    assert ann["kind"] == "obb"
    assert ann["polygon"] is None
    assert len(ann["obb"]) == 4
    assert names[0] == "box"


def test_import_yaml_pose_keeps_visibility(tmp_path: Path):
    line = "0 0.5 0.5 0.2 0.2 " + " ".join(
        f"{0.1 + i * 0.01:.3f} {0.2 + i * 0.01:.3f} {i % 3}" for i in range(17)
    )
    images, labels = _dataset(
        tmp_path,
        [line],
        "task: pose\nkpt_shape: [17, 3]\nnames: [person]\n",
    )
    _files, frames, _names, _splits, result = YOLOImporter().import_dataset(images, labels)
    assert result.kpt_shape == (17, 3)
    kpts = frames[0][0]["keypoints"]
    assert len(kpts) == 17
    assert [p[2] for p in kpts] == [i % 3 for i in range(17)]


def test_import_user_obb_rejects_xywhr(tmp_path: Path):
    images, labels = _dataset(tmp_path, ["0 0.5 0.5 0.2 0.4 0.7", "0 0.1 0.1 0.9 0.1 0.9 0.9 0.1 0.9"])
    _files, frames, _n, _s, result = YOLOImporter().import_dataset(
        images, labels, task_choice="obb"
    )
    assert len(frames[0]) == 1
    assert frames[0][0]["kind"] == "obb"
    assert result.parse_errors


def test_import_auto_without_yaml_does_not_eat_as_segment(tmp_path: Path):
    images, labels = _dataset(tmp_path, ["0 0.1 0.1 0.9 0.1 0.9 0.9 0.1 0.9"])
    _files, frames, _n, _s, result = YOLOImporter().import_dataset(
        images, labels, task_choice="auto"
    )
    assert result.needs_task_choice is True
    assert frames == {} or not any(frames.values())
    assert result.total_annotations == 0


def test_import_pose_infers_k_from_first_line(tmp_path: Path):
    good = "0 0.4 0.4 0.2 0.2 0.1 0.1 2 0.2 0.2 1"
    bad = "0 0.4 0.4 0.2 0.2 0.1 0.1 2"
    images, labels = _dataset(tmp_path, [good, bad])
    _files, frames, _n, _s, result = YOLOImporter().import_dataset(
        images, labels, task_choice="pose"
    )
    assert result.kpt_shape == (2, 3)
    assert result.kpt_shape_source == "inferred"
    assert len(frames[0]) == 1
    assert result.parse_errors


def test_import_skips_pose_when_k_conflicts(tmp_path: Path):
    line21 = "0 0.5 0.5 0.2 0.2 " + " ".join(["0.1 0.1 2"] * 21)
    images, labels = _dataset(tmp_path, [line21], "kpt_shape: [21, 3]\nnames: [p]\n")
    _files, frames, _n, _s, result = YOLOImporter().import_dataset(
        images, labels, task_choice="pose", existing_kpt_shape=(17, 3)
    )
    assert result.kpt_shape == (17, 3)
    assert not frames.get(0)
    assert result.skipped_incompatible >= 1


def test_import_detect_and_segment_regression(tmp_path: Path):
    images, labels = _dataset(
        tmp_path,
        ["1 0.5 0.5 0.2 0.4", "0 0.1 0.1 0.2 0.1 0.2 0.2"],
        "task: detect\nnames: [a, b]\n",
    )
    _f, frames, _n, _s, result = YOLOImporter().import_dataset(images, labels)
    assert result.detected_task == "detect"
    kinds = [a["kind"] for a in frames[0]]
    assert kinds == ["bbox"]
    assert result.parse_errors  # 分割行在 detect 任务下是错误

    images, labels = _dataset(
        tmp_path / "seg",
        ["0 0.1 0.1 0.2 0.1 0.2 0.2"],
        "task: segment\nnames: [poly]\n",
    )
    _f, frames, _n, _s, result = YOLOImporter().import_dataset(images, labels)
    assert frames[0][0]["kind"] == "polygon"
    assert len(frames[0][0]["polygon"]) == 3


def test_export_pose_obb_roundtrip_quantized(tmp_path: Path):
    exporter = YOLOExporter(tmp_path)
    pose_kpts = [(0.1234567, 0.2345678, 2), (0.3456789, 0.4567891, 1)]
    pose_ann = {
        "class_id": 0,
        "kind": "pose",
        "bbox": (0.5111111, 0.5222222, 0.1333333, 0.1444444),
        "keypoints": pose_kpts,
        "visible": True,
    }
    pose_path = tmp_path / "pose.txt"
    report = exporter.export_yolo_task([pose_ann, {"class_id": 0, "kind": "bbox", "bbox": (0.5, 0.5, 0.1, 0.1), "visible": True}], pose_path, "pose")
    assert report["written"] == 1
    assert report["skipped"]["bbox"] == 1
    parsed = parse_yolo_line(
        pose_path.read_text(encoding="utf-8").strip(), "pose", (2, 3)
    )
    assert parsed.ok
    for src, got in zip(pose_kpts, parsed.keypoints):
        assert abs(quantize6(src[0]) - got[0]) < 1e-5
        assert abs(quantize6(src[1]) - got[1]) < 1e-5
        assert src[2] == got[2]

    obb = ((0.1111111, 0.1222222), (0.9111111, 0.1222222), (0.9111111, 0.9222222), (0.1111111, 0.9222222))
    obb_ann = {"class_id": 3, "kind": "obb", "obb": obb, "bbox": (0.5, 0.5, 0.8, 0.8), "visible": True}
    obb_path = tmp_path / "obb.txt"
    report = exporter.export_yolo_task([obb_ann, pose_ann], obb_path, "obb")
    assert report["written"] == 1
    assert report["skipped"]["pose"] == 1
    parsed = parse_yolo_line(obb_path.read_text(encoding="utf-8").strip(), "obb")
    assert parsed.ok
    for src, got in zip(obb, parsed.obb):
        assert abs(quantize6(src[0]) - got[0]) < 1e-5
        assert abs(quantize6(src[1]) - got[1]) < 1e-5


def test_export_detect_skips_pose_and_obb():
    exporter = YOLOExporter(".")
    anns = [
        {"class_id": 0, "kind": "bbox", "bbox": (0.5, 0.5, 0.2, 0.2), "visible": True},
        {"class_id": 0, "kind": "polygon", "bbox": (0.4, 0.4, 0.1, 0.1), "polygon": [(0.3, 0.3), (0.5, 0.3), (0.4, 0.5)], "visible": True},
        {"class_id": 0, "kind": "obb", "obb": ((0.1, 0.1), (0.2, 0.1), (0.2, 0.2), (0.1, 0.2)), "bbox": (0.15, 0.15, 0.1, 0.1), "visible": True},
        {"class_id": 0, "kind": "pose", "bbox": (0.5, 0.5, 0.2, 0.2), "keypoints": [(0.1, 0.1, 2)], "visible": True},
    ]
    text, report = exporter.format_yolo_task(anns, "detect")
    lines = [ln for ln in text.splitlines() if ln.strip()]
    assert len(lines) == 2
    assert all(len(ln.split()) == 5 for ln in lines)
    assert report["skipped"]["obb"] == 1
    assert report["skipped"]["pose"] == 1

    text, report = exporter.format_yolo_task(anns, "segment")
    lines = [ln for ln in text.splitlines() if ln.strip()]
    assert len(lines) == 1
    assert len(lines[0].split()) == 7
    assert report["written"] == 1


def test_skip_message_mentions_counts():
    msg = format_export_skip_message(12, {"obb": 8, "bbox": 3}, "姿态")
    assert "已导出 12 条姿态" in msg
    assert "8 条旋转框" in msg


def test_project_replace_keeps_imported_kpt(tmp_path: Path):
    images, labels = _dataset(
        tmp_path,
        ["0 0.5 0.5 0.2 0.2 0.1 0.1 2 0.2 0.2 0"],
        "kpt_shape: [2, 3]\nnames: [p]\n",
    )
    files, frames, names, splits, result = YOLOImporter().import_dataset(
        images, labels, task_choice="pose"
    )
    doc = ProjectDocument()
    doc.replace_data(
        image_files=files,
        frame_annotations=frames,
        class_names=names,
        split_map=splits,
        kpt_shape=result.kpt_shape,
        kpt_names=result.kpt_names,
    )
    assert doc.kpt_shape == (2, 3)
    assert doc.frame_annotations[0][0].kind == "pose"


def test_annotation_from_import_dict():
    ann = Annotation.from_dict({
        "id": 1,
        "class_id": 0,
        "bbox": (0.5, 0.5, 0.2, 0.2),
        "kind": "obb",
        "obb": [(0.1, 0.1), (0.9, 0.1), (0.9, 0.9), (0.1, 0.9)],
    })
    assert ann.kind == "obb"
    assert ann.obb[2] == (0.9, 0.9)
