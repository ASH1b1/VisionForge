from __future__ import annotations

from pathlib import Path

import pytest

from src.gui.utils.yolo_label_format import (
    aabb_from_obb_corners,
    default_kpt_names,
    format_export_skip_message,
    normalize_task,
    parse_data_yaml,
    parse_yolo_line,
    resolve_import_task,
)


def test_normalize_task_aliases():
    assert normalize_task("detection") == "detect"
    assert normalize_task("seg") == "segment"
    assert normalize_task("segmentation") == "segment"
    assert normalize_task("pose") == "pose"
    assert normalize_task("obb") == "obb"
    assert normalize_task("rotated") == "obb"
    assert normalize_task("nope") is None
    assert normalize_task(None) is None


def test_resolve_user_choice_overrides_yaml(tmp_path: Path):
    yaml_path = tmp_path / "data.yaml"
    yaml_path.write_text("task: detect\nkpt_shape: [17, 3]\n", encoding="utf-8")
    meta = parse_data_yaml(yaml_path)
    resolved = resolve_import_task(meta, "obb")
    assert resolved.needs_user_choice is False
    assert resolved.task == "obb"
    assert resolved.source == "user"


def test_resolve_auto_uses_yaml_task(tmp_path: Path):
    yaml_path = tmp_path / "data.yaml"
    yaml_path.write_text("task: obb\nnames: [box]\n", encoding="utf-8")
    meta = parse_data_yaml(yaml_path)
    resolved = resolve_import_task(meta, "auto")
    assert resolved.task == "obb"
    assert resolved.source == "yaml_task"
    assert resolved.needs_user_choice is False


def test_resolve_auto_kpt_shape_means_pose(tmp_path: Path):
    yaml_path = tmp_path / "data.yaml"
    yaml_path.write_text("kpt_shape: [17, 3]\nnames: [person]\n", encoding="utf-8")
    meta = parse_data_yaml(yaml_path)
    assert meta.kpt_shape == (17, 3)
    resolved = resolve_import_task(meta, "auto")
    assert resolved.task == "pose"
    assert resolved.source == "yaml_kpt_shape"


def test_resolve_auto_without_yaml_requires_choice():
    resolved = resolve_import_task(None, "auto")
    assert resolved.task is None
    assert resolved.needs_user_choice is True


def test_parse_detect_ok_and_reject_segment_width():
    ok = parse_yolo_line("0 0.5 0.5 0.2 0.4", "detect")
    assert ok.ok
    assert ok.kind == "bbox"
    assert ok.bbox == pytest.approx((0.5, 0.5, 0.2, 0.4))

    bad = parse_yolo_line("0 0.1 0.1 0.2 0.2 0.3 0.3", "detect")
    assert not bad.ok
    assert bad.error


def test_parse_obb_eight_corners_not_polygon():
    line = "1 0.1 0.1 0.9 0.1 0.9 0.9 0.1 0.9"
    result = parse_yolo_line(line, "obb")
    assert result.ok
    assert result.kind == "obb"
    assert result.polygon is None
    assert result.obb == (
        (0.1, 0.1),
        (0.9, 0.1),
        (0.9, 0.9),
        (0.1, 0.9),
    )
    assert result.bbox == pytest.approx((0.5, 0.5, 0.8, 0.8))


def test_parse_obb_rejects_xywhr():
    result = parse_yolo_line("0 0.5 0.5 0.2 0.4 0.7", "obb")
    assert not result.ok
    assert "8" in (result.error or "")


def test_parse_segment_even_vertices():
    result = parse_yolo_line("0 0.1 0.1 0.2 0.1 0.2 0.2", "segment")
    assert result.ok
    assert result.kind == "polygon"
    assert result.polygon == [(0.1, 0.1), (0.2, 0.1), (0.2, 0.2)]


def test_parse_segment_rejects_odd_or_short():
    assert not parse_yolo_line("0 0.1 0.1 0.2", "segment").ok
    assert not parse_yolo_line("0 0.1 0.1", "segment").ok


def test_parse_pose_uses_kpt_shape():
    coords = ["0", "0.5", "0.5", "0.2", "0.2"]
    for i in range(3):
        coords.extend([f"0.{i+1}", f"0.{i+2}", "2" if i else "1"])
    result = parse_yolo_line(coords, "pose", kpt_shape=(3, 3))
    assert result.ok
    assert result.kind == "pose"
    assert result.bbox == pytest.approx((0.5, 0.5, 0.2, 0.2))
    assert len(result.keypoints) == 3
    assert result.keypoints[0][2] == 1
    assert result.keypoints[1][2] == 2


def test_parse_pose_infers_kpt_shape_3_then_2():
    d3 = parse_yolo_line(
        "0 0.4 0.4 0.2 0.2 0.1 0.1 2 0.2 0.2 1",
        "pose",
        kpt_shape=None,
    )
    assert d3.ok
    assert d3.inferred_kpt_shape == (2, 3)
    assert d3.keypoints[1] == pytest.approx((0.2, 0.2, 1))

    d2 = parse_yolo_line(
        "0 0.4 0.4 0.2 0.2 0.1 0.1 0.2 0.2",
        "pose",
        kpt_shape=None,
    )
    assert d2.ok
    assert d2.inferred_kpt_shape == (2, 2)
    assert d2.keypoints[0][2] == 2


def test_parse_pose_invalid_visibility_becomes_zero_with_warning():
    result = parse_yolo_line(
        "0 0.4 0.4 0.2 0.2 0.1 0.1 9",
        "pose",
        kpt_shape=(1, 3),
    )
    assert result.ok
    assert result.keypoints[0][2] == 0
    assert result.warning


def test_parse_pose_wrong_width_is_error():
    result = parse_yolo_line(
        "0 0.4 0.4 0.2 0.2 0.1 0.1 2",
        "pose",
        kpt_shape=(17, 3),
    )
    assert not result.ok


def test_aabb_from_obb_corners():
    bbox = aabb_from_obb_corners(((0.1, 0.2), (0.5, 0.2), (0.5, 0.8), (0.1, 0.8)))
    assert bbox == pytest.approx((0.3, 0.5, 0.4, 0.6))


def test_default_kpt_names_and_skip_message():
    assert default_kpt_names(2) == ["kpt_1", "kpt_2"]
    text = format_export_skip_message(
        written=120,
        skipped={"obb": 8, "bbox": 3},
        task_label="姿态",
    )
    assert "120" in text
    assert "姿态" in text
    assert "8" in text
    assert "旋转框" in text
    assert "3" in text
    assert "普通框" in text


def test_parse_yaml_names_list_and_dict(tmp_path: Path):
    p = tmp_path / "data.yaml"
    p.write_text("names:\n  0: car\n  1: bus\n", encoding="utf-8")
    meta = parse_data_yaml(p)
    assert meta.names[0] == "car"
    assert meta.names[1] == "bus"
    p.write_text("names: [person, bike]\nkpt_names: [nose, left_eye]\n", encoding="utf-8")
    meta = parse_data_yaml(p)
    assert meta.names[0] == "person"
    assert meta.kpt_names == ["nose", "left_eye"]
