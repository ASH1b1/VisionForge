from __future__ import annotations

from pathlib import Path

import numpy as np

from src.gui.models.onnx_yolo_decode import (
    arrange_detect_predictions,
    class_names_from_onnx_path,
    decode_ultralytics_detect,
    fallback_class_names,
    letterbox_rgb,
    parse_names_metadata,
)


def test_letterbox_square_image_no_pad():
    img = np.zeros((100, 100, 3), dtype=np.uint8)
    out, scale, pad = letterbox_rgb(img, (640, 640))
    assert out.shape == (640, 640, 3)
    assert abs(scale - 6.4) < 1e-6
    assert pad[0] == 0 and pad[1] == 0


def test_arrange_transposes_channels_first():
    raw = np.zeros((1, 6, 8400), dtype=np.float32)  # 2 类
    pred = arrange_detect_predictions(raw)
    assert pred.shape == (8400, 6)


def test_arrange_keeps_rows_last():
    raw = np.zeros((8400, 6), dtype=np.float32)
    pred = arrange_detect_predictions(raw)
    assert pred.shape == (8400, 6)


def test_arrange_rejects_end2end_nms():
    raw = np.zeros((1, 300, 6), dtype=np.float32)
    try:
        arrange_detect_predictions(raw)
        assert False, "expected ValueError"
    except ValueError as exc:
        assert "nms" in str(exc).lower() or "NMS" in str(exc)


def test_arrange_rejects_transposed_end2end_nms():
    raw = np.zeros((6, 300), dtype=np.float32)
    try:
        arrange_detect_predictions(raw)
        assert False, "expected ValueError"
    except ValueError as exc:
        assert "nms" in str(exc).lower() or "NMS" in str(exc)


def test_decode_one_box_on_unpadded_canvas():
    # 640 canvas, scale=6.4, 100x100 原图；中心 320,320 宽高 64 → 原图 50,50 边长 10
    pred = np.zeros((1, 6, 1), dtype=np.float32)
    pred[0, :, 0] = [320, 320, 64, 64, 0.1, 0.9]  # class 1
    boxes, labels, scores = decode_ultralytics_detect(
        pred, conf_threshold=0.25, scale=6.4, pad=(0.0, 0.0), orig_wh=(100, 100)
    )
    assert labels == [1]
    assert abs(scores[0] - 0.9) < 1e-5
    x1, y1, x2, y2 = boxes[0]
    assert abs(x1 - 45) < 1.0 and abs(y2 - 55) < 1.0


def test_decode_filters_below_conf_threshold():
    pred = np.zeros((1, 6), dtype=np.float32)
    pred[0] = [320, 320, 64, 64, 0.1, 0.2]
    boxes, labels, scores = decode_ultralytics_detect(
        pred, conf_threshold=0.25, scale=6.4, pad=(0.0, 0.0), orig_wh=(100, 100)
    )
    assert boxes == []
    assert labels == []
    assert scores == []


def test_parse_names_json_and_python_dict():
    assert parse_names_metadata('{"0": "person", "1": "car"}')[0] == "person"
    assert parse_names_metadata("{0: 'a', 1: 'b'}")[1] == "b"


def test_class_names_from_onnx_path_reads_data_yaml(tmp_path: Path):
    onnx_path = tmp_path / "best.onnx"
    onnx_path.write_bytes(b"")
    (tmp_path / "data.yaml").write_text(
        "names:\n  0: person\n  1: bicycle\n",
        encoding="utf-8",
    )
    names = class_names_from_onnx_path(onnx_path)
    assert names[0] == "person"
    assert names[1] == "bicycle"


def test_class_names_from_onnx_path_reads_stem_yaml(tmp_path: Path):
    onnx_path = tmp_path / "custom.onnx"
    onnx_path.write_bytes(b"")
    (tmp_path / "custom.yaml").write_text(
        "names: [cat, dog]\n",
        encoding="utf-8",
    )
    names = class_names_from_onnx_path(onnx_path)
    assert names[0] == "cat"
    assert names[1] == "dog"


def test_fallback_class_names():
    assert fallback_class_names(2) == {0: "class_0", 1: "class_1"}
