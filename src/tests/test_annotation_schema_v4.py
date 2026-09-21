from __future__ import annotations

from src.gui.models.project_document import Annotation, ProjectDocument


def test_from_dict_old_bbox_defaults_kind():
    ann = Annotation.from_dict({
        "id": 1,
        "class_id": 0,
        "bbox": [0.5, 0.5, 0.2, 0.2],
        "confidence": 1.0,
        "visible": True,
    })
    assert ann.kind == "bbox"
    assert ann.obb is None
    assert ann.keypoints is None


def test_from_dict_old_polygon_defaults_kind():
    ann = Annotation.from_dict({
        "id": 2,
        "class_id": 0,
        "bbox": [0.5, 0.5, 0.2, 0.2],
        "polygon": [(0.4, 0.4), (0.6, 0.4), (0.5, 0.6)],
    })
    assert ann.kind == "polygon"


def test_clone_deepcopies_obb_and_keypoints():
    ann = Annotation(
        id=3,
        class_id=0,
        confidence=1.0,
        bbox=(0.5, 0.5, 0.4, 0.4),
        kind="pose",
        keypoints=[(0.1, 0.2, 2), (0.3, 0.4, 1)],
        obb=((0.1, 0.1), (0.9, 0.1), (0.9, 0.9), (0.1, 0.9)),
    )
    cloned = ann.clone()
    cloned.keypoints[0] = (0.0, 0.0, 0)
    cloned.obb = ((0, 0), (1, 0), (1, 1), (0, 1))
    assert ann.keypoints[0] == (0.1, 0.2, 2)
    assert ann.obb[0] == (0.1, 0.1)


def test_to_dict_roundtrip_pose_obb():
    src = Annotation(
        id=4,
        class_id=1,
        confidence=0.9,
        bbox=(0.4, 0.5, 0.2, 0.3),
        kind="obb",
        obb=((0.3, 0.35), (0.5, 0.35), (0.5, 0.65), (0.3, 0.65)),
    )
    restored = Annotation.from_dict(src.to_dict())
    assert restored.kind == "obb"
    assert restored.obb == src.obb

    pose = Annotation(
        id=5,
        class_id=0,
        confidence=1.0,
        bbox=(0.5, 0.5, 0.2, 0.2),
        kind="pose",
        keypoints=[(0.4, 0.4, 2), (0.6, 0.6, 0)],
    )
    restored = Annotation.from_dict(pose.to_dict())
    assert restored.kind == "pose"
    assert restored.keypoints == pose.keypoints


def test_gsproj_v3_opens_and_save_writes_v4():
    doc = ProjectDocument.from_dict({
        "version": 3,
        "image_files": [],
        "frame_annotations": {
            "0": [{
                "id": 1,
                "class_id": 0,
                "bbox": [0.5, 0.5, 0.1, 0.1],
                "confidence": 1.0,
            }]
        },
        "class_names": {"0": "car"},
        "split_map": {},
        "current_frame": 0,
    })
    assert doc.kpt_shape is None
    assert doc.kpt_names is None
    assert doc.frame_annotations[0][0].kind == "bbox"
    payload = doc.to_dict()
    assert payload["version"] == 4
    assert payload["kpt_shape"] is None


def test_replace_data_keeps_kpt_metadata():
    doc = ProjectDocument()
    doc.replace_data(
        image_files=[],
        frame_annotations={},
        class_names={},
        kpt_shape=(17, 3),
        kpt_names=["nose"] + [f"kpt_{i}" for i in range(2, 18)],
    )
    assert doc.kpt_shape == (17, 3)
    assert len(doc.kpt_names) == 17
    payload = doc.to_dict()
    assert payload["kpt_shape"] == [17, 3]
    restored = ProjectDocument.from_dict(payload)
    assert restored.kpt_shape == (17, 3)
    assert restored.kpt_names[0] == "nose"


def test_add_annotation_sets_kind_from_polygon():
    doc = ProjectDocument()
    box = doc.add_annotation(0, (0.5, 0.5, 0.2, 0.2))
    assert box.kind == "bbox"
    poly = doc.add_annotation(
        0, (0.5, 0.5, 0.2, 0.2),
        polygon=[(0.4, 0.4), (0.6, 0.4), (0.5, 0.6)],
    )
    assert poly.kind == "polygon"
