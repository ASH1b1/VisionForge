"""YOLO label format helpers — yaml task/kpt_shape, line parse, OBB AABB.

Must stay free of Qt and GPU. Callers: YOLO importer/exporter tests.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import yaml

TASK_DETECT = "detect"
TASK_SEGMENT = "segment"
TASK_POSE = "pose"
TASK_OBB = "obb"
VALID_TASKS = (TASK_DETECT, TASK_SEGMENT, TASK_POSE, TASK_OBB)

_TASK_ALIASES = {
    "detect": TASK_DETECT,
    "detection": TASK_DETECT,
    "det": TASK_DETECT,
    "segment": TASK_SEGMENT,
    "segmentation": TASK_SEGMENT,
    "seg": TASK_SEGMENT,
    "pose": TASK_POSE,
    "keypoint": TASK_POSE,
    "keypoints": TASK_POSE,
    "obb": TASK_OBB,
    "rotated": TASK_OBB,
    "rotated_box": TASK_OBB,
}

KIND_LABELS = {
    "bbox": "普通框",
    "polygon": "多边形",
    "obb": "旋转框",
    "pose": "姿态",
}


@dataclass
class YamlMeta:
    task: str | None = None
    kpt_shape: tuple[int, int] | None = None
    names: dict[int, str] = field(default_factory=dict)
    kpt_names: list[str] | None = None


@dataclass
class TaskResolution:
    task: str | None
    needs_user_choice: bool
    source: str = ""


@dataclass
class YoloLineResult:
    ok: bool
    error: str | None = None
    warning: str | None = None
    class_id: int | None = None
    kind: str | None = None
    bbox: tuple[float, float, float, float] | None = None
    polygon: list[tuple[float, float]] | None = None
    obb: tuple[tuple[float, float], ...] | None = None
    keypoints: list[tuple[float, float, int]] | None = None
    inferred_kpt_shape: tuple[int, int] | None = None


def normalize_task(raw: Any) -> str | None:
    if raw is None:
        return None
    key = str(raw).strip().lower()
    if not key:
        return None
    return _TASK_ALIASES.get(key)


def _parse_kpt_shape(raw: Any) -> tuple[int, int] | None:
    if raw is None:
        return None
    if isinstance(raw, (list, tuple)) and len(raw) >= 2:
        try:
            k = int(raw[0])
            d = int(raw[1])
        except (TypeError, ValueError):
            return None
        if k > 0 and d in (2, 3):
            return (k, d)
    return None


def _parse_names(raw: Any) -> dict[int, str]:
    names: dict[int, str] = {}
    if isinstance(raw, dict):
        for idx, name in raw.items():
            try:
                names[int(idx)] = str(name)
            except (TypeError, ValueError):
                continue
    elif isinstance(raw, list):
        for idx, name in enumerate(raw):
            names[idx] = str(name)
    return names


def parse_data_yaml(path: str | Path) -> YamlMeta:
    meta = YamlMeta()
    try:
        with open(path, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f) or {}
    except Exception:
        return meta
    if not isinstance(data, dict):
        return meta
    meta.task = normalize_task(data.get("task"))
    meta.kpt_shape = _parse_kpt_shape(data.get("kpt_shape"))
    meta.names = _parse_names(data.get("names"))
    kpt_names = data.get("kpt_names")
    if isinstance(kpt_names, list) and kpt_names:
        meta.kpt_names = [str(n) for n in kpt_names]
    return meta


def find_data_yaml(image_dir: str | Path | None, label_dir: str | Path | None) -> Path | None:
    candidates: list[Path] = []
    if image_dir is not None:
        parent = Path(image_dir).parent
        candidates.append(parent / "data.yaml")
        candidates.append(Path(image_dir) / "data.yaml")
    if label_dir is not None:
        parent = Path(label_dir).parent
        candidates.append(parent / "data.yaml")
        candidates.append(Path(label_dir) / "data.yaml")
    seen: set[str] = set()
    for path in candidates:
        key = str(path)
        if key in seen:
            continue
        seen.add(key)
        if path.is_file():
            return path
    return None


def resolve_import_task(
    yaml_meta: YamlMeta | None,
    user_choice: str | None,
) -> TaskResolution:
    choice = normalize_task(user_choice)
    if user_choice and str(user_choice).strip().lower() not in ("", "auto") and choice:
        return TaskResolution(task=choice, needs_user_choice=False, source="user")
    meta = yaml_meta or YamlMeta()
    if meta.task:
        return TaskResolution(task=meta.task, needs_user_choice=False, source="yaml_task")
    if meta.kpt_shape:
        return TaskResolution(task=TASK_POSE, needs_user_choice=False, source="yaml_kpt_shape")
    return TaskResolution(task=None, needs_user_choice=True, source="")


def aabb_from_obb_corners(
    corners: Sequence[Sequence[float]],
) -> tuple[float, float, float, float]:
    xs = [float(c[0]) for c in corners]
    ys = [float(c[1]) for c in corners]
    xmin, xmax = min(xs), max(xs)
    ymin, ymax = min(ys), max(ys)
    return ((xmin + xmax) / 2.0, (ymin + ymax) / 2.0, xmax - xmin, ymax - ymin)


def default_kpt_names(k: int) -> list[str]:
    return [f"kpt_{i}" for i in range(1, int(k) + 1)]


def format_export_skip_message(
    written: int,
    skipped: Mapping[str, int] | None,
    task_label: str,
) -> str:
    text = f"已导出 {int(written)} 条{task_label}"
    parts = []
    for kind, count in (skipped or {}).items():
        if count:
            parts.append(f"{int(count)} 条{KIND_LABELS.get(kind, kind)}")
    if parts:
        text += f"，跳过 {'、'.join(parts)}"
    return text


def _split_parts(line: str | Sequence[str]) -> list[str]:
    if isinstance(line, str):
        return line.split()
    return [str(p) for p in line]


def infer_kpt_shape_from_n(n: int) -> tuple[int, int] | None:
    """n = number of floats after class_id. Prefer 3-dim keypoints."""
    rest = n - 4
    if rest <= 0:
        return None
    if rest % 3 == 0:
        return (rest // 3, 3)
    if rest % 2 == 0:
        return (rest // 2, 2)
    return None


def _bbox_from_polygon(polygon: Sequence[tuple[float, float]]) -> tuple[float, float, float, float]:
    xs = [p[0] for p in polygon]
    ys = [p[1] for p in polygon]
    xmin, xmax = min(xs), max(xs)
    ymin, ymax = min(ys), max(ys)
    return ((xmin + xmax) / 2.0, (ymin + ymax) / 2.0, xmax - xmin, ymax - ymin)


def parse_yolo_line(
    line: str | Sequence[str],
    task: str,
    kpt_shape: tuple[int, int] | None = None,
) -> YoloLineResult:
    parts = _split_parts(line)
    if len(parts) < 2:
        return YoloLineResult(ok=False, error="行过短")
    try:
        class_id = int(float(parts[0]))
        values = [float(p) for p in parts[1:]]
    except (TypeError, ValueError):
        return YoloLineResult(ok=False, error="无法解析数字")
    n = len(values)
    task_norm = normalize_task(task) or task

    if task_norm == TASK_DETECT:
        if n != 4:
            return YoloLineResult(ok=False, error=f"检测行需要 4 个数，实际 {n}")
        bbox = (values[0], values[1], values[2], values[3])
        return YoloLineResult(ok=True, class_id=class_id, kind="bbox", bbox=bbox)

    if task_norm == TASK_OBB:
        if n != 8:
            return YoloLineResult(ok=False, error=f"OBB 行需要 8 个角点坐标，实际 {n}")
        corners = (
            (values[0], values[1]),
            (values[2], values[3]),
            (values[4], values[5]),
            (values[6], values[7]),
        )
        return YoloLineResult(
            ok=True,
            class_id=class_id,
            kind="obb",
            bbox=aabb_from_obb_corners(corners),
            obb=corners,
        )

    if task_norm == TASK_SEGMENT:
        if n < 6 or n % 2 != 0:
            return YoloLineResult(ok=False, error=f"分割行需要偶数个坐标且至少 6 个数，实际 {n}")
        polygon = [(values[i], values[i + 1]) for i in range(0, n, 2)]
        return YoloLineResult(
            ok=True,
            class_id=class_id,
            kind="polygon",
            bbox=_bbox_from_polygon(polygon),
            polygon=polygon,
        )

    if task_norm == TASK_POSE:
        inferred = None
        shape = kpt_shape
        if shape is None:
            inferred = infer_kpt_shape_from_n(n)
            shape = inferred
        if shape is None:
            return YoloLineResult(ok=False, error=f"无法从行宽 {n} 推断 kpt_shape")
        k, d = shape
        expected = 4 + k * d
        if n != expected:
            return YoloLineResult(
                ok=False,
                error=f"姿态行需要 {expected} 个数（kpt_shape={list(shape)}），实际 {n}",
            )
        bbox = (values[0], values[1], values[2], values[3])
        rest = values[4:]
        keypoints: list[tuple[float, float, int]] = []
        warning = None
        if d == 3:
            for i in range(k):
                x, y, v_raw = rest[i * 3], rest[i * 3 + 1], rest[i * 3 + 2]
                v = int(round(v_raw))
                if v not in (0, 1, 2):
                    v = 0
                    warning = "可见性超出 0/1/2，已记为 0"
                keypoints.append((x, y, v))
        else:
            for i in range(k):
                keypoints.append((rest[i * 2], rest[i * 2 + 1], 2))
        return YoloLineResult(
            ok=True,
            class_id=class_id,
            kind="pose",
            bbox=bbox,
            keypoints=keypoints,
            inferred_kpt_shape=inferred,
            warning=warning,
        )

    return YoloLineResult(ok=False, error=f"未知任务: {task}")


def quantize6(value: float) -> float:
    return float(f"{float(value):.6f}")


def format_yolo_floats(values: Iterable[float]) -> str:
    return " ".join(f"{float(v):.6f}" for v in values)
