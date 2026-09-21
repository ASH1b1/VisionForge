"""ProjectDocument — unified project data model with built-in undo/redo.

Replaces the scattered instance variables in MainWindow and absorbs
AnnotationState's undo/redo stack into a single QObject-based model.
"""
from __future__ import annotations

import copy
import json
import os
import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Set

import cv2
from PySide6.QtCore import QObject, Signal


# ============================================================================
# Data classes (migrated from annotation_state.py)
# ============================================================================

def _infer_annotation_kind(
    kind: str | None,
    polygon: list | None,
    obb: Any = None,
    keypoints: Any = None,
) -> str:
    if kind in ("bbox", "polygon", "obb", "pose"):
        return kind
    if obb:
        return "obb"
    if keypoints:
        return "pose"
    if polygon and len(polygon) >= 3:
        return "polygon"
    return "bbox"


def _as_obb_corners(value: Any) -> tuple[tuple[float, float], ...] | None:
    if not value:
        return None
    corners = []
    for point in value:
        if point is None or len(point) < 2:
            continue
        corners.append((float(point[0]), float(point[1])))
    if len(corners) != 4:
        return None
    return tuple(corners)


def _as_keypoints(value: Any) -> list[tuple[float, float, int]] | None:
    if not value:
        return None
    points: list[tuple[float, float, int]] = []
    for point in value:
        if point is None or len(point) < 2:
            continue
        x = float(point[0])
        y = float(point[1])
        v = int(point[2]) if len(point) >= 3 else 2
        if v not in (0, 1, 2):
            v = 0
        points.append((x, y, v))
    return points or None


def _as_kpt_shape(value: Any) -> tuple[int, int] | None:
    if not value or not isinstance(value, (list, tuple)) or len(value) < 2:
        return None
    try:
        k = int(value[0])
        d = int(value[1])
    except (TypeError, ValueError):
        return None
    if k > 0 and d in (2, 3):
        return (k, d)
    return None


@dataclass
class Annotation:
    """Single annotation."""

    id: int
    class_id: int
    confidence: float
    bbox: tuple  # (x, y, w, h) normalized
    polygon: list | None = None
    visible: bool = True
    kind: str = "bbox"
    obb: tuple[tuple[float, float], ...] | None = None
    keypoints: list[tuple[float, float, int]] | None = None

    def clone(self) -> "Annotation":
        return Annotation(
            id=self.id,
            class_id=self.class_id,
            confidence=self.confidence,
            bbox=self.bbox,
            polygon=copy.deepcopy(self.polygon),
            visible=self.visible,
            kind=self.kind,
            obb=copy.deepcopy(self.obb),
            keypoints=copy.deepcopy(self.keypoints),
        )

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, Annotation):
            return False
        return self.id == other.id

    def to_dict(self) -> dict:
        data = {
            "id": self.id,
            "class_id": self.class_id,
            "confidence": self.confidence,
            "bbox": list(self.bbox),
            "polygon": self.polygon,
            "visible": self.visible,
            "kind": self.kind,
        }
        if self.obb is not None:
            data["obb"] = [list(p) for p in self.obb]
        if self.keypoints is not None:
            data["keypoints"] = [list(p) for p in self.keypoints]
        return data

    @classmethod
    def from_dict(cls, d: dict) -> "Annotation":
        polygon = d.get("polygon")
        obb = _as_obb_corners(d.get("obb"))
        keypoints = _as_keypoints(d.get("keypoints"))
        return cls(
            id=d["id"],
            class_id=d["class_id"],
            confidence=d.get("confidence", 1.0),
            bbox=tuple(d["bbox"]),
            polygon=polygon,
            visible=d.get("visible", True),
            kind=_infer_annotation_kind(d.get("kind"), polygon, obb, keypoints),
            obb=obb,
            keypoints=keypoints,
        )


@dataclass
class Action:
    """Undo/redo action record."""

    type: str  # "add", "delete", "modify", "batch", "clear"
    annotations: List[Annotation] = field(default_factory=list)
    previous_state: Optional[List[Annotation]] = None
    modified_ids: Optional[Dict[int, Dict[str, Any]]] = None
    new_values: Optional[Dict[int, Dict[str, Any]]] = None  # for redo of "modify"
    description: str = ""


# ============================================================================
# ProjectDocument
# ============================================================================

class ProjectDocument(QObject):
    """Unified project data model — single source of truth.

    Holds all project data and a per-frame undo/redo stack.
    Emits signals on every mutation so Views can react.
    """

    # --- Signals ---
    data_loaded = Signal()
    data_cleared = Signal()
    annotations_changed = Signal(int)  # frame_index
    current_annotations_changed = Signal(list)
    selection_changed = Signal(set)
    class_added = Signal(int, str)
    class_removed = Signal(int)
    class_renamed = Signal(int, str)
    split_changed = Signal()
    dirty_changed = Signal(bool)
    history_changed = Signal()
    action_performed = Signal(str)

    def __init__(self, parent: QObject | None = None):
        super().__init__(parent)
        self.image_files: List[Path] = []
        self.frame_annotations: Dict[int, List[Annotation]] = {}
        self.class_names: Dict[int, str] = {}
        self.split_map: Dict[int, str] = {}
        self.video_capture: Optional[cv2.VideoCapture] = None
        # Optional video source metadata for .gsproj round-trip (D3).
        # Keys: path (Path), fps, total_frames, width, height.
        self.video: Optional[Dict[str, Any]] = None
        self.project_path: Optional[Path] = None
        self.kpt_shape: Optional[tuple[int, int]] = None
        self.kpt_names: Optional[List[str]] = None
        self._is_modified: bool = False

        self.current_annotations: List[Annotation] = []
        self._selected_ids: Set[int] = set()
        self._hidden_classes: Set[int] = set()
        self._current_frame_index: int = -1

        self._histories: Dict[int, List[Action]] = {}
        self._history_indices: Dict[int, int] = {}
        self._next_id: int = 1
        self._history_enabled: bool = True

    @property
    def is_modified(self) -> bool:
        return self._is_modified

    @is_modified.setter
    def is_modified(self, value: bool) -> None:
        if self._is_modified != value:
            self._is_modified = value
            self.dirty_changed.emit(value)

    @property
    def can_undo(self) -> bool:
        return self._history_index >= 0

    @property
    def can_redo(self) -> bool:
        return self._history_index < len(self._history) - 1

    @property
    def _history_frame(self) -> int:
        return self._current_frame_index if self._current_frame_index >= 0 else 0

    @property
    def _history(self) -> List[Action]:
        return self._histories.setdefault(self._history_frame, [])

    @_history.setter
    def _history(self, value: List[Action]) -> None:
        self._histories[self._history_frame] = value

    @property
    def _history_index(self) -> int:
        return self._history_indices.get(self._history_frame, -1)

    @_history_index.setter
    def _history_index(self, value: int) -> None:
        self._history_indices[self._history_frame] = value

    @property
    def selected_ids(self) -> Set[int]:
        return self._selected_ids.copy()

    @staticmethod
    def _coerce_annotation(ann_data: Any) -> Annotation:
        """Normalize frame storage (Annotation object or dict) to Annotation."""
        if isinstance(ann_data, Annotation):
            return ann_data.clone()
        if isinstance(ann_data, dict):
            return Annotation.from_dict(ann_data)
        raise TypeError(f"Unsupported annotation storage type: {type(ann_data)!r}")

    def switch_to_frame(self, frame_index: int) -> None:
        """Save current-frame work back to frame_annotations, load new frame."""
        # Save current annotations to current frame
        self._sync_current_to_frame()
        # Load new frame's annotations (dict or Annotation)
        self.current_annotations = [
            self._coerce_annotation(a)
            for a in self.frame_annotations.get(frame_index, [])
        ]
        self._current_frame_index = frame_index
        self._selected_ids.clear()
        self.current_annotations_changed.emit(self.visible_annotations)
        self.selection_changed.emit(set())

    def _sync_current_to_frame(self) -> None:
        """Write current_annotations to frame_annotations for the active frame."""
        idx = self._current_frame_index if self._current_frame_index >= 0 else 0
        self.frame_annotations[idx] = [
            a.clone() for a in self.current_annotations
        ]

    def apply_polygons_to_frame(
        self, frame_index: int, pairs: List[tuple], *, record_history: bool = False
    ) -> int:
        """Apply polygons to a stored frame without requiring UI frame switch.

        pairs: [(ann_id, polygon_points|None), ...]. None polygons are skipped.
        Returns number of annotations updated.
        When record_history=True, pushes one undo Action onto the *target*
        frame's history stack (so undo works after switching to that frame).
        """
        if not pairs:
            return 0

        raw = self.frame_annotations.get(frame_index, [])
        anns = [self._coerce_annotation(a) for a in raw]
        by_id = {ann.id: ann for ann in anns}
        old_map: Dict[int, Dict[str, Any]] = {}
        new_map: Dict[int, Dict[str, Any]] = {}
        updated = 0
        for ann_id, polygon in pairs:
            if polygon is None:
                continue
            target = by_id.get(ann_id)
            if target is None:
                continue
            old_map[ann_id] = {"polygon": copy.deepcopy(target.polygon)}
            new_map[ann_id] = {"polygon": copy.deepcopy(polygon)}
            target.polygon = copy.deepcopy(polygon)
            updated += 1

        if not updated:
            return 0

        self.frame_annotations[frame_index] = anns
        self._mark_dirty()

        if record_history and self._history_enabled:
            self._push_action_for_frame(
                frame_index,
                Action(
                    type="modify",
                    annotations=[],
                    modified_ids=old_map,
                    new_values=new_map,
                    description=f"补分割 {updated} 个多边形",
                ),
            )

        # Keep in-memory current view in sync when editing the active frame
        active = self._current_frame_index if self._current_frame_index >= 0 else 0
        if frame_index == active:
            self.current_annotations = [a.clone() for a in anns]
            self.current_annotations_changed.emit(self.visible_annotations)

        return updated

    @property
    def visible_annotations(self) -> List[Annotation]:
        return [
            a for a in self.current_annotations
            if a.visible and a.class_id not in self._hidden_classes
        ]

    def enable_history(self, enabled: bool = True) -> None:
        self._history_enabled = enabled

    def add_annotation(
        self, class_id: int, bbox: tuple, confidence: float = 1.0,
        polygon: list | None = None, *, record_history: bool = True,
    ) -> Annotation:
        polygon = copy.deepcopy(polygon)
        ann = Annotation(
            id=self._next_id, class_id=class_id, confidence=confidence,
            bbox=bbox, polygon=polygon,
            kind=_infer_annotation_kind(None, polygon),
        )
        self._next_id += 1
        self.current_annotations.append(ann)
        if record_history and self._history_enabled:
            self._push_action(Action(
                type="add", annotations=[ann.clone()],
                description=f"添加标注 [{ann.id}]",
            ))
        self._mark_dirty()
        self._sync_current_to_frame()
        self.current_annotations_changed.emit(self.visible_annotations)
        return ann

    def add_annotations_batch(
        self, annotations_data: List[tuple], *, record_history: bool = True,
    ) -> List[Annotation]:
        added: List[Annotation] = []
        for data in annotations_data:
            if len(data) == 3:
                class_id, bbox, confidence = data
                polygon = None
            else:
                class_id, bbox, confidence, polygon = data
            polygon = copy.deepcopy(polygon)
            ann = Annotation(
                id=self._next_id, class_id=class_id, confidence=confidence,
                bbox=bbox, polygon=polygon,
                kind=_infer_annotation_kind(None, polygon),
            )
            self._next_id += 1
            self.current_annotations.append(ann)
            added.append(ann)
        if record_history and self._history_enabled and added:
            self._push_action(Action(
                type="add", annotations=[a.clone() for a in added],
                description=f"批量添加 {len(added)} 个标注",
            ))
        self._mark_dirty()
        self._sync_current_to_frame()
        self.current_annotations_changed.emit(self.visible_annotations)
        return added

    def delete_annotations(self, ids: Set[int], *, record_history: bool = True) -> None:
        if not ids:
            return
        if record_history and self._history_enabled:
            prev = [a.clone() for a in self.current_annotations if a.id in ids]
            self._push_action(Action(
                type="delete", annotations=prev,
                description=f"删除 {len(ids)} 个标注",
            ))
        self.current_annotations = [a for a in self.current_annotations if a.id not in ids]
        self._selected_ids -= ids
        self._mark_dirty()
        self._sync_current_to_frame()
        self.current_annotations_changed.emit(self.visible_annotations)
        self.selection_changed.emit(self._selected_ids)

    def update_annotation(self, ann_id: int, **kwargs: Any) -> None:
        old_values: Dict[str, Any] = {}
        new_values: Dict[str, Any] = {}
        target: Optional[Annotation] = None
        for ann in self.current_annotations:
            if ann.id == ann_id:
                target = ann
                break
        if target is None:
            return
        for key, value in kwargs.items():
            if hasattr(target, key):
                old_values[key] = copy.deepcopy(getattr(target, key))
                new_values[key] = copy.deepcopy(value)
                setattr(target, key, copy.deepcopy(value))
        if old_values and self._history_enabled:
            self._push_action(Action(
                type="modify", annotations=[],
                modified_ids={ann_id: old_values},
                new_values={ann_id: new_values},
                description=f"修改标注 [{ann_id}]",
            ))
        self._mark_dirty()
        self._sync_current_to_frame()
        self.current_annotations_changed.emit(self.visible_annotations)

    def apply_polygons_batch(
        self, pairs: List[tuple], *, record_history: bool = True
    ) -> int:
        """Apply polygons to current-frame annotations by id (one undo point).

        pairs: [(ann_id, polygon_points|None), ...]. None polygons are skipped.
        Returns number of annotations updated.
        """
        if not pairs:
            return 0

        old_map: Dict[int, Dict[str, Any]] = {}
        new_map: Dict[int, Dict[str, Any]] = {}
        by_id = {ann.id: ann for ann in self.current_annotations}
        for ann_id, polygon in pairs:
            if polygon is None:
                continue
            target = by_id.get(ann_id)
            if target is None:
                continue
            old_map[ann_id] = {"polygon": copy.deepcopy(target.polygon)}
            new_map[ann_id] = {"polygon": copy.deepcopy(polygon)}
            target.polygon = copy.deepcopy(polygon)

        if not new_map:
            return 0

        if record_history and self._history_enabled:
            self._push_action(Action(
                type="modify",
                annotations=[],
                modified_ids=old_map,
                new_values=new_map,
                description=f"补分割 {len(new_map)} 个多边形",
            ))
        self._mark_dirty()
        self._sync_current_to_frame()
        self.current_annotations_changed.emit(self.visible_annotations)
        return len(new_map)

    def set_selection(self, ids: Set[int]) -> None:
        self._selected_ids = ids.copy()
        self.selection_changed.emit(self._selected_ids)

    def clear_selection(self) -> None:
        self.set_selection(set())

    def clear_annotations(self, *, record_history: bool = True) -> None:
        if record_history and self._history_enabled and self.current_annotations:
            self._push_action(Action(
                type="clear",
                annotations=[a.clone() for a in self.current_annotations],
                description=f"清空 {len(self.current_annotations)} 个标注",
            ))
        self.current_annotations.clear()
        self._selected_ids.clear()
        self._mark_dirty()
        self._sync_current_to_frame()
        self.current_annotations_changed.emit([])
        self.selection_changed.emit(set())

    def undo(self) -> None:
        if not self.can_undo:
            return
        action = self._history[self._history_index]
        self._apply_reverse(action)
        self._history_index -= 1
        self._mark_dirty()
        self._sync_current_to_frame()
        self.history_changed.emit()
        self.current_annotations_changed.emit(self.visible_annotations)
        self.action_performed.emit(f"撤销: {action.description}")

    def redo(self) -> None:
        if not self.can_redo:
            return
        self._history_index += 1
        action = self._history[self._history_index]
        self._apply_forward(action)
        self._mark_dirty()
        self._sync_current_to_frame()
        self.history_changed.emit()
        self.current_annotations_changed.emit(self.visible_annotations)
        self.action_performed.emit(f"重做: {action.description}")

    def get_history(self) -> List[Action]:
        return self._history.copy()

    def clear_history(self) -> None:
        self._histories.clear()
        self._history_indices.clear()
        self.history_changed.emit()

    def add_class(self, class_id: int, name: str) -> None:
        self.class_names[class_id] = name
        self._mark_dirty()
        self.class_added.emit(class_id, name)

    def rename_class(self, class_id: int, new_name: str) -> None:
        if class_id in self.class_names:
            self.class_names[class_id] = new_name
            self._mark_dirty()
            self.class_renamed.emit(class_id, new_name)

    def remove_class(self, class_id: int) -> None:
        if class_id in self.class_names:
            del self.class_names[class_id]
            for frame_idx in self.frame_annotations:
                self.frame_annotations[frame_idx] = [
                    a for a in self.frame_annotations[frame_idx]
                    if a.class_id != class_id
                ]
            self.current_annotations = [
                a for a in self.current_annotations if a.class_id != class_id
            ]
            self._mark_dirty()
            self.class_removed.emit(class_id)
            self.current_annotations_changed.emit(self.visible_annotations)

    def set_split_map(self, split_map: Dict[int, str]) -> None:
        self.split_map = dict(split_map)
        self._mark_dirty()
        self.split_changed.emit()

    def toggle_class_visibility(self, class_id: int, visible: bool) -> None:
        if visible:
            self._hidden_classes.discard(class_id)
        else:
            self._hidden_classes.add(class_id)
        # Class hide is session UI state only — never mutate ann.visible
        # (that field is persisted in .gsproj and must stay True).
        self.current_annotations_changed.emit(self.visible_annotations)

    @staticmethod
    def _path_for_storage(path: Path, base_dir: Optional[Path]) -> str:
        """Prefer path relative to .gsproj parent; fall back to absolute."""
        path = Path(path)
        if base_dir is None:
            return str(path.resolve()) if (path.is_absolute() or bool(path.anchor)) else str(path)
        try:
            return path.resolve().relative_to(Path(base_dir).resolve()).as_posix()
        except ValueError:
            # Drive-qualify on Windows so rooted paths like /a/img.jpg round-trip
            # (pathlib Path.is_absolute() is False without a drive letter).
            return str(path.resolve())

    @staticmethod
    def _resolve_stored_path(path_str: str, base_dir: Optional[Path]) -> Path:
        """Resolve relative paths against base_dir; keep absolute as-is."""
        p = Path(path_str)
        # Windows: rooted paths without a drive (e.g. \a\img.jpg) are not
        # is_absolute(); resolve() drive-qualifies them instead of joining base.
        if p.is_absolute() or bool(p.anchor):
            return p.resolve() if p.anchor and not p.is_absolute() else p
        if base_dir is not None:
            return (Path(base_dir) / p).resolve()
        return p

    @staticmethod
    def _normalize_video_meta(video: Optional[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
        if not video or not video.get("path"):
            return None
        return {
            "path": Path(video["path"]),
            "fps": float(video.get("fps", 30.0) or 30.0),
            "total_frames": int(video.get("total_frames", 0) or 0),
            "width": int(video.get("width", 0) or 0),
            "height": int(video.get("height", 0) or 0),
        }

    def to_dict(self, *, base_dir: Optional[Path] = None) -> dict:
        # Flush current frame before serializing
        self._sync_current_to_frame()
        if base_dir is None and self.project_path is not None:
            base_dir = Path(self.project_path).parent
        data: Dict[str, Any] = {
            "version": 4,
            "image_files": [
                self._path_for_storage(p, base_dir) for p in self.image_files
            ],
            "frame_annotations": {
                str(k): [self._coerce_annotation(a).to_dict() for a in v]
                for k, v in self.frame_annotations.items()
            },
            "class_names": {str(k): v for k, v in self.class_names.items()},
            "split_map": {str(k): v for k, v in self.split_map.items()},
            "current_frame": self._current_frame_index if self._current_frame_index >= 0 else 0,
            "kpt_shape": list(self.kpt_shape) if self.kpt_shape else None,
            "kpt_names": list(self.kpt_names) if self.kpt_names else None,
        }
        video = self._normalize_video_meta(self.video)
        if video is not None:
            data["video"] = {
                "path": self._path_for_storage(video["path"], base_dir),
                "fps": video["fps"],
                "total_frames": video["total_frames"],
                "width": video["width"],
                "height": video["height"],
            }
        return data

    @classmethod
    def from_dict(cls, d: dict, *, base_dir: Optional[Path] = None) -> "ProjectDocument":
        doc = cls()
        _IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".tif", ".webp"}
        doc.image_files = [
            cls._resolve_stored_path(p, base_dir)
            for p in d.get("image_files", [])
            if Path(p).suffix.lower() in _IMAGE_EXTS
        ]
        doc.frame_annotations = {
            int(k): [Annotation.from_dict(a) for a in v]
            for k, v in d.get("frame_annotations", {}).items()
        }
        doc.class_names = {int(k): v for k, v in d.get("class_names", {}).items()}
        doc.split_map = {int(k): v for k, v in d.get("split_map", {}).items()}
        doc.kpt_shape = _as_kpt_shape(d.get("kpt_shape"))
        names = d.get("kpt_names")
        doc.kpt_names = [str(n) for n in names] if isinstance(names, list) and names else None
        doc._current_frame_index = d.get("current_frame", 0)
        raw_video = d.get("video")
        if isinstance(raw_video, dict) and raw_video.get("path"):
            resolved = dict(raw_video)
            resolved["path"] = cls._resolve_stored_path(str(raw_video["path"]), base_dir)
            doc.video = cls._normalize_video_meta(resolved)
        return doc

    def save_to_file(self, path: Path) -> None:
        path = Path(path)
        base_dir = path.parent
        data = json.dumps(self.to_dict(base_dir=base_dir), ensure_ascii=False, indent=2)
        path.parent.mkdir(parents=True, exist_ok=True)
        temp_path: Optional[Path] = None
        try:
            with tempfile.NamedTemporaryFile(
                mode="w",
                encoding="utf-8",
                dir=path.parent,
                prefix=f".{path.name}.",
                suffix=".tmp",
                delete=False,
            ) as temp_file:
                temp_file.write(data)
                temp_file.flush()
                os.fsync(temp_file.fileno())
                temp_path = Path(temp_file.name)
            os.replace(temp_path, path)
        except Exception:
            if temp_path is not None:
                temp_path.unlink(missing_ok=True)
            raise
        self.project_path = path
        self.is_modified = False

    def load_from_file(self, path: Path) -> None:
        path = Path(path)
        data = json.loads(path.read_text(encoding="utf-8"))
        doc = ProjectDocument.from_dict(data, base_dir=path.parent)
        self.replace_data(
            image_files=doc.image_files,
            frame_annotations=doc.frame_annotations,
            class_names=doc.class_names,
            split_map=doc.split_map,
            current_frame_index=doc._current_frame_index,
            project_path=path,
            video=doc.video,
            kpt_shape=doc.kpt_shape,
            kpt_names=doc.kpt_names,
            emit_loaded=False,
        )
        self.data_loaded.emit()

    def replace_data(
        self,
        *,
        image_files: Optional[List[Path]] = None,
        frame_annotations: Optional[Dict[int, List[Any]]] = None,
        class_names: Optional[Dict[int, str]] = None,
        split_map: Optional[Dict[int, str]] = None,
        current_frame_index: int = 0,
        project_path: Optional[Path] = None,
        video_capture: Optional[cv2.VideoCapture] = None,
        video: Optional[Dict[str, Any]] = None,
        kpt_shape: Optional[tuple[int, int]] = None,
        kpt_names: Optional[List[str]] = None,
        emit_loaded: bool = True,
    ) -> None:
        """Atomically replace the active project session with normalized data."""
        normalized_frames = {
            int(frame_idx): [self._coerce_annotation(a) for a in annotations]
            for frame_idx, annotations in (frame_annotations or {}).items()
        }

        old_capture = self.video_capture
        if old_capture is not None and old_capture is not video_capture:
            old_capture.release()

        self.image_files = [Path(p) for p in (image_files or [])]
        self.frame_annotations = normalized_frames
        self.class_names = dict(class_names or {})
        self.split_map = dict(split_map or {})
        self.video_capture = video_capture
        self.video = self._normalize_video_meta(video)
        self.kpt_shape = _as_kpt_shape(kpt_shape) if kpt_shape is not None else None
        self.kpt_names = list(kpt_names) if kpt_names else None
        self.project_path = Path(project_path) if project_path is not None else None
        self._current_frame_index = current_frame_index
        self.current_annotations = [
            a.clone() for a in self.frame_annotations.get(current_frame_index, [])
        ]
        self._selected_ids.clear()
        self._hidden_classes.clear()
        self.clear_history()

        all_ids = [
            ann.id
            for annotations in self.frame_annotations.values()
            for ann in annotations
        ]
        self._next_id = max(all_ids, default=0) + 1
        self.is_modified = False
        self.current_annotations_changed.emit(self.visible_annotations)
        self.selection_changed.emit(set())
        if emit_loaded:
            self.data_loaded.emit()

    def _mark_dirty(self) -> None:
        self.is_modified = True

    def _push_action(self, action: Action) -> None:
        self._push_action_for_frame(self._history_frame, action)

    def _push_action_for_frame(self, frame_index: int, action: Action) -> None:
        history = self._histories.setdefault(frame_index, [])
        index = self._history_indices.get(frame_index, -1)
        history = history[: index + 1]
        history.append(action)
        self._histories[frame_index] = history
        self._history_indices[frame_index] = index + 1
        self.history_changed.emit()
        self.action_performed.emit(action.description)

    def _apply_reverse(self, action: Action) -> None:
        if action.type == "add":
            ids_to_remove = {a.id for a in action.annotations}
            self.current_annotations = [
                a for a in self.current_annotations if a.id not in ids_to_remove
            ]
            self._next_id = max([a.id for a in self.current_annotations] + [0]) + 1
        elif action.type == "delete":
            for ann in action.annotations:
                ann.visible = True
                if ann not in self.current_annotations:
                    self.current_annotations.append(ann)
            self._next_id = max([a.id for a in self.current_annotations] + [0]) + 1
        elif action.type == "modify":
            for ann_id, old_values in (action.modified_ids or {}).items():
                for ann in self.current_annotations:
                    if ann.id == ann_id:
                        for key, value in old_values.items():
                            if hasattr(ann, key):
                                setattr(ann, key, copy.deepcopy(value))
                        break
        elif action.type == "clear":
            for ann in action.annotations:
                ann.visible = True
                if ann not in self.current_annotations:
                    self.current_annotations.append(ann)
            self._next_id = max([a.id for a in self.current_annotations] + [0]) + 1

    def _apply_forward(self, action: Action) -> None:
        if action.type == "add":
            for ann in action.annotations:
                if ann not in self.current_annotations:
                    self.current_annotations.append(ann.clone())
        elif action.type == "delete":
            ids_to_remove = {a.id for a in action.annotations}
            self.current_annotations = [
                a for a in self.current_annotations if a.id not in ids_to_remove
            ]
        elif action.type == "modify":
            for ann_id, new_values in (action.new_values or {}).items():
                for ann in self.current_annotations:
                    if ann.id == ann_id:
                        for key, value in new_values.items():
                            if hasattr(ann, key):
                                setattr(ann, key, copy.deepcopy(value))
                        break
        elif action.type == "clear":
            self.current_annotations.clear()
