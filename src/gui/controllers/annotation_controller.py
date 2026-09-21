"""AnnotationController — CRUD, undo/redo, class management."""
from __future__ import annotations
import copy
from typing import Optional, Set
from PySide6.QtCore import QObject
from ..models.project_document import ProjectDocument
from ..models.app_state import AppStateManager


class AnnotationController(QObject):
    """Orchestrates all annotation mutations via ProjectDocument."""

    def __init__(self, project: ProjectDocument, app_state: AppStateManager,
                 parent: QObject | None = None):
        super().__init__(parent)
        self._project = project
        self._app_state = app_state

    def add_annotation(self, class_id: int, bbox: tuple,
                       polygon: list | None = None, *, confidence: float = 1.0,
                       record_history: bool = True):
        return self._project.add_annotation(class_id, bbox, confidence=confidence,
                                            polygon=polygon, record_history=record_history)

    def add_annotations_batch(self, annotations_data: list, *, record_history: bool = True):
        return self._project.add_annotations_batch(annotations_data, record_history=record_history)

    def delete_annotation(self, ann_id: int) -> None:
        self._project.delete_annotations({ann_id})

    def delete_selected(self) -> None:
        if self._project.selected_ids:
            self._project.delete_annotations(self._project.selected_ids)

    def modify_annotation(self, ann_id: int, new_bbox: tuple) -> None:
        self._project.update_annotation(ann_id, bbox=new_bbox)

    def edit_polygon(self, ann_id: int, polygon: list, bbox: tuple) -> None:
        self._project.update_annotation(
            ann_id,
            polygon=copy.deepcopy(list(polygon)),
            bbox=tuple(bbox),
            kind="polygon",
        )

    def modify_annotation_class(self, ann_id: int, new_class: int) -> None:
        self._project.update_annotation(ann_id, class_id=new_class)

    def select_annotation(self, ann_id: int) -> None:
        if ann_id >= 0:
            self._project.set_selection({ann_id})
        else:
            self._project.clear_selection()

    def undo(self) -> None:
        self._project.undo()

    def redo(self) -> None:
        self._project.redo()

    def change_frame(self, frame_index: int) -> None:
        """Switch current frame, persisting changes from the previous frame."""
        # Update app_state first so canvas knows which frame we're on
        self._app_state.set_frame(frame_index)
        self._project.switch_to_frame(frame_index)

    def add_class(self, name: str) -> Optional[int]:
        existing = self._project.class_names
        class_id = max(existing.keys()) + 1 if existing else 0
        self._project.add_class(class_id, name)
        return class_id

    def rename_class(self, class_id: int, new_name: str) -> None:
        self._project.rename_class(class_id, new_name)

    def delete_class(self, class_id: int) -> None:
        self._project.remove_class(class_id)

    def replace_class(self, source_class_id: int, target_name: str,
                      scope: str = "current") -> int:
        count = 0
        target_id = None
        for cid, cname in self._project.class_names.items():
            if cname == target_name:
                target_id = cid
                break
        if target_id is None:
            target_id = self.add_class(target_name)
        if scope == "current":
            for ann in self._project.current_annotations:
                if ann.class_id == source_class_id:
                    ann.class_id = target_id
                    count += 1
        elif scope == "all":
            for frame_anns in self._project.frame_annotations.values():
                for ann in frame_anns:
                    if ann.class_id == source_class_id:
                        ann.class_id = target_id
                        count += 1
            for ann in self._project.current_annotations:
                if ann.class_id == source_class_id:
                    ann.class_id = target_id
                    count += 1
        if count > 0:
            self._project._mark_dirty()
            self._project.current_annotations_changed.emit(
                self._project.visible_annotations)
        return count

    def toggle_class_visibility(self, class_id: int, visible: bool) -> None:
        self._project.toggle_class_visibility(class_id, visible)

    def apply_polygons_batch(
        self, pairs: list, *, record_history: bool = True
    ) -> int:
        """Apply polygons to annotations by id. One undo point when record_history.

        pairs: [(ann_id, polygon_points|None), ...]
        Skips None polygons. Returns number updated.
        """
        return self._project.apply_polygons_batch(pairs, record_history=record_history)

    def apply_polygons_to_frame(
        self, frame_index: int, pairs: list, *, record_history: bool = False
    ) -> int:
        """Apply polygons to a stored frame (dict or Annotation) without UI switch."""
        return self._project.apply_polygons_to_frame(
            frame_index, pairs, record_history=record_history
        )

    def save_current_frame_to_project(self) -> None:
        idx = self._app_state._state.current_frame
        self._project.frame_annotations[idx] = [
            a.clone() for a in self._project.current_annotations]

    @property
    def current_annotations(self):
        return self._project.visible_annotations

    @property
    def selected_ids(self) -> Set[int]:
        return self._project.selected_ids
