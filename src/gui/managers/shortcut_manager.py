from PySide6.QtWidgets import QApplication
from PySide6.QtCore import QObject, Signal
from PySide6.QtGui import QKeySequence, QShortcut

from ..models.app_state import Tool


class ShortcutManager(QObject):
    """快捷键管理器"""

    def __init__(self, main_window):
        super().__init__(main_window)
        self.window = main_window
        self._shortcuts = {}
        self._setup_global_shortcuts()
        self._setup_video_shortcuts()
        self._setup_tool_shortcuts()
        self._setup_annotation_shortcuts()
        self._setup_view_shortcuts()

    def _add_shortcut(self, key_sequence, callback, description=""):
        """添加快捷键"""
        shortcut = QShortcut(QKeySequence(key_sequence), self.window)
        shortcut.activated.connect(callback)
        self._shortcuts[key_sequence] = {"callback": callback, "description": description}

    def _setup_global_shortcuts(self):
        """全局快捷键"""
        self._add_shortcut("Ctrl+O", self.window._open_video, "Open Video")
        self._add_shortcut("Ctrl+Shift+O", self.window._open_images, "Open Images")
        self._add_shortcut("Ctrl+Shift+P", self.window._open_project, "Open Project")
        self._add_shortcut("Ctrl+S", self.window._save, "Save")
        self._add_shortcut("Ctrl+E", self.window._export_yolo_dataset, "Export YOLO Dataset")
        # Ctrl+Z / Ctrl+Y 只走菜单 QAction（QKeySequence.Undo/Redo）。
        # 这里再注册一次会和菜单快捷键冲突，Qt 会当成 Ambiguous shortcut，按下去没有反应。
        self._add_shortcut("Ctrl+B", lambda: self._toggle_dock(self.window.explorer_dock), "Toggle Explorer")
        self._add_shortcut("Ctrl+Shift+B", lambda: self._toggle_dock(self.window.inspector_dock), "Toggle Inspector")
        self._add_shortcut("F11", self.window._toggle_fullscreen, "Fullscreen")
        self._add_shortcut("Esc", self._cancel_action, "Cancel")
        self._add_shortcut("Ctrl+Q", self.window.close, "Quit")

    def _setup_video_shortcuts(self):
        """视频控制快捷键"""
        self._add_shortcut("Space", self._toggle_playback, "Play/Pause")
        self._add_shortcut("Left", self._previous_frame, "Previous Frame")
        self._add_shortcut("Right", self._next_frame, "Next Frame")
        self._add_shortcut("Shift+Left", lambda: self._skip_frames(-10), "Rewind 10 frames")
        self._add_shortcut("Shift+Right", lambda: self._skip_frames(10), "Forward 10 frames")
        self._add_shortcut("Home", self._first_frame, "First Frame")
        self._add_shortcut("End", self._last_frame, "Last Frame")
        self._add_shortcut("Up", self._previous_frame, "Previous Frame")
        self._add_shortcut("Down", self._next_frame, "Next Frame")

    def _setup_tool_shortcuts(self):
        """工具切换快捷键"""
        self._add_shortcut("V", lambda: self._select_tool(Tool.SELECT), "Select Tool")
        self._add_shortcut("R", lambda: self._select_tool(Tool.RECTANGLE), "Rectangle Tool")
        self._add_shortcut("P", lambda: self._select_tool(Tool.POLYGON), "Polygon Tool")
        self._add_shortcut("E", lambda: self._select_tool(Tool.ERASER), "Eraser Tool")
        self._add_shortcut("M", lambda: self._select_tool(Tool.MOVE), "Move Tool")
        self._add_shortcut("H", lambda: self._select_tool(Tool.HAND), "Hand Tool")
        self._add_shortcut("A", lambda: self._select_tool(Tool.SAM_CLICK), "SAM Click Tool")

    def _setup_annotation_shortcuts(self):
        """标注编辑快捷键"""
        self._add_shortcut("Delete", self._handle_delete, "Delete Selected")
        self._add_shortcut("Backspace", self._handle_backspace, "Delete Selected / Polygon Undo")
        self._add_shortcut("Ctrl+D", self._deselect_all, "Deselect All")
        self._add_shortcut("Ctrl+A", self._select_all, "Select All Annotations")
        self._add_shortcut("Tab", self._next_annotation, "Next Annotation")
        self._add_shortcut("Shift+Tab", self._prev_annotation, "Previous Annotation")
        self._add_shortcut("Enter", self._handle_enter, "Edit Selected / Finish Polygon")
        self._add_shortcut("Return", self._handle_enter, "Confirm SAM preview / Finish Polygon")

        # 类别快捷键 1-9
        for i in range(1, 10):
            self._add_shortcut(str(i), lambda idx=i: self._select_class(idx - 1), f"Select Class {i}")
        self._add_shortcut("0", lambda: self._select_class(9), "Select Class 10")

    def _setup_view_shortcuts(self):
        """视图操作快捷键"""
        self._add_shortcut("Ctrl+0", self._reset_view, "Reset View")
        self._add_shortcut("Ctrl++", self._zoom_in, "Zoom In")
        self._add_shortcut("Ctrl+=", self._zoom_in, "Zoom In")
        self._add_shortcut("Ctrl+-", self._zoom_out, "Zoom Out")
        self._add_shortcut("Ctrl+/", self._show_shortcuts, "Show Shortcuts")

    # ==================== Callback Methods ====================

    def _toggle_dock(self, dock):
        """切换面板显示"""
        dock.setVisible(not dock.isVisible())

    def _toggle_playback(self):
        """切换播放"""
        if hasattr(self.window, '_toggle_playback'):
            self.window._toggle_playback()

    def _previous_frame(self):
        """上一帧"""
        self.window._prev_frame()

    def _next_frame(self):
        """下一帧"""
        self.window._next_frame()

    def _skip_frames(self, count):
        """跳转多帧"""
        current = self.window.state_manager._state.current_frame
        self.window.state_manager.set_frame(current + count)

    def _first_frame(self):
        """首帧"""
        self.window._first_frame()

    def _last_frame(self):
        """末帧"""
        self.window._last_frame()

    def _select_tool(self, tool: Tool):
        """选择工具"""
        self.window.state_manager.set_tool(tool)

    def _sam_click_ctrl(self):
        return getattr(self.window, "sam_click_ctrl", None)

    def _cancel_action(self):
        """取消当前操作"""
        ctrl = self._sam_click_ctrl()
        if ctrl is not None and self.window.canvas._tool == Tool.SAM_CLICK:
            ctrl.discard_session()
            return
        # 如果正在绘制 polygon，取消绘制
        if (self.window.canvas._tool == Tool.POLYGON
                and self.window.canvas._polygon_vertices):
            self.window.canvas._cancel_polygon()
            return

        # 取消选中
        self.window.project.clear_selection()
        self.window.canvas._hovered_annotation_id = None
        self.window.canvas.update()

    def _deselect_all(self):
        """取消选中"""
        ctrl = self._sam_click_ctrl()
        if ctrl is not None and self.window.canvas._tool == Tool.SAM_CLICK:
            ctrl.deselect_to_create()
            return
        self.window.project.clear_selection()
        self.window.canvas._hovered_annotation_id = None
        self.window.canvas.update()

    def _select_all(self):
        """全选标注"""
        all_ids = {ann.id for ann in self.window.project.visible_annotations}
        self.window.project.set_selection(all_ids)
        self.window.canvas.update()

    def _next_annotation(self):
        """下一个标注"""
        annotations = self.window.project.visible_annotations
        if not annotations:
            return

        current_ids = self.window.project.selected_ids
        if not current_ids:
            self.window.project.set_selection({annotations[0].id})
        else:
            current_id = max(current_ids)
            found = False
            for ann in annotations:
                if found:
                    self.window.project.set_selection({ann.id})
                    return
                if ann.id == current_id:
                    found = True
            # Wrap to first
            if annotations:
                self.window.project.set_selection({annotations[0].id})

        self.window.canvas.update()

    def _prev_annotation(self):
        """上一个标注"""
        annotations = self.window.project.visible_annotations
        if not annotations:
            return

        current_ids = self.window.project.selected_ids
        if not current_ids:
            self.window.project.set_selection({annotations[-1].id})
        else:
            current_id = min(current_ids)
            prev_ann = None
            for ann in reversed(annotations):
                if ann.id == current_id:
                    if prev_ann is not None:
                        self.window.project.set_selection({prev_ann.id})
                    else:
                        self.window.project.set_selection({annotations[-1].id})
                    return
                prev_ann = ann

        self.window.canvas.update()

    def _edit_selected(self):
        """编辑选中的标注"""
        selected_ids = self.window.project.selected_ids
        if not selected_ids:
            return

        # Focus on inspector for editing
        self.window.inspector_dock.show()
        self.window.inspector.raise_()

    def _handle_delete(self):
        if self.window.canvas._tool == Tool.SAM_CLICK:
            return
        if self.window.canvas.try_delete_highlighted_vertex():
            return
        self.window._delete_selected()

    def _handle_backspace(self):
        """Backspace: 删除选中 / polygon 撤销顶点 / 删除高亮顶点"""
        ctrl = self._sam_click_ctrl()
        if ctrl is not None and self.window.canvas._tool == Tool.SAM_CLICK:
            ctrl.undo_last_point()
            return
        if (self.window.canvas._tool == Tool.POLYGON
                and self.window.canvas._polygon_vertices):
            self.window.canvas._polygon_vertices.pop()
            self.window.canvas.update()
            return
        if self.window.canvas.try_delete_highlighted_vertex():
            return
        self.window._delete_selected()

    def _handle_enter(self):
        """Enter: 编辑选中 / polygon 闭合"""
        ctrl = self._sam_click_ctrl()
        if ctrl is not None and self.window.canvas._tool == Tool.SAM_CLICK:
            ctrl.commit()
            return
        if (self.window.canvas._tool == Tool.POLYGON
                and len(self.window.canvas._polygon_vertices) >= 3):
            self.window.canvas._finish_polygon()
            return
        self._edit_selected()

    def _select_class(self, class_id: int):
        """选择类别"""
        self.window.explorer.set_selected_class(class_id)

    def _reset_view(self):
        """重置视图"""
        self.window.canvas.set_zoom(1.0)

    def _zoom_in(self):
        """放大"""
        self.window._set_zoom(self.window.canvas._zoom * 1.1)

    def _zoom_out(self):
        """缩小"""
        self.window._set_zoom(self.window.canvas._zoom * 0.9)

    def _show_shortcuts(self):
        """显示快捷键帮助"""
        self.window._show_shortcuts()
