from PySide6.QtCore import QObject, Signal
from enum import Enum
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional


class Tool(Enum):
    SELECT = "select"
    RECTANGLE = "rectangle"
    ERASER = "eraser"
    POLYGON = "polygon"
    MOVE = "move"
    HAND = "hand"
    SAM_CLICK = "sam_click"


@dataclass
class AppState:
    """全局应用状态"""
    current_file: Optional[Path] = None
    current_frame: int = 0
    total_frames: int = 0
    zoom_level: float = 1.0
    pan_offset: tuple = (0, 0)
    selected_tool: Tool = Tool.SELECT
    selected_class_id: Optional[int] = None
    is_playing: bool = False
    fps: float = 30.0
    gpu_enabled: bool = True


class AppStateManager(QObject):
    """应用状态管理器"""

    # Signals
    file_changed = Signal(object)  # Path
    frame_changed = Signal(int)  # frame index
    zoom_changed = Signal(float)  # zoom level
    tool_changed = Signal(Tool)  # current tool
    class_changed = Signal(int)  # class id
    playback_changed = Signal(bool)  # is playing

    def __init__(self):
        super().__init__()
        self._state = AppState()

    @property
    def state(self) -> AppState:
        return self._state

    def set_file(self, path: Optional[Path]):
        self._state.current_file = path
        self.file_changed.emit(path)

    def set_frame(self, index: int):
        self._state.current_frame = max(0, min(index, self._state.total_frames - 1))
        self.frame_changed.emit(self._state.current_frame)

    def set_zoom(self, level: float):
        self._state.zoom_level = max(0.1, min(level, 4.0))
        self.zoom_changed.emit(self._state.zoom_level)

    def set_tool(self, tool: Tool):
        self._state.selected_tool = tool
        self.tool_changed.emit(tool)

    def set_class(self, class_id: Optional[int]):
        self._state.selected_class_id = class_id
        if class_id is not None:
            self.class_changed.emit(class_id)

    def set_playing(self, playing: bool):
        self._state.is_playing = playing
        self.playback_changed.emit(playing)

    def toggle_playback(self):
        self.set_playing(not self._state.is_playing)
