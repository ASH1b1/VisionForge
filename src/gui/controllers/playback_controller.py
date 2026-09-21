"""PlaybackController — video playback and frame navigation."""
from __future__ import annotations
from PySide6.QtCore import QObject, QTimer
from ..models.project_document import ProjectDocument
from ..models.app_state import AppStateManager


class PlaybackController(QObject):
    """Controls video playback and frame navigation."""

    def __init__(self, project: ProjectDocument, app_state: AppStateManager,
                 parent: QObject | None = None):
        super().__init__(parent)
        self._project = project
        self._app_state = app_state
        self._timer = QTimer()
        self._timer.timeout.connect(self._on_tick)

    def next_frame(self) -> None:
        # Sync using app_state frame so ProjectDocument index cannot lag behind UI
        self._project._current_frame_index = self._app_state._state.current_frame
        self._project._sync_current_to_frame()
        self._app_state.set_frame(self._app_state._state.current_frame + 1)

    def prev_frame(self) -> None:
        self._project._current_frame_index = self._app_state._state.current_frame
        self._project._sync_current_to_frame()
        self._app_state.set_frame(self._app_state._state.current_frame - 1)

    def go_to_frame(self, index: int) -> None:
        self._project._current_frame_index = self._app_state._state.current_frame
        self._project._sync_current_to_frame()
        self._app_state.set_frame(index)

    def first_frame(self) -> None:
        self._project._current_frame_index = self._app_state._state.current_frame
        self._project._sync_current_to_frame()
        self._app_state.set_frame(0)

    def last_frame(self) -> None:
        self._project._current_frame_index = self._app_state._state.current_frame
        self._project._sync_current_to_frame()
        self._app_state.set_frame(self._app_state._state.total_frames - 1)

    def play(self) -> None:
        self._timer.start(int(1000 / self._app_state._state.fps))
        self._app_state.set_playing(True)

    def pause(self) -> None:
        self._timer.stop()
        self._app_state.set_playing(False)

    def stop(self) -> None:
        self._timer.stop()
        self._app_state.set_playing(False)
        self._project._current_frame_index = self._app_state._state.current_frame
        self._project._sync_current_to_frame()
        self._app_state.set_frame(0)

    def toggle_playback(self) -> None:
        if self._app_state._state.is_playing:
            self.pause()
        else:
            self.play()

    def set_fps(self, fps: float) -> None:
        self._app_state._state.fps = max(1, fps)

    def is_playing(self) -> bool:
        return self._app_state._state.is_playing

    def _on_tick(self) -> None:
        current = self._app_state._state.current_frame
        if current >= self._app_state._state.total_frames - 1:
            self.pause()
        else:
            self._project._current_frame_index = current
            self._project._sync_current_to_frame()
            self._app_state.set_frame(current + 1)
