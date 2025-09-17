from __future__ import annotations

from pathlib import Path
from typing import Iterable, Optional

from PyQt5.QtCore import QSignalBlocker, Qt, QTimer
from PyQt5.QtWidgets import (
    QDockWidget,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMessageBox,
    QPushButton,
    QSlider,
    QVBoxLayout,
    QWidget,
)

from .selection_bus import SelectionBus
from .session_model import SessionModel



def _format_millis(ms: int) -> str:
    ms = max(0, int(ms))
    seconds = ms // 1000
    minutes, secs = divmod(seconds, 60)
    hours, minutes = divmod(minutes, 60)
    if hours:
        return f"{hours:02d}:{minutes:02d}:{secs:02d}"
    return f"{minutes:02d}:{secs:02d}"


def _parse_time_string(text: str) -> Optional[int]:
    text = text.strip()
    if not text:
        return None
    parts = text.split(":")
    try:
        total_seconds = 0.0
        for part in parts:
            part = part.strip()
            if not part:
                return None
            total_seconds = total_seconds * 60.0 + float(part)
    except ValueError:
        return None
    return int(total_seconds * 1000)


class AudioPlayerDock(QDockWidget):
    """Dock widget that provides simple audio playback controls."""


    def __init__(self, bus: SelectionBus, parent: QWidget | None = None) -> None:
        super().__init__("Audio Player", parent)
        self.bus = bus
        import os
        from pathlib import Path
        def _ensure_vlc_paths():
            # Resolve relative to this file (/utils/audio_player.py)
            here = Path(__file__).resolve().parent
            base = here.parent / "vlc_player_folder"
            print(base)
            os.environ.setdefault("PYTHON_VLC_LIB_PATH", str(base / "libvlc.dll"))
            os.environ.setdefault("VLC_PLUGIN_PATH", str(base / "plugins"))

        _ensure_vlc_paths()
        import vlc

        self._session: SessionModel | None = None
        self._selected_index: int | None = None
        self._loaded_index: int | None = None
        self._duration_ms: int = 0
        self._user_seeking = False
        self._vlc_error: str | None = None

        container = QWidget(self)
        layout = QVBoxLayout(container)
        layout.setContentsMargins(8, 8, 8, 8)
        layout.setSpacing(8)

        self.track_label = QLabel("No track selected", container)
        self.track_label.setWordWrap(True)
        layout.addWidget(self.track_label)

        controls = QHBoxLayout()
        self.play_button = QPushButton("Play", container)
        controls.addWidget(self.play_button)

        self.current_time_edit = QLineEdit("00:00", container)
        self.current_time_edit.setAlignment(Qt.AlignCenter)
        self.current_time_edit.setFixedWidth(80)
        controls.addWidget(self.current_time_edit)

        self.total_time_label = QLabel("/ 00:00", container)
        controls.addWidget(self.total_time_label)
        controls.addStretch()
        layout.addLayout(controls)

        self.position_slider = QSlider(Qt.Horizontal, container)
        self.position_slider.setEnabled(False)
        layout.addWidget(self.position_slider)

        self.setWidget(container)

        self._update_timer = QTimer(self)
        self._update_timer.setInterval(200)

        self.play_button.clicked.connect(self._toggle_playback)
        self.position_slider.sliderPressed.connect(self._on_slider_pressed)
        self.position_slider.sliderReleased.connect(self._on_slider_released)
        self.position_slider.sliderMoved.connect(self._on_slider_moved)
        self._update_timer.timeout.connect(self._refresh_position)
        self.current_time_edit.returnPressed.connect(self._apply_time_edit)
        self.current_time_edit.editingFinished.connect(self._apply_time_edit)
        self.bus.imagesChanged.connect(self._on_images_selected)

        self.current_time_edit.setEnabled(False)

        if vlc is None:
            self._instance = None
            self._player = None
            self._vlc_error = "python-vlc is not available. Install VLC to enable playback."
            self.play_button.setEnabled(False)
        else:
            try:                
                self._instance = vlc.Instance()
                self._player = self._instance.media_player_new()
            except Exception as exc:
                try:
                    vlc_path = Path("F:/PhD/Projects/HyperSonograph/vlc_player_folder/libvlc.dll")
                    vlc.Instance(["--no-xlib"], libvlc_path=str(vlc_path))
                    self._player = self._instance.media_player_new()
                except Exception as exc:
                    self._player = None
                    self._instance = None
                    self._vlc_error = str(exc)
                    self.play_button.setEnabled(False)
            else:
                event_manager = self._player.event_manager()
                event_manager.event_attach(vlc.EventType.MediaPlayerEndReached, self._on_media_finished)

        self._update_title()
        self._update_play_button_state()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def set_session(self, session: SessionModel | None) -> None:
        """Update the player with a new session."""
        self._session = session
        self._selected_index = None
        self._reset_player()
        print('setting session')
        self._update_title()
        self._update_play_button_state()

    # ------------------------------------------------------------------
    # Selection handling
    # ------------------------------------------------------------------
    def _on_images_selected(self, indices: Iterable[int]) -> None:
        if not self._session:
            self._selected_index = None
            self._update_play_button_state()
            self._update_title()
            return

        chosen: Optional[int] = None
        for idx in indices:
            if 0 <= idx < len(self._session.im_list):
                chosen = idx
                break
        self._selected_index = chosen
        self._update_play_button_state()
        self._update_title()

    # ------------------------------------------------------------------
    # Playback control helpers
    # ------------------------------------------------------------------
    def _toggle_playback(self) -> None:
        if not self._player:
            if self._vlc_error:
                QMessageBox.warning(self, "Audio Player", self._vlc_error)
            return

        if self._player.is_playing():
            self._player.pause()
            self.play_button.setText("Play")
        else:
            if not self._ensure_media_loaded():
                return
            result = self._player.play()
            if result == -1:  # pragma: no cover - VLC error handling
                QMessageBox.warning(self, "Audio Player", "Unable to play the selected audio file.")
                return
            self.play_button.setText("Pause")
            self._update_timer.start()
        self._update_title()
        self._update_play_button_state()

    def _ensure_media_loaded(self) -> bool:
        if not self._player or not self._session:
            return False

        if self._player.get_media() and (
            self._selected_index is None or self._selected_index == self._loaded_index
        ):
            return True

        if self._selected_index is None:
            if self._loaded_index is not None and self._player.get_media():
                return True
            QMessageBox.information(self, "Audio Player", "Select a song to play.")
            return False

        return self._load_index(self._selected_index)

    def _load_index(self, idx: int) -> bool:
        if not self._session or not self._player or not self._instance:
            return False
        if not (0 <= idx < len(self._session.im_list)):
            return False

        path = self._session.im_list[idx]
        if not Path(path).exists():
            QMessageBox.warning(self, "Audio Player", f"File not found:\n{path}")
            return False

        if hasattr(self._instance, "media_new_path"):
            media = self._instance.media_new_path(str(path))
        else:  # pragma: no cover - fallback for older VLC bindings
            media = self._instance.media_new(str(path))
        if not media:  # pragma: no cover - defensive
            QMessageBox.warning(self, "Audio Player", "Failed to load the selected audio file.")
            return False

        self._player.stop()
        self._player.set_media(media)
        self._loaded_index = idx
        self._duration_ms = 0
        with QSignalBlocker(self.position_slider):
            self.position_slider.setRange(0, 0)
            self.position_slider.setValue(0)
        self.position_slider.setEnabled(True)
        self.current_time_edit.setEnabled(True)
        if not self.current_time_edit.hasFocus():
            self.current_time_edit.setText("00:00")
        self.total_time_label.setText("/ 00:00")
        self._update_timer.start()
        self._update_title()
        self._update_play_button_state()
        return True

    def _reset_player(self) -> None:
        if self._player:
            self._player.stop()
        self._loaded_index = None
        self._duration_ms = 0
        self._update_timer.stop()
        with QSignalBlocker(self.position_slider):
            self.position_slider.setRange(0, 0)
            self.position_slider.setValue(0)
        self.position_slider.setEnabled(False)
        self.current_time_edit.setEnabled(False)
        if not self.current_time_edit.hasFocus():
            self.current_time_edit.setText("00:00")
        self.total_time_label.setText("/ 00:00")
        self.play_button.setText("Play")
        self._update_play_button_state()
        self._update_title()

    # ------------------------------------------------------------------
    # UI updates
    # ------------------------------------------------------------------
    def _update_title(self) -> None:
        if self._vlc_error:
            self.track_label.setText(self._vlc_error)
            return

        if self._player and self._player.is_playing() and self._loaded_index is not None:
            name = self._track_name(self._loaded_index)
            self.track_label.setText(f"Now playing: {name}")
        elif self._loaded_index is not None:
            name = self._track_name(self._loaded_index)
            self.track_label.setText(f"Loaded: {name}")
        elif self._selected_index is not None:
            name = self._track_name(self._selected_index)
            self.track_label.setText(f"Selected: {name}")
        else:
            if self._session is None:
                self.track_label.setText("No session loaded")
            else:
                self.track_label.setText("No track selected")

    def _update_play_button_state(self) -> None:
        enabled = (
            self._player is not None
            and self._session is not None
            and (self._selected_index is not None or self._loaded_index is not None)
        )
        self.play_button.setEnabled(enabled)

    def _track_name(self, idx: int) -> str:
        if not self._session or not (0 <= idx < len(self._session.im_list)):
            return "Unknown"
        return Path(self._session.im_list[idx]).name

    # ------------------------------------------------------------------
    # Timed updates and seeking
    # ------------------------------------------------------------------
    def _refresh_position(self) -> None:
        if not self._player or not self._player.get_media():
            return

        length = int(self._player.get_length())
        if length > 0 and length != self._duration_ms:
            self._duration_ms = length
            with QSignalBlocker(self.position_slider):
                self.position_slider.setRange(0, length)
            self.total_time_label.setText(f"/ {_format_millis(length)}")

        current = int(self._player.get_time())
        if not self._user_seeking:
            with QSignalBlocker(self.position_slider):
                self.position_slider.setValue(current)
            if not self.current_time_edit.hasFocus():
                self.current_time_edit.setText(_format_millis(current))

        if length > 0 and current >= length and not self._player.is_playing():
            self._handle_finished()

    def _on_slider_pressed(self) -> None:
        self._user_seeking = True

    def _on_slider_released(self) -> None:
        self._user_seeking = False
        self._seek_to(self.position_slider.value())

    def _on_slider_moved(self, value: int) -> None:
        if not self.current_time_edit.hasFocus():
            self.current_time_edit.setText(_format_millis(value))

    def _seek_to(self, value: int) -> None:
        if not self._player or not self._player.get_media():
            return
        if self._duration_ms > 0:
            value = max(0, min(value, self._duration_ms))
        self._player.set_time(int(value))
        if not self.current_time_edit.hasFocus():
            self.current_time_edit.setText(_format_millis(value))

    def _apply_time_edit(self) -> None:
        if not self._player or not self._player.get_media():
            self.current_time_edit.setText("00:00")
            return

        desired = _parse_time_string(self.current_time_edit.text())
        if desired is None:
            self.current_time_edit.setText(_format_millis(self.position_slider.value()))
            return

        if self._duration_ms > 0:
            desired = max(0, min(desired, self._duration_ms))

        self._player.set_time(desired)
        with QSignalBlocker(self.position_slider):
            self.position_slider.setValue(desired)
        self.current_time_edit.setText(_format_millis(desired))

    def _handle_finished(self) -> None:
        if self._player:
            self._player.stop()
        self.play_button.setText("Play")
        with QSignalBlocker(self.position_slider):
            self.position_slider.setValue(0)
        if not self.current_time_edit.hasFocus():
            self.current_time_edit.setText("00:00")
        self._update_timer.stop()
        self._update_title()
        self._update_play_button_state()

    # ------------------------------------------------------------------
    # VLC callbacks and Qt events
    # ------------------------------------------------------------------
    def _on_media_finished(self, event) -> None:  # pragma: no cover - triggered by VLC thread
        QTimer.singleShot(0, self._handle_finished)

    def closeEvent(self, event) -> None:  # pragma: no cover - Qt lifecycle
        self._reset_player()
        super().closeEvent(event)