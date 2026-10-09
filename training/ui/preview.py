"""Embedded latest saved evaluation. Never starts physics, scoring or a worker."""
from time import perf_counter
from PySide6 import QtCore, QtWidgets


class EvaluationPreview(QtWidgets.QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.pending = None
        self.loaded_key = None
        self.replay_window = None
        layout = QtWidgets.QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        self.caption = QtWidgets.QLabel('Latest saved deterministic evaluation · not live collection')
        self.caption.setWordWrap(True)
        layout.addWidget(self.caption)
        self.placeholder = QtWidgets.QLabel('Waiting for the first saved evaluation.\nTraining can collect and optimize before a replay is available.')
        self.placeholder.setAlignment(QtCore.Qt.AlignmentFlag.AlignCenter)
        self.placeholder.setWordWrap(True)
        layout.addWidget(self.placeholder, 1)
        self.body = QtWidgets.QVBoxLayout()
        layout.addLayout(self.body, 1)
        self.note = QtWidgets.QLabel('Saved states only; playback does not re-simulate. One fixed initial case is not a robustness test.')
        self.note.setWordWrap(True)
        layout.addWidget(self.note)

    def set_result(self, directory, event):
        key = (str(directory), event.get('round'), event.get('replay')) if event else (str(directory), None, None)
        if self.pending and self.pending[0] == key:
            return
        self.pending = (key, directory, event)
        if self.isVisible():
            self.load_pending()

    def _create_replay(self, directory, event):
        # Lazy import/render: the settings/log tabs do not instantiate VTK.
        from training.ui.replay_window import ReplayWindow
        return ReplayWindow(directory, event['replay'], event, self)

    def load_pending(self):
        if not self.pending or self.loaded_key == self.pending[0]:
            return
        key, directory, event = self.pending
        self.dispose()
        self.loaded_key = key
        self.placeholder.show()
        self.caption.show()
        self.note.show()
        if event is None:
            self.caption.setText('Latest saved deterministic evaluation · not live collection')
            self.placeholder.setText('No saved evaluation replay for this run yet.\nThe phase indicator above shows whether training is collecting or optimizing.')
            return
        inherited = ' · inherited from parent' if event.get('inherited') else ''
        self.caption.setText(f"Evaluation replay · iteration {event['round']}{inherited} · {directory.name}")
        window = None
        try:
            window = self._create_replay(directory, event)
            window.setWindowFlags(QtCore.Qt.WindowType.Widget)
            window.telemetry.setWordWrap(True)
            window.result.setWordWrap(True)
            window.timer.setInterval(50)  # At most 20 Hz playback; no inference.
            self.body.addWidget(window)
            self.replay_window = window
            self.placeholder.hide()
            self.caption.hide()  # Shared replay already labels its saved source.
            self.note.hide()  # Its footer and Details retain these limitations.
            window.show()
            window.toggle()  # Play each newly saved result once, not an endless loop.
        except Exception as exc:
            if window is not None:
                window.close()
            self.placeholder.setText(f'Replay unavailable: {exc}\nTraining is independent of this viewer; details remain in Results / Log.')

    def dispose(self):
        if self.replay_window is not None:
            window, self.replay_window = self.replay_window, None
            self.body.removeWidget(window)
            window.close()

    def showEvent(self, event):
        super().showEvent(event)
        self.load_pending()
        if self.replay_window is not None:
            self.replay_window.last = perf_counter()
            self.replay_window.timer.start(50)

    def hideEvent(self, event):
        if self.replay_window is not None:
            self.replay_window.pause()
            self.replay_window.timer.stop()
        super().hideEvent(event)

    def closeEvent(self, event):
        self.dispose()
        super().closeEvent(event)
