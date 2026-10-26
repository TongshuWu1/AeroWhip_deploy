"""Explicit recorded comparisons with cooperative background execution."""

from pathlib import Path
from threading import Event

from PySide6.QtCore import QThread, Signal
from PySide6.QtWidgets import (QComboBox, QDialog, QDoubleSpinBox, QFormLayout,
                               QHBoxLayout, QLabel, QProgressBar, QPushButton,
                               QVBoxLayout)

from aerowhip.paths import ROOT
from aerowhip.physics.model_configuration import DEFAULT_FIT_MODEL, DEFAULT_MODEL, load_model_configuration


class ComparisonWorker(QThread):
    progress_changed = Signal(int, int)
    completed = Signal(object)
    failed = Signal(str)
    stopped = Signal()

    def __init__(self, path, model_path, options, parent=None):
        super().__init__(parent)
        self.path, self.model_path, self.options = path, model_path, options
        self.cancelled = Event()

    def run(self):
        try:
            from aerowhip.fitting.calibration import compare_recording, save_comparison
            metadata, arrays = compare_recording(
                self.path, self.model_path, **self.options, device='cpu',
                progress=self.progress_changed.emit, cancelled=self.cancelled.is_set)
            if self.cancelled.is_set():
                raise InterruptedError()
            path = save_comparison(metadata, arrays)
            self.completed.emit(path)
        except InterruptedError:
            self.stopped.emit()
        except Exception as exc:
            self.failed.emit(str(exc))


class CalibrationWindow(QDialog):
    def __init__(self, take, *, start_s, duration_s, parent=None):
        super().__init__(parent)
        self.take = take
        self.worker = None
        self.result_path = None
        self._close_when_done = False
        self.setWindowTitle('Calibration comparison')
        self.setMinimumWidth(480)
        layout = QVBoxLayout(self)
        name = QLabel(take.path.name)
        name.setWordWrap(True)
        layout.addWidget(name)
        form = QFormLayout()
        layout.addLayout(form)
        self.mode = QComboBox()
        self.mode.addItem('Cable', 'prescribed_cable')
        self.mode.addItem('Coupled', 'coupled')
        form.addRow('Mode', self.mode)
        self.model = QComboBox()
        for path in (DEFAULT_MODEL, *sorted((ROOT/'models').glob('*.json'))):
            try:
                config = load_model_configuration(path)
                if not all(isinstance(config.get(key), dict) for key in
                           ('drone', 'cable', 'attachment', 'controller', 'simulation')):
                    continue
            except (OSError, ValueError, AttributeError):
                continue
            label = 'Current setup' if path == DEFAULT_MODEL else 'Snapshot: '+path.name
            self.model.addItem(label, str(path))
        default = self.model.findData(str(DEFAULT_FIT_MODEL))
        if default >= 0:
            self.model.setCurrentIndex(default)
        form.addRow('Model', self.model)
        self.start = QDoubleSpinBox()
        self.duration = QDoubleSpinBox()
        for control in (self.start, self.duration):
            control.setDecimals(2)
            control.setSingleStep(.01)
            control.setSuffix(' s')
            control.setKeyboardTracking(False)
        self.start.setRange(.1, max(.1, float(take.time[-1])-.01))
        self.start.setValue(start_s)
        self.duration.setRange(.01, max(.01, float(take.time[-1])-.1))
        self.duration.setValue(min(2.5, duration_s))
        form.addRow('Start', self.start)
        form.addRow('Duration', self.duration)
        self.description = QLabel()
        self.description.setWordWrap(True)
        layout.addWidget(self.description)
        note = QLabel('Start selects the first measured sample at or after the requested time, with 0.10 s of causal history before it. Measurements remain on their native 0.01 s grid. Missing data or unsupported inputs are rejected.')
        note.setWordWrap(True)
        layout.addWidget(note)
        self.progress = QProgressBar()
        self.progress.setRange(0, 100)
        self.progress.setValue(0)
        layout.addWidget(self.progress)
        self.status = QLabel('Ready. Run starts a CPU comparison and saves its completed result.')
        self.status.setWordWrap(True)
        layout.addWidget(self.status)
        row = QHBoxLayout()
        self.run_button = QPushButton('Run')
        self.run_button.setToolTip('Run a CPU comparison over this interval. No fitting or model update is performed.')
        self.stop_button = QPushButton('Stop')
        self.close_button = QPushButton('Close')
        self.run_button.setEnabled(self.model.count() > 0)
        self.stop_button.setEnabled(False)
        for button in (self.run_button, self.stop_button, self.close_button):
            row.addWidget(button)
        layout.addLayout(row)
        self.run_button.clicked.connect(self.start_comparison)
        self.stop_button.clicked.connect(self.stop)
        self.close_button.clicked.connect(self.reject)
        self.mode.currentIndexChanged.connect(self.mode_changed)
        self.mode_changed()

    def mode_changed(self):
        mode = self.mode.currentData()
        text = {
            'prescribed_cable': 'Measured drone motion prescribes the attachment for component identification. Requires the unfitted zero correction; this is not a second flight simulator.',
            'coupled': 'The sampled geometric controller and four motors drive the coupled plant. Initial motor thrusts are provisional, not measured in old recordings.',
        }
        self.description.setText(text[mode])

    def start_comparison(self):
        if self.worker is not None or self.model.currentData() is None:
            return
        start, duration = self.start.value(), self.duration.value()
        if start+duration > float(self.take.time[-1])+1e-8:
            self.status.setText('Start plus Duration exceeds the recording. Shorten the interval.')
            return
        mode = self.mode.currentData()
        options = dict(start_s=start, duration_s=duration,
                       mode=mode)
        self.result_path = None
        self.worker = ComparisonWorker(self.take.path, self.model.currentData(), options, self)
        self.worker.progress_changed.connect(self.show_progress)
        self.worker.completed.connect(self.comparison_completed)
        self.worker.failed.connect(lambda message: self.status.setText(f'Comparison failed: {message}'))
        self.worker.stopped.connect(lambda: self.status.setText('Stopped. No comparison result was saved.'))
        self.worker.finished.connect(self.worker_finished)
        for control in (self.mode, self.model, self.start, self.duration, self.run_button):
            control.setEnabled(False)
        self.stop_button.setEnabled(True)
        self.progress.setRange(0, 0)
        self.status.setText('Preparing source-verified measurements…')
        self.worker.start()

    def show_progress(self, completed, total):
        self.progress.setRange(0, max(1, total))
        self.progress.setValue(completed)
        if self.worker is None or not self.worker.cancelled.is_set():
            self.status.setText(f'Comparing {completed} / {total} measured steps')

    def comparison_completed(self, path):
        self.result_path = Path(path)
        self.status.setText(f'Saved: {path}')

    def worker_finished(self):
        self.worker.deleteLater()
        self.worker = None
        self.stop_button.setEnabled(False)
        if self.result_path is not None:
            self.accept()
        elif self._close_when_done:
            super().reject()
        else:
            for control in (self.mode, self.model, self.start, self.duration, self.run_button):
                control.setEnabled(True)
            self.mode_changed()
            if self.progress.maximum() == 0:
                self.progress.setRange(0, 100)
                self.progress.setValue(0)

    def stop(self):
        if self.worker is not None:
            self.worker.cancelled.set()
            self.stop_button.setEnabled(False)
            self.status.setText('Stopping after the current integration step…')

    def reject(self):
        if self.worker is not None:
            self._close_when_done = True
            self.stop()
            return
        super().reject()

    def closeEvent(self, event):
        if self.worker is not None:
            self._close_when_done = True
            self.stop()
            event.ignore()
        else:
            super().closeEvent(event)
