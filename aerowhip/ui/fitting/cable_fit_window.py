"""Visible cable identification in Take Processing; no automatic model selection."""
import json
import os
from threading import Event

from PySide6.QtCore import QThread, Signal, QUrl
from PySide6.QtGui import QDesktopServices
from PySide6.QtWidgets import (QDialog, QDoubleSpinBox, QFileDialog, QFormLayout,
    QHBoxLayout, QLabel, QLineEdit, QPlainTextEdit, QProgressBar, QPushButton,
    QSpinBox, QTableWidget, QTableWidgetItem, QVBoxLayout)

from aerowhip.physics.model_configuration import DEFAULT_FIT_MODEL
from aerowhip.paths import project_path
from aerowhip.ui import style
from aerowhip.ui.plotting import PlotCanvas, message
from aerowhip.ui.style import ROLE


class CableFitWorker(QThread):
    progress_changed = Signal(object)
    completed = Signal(object)
    failed = Signal(str)
    stopped = Signal()

    def __init__(self, root, model_path, settings, parent=None):
        super().__init__(parent)
        self.root, self.model_path, self.settings = root, model_path, settings
        self.cancelled = Event()

    def run(self):
        try:
            from aerowhip.fitting.cable_fit import fit_cable
            folder = fit_cable(self.root, self.model_path, self.settings,
                progress=self.progress_changed.emit, cancelled=self.cancelled.is_set)
            self.completed.emit(folder)
        except InterruptedError:
            self.stopped.emit()
        except Exception as exc:
            self.failed.emit(str(exc))


class CableFitWindow(QDialog):
    def __init__(self, root, parent=None):
        super().__init__(parent)
        self.root = project_path(root)
        self.worker = self.result_path = self.run_folder = None
        self._close_when_done = False
        self.setWindowTitle('Cable fit')
        self.resize(760, 820)
        style.apply(self)
        layout = QVBoxLayout(self)
        form = QFormLayout()
        layout.addLayout(form)
        folder = QLabel(str(self.root))
        folder.setWordWrap(True)
        form.addRow('Recordings', folder)
        row = QHBoxLayout()
        self.model = QLineEdit(str(DEFAULT_FIT_MODEL))
        self.model.setReadOnly(True)
        self.browse = QPushButton('Browse…')
        self.browse.clicked.connect(self.choose_model)
        row.addWidget(self.model, 1)
        row.addWidget(self.browse)
        form.addRow('Fit setup', row)
        self.workers = QSpinBox()
        self.workers.setRange(1, min(32, os.cpu_count() or 1))
        self.workers.setValue(min(4, os.cpu_count() or 1))
        form.addRow('CPU workers', self.workers)
        self.rounds = QSpinBox()
        self.rounds.setRange(0, 6)
        self.rounds.setValue(2)
        self.rounds.setToolTip('Joint 3x3x3 log-spaced EI/Cb/external-drag grid plus baseline, then local refinement. Drag bounds are 0.25–4× the input value.')
        form.addRow('Refinement rounds', self.rounds)
        self.lower, self.upper = QDoubleSpinBox(), QDoubleSpinBox()
        for spin in (self.lower, self.upper):
            spin.setDecimals(3)
            spin.setSingleStep(.1)
        self.lower.setRange(.001, .999)
        self.upper.setRange(1.001, 100.)
        self.lower.setValue(.1)
        self.upper.setValue(10.)
        form.addRow('Minimum EI/Cb multiplier', self.lower)
        form.addRow('Maximum EI/Cb multiplier', self.upper)
        from aerowhip.fitting.run_fit import SPLIT
        description = QLabel(f'{len(SPLIT["training"])} fitting / {len(SPLIT["comparison"])} comparison takes. '
            'Early 1.7 s and late 2.0 s windows. '
            'Measured attachment position and orientation; residual off. EI, Cb and external drag are fitted. '
            'External-drag multipliers: 0.25–4. GJ, Ct and drone parameters stay fixed. Initial relative twist is assumed zero.')
        description.setWordWrap(True)
        description.setObjectName('note')
        layout.addWidget(description)
        self.progress = QProgressBar()
        self.progress.setValue(0)
        layout.addWidget(self.progress)
        self.status = QLabel('Ready. Start runs the complete fixed-split fit; the active model is never replaced.')
        self.status.setWordWrap(True)
        layout.addWidget(self.status)
        # Convergence: every completed trial's fitting-set marker RMSE and the best so far.
        self.trials = []
        self.convergence = PlotCanvas(self, minimum=(400, 220))
        layout.addWidget(self.convergence, 2)
        self.plot_trials()
        self.log = QPlainTextEdit()
        self.log.setReadOnly(True)
        self.log.setMaximumBlockCount(500)
        self.log.setMaximumHeight(120)
        layout.addWidget(self.log, 1)
        self.results = QTableWidget(0, 3)
        self.results.setHorizontalHeaderLabels(['Metric', 'Before (cm)', 'After (cm)'])
        self.results.setEditTriggers(QTableWidget.EditTrigger.NoEditTriggers)
        self.results.setMaximumHeight(165)
        layout.addWidget(self.results)
        buttons = QHBoxLayout()
        self.start_button, self.stop_button = QPushButton('Start', objectName='primary'), QPushButton('Stop')
        self.open_button, self.close_button = QPushButton('Open results'), QPushButton('Close')
        self.stop_button.setEnabled(False)
        self.open_button.setEnabled(False)
        for button in (self.start_button, self.stop_button, self.open_button, self.close_button):
            buttons.addWidget(button)
        layout.addLayout(buttons)
        self.start_button.clicked.connect(self.start_fit)
        self.stop_button.clicked.connect(self.stop)
        self.close_button.clicked.connect(self.reject)
        self.open_button.clicked.connect(self.open_results)

    def choose_model(self):
        path, _ = QFileDialog.getOpenFileName(self, 'Fit setup or saved model', self.model.text(), 'Configuration (*.json)')
        if path:
            try:
                self.model.setText(str(project_path(path)))
            except ValueError as exc:
                self.status.setText(str(exc))

    def start_fit(self):
        if self.worker is not None:
            return
        from aerowhip.fitting.cable_fit import CableFitSettings
        settings = CableFitSettings(workers=self.workers.value(), refinements=self.rounds.value(),
            minimum_scale=self.lower.value(), maximum_scale=self.upper.value())
        self.result_path = self.run_folder = None
        self._close_when_done = False
        self.log.clear()
        self.trials = []
        self.plot_trials()
        self.results.setRowCount(0)
        self.open_button.setEnabled(False)
        self.worker = CableFitWorker(self.root, self.model.text(), settings, self)
        self.worker.progress_changed.connect(self.show_progress)
        self.worker.completed.connect(self.fit_completed)
        self.worker.failed.connect(lambda message: self.status.setText(f'Fit failed: {message}'))
        self.worker.stopped.connect(lambda: self.status.setText('Stopped. Partial logs retained; active model unchanged.'))
        self.worker.finished.connect(self.worker_finished)
        for control in (self.browse, self.workers, self.rounds, self.lower, self.upper, self.start_button):
            control.setEnabled(False)
        self.stop_button.setEnabled(True)
        self.progress.setRange(0, 0)
        self.status.setText('Preparing recordings…')
        self.worker.start()

    def show_progress(self, event):
        self.run_folder = project_path(event['run'])
        self.open_button.setEnabled(True)
        if self.worker is not None and self.worker.cancelled.is_set():
            return
        total, done = event.get('total', 1), event.get('completed', 0)
        self.progress.setRange(0, max(1, total))
        self.progress.setValue(done)
        message = event.get('message', event['phase'])
        self.status.setText(f'{message} · {done}/{total} · {event["elapsed_s"]/60:.1f} min')
        if event['phase'] == 'trial_complete':
            error = event['metrics']['cable_marker_rmse_m']*100
            self.log.appendPlainText(f'{message}: marker RMSE {error:.5f} cm')
            self.trials.append(error)
            self.plot_trials()
        elif event['phase'] in ('failed', 'stopped') or done == 0:
            self.log.appendPlainText(message)

    def plot_trials(self, before=None):
        c = self.convergence
        with c.style():
            c.reset()
            ax = c.figure.add_subplot()
            if not self.trials:
                message(ax, 'Marker RMSE of each trial appears here during the fit')
            else:
                n = range(1, len(self.trials)+1)
                ax.plot(n, self.trials, 'o', color=ROLE['model'], markersize=3, label='trial')
                ax.step(n, list(accumulate_min(self.trials)), where='post', color=ROLE['measured'],
                        linewidth=1.1, label='best so far')
                if before is not None:
                    ax.axhline(before, color=ROLE['reference'], linestyle='--', linewidth=.9, label='input model')
                ax.set_xlabel('trial')
                ax.set_ylabel('marker RMSE (cm)')
                ax.set_title('Convergence · fitting takes')
                ax.legend(loc='upper right')
        c.draw_idle()

    def fit_completed(self, folder):
        self.result_path = project_path(folder)
        summary = json.loads((self.result_path/'summary.json').read_text(encoding='utf-8'))
        rows = []
        for role, label in (('training', 'Fitting'), ('comparison', 'Comparison')):
            for metric, title in (('cable_marker_rmse_m', 'markers'), ('tip_rmse_m', 'tip')):
                rows.append((f'{label} {title}', summary[role+'_before'][metric]*100,
                             summary[role+'_after'][metric]*100))
        self.results.setRowCount(len(rows))
        for i, row in enumerate(rows):
            for j, value in enumerate(row):
                self.results.setItem(i, j, QTableWidgetItem(value if isinstance(value, str) else f'{value:.5f}'))
        self.results.resizeColumnsToContents()
        self.plot_trials(summary['training_before']['cable_marker_rmse_m']*100)
        self.log.appendPlainText('Conditional EI/Cb/external-drag fit; GJ/Ct and initial twist remain assumptions.')
        boundary = ' Best is at the search boundary.' if summary['boundary_best'] else ''
        self.status.setText(f"Candidate EI: {summary['EI_n_m2']:.7g} N m²; Cb: {summary['Cb_n_m2_s']:.7g} N m² s; external drag: {summary['external_drag_s_inv']:.7g} s⁻¹.{boundary} "
                           'Active model unchanged. Before/after replays are in Comparison.')
        self.log.appendPlainText(str(self.result_path/'candidate_model.json'))

    def worker_finished(self):
        self.worker.deleteLater()
        self.worker = None
        self.stop_button.setEnabled(False)
        for control in (self.browse, self.workers, self.rounds, self.lower, self.upper, self.start_button):
            control.setEnabled(True)
        if self.progress.maximum() == 0:
            self.progress.setRange(0, 100)
            self.progress.setValue(0)
        if self._close_when_done:
            super().reject()

    def open_results(self):
        if self.run_folder is not None:
            QDesktopServices.openUrl(QUrl.fromLocalFile(str(self.run_folder)))

    def stop(self):
        if self.worker is not None:
            self.worker.cancelled.set()
            self.stop_button.setEnabled(False)
            self.status.setText('Stopping at a replay boundary and waiting for workers…')

    def reject(self):
        if self.worker is not None:
            self._close_when_done = True
            self.stop()
        else:
            super().reject()

    def closeEvent(self, event):
        if self.worker is not None:
            self._close_when_done = True
            self.stop()
            event.ignore()
        else:
            super().closeEvent(event)


def accumulate_min(values):
    best = float('inf')
    for value in values:
        best = min(best, value)
        yield best
