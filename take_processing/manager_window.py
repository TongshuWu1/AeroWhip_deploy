"""Take collection, source review and non-destructive interval selection."""
from copy import deepcopy
from pathlib import Path
import numpy as np
from PySide6.QtCore import Qt, QThread, Signal, QTimer
from PySide6.QtWidgets import (QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, QSplitter,
    QPushButton, QTreeWidget, QTreeWidgetItem, QLabel, QDoubleSpinBox, QFormLayout,
    QGroupBox, QFileDialog, QMessageBox, QScrollArea, QComboBox)

from take_processing.library import Library, verify_sources
from take_processing.quality import QualityTimeline, summarize
from take_processing.replay_window import ReplayWindow
from take_processing.position_plots import PositionPlots
from take_processing.baseline import discover_baselines, load_baseline


class Importer(QThread):
    imported = Signal(object)
    failed = Signal(str)

    def __init__(self, library, folder, *, refresh_existing=True):
        super().__init__()
        self.library, self.folder = library, folder
        self.refresh_existing = refresh_existing

    def run(self):
        try:
            self.imported.emit(self.library.import_folder(self.folder, refresh_existing=self.refresh_existing))
        except Exception as exc:
            self.failed.emit(str(exc))


class TakeManager(QMainWindow):
    def __init__(self, folder, library_path):
        super().__init__()
        self.setWindowTitle("Take Manager")
        self.resize(1280, 760)
        self.folder = Path(folder).resolve()
        self.library = Library(library_path)
        self.importer = None
        self.current_key = None
        self.dirty = False
        self.first = self.last = 0
        self.setStyleSheet("QWidget {font-size: 12px;} QPushButton {padding: 5px 8px;} QGroupBox {font-weight: 600;}")
        splitter = QSplitter()
        self.setCentralWidget(splitter)
        left = QWidget()
        left_layout = QVBoxLayout(left)
        self.import_button = QPushButton("Import folder")
        self.import_button.clicked.connect(self.choose_folder)
        left_layout.addWidget(self.import_button)
        self.tree = QTreeWidget()
        self.tree.setHeaderLabels(["Sessions / Takes"])
        self.tree.setMinimumWidth(200)
        left_layout.addWidget(self.tree)
        self.count_label = QLabel()
        left_layout.addWidget(self.count_label)
        splitter.addWidget(left)
        center = QWidget()
        center_layout = QVBoxLayout(center)
        center_layout.setContentsMargins(0, 0, 0, 0)
        self.replay = ReplayWindow()
        self.replay.setWindowFlags(Qt.WindowType.Widget)
        self.replay.open_button.hide()
        self.replay.takes.hide()
        views = QSplitter(Qt.Orientation.Vertical)
        views.addWidget(self.replay)
        self.position_plots = PositionPlots()
        views.addWidget(self.position_plots)
        views.setStretchFactor(0, 3)
        views.setStretchFactor(1, 2)
        views.setSizes([440, 240])
        center_layout.addWidget(views, 1)
        self.quality_timeline = QualityTimeline()
        center_layout.addWidget(self.quality_timeline)
        splitter.addWidget(center)
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setMinimumWidth(235)
        panel = QWidget()
        panel_layout = QVBoxLayout(panel)
        self.name_label = QLabel("Select a take")
        self.name_label.setWordWrap(True)
        panel_layout.addWidget(self.name_label)
        self.source_label = QLabel()
        self.source_label.setWordWrap(True)
        source_box = QGroupBox("Files")
        QVBoxLayout(source_box).addWidget(self.source_label)
        panel_layout.addWidget(source_box)
        self.full_quality = QLabel()
        self.full_quality.setWordWrap(True)
        full_box = QGroupBox("Full take")
        QVBoxLayout(full_box).addWidget(self.full_quality)
        panel_layout.addWidget(full_box)
        alignment_box = QGroupBox("Alignment")
        self.alignment_label = QLabel("Select a take")
        self.alignment_label.setWordWrap(True)
        self.alignment_label.setTextFormat(Qt.TextFormat.PlainText)
        self.alignment_label.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
        QVBoxLayout(alignment_box).addWidget(self.alignment_label)
        panel_layout.addWidget(alignment_box)
        baseline_box=QGroupBox('Baseline')
        baseline_layout=QVBoxLayout(baseline_box)
        self.baselines=QComboBox()
        self.baselines.addItem('None',None)
        self.baselines.currentIndexChanged.connect(self.select_baseline)
        baseline_layout.addWidget(self.baselines)
        self.baseline_label=QLabel('No saved comparison')
        self.baseline_label.setWordWrap(True)
        baseline_layout.addWidget(self.baseline_label)
        self.baseline_start=QPushButton('Go to baseline')
        self.baseline_start.setEnabled(False)
        self.baseline_start.clicked.connect(self.seek_baseline)
        baseline_layout.addWidget(self.baseline_start)
        panel_layout.addWidget(baseline_box)
        self.interval_box = QGroupBox("Interval")
        interval = QVBoxLayout(self.interval_box)
        form = QFormLayout()
        self.start = QDoubleSpinBox()
        self.end = QDoubleSpinBox()
        for control in (self.start, self.end):
            control.setDecimals(3)
            control.setSuffix(" s")
            control.setSingleStep(.01)
            control.setKeyboardTracking(False)
            control.valueChanged.connect(self.interval_changed)
        form.addRow("Start", self.start)
        form.addRow("End", self.end)
        interval.addLayout(form)
        buttons = QHBoxLayout()
        for text, callback in (("Set start", self.set_start), ("Set end", self.set_end)):
            button = QPushButton(text)
            button.clicked.connect(callback)
            buttons.addWidget(button)
        interval.addLayout(buttons)
        entire = QPushButton("Full take")
        entire.clicked.connect(self.select_full)
        interval.addWidget(entire)
        self.selection_quality = QLabel()
        self.selection_quality.setWordWrap(True)
        interval.addWidget(self.selection_quality)
        self.save_button = QPushButton("Save interval")
        self.save_button.clicked.connect(self.save_interval)
        interval.addWidget(self.save_button)
        self.selection_status = QLabel()
        interval.addWidget(self.selection_status)
        self.interval_box.setEnabled(False)
        panel_layout.addWidget(self.interval_box)
        panel_layout.addStretch()
        scroll.setWidget(panel)
        splitter.addWidget(scroll)
        splitter.setStretchFactor(1, 1)
        splitter.setSizes([240, 790, 250])
        self.tree.currentItemChanged.connect(self.select_item)
        self.replay.take_loaded.connect(self.take_loaded)
        self.replay.take_failed.connect(self.take_failed)
        self.replay.frame_changed.connect(self.playback_changed)
        self.quality_timeline.seek_requested.connect(self.seek_time)
        self.position_plots.seek_requested.connect(self.seek_time)
        for name, check in self.replay.checks.items():
            check.toggled.connect(lambda visible, layer=name: self.position_plots.set_layer(layer, visible))
        self.populate_tree()
        self.startup_timer = QTimer(self)
        self.startup_timer.setSingleShot(True)
        self.startup_timer.timeout.connect(self.scan_on_startup)
        self.startup_timer.start(0)

    def scan_on_startup(self):
        if not self.folder.is_dir():
            self.statusBar().showMessage(f"Recording folder not found: {self.folder}")
            return
        self.import_folder(self.folder, refresh_existing=False)

    def busy(self):
        return self.importer is not None or self.replay.worker is not None

    def populate_tree(self):
        self.tree.blockSignals(True)
        self.tree.clear()
        groups = {}
        for key, entry in sorted(self.library.entries.items(), key=lambda pair: (pair[1]["session"], pair[1]["name"])):
            group_key = (entry["session"], entry["grouping"],
                         str(Path(entry["path"]).parent) if entry["grouping"] == "folder" else "")
            if group_key not in groups:
                group = QTreeWidgetItem(self.tree, [entry["session"]])
                group.setToolTip(0, f"Grouped by {entry['grouping']}; no fitting role assigned")
                groups[group_key] = group
            item = QTreeWidgetItem(groups[group_key], [entry["name"]])
            item.setData(0, Qt.ItemDataRole.UserRole, key)
            item.setToolTip(0, entry["path"])
            if entry.get("selection"):
                item.setText(0, entry["name"] + " · saved")
        self.tree.expandAll()
        self.tree.blockSignals(False)
        self.count_label.setText(f"{len(self.library.entries)} takes · {len(groups)} groups")

    def choose_folder(self):
        if self.busy() or not self.confirm_pending():
            return
        folder = QFileDialog.getExistingDirectory(self, "Import recordings", str(self.folder))
        if folder:
            self.import_folder(folder)

    def import_folder(self, folder, *, refresh_existing=True):
        if self.busy():
            return
        self.folder = Path(folder).resolve()
        self.replay.pause()
        self.tree.setEnabled(False)
        self.import_button.setEnabled(False)
        self.interval_box.setEnabled(False)
        self.statusBar().showMessage("Indexing source files…")
        self.importer = Importer(self.library, self.folder, refresh_existing=refresh_existing)
        self.importer.imported.connect(self.imported)
        self.importer.failed.connect(self.import_failed)
        self.importer.finished.connect(self.import_finished)
        self.importer.start()

    def imported(self, report):
        self.current_key = None
        self.dirty = False
        self.replay.load_path(None)
        self.position_plots.clear()
        self.clear_baselines()
        self.alignment_label.setText("Select a take")
        self.quality_timeline.take = None
        self.quality_timeline.update()
        self.name_label.setText("Select a take")
        self.source_label.clear()
        self.full_quality.clear()
        self.selection_quality.clear()
        self.selection_status.clear()
        self.populate_tree()
        text = f"{report['added']} added · {report['duplicates']} duplicates skipped"
        if report["updated"]:
            text += f" · {report['updated']} updated; old intervals cleared"
        if report.get("changed"):
            text += f" · {report['changed']} changed sources; use Import folder to review"
        if report["errors"]:
            text += f" · {len(report['errors'])} could not be imported"
            dialog = QMessageBox(QMessageBox.Icon.Warning, "Import", text, parent=self)
            dialog.setDetailedText("\n".join(report["errors"]))
            dialog.exec()
        self.statusBar().showMessage(text)

    def import_failed(self, message):
        self.statusBar().showMessage(message)

    def import_finished(self):
        self.importer.deleteLater()
        self.importer = None
        self.tree.setEnabled(True)
        self.import_button.setEnabled(True)
        self.interval_box.setEnabled(self.replay.data is not None)

    def select_item(self, item, previous):
        key = item.data(0, Qt.ItemDataRole.UserRole) if item else None
        if key is None or key == self.current_key:
            return
        if self.busy() or not self.confirm_pending():
            self.tree.blockSignals(True)
            self.tree.setCurrentItem(previous)
            self.tree.blockSignals(False)
            return
        self.current_key = key
        self.dirty = False
        entry = self.library.entries[key]
        self.name_label.setText(entry["name"])
        self.source_label.setText("\n".join(
            f"{label}: {'Found' if entry['sources'].get(name, {}).get('sha256') else 'Missing'}"
            for name, label in (("tracking", "Tracking"), ("samples", "Samples"), ("commands", "Commands"), ("reference", "Reference"))))
        self.source_label.setToolTip("\n".join(f"{name}: {s['path']}" for name, s in entry["sources"].items()))
        self.full_quality.setText("Loading…")
        self.position_plots.clear()
        self.clear_baselines()
        self.alignment_label.setText("Loading…")
        self.selection_quality.clear()
        self.selection_status.clear()
        self.interval_box.setEnabled(False)
        self.quality_timeline.take = None
        self.quality_timeline.update()
        snapshot = deepcopy(entry)
        self.replay.load_path(Path(entry["path"]), verify=lambda: verify_sources(snapshot))

    @staticmethod
    def quality_text(q):
        return (f"{q['frames']:,} frames · {q['duration']:.2f} s\n"
                f"Pose valid: {q['pose_percent']:.1f}%\n"
                f"All markers valid: {q['complete_percent']:.1f}%\n"
                f"Tracking gaps: {q['gaps']}")

    def take_loaded(self, take):
        self.position_plots.set_take(take)
        alignment = take.alignment
        if alignment is None:
            self.alignment_label.setText("Unavailable\nNo saved clock alignment")
        else:
            state = "Verified" if alignment.get("clock_verified") is True else "Estimated / unverified"
            application = "Applied" if take.alignment_applied else "Not applied"
            offset = alignment.get("offset_s")
            offset_text = f"{offset:+.6f} s" if isinstance(offset, (int, float)) and np.isfinite(offset) else "Unavailable"
            source = alignment.get("source") or "Not recorded"
            text = f"{state} · {application}\nOffset: {offset_text}\nLogger time = tracking time + offset\n\nSource: {source}"
            if take.alignment_error:
                text += f"\n\n{take.alignment_error}"
            self.alignment_label.setText(text)
        q = summarize(take, 0, len(take.time)-1)
        self.full_quality.setText(f"{q['rate']:.1f} Hz (median)\n" + self.quality_text(q))
        self.full_quality.setToolTip("\n".join(f"Marker {i+1}: {v:.1f}% valid" for i, v in enumerate(q["marker_percent"])) +
                                    f"\nLargest tracking gap: {q['largest_gap']:.3f} s")
        self.quality_timeline.set_take(take)
        for spin in (self.start, self.end):
            spin.blockSignals(True)
            spin.setRange(0, float(take.time[-1]))
            spin.blockSignals(False)
        self.first, self.last = 0, len(take.time)-1
        saved = self.library.entries[self.current_key].get("selection")
        if saved:
            first, last = saved["first_index"], saved["last_index"]
            if (0 <= first < last < len(take.time) and take.frame[first] == saved["first_frame"]
                    and take.frame[last] == saved["last_frame"]):
                self.first, self.last = first, last
        self.interval_box.setEnabled(True)
        self.update_interval()
        root=Path(__file__).resolve().parents[1]/'runs/coupled_baseline'
        self.baselines.blockSignals(True)
        for path in discover_baselines(take,root):
            self.baselines.addItem(path.parent.name,path)
        self.baselines.blockSignals(False)
        if self.baselines.count()>1:
            self.baselines.setCurrentIndex(1)
        self.statusBar().showMessage("Raw files unchanged · interval endpoints are inclusive")

    def clear_baselines(self):
        self.baselines.blockSignals(True)
        self.baselines.clear()
        self.baselines.addItem('None',None)
        self.baselines.blockSignals(False)
        self.baseline_label.setText('No saved comparison')
        self.baseline_start.setEnabled(False)

    def select_baseline(self):
        self.replay.set_prediction(None)
        self.position_plots.set_prediction(None)
        self.baseline_start.setEnabled(False)
        path=self.baselines.currentData()
        if path is None or self.replay.data is None:
            self.baseline_label.setText('No comparison selected')
            return
        try:
            baseline=load_baseline(path,self.replay.data)
            self.replay.set_prediction(baseline)
            self.position_plots.set_prediction(baseline)
            metrics=baseline.metadata['metrics']
            self.baseline_label.setText(f"{baseline.time[0]:.2f}–{baseline.time[-1]:.2f} s\n"
                f"Position RMSE: {100*metrics['tracking_position_rmse_m']:.1f} cm\n"
                f"Tip RMSE: {100*metrics['tip_rmse_m']:.1f} cm")
            self.baseline_label.setToolTip(f"Saved run: {path}\nErrors cover the full saved baseline, not the selected interval.\nModel: {baseline.metadata['model_sha256']}")
            self.baseline_start.setEnabled(True)
        except (ValueError,KeyError,OSError,IndexError) as exc:
            self.replay.set_prediction(None)
            self.position_plots.set_prediction(None)
            self.baseline_label.setText(f'Unavailable: {exc}')

    def seek_baseline(self):
        if self.replay.prediction is not None:
            self.replay.pause()
            self.seek_time(self.replay.prediction.time[0])

    def take_failed(self, message):
        self.position_plots.clear()
        self.alignment_label.setText("Unavailable\nTake could not be loaded")
        self.full_quality.setText("Could not load take")
        self.full_quality.setToolTip(message)
        self.statusBar().showMessage(message)

    def interval_changed(self):
        take = self.replay.data
        if take is None:
            return
        self.first = int(np.argmin(np.abs(take.time-self.start.value())))
        self.last = int(np.argmin(np.abs(take.time-self.end.value())))
        self.update_interval()

    def update_interval(self):
        take = self.replay.data
        if take is None:
            return
        for spin, index in ((self.start, self.first), (self.end, self.last)):
            spin.blockSignals(True)
            spin.setValue(float(take.time[index]))
            spin.blockSignals(False)
        saved = self.library.entries[self.current_key].get("selection")
        baseline = (saved["first_index"], saved["last_index"]) if saved else (0, len(take.time)-1)
        self.dirty = (self.first, self.last) != baseline
        valid = self.first < self.last
        self.save_button.setEnabled(valid)
        self.selection_quality.setText(self.quality_text(summarize(take, self.first, self.last))
                                       if valid else "Start must be before End")
        self.selection_status.setText("Unsaved" if self.dirty else ("Saved" if saved else "Not saved"))
        self.quality_timeline.first, self.quality_timeline.last = self.first, self.last
        self.quality_timeline.update()
        self.position_plots.set_interval(self.first, self.last)

    def set_start(self):
        self.first = self.replay.timeline.value()
        self.update_interval()

    def set_end(self):
        self.last = self.replay.timeline.value()
        self.update_interval()

    def select_full(self):
        self.first, self.last = 0, len(self.replay.data.time)-1
        self.update_interval()

    def save_interval(self):
        try:
            self.library.save_selection(self.current_key, self.replay.data, self.first, self.last)
        except Exception as exc:
            QMessageBox.warning(self, "Save interval", str(exc))
            return False
        self.dirty = False
        self.selection_status.setText("Saved")
        item = self.tree.currentItem()
        if item and item.data(0, Qt.ItemDataRole.UserRole) == self.current_key:
            item.setText(0, self.library.entries[self.current_key]["name"] + " · saved")
        self.statusBar().showMessage("Interval saved · raw files unchanged")
        return True

    def confirm_pending(self):
        if not self.dirty:
            return True
        choice = QMessageBox.question(self, "Interval", "Save this interval before leaving the take?",
            QMessageBox.StandardButton.Save | QMessageBox.StandardButton.Discard | QMessageBox.StandardButton.Cancel)
        if choice == QMessageBox.StandardButton.Save:
            return self.save_interval()
        return choice == QMessageBox.StandardButton.Discard

    def playback_changed(self, time):
        self.position_plots.set_cursor(time)
        self.quality_timeline.current = time
        self.quality_timeline.update()

    def seek_time(self, time):
        if self.replay.data is not None:
            index = int(np.argmin(np.abs(self.replay.data.time-time)))
            self.replay.timeline.setValue(index)

    def closeEvent(self, event):
        if self.busy():
            self.statusBar().showMessage("Finishing file operation; close again when ready")
            event.ignore()
            return
        if not self.confirm_pending():
            event.ignore()
            return
        self.replay.close()
        event.accept()
