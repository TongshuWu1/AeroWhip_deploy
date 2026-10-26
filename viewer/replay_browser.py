"""Stage/run/file selection for saved motion, shared with Training's replay UI."""
import argparse
import json
from pathlib import Path
import numpy as np
from PySide6 import QtCore, QtGui, QtWidgets
from project_paths import ROOT, project_path
from training.artifacts import RUNS
from viewer.replay_data import PresentationData, read_events


def discover_runs(root=ROOT):
    folder = project_path(root) / RUNS.relative_to(ROOT)
    runs = [directory for directory in folder.iterdir()
            if directory.is_dir() and (directory / 'configuration.json').exists()] if folder.exists() else []
    return sorted(runs, key=lambda p: p.name, reverse=True)


def resolve_selection(path):
    """Checkpoint selection locates its run only: no actor loading/inference."""
    path = project_path(path)
    if not path.exists():
        raise ValueError('Selected file/folder does not exist')
    if path.is_dir():
        directory, filename = path, None
    elif path.suffix.lower() in ('.pt', '.json', '.npz', '.jsonl'):
        directory = path.parent
        filename = path.name if path.suffix.lower() == '.npz' else None
    else:
        raise ValueError('Select a training run folder, checkpoint, or saved NPZ replay')
    if not (directory / 'configuration.json').exists() or not (directory / 'scene.toml').exists():
        raise ValueError('Selected file/folder has no associated run configuration and scene')
    return directory, filename


def replay_choices(directory):
    """Include unavailable evaluations explicitly, so latest failure stays visible."""
    rows = read_events(directory)
    seen, choices = set(), []
    for row in reversed(rows):
        filename = row.get('replay')
        if filename and filename not in seen:
            seen.add(filename)
            choices.append((filename, row))
        elif row.get('phase') == 'evaluation' and not filename:
            choices.append((None, row))
    for path in sorted(Path(directory).glob('*.npz'), reverse=True):
        if path.name not in seen:
            choices.append((path.name, {}))
    return choices


class ReplayBrowser(QtWidgets.QMainWindow):
    def __init__(self, selected=None, filename=None, *, root=ROOT):
        super().__init__()
        self.root = project_path(root)
        self.extra_runs = set()
        self.motion = None
        self.setWindowTitle('AeroWhip · Replay')
        self.resize(1440, 1000)
        self.setMinimumSize(1000, 700)
        from training.ui.replay_window import STYLE
        self.setStyleSheet(STYLE)
        central = QtWidgets.QWidget(objectName='page')
        self.setCentralWidget(central)
        layout = QtWidgets.QVBoxLayout(central)
        layout.setContentsMargins(8, 8, 8, 8)
        row = QtWidgets.QHBoxLayout()
        self.runs = QtWidgets.QComboBox()
        self.runs.setMinimumWidth(380)
        row.addWidget(self.runs, 1)
        for name, callback in (('Refresh', self.refresh_runs), ('Run folder…', self.choose_folder),
                               ('Training / replay file…', self.choose_file)):
            button = QtWidgets.QPushButton(name)
            button.clicked.connect(callback)
            row.addWidget(button)
        layout.addLayout(row)
        replay_row = QtWidgets.QHBoxLayout()
        replay_row.addWidget(QtWidgets.QLabel('Saved motion'))
        self.replays = QtWidgets.QComboBox()
        replay_row.addWidget(self.replays, 1)
        layout.addLayout(replay_row)
        self.message = QtWidgets.QLabel()
        self.message.setWordWrap(True)
        layout.addWidget(self.message)
        self.body = QtWidgets.QVBoxLayout()
        layout.addLayout(self.body, 1)
        self.runs.currentIndexChanged.connect(self.select_run)
        self.replays.currentIndexChanged.connect(self.select_replay)
        if selected is not None:
            directory, selected_filename = resolve_selection(selected)
            filename = filename or selected_filename
            self.extra_runs.add(directory)
            self.refresh_runs(directory)
            if filename:
                self.select_filename(filename)
        else:
            self.refresh_runs()

    def refresh_runs(self, preferred=None):
        # QPushButton passes bool; only Path/string is a selection request.
        preferred = preferred if isinstance(preferred, (str, Path)) else self.runs.currentData()
        try:
            directories = sorted(set(discover_runs(self.root)) | self.extra_runs,
                                 key=lambda p: p.name, reverse=True)
        except (OSError, ValueError, KeyError) as exc:
            self.dispose()
            self.message.setText('Discovery unavailable: '+str(exc))
            return
        with QtCore.QSignalBlocker(self.runs):
            self.runs.clear()
            for directory in directories:
                self.runs.addItem(f'{directory.parent.name} / {directory.name}', directory)
                self.runs.setItemData(self.runs.count() - 1, str(directory), QtCore.Qt.ItemDataRole.ToolTipRole)
            for i in range(self.runs.count()):
                if self.runs.itemData(i) == preferred:
                    self.runs.setCurrentIndex(i)
                    break
        self.select_run()

    def select_run(self, *_):
        directory = self.runs.currentData()
        try:
            choices = replay_choices(directory) if directory else []
        except (OSError, ValueError, KeyError) as exc:
            with QtCore.QSignalBlocker(self.replays):
                self.replays.clear()
            self.dispose()
            self.message.setText('Replay unavailable: '+str(exc))
            return
        with QtCore.QSignalBlocker(self.replays):
            self.replays.clear()
            for filename, event in choices:
                phase = event.get('phase', 'Saved file').replace('_', ' ')
                label = f"Collection {event.get('round', '—')} · {phase} · {filename or 'NO REPLAY (invalid/unavailable)'}"
                self.replays.addItem(label, (filename, event))
            # Latest evaluation means latest recorded evaluation, even if invalid.
            for i, (_, event) in enumerate(choices):
                if event.get('phase') == 'evaluation':
                    self.replays.setCurrentIndex(i)
                    break
        self.select_replay()

    def dispose(self):
        if self.motion is not None:
            motion, self.motion = self.motion, None
            self.body.removeWidget(motion)
            motion.close()

    def select_replay(self, *_):
        self.dispose()
        directory, selection = self.runs.currentData(), self.replays.currentData()
        if not directory or not selection:
            self.message.setText('No saved motion available. Choose a training run folder or file; no simulation will be started.')
            return
        filename, event = selection
        if not filename:
            self.message.setText('This evaluation has no saved replay. It is not replaced by an older successful motion.')
            return
        try:
            from training.ui.replay_window import ReplayWindow
            motion = ReplayWindow(directory, filename, event, self)
            motion.setWindowFlags(QtCore.Qt.WindowType.Widget)
            self.body.addWidget(motion)
            self.motion = motion
            motion.show()
            self.message.setText('Saved motion · no new simulation or policy inference')
            self.message.setToolTip(str(directory) + '\nCheckpoint selection locates a run; it does not identify the actor of every historical replay.')
        except Exception as exc:
            self.message.setText(f'Replay unavailable: {exc}. No worker started and no older motion substituted.')

    def select_filename(self, filename):
        for i in range(self.replays.count()):
            if self.replays.itemData(i)[0] == filename:
                self.replays.setCurrentIndex(i)
                return
        raise ValueError('Selected NPZ is not saved in this run')

    def open_selection(self, path):
        directory, filename = resolve_selection(path)
        self.extra_runs.add(directory)
        self.refresh_runs(directory)
        if filename:
            self.select_filename(filename)

    def choose_folder(self):
        path = QtWidgets.QFileDialog.getExistingDirectory(self, 'Select training run', str(self.root / 'runs'))
        if path:
            self.try_open(path)

    def choose_file(self):
        path, _ = QtWidgets.QFileDialog.getOpenFileName(self, 'Select training checkpoint or replay',
            str(self.root / 'runs'), 'Training / replay files (*.pt *.npz *.json *.jsonl)')
        if path:
            self.try_open(path)

    def try_open(self, path):
        try:
            self.open_selection(path)
        except Exception as exc:
            QtWidgets.QMessageBox.warning(self, 'Replay selection', str(exc))

    def closeEvent(self, event):
        self.dispose()
        super().closeEvent(event)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', '--file', dest='selected', type=Path)
    parser.add_argument('--replay', help='Saved NPZ basename within the selected run')
    parser.add_argument('--audit', action='store_true', help='Read saved event metrics without a GUI')
    parser.add_argument('--snapshot', type=Path, help='Render the selected saved replay and exit')
    parser.add_argument('--flight-view', action='store_true', help='Open saved states in the docked flight viewer')
    parser.add_argument('--measurement-report', type=Path,
                        help='Open a hash-pinned offline measurement review in the existing replay viewer')
    parser.add_argument('--case', type=int, dest='case_index',
                        help='Zero-based measurement case index (default: 0)')
    parser.add_argument('--event', type=int, dest='event_index',
                        help='Zero-based measurement event index (default: report selection)')
    args = parser.parse_args()
    measurement = None
    if args.measurement_report is not None:
        if args.selected or args.replay or args.flight_view:
            parser.error('--measurement-report selects its own saved replay; do not combine with --run/--file/--replay/--flight-view')
        measurement = PresentationData.load_measurement(args.measurement_report,
            0 if args.case_index is None else args.case_index, args.event_index)
    elif args.case_index is not None or args.event_index is not None:
        parser.error('--case and --event require --measurement-report')
    if args.audit:
        if measurement is None:
            if args.selected is None:
                parser.error('--audit requires --run/--file or --measurement-report')
            directory, filename = resolve_selection(args.selected)
            measurement_data = PresentationData.load(directory, args.replay or filename)
        else:
            measurement_data = measurement
        print(json.dumps(measurement_data.evidence()), flush=True)
        return 0
    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
    if args.flight_view:
        if args.selected is None:
            parser.error('--flight-view requires --run/--file')
        directory, filename = resolve_selection(args.selected)
        data = PresentationData.load(directory, args.replay or filename)
        from viewer.pyvista_viewer import PyVistaViewer
        from viewer.sources import ReplaySource
        from viewer.simulation_window import SimulationWindow
        window = SimulationWindow(PyVistaViewer(data.scene), ReplaySource(data.replay, data.scene))
        window.show()
        if args.snapshot:
            def capture_flight():
                from viewer.simulation_window import capture_window
                capture_window(window, project_path(args.snapshot))
                print(json.dumps(data.evidence()), flush=True)
                window.close()
                app.quit()
            QtCore.QTimer.singleShot(700, capture_flight)
        return app.exec()
    app.setFont(QtGui.QFont('Segoe UI', 10))
    if measurement is None:
        window = ReplayBrowser(args.selected, args.replay)
    else:
        from training.ui.replay_window import ReplayWindow
        window = ReplayWindow(presentation_data=measurement)
        window.jump_to_measurement()
    window.show()
    if args.snapshot:
        def save():
            motion = window.motion if measurement is None else window
            if motion is None:
                print(window.message.text(), flush=True)
                app.exit(1)
                return
            if measurement is not None:
                motion.jump_to_measurement()
            pixels = np.ascontiguousarray(motion.plotter.screenshot(return_img=True))
            frame = window.grab()
            child = motion.plotter.interactor
            top_left = child.mapTo(window, QtCore.QPoint(0, 0))
            vtk_image = QtGui.QImage(pixels.data, pixels.shape[1], pixels.shape[0], pixels.strides[0],
                                    QtGui.QImage.Format.Format_RGB888).copy()
            painter = QtGui.QPainter(frame)
            painter.drawImage(QtCore.QRect(top_left, child.size()), vtk_image)
            painter.end()
            if not frame.save(str(project_path(args.snapshot))):
                app.exit(1)
                return
            print(json.dumps(motion.data.evidence()), flush=True)
            window.close()
            app.quit()
        QtCore.QTimer.singleShot(700, save)
    return app.exec()
