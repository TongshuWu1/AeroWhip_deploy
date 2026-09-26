"""User-owned training process and saved-run inspection, separate from simulation."""

import hashlib
from copy import deepcopy
import json
from pathlib import Path
import sys
from time import perf_counter

from PySide6 import QtCore, QtGui, QtWidgets

from .plots import MetricPlot
from .records import RunHistory, load_history


def duration(seconds):
    if seconds is None:
        return '—'
    seconds = max(0, int(seconds))
    return f'{seconds//3600:02d}:{seconds//60%60:02d}:{seconds%60:02d}'


class TrainingWindow(QtWidgets.QMainWindow):
    def __init__(self, root, configuration, configuration_loader, *, worker_module='tasks.whip.worker', runs_directory=None,
                 resume_inspector=None, resume_checkpoint=None, total_updates=None, additional_runs_directories=()):
        super().__init__()
        self.root, self.configuration_loader = Path(root), configuration_loader
        self.worker_module = worker_module
        self.resume_inspector = resume_inspector
        self.resume_info = None
        self.runs_directory = Path(runs_directory) if runs_directory is not None else self.root/'runs'
        self.run_directories = (self.runs_directory, *(Path(p) for p in additional_runs_directories))
        self.process = None
        self.preview = None
        self.preview_hash = None
        self.history = RunHistory()
        self._started = None
        self._elapsed = None
        self._stop_requested = False
        self._close_when_finished = False
        self._terminal = None
        self._stdout = b''
        self._active_seed = None
        self.setWindowTitle('Training')
        self.resize(1180, 850)
        self.setMinimumSize(960, 740)
        central = QtWidgets.QWidget()
        self.setCentralWidget(central)
        layout = QtWidgets.QVBoxLayout(central)
        layout.setContentsMargins(18, 16, 18, 16)
        layout.setSpacing(12)
        heading = QtWidgets.QHBoxLayout()
        title = QtWidgets.QLabel('Training')
        font = title.font(); font.setPointSize(20); font.setBold(True); title.setFont(font)
        heading.addWidget(title)
        self.status = QtWidgets.QLabel('Ready')
        heading.addWidget(self.status, 1)
        self.start_button = QtWidgets.QPushButton('Start')
        self.start_button.setMinimumWidth(100)
        self.start_button.setToolTip('Start the selected new or resumed experiment in a separate saved run.')
        self.start_button.clicked.connect(self.start)
        self.stop_button = QtWidgets.QPushButton('Stop')
        self.stop_button.setMinimumWidth(100)
        self.stop_button.setEnabled(False)
        self.stop_button.setToolTip('Finish the current update and save a checkpoint before exiting.')
        self.stop_button.clicked.connect(self.request_stop)
        heading.addWidget(self.start_button); heading.addWidget(self.stop_button)
        layout.addLayout(heading)

        setup = QtWidgets.QGroupBox('Experiment setup')
        form = QtWidgets.QVBoxLayout(setup)
        row = QtWidgets.QHBoxLayout()
        self.config_combo = QtWidgets.QComboBox()
        self.config_combo.setMinimumWidth(300)
        for path in sorted((self.root/'configs/experiments').glob('*.toml')):
            self.add_path(self.config_combo, path)
        self.config_combo.setCurrentIndex(self.add_path(self.config_combo, configuration))
        self.config_combo.currentIndexChanged.connect(self.preview_configuration)
        self.config_browse = QtWidgets.QPushButton('Browse…')
        self.config_browse.clicked.connect(self.browse_configuration)
        row.addWidget(self.config_combo, 1); row.addWidget(self.config_browse)
        form.addLayout(row)
        controls = QtWidgets.QHBoxLayout()
        self.run_mode = QtWidgets.QComboBox()
        self.run_mode.addItems(['New', 'Resume'])
        self.run_mode.setCurrentIndex(1 if resume_checkpoint is not None else 0)
        controls.addWidget(self.run_mode)
        self.resume_path = QtWidgets.QLineEdit(str(resume_checkpoint) if resume_checkpoint is not None else '')
        self.resume_path.setReadOnly(True)
        self.resume_path.setPlaceholderText('Checkpoint')
        controls.addWidget(self.resume_path, 1)
        self.resume_browse = QtWidgets.QPushButton('Browse…')
        self.resume_browse.clicked.connect(self.browse_checkpoint)
        controls.addWidget(self.resume_browse)
        controls.addWidget(QtWidgets.QLabel('Total updates'))
        self.total_updates = QtWidgets.QSpinBox()
        self.total_updates.setRange(1, 2**31-1)
        if total_updates is None:
            try:
                total_updates = configuration_loader(configuration)[0]['experiment']['updates']
            except (OSError, ValueError, TypeError, KeyError):
                total_updates = 1
        self.total_updates.setValue(total_updates)
        self.total_updates.setToolTip('Final cumulative update number, including completed updates when resuming.')
        controls.addWidget(self.total_updates)
        form.addLayout(controls)
        self.resume_details = QtWidgets.QLabel()
        self.resume_details.setTextFormat(QtCore.Qt.TextFormat.PlainText)
        form.addWidget(self.resume_details)
        self.run_mode.currentIndexChanged.connect(self.preview_configuration)
        self.total_updates.valueChanged.connect(self.preview_configuration)
        self.config_details = QtWidgets.QLabel()
        self.config_details.setWordWrap(True)
        self.config_details.setTextFormat(QtCore.Qt.TextFormat.PlainText)
        form.addWidget(self.config_details)
        layout.addWidget(setup)

        cards = QtWidgets.QHBoxLayout()
        self.cards = {}
        for label in ('Seed', 'Update', 'Episodes', 'Elapsed', 'Update time', 'Remaining (est.)'):
            box = QtWidgets.QGroupBox(label)
            box_layout = QtWidgets.QVBoxLayout(box)
            value = QtWidgets.QLabel('—')
            value_font = value.font(); value_font.setPointSize(13); value.setFont(value_font)
            box_layout.addWidget(value)
            cards.addWidget(box, 1)
            self.cards[label] = value
        self.cards['Remaining (est.)'].setToolTip('Estimated from completed update times; excludes extra policy evaluations and setup.')
        self.cards['Episodes'].setToolTip('Sampled episodes; separate deterministic policy evaluations are not included.')
        layout.addLayout(cards)
        self.progress_bar = QtWidgets.QProgressBar()
        self.progress_bar.setRange(0, 1); self.progress_bar.setValue(0)
        self.progress_bar.setFormat('%v / %m updates')
        layout.addWidget(self.progress_bar)

        row = QtWidgets.QHBoxLayout()
        row.addWidget(QtWidgets.QLabel('Run'))
        self.run_combo = QtWidgets.QComboBox()
        self.run_combo.setSizePolicy(QtWidgets.QSizePolicy.Policy.Expanding, QtWidgets.QSizePolicy.Policy.Fixed)
        self.run_combo.currentIndexChanged.connect(self.choose_run)
        row.addWidget(self.run_combo, 1)
        self.run_browse = QtWidgets.QPushButton('Browse…')
        self.run_browse.clicked.connect(self.browse_run)
        self.refresh_button = QtWidgets.QPushButton('Refresh')
        self.refresh_button.clicked.connect(lambda: (self.refresh_runs(), self.choose_run()))
        row.addWidget(self.run_browse); row.addWidget(self.refresh_button)
        row.addWidget(QtWidgets.QLabel('Plot seed'))
        self.seed_combo = QtWidgets.QComboBox()
        self.seed_combo.setMinimumWidth(65)
        self.seed_combo.currentIndexChanged.connect(self.refresh_plots)
        row.addWidget(self.seed_combo)
        layout.addLayout(row)

        self.tabs = QtWidgets.QTabWidget()
        charts = QtWidgets.QWidget()
        grid = QtWidgets.QGridLayout(charts)
        grid.setContentsMargins(0, 8, 0, 0)
        self.return_plot = MetricPlot('Episode return')
        self.return_plot.setToolTip('Batch mean includes rejected episodes. Valid mean excludes them. Policy evaluation uses deterministic actions in a separate batch-one rollout.')
        self.component_plot = MetricPlot('Event components', limits=(0., 1.))
        self.component_plot.setToolTip('Component means over valid episodes with an eligible scored event. Not physical whip-success rates or additive return terms.')
        self.validity_plot = MetricPlot('Episode validity', limits=(0., 1.))
        self.validity_plot.setToolTip('Validity is command/numerical/envelope eligibility, not target success or a physical safety certificate.')
        grid.addWidget(self.return_plot, 0, 0, 1, 2)
        grid.addWidget(self.component_plot, 1, 0); grid.addWidget(self.validity_plot, 1, 1)
        self.event_details = QtWidgets.QLabel()
        self.event_details.setWordWrap(True)
        self.event_details.hide()
        grid.addWidget(self.event_details, 2, 0, 1, 2)
        grid.setRowStretch(0, 1); grid.setRowStretch(1, 1)
        self.tabs.addTab(charts, 'Learning curves')

        results = QtWidgets.QWidget()
        results_layout = QtWidgets.QVBoxLayout(results)
        self.run_details = QtWidgets.QLabel('No run selected')
        self.run_details.setTextFormat(QtCore.Qt.TextFormat.PlainText)
        self.run_details.setWordWrap(True)
        self.run_details.setTextInteractionFlags(QtCore.Qt.TextInteractionFlag.TextSelectableByMouse)
        results_layout.addWidget(self.run_details)
        self.files = QtWidgets.QTreeWidget()
        self.files.setHeaderLabels(['Seed', 'Update', 'Policy return', 'Checkpoint', 'Replay CSV'])
        self.files.setRootIsDecorated(False)
        self.files.setSelectionMode(QtWidgets.QAbstractItemView.SelectionMode.SingleSelection)
        self.files.header().setSectionResizeMode(QtWidgets.QHeaderView.ResizeMode.ResizeToContents)
        self.files.header().setStretchLastSection(True)
        self.files.itemSelectionChanged.connect(self.update_file_buttons)
        results_layout.addWidget(self.files, 1)
        buttons = QtWidgets.QHBoxLayout()
        self.folder_button = QtWidgets.QPushButton('Open run folder')
        self.folder_button.clicked.connect(self.open_folder)
        self.checkpoint_button = QtWidgets.QPushButton('Copy checkpoint path')
        self.checkpoint_button.clicked.connect(lambda: self.copy_path(3))
        self.csv_button = QtWidgets.QPushButton('Copy CSV path')
        self.csv_button.clicked.connect(lambda: self.copy_path(4))
        for button in (self.folder_button, self.checkpoint_button, self.csv_button):
            buttons.addWidget(button)
        buttons.addStretch()
        results_layout.addLayout(buttons)
        self.tabs.addTab(results, 'Results')
        self.log = QtWidgets.QPlainTextEdit()
        self.log.setReadOnly(True); self.log.setMaximumBlockCount(1500)
        self.tabs.addTab(self.log, 'Log')
        layout.addWidget(self.tabs, 1)
        self.message = QtWidgets.QLabel()
        self.message.setTextFormat(QtCore.Qt.TextFormat.PlainText)
        self.message.setWordWrap(True)
        layout.addWidget(self.message)
        self.timer = QtCore.QTimer(self)
        self.timer.setInterval(250)
        self.timer.timeout.connect(self.refresh_clock)
        self.timer.start()
        self.refresh_runs()
        self.preview_configuration()
        self.refresh_plots()
        self.update_file_buttons()

    @property
    def active(self):
        return self.process is not None

    @staticmethod
    def add_path(combo, path):
        value = str(Path(path).resolve())
        index = combo.findData(value)
        if index < 0:
            combo.addItem(Path(value).name, value)
            index = combo.count()-1
            combo.setItemData(index, value, QtCore.Qt.ItemDataRole.ToolTipRole)
        return index

    def preview_configuration(self, *_):
        if self.active:
            return
        self.preview = None
        self.resume_info = None
        self.start_button.setEnabled(False)
        try:
            path = Path(self.config_combo.currentData())
            digest = hashlib.sha256(path.read_bytes()).hexdigest()
            raw, *_ = self.configuration_loader(path)
            raw = deepcopy(raw)
            if hashlib.sha256(path.read_bytes()).hexdigest() != digest:
                raise ValueError('Configuration changed while loading; select it again.')
            exp, trajectory = raw['experiment'], raw['trajectory']
            training = exp['mode'] == 'training'
            self.total_updates.setEnabled(training)
            self.run_mode.setEnabled(training and self.resume_inspector is not None)
            resuming = training and self.run_mode.currentText() == 'Resume'
            self.resume_browse.setEnabled(resuming)
            self.resume_path.setEnabled(resuming)
            self.resume_details.clear()
            if training:
                exp['updates'] = self.total_updates.value()
            if resuming:
                if self.resume_inspector is None or not self.resume_path.text():
                    raise ValueError('Select a checkpoint to resume')
                self.resume_info = self.resume_inspector(Path(self.resume_path.text()), raw)
                done = self.resume_info['update']
                self.resume_details.setText(f"Resume update {done} → {exp['updates']} · {exp['updates']-done} additional updates")
            exploration = raw.get('exploration')
            exploration_text = (f"Smooth · Time: {exploration['smoothness_time_s']:g} s · Scale: {exploration['amplitude_scale']:g}"
                                if exploration else 'Independent')
            budget = (f"Seed: {exp['seeds'][0]}  ·  Samples: {exp['batch_size']}  ·  Optimizer updates: 0"
                      if exp['mode'] == 'initialization' else
                      f"Seeds: {', '.join(map(str, exp['seeds']))}  ·  Batch: {exp['batch_size']}  ·  Updates/seed: {exp['updates']}  ·  Evaluate every {exp['evaluation_interval']} updates")
            stage1 = exp['name'] == 'whip_stage1'
            if stage1:
                request = raw['request']
                speed_label = 'Minimum tip speed' if raw['matching'].get('speed_mode', 'match') == 'minimum' else 'Tip speed'
                target_text = (f"Stage 1 · Release height: {request['release_height_world_m']:g} m · "
                    f"{speed_label}: {request['tip_speed_mps']:g} m/s · Heading/elevation: "
                    f"{request['heading_rad']:g}/{request['elevation_rad']:g} rad")
                if 'strike_surface' in raw:
                    wall = raw['strike_surface']
                    target_text = (f"Directed Stage 1 · Wall: {wall['distance_m']:g} m forward · "
                        f"Height: {request['release_height_world_m']:g} m · "
                        f"Region: {wall['width_m']:g} × {wall['height_m']:g} m · "
                        f"Entry ≤ {wall['maximum_entry_angle_deg']:g}° · Forward speed ≥ {request['tip_speed_mps']:g} m/s")
                exploration_text = (f"CEM · Initial σ: {raw['cem']['initial_std']:g} · "
                    f"Elites: {raw['cem']['elites']} · {exploration_text}")
            else:
                target_text = f"Target: {tuple(exp['target_position_world_m'])} m  ·  Width (σ): {raw['reward']['target_width_m']:g} m"
            if 'reach' in raw:
                reach = raw['reach']
                target_text = (f"Reach window: {tuple(exp['target_position_world_m'])} m · "
                    f"{reach['window_width_m']:g} × {reach['window_height_m']:g} m · "
                    f"Quality bonus ≤ {reach['whip_bonus_weight']:.0%}")
            self.config_details.setText(
                f"Model: {Path(exp['model']).name}  ·  {exp['device'].upper()}  ·  {exp['mode'].capitalize()}\n"
                f"{target_text}  ·  {trajectory['control_count']} spline controls\n"
                f"Commands: {trajectory.get('command_rate_hz', 100):g} Hz · Feedback: 100 Hz\n"
                f"{budget}\n"
                f"Exploration: {exploration_text}")
            if 'workspace' in raw:
                workspace = raw['workspace']
                self.config_details.setText(self.config_details.text()+
                    f"\nWorkspace: cylinder · Radius {workspace['radius_m']:g} m · "
                    f"Z {workspace['height_lower_world_m']:g}–{workspace['height_upper_world_m']:g} m")
            self.config_details.setToolTip(str(path))
            self.preview, self.preview_hash = raw, digest
            if self.history.path is None and not self.history.updates:
                self.history.configuration = raw
                self.refresh_plots()
            self.start_button.setEnabled(True)
            self.message.clear()
        except (OSError, ValueError, TypeError, KeyError) as exc:
            self.config_details.setText('Cannot load experiment')
            self.message.setText(str(exc))

    def browse_checkpoint(self):
        path, _ = QtWidgets.QFileDialog.getOpenFileName(self, 'Resume checkpoint',
            str(self.runs_directory), 'Policy checkpoint (*.pt)')
        if path:
            self.resume_path.setText(path)
            self.preview_configuration()

    def browse_configuration(self):
        path, _ = QtWidgets.QFileDialog.getOpenFileName(self, 'Select experiment',
            str(self.root/'configs/experiments'), 'Experiment files (*.toml)')
        if path:
            self.config_combo.setCurrentIndex(self.add_path(self.config_combo, path))
            self.preview_configuration()

    def start(self):
        if self.active or self.preview is None:
            return
        path = Path(self.config_combo.currentData())
        options = {}
        try:
            if hashlib.sha256(path.read_bytes()).hexdigest() != self.preview_hash:
                self.preview_configuration()
                self.message.setText('Configuration changed. Review the updated settings, then press Start.')
                return
            if self.preview['experiment']['mode'] == 'training':
                options['total_updates'] = self.preview['experiment']['updates']
            if self.resume_info is not None:
                current = self.resume_inspector(Path(self.resume_path.text()), self.preview)
                if current != self.resume_info:
                    self.preview_configuration()
                    self.message.setText('Checkpoint records changed. Review the selection, then press Start.')
                    return
                options.update(resume_checkpoint=current['checkpoint'], expected_resume_sha256=current['sha256'])
        except (OSError, ValueError, TypeError, KeyError) as exc:
            self.message.setText(str(exc)); return
        self.history = RunHistory(configuration=self.preview, status='Starting')
        if self.resume_info is not None:
            self.history.add_prefix(self.resume_info['history'])
        self._started, self._elapsed = perf_counter(), None
        self._stop_requested = self._close_when_finished = False
        self._terminal, self._stdout, self._active_seed = None, b'', None
        self.seed_combo.clear(); self.log.clear(); self.message.clear()
        for seed in self.history.seeds:
            self.ensure_seed(seed)
        self.status.setText('Starting')
        self.process = QtCore.QProcess(self)
        self.process.setWorkingDirectory(str(self.root))
        environment = QtCore.QProcessEnvironment.systemEnvironment()
        environment.insert('PYTHONUTF8', '1')
        self.process.setProcessEnvironment(environment)
        self.process.readyReadStandardOutput.connect(self.read_stdout)
        self.process.readyReadStandardError.connect(self.read_stderr)
        self.process.started.connect(self.process_started)
        self.process.finished.connect(self.process_finished)
        self.process.errorOccurred.connect(self.process_error)
        self.set_running_controls(True)
        self.refresh_plots(); self.refresh_files(); self.refresh_progress()
        self.process.start(sys.executable, ['-u', '-m', self.worker_module, str(path), self.preview_hash,
                                            json.dumps(options, allow_nan=False)])

    def set_running_controls(self, running):
        for widget in (self.config_combo, self.config_browse, self.run_combo, self.run_browse, self.refresh_button,
                       self.run_mode, self.resume_path, self.resume_browse, self.total_updates):
            widget.setEnabled(not running)
        self.start_button.setEnabled(not running and self.preview is not None)
        self.stop_button.setEnabled(running and not self._stop_requested)

    def process_started(self):
        if self._stop_requested:
            self.process.write(b'stop\n')

    def request_stop(self):
        if not self.active or self._stop_requested:
            return
        self._stop_requested = True
        self.stop_button.setEnabled(False)
        self.status.setText('Stopping')
        self.message.setText('Finishing the current operation and saving the checkpoint…')
        if self.process.state() == QtCore.QProcess.ProcessState.Running:
            self.process.write(b'stop\n')

    def read_stdout(self):
        if not self.active:
            return
        self._stdout += bytes(self.process.readAllStandardOutput())
        while b'\n' in self._stdout:
            line, self._stdout = self._stdout.split(b'\n', 1)
            self.handle_line(line.decode('utf-8', errors='replace'))

    def read_stderr(self):
        if self.active:
            text = bytes(self.process.readAllStandardError()).decode('utf-8', errors='replace').strip()
            if text:
                self.log.appendPlainText(text)

    def handle_line(self, line):
        try:
            event = json.loads(line)
        except ValueError:
            if line.strip():
                self.log.appendPlainText(line.strip())
            return
        if not isinstance(event, dict) or 'event' not in event:
            self.log.appendPlainText(line)
            return
        kind = event['event']
        if kind == 'run':
            if event.get('resume') is not None:
                self.history = load_history(event['path'])
                for seed in self.history.seeds:
                    self.ensure_seed(seed)
            self.history.path = Path(event['path'])
            self.history.configuration = event['configuration']
            self.run_combo.blockSignals(True)
            self.run_combo.setCurrentIndex(self.add_path(self.run_combo, self.history.path))
            self.run_combo.blockSignals(False)
            self.refresh_files()
        elif kind == 'phase':
            if not self._stop_requested:
                self.status.setText(event['phase'])
            if 'seed' in event:
                seed = event['seed']
                self.cards['Seed'].setText(str(seed))
                exp = self.history.configuration['experiment']
                self.cards['Update'].setText('0' if exp['mode'] == 'initialization' else f"{event['update']} / {exp['updates']}")
                if self._active_seed != seed:
                    self._active_seed = seed
                    self.ensure_seed(seed, select=True)
        elif kind == 'update':
            record = event['record']
            self.history.add_update(record)
            self.ensure_seed(record['seed'])
            self.log.appendPlainText(f"Seed {record['seed']} · Update {record['update']} · Return {record['samples']['return_mean']:.6g} · {record['elapsed_s']:.2f} s")
            self.refresh_plots(); self.refresh_progress()
        elif kind == 'evaluation':
            self.history.add_evaluation(event['seed'], event['update'], event['metrics'])
            self.ensure_seed(event['seed'])
            self.refresh_plots(); self.refresh_files()
        elif kind == 'seed_finished':
            self.refresh_files()
        elif kind == 'initialization':
            self.log.appendPlainText('Initialization samples saved; zero optimizer updates.')
            self.history.initialization = event['metrics']
            self.refresh_progress(); self.refresh_plots()
        elif kind == 'finished':
            self._terminal = event['summary']['status'].capitalize()
        elif kind == 'error':
            self._terminal = 'Failed'
            self.message.setText(event['message'])
            self.log.appendPlainText(event['message'])

    def process_error(self, error):
        if not self.active:
            return
        self.message.setText(self.process.errorString())
        self.log.appendPlainText(self.process.errorString())
        if error == QtCore.QProcess.ProcessError.FailedToStart:
            self.process_finished(-1, QtCore.QProcess.ExitStatus.CrashExit)

    def process_finished(self, code, exit_status):
        if not self.active:
            return
        self.read_stdout(); self.read_stderr()
        if self._stdout.strip():
            self.handle_line(self._stdout.decode('utf-8', errors='replace'))
        self._elapsed = perf_counter()-self._started
        self._started = None
        final_status = self._terminal if code == 0 and self._terminal else 'Failed'
        self.process.deleteLater(); self.process = None
        self.history.status = final_status
        if self.history.path:
            try:
                self.history = load_history(self.history.path)
                self.history.status = final_status
            except (OSError, ValueError, KeyError, TypeError) as exc:
                self.log.appendPlainText(str(exc))
        self.status.setText(final_status)
        if final_status == 'Failed' and not self.message.text():
            self.message.setText('Worker exited without completing the run. See Log and saved run files.')
        elif final_status != 'Failed':
            saved = any(row[3] is not None for row in self.history.policy_files())
            self.message.setText(('Checkpoint and completed records saved.' if saved else 'Stopped before training started.')
                                 if final_status == 'Stopped' else 'Run completed.')
        self.set_running_controls(False)
        terminal_message = self.message.text()
        self.preview_configuration()
        if terminal_message:
            self.message.setText(terminal_message)
        self.refresh_runs(); self.refresh_files(); self.refresh_plots(); self.refresh_progress(); self.refresh_clock()
        if self._close_when_finished:
            self.close()

    def ensure_seed(self, seed, *, select=False):
        index = self.seed_combo.findData(seed)
        if index < 0:
            self.seed_combo.addItem(str(seed), seed)
            index = self.seed_combo.count()-1
        if select:
            self.seed_combo.setCurrentIndex(index)

    def refresh_runs(self):
        if self.active:
            return
        selected = str(self.history.path) if self.history.path else self.run_combo.currentData()
        self.run_combo.blockSignals(True)
        self.run_combo.clear(); self.run_combo.addItem('Select saved run', None)
        paths = {path for folder in self.run_directories for path in folder.glob('*')}
        for path in sorted(paths, key=lambda p: p.name, reverse=True):
            if (path/'protocol.json').is_file():
                self.add_path(self.run_combo, path)
        if selected:
            self.run_combo.setCurrentIndex(self.add_path(self.run_combo, selected))
        self.run_combo.blockSignals(False)

    def browse_run(self):
        path = QtWidgets.QFileDialog.getExistingDirectory(self, 'Select run', str(self.root/'runs'))
        if path:
            self.run_combo.setCurrentIndex(self.add_path(self.run_combo, path))
            self.choose_run()

    def choose_run(self, *_):
        if self.active or not self.run_combo.currentData():
            return
        try:
            history = load_history(self.run_combo.currentData())
        except (OSError, ValueError, KeyError, TypeError) as exc:
            self.message.setText(f'Cannot read run: {exc}')
            return
        self.history = history
        self._elapsed = history.elapsed_seconds
        self.status.setText(history.status)
        self.seed_combo.clear()
        for seed in history.seeds:
            self.ensure_seed(seed)
        self.message.clear()
        self.refresh_progress(); self.refresh_plots(); self.refresh_files(); self.refresh_clock()

    def refresh_plots(self, *_):
        seed = self.seed_combo.currentData()
        stage1 = self.history.configuration.get('experiment', {}).get('name') == 'whip_stage1'
        records = sorted(self.history.updates.get(seed, {}).items())
        evaluations = sorted(self.history.evaluations.get(seed, {}).items())
        initial = self.history.initialization
        if initial.get('seed') == seed and 'return_mean' in initial:
            records = [(0, {'samples': initial})]+records
        samples = lambda key: [(step, row['samples'].get(key)) for step, row in records]
        self.return_plot.set_series([
            ('Batch mean', samples('return_mean'), False),
            ('Valid mean', samples('valid_return_mean'), False),
            ('Mean evaluation' if stage1 else 'Policy evaluation',
             [(step, row.get('return_mean')) for step, row in evaluations], True)])
        reach = 'reach' in self.history.configuration
        self.component_plot.title = 'Stage 1 components' if stage1 else ('Reach components' if reach else 'Event components')
        self.component_plot.setToolTip(
            'Means over valid episodes at the selected reach sample. Quality is the existing B × V proxy; bonus is additive.'
            if reach else 'Means over valid episodes with an eligible proxy event; not a physical whip-success rate.')
        components = (('Progress', 'reach_progress_mean'), ('Quality', 'quality_mean'), ('Bonus', 'reach_bonus_mean')) if reach else (
            ('Bend', 'bend_mean'), ('Speed', 'speed_score_mean'), ('Target', 'target_score_mean'))
        if stage1:
            components = (('Mechanics', 'mechanical_mean'), ('Matching', 'matching_mean'), ('Time', 'time_factor_mean'))
            self.component_plot.setToolTip('Multiplicative factors at the same event; means over valid positive-score events, not whip-success rates.')
        self.component_plot.set_series([(label, samples(key), False) for label, key in components])
        self.event_details.setVisible(stage1)
        latest = records[-1][1]['samples'] if records else {}
        def number(key, unit):
            value = latest.get(key)
            return '—' if value is None else f'{value:.3f} {unit}'
        self.event_details.setText('Batch event means · Height: '+number('release_height_world_m_mean', 'm')+
            ' · Tip speed: '+number('tip_speed_mps_mean', 'm/s')+' · Time: '+number('release_time_s_mean', 's'))
        directed = 'strike_surface' in self.history.configuration
        if directed:
            self.event_details.setText(self.event_details.text()+' · Wall distance: '+number('wall_distance_m_mean', 'm')+
                ' · Forward speed: '+number('forward_mps_mean', 'm/s'))
        valid = samples('valid_fraction')
        series = [('Valid', valid, False),
            ('Rejected', [(step, 1-value if value is not None else None) for step, value in valid], False)]
        if directed:
            series.append(('Wall strike', samples('strike_fraction'), False))
        self.validity_plot.set_series(series)
        self.validity_plot.setToolTip('Wall strike is the fraction of valid episodes with a qualified forward crossing anywhere in the episode, not a whip-success rate.'
            if directed else 'Validity is command/numerical/envelope eligibility, not target success or a physical safety certificate.')

    def refresh_progress(self):
        exp = self.history.configuration.get('experiment', {})
        total = exp.get('updates', 1)*len(exp.get('seeds', [0]))
        if exp.get('mode') == 'initialization':
            self.progress_bar.setRange(0, 0 if self.active else 1)
            if not self.active:
                self.progress_bar.setValue(1 if self.history.status == 'Completed' else 0)
            self.progress_bar.setFormat('Initialization')
        else:
            self.progress_bar.setRange(0, total)
            self.progress_bar.setValue(self.history.completed_updates)
            self.progress_bar.setFormat('%v / %m updates')
        self.cards['Episodes'].setText(f'{self.history.episodes:,}')
        if not self.active:
            seeds = self.history.seeds
            seed = seeds[-1] if seeds else None
            update = max(self.history.updates.get(seed, {}), default=0)
            self.cards['Seed'].setText(str(seed) if seed is not None else '—')
            self.cards['Update'].setText(f"{update} / {exp.get('updates', '—')}" if seeds else '—')
            if exp.get('mode') == 'initialization':
                self.cards['Update'].setText('0')
        times = [row['elapsed_s'] for rows in self.history.updates.values() for row in rows.values()]
        mean = sum(times[-10:])/len(times[-10:]) if times else None
        self.cards['Update time'].setText(f'{mean:.2f} s' if mean is not None else '—')
        remaining = (total-self.history.completed_updates)*mean if self.active and mean is not None else None
        self.cards['Remaining (est.)'].setText(duration(remaining))

    def refresh_clock(self):
        elapsed = perf_counter()-self._started if self._started is not None else self._elapsed
        self.cards['Elapsed'].setText(duration(elapsed))

    def refresh_files(self):
        self.files.clear()
        for seed, update, value, checkpoint, csv in self.history.policy_files():
            item = QtWidgets.QTreeWidgetItem([str(seed), str(update), f'{value:.6g}' if value is not None else '—',
                checkpoint.name if checkpoint else '—', csv.name if csv else '—'])
            for col, path in ((3, checkpoint), (4, csv)):
                item.setData(col, QtCore.Qt.ItemDataRole.UserRole, str(path) if path else None)
                item.setToolTip(col, str(path) if path else 'Not saved')
            self.files.addTopLevelItem(item)
        if self.history.path:
            exp = self.history.configuration.get('experiment', {})
            task = (f"Stage 1 · Request: {self.history.configuration['request']}" if exp.get('name') == 'whip_stage1'
                    else f"Target: {exp.get('target_position_world_m', '—')}")
            self.run_details.setText(f"{self.history.path}\nModel: {exp.get('model', '—')} · {task}")
        else:
            self.run_details.setText('No run selected')
        self.update_file_buttons()

    def update_file_buttons(self):
        item = self.files.currentItem()
        self.folder_button.setEnabled(self.history.path is not None)
        for col, button in ((3, self.checkpoint_button), (4, self.csv_button)):
            button.setEnabled(item is not None and bool(item.data(col, QtCore.Qt.ItemDataRole.UserRole)))

    def copy_path(self, col):
        item = self.files.currentItem()
        if item and item.data(col, QtCore.Qt.ItemDataRole.UserRole):
            QtWidgets.QApplication.clipboard().setText(item.data(col, QtCore.Qt.ItemDataRole.UserRole))

    def open_folder(self):
        if self.history.path:
            QtGui.QDesktopServices.openUrl(QtCore.QUrl.fromLocalFile(str(self.history.path)))

    def closeEvent(self, event):
        if not self.active:
            self.timer.stop()
            event.accept()
            return
        event.ignore()
        if self._close_when_finished:
            return
        answer = QtWidgets.QMessageBox.question(self, 'Stop training?',
            'Stop after the current operation, save the checkpoint, and close?',
            QtWidgets.QMessageBox.StandardButton.Yes | QtWidgets.QMessageBox.StandardButton.No,
            QtWidgets.QMessageBox.StandardButton.No)
        if answer == QtWidgets.QMessageBox.StandardButton.Yes:
            self._close_when_finished = True
            self.request_stop()
