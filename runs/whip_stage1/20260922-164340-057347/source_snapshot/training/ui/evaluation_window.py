"""User-started policy evaluation controls and playback, separate from training."""

from dataclasses import replace
from datetime import datetime
import json
from pathlib import Path
import sys
from time import perf_counter

import numpy as np
from PySide6 import QtCore, QtWidgets


class EvaluationWindow(QtWidgets.QWidget):
    def __init__(self, root, backend):
        super().__init__()
        self.root, self.backend = Path(root), backend
        self.selection = self.process = self.output = None
        self.replay = self.simulation_window = self.viewer = None
        self.summary = self.scene = None
        self._close_when_finished = False
        self.setWindowTitle('Policy Evaluation')
        self.resize(620, 780)
        layout = QtWidgets.QVBoxLayout(self)
        form = QtWidgets.QFormLayout()
        layout.addLayout(form)
        self.run_combo = QtWidgets.QComboBox()
        self.run_combo.setSizeAdjustPolicy(QtWidgets.QComboBox.SizeAdjustPolicy.AdjustToMinimumContentsLengthWithIcon)
        self.run_combo.setMinimumContentsLength(22)
        self.refresh_button = QtWidgets.QPushButton('Refresh Best')
        row = QtWidgets.QHBoxLayout()
        row.addWidget(self.run_combo, 1)
        row.addWidget(self.refresh_button)
        form.addRow('Run', row)
        self.policy_combo = QtWidgets.QComboBox()
        self.browse_button = QtWidgets.QPushButton('Browse…')
        row = QtWidgets.QHBoxLayout()
        row.addWidget(self.policy_combo, 1)
        row.addWidget(self.browse_button)
        form.addRow('Policy', row)
        self.details = QtWidgets.QLabel()
        self.details.setWordWrap(True)
        layout.addWidget(self.details)
        self.evaluate_button = QtWidgets.QPushButton('Evaluate')
        self.evaluate_button.setMinimumHeight(34)
        layout.addWidget(self.evaluate_button)
        self.status = QtWidgets.QLabel('Select a policy. Evaluation starts only when you press Evaluate.')
        self.status.setWordWrap(True)
        layout.addWidget(self.status)
        self.progress = QtWidgets.QProgressBar()
        self.progress.setRange(0, 0)
        self.progress.hide()
        layout.addWidget(self.progress)
        box = QtWidgets.QGroupBox('Evaluation')
        metrics = QtWidgets.QFormLayout(box)
        self.loaded_label = QtWidgets.QLabel('—')
        self.loaded_label.setWordWrap(True)
        metrics.addRow('Loaded policy', self.loaded_label)
        self.metric_fields = {}
        self.metric_labels = {}
        for key, label in (
            ('return_mean', 'Reward'), ('score_mean', 'Whip + target score'),
            ('acceleration_cost_mean', 'Acceleration penalty'), ('bend_mean', 'Bend component'),
            ('speed_score_mean', 'Speed component'), ('target_score_mean', 'Target component'),
            ('closest_distance_mean_m', 'Closest target distance (m)'),
            ('selected_distance_mean_m', 'Scored-event distance (m)'),
            ('selected_tip_speed_mean_mps', 'Scored-event tip speed (m/s)')):
            field = QtWidgets.QLabel('—')
            field.setTextInteractionFlags(QtCore.Qt.TextInteractionFlag.TextSelectableByMouse)
            metrics.addRow(label, field)
            self.metric_fields[key] = field
            self.metric_labels[key] = (metrics.labelForField(field), label)
        self.comparison = QtWidgets.QLabel('—')
        self.comparison.setWordWrap(True)
        metrics.addRow('Reproduction', self.comparison)
        layout.addWidget(box)
        controls = QtWidgets.QHBoxLayout()
        self.play_button = QtWidgets.QPushButton('Play')
        self.replay_button = QtWidgets.QPushButton('Replay')
        self.speed_combo = QtWidgets.QComboBox()
        for label, value in (('0.1×', .1), ('0.25×', .25), ('0.5×', .5), ('1×', 1.)):
            self.speed_combo.addItem(label, value)
        self.speed_combo.setCurrentIndex(3)
        controls.addWidget(self.play_button)
        controls.addWidget(self.replay_button)
        controls.addWidget(QtWidgets.QLabel('Speed'))
        controls.addWidget(self.speed_combo)
        layout.addLayout(controls)
        self.timeline = QtWidgets.QSlider(QtCore.Qt.Orientation.Horizontal)
        self.timeline.setRange(0, 0)
        layout.addWidget(self.timeline)
        self.time_label = QtWidgets.QLabel('0.00 / 0.00 s')
        layout.addWidget(self.time_label)
        self.log = QtWidgets.QPlainTextEdit()
        self.log.setReadOnly(True)
        self.log.setMaximumBlockCount(200)
        self.log.setMaximumHeight(90)
        layout.addWidget(self.log)
        self.refresh_button.clicked.connect(self.refresh_runs)
        self.run_combo.currentIndexChanged.connect(self.select_run)
        self.policy_combo.currentIndexChanged.connect(self.select_policy)
        self.browse_button.clicked.connect(self.browse_policy)
        self.evaluate_button.clicked.connect(self.start_evaluation)
        self.play_button.clicked.connect(self.toggle_play)
        self.replay_button.clicked.connect(self.restart_play)
        self.timeline.valueChanged.connect(self.seek)
        self.speed_combo.currentIndexChanged.connect(self.change_speed)
        self.timer = QtCore.QTimer(self)
        self.timer.timeout.connect(self.refresh_playback)
        self.timer.start(50)
        self.set_playback_enabled(False)
        self.refresh_runs()

    def refresh_runs(self):
        selected = self.run_combo.currentData()
        runs = []
        paths = [p for name in ('whip_single_scenario', 'whip_stage1') for p in (self.root/'runs'/name).glob('*')]
        for path in sorted(paths, key=lambda p: p.name, reverse=True):
            try:
                protocol = self.backend.read_json(path/'protocol.json')
                if (protocol.get('schema') in ('whip_single_scenario_v1', 'whip_stage1_v1')
                        and protocol['experiment']['experiment']['mode'] == 'training'):
                    runs.append(str(path.resolve()))
            except (OSError, ValueError, KeyError):
                continue
        if selected and selected not in runs:
            runs.append(selected)
        self.run_combo.blockSignals(True)
        self.run_combo.clear()
        for path in runs:
            self.run_combo.addItem(Path(path).name, path)
        if selected in runs:
            self.run_combo.setCurrentIndex(runs.index(selected))
        self.run_combo.blockSignals(False)
        self.select_run()

    def select_run(self, *_):
        run = self.run_combo.currentData()
        records = self.backend.evaluated_policies(run) if run else []
        best = self.backend.best_policy(records) if records else None
        self.policy_combo.blockSignals(True)
        self.policy_combo.clear()
        for record in records:
            label = f"Seed {record['seed']} · Update {record['update']} · {record['metrics']['return_mean']:.6f}"
            self.policy_combo.addItem(label+(' · Best' if record == best else ''), record)
        if best is not None:
            self.policy_combo.setCurrentIndex(records.index(best))
        self.policy_combo.blockSignals(False)
        self.select_policy()

    def select_policy(self, *_):
        self.selection = None
        self.evaluate_button.setEnabled(False)
        row = self.policy_combo.currentData()
        if row is None:
            self.details.setText('No completed policy evaluation in this run. Refresh after an evaluation is saved.')
            return
        try:
            record = self.backend.inspect_policy(row['checkpoint'])
        except (OSError, ValueError, KeyError) as exc:
            self.details.setText(str(exc))
            return
        self.selection = record
        raw = record['protocol']['experiment']
        if raw['experiment']['name'] == 'whip_stage1':
            request = raw['request']
            speed_label = 'Minimum tip speed' if raw['matching'].get('speed_mode', 'match') == 'minimum' else 'Tip speed'
            self.details.setText(f"Stage 1 · Height: {request['release_height_world_m']:g} m · "
                f"{speed_label}: {request['tip_speed_mps']:g} m/s\n"
                'Best is the highest evaluated CEM mean within this run, not the best sampled trajectory.')
        else:
            self.details.setText(f"Target: {tuple(raw['experiment']['target_position_world_m'])} m · "
                                 f"Width (σ): {raw['reward']['target_width_m']:g} m\n"
                                 'Best is the highest completed deterministic return within this run.')
        if 'reach' in raw:
            reach = raw['reach']
            self.details.setText(f"Reach window: {tuple(raw['experiment']['target_position_world_m'])} m · "
                f"{reach['window_width_m']:g} × {reach['window_height_m']:g} m\n"
                'Best is the highest completed deterministic return within this run.')
        if 'workspace' in raw:
            workspace = raw['workspace']
            self.details.setText(self.details.text()+
                f"\nWorkspace: radius {workspace['radius_m']:g} m · "
                f"Z {workspace['height_lower_world_m']:g}–{workspace['height_upper_world_m']:g} m")
        self.details.setToolTip(record['checkpoint'])
        self.evaluate_button.setEnabled(True)

    def browse_policy(self):
        path, _ = QtWidgets.QFileDialog.getOpenFileName(self, 'Select evaluated policy',
            str(self.root/'runs'), 'Policy checkpoint (*.pt)')
        if not path:
            return
        try:
            record = self.backend.inspect_policy(path)
        except (OSError, ValueError, KeyError) as exc:
            QtWidgets.QMessageBox.warning(self, 'Cannot select policy', str(exc))
            return
        index = self.run_combo.findData(record['run'])
        if index < 0:
            self.run_combo.addItem(Path(record['run']).name, record['run'])
            index = self.run_combo.count()-1
        self.run_combo.setCurrentIndex(index)
        self.select_run()
        for index in range(self.policy_combo.count()):
            if self.policy_combo.itemData(index)['checkpoint'] == record['checkpoint']:
                self.policy_combo.setCurrentIndex(index)
                break

    def set_playback_enabled(self, enabled):
        for widget in (self.play_button, self.replay_button, self.timeline, self.speed_combo):
            widget.setEnabled(enabled)

    def start_evaluation(self):
        if self.process is not None or self.selection is None:
            return
        # Pin the previewed identities. The worker checks them again before simulation.
        selected = self.selection
        if self.simulation_window is not None:
            self.simulation_window.close()
        self.replay = self.summary = None
        self.set_playback_enabled(False)
        for label in self.metric_fields.values():
            label.setText('—')
        self.loaded_label.setText(f"Seed {selected['seed']} · Update {selected['update']}")
        self.comparison.setText('—')
        self.log.clear()
        self.status.setText('Preparing evaluation… Training is not modified.')
        self.output = self.root/'runs/whip_evaluation'/datetime.now().strftime('%Y%m%d-%H%M%S-%f')
        self.progress.show()
        for widget in (self.run_combo, self.policy_combo, self.refresh_button, self.browse_button, self.evaluate_button):
            widget.setEnabled(False)
        process = QtCore.QProcess(self)
        self.process = process
        process.setWorkingDirectory(str(self.root))
        environment = QtCore.QProcessEnvironment.systemEnvironment()
        environment.insert('PYTHONIOENCODING', 'utf-8')
        process.setProcessEnvironment(environment)
        executable = Path(sys.executable)
        if executable.name.lower() == 'pythonw.exe':
            executable = executable.with_name('python.exe')
        process.setProgram(str(executable))
        process.setArguments(['-u', '-m', self.backend.__name__, selected['checkpoint'],
                              str(self.output), json.dumps(selected['hashes'])])
        process.setProcessChannelMode(QtCore.QProcess.ProcessChannelMode.MergedChannels)
        process.readyReadStandardOutput.connect(self.read_output)
        process.finished.connect(self.finished)
        process.errorOccurred.connect(self.process_error)
        process.start()

    def read_output(self):
        if self.process is not None:
            text = bytes(self.process.readAllStandardOutput()).decode('utf-8', errors='replace').strip()
            if text:
                self.log.appendPlainText(text)
                phases = ('Loading saved model', 'Preparing one simulation',
                          'Evaluating at 100 Hz', 'Evaluation complete.')
                for line in text.splitlines():
                    if not self._close_when_finished and line.startswith(phases):
                        self.status.setText(line)

    def process_error(self, error):
        if error == QtCore.QProcess.ProcessError.FailedToStart:
            self.log.appendPlainText(self.process.errorString())
            self.finished(-1, QtCore.QProcess.ExitStatus.CrashExit)

    def finished(self, code, exit_status):
        if self.process is None:
            return
        self.read_output()
        process, self.process = self.process, None
        process.deleteLater()
        self.progress.hide()
        for widget in (self.run_combo, self.policy_combo, self.refresh_button, self.browse_button):
            widget.setEnabled(True)
        self.evaluate_button.setEnabled(self.selection is not None)
        try:
            if code != 0 or exit_status != QtCore.QProcess.ExitStatus.NormalExit:
                raise RuntimeError('Evaluation failed. See the log; training was not changed.')
            self.load_result(self.output, show_scene=not self._close_when_finished)
        except Exception as exc:
            self.status.setText(str(exc))
            self.log.appendPlainText(str(exc))
        if self._close_when_finished:
            self.close()

    def load_result(self, output, *, show_scene=True):
        from simulation.configuration import load_configuration
        from viewer.replay import StateReplay
        summary = self.backend.read_json(Path(output)/'summary.json')
        self.summary = summary
        self.loaded_label.setText(f"{Path(summary['source_run']).name}\nSeed {summary['seed']} · Update {summary['update']}")
        reach_fields = {
            'score_mean': ('score_mean', 'Reach score'),
            'target_score_mean': ('reach_progress_mean', 'Reach progress'),
            'closest_distance_mean_m': ('closest_window_distance_mean_m', 'Closest window distance (m)'),
            'selected_distance_mean_m': ('selected_window_distance_mean_m', 'Scored reach distance (m)'),
        } if 'reach' in summary else {}
        if 'stage1' in summary:
            reach_fields = {
                'score_mean': ('score_mean', 'Stage 1 score'),
                'bend_mean': ('mechanical_mean', 'Mechanics'),
                'speed_score_mean': ('matching_mean', 'Request matching'),
                'target_score_mean': ('time_factor_mean', 'Time factor'),
                'closest_distance_mean_m': ('release_height_world_m_mean', 'Release height (m)'),
                'selected_distance_mean_m': ('release_time_s_mean', 'Release time (s)'),
                'selected_tip_speed_mean_mps': ('tip_speed_mps_mean', 'Release tip speed (m/s)'),
            }
        for key, field in self.metric_fields.items():
            label, default_label = self.metric_labels[key]
            metric_key, text = reach_fields.get(key, (key, default_label))
            label.setText(text)
            value = summary['metrics'][metric_key]
            field.setText('—' if value is None else f'{value:.6f}')
        same = summary['metrics_equal'] and summary['saved_action_equal']
        self.comparison.setText('Matches saved evaluation exactly' if same else
            f"Differs from saved evaluation · reward Δ {summary['return_difference']:+.9g}")
        self.comparison.setStyleSheet('' if same else 'color: #a65d10;')
        reasons = summary['rejection_reasons']
        self.status.setText('Rejected: '+', '.join(reasons) if reasons else 'Ready · simulated evaluation, not real-flight evidence')
        self.status.setToolTip(str(output))
        if not summary['playable']:
            self.status.setText(self.status.text()+'. No numerically valid motion to replay.')
            return
        with np.load(Path(output)/'rollout.npz', allow_pickle=False) as data:
            self.replay = StateReplay({key: data[key] for key in data.files})
        scene = load_configuration(Path(output)/'scene.toml')
        target = summary.get('target_position_world_m')
        self.scene = replace(scene, viewer=replace(scene.viewer,
            target_position_world_m=None if target is None else np.asarray(target, dtype=float)))
        self.replay.speed = self.speed_combo.currentData()
        self.timeline.blockSignals(True)
        self.timeline.setRange(0, len(self.replay.time)-1)
        self.timeline.setValue(0)
        self.timeline.blockSignals(False)
        self.set_playback_enabled(True)
        if show_scene:
            self.open_scene()
        self.refresh_playback()

    def open_scene(self):
        if self.simulation_window is not None or self.replay is None:
            return
        import pyvista as pv
        from simulation.controller_interface import ControlInput
        from viewer.pyvista_viewer import PyVistaViewer
        from viewer.simulation_window import SimulationWindow
        viewer = PyVistaViewer(self.scene)
        window = SimulationWindow(viewer, self.replay, ControlInput)
        window.setWindowTitle(f"Policy Evaluation · Update {self.summary['update']}")
        self.viewer, self.simulation_window = viewer, window
        window.closed.connect(self.scene_closed)
        a = self.replay.arrays
        for name, points, color in (('Command', a['commands'][:, :3], '#397dcc'),
                                    ('Drone', a['drone_positions'], '#178760'),
                                    ('Cable tip', a['cable_positions'][:, -1], '#cc6c34')):
            window.plotter.add_mesh(pv.lines_from_points(points), color=color, line_width=2,
                                    lighting=False, label=name, pickable=False)
        workspace_bounds = None
        if self.summary.get('workspace') is not None:
            from viewer.workspace import add_cylinder_boundary
            workspace_bounds = add_cylinder_boundary(window.plotter, **self.summary['workspace'])
        window_points = np.empty((0, 3))
        if 'reach' in self.summary:
            center = np.asarray(self.summary['target_position_world_m'], dtype=float)
            initial = np.asarray(self.summary['reach_initial_position_world_m'], dtype=float)
            normal = center[:2]-initial[:2]
            normal /= np.linalg.norm(normal)
            side = np.array([-normal[1], normal[0], 0.])
            reach = self.summary['reach']
            window_points = np.array([center+s*side*reach['window_width_m']/2
                +np.array([0., 0., z*reach['window_height_m']/2]) for s, z in
                ((-1, -1), (1, -1), (1, 1), (-1, 1))])
            window.plotter.add_mesh(pv.lines_from_points(np.vstack((window_points, window_points[:1]))),
                color='#be3884', line_width=3, lighting=False, pickable=False, label='Reach window')
        legend_size = (.24, .20) if 'reach' in self.summary else ((.24, .16) if workspace_bounds is not None else (.18, .12))
        window.plotter.add_legend(bcolor='white', size=legend_size)
        target = self.summary.get('target_position_world_m')
        target_points = np.empty((0, 3)) if target is None else np.asarray(target)[None]
        points = np.concatenate((a['commands'][:, :3], a['drone_positions'],
                                 a['cable_positions'].reshape(-1, 3),
                                 target_points, window_points))
        bounds = np.column_stack((points.min(0)-.12, points.max(0)+.12)).ravel()
        if workspace_bounds is not None:
            bounds[::2] = np.minimum(bounds[::2], workspace_bounds[::2]-.12)
            bounds[1::2] = np.maximum(bounds[1::2], workspace_bounds[1::2]+.12)
        window.plotter.reset_camera(bounds=bounds)
        viewer._initial_camera = list(window.plotter.camera_position)
        window.show()
        window.plotter.render()

    def scene_closed(self):
        if self.replay is not None:
            self.replay.running = False
        self.simulation_window = self.viewer = None

    def render_state(self):
        if self.viewer is not None:
            self.viewer._last_tick = perf_counter()
            self.viewer._update_geometry(self.replay.state)
            self.viewer._update_status()

    def toggle_play(self):
        if self.replay is not None:
            self.open_scene()
            self.replay.toggle_running()
            self.render_state()
            self.refresh_playback()

    def restart_play(self):
        if self.replay is not None:
            self.open_scene()
            self.replay.reset()
            self.replay.running = True
            self.render_state()
            self.refresh_playback()

    def seek(self, index):
        if self.replay is not None:
            self.replay.running = False
            self.replay.seek(index)
            self.render_state()
            self.refresh_playback()

    def change_speed(self, *_):
        if self.replay is not None:
            self.replay.speed = self.speed_combo.currentData()

    def refresh_playback(self):
        if self.replay is None:
            return
        self.timeline.blockSignals(True)
        self.timeline.setValue(self.replay.index)
        self.timeline.blockSignals(False)
        self.time_label.setText(f'{self.replay.state.time_s:.2f} / {self.replay.duration_s:.2f} s')
        self.play_button.setText('Pause' if self.replay.running else 'Play')

    def closeEvent(self, event):
        if self.process is not None:
            answer = QtWidgets.QMessageBox.question(self, 'Evaluation running',
                'Close after this evaluation finishes? Training will continue unchanged.',
                QtWidgets.QMessageBox.StandardButton.Yes | QtWidgets.QMessageBox.StandardButton.No)
            if answer == QtWidgets.QMessageBox.StandardButton.Yes:
                self._close_when_finished = True
                self.status.setText('Finishing this evaluation before closing…')
                if self.process is None:  # It may finish inside the modal event loop.
                    QtCore.QTimer.singleShot(0, self.close)
            event.ignore()
            return
        self.timer.stop()
        if self.simulation_window is not None:
            self.simulation_window.close()
        event.accept()
