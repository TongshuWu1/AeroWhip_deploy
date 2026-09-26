"""User-started policy evaluation controls and playback, separate from training."""

from dataclasses import replace
from datetime import datetime
import json
from pathlib import Path, PureWindowsPath
import sys
from time import perf_counter

import numpy as np
from PySide6 import QtCore, QtWidgets

from .records import run_display_name


class EvaluationWindow(QtWidgets.QWidget):
    def __init__(self, root, backend):
        super().__init__()
        self.root, self.backend = Path(root).resolve(), backend
        self.runs_folder = self.root/'runs'
        self.saved_selection = None
        self.selection = self.process = self.output = None
        self.replay = self.simulation_window = self.viewer = None
        self.scene_controls = None
        self.summary = self.scene = None
        self.release_index = self.release_position = self.desired_height = None
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
        self.folder_button = QtWidgets.QPushButton('Folder…')
        self.folder_button.setToolTip(str(self.runs_folder))
        row = QtWidgets.QHBoxLayout()
        row.addWidget(self.run_combo, 1)
        row.addWidget(self.folder_button)
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
        self.saved_button = QtWidgets.QPushButton('View saved')
        self.open_replay_button = QtWidgets.QPushButton('Open replay…')
        row = QtWidgets.QHBoxLayout()
        row.addWidget(self.saved_button)
        row.addWidget(self.evaluate_button)
        row.addWidget(self.open_replay_button)
        layout.addLayout(row)
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
        self.release_fields = {}
        for key, label in (('position', 'Release XYZ (m)'), ('height', 'Desired height (m)'),
                           ('wall', 'Wall distance (m)'), ('entry', 'Entry angle (°)'),
                           ('forward', 'Forward speed (m/s)'), ('strike', 'Wall strike')):
            field = QtWidgets.QLabel('—')
            field.setTextInteractionFlags(QtCore.Qt.TextInteractionFlag.TextSelectableByMouse)
            metrics.addRow(label, field)
            self.release_fields[key] = (metrics.labelForField(field), field)
            metrics.labelForField(field).hide()
            field.hide()
        self.comparison = QtWidgets.QLabel('—')
        self.comparison.setWordWrap(True)
        metrics.addRow('Reproduction', self.comparison)
        layout.addWidget(box)
        controls = QtWidgets.QHBoxLayout()
        self.play_button = QtWidgets.QPushButton('Play')
        self.replay_button = QtWidgets.QPushButton('Replay')
        self.release_button = QtWidgets.QPushButton('Release')
        self.release_button.setToolTip('Pause at the scored cable-tip release event; the cable stays attached.')
        self.release_button.hide()
        self.speed_combo = QtWidgets.QComboBox()
        for label, value in (('0.1×', .1), ('0.25×', .25), ('0.5×', .5), ('1×', 1.)):
            self.speed_combo.addItem(label, value)
        self.speed_combo.setCurrentIndex(3)
        controls.addWidget(self.play_button)
        controls.addWidget(self.replay_button)
        controls.addWidget(self.release_button)
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
        self.folder_button.clicked.connect(self.choose_runs_folder)
        self.run_combo.currentIndexChanged.connect(self.select_run)
        self.policy_combo.currentIndexChanged.connect(self.select_policy)
        self.browse_button.clicked.connect(self.browse_policy)
        self.evaluate_button.clicked.connect(self.start_evaluation)
        self.saved_button.clicked.connect(self.view_saved)
        self.open_replay_button.clicked.connect(self.open_saved_replay)
        self.play_button.clicked.connect(self.toggle_play)
        self.replay_button.clicked.connect(self.restart_play)
        self.release_button.clicked.connect(self.seek_release)
        self.timeline.valueChanged.connect(self.seek)
        self.timeline.sliderPressed.connect(self.pause_playback)
        self.speed_combo.currentIndexChanged.connect(self.change_speed)
        self.timer = QtCore.QTimer(self)
        self.timer.timeout.connect(self.refresh_playback)
        self.timer.start(50)
        self.set_playback_enabled(False)
        self.refresh_runs()

    def refresh_runs(self):
        selected = self.run_combo.currentData()
        self._discovery_error = None
        try:
            runs = [str(path) for path in self.backend.discover_runs(self.runs_folder)]
        except OSError as exc:
            runs = []
            self._discovery_error = str(exc)
        self.run_combo.blockSignals(True)
        self.run_combo.clear()
        for path in runs:
            self.run_combo.addItem(run_display_name(path), path)
            self.run_combo.setItemData(self.run_combo.count()-1, path, QtCore.Qt.ItemDataRole.ToolTipRole)
        if selected in runs:
            self.run_combo.setCurrentIndex(runs.index(selected))
        self.run_combo.blockSignals(False)
        self.select_run()

    def choose_runs_folder(self):
        folder = QtWidgets.QFileDialog.getExistingDirectory(self, 'Select results folder', str(self.runs_folder))
        if folder:
            self.runs_folder = Path(folder).resolve()
            self.folder_button.setToolTip(str(self.runs_folder))
            self.refresh_runs()

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
        self.selection = self.saved_selection = None
        self.evaluate_button.setEnabled(False)
        self.saved_button.setEnabled(False)
        row = self.policy_combo.currentData()
        if row is None:
            if not self.run_combo.count():
                self.details.setText(self._discovery_error or
                    f'No saved training runs found in:\n{self.runs_folder}\n'
                    'Choose Folder or copy/download the saved runs. Source code alone does not contain results.')
            else:
                self.details.setText('No completed policy evaluations in this run. Copy the complete run folder, including seed folders and policy .pt/.json/.npz files.')
            return
        try:
            record = self.backend.inspect_saved_policy(row['checkpoint'])
        except (OSError, ValueError, KeyError) as exc:
            self.details.setText(str(exc))
            return
        self.saved_selection = record
        checkpoint = Path(record['checkpoint'])
        has_motion = checkpoint.with_name(checkpoint.stem+'_motion.npz').is_file()
        self.saved_button.setEnabled(has_motion)
        self.saved_button.setToolTip('Play recorded states; no physics or training.' if has_motion else
            'No dense motion was saved for this checkpoint. Open an existing replay or Evaluate.')
        raw = record['protocol']['experiment']
        if raw['experiment']['name'] == 'whip_stage1':
            request = raw['request']
            speed_label = 'Minimum tip speed' if raw['matching'].get('speed_mode', 'match') == 'minimum' else 'Tip speed'
            self.details.setText(f"Stage 1 · Height: {request['release_height_world_m']:g} m · "
                f"{speed_label}: {request['tip_speed_mps']:g} m/s\n"
                'Best is the highest evaluated CEM mean within this run, not the best sampled trajectory.')
            if 'strike_surface' in raw:
                wall = raw['strike_surface']
                self.details.setText(f"Directed Stage 1 · Wall: {wall['distance_m']:g} m forward · "
                    f"Height: {request['release_height_world_m']:g} m\n"
                    f"Region: {wall['width_m']:g} × {wall['height_m']:g} m · "
                    f"Entry ≤ {wall['maximum_entry_angle_deg']:g}° · Forward ≥ {request['tip_speed_mps']:g} m/s")
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
        try:
            self.selection = self.backend.inspect_policy(row['checkpoint'])
        except (OSError, ValueError, KeyError) as exc:
            self.details.setText(self.details.text()+f'\nRe-evaluation unavailable: {exc}')
        self.evaluate_button.setEnabled(self.selection is not None)

    def browse_policy(self):
        path, _ = QtWidgets.QFileDialog.getOpenFileName(self, 'Select evaluated policy',
            str(self.runs_folder), 'Policy checkpoint (*.pt)')
        if not path:
            return
        try:
            record = self.backend.inspect_saved_policy(path)
        except (OSError, ValueError, KeyError) as exc:
            QtWidgets.QMessageBox.warning(self, 'Cannot select policy', str(exc))
            return
        index = self.run_combo.findData(record['run'])
        if index < 0:
            self.run_combo.addItem(run_display_name(record['run']), record['run'])
            index = self.run_combo.count()-1
        self.run_combo.setCurrentIndex(index)
        self.select_run()
        for index in range(self.policy_combo.count()):
            if self.policy_combo.itemData(index)['checkpoint'] == record['checkpoint']:
                self.policy_combo.setCurrentIndex(index)
                break

    def view_saved(self):
        if self.process is not None or self.saved_selection is None:
            return
        try:
            result = self.backend.saved_motion(self.saved_selection['checkpoint'])
            self.load_result(None, saved=result)
        except (OSError, ValueError, KeyError) as exc:
            self.status.setText(str(exc))

    def open_saved_replay(self):
        path, _ = QtWidgets.QFileDialog.getOpenFileName(self, 'Open saved replay',
            str(self.runs_folder), 'Replay summary (summary.json)')
        if path:
            try:
                result = self.backend.saved_replay(Path(path).parent)
                self.load_result(Path(path).parent, saved=result)
            except (OSError, ValueError, KeyError) as exc:
                self.status.setText(str(exc))

    def set_playback_enabled(self, enabled):
        for widget in (self.play_button, self.replay_button, self.timeline, self.speed_combo):
            widget.setEnabled(enabled)
        self.release_button.setEnabled(enabled and self.release_index is not None)

    def start_evaluation(self):
        if self.process is not None or self.selection is None:
            return
        # Pin the previewed identities. The worker checks them again before simulation.
        selected = self.selection
        if self.simulation_window is not None:
            self.simulation_window.close()
        self.replay = self.summary = None
        self.configure_release()
        self.set_playback_enabled(False)
        for label in self.metric_fields.values():
            label.setText('—')
        self.loaded_label.setText(f"Seed {selected['seed']} · Update {selected['update']}")
        self.comparison.setText('—')
        self.log.clear()
        self.status.setText('Preparing evaluation… Training is not modified.')
        self.output = self.root/'runs/whip_evaluation'/datetime.now().strftime('%Y%m%d-%H%M%S-%f')
        self.progress.show()
        for widget in (self.run_combo, self.policy_combo, self.refresh_button, self.browse_button,
                       self.evaluate_button, self.folder_button, self.saved_button, self.open_replay_button):
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
        for widget in (self.run_combo, self.policy_combo, self.refresh_button, self.browse_button,
                       self.folder_button, self.open_replay_button):
            widget.setEnabled(True)
        self.saved_button.setEnabled(self.saved_selection is not None and Path(
            self.saved_selection['checkpoint']).with_name(
            Path(self.saved_selection['checkpoint']).stem+'_motion.npz').is_file())
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

    def load_result(self, output, *, show_scene=True, saved=None):
        from simulation.configuration import load_configuration
        from viewer.replay import StateReplay
        if self.simulation_window is not None:
            self.simulation_window.close()
        self.replay = self.summary = None
        self.configure_release()
        self.set_playback_enabled(False)
        summary = saved[0] if saved is not None else self.backend.read_json(Path(output)/'summary.json')
        self.summary = summary
        source_run = summary['source_run']
        name = run_display_name(source_run) if Path(source_run).is_dir() else PureWindowsPath(source_run).name
        self.loaded_label.setText(f"{name}\nSeed {summary['seed']} · Update {summary['update']}")
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
        if summary.get('saved_only'):
            self.comparison.setText('Saved result · not re-evaluated')
            self.comparison.setStyleSheet('')
        else:
            same = summary['metrics_equal'] and summary['saved_action_equal']
            self.comparison.setText('Matches saved evaluation exactly' if same else
                f"Differs from saved evaluation · reward Δ {summary['return_difference']:+.9g}")
            self.comparison.setStyleSheet('' if same else 'color: #a65d10;')
        reasons = summary['rejection_reasons']
        self.status.setText('Rejected: '+', '.join(reasons) if reasons else 'Ready · simulated evaluation, not real-flight evidence')
        self.status.setToolTip(str(saved[2].parent if saved is not None else output))
        if not summary['playable']:
            self.status.setText(self.status.text()+'. No numerically valid motion to replay.')
            return
        if saved is not None:
            self.replay = StateReplay(saved[1])
        else:
            with np.load(Path(output)/'rollout.npz', allow_pickle=False) as data:
                self.replay = StateReplay({key: data[key] for key in data.files})
        self.configure_release()
        scene = load_configuration(saved[2] if saved is not None else Path(output)/'scene.toml')
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

    def configure_release(self):
        """Display the saved selected event, including its crossing interpolation."""
        self.release_index = self.release_position = self.desired_height = None
        stage1 = self.summary is not None and 'stage1' in self.summary
        directed = stage1 and 'strike_surface' in self.summary
        self.release_button.setText('Event' if directed else 'Release')
        self.release_button.setToolTip('Pause at the nearest 100 Hz state (within 5 ms); the marker uses the interpolated scored event.'
            if directed else 'Pause at the scored cable-tip release event; the cable stays attached.')
        self.release_button.setVisible(stage1)
        self.release_button.setEnabled(False)
        for key, (label, field) in self.release_fields.items():
            visible = stage1 and (directed or key in ('position', 'height'))
            label.setVisible(visible)
            field.setVisible(visible)
            field.setText('—')
        self.release_fields['position'][0].setText('Event XYZ (m)' if directed else 'Release XYZ (m)')
        if not stage1:
            return
        if directed:
            metrics = self.summary['metrics']
            for field, key, factor in (('wall', 'wall_distance_m_mean', 1),
                                       ('entry', 'direction_error_rad_mean', 180/np.pi),
                                       ('forward', 'forward_mps_mean', 1)):
                value = metrics.get(key)
                self.release_fields[field][1].setText('—' if value is None else f'{value*factor:.6f}')
            success, selected = metrics.get('strike_fraction'), metrics.get('directed_strike_mean')
            self.release_fields['strike'][1].setText('—' if success is None else
                ('Selected event' if selected == 1 else ('Elsewhere in episode' if success == 1 else 'No')))
            self.release_fields['strike'][1].setToolTip('Forward plane crossing inside the region with the configured angle/speed. Not a whip classification.')
        height = self.summary['stage1'].get('release_height_world_m')
        if isinstance(height, (int, float)) and not isinstance(height, bool) and np.isfinite(height):
            self.desired_height = float(height)
            self.release_fields['height'][1].setText(f'{height:.6f}')
        indices = self.summary.get('event_indices', [])
        if (self.replay is not None and len(indices) == 3
                and all(type(index) is int for index in indices)
                and 0 <= indices[0] < indices[1] <= indices[2] < len(self.replay.time)):
            self.release_index = indices[2]
            self.release_position = self.replay.arrays['cable_positions'][self.release_index, -1].copy()
            if directed:
                fraction = self.summary.get('event_fraction')
                if (not isinstance(fraction, (int, float)) or not np.isfinite(fraction)
                        or not 0 <= fraction <= 1 or (fraction > 0 and self.release_index+1 >= len(self.replay.time))):
                    raise ValueError('Invalid saved crossing interpolation')
                if fraction > 0:
                    next_point = self.replay.arrays['cable_positions'][self.release_index+1, -1]
                    self.release_position += fraction*(next_point-self.release_position)
                    self.release_index += int(fraction >= .5)
            self.release_fields['position'][1].setText(', '.join(f'{value:.6f}' for value in self.release_position))

    def add_release_guides(self, plotter):
        """Evaluation-only world-space guides; no task logic in the simulator."""
        import pyvista as pv
        plane_points = np.empty((0, 3))
        surface = self.summary.get('strike_surface')
        if surface is not None:
            center = np.asarray(surface['center_world_m'])
            normal = np.asarray(surface['normal_world'])
            side = np.array([-normal[1], normal[0], 0.])*surface['width_m']/2
            up = np.array([0., 0., surface['height_m']/2])
            plane_points = np.array([center-side-up, center+side-up, center+side+up, center-side+up])
            plotter.add_mesh(pv.PolyData(plane_points, faces=[4, 0, 1, 2, 3]),
                name='strike_region', color='#397dcc', opacity=.15, lighting=False, pickable=False)
            plotter.add_mesh(pv.lines_from_points(np.vstack((plane_points, plane_points[:1]))),
                name='strike_outline', label='Strike region', color='#397dcc', line_width=3, lighting=False, pickable=False)
            plotter.add_mesh(pv.Arrow(start=center-.25*normal, direction=normal, scale=.5),
                name='strike_direction', label='Entry direction', color='#397dcc', lighting=False, pickable=False)
        elif self.desired_height is not None:
            xy = self.replay.arrays['cable_positions'][..., :2].reshape(-1, 2)
            lower, upper = xy.min(0)-.12, xy.max(0)+.12
            plane_points = np.array([[x, y, self.desired_height] for x, y in
                ((lower[0], lower[1]), (upper[0], lower[1]),
                 (upper[0], upper[1]), (lower[0], upper[1]))])
            plotter.add_mesh(pv.PolyData(plane_points, faces=[4, 0, 1, 2, 3]),
                name='release_height_plane', color='#b8942a', opacity=.10,
                lighting=False, pickable=False)
            plotter.add_mesh(pv.lines_from_points(np.vstack((plane_points, plane_points[:1]))),
                name='release_height_outline', label='Desired height', color='#b8942a',
                opacity=.7, line_width=2, lighting=False, pickable=False)
        if self.release_position is not None:
            plotter.add_mesh(pv.Sphere(radius=.017, center=self.release_position),
                name='release_point', label='Scored event' if surface is not None else 'Release', color='#df3c4b',
                lighting=False, pickable=False)
            for axis in range(3):
                arm = np.eye(3)[axis]*.04
                plotter.add_mesh(pv.Line(self.release_position-arm, self.release_position+arm),
                    name=f'release_cross_{axis}', color='#df3c4b', line_width=3,
                    lighting=False, pickable=False)
        return plane_points

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
        self.add_scene_controls(window)
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
        release_points = self.add_release_guides(window.plotter)
        legend_size = (.24, .24) if 'stage1' in self.summary else ((.24, .20) if 'reach' in self.summary else ((.24, .16) if workspace_bounds is not None else (.18, .12)))
        window.plotter.add_legend(bcolor='white', size=legend_size)
        target = self.summary.get('target_position_world_m')
        target_points = np.empty((0, 3)) if target is None else np.asarray(target)[None]
        points = np.concatenate((a['commands'][:, :3], a['drone_positions'],
                                 a['cable_positions'].reshape(-1, 3),
                                 target_points, window_points, release_points))
        bounds = np.column_stack((points.min(0)-.12, points.max(0)+.12)).ravel()
        if workspace_bounds is not None:
            bounds[::2] = np.minimum(bounds[::2], workspace_bounds[::2]-.12)
            bounds[1::2] = np.maximum(bounds[1::2], workspace_bounds[1::2]+.12)
        window.plotter.reset_camera(bounds=bounds)
        viewer._initial_camera = list(window.plotter.camera_position)
        window.show()
        window.plotter.render()

    def add_scene_controls(self, window):
        """Replay-only controls; the standalone simulation UI stays unchanged."""
        bar = QtWidgets.QToolBar('Playback', window)
        bar.setMovable(False)
        bar.setFloatable(False)
        controls = QtWidgets.QWidget(bar)
        row = QtWidgets.QHBoxLayout(controls)
        row.setContentsMargins(10, 5, 10, 5)
        play = QtWidgets.QPushButton('Play')
        play.setMinimumWidth(72)
        play.setToolTip('Pause at the current frame or resume playback. Space also toggles playback.')
        play.clicked.connect(self.toggle_play)
        row.addWidget(play)
        timeline = QtWidgets.QSlider(QtCore.Qt.Orientation.Horizontal)
        timeline.setRange(0, len(self.replay.time)-1)
        timeline.setMinimumWidth(180)
        timeline.setToolTip('Scrub to a saved frame. Seeking pauses playback.')
        timeline.sliderPressed.connect(self.pause_playback)
        timeline.valueChanged.connect(self.seek)
        row.addWidget(timeline, 1)
        clock = QtWidgets.QLabel()
        clock.setMinimumWidth(100)
        row.addWidget(clock)
        row.addWidget(QtWidgets.QLabel('Speed'))
        speed = QtWidgets.QComboBox()
        for index in range(self.speed_combo.count()):
            speed.addItem(self.speed_combo.itemText(index), self.speed_combo.itemData(index))
        speed.currentIndexChanged.connect(self.speed_combo.setCurrentIndex)
        row.addWidget(speed)
        controls.setSizePolicy(QtWidgets.QSizePolicy.Policy.Expanding, QtWidgets.QSizePolicy.Policy.Preferred)
        bar.addWidget(controls)
        window.addToolBar(QtCore.Qt.ToolBarArea.BottomToolBarArea, bar)
        self.scene_controls = dict(play=play, timeline=timeline, clock=clock, speed=speed)
        self.refresh_playback()

    def scene_closed(self):
        if self.replay is not None:
            self.replay.running = False
        self.simulation_window = self.viewer = None
        self.scene_controls = None

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

    def pause_playback(self):
        if self.replay is not None:
            self.replay.running = False
            self.render_state()
            self.refresh_playback()

    def seek(self, index):
        if self.replay is not None:
            self.replay.running = False
            self.replay.seek(index)
            self.render_state()
            self.refresh_playback()

    def seek_release(self):
        if self.replay is not None and self.release_index is not None:
            self.open_scene()
            self.seek(self.release_index)

    def change_speed(self, *_):
        if self.replay is not None:
            self.replay.speed = self.speed_combo.currentData()
            if self.viewer is not None:
                self.viewer._last_tick = perf_counter()
            self.refresh_playback()

    def refresh_playback(self):
        if self.replay is None:
            return
        panels = [dict(play=self.play_button, timeline=self.timeline, clock=self.time_label, speed=self.speed_combo)]
        if self.scene_controls is not None:
            panels.append(self.scene_controls)
        for panel in panels:
            with QtCore.QSignalBlocker(panel['timeline']):
                panel['timeline'].setValue(self.replay.index)
            with QtCore.QSignalBlocker(panel['speed']):
                panel['speed'].setCurrentIndex(panel['speed'].findData(self.replay.speed))
            panel['clock'].setText(f'{self.replay.state.time_s:.2f} / {self.replay.duration_s:.2f} s')
            panel['play'].setText('Pause' if self.replay.running else 'Play')

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
