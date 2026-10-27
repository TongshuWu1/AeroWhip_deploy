"""Docked flight viewer shared by local simulation and saved-state replay."""
from time import perf_counter
import numpy as np
# Complete VTK's native module registration before PyVista's lazy imports.
# With the installed VTK/PyVista versions, partial initialization intermittently
# crashes in vtkParallelCore (pickle_support) or later native-object cleanup.
import vtk as _vtk  # noqa: F401
from PySide6 import QtCore, QtGui, QtWidgets
from pyvistaqt import QtInteractor

from aerowhip.viewer.drone_info import roll_pitch_yaw_deg
from aerowhip.viewer.sources import DisplayFeed, SimulationSource, ReplaySource
from aerowhip.viewer.time_histories import TimeHistories
from aerowhip.ui.style import stylesheet


# Dark flight viewer in the shared visual language (aerowhip.ui.style).
STYLE = stylesheet(dark=True)+'''
QLabel#objectTitle { font-size: 17px; font-weight: 600; }
QLabel#badge { background: #244766; color: #cfe6fb; border-radius: 3px; padding: 3px 8px; }
QFrame#playback { background: #22272e; border-top: 1px solid #3a424c; }
QWidget#inspector, QTreeWidget { border: 0; }
'''


def label(text='', *, name=None):
    widget = QtWidgets.QLabel(text)
    widget.setTextFormat(QtCore.Qt.TextFormat.PlainText)
    if name:
        widget.setObjectName(name)
    widget.setTextInteractionFlags(QtCore.Qt.TextInteractionFlag.TextSelectableByMouse)
    return widget


class Inspector(QtWidgets.QWidget):
    def __init__(self):
        super().__init__()
        self.setObjectName('inspector')
        layout = QtWidgets.QVBoxLayout(self)
        layout.setContentsMargins(14, 14, 14, 14)
        layout.setSpacing(7)
        self.title = label('No robot', name='objectTitle')
        self.source, self.timestamp, self.frame = label(name='muted'), label(name='muted'), label(name='muted')
        for widget in (self.title, self.source, self.timestamp, self.frame):
            layout.addWidget(widget)
        self.values = {}
        for section, rows in (
            ('WORLD STATE', (('Position', 'm'), ('Velocity', 'm/s'), ('Acceleration', 'm/s²'))),
            ('BODY ATTITUDE / RATE', (('Roll / pitch / yaw', 'deg'), ('Angular velocity', 'rad/s'))),
        ):
            layout.addWidget(label(section, name='section'))
            for name, unit in rows:
                layout.addWidget(label(f'{name} · {unit}', name='muted'))
                row = QtWidgets.QHBoxLayout()
                self.values[name] = []
                for axis in ('RPY' if name == 'Roll / pitch / yaw' else 'XYZ'):
                    field = label(f'{axis}  —')
                    field.setMinimumWidth(68)
                    font = QtGui.QFont('monospace'); font.setPixelSize(12)
                    field.setFont(font)
                    row.addWidget(field, 1)
                    self.values[name].append(field)
                layout.addLayout(row)
        layout.addWidget(label('CABLE', name='section'))
        self.cable = label(name='muted'); self.cable.setWordWrap(True)
        self.tip = label(); self.tip.setWordWrap(True)
        layout.addWidget(self.cable); layout.addWidget(self.tip)
        self.tracking = label(name='muted'); self.tracking.setWordWrap(True)
        self.tracking.hide(); layout.addWidget(self.tracking)
        layout.addWidget(label('MODEL FORCES', name='section'))
        self.forces = label(); self.forces.setWordWrap(True)
        layout.addWidget(self.forces)
        self.error = label(name='error'); self.error.setWordWrap(True); self.error.hide()
        layout.addWidget(self.error)
        layout.addStretch()

    def update_frame(self, frame, robot_name, error=None):
        robot = next((r for r in frame.robots if r.name == robot_name), None)
        self.title.setText('No robot' if robot is None else robot.name)
        self.source.setText('Unavailable' if robot is None else robot.provenance.value)
        self.timestamp.setText('—' if robot is None else f'State time  {robot.time_s:.3f} s · lag {max(0., frame.time_s-robot.time_s):.3f} s')
        self.frame.setText(f'Frame  {frame.frame_id} · Z up · SI units')
        values = {} if robot is None else {
            'Position': robot.drone.position_world_m,
            'Velocity': robot.drone.linear_velocity_world_mps,
            'Acceleration': robot.drone.linear_acceleration_world_mps2,
            'Roll / pitch / yaw': roll_pitch_yaw_deg(robot.drone.rotation_body_to_world),
            'Angular velocity': robot.drone.angular_velocity_body_radps,
        }
        for name, fields in self.values.items():
            vector = values.get(name)
            axes = 'RPY' if name == 'Roll / pitch / yaw' else 'XYZ'
            for axis, index, field in zip(axes, range(3), fields):
                value = None if vector is None else round(float(vector[index]), 3)
                field.setText(f'{axis}  —' if value is None else f'{axis} {0. if value == 0. else value:+.3f}')
        cable = None if robot is None else robot.cable
        self.cable.setText('Unavailable' if cable is None else
            f'{cable.provenance.value}{" · stale" if frame.time_s-cable.time_s > .25 else ""} · {len(cable.positions_world_m)} nodes\n'
            f'State time {cable.time_s:.3f} s · lag {max(0., frame.time_s-cable.time_s):.3f} s')
        self.tip.setText('Tip position unavailable' if cable is None else 'Tip [m]  '+
                         '  '.join(f'{axis} {v:+.3f}' for axis, v in zip('XYZ', cable.positions_world_m[-1])))
        tracking = None if robot is None else robot.cable_tracking
        self.tracking.setVisible(tracking is not None)
        if tracking is not None:
            count = int(tracking.observed.sum())
            confidence = (f'{tracking.confidence[tracking.observed].min():.2f}' if count else '—')
            self.tracking.setText(f'Labelled cable (blue) · {count}/{len(tracking.observed)} markers\n'
                f'{tracking.status}\nLag {max(0., frame.time_s-tracking.time_s):.3f} s'
                f'{" · stale (hidden)" if frame.time_s-tracking.time_s > .25 else ""}\n'
                f'Min association confidence: {confidence} (heuristic)\n'
                'Blue: estimated identities · gray: raw candidates')
        forces = None if robot is None else robot.forces
        self.forces.setText('Unavailable' if forces is None else
            f'Thrust  {np.linalg.norm(forces.thrust_world_n):.3f} N\n'
            f'Cable on drone  {np.linalg.norm(forces.cable_force_world_n):.3f} N')
        self.error.setText(error or ''); self.error.setVisible(bool(error))


class SimulationWindow(QtWidgets.QMainWindow):
    closed = QtCore.Signal()

    def __init__(self, viewer, simulation, control_source=None):
        super().__init__()
        self.viewer = viewer
        self.controller_panel = None
        self._closing = False
        self._tree_signature = None
        self._changing_display = False
        self.selected_robot = None
        self.source = simulation if isinstance(simulation, (SimulationSource, ReplaySource, DisplayFeed)) else SimulationSource(
            simulation, **({} if control_source is None else {'control_source': control_source}))
        self.previous_controller = None
        from aerowhip.trajectories.full_state import FullStateTrajectory
        if self.source.has_trajectory and not isinstance(self.source.command_source, FullStateTrajectory):
            self.previous_controller = self.source.command_source, self.source.command_reset
        self.setWindowTitle('AeroWhip Simulator')
        self.setStyleSheet(STYLE)
        self.resize(*(round(size/self.devicePixelRatioF()) for size in viewer._configuration.viewer.window_size_px))
        self.setMinimumSize(960, 620)
        self.setDockOptions(QtWidgets.QMainWindow.DockOption.AllowTabbedDocks |
                            QtWidgets.QMainWindow.DockOption.AllowNestedDocks)
        central = QtWidgets.QWidget(objectName='flightPage')
        layout = QtWidgets.QVBoxLayout(central); layout.setContentsMargins(0, 0, 0, 0); layout.setSpacing(0)
        self.plotter = QtInteractor(central, auto_update=False, multi_samples=4)
        layout.addWidget(self.plotter.interactor, 1)
        self._build_playback(layout)
        self.setCentralWidget(central)
        self.tree = QtWidgets.QTreeWidget()
        self.tree.setHeaderHidden(True); self.tree.setIndentation(16)
        self.tree.setMinimumWidth(190)
        display_page = QtWidgets.QWidget(objectName='flightPage')
        display_layout = QtWidgets.QVBoxLayout(display_page)
        display_layout.setContentsMargins(0, 0, 0, 0)
        display_layout.addWidget(self.tree, 1)
        scale_row = QtWidgets.QHBoxLayout()
        scale_row.setContentsMargins(10, 8, 10, 8)
        scale_row.addWidget(label('Force scale', name='muted'))
        self.force_scale = QtWidgets.QDoubleSpinBox()
        self.force_scale.setRange(.01, 2.); self.force_scale.setDecimals(2)
        self.force_scale.setSingleStep(.05); self.force_scale.setSuffix(' m/N'); self.force_scale.setValue(.25)
        self.force_scale.setToolTip('Arrow length per newton; display only.')
        self.force_scale.valueChanged.connect(viewer.set_force_scale)
        scale_row.addWidget(self.force_scale)
        display_layout.addLayout(scale_row)
        self.displays = self._dock('Displays', display_page, QtCore.Qt.DockWidgetArea.LeftDockWidgetArea)
        self.panel = Inspector()
        scroll = QtWidgets.QScrollArea(); scroll.setWidgetResizable(True); scroll.setWidget(self.panel)
        scroll.setHorizontalScrollBarPolicy(QtCore.Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        scroll.setMinimumWidth(280)
        self.sidebar = self._dock('Inspector', scroll, QtCore.Qt.DockWidgetArea.RightDockWidgetArea)
        self.histories = TimeHistories()
        self.history_dock = self._dock('Time histories · last 10 s', self.histories,
                                       QtCore.Qt.DockWidgetArea.BottomDockWidgetArea)
        self._build_toolbar()
        menu = self.menuBar().addMenu('View')
        menu.addAction(self.displays.toggleViewAction()); menu.addAction(self.sidebar.toggleViewAction())
        menu.addAction(self.history_dock.toggleViewAction())
        hint = ('Space  Play / pause    N  Step    T  Reset    R  Frame scene'
                if self.source.has_playback else 'Read-only display    R  Frame scene')
        self.statusBar().addWidget(label(hint, name='muted'))
        self.freshness = label(name='muted'); self.statusBar().addPermanentWidget(self.freshness)
        self.statusBar().setSizeGripEnabled(False)
        viewer._simulation, viewer._window = self.source, self
        viewer._dynamics_error = self.source.latest.error
        # Force arrows start hidden so the physical vehicle/cable stay prominent.
        viewer._overlay_visibility.update(thrust=False, cable=False)
        if isinstance(self.source.command_source, FullStateTrajectory):
            viewer.set_trajectory(self.source.command_source)
        viewer.create_plotter(self.source.latest.frame, plotter=self.plotter)
        viewer._last_revision = self.source.latest.revision
        self.tree.itemChanged.connect(self._display_changed)
        self.tree.currentItemChanged.connect(self._selection_changed)
        self.shortcuts = []
        for key, callback in (('Space', viewer._toggle_simulation), ('N', viewer._step_simulation),
                              ('T', viewer._reset_simulation), ('R', viewer._reset_camera),
                              ('I', viewer._toggle_info), ('B', viewer._toggle_body_axes),
                              ('W', viewer._toggle_world_axes), ('L', viewer._toggle_labels)):
            if key in ('Space', 'N', 'T') and not self.source.has_playback:
                continue
            shortcut = QtGui.QShortcut(QtGui.QKeySequence(key), self)
            shortcut.setAutoRepeat(False); shortcut.activated.connect(callback); self.shortcuts.append(shortcut)
        viewer._last_tick = perf_counter()
        self.simulation_timer = QtCore.QTimer(self)
        self.simulation_timer.setTimerType(QtCore.Qt.TimerType.PreciseTimer)
        self.simulation_timer.timeout.connect(viewer._on_timer); self.simulation_timer.start(16)
        self.telemetry_timer = QtCore.QTimer(self)
        self.telemetry_timer.timeout.connect(self.update_telemetry); self.telemetry_timer.start(100)
        # Sample the time histories faster than the readouts refresh.
        self.history_timer = QtCore.QTimer(self)
        self.history_timer.timeout.connect(lambda: self.histories.add(self.source.latest.frame, self.selected_robot))
        self.history_timer.start(20)
        # Python signal handlers may run inside a VTK render callback. They only
        # set an Event; window/GL destruction is deferred to a Qt timer callback.
        self.shutdown_timer = QtCore.QTimer(self)
        self.shutdown_timer.timeout.connect(self._check_shutdown)
        self.shutdown_timer.start(50)
        self.update_telemetry()
        self.resizeDocks([self.displays, self.sidebar], [220, 300], QtCore.Qt.Orientation.Horizontal)
        self.resizeDocks([self.history_dock], [210], QtCore.Qt.Orientation.Vertical)

    def _dock(self, title, widget, area):
        dock = QtWidgets.QDockWidget(title, self)
        dock.setObjectName(title); dock.setWidget(widget); self.addDockWidget(area, dock)
        return dock

    def _build_toolbar(self):
        toolbar = self.addToolBar('Camera'); toolbar.setMovable(False)
        toolbar.addWidget(label('AEROWHIP', name='muted')); toolbar.addSeparator()
        self.source_label = label(self.source.latest.frame.source_label, name='badge')
        self.source_label.setMaximumWidth(440)
        self.source_label.setSizePolicy(QtWidgets.QSizePolicy.Policy.Preferred, QtWidgets.QSizePolicy.Policy.Preferred)
        self.source_label.setMinimumWidth(200)
        toolbar.addWidget(self.source_label)
        self.status = label('Paused', name='muted'); toolbar.addWidget(self.status)
        spacer = QtWidgets.QWidget(); spacer.setSizePolicy(QtWidgets.QSizePolicy.Policy.Expanding, QtWidgets.QSizePolicy.Policy.Preferred)
        toolbar.addWidget(spacer)
        self.camera = QtWidgets.QComboBox(); self.camera.addItems(['Orbit', 'Top', 'Front', 'Side'])
        self.camera.currentTextChanged.connect(self.viewer.camera_view)
        toolbar.addWidget(self.camera)
        for text, callback in (('Frame scene', self.viewer._reset_camera),
                               ('Frame drone', lambda: self.viewer.camera_view(self.camera.currentText(), robot=self.selected_robot)),
                               ('Clear trails', self.viewer.clear_trails)):
            button = QtWidgets.QPushButton(text); button.clicked.connect(callback); toolbar.addWidget(button)
        self.follow = QtWidgets.QCheckBox('Follow drone')
        self.follow.toggled.connect(lambda checked: self.viewer.set_follow(self.selected_robot if checked else None))
        toolbar.addWidget(self.follow)
        self.addToolBarBreak()
        recordings = self.addToolBar('Recordings'); recordings.setMovable(False)
        self.record_button = self.stop_record_button = self.replay_record_button = None
        self.record_status = None
        if getattr(self.source, 'recording_directory', None) is not None:
            self.record_button = QtWidgets.QPushButton('Record')
            self.record_button.clicked.connect(self.start_recording)
            self.stop_record_button = QtWidgets.QPushButton('Stop && save')
            self.stop_record_button.clicked.connect(self.stop_recording)
            self.replay_record_button = QtWidgets.QPushButton('Replay last')
            self.replay_record_button.clicked.connect(lambda: self.open_recording(self.source.last_recording))
            for button in (self.record_button, self.stop_record_button, self.replay_record_button):
                recordings.addWidget(button)
            self.record_status = label('Ready to record', name='muted')
            recordings.addWidget(self.record_status)
        open_recording = QtWidgets.QPushButton('Open recording…')
        open_recording.clicked.connect(lambda: self.open_recording())
        recordings.addWidget(open_recording)

    def start_recording(self):
        try:
            path = self.source.start_recording()
            print(f'Recording: {path}', flush=True)
        except Exception as exc:
            QtWidgets.QMessageBox.warning(self, 'Recording', str(exc))
        self.update_telemetry()

    def stop_recording(self):
        path = self.source.stop_recording()
        error = self.source.recording_status['error']
        print(error or f'Recording saved: {path}', flush=True)
        self.update_telemetry()

    def open_recording(self, path=None):
        import subprocess
        import sys
        from aerowhip.paths import ROOT as root
        if path is None:
            start = getattr(self.source, 'recording_directory', None) or root/'output/custom_real'
            path, _ = QtWidgets.QFileDialog.getOpenFileName(self, 'Open saved recording',
                str(start), 'Display recording (recording.json)')
        if path:
            subprocess.Popen([sys.executable, str(root/'scripts/play_recording.py'), str(path)],
                start_new_session=True)

    def _build_playback(self, parent):
        bar = QtWidgets.QFrame(objectName='playback'); row = QtWidgets.QHBoxLayout(bar)
        row.setContentsMargins(12, 10, 12, 10); row.setSpacing(8)
        self.timeline = None
        self.clock = label('0.000 s')
        if not self.source.has_playback:
            row.addWidget(label('Live state', name='muted')); row.addWidget(self.clock); row.addStretch()
            parent.addWidget(bar)
            return
        self.play_button = QtWidgets.QPushButton('Play', objectName='primary')
        self.play_button.clicked.connect(self.viewer._toggle_simulation)
        self.step_button = QtWidgets.QPushButton('Step'); self.step_button.clicked.connect(self.viewer._step_simulation)
        self.reset_button = QtWidgets.QPushButton('Reset'); self.reset_button.clicked.connect(self.viewer._reset_simulation)
        for widget in (self.play_button, self.step_button, self.reset_button): row.addWidget(widget)
        self.speed = QtWidgets.QComboBox()
        for value in (.25, .5, 1.): self.speed.addItem(f'{value:g}×', value)
        self.speed.setCurrentIndex(2); self.speed.currentIndexChanged.connect(
            lambda _: self.viewer.set_playback_speed(self.speed.currentData()))
        self.speed.setToolTip('Pacing only; physical integration and command clocks stay unchanged.')
        row.addWidget(self.speed)
        row.addWidget(self.clock)
        if self.source.is_replay:
            self.timeline = QtWidgets.QSlider(QtCore.Qt.Orientation.Horizontal)
            self.timeline.setRange(0, len(self.source.replay.time)-1)
            self.timeline.sliderPressed.connect(lambda: setattr(self.source, 'running', False))
            self.timeline.valueChanged.connect(self.seek)
            row.addWidget(self.timeline, 1)
        else:
            row.addStretch()
            if self.source.has_trajectory:
                self.trajectory_button = QtWidgets.QPushButton('Trajectory…')
                self.trajectory_button.clicked.connect(self.open_controller); row.addWidget(self.trajectory_button)
        parent.addWidget(bar)

    def _item(self, parent, text, data=None, checked=True):
        item = QtWidgets.QTreeWidgetItem(parent, [text])
        if data is not None:
            item.setData(0, QtCore.Qt.ItemDataRole.UserRole, data)
            item.setFlags(item.flags() | QtCore.Qt.ItemFlag.ItemIsUserCheckable)
            item.setCheckState(0, QtCore.Qt.CheckState.Checked if checked else QtCore.Qt.CheckState.Unchecked)
            self._items[tuple(data)] = item
        item.setExpanded(True)
        return item

    def _rebuild_tree(self, frame):
        self._items = {}
        with QtCore.QSignalBlocker(self.tree):
            self.tree.clear()
            world = self._item(self.tree, f'World · {frame.frame_id}')
            for text, key in (('Floor', 'floor'), ('Metric grid', 'grid'), ('World axes', 'world_axes')):
                self._item(world, text, ('world', key, None))
            robots = self._item(self.tree, 'Robots')
            for robot in frame.robots:
                item = self._item(robots, robot.name, ('robot', 'robot', robot.name))
                self._item(item, robot.provenance.value)
                self._item(item, 'Body axes', ('robot', 'body_axes', robot.name), False)
                if robot.cable is not None:
                    kind = robot.cable.provenance.value
                    cable = self._item(item, f'Cable · {kind} (green tube{", translucent" if robot.cable_tracking is not None else ""})',
                                       ('robot', 'cable', robot.name))
                    self._item(cable, 'Simulated markers', ('robot', 'markers', robot.name))
                else:
                    self._item(item, 'Cable · unavailable')
                if robot.cable_tracking is not None:
                    tracking = self._item(item, 'Cable · Labelled from OptiTrack (blue spheres, C1-C10)', ('robot', 'tracking', robot.name))
                    self._item(tracking, 'Raw marker cloud (gray)', ('robot', 'raw_cloud', robot.name))
            overlays = self._item(self.tree, 'Overlays')
            for key, text in (('reference', 'Reference path'), ('trail', 'Drone trail'),
                              ('tip_trail', 'Cable-tip trail'), ('thrust', 'Model thrust'), ('cable', 'Model cable force')):
                item = self._item(overlays, text, ('overlay', key, None), key not in ('thrust', 'cable'))
                if (key in ('thrust', 'cable') and not hasattr(self.source, 'set_force_telemetry')
                        and not any(r.forces is not None for r in frame.robots)):
                    item.setDisabled(True); item.setToolTip(0, 'Model force telemetry is unavailable')
            self._item(overlays, 'Robot / marker labels', ('world', 'labels', None), False)
            names = [r.name for r in frame.robots]
            if self.selected_robot not in names:
                self.selected_robot = names[0] if names else None
            if self.selected_robot:
                self.tree.setCurrentItem(self._items[('robot', 'robot', self.selected_robot)])
        self.sync_display_checks()

    def sync_display_checks(self):
        if not hasattr(self, '_items'):
            return
        viewer = self.viewer
        with QtCore.QSignalBlocker(self.tree):
            for (kind, key, robot), item in self._items.items():
                if kind == 'overlay':
                    checked = viewer._overlay_visibility.get(key, True)
                elif kind == 'robot':
                    checked = viewer._robot_visibility.get(robot, {}).get(key, viewer._body_axes_visible if key == 'body_axes' else True)
                else:
                    checked = viewer._world_visible.get(key, viewer._world_axes_visible if key == 'world_axes' else viewer._labels_visible)
                item.setCheckState(0, QtCore.Qt.CheckState.Checked if checked else QtCore.Qt.CheckState.Unchecked)

    def _display_changed(self, item, column):
        data = item.data(0, QtCore.Qt.ItemDataRole.UserRole)
        if data is None: return
        kind, key, robot = data
        visible = item.checkState(0) == QtCore.Qt.CheckState.Checked
        # Do not delete/rebuild QTreeWidgetItems inside their own itemChanged
        # signal. The telemetry timer reconciles structural changes afterward.
        self._changing_display = True
        try:
            if kind == 'overlay': self.viewer.set_overlay_visible(key, visible)
            else: self.viewer.set_display_visible(key, visible, robot=robot)
        finally:
            self._changing_display = False

    def _selection_changed(self, item, previous):
        if item is None: return
        while item is not None:
            data = item.data(0, QtCore.Qt.ItemDataRole.UserRole)
            if data is not None and data[0] == 'robot':
                self.selected_robot = data[2]
                if self.follow.isChecked(): self.viewer.set_follow(self.selected_robot)
                self.update_telemetry()
                break
            item = item.parent()

    def update_telemetry(self):
        snapshot = self.source.latest
        status, age = snapshot.status()
        error = snapshot.error or self.viewer._dynamics_error
        if self.record_status is not None:
            recording = self.source.recording_status
            self.record_button.setEnabled(not recording['active'])
            self.stop_record_button.setEnabled(recording['active'])
            self.replay_record_button.setEnabled(self.source.last_recording is not None)
            text = (recording['error'] or
                (f'Recording · {recording["duration_s"]:.1f} s · {recording["frames"]} frames'
                 if recording['active'] else f'Saved · {recording["path"].parent.name}'
                 if recording['path'] else 'Ready to record'))
            self.record_status.setText(text)
            self.record_status.setToolTip(str(recording['path'] or ''))
        signature = tuple((r.name, r.provenance, None if r.cable is None else r.cable.provenance,
                           r.cable_tracking is not None,
                           r.forces is not None if not hasattr(self.source, 'set_force_telemetry') else None)
                          for r in snapshot.frame.robots)
        if signature != self._tree_signature and not self._changing_display:
            self._tree_signature = signature; self._rebuild_tree(snapshot.frame)
        self.panel.update_frame(snapshot.frame, self.selected_robot, error)
        self.histories.add(snapshot.frame, self.selected_robot)
        self.histories.redraw()
        self.source_label.setText(snapshot.frame.source_label)
        self.source_label.setToolTip(snapshot.frame.source_label)
        self.status.setText('Error' if error else status)
        self.status.setStyleSheet('color: #E69F00;' if status == 'Stale' or error else '')
        self.freshness.setText('Saved arrays · no physics' if self.source.is_replay else
            f'Snapshot age {age*1000:.0f} ms' if snapshot.running or error else 'Paused snapshot')
        if self.source.has_playback:
            self.play_button.setText('Pause' if snapshot.running else 'Play')
            enabled = not error or getattr(self.source, 'can_replay_errors', False)
            self.play_button.setEnabled(enabled); self.step_button.setEnabled(enabled)
        if self.timeline is not None:
            with QtCore.QSignalBlocker(self.timeline): self.timeline.setValue(snapshot.step_count)
            time_s = getattr(self.source, 'playback_time_s', snapshot.frame.time_s)
            self.clock.setText(f'{time_s:.3f} / {self.source.replay.duration_s:.3f} s')
        else:
            self.clock.setText(f'{snapshot.frame.time_s:.3f} s')

    def seek(self, index):
        if getattr(self.source, 'can_replay_errors', False):
            self.viewer.clear_trails()
        self.viewer._action(lambda: self.source.seek(index))

    def attach_controller(self, panel):
        self.controller_panel = panel
        panel.closed.connect(lambda: self._controller_closed(panel))

    def _controller_closed(self, panel):
        if self.controller_panel is panel: self.controller_panel = None
        self.update_telemetry()

    def open_controller(self):
        if not self.source.has_trajectory: return
        if self.controller_panel is not None:
            self.controller_panel.show(); self.controller_panel.raise_(); self.controller_panel.activateWindow()
            return
        from aerowhip.ui.controller.controller_window import ControllerWindow
        from aerowhip.trajectories.full_state import FullStateTrajectory
        from aerowhip.trajectories.presets import simple_motion
        from aerowhip.physics.state import quaternion_xyzw_to_rotation_matrix
        sim = self.source
        trajectory = sim.command_source
        name = 'Trajectory'
        if not isinstance(trajectory, FullStateTrajectory):
            initial = sim.configuration.initial_state
            rotation = quaternion_xyzw_to_rotation_matrix(initial.quaternion_body_to_world_xyzw)
            trajectory = simple_motion('Hover', initial.position_world_m, float(np.arctan2(rotation[1, 0], rotation[0, 0])))
            name = 'Hover'
        panel = ControllerWindow(self.viewer, sim, trajectory, previous_controller=self.previous_controller, trajectory_name=name)
        self.attach_controller(panel); panel.show()

    def closeEvent(self, event):
        if self._closing:
            event.accept(); return
        self._closing = True
        self.shutdown_timer.stop()
        self.simulation_timer.stop(); self.telemetry_timer.stop(); self.history_timer.stop()
        if self.controller_panel is not None: self.controller_panel.close()
        self.source.close()
        self.plotter.close()
        self.viewer._window = self.viewer._simulation = self.viewer._plotter = self.viewer._overlays = None
        self.closed.emit(); event.accept()

    def _check_shutdown(self):
        stop = getattr(self.viewer, 'stop_requested', None)
        if stop is not None and stop.is_set() and not self._closing:
            self.close()


def _show(viewer, source):
    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
    window = None
    try:
        window = SimulationWindow(viewer, source)
        window.showMaximized() if viewer._configuration.viewer.start_maximized else window.show()
        loop = QtCore.QEventLoop(); window.closed.connect(loop.quit); loop.exec()
    finally:
        if window is not None: window.close()
        source.close()


def show_simulation(viewer, simulation, control_source):
    _show(viewer, SimulationSource(simulation, control_source=control_source))


def show_replay(viewer, replay):
    _show(viewer, ReplaySource(replay, viewer._configuration))


def show_source(viewer, source):
    _show(viewer, source)


def capture_window(window, path):
    """Qt grabs omit the native OpenGL child on some platforms; composite it."""
    pixels = np.ascontiguousarray(window.plotter.screenshot(return_img=True))
    frame = window.grab()
    child = window.plotter.interactor
    image = QtGui.QImage(pixels.data, pixels.shape[1], pixels.shape[0], pixels.strides[0],
                         QtGui.QImage.Format.Format_RGB888).copy()
    painter = QtGui.QPainter(frame)
    painter.drawImage(QtCore.QRect(child.mapTo(window, QtCore.QPoint(0, 0)), child.size()), image)
    painter.end()
    if not frame.save(str(path)):
        raise OSError('Could not save viewer capture: '+str(path))
