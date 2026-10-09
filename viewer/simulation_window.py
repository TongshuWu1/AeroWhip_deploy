"""Qt simulation window: 3D scene, playback toggle, telemetry and keyboard hints."""

from time import perf_counter

from PySide6 import QtCore, QtGui, QtWidgets
from pyvistaqt import QtInteractor

from viewer.drone_info import roll_pitch_yaw_deg
from viewer.overlays import COLORS


class TelemetryPanel(QtWidgets.QWidget):
    def __init__(self):
        super().__init__()
        self.setObjectName("telemetry")
        self.setMinimumWidth(270)
        self.setMaximumWidth(330)
        layout = QtWidgets.QVBoxLayout(self)
        layout.setContentsMargins(18, 20, 18, 18)
        layout.setSpacing(16)
        title = QtWidgets.QLabel("Simulation")
        title.setObjectName("title")
        layout.addWidget(title)
        status_row = QtWidgets.QHBoxLayout()
        self.status = QtWidgets.QLabel("Paused")
        self.status.setObjectName("status")
        self.time = QtWidgets.QLabel("0.00 s")
        self.time.setAlignment(QtCore.Qt.AlignmentFlag.AlignRight | QtCore.Qt.AlignmentFlag.AlignVCenter)
        status_row.addWidget(self.status)
        status_row.addWidget(self.time)
        layout.addLayout(status_row)
        self.play_button = QtWidgets.QPushButton("Play")
        self.play_button.setObjectName("playbackToggle")
        self.play_button.setMinimumHeight(36)
        self.play_button.setToolTip("Run or pause here without resetting the current state.")
        playback = QtWidgets.QHBoxLayout()
        playback.addWidget(self.play_button, 1)
        self.step_button = QtWidgets.QPushButton('Step')
        self.step_button.setToolTip('Pause and advance one simulation output step.')
        self.reset_button = QtWidgets.QPushButton('Reset')
        playback.addWidget(self.step_button)
        playback.addWidget(self.reset_button)
        layout.addLayout(playback)
        controls = QtWidgets.QHBoxLayout()
        self.speed = QtWidgets.QComboBox()
        for label, speed in (('0.25×', .25), ('0.5×', .5), ('1×', 1.)):
            self.speed.addItem(label, speed)
        self.speed.setCurrentIndex(2)
        self.speed.setToolTip('Playback pacing only; physics and command timesteps are unchanged.')
        controls.addWidget(self.speed)
        self.trajectory_button = QtWidgets.QPushButton('Trajectory…')
        controls.addWidget(self.trajectory_button, 1)
        layout.addLayout(controls)
        display = QtWidgets.QGroupBox('Display')
        display_layout = QtWidgets.QGridLayout(display)
        self.overlays = {}
        for i, (key, label) in enumerate((('reference', 'Reference path'), ('trail', 'Drone trail'),
                                         ('tip_trail', 'Cable-tip trail'),
                                         ('thrust', 'Thrust'), ('cable', 'Cable force'))):
            checkbox = QtWidgets.QCheckBox(label)
            checkbox.setChecked(True)
            checkbox.setStyleSheet(f'color: {COLORS[key]};')
            display_layout.addWidget(checkbox, i//2, i%2)
            self.overlays[key] = checkbox
        self.force_scale = QtWidgets.QDoubleSpinBox()
        self.force_scale.setRange(.01, 2.)
        self.force_scale.setDecimals(2)
        self.force_scale.setSingleStep(.05)
        self.force_scale.setSuffix(' m/N')
        self.force_scale.setValue(.25)
        self.force_scale.setToolTip('Arrow length per newton, shared by both forces. Visual only.')
        display_layout.addWidget(QtWidgets.QLabel('Force scale'), 3, 0)
        display_layout.addWidget(self.force_scale, 3, 1)
        self.force_values = {}
        for row, (key, label) in enumerate((('thrust', 'Thrust'), ('cable', 'Cable force')), 4):
            heading = QtWidgets.QLabel(label)
            heading.setStyleSheet(f'color: {COLORS[key]};')
            value = QtWidgets.QLabel('-- N')
            value.setAlignment(QtCore.Qt.AlignmentFlag.AlignRight)
            value.setToolTip('Instantaneous simulated force magnitude; not measured motor data.')
            display_layout.addWidget(heading, row, 0)
            display_layout.addWidget(value, row, 1)
            self.force_values[key] = value
        layout.addWidget(display)
        self.fields = {}
        for name, unit, axes in (("Position", "m", ("X", "Y", "Z")),
                                 ("Velocity", "m/s", ("X", "Y", "Z")),
                                 ("Acceleration", "m/s²", ("X", "Y", "Z")),
                                 ("Attitude", "deg", ("Roll", "Pitch", "Yaw"))):
            card = QtWidgets.QFrame()
            card.setObjectName("card")
            grid = QtWidgets.QGridLayout(card)
            grid.setContentsMargins(12, 10, 12, 12)
            grid.setHorizontalSpacing(8)
            heading = QtWidgets.QLabel(name)
            heading.setObjectName("heading")
            units = QtWidgets.QLabel(unit)
            units.setObjectName("unit")
            units.setAlignment(QtCore.Qt.AlignmentFlag.AlignRight)
            grid.addWidget(heading, 0, 0, 1, 2)
            grid.addWidget(units, 0, 2)
            self.fields[name] = []
            for column, axis in enumerate(axes):
                label = QtWidgets.QLabel(axis)
                label.setObjectName("axis")
                value = QtWidgets.QLabel("--")
                value.setObjectName("value")
                value.setMinimumWidth(58)
                value.setAlignment(QtCore.Qt.AlignmentFlag.AlignRight | QtCore.Qt.AlignmentFlag.AlignVCenter)
                value.setTextInteractionFlags(QtCore.Qt.TextInteractionFlag.TextSelectableByMouse)
                grid.addWidget(label, 1, column)
                grid.addWidget(value, 2, column)
                grid.setColumnStretch(column, 1)
                self.fields[name].append(value)
            if name != "Attitude":
                card.setToolTip("World frame")
            layout.addWidget(card)
        self.error = QtWidgets.QLabel()
        self.error.setObjectName("error")
        self.error.setWordWrap(True)
        self.error.hide()
        layout.addWidget(self.error)
        layout.addStretch()
        self.setStyleSheet("""
            QWidget#telemetry { background: #f4f6f8; color: #243342; }
            QLabel { color: #243342; font: 10pt 'Segoe UI'; background: transparent; }
            QLabel#title { font-size: 18pt; font-weight: 600; }
            QLabel#status { color: #536b7d; font-weight: 600; }
            QPushButton#playbackToggle { background: #286ca5; color: white;
                border: none; border-radius: 5px; font: 11pt 'Segoe UI'; padding: 7px; }
            QPushButton#playbackToggle:hover { background: #205887; }
            QPushButton#playbackToggle:disabled { background: #879aaa; }
            QFrame#card { background: white; border: 1px solid #dce3e9; border-radius: 7px; }
            QLabel#heading { font-weight: 600; }
            QLabel#unit, QLabel#axis { color: #6b7c89; font-size: 9pt; }
            QLabel#value { background: #f3f6f8; border-radius: 4px; padding: 5px;
                           font: 12pt 'Consolas'; }
            QLabel#error { color: #a33131; }
        """)

    def update_state(self, state, running, error=None):
        self.status.setText("Error" if error else "Running" if running else "Paused")
        self.play_button.setText("Pause" if running else "Play")
        self.play_button.setEnabled(not bool(error))
        self.step_button.setEnabled(not bool(error))
        self.time.setText(f"{state.time_s:.2f} s")
        drone = state.drone
        for key, vector in (('thrust', None if state.forces is None else state.forces.thrust_world_n),
                            ('cable', None if state.forces is None else state.forces.cable_force_world_n)):
            magnitude = None if vector is None else float(sum(float(v)**2 for v in vector)**.5)
            self.force_values[key].setText('-- N' if magnitude is None else f'{magnitude:.3f} N')
        values = {"Position": drone.position_world_m,
                  "Velocity": drone.linear_velocity_world_mps,
                  "Acceleration": drone.linear_acceleration_world_mps2,
                  "Attitude": roll_pitch_yaw_deg(drone.rotation_body_to_world)}
        for name, vector in values.items():
            decimals = 1 if name == "Attitude" else 2
            for index, field in enumerate(self.fields[name]):
                value = None if vector is None else round(float(vector[index]), decimals)
                text = "--" if value is None else f"{0.0 if value == 0 else value:.{decimals}f}"
                if field.text() != text:
                    field.setText(text)
        self.error.setText(error or "")
        self.error.setVisible(bool(error))


class SimulationWindow(QtWidgets.QMainWindow):
    closed = QtCore.Signal()

    def __init__(self, viewer, simulation, control_source):
        super().__init__()
        self.viewer = viewer
        self.controller_panel = None
        from trajectories.full_state import FullStateTrajectory
        self.previous_controller = (None if isinstance(simulation.command_source, FullStateTrajectory)
                                    else (simulation.command_source, simulation.command_reset))
        self.setWindowTitle("AeroWhip Simulation")
        # Configuration is in physical pixels; Qt sizes widgets in logical
        # pixels. Avoid silently enlarging the render on a high-DPI monitor.
        scale = self.devicePixelRatioF()
        self.resize(*(round(size / scale) for size in viewer._configuration.viewer.window_size_px))
        self.setMinimumSize(760, 480)
        central = QtWidgets.QWidget()
        layout = QtWidgets.QHBoxLayout(central)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)
        self.plotter = QtInteractor(central, auto_update=False, multi_samples=4)
        layout.addWidget(self.plotter.interactor, 1)
        self.panel = TelemetryPanel()
        self.panel.play_button.clicked.connect(viewer._toggle_simulation)
        self.panel.step_button.clicked.connect(viewer._step_simulation)
        self.panel.reset_button.clicked.connect(viewer._reset_simulation)
        self.panel.speed.currentIndexChanged.connect(lambda _: viewer.set_playback_speed(self.panel.speed.currentData()))
        self.panel.trajectory_button.clicked.connect(self.open_controller)
        self.sidebar = QtWidgets.QScrollArea()
        self.sidebar.setWidgetResizable(True)
        self.sidebar.setFrameShape(QtWidgets.QFrame.Shape.NoFrame)
        self.sidebar.setHorizontalScrollBarPolicy(QtCore.Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        self.sidebar.setMinimumWidth(290)
        self.sidebar.setMaximumWidth(330)
        self.sidebar.setWidget(self.panel)
        layout.addWidget(self.sidebar)
        self.setCentralWidget(central)
        hints = QtWidgets.QLabel("Space: Run/Pause     N: Step     T: Reset     R: Camera     I: Info")
        hints.setStyleSheet("padding: 5px 10px; color: #536372; font: 9pt 'Segoe UI';")
        self.statusBar().addWidget(hints)
        self.statusBar().setSizeGripEnabled(False)
        self.statusBar().setStyleSheet("QStatusBar { background: #f4f6f8; border-top: 1px solid #dce3e9; } QStatusBar::item { border: none; }")
        viewer._simulation = simulation
        viewer._control_source = control_source
        viewer._window = self
        if hasattr(simulation, 'set_force_telemetry'):
            simulation.set_force_telemetry(True)
        else:
            for key in ('thrust', 'cable'):
                self.panel.overlays[key].setChecked(False)
                self.panel.overlays[key].setEnabled(False)
        if isinstance(simulation.command_source, FullStateTrajectory):
            viewer.set_trajectory(simulation.command_source)
        viewer.create_plotter(simulation.state, plotter=self.plotter)
        for key, checkbox in self.panel.overlays.items():
            checkbox.toggled.connect(lambda checked, name=key: viewer.set_overlay_visible(name, checked))
        self.panel.force_scale.valueChanged.connect(viewer.set_force_scale)
        self.shortcuts = []
        for key, callback in (("Space", viewer._toggle_simulation), ("N", viewer._step_simulation),
                              ("T", viewer._reset_simulation), ("R", viewer._reset_camera),
                              ("I", viewer._toggle_info), ("B", viewer._toggle_body_axes),
                              ("W", viewer._toggle_world_axes), ("L", viewer._toggle_labels)):
            shortcut = QtGui.QShortcut(QtGui.QKeySequence(key), self)
            shortcut.setAutoRepeat(False)
            shortcut.activated.connect(callback)
            self.shortcuts.append(shortcut)
        viewer._last_tick = perf_counter()
        # Qt timers continue during VTK camera drags. Physics still advances
        # in fixed 10 ms steps; telemetry only reads the latest snapshot.
        self.simulation_timer = QtCore.QTimer(self)
        self.simulation_timer.setTimerType(QtCore.Qt.TimerType.PreciseTimer)
        self.simulation_timer.timeout.connect(viewer._on_timer)
        self.simulation_timer.start(33)
        self.telemetry_timer = QtCore.QTimer(self)
        self.telemetry_timer.setTimerType(QtCore.Qt.TimerType.PreciseTimer)
        self.telemetry_timer.timeout.connect(self.update_telemetry)
        self.telemetry_timer.start(50)
        self.update_telemetry()

    def update_telemetry(self):
        simulation = self.viewer._simulation
        if simulation is not None:
            self.panel.update_state(simulation.state, simulation.running, self.viewer._dynamics_error)

    def attach_controller(self, panel):
        self.controller_panel = panel
        panel.closed.connect(lambda: self._controller_closed(panel))

    def _controller_closed(self, panel):
        if self.controller_panel is panel:
            self.controller_panel = None
        self.update_telemetry()

    def open_controller(self):
        if self.controller_panel is not None:
            self.controller_panel.show()
            self.controller_panel.raise_()
            self.controller_panel.activateWindow()
            return
        import numpy as np
        from controllers.full_state_controller.controller_window import ControllerWindow
        from trajectories.full_state import FullStateTrajectory
        from trajectories.presets import simple_motion
        from simulation.state import quaternion_xyzw_to_rotation_matrix
        sim = self.viewer._simulation
        trajectory = sim.command_source
        name = 'Trajectory'
        if not isinstance(trajectory, FullStateTrajectory):
            initial = sim.configuration.initial_state
            rotation = quaternion_xyzw_to_rotation_matrix(initial.quaternion_body_to_world_xyzw)
            trajectory = simple_motion('Hover', initial.position_world_m,
                                       float(np.arctan2(rotation[1, 0], rotation[0, 0])))
            name = 'Hover'
        panel = ControllerWindow(self.viewer, sim, trajectory,
                                 previous_controller=self.previous_controller, trajectory_name=name)
        self.attach_controller(panel)
        panel.show()

    def closeEvent(self, event):
        self.simulation_timer.stop()
        self.telemetry_timer.stop()
        if self.controller_panel is not None:
            self.controller_panel.close()
        self.plotter.close()
        self.viewer._window = None
        self.viewer._simulation = None
        self.viewer._plotter = None
        self.viewer._overlays = None
        self.closed.emit()
        event.accept()


def show_simulation(viewer, simulation, control_source):
    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
    window = SimulationWindow(viewer, simulation, control_source)
    if viewer._configuration.viewer.start_maximized:
        window.showMaximized()
    else:
        window.show()
    loop = QtCore.QEventLoop()
    window.closed.connect(loop.quit)
    try:
        loop.exec()
    finally:
        window.close()
