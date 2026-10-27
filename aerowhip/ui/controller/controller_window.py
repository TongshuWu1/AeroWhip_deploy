"""Separate full-state command preview and playback panel; no physics."""

from pathlib import Path
from PySide6 import QtCore, QtGui, QtWidgets

from aerowhip.trajectories.full_state import FullStateTrajectory, COLUMNS
from aerowhip.paths import ROOT, project_path
from aerowhip.ui.controller.bspline_window import BSplineWindow
from aerowhip.ui.controller.preset_window import PresetWindow
from aerowhip.ui import style
from aerowhip.ui.plotting import PlotCanvas
from aerowhip.ui.style import AXIS


class TrajectoryPlot(PlotCanvas):
    """Reference position, velocity or acceleration per world axis, with the playback time."""
    def __init__(self):
        super().__init__(dark=True, minimum=(400, 250))
        self.cursor = 0.
        self._shown = None

    def set_trajectory(self, trajectory, component):
        values = trajectory.preview(component)
        unit = {'Position': 'm', 'Velocity': 'm/s', 'Acceleration': 'm/s²'}[component]
        with self.style():
            self.reset()
            axes = self.figure.subplots(3, 1, sharex=True)
            for axis, ax in enumerate(axes):
                ax.plot(trajectory.time_s, values[:, axis], color=AXIS[axis])
                ax.set_ylabel(f"{'xyz'[axis]} ({unit})")
                self.add_cursor(ax, 0.)
            axes[0].set_title(f'Reference {component.lower()} (world, tracking origin)')
            axes[-1].set_xlabel('time (s)')
            axes[-1].set_xlim(0, trajectory.duration_s)
        self._shown = None
        self.draw_idle()

    def show_cursor(self, time_s):
        if time_s != self._shown:
            self._shown = time_s
            self.set_cursor(time_s)


class ControllerWindow(QtWidgets.QWidget):
    closed = QtCore.Signal()

    def __init__(self, viewer, simulation, trajectory, source=None, *,
                 previous_controller=None, trajectory_name=None):
        super().__init__()
        self.viewer, self.simulation, self.trajectory = viewer, simulation, trajectory
        self._closing = False
        self.spline_window = None
        self.preset_window = None
        self.previous_controller = previous_controller
        self.setWindowTitle('Full-State Controller')
        self.resize(600, 620)
        self.setObjectName('page')
        self.setAttribute(QtCore.Qt.WidgetAttribute.WA_StyledBackground, True)
        style.apply(self, dark=True)
        layout = QtWidgets.QVBoxLayout(self)
        row = QtWidgets.QHBoxLayout()
        self.preset_button = QtWidgets.QPushButton('Presets…')
        self.preset_button.clicked.connect(self.choose_preset)
        row.addWidget(self.preset_button)
        self.load_button = QtWidgets.QPushButton('Load CSV')
        self.load_button.clicked.connect(self.choose_file)
        self.filename = QtWidgets.QLabel()
        self.filename.setWordWrap(True)
        row.addWidget(self.load_button)
        self.spline_button = QtWidgets.QPushButton('B-spline')
        self.spline_button.clicked.connect(self.edit_spline)
        row.addWidget(self.spline_button)
        self.export_button = QtWidgets.QPushButton('Export CSV')
        self.export_button.clicked.connect(self.export_csv)
        row.addWidget(self.export_button)
        row.addWidget(self.filename, 1)
        layout.addLayout(row)
        self.details = QtWidgets.QLabel()
        layout.addWidget(self.details)
        self.component = QtWidgets.QComboBox()
        self.component.addItems(['Position', 'Velocity', 'Acceleration'])
        self.component.currentTextChanged.connect(self.update_preview)
        layout.addWidget(self.component)
        self.plot = TrajectoryPlot()
        layout.addWidget(self.plot, 1)
        units = QtWidgets.QLabel('World P/V/A: m, m/s, m/s² · Yaw: rad · Desired body rates: rad/s', objectName='note')
        units.setWordWrap(True)
        units.setToolTip('CSV header: '+','.join(COLUMNS)+'\nOriginal timestamps, starting at zero. Position uses the tracking origin. Acceleration excludes gravity.')
        layout.addWidget(units)
        self.error = QtWidgets.QLabel()
        self.error.setObjectName('error')
        self.error.setWordWrap(True)
        layout.addWidget(self.error)
        self.status = QtWidgets.QLabel()
        layout.addWidget(self.status)
        self.progress = QtWidgets.QProgressBar()
        self.progress.setRange(0, trajectory.sample_count-1)
        layout.addWidget(self.progress)
        buttons = QtWidgets.QHBoxLayout()
        self.apply_button = QtWidgets.QPushButton('Apply')
        self.apply_button.setToolTip('Apply this reference and reset the simulation to its initial state.')
        self.apply_button.clicked.connect(self.apply_preview)
        self.run_button = QtWidgets.QPushButton('Run')
        self.reset_button = QtWidgets.QPushButton('Reset')
        self.run_button.clicked.connect(self.toggle)
        self.reset_button.clicked.connect(self.reset)
        buttons.addWidget(self.apply_button)
        buttons.addWidget(self.run_button)
        buttons.addWidget(self.reset_button)
        self.manual_button = QtWidgets.QPushButton('Manual')
        self.manual_button.setToolTip('Return to the previous manual controller and reset the simulation.')
        self.manual_button.setVisible(previous_controller is not None)
        self.manual_button.clicked.connect(self.use_previous_controller)
        buttons.addWidget(self.manual_button)
        layout.addLayout(buttons)
        self.describe(source, name=trajectory_name)
        if simulation.command_source is trajectory:
            self.viewer.set_trajectory(trajectory)
        self.update_preview()
        self.timer = QtCore.QTimer(self)
        self.timer.timeout.connect(self.refresh)
        self.timer.start(50)
        self.refresh()

    def describe(self, source, generated=False, name=None):
        self.source, self.generated, self.trajectory_name = source, generated, name
        self.filename.setText(name or ('B-spline' if generated else Path(source).name if source else 'Gentle move'))
        self.filename.setToolTip('Generated command trajectory' if generated or name else
                                 str(project_path(source)) if source else 'Built-in 0.25 m X movement')
        self.details.setText(f'{self.trajectory.duration_s:g} s · {self.trajectory.sample_count} samples · {self.trajectory.rate_label}')
        self.details.setToolTip('Analytic preset rate feedforward' if self.trajectory.jerk_world is not None
            else 'Supplied CSV body-rate feedforward' if self.trajectory.has_body_rates
            else 'PVA reference; no body-rate feedforward')

    def update_preview(self, *_):
        self.plot.set_trajectory(self.trajectory, self.component.currentText())

    def choose_file(self):
        folder = ROOT/'data/trajectories'
        path, _ = QtWidgets.QFileDialog.getOpenFileName(self, 'Load trajectory', str(folder), 'CSV files (*.csv)')
        if path:
            self.load_path(path)

    def load_path(self, path):
        if self.simulation.running:
            self.error.setText('Pause before loading a trajectory.')
            return False
        try:
            candidate = FullStateTrajectory.load(path)
            candidate.validate_initial(self.simulation.configuration)
            candidate.validate_controller()
        except (OSError, ValueError, TypeError) as exc:
            self.error.setText(str(exc))
            return False
        self.use_trajectory(candidate, path)
        return True

    def use_trajectory(self, candidate, source=None, generated=False, name=None):
        # No playback state changes until all input checks succeed.
        if self.simulation.running:
            raise ValueError('Pause before applying a trajectory.')
        candidate.validate_initial(self.simulation.configuration)
        candidate.validate_controller()
        self.trajectory = candidate
        self.simulation.command_source = candidate
        self.simulation.command_reset = None
        self.viewer._reset_simulation()
        self.viewer.set_trajectory(candidate)
        self.progress.setRange(0, candidate.sample_count-1)
        self.describe(source, generated, name)
        self.error.clear()
        self.update_preview()
        self.refresh()

    def apply_preview(self):
        self.use_trajectory(self.trajectory, self.source, self.generated, self.trajectory_name)

    def choose_preset(self):
        if self.simulation.running:
            self.error.setText('Pause before changing a trajectory.')
            return
        if self.preset_window is None:
            self.preset_window = PresetWindow(self.simulation.configuration, self)
        if self.preset_window.exec() == QtWidgets.QDialog.DialogCode.Accepted:
            if self.simulation.running:
                self.error.setText('Pause before changing a trajectory.')
                return
            self.use_trajectory(self.preset_window.trajectory, name=self.preset_window.motion.currentText())

    def use_previous_controller(self):
        if self.previous_controller is None or self.simulation.running:
            return
        self.simulation.command_source, self.simulation.command_reset = self.previous_controller
        self.viewer._reset_simulation()
        self.viewer.set_trajectory(None)
        self.refresh()

    def edit_spline(self):
        if self.simulation.running:
            self.error.setText('Pause before changing a trajectory.')
            return
        if self.spline_window is None:
            self.spline_window = BSplineWindow(self.simulation.configuration, self)
        if self.spline_window.exec() == QtWidgets.QDialog.DialogCode.Accepted:
            # Recheck playback state before replacing its command source.
            if self.simulation.running:
                self.error.setText('Pause before changing a trajectory.')
                return
            self.use_trajectory(self.spline_window.trajectory, generated=True)

    def export_csv(self):
        folder = ROOT/'data/trajectories'
        dialog = QtWidgets.QFileDialog(self, 'Export trajectory', str(folder/'trajectory.csv'),
                                       'CSV files (*.csv)')
        dialog.setAcceptMode(QtWidgets.QFileDialog.AcceptMode.AcceptSave)
        dialog.setDefaultSuffix('csv')
        if dialog.exec() == QtWidgets.QDialog.DialogCode.Accepted:
            path = dialog.selectedFiles()[0]
            try:
                dynamics = self.simulation.dynamics
                transform = dynamics.controller.prepare_reference
                self.trajectory.save(path, transform=transform)
            except (OSError, ValueError) as exc:
                self.error.setText(str(exc))
                return
            self.error.clear()

    def toggle(self):
        if self.simulation.command_source is not self.trajectory:
            return
        self.viewer._toggle_simulation()
        self.refresh()

    def reset(self):
        self.viewer._reset_simulation()
        self.refresh()

    def refresh(self):
        sim = self.simulation
        elapsed = sim.step_count*sim.timestep_s
        fault = self.viewer._dynamics_error
        active = sim.command_source is self.trajectory
        state = ('Error' if fault else 'Holding' if sim.running and elapsed >= self.trajectory.duration_s
                 else 'Running' if sim.running else 'Ready' if sim.step_count == 0 else 'Paused')
        self.status.setText(f'{state} · Simulation {elapsed:.2f} s · Trajectory {min(elapsed, self.trajectory.duration_s):.2f} / {self.trajectory.duration_s:g} s')
        if not active:
            self.status.setText('Preview · Apply to reset and use this trajectory')
        self.status.setToolTip(fault or '')
        self.run_button.setText('Pause' if sim.running else 'Run')
        self.run_button.setEnabled(fault is None and active)
        self.apply_button.setEnabled(not sim.running and not active)
        self.preset_button.setEnabled(not sim.running)
        self.manual_button.setEnabled(not sim.running and active)
        self.load_button.setEnabled(not sim.running)
        self.spline_button.setEnabled(not sim.running)
        self.progress.setValue(self.trajectory.index_at(elapsed if active else 0.))
        self.plot.show_cursor(elapsed if active else 0.)

    def closeEvent(self, event):
        if not self._closing:
            self._closing = True
            self.timer.stop()
            self.simulation.running = False
            self.closed.emit()
        event.accept()
