"""Separate full-state command preview and playback panel; no physics."""

from pathlib import Path
from PySide6 import QtCore, QtGui, QtWidgets

from trajectories.full_state import FullStateTrajectory, COLUMNS
from project_paths import project_path
from controllers.full_state_controller.bspline_window import BSplineWindow
from controllers.full_state_controller.preset_window import PresetWindow


class TrajectoryPlot(QtWidgets.QWidget):
    def __init__(self):
        super().__init__()
        self.setMinimumSize(400, 250)
        self.values = None
        self.duration = 1.
        self.cursor = 0.
        self.unit = 'm'
        self.cache = None

    def set_trajectory(self, trajectory, component):
        self.values = trajectory.preview(component)
        self.times = trajectory.time_s
        self.duration = trajectory.duration_s
        self.unit = {'Position': 'm', 'Velocity': 'm/s', 'Acceleration': 'm/s²'}[component]
        self.cache = None
        self.update()

    def resizeEvent(self, event):
        self.cache = None
        super().resizeEvent(event)

    def rectangles(self):
        h = (self.height()-24)/3
        return [QtCore.QRectF(65, 8+i*h, self.width()-82, h-22) for i in range(3)]

    def paintEvent(self, event):
        if self.values is None:
            return
        if self.cache is None:
            ratio = self.devicePixelRatioF()
            self.cache = QtGui.QPixmap(round(self.width()*ratio), round(self.height()*ratio))
            self.cache.setDevicePixelRatio(ratio)
            self.cache.fill(self.palette().base().color())
            painter = QtGui.QPainter(self.cache)
            painter.setRenderHint(QtGui.QPainter.RenderHint.Antialiasing)
            for axis, rect in enumerate(self.rectangles()):
                values = self.values[:, axis]
                lo, hi = float(values.min()), float(values.max())
                margin = max((hi-lo)*.1, .01)
                lo -= margin
                hi += margin
                painter.setPen(self.palette().mid().color())
                painter.drawRect(rect)
                painter.setPen(self.palette().text().color())
                painter.drawText(4, int(rect.top()+14), f'{"XYZ"[axis]} ({self.unit})')
                painter.drawText(4, int(rect.top()+31), f'{hi:.3g}')
                painter.drawText(4, int(rect.bottom()), f'{lo:.3g}')
                path = QtGui.QPainterPath()
                for i, value in enumerate(values):
                    point = QtCore.QPointF(rect.left()+rect.width()*self.times[i]/self.duration,
                                          rect.bottom()-rect.height()*(value-lo)/(hi-lo))
                    if i == 0:
                        path.moveTo(point)
                    else:
                        path.lineTo(point)
                painter.setPen(QtGui.QPen(QtGui.QColor(('#397dcc', '#349765', '#cf7650')[axis]), 1.7))
                painter.drawPath(path)
            painter.setPen(self.palette().text().color())
            painter.drawText(65, self.height()-4, '0 s')
            painter.drawText(self.width()-85, self.height()-4, f'{self.duration:g} s')
            painter.end()
        painter = QtGui.QPainter(self)
        painter.drawPixmap(0, 0, self.cache)
        painter.setPen(QtGui.QPen(self.palette().text().color(), 1, QtCore.Qt.PenStyle.DashLine))
        for rect in self.rectangles():
            x = rect.left()+rect.width()*min(self.cursor/self.duration, 1.)
            painter.drawLine(QtCore.QPointF(x, rect.top()), QtCore.QPointF(x, rect.bottom()))
        painter.end()


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
        self.resize(550, 530)
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
        units = QtWidgets.QLabel('World P/V/A: m, m/s, m/s² · Yaw: rad · Desired body rates: rad/s')
        units.setWordWrap(True)
        units.setToolTip('CSV header: '+','.join(COLUMNS)+'\nOriginal timestamps, starting at zero. Position uses the tracking origin. Acceleration excludes gravity.')
        layout.addWidget(units)
        self.error = QtWidgets.QLabel()
        self.error.setStyleSheet('color: #c24444;')
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
        path, _ = QtWidgets.QFileDialog.getOpenFileName(self, 'Load trajectory', '', 'CSV files (*.csv)')
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
        folder = Path(__file__).resolve().parents[2]/'data/trajectories'
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
        self.plot.cursor = elapsed if active else 0.
        self.plot.update()

    def closeEvent(self, event):
        if not self._closing:
            self._closing = True
            self.timer.stop()
            self.simulation.running = False
            self.closed.emit()
        event.accept()
