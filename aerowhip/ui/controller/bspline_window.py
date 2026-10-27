"""Command-shape editor for the separate full-state controller panel."""

import numpy as np
from PySide6 import QtWidgets

from aerowhip.trajectories.bspline import BSplineCommand
from aerowhip.physics.state import quaternion_xyzw_to_rotation_matrix
from aerowhip.trajectories.full_state import FullStateTrajectory


class BSplineWindow(QtWidgets.QDialog):
    def __init__(self, configuration, parent=None):
        super().__init__(parent)
        self.configuration = configuration
        self.trajectory = None
        self.setWindowTitle('B-spline')
        self.resize(460, 440)
        layout = QtWidgets.QVBoxLayout(self)
        initial = configuration.initial_state
        start = ', '.join(f'{v:g}' for v in initial.position_world_m)
        layout.addWidget(QtWidgets.QLabel(f'Start: ({start}) m'))
        form = QtWidgets.QFormLayout()
        self.duration = self.spin(.01, 60., 3., .1)
        self.speed = self.spin(.01, 100., 2., .1)
        self.acceleration = self.spin(.01, 1000., 6., .1)
        form.addRow('Duration (s)', self.duration)
        form.addRow('Speed limit (m/s)', self.speed)
        form.addRow('Acceleration limit (m/s²)', self.acceleration)
        for field in (self.speed, self.acceleration):
            field.setToolTip('Editable command limit, not a certified aircraft limit. '
                             'Checked with a conservative continuous-time spline bound.')
        layout.addLayout(form)
        label = QtWidgets.QLabel('Control offsets from start (m)')
        label.setToolTip('World XYZ offsets. Interior controls shape the curve; '
                         'the drone does not necessarily pass through them. '
                         'The last row is the exact endpoint. '
                         'Initial velocity comes from the scene; initial command acceleration is zero. '
                         'Final velocity and acceleration are zero; yaw is held at its initial value.')
        layout.addWidget(label)
        self.points = QtWidgets.QTableWidget(3, 3)
        self.points.setHorizontalHeaderLabels(['X', 'Y', 'Z'])
        self.points.horizontalHeader().setSectionResizeMode(QtWidgets.QHeaderView.ResizeMode.Stretch)
        for row, x in enumerate((.08, .17, .25)):
            for axis, value in enumerate((x, 0., 0.)):
                self.points.setItem(row, axis, QtWidgets.QTableWidgetItem(f'{value:g}'))
        self.relabel()
        layout.addWidget(self.points)
        row = QtWidgets.QHBoxLayout()
        self.add_button = QtWidgets.QPushButton('Add point')
        self.remove_button = QtWidgets.QPushButton('Remove point')
        self.add_button.clicked.connect(self.add_point)
        self.remove_button.clicked.connect(self.remove_point)
        row.addWidget(self.add_button)
        row.addWidget(self.remove_button)
        layout.addLayout(row)
        self.error = QtWidgets.QLabel()
        self.error.setWordWrap(True)
        self.error.setObjectName('error')
        layout.addWidget(self.error)
        buttons = QtWidgets.QDialogButtonBox()
        buttons.addButton('Use trajectory', QtWidgets.QDialogButtonBox.ButtonRole.AcceptRole)
        buttons.addButton(QtWidgets.QDialogButtonBox.StandardButton.Cancel)
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        layout.addWidget(buttons)

    @staticmethod
    def spin(minimum, maximum, value, step):
        field = QtWidgets.QDoubleSpinBox()
        field.setRange(minimum, maximum)
        field.setDecimals(2)
        field.setSingleStep(step)
        field.setValue(value)
        return field

    def relabel(self):
        self.points.setVerticalHeaderLabels(
            [f'Control {i+1}' for i in range(self.points.rowCount()-1)]+['End'])

    def add_point(self):
        row = self.points.rowCount()-1
        self.points.insertRow(row)
        for axis in range(3):
            item = self.points.item(row+1, axis)
            self.points.setItem(row, axis, QtWidgets.QTableWidgetItem(item.text() if item else '0'))
        self.relabel()

    def remove_point(self):
        # Keep at least one free shape control and the endpoint.
        if self.points.rowCount() > 2:
            row = self.points.currentRow()
            if row < 0 or row == self.points.rowCount()-1:
                row = self.points.rowCount()-2
            self.points.removeRow(row)
            self.relabel()

    def accept(self):
        try:
            offsets = np.array([[float(self.points.item(row, axis).text())
                                 for axis in range(3)] for row in range(self.points.rowCount())])
            initial = self.configuration.initial_state
            rotation = quaternion_xyzw_to_rotation_matrix(initial.quaternion_body_to_world_xyzw)
            yaw = float(np.arctan2(rotation[1, 0], rotation[0, 0]))
            command = BSplineCommand(
                initial.position_world_m, initial.position_world_m+offsets[-1],
                initial.position_world_m+offsets[:-1], self.duration.value(),
                start_velocity=initial.linear_velocity_world_mps, yaw=yaw)
            candidate = FullStateTrajectory(command.samples(
                FullStateTrajectory.timestep_s, self.speed.value(), self.acceleration.value()))
            candidate.validate_initial(self.configuration)
        except (AttributeError, TypeError, ValueError) as exc:
            self.error.setText(str(exc) or 'Enter numeric XYZ offsets.')
            return
        self.trajectory = candidate
        self.error.clear()
        super().accept()
