"""Simple motion selection in the separate controller, not the physics viewer."""

import numpy as np
from PySide6 import QtWidgets

from aerowhip.physics.state import quaternion_xyzw_to_rotation_matrix
from aerowhip.trajectories.presets import MOTIONS, simple_motion


class PresetWindow(QtWidgets.QDialog):
    def __init__(self, configuration, parent=None):
        super().__init__(parent)
        self.configuration = configuration
        self.trajectory = None
        self.setWindowTitle('Motion')
        self.setMinimumWidth(330)
        layout = QtWidgets.QVBoxLayout(self)
        form = QtWidgets.QFormLayout()
        layout.addLayout(form)
        self.motion = QtWidgets.QComboBox()
        self.motion.addItems(MOTIONS)
        self.motion.setCurrentText('Move')
        form.addRow('Motion', self.motion)
        self.displacement = []
        for axis, value in zip('XYZ', (.25, 0., 0.)):
            field = QtWidgets.QDoubleSpinBox()
            field.setRange(-5., 5.)
            field.setDecimals(3)
            field.setSingleStep(.05)
            field.setSuffix(' m')
            field.setValue(value)
            field.setToolTip('World-frame displacement from the initial tracking position; not a physical capability limit.')
            form.addRow(f'{axis} displacement', field)
            self.displacement.append(field)
        self.duration = QtWidgets.QDoubleSpinBox()
        self.duration.setRange(.2, 120.)
        self.duration.setDecimals(2)
        self.duration.setSingleStep(.5)
        self.duration.setSuffix(' s')
        self.duration.setValue(4.)
        self.duration.setToolTip('Total motion time. Back and forth uses half for each leg.')
        form.addRow('Duration', self.duration)
        self.error = QtWidgets.QLabel()
        self.error.setWordWrap(True)
        self.error.setObjectName('error')
        layout.addWidget(self.error)
        buttons = QtWidgets.QDialogButtonBox(QtWidgets.QDialogButtonBox.StandardButton.Cancel)
        buttons.addButton('Apply', QtWidgets.QDialogButtonBox.ButtonRole.AcceptRole)
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        layout.addWidget(buttons)
        self.motion.currentTextChanged.connect(self.update_fields)
        self.update_fields()

    def update_fields(self, *_):
        for field in self.displacement:
            field.setEnabled(self.motion.currentText() != 'Hover')

    def accept(self):
        initial = self.configuration.initial_state
        rotation = quaternion_xyzw_to_rotation_matrix(initial.quaternion_body_to_world_xyzw)
        try:
            candidate = simple_motion(self.motion.currentText(), initial.position_world_m,
                float(np.arctan2(rotation[1, 0], rotation[0, 0])),
                displacement=[field.value() for field in self.displacement],
                duration_s=self.duration.value())
            candidate.validate_initial(self.configuration)
        except (ValueError, TypeError) as exc:
            self.error.setText(str(exc))
            return
        self.trajectory = candidate
        super().accept()
