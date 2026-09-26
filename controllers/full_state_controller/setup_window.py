"""Select a scene and trajectory before creating the simulation."""

from pathlib import Path

import numpy as np
from PySide6 import QtWidgets

from simulation.configuration import load_configuration
from simulation.state import quaternion_xyzw_to_rotation_matrix
from trajectories.full_state import FullStateTrajectory


class SetupWindow(QtWidgets.QDialog):
    def __init__(self, root, scene_path, trajectory_path=None):
        super().__init__()
        self.root = Path(root)
        self.selection = None
        self.setWindowTitle('Full-State Controller')
        self.setMinimumWidth(480)
        layout = QtWidgets.QVBoxLayout(self)
        form = QtWidgets.QFormLayout()
        layout.addLayout(form)
        self.scene = QtWidgets.QComboBox()
        self.trajectory = QtWidgets.QComboBox()
        for path in sorted((self.root/'configs/scenes').glob('*.toml')):
            self.add_path(self.scene, path)
        self.scene.setCurrentIndex(self.add_path(self.scene, scene_path))
        self.trajectory.addItem('Gentle move', None)
        for path in sorted((self.root/'data/trajectories').glob('*.csv')):
            self.add_path(self.trajectory, path)
        if trajectory_path is not None:
            self.trajectory.setCurrentIndex(self.add_path(self.trajectory, trajectory_path))
        for label, combo, callback in (
            ('Scene', self.scene, self.choose_scene),
            ('Trajectory', self.trajectory, self.choose_trajectory),
        ):
            row = QtWidgets.QHBoxLayout()
            row.addWidget(combo, 1)
            button = QtWidgets.QPushButton('Browse…')
            button.clicked.connect(callback)
            row.addWidget(button)
            form.addRow(label, row)
        self.details = QtWidgets.QLabel()
        self.details.setWordWrap(True)
        layout.addWidget(self.details)
        self.error = QtWidgets.QLabel()
        self.error.setWordWrap(True)
        self.error.setStyleSheet('color: #c24444;')
        layout.addWidget(self.error)
        buttons = QtWidgets.QDialogButtonBox()
        self.open_button = buttons.addButton('Open Simulation', QtWidgets.QDialogButtonBox.ButtonRole.AcceptRole)
        buttons.addButton(QtWidgets.QDialogButtonBox.StandardButton.Cancel)
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        layout.addWidget(buttons)
        self.scene.currentIndexChanged.connect(self.validate_selection)
        self.trajectory.currentIndexChanged.connect(self.validate_selection)
        self.validate_selection()

    @staticmethod
    def add_path(combo, path):
        path = str(Path(path).resolve())
        index = combo.findData(path)
        if index < 0:
            combo.addItem(Path(path).name, path)
            index = combo.count()-1
        return index

    def choose_scene(self):
        path, _ = QtWidgets.QFileDialog.getOpenFileName(
            self, 'Select scene', self.scene.currentData() or str(self.root/'configs/scenes'),
            'Scene files (*.toml)')
        if path:
            self.scene.setCurrentIndex(self.add_path(self.scene, path))
            self.validate_selection()

    def choose_trajectory(self):
        path, _ = QtWidgets.QFileDialog.getOpenFileName(
            self, 'Select trajectory', self.trajectory.currentData() or str(self.root/'data/trajectories'),
            'CSV files (*.csv)')
        if path:
            self.trajectory.setCurrentIndex(self.add_path(self.trajectory, path))
            self.validate_selection()

    def validate_selection(self, *_):
        self.selection = None
        self.open_button.setEnabled(False)
        self.details.clear()
        self.error.clear()
        scene_path = self.scene.currentData()
        source = self.trajectory.currentData()
        self.scene.setToolTip(scene_path or '')
        self.trajectory.setToolTip(source or 'Built-in 0.25 m X movement')
        try:
            if scene_path is None:
                raise ValueError('Select a scene.')
            config = load_configuration(scene_path)
            initial = config.initial_state
            if source is None:
                rotation = quaternion_xyzw_to_rotation_matrix(initial.quaternion_body_to_world_xyzw)
                yaw = float(np.arctan2(rotation[1, 0], rotation[0, 0]))
                trajectory = FullStateTrajectory.gentle_move(initial.position_world_m, yaw)
            else:
                trajectory = FullStateTrajectory.load(source)
            position = ', '.join(f'{value:g}' for value in initial.position_world_m)
            self.details.setText(f'Start: ({position}) m · {trajectory.duration_s:g} s · {trajectory.rate_label}')
            trajectory.validate_initial(config)
        except (OSError, ValueError, TypeError, KeyError) as exc:
            self.error.setText(str(exc))
            return False
        self.selection = (config, trajectory, Path(source) if source else None)
        self.open_button.setEnabled(True)
        return True

    def accept(self):
        # Re-read files in case they changed after selection.
        if self.validate_selection():
            super().accept()
