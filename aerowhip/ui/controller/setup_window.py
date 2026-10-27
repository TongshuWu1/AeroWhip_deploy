"""Choose inputs for the single controller–motor–plant pipeline."""
from pathlib import Path
import numpy as np
from PySide6 import QtWidgets
from aerowhip.physics.configuration import DEFAULT_MODEL, load_configuration
from aerowhip.physics.model_configuration import load_model_configuration
from aerowhip.physics.actuator import Actuator
from aerowhip.physics.state import quaternion_xyzw_to_rotation_matrix
from aerowhip.trajectories.full_state import FullStateTrajectory
from aerowhip.paths import project_path
from aerowhip.ui import style


class SetupWindow(QtWidgets.QDialog):
    def __init__(self, root, scene_path, trajectory_path=None, model_path=DEFAULT_MODEL):
        super().__init__()
        self.root = project_path(root)
        self.model_path = project_path(model_path)
        self.selection = None
        self.setWindowTitle('Full-State Controller')
        style.apply(self)
        self.setMinimumWidth(480)
        layout = QtWidgets.QVBoxLayout(self)
        form = QtWidgets.QFormLayout()
        layout.addLayout(form)
        self.scene = QtWidgets.QComboBox()
        self.trajectory = QtWidgets.QComboBox()
        self.setup = QtWidgets.QLineEdit()
        self.setup.setReadOnly(True)
        for path in sorted((self.root/'configs/scenes').glob('*.toml')):
            self.add_path(self.scene,path)
        self.scene.setCurrentIndex(self.add_path(self.scene,scene_path))
        self.trajectory.addItem('Gentle move',None)
        for path in sorted((self.root/'data/trajectories').glob('*.csv')):
            self.add_path(self.trajectory,path)
        if trajectory_path is not None:
            self.trajectory.setCurrentIndex(self.add_path(self.trajectory,trajectory_path))
        for name,widget,callback in (('Scene',self.scene,self.choose_scene),
                ('Setup',self.setup,self.choose_setup),('Trajectory',self.trajectory,self.choose_trajectory)):
            row = QtWidgets.QHBoxLayout()
            row.addWidget(widget,1)
            button = QtWidgets.QPushButton('Browse…')
            button.clicked.connect(callback)
            row.addWidget(button)
            form.addRow(name,row)
        self.details = QtWidgets.QLabel()
        self.error = QtWidgets.QLabel()
        self.error.setWordWrap(True)
        self.error.setObjectName('error')
        layout.addWidget(self.details)
        layout.addWidget(self.error)
        buttons = QtWidgets.QDialogButtonBox()
        self.open_button = buttons.addButton('Open Simulation',QtWidgets.QDialogButtonBox.ButtonRole.AcceptRole)
        buttons.addButton(QtWidgets.QDialogButtonBox.StandardButton.Cancel)
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        layout.addWidget(buttons)
        self.scene.currentIndexChanged.connect(self.validate_selection)
        self.trajectory.currentIndexChanged.connect(self.validate_selection)
        self.validate_selection()

    @staticmethod
    def add_path(combo,path):
        path = str(project_path(path))
        index = combo.findData(path)
        if index < 0:
            combo.addItem(Path(path).name,path)
            index = combo.count()-1
        return index

    def choose_scene(self):
        path,_ = QtWidgets.QFileDialog.getOpenFileName(self,'Scene',str(self.root/'configs/scenes'),'Scene (*.toml)')
        if path:
            try:
                self.scene.setCurrentIndex(self.add_path(self.scene,path))
            except ValueError as exc:
                self.error.setText(str(exc))

    def choose_trajectory(self):
        path,_ = QtWidgets.QFileDialog.getOpenFileName(self,'Trajectory',str(self.root/'data/trajectories'),'CSV (*.csv)')
        if path:
            try:
                self.trajectory.setCurrentIndex(self.add_path(self.trajectory,path))
            except ValueError as exc:
                self.error.setText(str(exc))

    def choose_setup(self):
        path,_ = QtWidgets.QFileDialog.getOpenFileName(self,'Setup',str(self.root/'configs'),'Configuration (*.json)')
        if path:
            self.load_setup(path)

    def load_setup(self,path):
        try:
            self.model_path = project_path(path)
        except ValueError as exc:
            self.error.setText(str(exc))
            return False
        return self.validate_selection()

    def validate_selection(self,*_):
        self.selection = None
        self.open_button.setEnabled(False)
        self.error.clear()
        self.details.clear()
        self.setup.setText(self.model_path.name)
        self.setup.setToolTip(str(self.model_path))
        try:
            config = load_configuration(self.scene.currentData())
            model = load_model_configuration(self.model_path)
            Actuator(model['actuator'])
            source = self.trajectory.currentData()
            initial = config.initial_state
            R = quaternion_xyzw_to_rotation_matrix(initial.quaternion_body_to_world_xyzw)
            trajectory = (FullStateTrajectory.load(source) if source else
                FullStateTrajectory.gentle_move(initial.position_world_m,float(np.arctan2(R[1,0],R[0,0]))))
            trajectory.validate_initial(config)
            trajectory.validate_controller()
            self.details.setText(f'{trajectory.duration_s:g} s · {trajectory.rate_label}')
        except (OSError,ValueError,TypeError,KeyError) as exc:
            self.error.setText(str(exc))
            return False
        self.selection = config,trajectory,Path(source) if source else None,model
        self.open_button.setEnabled(True)
        return True

    def accept(self):
        if self.validate_selection():
            super().accept()
