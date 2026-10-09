"""Display-only saved paths and independent toggles; no physics or inference."""
import numpy as np
from PySide6 import QtCore, QtWidgets
from viewer.overlays import COLORS


class SavedTrajectoryControls(QtWidgets.QGroupBox):
    def __init__(self, plotter, arrays, parent=None, *, pv=None):
        super().__init__('Trajectories · entire saved replay', parent)
        if pv is None:
            import pyvista as pv
        self.plotter = plotter
        self.paths = {
            'drone': np.asarray(arrays['drone_positions'], dtype=float).copy(),
            'tip': np.asarray(arrays['cable_positions'], dtype=float)[:, -1].copy(),
        }
        labels = {'drone': 'Drone trajectory', 'tip': 'Cable-tip trajectory'}
        colors = {'drone': COLORS['trail'], 'tip': COLORS['tip_trail']}
        command = arrays.get('command')
        # Velocity/CTBR commands are NOT position trajectories. Never integrate
        # them or label a desired velocity as an actual drone/tip path.
        if command is not None and np.asarray(command).ndim == 2 and np.asarray(command).shape[1] == 13:
            self.paths['reference'] = np.asarray(command, dtype=float)[:, :3].copy()
            labels['reference'] = 'Requested position path'
            colors['reference'] = COLORS['reference']
        layout = QtWidgets.QHBoxLayout(self)
        self.actors, self.checkboxes = {}, {}
        for key, points in self.paths.items():
            if points.ndim != 2 or points.shape[1] != 3 or len(points) < 2 or not np.isfinite(points).all():
                raise ValueError('Trajectory requires at least two finite saved XYZ points')
            actor = plotter.add_mesh(pv.lines_from_points(points), name='saved_path_' + key,
                color=colors[key], line_width=3, lighting=False, pickable=False, reset_camera=False)
            self.actors[key] = actor
            checkbox = QtWidgets.QCheckBox(labels[key], self)
            checkbox.setStyleSheet('color: ' + colors[key] + ';')
            checkbox.setToolTip('Only show/hide this entire saved path; playback and training are unchanged.')
            checkbox.setChecked(True)
            checkbox.toggled.connect(lambda visible, name=key: self.set_visible(name, visible))
            self.checkboxes[key] = checkbox
            layout.addWidget(checkbox)
        layout.addStretch()
        self.setToolTip('Actual sampled positions, world XYZ in metres. Cable tip is the final cable node. Full saved paths are visual context, not policy inputs.')

    def set_visible(self, name, visible):
        self.actors[name].SetVisibility(bool(visible))
        self.plotter.render()


def attach_saved_trajectories(window, viewer):
    if hasattr(window, 'trajectory_controls'):
        return window.trajectory_controls
    controls = SavedTrajectoryControls(window.plotter, window.replay.arrays, window.centralWidget())
    window.centralWidget().layout().addWidget(controls)
    window.trajectory_controls = controls
    # Avoid a second partial trail remaining visible when the full path is off.
    for key in ('trail', 'tip_trail'):
        viewer._overlays.visible[key] = False
        viewer._overlays.actors[key].SetVisibility(False)
    window.plotter.render()
    return controls


def schedule_saved_trajectories(plotter, viewer):
    if not hasattr(plotter, 'parentWidget'):
        return  # Non-Qt/off-screen plotters have no replay-window controls.
    owner = plotter.parentWidget()
    while owner is not None and not hasattr(owner, 'replay'):
        owner = owner.parentWidget()
    if owner is None or not hasattr(owner.replay, 'arrays'):
        return
    # ReplayWindow subclasses expect playback controls at layout index1. Wait
    # until the whole constructor completes, then append this display group.
    def install():
        from shiboken6 import isValid
        if isValid(owner):
            attach_saved_trajectories(owner, viewer)
    QtCore.QTimer.singleShot(0, install)
