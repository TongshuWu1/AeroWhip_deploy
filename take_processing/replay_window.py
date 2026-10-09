"""Standalone playback; no simulation, fitting, or recording writes."""
from pathlib import Path
from time import perf_counter
from types import SimpleNamespace
import numpy as np
import pyvista as pv
from pyvistaqt import QtInteractor
from PySide6.QtCore import Qt, QTimer, QThread, Signal
from PySide6.QtWidgets import (QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QPushButton, QComboBox, QSlider, QLabel, QCheckBox, QFileDialog)
from simulation.configuration import load_configuration
from simulation.state import DroneState, quaternion_xyzw_to_rotation_matrix
from viewer.drone_model import add_drone_model
from viewer.environment import add_environment, configure_lighting
from take_processing.take_data import discover, load_take
from project_paths import project_path

ROOT = Path(__file__).resolve().parents[1]
COLORS = {"Desired": "#2563eb", "Command": "#e88724", "Measured": "#15803d", "Predicted": "#ad70dc"}


def segments(points, times=None, dashed=False):
    points = np.asarray(points, dtype=float)
    ok = np.isfinite(points).all(axis=1)
    edges = ok[:-1] & ok[1:]
    if times is not None and len(times) > 1:
        edges &= np.diff(times) <= 1.5 * np.median(np.diff(times))
    ids = np.flatnonzero(edges)
    if dashed:
        ids = ids[ids % 6 < 3]
    mesh = pv.PolyData(np.nan_to_num(points))
    mesh.verts = np.empty(0, dtype=int)
    if len(ids):
        mesh.lines = np.column_stack([np.full(len(ids), 2), ids, ids+1]).ravel()
    return mesh


class Loader(QThread):
    loaded = Signal(object)
    failed = Signal(str)

    def __init__(self, path, verify=None):
        super().__init__()
        self.path = path
        self.verify = verify

    def run(self):
        try:
            if self.verify is not None:
                self.verify()
            take = load_take(self.path)
            if self.verify is not None:
                self.verify()
            self.loaded.emit(take)
        except Exception as exc:
            self.failed.emit(str(exc))


class ReplayWindow(QMainWindow):
    take_loaded = Signal(object)
    take_failed = Signal(str)
    frame_changed = Signal(float)

    def __init__(self, folder=None):
        super().__init__()
        self.setWindowTitle("Take Replay")
        scale = self.devicePixelRatioF()
        self.resize(round(1440/scale), round(900/scale))
        self.data = None
        self.worker = None
        self.playing = False
        self.elapsed = 0.0
        self.layers = {}
        self.prediction = None
        self.setStyleSheet("QWidget {font-size: 13px;} QPushButton {padding: 5px 12px;}")
        body = QWidget()
        self.setCentralWidget(body)
        layout = QVBoxLayout(body)
        top = QHBoxLayout()
        layout.addLayout(top)
        self.open_button = QPushButton("Open folder")
        self.open_button.clicked.connect(self.open_folder)
        top.addWidget(self.open_button)
        self.takes = QComboBox()
        top.addWidget(self.takes, 1)
        self.plotter = QtInteractor(body, auto_update=False, multi_samples=4)
        layout.addWidget(self.plotter.interactor, 1)
        options = QHBoxLayout()
        layout.addLayout(options)
        self.checks = {}
        for name, color in COLORS.items():
            check = QCheckBox(name)
            check.setChecked(True)
            check.setStyleSheet(f"color: {color}; font-weight: 600;")
            check.toggled.connect(self.update_visibility)
            options.addWidget(check)
            self.checks[name] = check
        options.addStretch()
        reset = QPushButton("Reset camera")
        reset.clicked.connect(self.reset_camera)
        options.addWidget(reset)
        controls = QHBoxLayout()
        layout.addLayout(controls)
        self.play = QPushButton("Play")
        self.play.clicked.connect(self.toggle)
        controls.addWidget(self.play)
        self.timeline = QSlider(Qt.Orientation.Horizontal)
        self.timeline.valueChanged.connect(self.seek)
        controls.addWidget(self.timeline, 1)
        self.time_label = QLabel("0.00 s")
        self.time_label.setMinimumWidth(160)
        controls.addWidget(self.time_label)
        self.speed = QComboBox()
        for speed in (.25, .5, 1., 2.):
            self.speed.addItem(f"{speed:g}×", speed)
        self.speed.setCurrentIndex(2)
        controls.addWidget(self.speed)
        self.timer = QTimer(self)
        self.timer.setInterval(33)
        self.timer.timeout.connect(self.tick)
        self.takes.currentIndexChanged.connect(self.select_take)
        if folder is not None:
            self.populate(folder)
        else:
            self.play.setEnabled(False)
            self.timeline.setEnabled(False)

    def populate(self, folder):
        self.folder = project_path(folder)
        self.takes.blockSignals(True)
        self.takes.clear()
        for path in discover(self.folder):
            self.takes.addItem(str(path.relative_to(self.folder)), path)
        self.takes.blockSignals(False)
        self.select_take()

    def open_folder(self):
        folder = QFileDialog.getExistingDirectory(self, "Take folder", str(self.folder))
        if folder:
            self.populate(folder)

    def pause(self):
        self.playing = False
        self.timer.stop()
        self.play.setText("Play")

    def select_take(self):
        self.load_path(self.takes.currentData())

    def load_path(self, path, verify=None):
        if self.worker is not None:
            return
        self.pause()
        self.data = None
        self.prediction = None
        self.checks['Predicted'].setEnabled(False)
        self.play.setEnabled(False)
        self.timeline.setEnabled(False)
        self.plotter.clear()
        self.layers = {}
        self.plotter.render()
        if path is None:
            self.statusBar().showMessage("No takes found")
            return
        self.takes.setEnabled(False)
        self.open_button.setEnabled(False)
        self.statusBar().showMessage("Loading…")
        self.worker = Loader(path, verify)
        self.worker.loaded.connect(self.loaded)
        self.worker.failed.connect(self.load_error)
        self.worker.finished.connect(self.load_finished)
        self.worker.start()

    def load_error(self, message):
        self.statusBar().showMessage(message)
        self.take_failed.emit(message)

    def load_finished(self):
        self.worker.deleteLater()
        self.worker = None
        self.takes.setEnabled(True)
        self.open_button.setEnabled(True)

    def loaded(self, data):
        try:
            self.build_scene(data)
            self.take_loaded.emit(data)
        except Exception as exc:
            self.data = None
            self.play.setEnabled(False)
            self.timeline.setEnabled(False)
            self.statusBar().showMessage(str(exc))
            self.take_failed.emit(str(exc))

    def build_scene(self, data):
        self.data = data
        self.elapsed = 0.0
        config = load_configuration(ROOT / "configs/scenes/default.toml")
        configure_lighting(self.plotter, pv, config.viewer)
        add_environment(self.plotter, pv, config.viewer)
        self.plotter.add_axes()
        self.layers = {}
        for name, positions, times in (("Desired", data.desired, data.desired_time),
                ("Command", data.command, data.command_time), ("Measured", data.position, data.time)):
            self.layers[name] = []
            self.checks[name].setEnabled(positions is not None)
            descriptions = {
                "Desired": "Frozen desired tracked-origin trajectory; not predicted execution",
                "Command": "Logged position setpoints; current marker holds the latest received command",
                "Measured": "OptiTrack pose and cable markers; CAD and connecting lines are visual guides",
            }
            self.checks[name].setToolTip(descriptions[name] if positions is not None else "Unavailable for this take")
            if positions is not None:
                mesh = segments(positions, times, dashed=name == "Desired")
                self.layers[name].append(self.plotter.add_mesh(mesh, color=COLORS[name], line_width=3,
                    lighting=False, render=False))
                dot = self.plotter.add_mesh(pv.Sphere(radius=.012), color=COLORS[name], render=False)
                self.layers[name].append(dot)
        # Build CAD at the measured top-plate origin, without running physics.
        self.attachment = np.array([0., 0., -.051])
        drone = DroneState(np.zeros(3), np.array([0., 0., 0., 1.]), np.zeros(3), np.zeros(3))
        actors, rotors = add_drone_model(self.plotter, pv,
            SimpleNamespace(drone=drone, attachment_position_world_m=self.attachment), config.viewer)
        self.drone_actors = actors + [a for rotor in rotors for a in rotor.actors]
        self.layers["Measured"].extend(self.drone_actors)
        self.marker_mesh = pv.PolyData(np.zeros((10, 3)))
        self.marker_actor = self.plotter.add_mesh(self.marker_mesh, color=COLORS["Measured"],
            point_size=8, render_points_as_spheres=True, render=False)
        self.cable_mesh = segments(np.zeros((11, 3)))
        self.cable_actor = self.plotter.add_mesh(self.cable_mesh, color=COLORS["Measured"], line_width=3, render=False)
        self.layers["Measured"].extend([self.marker_actor, self.cable_actor])
        self.timeline.blockSignals(True)
        self.timeline.setRange(0, len(data.time)-1)
        self.timeline.setValue(0)
        self.timeline.blockSignals(False)
        self.play.setEnabled(True)
        self.timeline.setEnabled(True)
        self.statusBar().showMessage(data.status)
        self.reset_camera()
        self.draw_frame(0)

    def reset_camera(self):
        if self.data is None:
            return
        d = self.data
        points = [d.position, d.cable.reshape(-1, 3)]
        points += [p for p in (d.command, d.desired) if p is not None]
        if self.prediction is not None:
            points.extend([self.prediction.position,self.prediction.markers.reshape(-1,3)])
        points = np.concatenate(points)
        points = points[np.isfinite(points).all(axis=1)]
        if not len(points):
            return
        low, high = points.min(0)-.15, points.max(0)+.15
        center = (low+high)/2
        distance = max(np.linalg.norm(high-low), 1.)
        self.plotter.camera_position = [center + distance*np.array([.8, -1.2, .65]), center, (0, 0, 1)]
        self.plotter.reset_camera_clipping_range()
        self.plotter.render()

    def update_visibility(self):
        if self.data is not None:
            self.draw_frame(self.timeline.value())

    def set_prediction(self, prediction):
        for actor in self.layers.pop('Predicted', []):
            self.plotter.remove_actor(actor, render=False)
        self.prediction = prediction
        self.checks['Predicted'].setEnabled(prediction is not None)
        prescribed = prediction is not None and prediction.metadata.get('mode') == 'prescribed_cable'
        self.checks['Predicted'].setText('Predicted cable' if prescribed else 'Predicted')
        self.checks['Predicted'].setToolTip(
            'Saved cable prediction driven by measured attachment motion; drone pose is prescribed, not predicted'
            if prescribed else 'Saved coupled-model baseline; shown only over its simulated interval')
        if prediction is not None:
            color=COLORS['Predicted']
            path=self.plotter.add_mesh(segments(prediction.position,prediction.time),color=color,line_width=3,render=False)
            # A simple translucent body outline makes predicted attitude distinct from measured CAD.
            self.predicted_body=self.plotter.add_mesh(pv.Box(bounds=(-.06,.06,-.06,.06,-.036,0)),
                color=color,style='wireframe',line_width=2,opacity=.65,render=False)
            self.predicted_cable=segments(np.zeros((11,3)))
            cable=self.plotter.add_mesh(self.predicted_cable,color=color,line_width=3,opacity=.7,render=False)
            self.predicted_markers=pv.PolyData(np.zeros((10,3)))
            markers=self.plotter.add_mesh(self.predicted_markers,color=color,point_size=7,render_points_as_spheres=True,render=False)
            self.layers['Predicted']=[path,self.predicted_body,cable,markers]
        if self.data is not None:
            self.draw_frame(self.timeline.value())

    def draw_frame(self, index):
        d = self.data
        if d is None:
            return
        for name, actors in self.layers.items():
            for actor in actors:
                actor.SetVisibility(self.checks[name].isChecked())
        now = self.elapsed
        for name, times, positions in (("Desired", d.desired_time, d.desired),
                ("Command", d.command_time, d.command), ("Measured", d.time, d.position)):
            if positions is None:
                continue
            j = np.searchsorted(times, now, side="right")-1
            visible = (0 <= j < len(times) and now <= times[-1] and
                       np.isfinite(positions[max(0, j)]).all())
            if visible:
                limit = .2 if name == "Command" else 1.5*np.median(np.diff(times))
                visible = now-times[j] <= limit
            actor = self.layers[name][1]
            actor.SetVisibility(bool(visible and self.checks[name].isChecked()))
            if visible:
                actor.SetPosition(*positions[j])
        p, q = d.position[index], d.quaternion[index]
        fresh = now-d.time[index] <= 1.5*np.median(np.diff(d.time))
        pose_ok = fresh and np.isfinite(p).all() and np.isfinite(q).all() and np.linalg.norm(q)>1e-12
        transform = np.eye(4)
        root = np.full(3, np.nan)
        if pose_ok:
            transform[:3, :3] = quaternion_xyzw_to_rotation_matrix(q)
            transform[:3, 3] = p
            root = p + transform[:3, :3] @ self.attachment
        for actor in self.drone_actors:
            actor.SetVisibility(bool(pose_ok and self.checks["Measured"].isChecked()))
            if pose_ok:
                actor.user_matrix = transform
        cable = d.cable[index].copy() if fresh else np.full((10, 3), np.nan)
        valid = np.isfinite(cable).all(axis=1)
        self.marker_mesh.copy_from(pv.PolyData(cable[valid] if valid.any() else np.zeros((1, 3))))
        self.marker_actor.SetVisibility(bool(valid.any() and self.checks["Measured"].isChecked()))
        self.cable_mesh.copy_from(segments(np.vstack([root, cable])))
        self.cable_actor.SetVisibility(self.cable_mesh.n_lines > 0 and self.checks["Measured"].isChecked())
        if self.prediction is not None:
            b=self.prediction
            j=int(np.searchsorted(b.time,now+1e-9,side='right')-1)
            visible=0<=j<len(b.time) and b.time[0]-1e-9<=now<=b.time[-1]+1e-9
            for actor in self.layers['Predicted'][1:]:
                actor.SetVisibility(bool(visible and self.checks['Predicted'].isChecked()))
            if b.metadata.get('mode') == 'prescribed_cable':
                # The measured drone is the input, not a second prediction.
                self.layers['Predicted'][0].SetVisibility(False)
                self.predicted_body.SetVisibility(False)
            if visible:
                transform=np.eye(4);transform[:3,:3]=b.rotation[j];transform[:3,3]=b.position[j]
                self.predicted_body.user_matrix=transform
                root=b.position[j]+b.rotation[j]@b.attachment
                self.predicted_cable.copy_from(segments(np.vstack([root,b.markers[j]])))
                self.predicted_markers.points=b.markers[j].copy()
        self.time_label.setText(f"{now:.2f} / {d.time[-1]:.2f} s")
        self.plotter.render()
        self.frame_changed.emit(float(now))

    def seek(self, index):
        if self.data is not None:
            self.elapsed = float(self.data.time[index])
            self.last_tick = perf_counter()
            self.draw_frame(index)

    def toggle(self):
        if self.data is None:
            return
        if self.playing:
            self.pause()
            return
        if self.elapsed >= self.data.time[-1]:
            self.timeline.setValue(0)
        self.last_tick = perf_counter()
        self.playing = True
        self.play.setText("Pause")
        self.timer.start()

    def tick(self):
        now = perf_counter()
        self.elapsed = min(self.data.time[-1], self.elapsed+(now-self.last_tick)*self.speed.currentData())
        self.last_tick = now
        index = max(0, np.searchsorted(self.data.time, self.elapsed, side="right")-1)
        self.timeline.blockSignals(True)
        self.timeline.setValue(int(index))
        self.timeline.blockSignals(False)
        self.draw_frame(index)
        if self.elapsed >= self.data.time[-1]:
            self.pause()

    def closeEvent(self, event):
        if self.worker is not None:
            self.statusBar().showMessage("Finishing load; close again when ready")
            event.ignore()
            return
        self.pause()
        self.plotter.close()
        event.accept()
