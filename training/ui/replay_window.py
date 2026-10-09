"""Shared saved-motion presentation for Training and the main Replay browser."""
import json
from time import perf_counter

from viewer.replay_data import PresentationData, velocity_metrics

import numpy as np
from PySide6 import QtCore, QtGui, QtWidgets
from pyvistaqt import QtInteractor
import pyvista as pv
from viewer.pyvista_viewer import PyVistaViewer
from viewer.replay_trajectories import SavedTrajectoryControls
from viewer.workspace import add_cylinder_boundary, add_strike_surface, add_event_marker


STYLE = """
QMainWindow, QWidget#page { background: #edf2f6; color: #213547; }
QLabel { color: #213547; font-family: 'Segoe UI'; font-size: 13px; }
QLabel#heading { font-size: 24px; font-weight: 650; }
QLabel#subtle { color: #607486; font-size: 12px; }
QLabel#metric { color: #087d79; font-size: 36px; font-weight: 650; }
QLabel#smallMetric { font-size: 18px; font-weight: 600; }
QFrame#card { background: white; border: 1px solid #dce5ec; border-radius: 10px; }
QLabel#cardTitle { font-size: 12px; font-weight: 650; color: #607486; }
QLabel#success { color: #087d79; font-size: 14px; font-weight: 600; }
QPushButton { background: white; border: 1px solid #ccd8e2; border-radius: 5px;
              padding: 7px 13px; color: #213547; font-size: 12px; }
QPushButton:hover { background: #e1f2f0; border-color: #459b95; }
QPushButton#primary { background: #087d79; color: white; border: 0; font-weight: 600; }
QComboBox { background: white; color: #213547; border: 1px solid #ccd8e2; border-radius: 4px; padding: 6px; }
QComboBox QAbstractItemView, QPlainTextEdit { background: white; color: #213547; }
QGroupBox { border: 0; font-size: 12px; color: #607486; padding-top: 17px; }
QGroupBox::title { subcontrol-origin: margin; }
QCheckBox { font-size: 12px; color: #52687a; }
QSlider::groove:horizontal { background: #cdd9e3; height: 5px; border-radius: 2px; }
QSlider::handle:horizontal { background: #087d79; width: 12px; margin: -4px 0; border-radius: 6px; }
"""


def label(text='', name=None, wrap=False):
    widget = QtWidgets.QLabel(text)
    # Give layouts the final font metrics before stylesheet inheritance/polish.
    # Otherwise Qt can cache the default 13px height for the large speed label.
    font = QtGui.QFont('Segoe UI')
    font.setPixelSize({'heading': 24, 'metric': 36, 'smallMetric': 18,
                       'cardTitle': 12, 'subtle': 12, 'success': 14}.get(name, 13))
    font.setBold(name in ('heading', 'metric', 'smallMetric', 'cardTitle', 'success'))
    widget.setFont(font)
    if name:
        widget.setObjectName(name)
    widget.setWordWrap(wrap)
    return widget


def vector_text(vector, digits=3):
    if vector is None:
        return 'Undefined at zero speed'
    return '   '.join(f'{axis} {value:+.{digits}f}' for axis, value in zip('XYZ', vector))


class ReplayWindow(QtWidgets.QMainWindow):
    def __init__(self, directory, filename=None, event=None, parent=None, *, off_screen=False):
        super().__init__(parent)
        self.setAttribute(QtCore.Qt.WidgetAttribute.WA_DeleteOnClose)
        data = PresentationData.load(directory, filename, event)
        self.data, self.replay = data, data.replay
        self.setWindowTitle(f'AeroWhip Replay · Stage {data.stage} · {data.directory.name}')
        self.resize(1440, 940)
        self.setMinimumSize(0, 0)
        self.setStyleSheet(STYLE)
        central = QtWidgets.QWidget(objectName='page')
        self.setCentralWidget(central)
        layout = QtWidgets.QVBoxLayout(central)
        layout.setContentsMargins(20, 16, 20, 14)
        layout.setSpacing(10)
        title_row = QtWidgets.QHBoxLayout()
        heading = QtWidgets.QVBoxLayout()
        heading.addWidget(label(f'Stage {data.stage} · Cable-tip motion', 'heading'))
        heading.addWidget(label(f"{data.event.get('phase', 'Saved states').replace('_', ' ')} · collection {data.event.get('round', '—')} · {data.filename}", 'subtle'))
        title_row.addLayout(heading, 1)
        for text, callback in (('3D view', self.overview_camera), ('Target view', self.target_camera),
                               ('Details', self.show_details)):
            button = QtWidgets.QPushButton(text)
            button.clicked.connect(callback)
            title_row.addWidget(button)
        layout.addLayout(title_row)
        body = QtWidgets.QHBoxLayout()
        body.setSpacing(16)
        self.plotter = QtInteractor(central, off_screen=off_screen, auto_update=False)
        body.addWidget(self.plotter.interactor, 1)
        sidebar = QtWidgets.QWidget()
        sidebar.setFixedWidth(340)
        cards = QtWidgets.QVBoxLayout(sidebar)
        cards.setContentsMargins(0, 0, 0, 0)
        cards.setSpacing(10)
        self.viewer = PyVistaViewer(data.scene)
        self.viewer.create_plotter(self.replay.state, plotter=self.plotter)
        self.viewer._toggle_world_axes()
        self.plotter.add_axes(interactive=False, color='#344958', line_width=2)
        add_cylinder_boundary(self.plotter, **data.config.workspace.description(data.scene.initial_state.position_world_m))
        corners = add_strike_surface(self.plotter, **{key: data.surface[key] for key in
                                                    ('center_world_m', 'normal_world', 'width_m', 'height_m')})
        self.strike_arrow = None
        if data.strike is not None:
            add_event_marker(self.plotter, data.strike['position'])
            if data.strike['direction'] is not None:
                self.strike_arrow = self.plotter.add_mesh(pv.Arrow(start=data.strike['position'],
                    direction=data.strike['direction'], scale=.32), name='recorded_strike_velocity',
                    color='#df3c4b', lighting=False, pickable=False)
        self.tip_arrow = self.plotter.add_mesh(pv.Arrow(), name='live_tip_velocity',
            color='#087d79', lighting=False, pickable=False)
        self._points = np.vstack((self.replay.arrays['cable_positions'].reshape(-1, 3), corners,
                                  self.replay.arrays['drone_positions']))
        self._camera_ready = True
        self.overview_camera()
        # Reuse original paths while suppressing deferred duplicate insertion.
        self.trajectory_controls = SavedTrajectoryControls(self.plotter, self.replay.arrays, central)
        self.trajectory_controls.setTitle('Saved paths · full replay')
        for key in ('trail', 'tip_trail'):
            self.viewer._overlays.visible[key] = False
            self.viewer._overlays.actors[key].SetVisibility(False)

        s = data.strike
        strike_card = self.card(cards, 'RECORDED EVENT · CABLE TIP')
        strike_card.addWidget(label('—' if s is None else f"{s['speed']:.2f} m/s", 'metric'))
        if s is None:
            strike_card.addWidget(label('No selected event recorded', 'subtle'))
        else:
            strike_card.addWidget(label(f"Forward speed  {s['forward']:.2f} m/s"))
            angle = 'Undefined' if s['angle_deg'] is None else f"{s['angle_deg']:.2f}°"
            strike_card.addWidget(label('Entry angle  ' + angle, 'smallMetric'))
            strike_card.addWidget(label('Direction · world unit vector', 'cardTitle'))
            strike_card.addWidget(label(vector_text(s['direction'])))
            strike_card.addWidget(label(f"{s['time_s']:.3f} s · {s['kind']}", 'subtle', True))

        position_card = self.card(cards, 'EVENT POSITION · WORLD FRAME')
        if s is not None:
            position_card.addWidget(label(vector_text(s['position'], 4) + ' m'))
            position_card.addWidget(label(f"Target-center offset  {s['center_error_m'] * 1000:.2f} mm", 'smallMetric'))
            position_card.addWidget(label(f"Lateral {s['lateral_m'] * 1000:+.2f} mm · Vertical {s['vertical_m'] * 1000:+.2f} mm"))
        else:
            position_card.addWidget(label('No recorded event position', 'subtle'))
        width, height = data.surface['width_m'], data.surface['height_m']
        position_card.addWidget(label(f'Target {width * 100:g} × {height * 100:g} cm · '
                                     f'±{width * 50:g} / ±{height * 50:g} cm', 'subtle'))
        live_card = self.card(cards, 'LIVE PLAYBACK · CABLE TIP')
        self.live_speed = label('', 'smallMetric')
        self.live_position = label()
        self.live_direction = label()
        self.drone_position = label('', 'subtle')
        self.telemetry = self.drone_position
        self.live_status = label('', 'subtle')
        for widget in (self.live_speed, self.live_position, self.live_direction,
                       self.drone_position, self.live_status):
            live_card.addWidget(widget)

        outcome_card = self.card(cards, 'SAVED FINAL OUTCOME')
        self.outcome = label(self.outcome_text(), 'success', True)
        self.result = self.outcome
        outcome_card.addWidget(self.outcome)
        components = data.event.get('components', {})
        if 'final_position_error_m' in components and 'final_speed_mps' in components:
            outcome_card.addWidget(label(f"Drone return  {components['final_position_error_m'] * 100:.2f} cm error"
                                         f" · {components['final_speed_mps']:.3f} m/s", wrap=True))
        cards.addStretch(1)
        scroll = QtWidgets.QScrollArea()
        scroll.setFrameShape(QtWidgets.QFrame.Shape.NoFrame)
        scroll.setWidgetResizable(True)
        scroll.setFixedWidth(360)
        scroll.setWidget(sidebar)
        scroll.setStyleSheet('QScrollArea, QScrollArea > QWidget > QWidget { background: #edf2f6; }')
        body.addWidget(scroll)
        layout.addLayout(body, 1)

        controls = QtWidgets.QHBoxLayout()
        self.play = QtWidgets.QPushButton('Play', objectName='primary')
        self.play.clicked.connect(self.toggle)
        controls.addWidget(self.play)
        for text, callback in (('Step', self.step), ('Reset', self.reset), ('Jump to event', self.jump_to_strike)):
            button = QtWidgets.QPushButton(text)
            button.clicked.connect(callback)
            controls.addWidget(button)
        self.speed = QtWidgets.QComboBox()
        self.speed.setMinimumWidth(70)
        for value in (.1, .25, .5, 1):
            self.speed.addItem(f'{value:g}×', value)
        self.speed.setCurrentIndex(2)
        self.replay.speed = .5
        self.speed.currentIndexChanged.connect(lambda _: setattr(self.replay, 'speed', self.speed.currentData()))
        controls.addWidget(self.speed)
        self.slider = QtWidgets.QSlider(QtCore.Qt.Orientation.Horizontal)
        self.slider.setRange(0, len(self.replay.time) - 1)
        self.slider.sliderPressed.connect(self.pause)
        self.slider.valueChanged.connect(self.seek)
        controls.addWidget(self.slider, 1)
        self.clock = label()
        self.clock.setMinimumWidth(145)
        controls.addWidget(self.clock)
        layout.addLayout(controls)
        lower = QtWidgets.QHBoxLayout()
        lower.addWidget(self.trajectory_controls, 1)
        self.workspace_toggle = QtWidgets.QCheckBox('Workspace boundary')
        self.workspace_toggle.setChecked(True)
        self.workspace_toggle.toggled.connect(lambda shown: self.visibility('workspace', shown))
        lower.addWidget(self.workspace_toggle)
        self.arrows_toggle = QtWidgets.QCheckBox('Direction arrows')
        self.arrows_toggle.setChecked(True)
        self.arrows_toggle.toggled.connect(lambda _: self.refresh())
        lower.addWidget(self.arrows_toggle)
        layout.addLayout(lower)
        layout.addWidget(label('Red: recorded event + direction   ·   Teal: current tip direction   ·   Blue: target entry   '
                               '·   Arrows show direction, not speed scale', 'subtle'))
        layout.addWidget(label('Saved simulation only · virtual target, no contact force · not robustness or flight evidence', 'subtle'))
        self.shortcuts = []
        for key, callback in (('Space', self.toggle), ('Right', self.step), ('R', self.reset), ('S', self.jump_to_strike)):
            shortcut = QtGui.QShortcut(QtGui.QKeySequence(key), self)
            shortcut.activated.connect(callback)
            self.shortcuts.append(shortcut)
        self.timer = QtCore.QTimer(self)
        self.timer.timeout.connect(self.tick)
        self.timer.start(33)
        self.last = perf_counter()
        self.refresh()

    def card(self, parent_layout, title):
        frame = QtWidgets.QFrame(objectName='card')
        layout = QtWidgets.QVBoxLayout(frame)
        layout.setContentsMargins(12, 9, 12, 9)
        layout.setSpacing(4)
        layout.setSizeConstraint(QtWidgets.QLayout.SizeConstraint.SetMinimumSize)
        layout.addWidget(label(title, 'cardTitle'))
        parent_layout.addWidget(frame)
        return layout

    def outcome_text(self):
        c = self.data.event.get('components', {})
        items = [('Geometric hit', 'geometric_hit_success'), ('Strict strike', 'strict_strike_success')]
        if self.data.stage == 4:
            items += [('Recovery', 'recovery_success'), ('Cable settling', 'cable_settling_success')]
        outcomes = [name + (' PASS' if c[key] == 1 else ' NOT MET') for name, key in items if key in c]
        if self.data.event.get('valid') is False:
            outcomes.insert(0, 'INVALID rollout · finite saved prefix only')
        return '  ·  '.join(outcomes) or 'No outcome recorded'

    def visibility(self, name, shown):
        self.plotter.renderer.actors[name].SetVisibility(shown)
        self.plotter.render()

    def overview_camera(self):
        if not getattr(self, '_camera_ready', False):
            return
        center = .5 * (self._points.min(0) + self._points.max(0))
        self.plotter.camera_position = [center + np.array([2.5, -4., 2.]), center, (0, 0, 1)]
        self.plotter.camera.view_angle = 30
        self.plotter.reset_camera(bounds=np.column_stack((self._points.min(0), self._points.max(0))).ravel())
        self.plotter.camera.zoom(1.2)
        self.plotter.render()

    def target_camera(self):
        center = np.asarray(self.data.surface['center_world_m'])
        normal = np.asarray(self.data.surface['normal_world'])
        self.plotter.camera_position = [center - 1.8 * normal, center, (0, 0, 1)]
        self.plotter.camera.view_angle = 25
        self.plotter.render()

    def show_details(self):
        dialog = QtWidgets.QDialog(self)
        dialog.setWindowTitle('Recorded artifact and scientific details')
        dialog.resize(850, 650)
        layout = QtWidgets.QVBoxLayout(dialog)
        text = QtWidgets.QPlainTextEdit()
        text.setReadOnly(True)
        evidence = self.data.evidence()
        evidence['recorded_event'] = self.data.event
        text.setPlainText(json.dumps(evidence, indent=2))
        layout.addWidget(text)
        close = QtWidgets.QPushButton('Close')
        close.clicked.connect(dialog.accept)
        layout.addWidget(close)
        dialog.exec()

    def refresh(self):
        a, index = self.replay.arrays, self.replay.index
        self.viewer._update_geometry(self.replay.state)
        velocity = a['cable_velocities'][index, -1]
        position = a['cable_positions'][index, -1]
        metrics = velocity_metrics(velocity, self.data.surface['normal_world'])
        self.live_speed.setText(f"{metrics['speed']:.2f} m/s")
        self.live_position.setText('Position [m]  ' + vector_text(position))
        self.live_direction.setText('Direction  ' + vector_text(metrics['direction']))
        self.drone_position.setText('Drone [m]  ' + vector_text(a['drone_positions'][index]))
        recovery_start = self.data.event.get('components', {}).get('recovery_start_s')
        phase = ('Recovery' if recovery_start is not None and self.replay.state.time_s >= recovery_start
                 else 'Whip motion')
        self.live_status.setText(phase)
        if self.strike_arrow is not None:
            self.strike_arrow.SetVisibility(self.arrows_toggle.isChecked())
        direction = metrics['direction']
        self.tip_arrow.SetVisibility(self.arrows_toggle.isChecked() and direction is not None)
        if direction is not None:
            helper = np.eye(3)[int(np.argmin(np.abs(direction)))]
            side = np.cross(direction, helper)
            side /= np.linalg.norm(side)
            transform = np.eye(4)
            transform[:3, :3] = .22 * np.column_stack((direction, side, np.cross(direction, side)))
            transform[:3, 3] = position
            self.tip_arrow.user_matrix = transform
        self.slider.blockSignals(True)
        self.slider.setValue(index)
        self.slider.blockSignals(False)
        self.clock.setText(f'{self.replay.state.time_s:.2f} / {self.replay.duration_s:.2f} s')
        self.play.setText('Pause' if self.replay.running else 'Play')
        self.plotter.render()

    def tick(self):
        now = perf_counter()
        elapsed, self.last = now - self.last, now
        if self.replay.advance(elapsed):
            self.refresh()

    def toggle(self):
        self.replay.toggle_running()
        self.last = perf_counter()
        self.refresh()

    def pause(self):
        self.replay.running = False
        self.refresh()

    def seek(self, index):
        self.replay.running = False
        self.replay.seek(index)
        self.refresh()

    def step(self):
        self.replay.step()
        self.refresh()

    def reset(self):
        self.replay.reset()
        self.refresh()

    def jump_to_strike(self):
        if self.data.strike is not None:
            self.seek(min(len(self.replay.time) - 1, int(np.searchsorted(self.replay.time, self.data.strike['time_s']))))

    def closeEvent(self, event):
        self.timer.stop()
        self.plotter.close()
        super().closeEvent(event)
