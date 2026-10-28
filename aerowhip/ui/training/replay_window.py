"""Shared saved-motion presentation for Training and the main Replay browser."""
import json
from time import perf_counter

from aerowhip.viewer.replay_data import PresentationData, velocity_metrics

import numpy as np
from PySide6 import QtCore, QtGui, QtWidgets
from pyvistaqt import QtInteractor
import pyvista as pv
from aerowhip.viewer.pyvista_viewer import PyVistaViewer
from aerowhip.viewer.replay_trajectories import SavedTrajectoryControls
from aerowhip.viewer.workspace import add_cylinder_boundary
from aerowhip.ui.motion_plots import MotionPlots
from aerowhip.ui.style import PALETTE, stylesheet


STYLE = stylesheet(False)


def label(text='', name=None, wrap=False):
    widget = QtWidgets.QLabel(text)
    if name:
        widget.setObjectName(name)
    widget.setWordWrap(wrap)
    return widget


def vector_text(vector, digits=3):
    if vector is None:
        return 'Undefined at zero speed'
    return ', '.join(f'{value:+.{digits}f}' for value in vector)


class MeasurementSlider(QtWidgets.QSlider):
    """Timeline annotation only; candidate marks never change playback states."""
    def __init__(self, candidate):
        super().__init__(QtCore.Qt.Orientation.Horizontal)
        self.candidate = candidate
        self.setMinimumHeight(25)

    def paintEvent(self, event):
        super().paintEvent(event)
        if self.candidate is None or self.maximum() == self.minimum():
            return
        option = QtWidgets.QStyleOptionSlider()
        self.initStyleOption(option)
        style = self.style()
        groove = style.subControlRect(QtWidgets.QStyle.ComplexControl.CC_Slider, option,
                                      QtWidgets.QStyle.SubControl.SC_SliderGroove, self)
        handle = style.subControlRect(QtWidgets.QStyle.ComplexControl.CC_Slider, option,
                                      QtWidgets.QStyle.SubControl.SC_SliderHandle, self)
        def x(index):
            return groove.x()+handle.width()//2+QtWidgets.QStyle.sliderPositionFromValue(
                self.minimum(), self.maximum(), index, max(0, groove.width()-handle.width()), option.upsideDown)
        painter = QtGui.QPainter(self)
        painter.setPen(QtGui.QPen(QtGui.QColor('#b67312'), 3))
        y = self.height()-4
        painter.drawLine(x(self.candidate['start_index']), y, x(self.candidate['end_index']), y)
        painter.drawLine(x(self.candidate['peak_index']), y-4, x(self.candidate['peak_index']), y+2)


class ReplayWindow(QtWidgets.QMainWindow):
    def __init__(self, directory=None, filename=None, event=None, parent=None, *, off_screen=False,
                 presentation_data=None):
        super().__init__(parent)
        self.setAttribute(QtCore.Qt.WidgetAttribute.WA_DeleteOnClose)
        data = presentation_data or PresentationData.load(directory, filename, event)
        self.data, self.replay = data, data.replay
        review = data.measurement_event
        self.measurement_candidate = review['event'] if review is not None else None
        self.setWindowTitle(f'AeroWhip Replay · {data.directory.name}')
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
        heading.addWidget(label('Saved motion', 'heading'))
        heading.addWidget(label(f"{data.event.get('phase', 'Saved states').replace('_', ' ')} · collection {data.event.get('round', '—')} · {data.filename}", 'subtle'))
        title_row.addLayout(heading, 1)
        for text, callback in (('3D view', self.overview_camera), ('Details', self.show_details)):
            button = QtWidgets.QPushButton(text)
            button.clicked.connect(callback)
            title_row.addWidget(button)
        layout.addLayout(title_row)
        if review is not None:
            self.review_label = label(f"Offline measurement review · {review['case_name']} · "
                f"{review['evidence_kind']} · UNREVIEWED / uncertain", wrap=True)
            layout.addWidget(self.review_label)
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
        add_cylinder_boundary(self.plotter, **data.config.environment.workspace.description(data.scene.initial_state.position_world_m))
        target = data.target
        if target is not None:       # the strike target with its hit radius
            self.plotter.add_mesh(pv.Sphere(radius=float(data.config.task.hit_radius_m), center=target), color='#E69F00',
                                  opacity=.35, name='strike_target')
            self.plotter.add_mesh(pv.Sphere(radius=.012, center=target), color='#E69F00', name='strike_target_centre')
        candidate = self.measurement_candidate
        if candidate is not None:
            positions = self.replay.arrays['cable_positions'][:, -1]
            points = positions[candidate['start_index']:candidate['end_index']+1]
            if len(points) > 1:
                self.plotter.add_mesh(pv.lines_from_points(points), color='#b67312', line_width=5,
                    name='measurement_candidate_interval', lighting=False, pickable=False)
            self.plotter.add_mesh(pv.Sphere(radius=.013, center=positions[candidate['peak_index']]),
                color='#b67312', name='measurement_candidate_peak', lighting=False, pickable=False)
        self.tip_arrow = self.plotter.add_mesh(pv.Arrow(), name='live_tip_velocity',
            color=PALETTE['green'], lighting=False, pickable=False)
        self._points = np.vstack((self.replay.arrays['cable_positions'].reshape(-1, 3),
                                  self.replay.arrays['drone_positions']))
        self._camera_ready = True
        self.overview_camera()
        # Reuse original paths while suppressing deferred duplicate insertion.
        self.trajectory_controls = SavedTrajectoryControls(self.plotter, self.replay.arrays, central)
        self.trajectory_controls.setTitle('Saved paths · full replay')
        for key in ('trail', 'tip_trail'):
            self.viewer._overlays.visible[key] = False
            self.viewer._overlays.actors[key].SetVisibility(False)

        if review is not None:
            review_card = self.card(cards, 'OFFLINE CANDIDATE · UNREVIEWED')
            if candidate is None:
                review_card.addWidget(label('No candidate event selected; inspect the full saved replay.', wrap=True))
            else:
                review_card.addWidget(label(f"Candidate {candidate['id']} · {candidate['start_s']:.3f}–"
                    f"{candidate['end_s']:.3f} s", 'smallMetric', True))
                review_card.addWidget(label(f"Measurement peak {candidate['peak_s']:.3f} s"))
                for note in candidate['notes']:
                    review_card.addWidget(label(note, 'subtle', True))
            review_card.addWidget(label('Candidate evidence is uncertain. No whip, strike or training '
                                         'success is inferred from these marks.', 'subtle', True))
        # Compact readout at the cursor; the time histories below carry the motion.
        self.outcome = label(self.outcome_text(), 'success', True)
        self.result = self.outcome
        cards.addWidget(self.outcome)
        self.live_speed = self.live_position = label('', 'muted', True)
        self.live_direction = self.drone_position = self.live_status = self.live_position
        self.telemetry = self.live_position
        cards.addWidget(self.live_position)
        self.motion = MotionPlots(minimum=(340, 300))
        self.motion.set_arrays(self.replay.arrays, self.data.output_direction)
        self.motion.seek_requested.connect(lambda t: self.seek(int(np.argmin(abs(self.replay.time-t)))))
        cards.addWidget(self.motion, 1)
        sidebar.setFixedWidth(500)
        body.addWidget(sidebar)
        layout.addLayout(body, 1)

        controls = QtWidgets.QHBoxLayout()
        self.play = QtWidgets.QPushButton('Play', objectName='primary')
        self.play.clicked.connect(self.toggle)
        controls.addWidget(self.play)
        for text, callback in (('Step', self.step), ('Reset', self.reset)):
            button = QtWidgets.QPushButton(text)
            button.clicked.connect(callback)
            controls.addWidget(button)
        if candidate is not None:
            self.measurement_jump = QtWidgets.QPushButton('Jump to candidate peak')
            self.measurement_jump.clicked.connect(self.jump_to_measurement)
            controls.addWidget(self.measurement_jump)
            button = QtWidgets.QPushButton('Candidate start')
            button.clicked.connect(lambda: self.seek(candidate['start_index']))
            controls.addWidget(button)
        self.speed = QtWidgets.QComboBox()
        self.speed.setMinimumWidth(70)
        for value in (.1, .25, .5, 1):
            self.speed.addItem(f'{value:g}×', value)
        self.speed.setCurrentIndex(2)
        self.replay.speed = .5
        self.speed.currentIndexChanged.connect(lambda _: setattr(self.replay, 'speed', self.speed.currentData()))
        controls.addWidget(self.speed)
        self.slider = (MeasurementSlider(candidate) if candidate is not None
                       else QtWidgets.QSlider(QtCore.Qt.Orientation.Horizontal))
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
        layout.addWidget(label('Green arrow: current tip velocity direction (length not to scale). '
                               'Saved states only; playback does not re-simulate.', 'note'))
        if review is not None:
            layout.addWidget(label('Amber: unreviewed candidate interval and measurement peak · '
                'marks are separate from the saved outcome', 'subtle'))
        self.shortcuts = []
        for key, callback in (('Space', self.toggle), ('Right', self.step), ('R', self.reset)):
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
        if 'strike' in c:
            parts=['STRIKE' if c.get('strike') else 'no strike', 'whip event' if c.get('whip_event') else 'no whip event',
                   'recovered' if c.get('recovered') else 'not recovered']
            if self.data.event.get('valid') is False: parts.append('physical violation')
            speed=c.get('peak_tip_speed_mps')
            return ' · '.join(parts)+(f'\nPeak tip speed {speed:.2f} m/s' if isinstance(speed,(int,float)) else '')
        if self.data.event.get('valid') is False:
            return 'INVALID rollout · finite saved prefix only'
        return 'No outcome recorded'

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
        metrics = velocity_metrics(velocity, self.data.output_direction)
        phase = ''
        if self.measurement_candidate is not None:
            candidate = self.measurement_candidate
            location = ('inside' if candidate['start_index'] <= index <= candidate['end_index']
                        else 'before' if index < candidate['start_index'] else 'after')
            phase = f' · {location} unreviewed candidate'
        self.live_position.setText(f"Tip {metrics['speed']:.2f} m/s at [{vector_text(position, 2)}] m · "
                                   f"drone at [{vector_text(a['drone_positions'][index], 2)}] m{phase}")
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
        self.motion.set_cursor(float(self.replay.time[index]))
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

    def jump_to_measurement(self):
        if self.measurement_candidate is not None:
            self.seek(self.measurement_candidate['peak_index'])

    def closeEvent(self, event):
        self.timer.stop()
        self.plotter.close()
        super().closeEvent(event)
