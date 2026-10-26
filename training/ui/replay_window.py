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
from viewer.workspace import add_cylinder_boundary


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
        heading.addWidget(label('Cable-tip motion', 'heading'))
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
            color='#087d79', lighting=False, pickable=False)
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
        layout.addWidget(label('Teal: current tip direction · arrow shows direction, not speed scale', 'subtle'))
        if review is not None:
            layout.addWidget(label('Amber: unreviewed candidate interval and measurement peak · '
                'marks are separate from the saved outcome', 'subtle'))
        layout.addWidget(label('Saved states only · not robustness or flight evidence', 'subtle'))
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
        if 'continuous_whip_credit' in c:
            outcome=(('Full horizon completed' if self.data.event.get('valid') else 'Full horizon completed · physical violation')
                     if c.get('completed_horizon')==1 else 'Aborted · finite saved prefix')
            return outcome+'\nContinuous reward, not a binary whip-success test.'
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
        self.live_speed.setText(f"{metrics['speed']:.2f} m/s")
        self.live_position.setText('Position [m]  ' + vector_text(position))
        self.live_direction.setText('Direction  ' + vector_text(metrics['direction']))
        self.drone_position.setText('Drone [m]  ' + vector_text(a['drone_positions'][index]))
        phase = 'Whip motion'
        if self.data.measurement_event is not None:
            phase = 'Saved motion'
        if self.measurement_candidate is not None:
            candidate = self.measurement_candidate
            location = ('inside' if candidate['start_index'] <= index <= candidate['end_index']
                        else 'before' if index < candidate['start_index'] else 'after')
            phase += f' · {location} unreviewed candidate'
        self.live_status.setText(phase)
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

    def jump_to_measurement(self):
        if self.measurement_candidate is not None:
            self.seek(self.measurement_candidate['peak_index'])

    def closeEvent(self, event):
        self.timer.stop()
        self.plotter.close()
        super().closeEvent(event)
