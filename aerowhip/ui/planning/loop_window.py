"""AeroWhip Identification Loop: plan, fly, measure, fit and check the drone + cable model, in one window.

    python -m aerowhip.ui.planning

Left rail: the five steps of the loop with their state, and the planned flights (newest first).
Centre: the selected flight in 3D (lab floor, flight area, the drone and the simulated cable played
back along the planned path; the next 3 s of the path highlighted) with its timeline, or the
Information, Safety, Envelope and Log pages. Right: the flight's key numbers, the values at the
playback cursor, and the actions of the selected step.

Jobs (optimise, preview, envelope, label) run as their own processes, one at a time, with their
output in runs/planning/jobs/; closing the window does not stop them. The 3D preview is one
simulation of the adopted model (scripts/preview_trajectory.py, a few seconds), made when a flight
without one is selected. The window never flies: Fly copies the trajectory into the flight stack's
calibration_data/plans/ (the container sees /workspace/calibration_data) and puts the command on the
clipboard. While a flight script runs on this computer, every computing action is locked: a
simulation running during a flight once stalled the host loop and dropped the drone.
"""
from datetime import datetime
import shutil
import sys
import time

import numpy as np
from PySide6 import QtCore, QtGui, QtWidgets

from aerowhip.fitting import calibration_set
from aerowhip.paths import ROOT
from aerowhip.planning.simulate import FLIGHT_STACK
from aerowhip.ui import style
from aerowhip.ui.planning import loop_data as ld
from aerowhip.ui.planning.scene import Scene3D
from aerowhip.ui.planning.views import BAD, GOOD, MUTED, WARN, EnvelopeView, InformationView, SafetyView, Strip, label

ACCENT = '#3dd6c6'
T = style._TOKENS[True]
EXTRA_STYLE = f'''
QFrame#topbar {{ background: #161b21; border-bottom: 1px solid {T['border']}; }}
QLabel#brand {{ font-size: 17px; font-weight: 700; letter-spacing: 1px; color: {T['text']}; }}
QLabel#brandSub {{ font-size: 13px; color: {T['muted']}; }}
QLabel#pill {{ border-radius: 10px; padding: 3px 10px; font-weight: 600; font-size: 12px; }}
QFrame#rail, QFrame#inspector {{ background: #181d23; }}
QLabel#railTitle {{ color: {T['muted']}; font-size: 11px; font-weight: 700; letter-spacing: 1.5px; padding: 6px 4px 2px 4px; }}
QPushButton#step {{ text-align: left; padding: 9px 10px; border: 1px solid transparent; border-radius: 6px; background: transparent; }}
QPushButton#step:hover {{ background: {T['hover']}; }}
QPushButton#step:checked {{ background: #22303c; border-color: #2d4558; }}
QListWidget#library {{ background: transparent; border: 0; }}
QListWidget#library::item {{ padding: 8px 6px; border-radius: 6px; margin: 1px 0; }}
QListWidget#library::item:selected {{ background: #22303c; color: {T['text']}; }}
QListWidget#library::item:hover {{ background: {T['hover']}; }}
QFrame#tile {{ background: #1f262e; border: 1px solid #2a333d; border-radius: 8px; }}
QLabel#tileValue {{ font-size: 19px; font-weight: 650; color: {T['text']}; }}
QLabel#tileLabel {{ font-size: 11px; color: {T['muted']}; letter-spacing: .5px; }}
QLabel#heading {{ font-size: 17px; font-weight: 650; }}
QLabel#title {{ font-size: 20px; font-weight: 700; }}
QTabBar#segments::tab {{ padding: 7px 16px; border: 0; border-bottom: 2px solid transparent; background: transparent; color: {T['muted']}; }}
QTabBar#segments::tab:selected {{ color: {T['text']}; border-bottom: 2px solid {ACCENT}; }}
QToolButton#camera {{ background: rgba(30,36,44,210); border: 1px solid #2f3a45; border-radius: 5px; padding: 4px 10px; color: {T['text']}; }}
QToolButton#camera:hover {{ background: #2a3440; }}
QToolButton#camera:checked {{ background: #22303c; border-color: {ACCENT}; color: {ACCENT}; }}
QToolButton#play {{ background: {ACCENT}; color: #0e1418; border: 0; border-radius: 16px; font-size: 15px; font-weight: 700; }}
QPushButton#primary {{ background: {ACCENT}; color: #0e1418; border-color: {ACCENT}; font-weight: 650; padding: 7px 14px; }}
QPushButton#primary:disabled {{ background: {T['hover']}; color: {T['disabled']}; border-color: {T['border']}; }}
QFrame#panel {{ background: #1f262e; border: 1px solid #2a333d; border-radius: 8px; }}
QTableWidget {{ alternate-background-color: #1c2229; }}
QAbstractSpinBox {{ background: {T['panel']}; color: {T['text']}; border: 1px solid {T['border']}; border-radius: 3px; padding: 4px 6px; }}
'''
STEPS = (('Plan', 'Optimise the next flight'), ('Fly', 'Run it in the lab'), ('Measure', 'What the cable really did'),
         ('Label & fit', 'Learn from the flight'), ('Check', 'Held-out validation'))


def pill(text, colour):
    w = QtWidgets.QLabel(text)
    w.setObjectName('pill')
    w.setFixedHeight(24)
    w.setSizePolicy(QtWidgets.QSizePolicy.Policy.Maximum, QtWidgets.QSizePolicy.Policy.Fixed)
    set_pill(w, text, colour)
    return w


def set_pill(w, text, colour):
    w.setText(text)
    w.setStyleSheet(f'color: {colour}; border: 1px solid {colour}; background: transparent; border-radius: 11px; '
                    'padding: 3px 11px; font-weight: 600; font-size: 12px;')


def button(text, primary=False):
    b = QtWidgets.QPushButton(text)
    if primary:
        b.setObjectName('primary')
    b.setCursor(QtCore.Qt.CursorShape.PointingHandCursor)
    return b


class Tile(QtWidgets.QFrame):
    def __init__(self, caption):
        super().__init__()
        self.setObjectName('tile')
        layout = QtWidgets.QVBoxLayout(self)
        layout.setContentsMargins(12, 9, 12, 10)
        layout.setSpacing(2)
        self.value = label('—', 'tileValue', wrap=False)
        layout.addWidget(self.value)
        layout.addWidget(label(caption.upper(), 'tileLabel', wrap=True))

    def set(self, text, colour=None):
        self.value.setText(text)
        self.value.setStyleSheet(f'color: {colour};' if colour else '')


class StepButton(QtWidgets.QPushButton):
    def __init__(self, number, title, subtitle):
        super().__init__()
        self.setObjectName('step')
        self.setCheckable(True)
        self.setCursor(QtCore.Qt.CursorShape.PointingHandCursor)
        layout = QtWidgets.QHBoxLayout(self)
        layout.setContentsMargins(10, 6, 10, 6)
        self.badge = QtWidgets.QLabel(str(number))
        self.badge.setFixedSize(26, 26)
        self.badge.setAlignment(QtCore.Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(self.badge)
        text = QtWidgets.QVBoxLayout()
        text.setSpacing(0)
        text.addWidget(label(title, None, wrap=False))
        self.state = label(subtitle, 'tileLabel', wrap=False)
        text.addWidget(self.state)
        layout.addLayout(text, 1)
        self.setMinimumHeight(52)
        self.set_state(subtitle, MUTED)

    def set_state(self, text, colour):
        self.state.setText(text)
        self.state.setStyleSheet(f'color: {colour};')
        self.badge.setStyleSheet(f'border: 2px solid {colour}; border-radius: 13px; color: {colour}; font-weight: 700;')


class LoopWindow(QtWidgets.QMainWindow):
    def __init__(self, off_screen_plotter=None):
        super().__init__()
        self.setWindowTitle('AeroWhip · Identification Loop')
        self.setStyleSheet(style.stylesheet(True)+EXTRA_STYLE)
        try:
            self.setup = calibration_set.default_setup()
        except Exception:
            self.setup = 'cf_6'
        self.job = ld.Job.latest()
        self.was_running = self.job is not None and self.job.running
        self.paths, self.data, self.preview, self.report = [], None, None, None
        self.playing, self.play_t, self.speed, self.last_tick = False, 0., 1., None
        self.flying = False
        self.auto_previewed = set()

        central = QtWidgets.QWidget(objectName='page')
        outer = QtWidgets.QVBoxLayout(central)
        outer.setContentsMargins(0, 0, 0, 0)
        outer.setSpacing(0)
        outer.addWidget(self._topbar())
        body = QtWidgets.QHBoxLayout()
        body.setSpacing(0)
        body.addWidget(self._rail())
        body.addWidget(self._centre(off_screen_plotter), 1)
        body.addWidget(self._inspector())
        outer.addLayout(body, 1)
        self.setCentralWidget(central)
        self.statusBar().showMessage(f'Plans in {ld.PLANNING}. Jobs run on their own: closing this window does not stop them.')

        self.timer = QtCore.QTimer(self, interval=1000, timeout=self.poll)
        self.timer.start()
        self.player = QtCore.QTimer(self, interval=33, timeout=self.tick)
        self.steps[0].setChecked(True)
        self.refresh()
        self.poll()

    # ------------------------------------------------------------ layout
    def _topbar(self):
        bar = QtWidgets.QFrame(objectName='topbar')
        bar.setFixedHeight(54)
        row = QtWidgets.QHBoxLayout(bar)
        row.setContentsMargins(18, 0, 18, 0)
        row.setSpacing(12)
        row.addWidget(label('AEROWHIP', 'brand', wrap=False))
        row.addWidget(label('Identification loop', 'brandSub', wrap=False))
        row.addSpacing(18)
        row.addWidget(label('Drone', 'brandSub', wrap=False))
        self.setup_edit = QtWidgets.QLineEdit(self.setup)
        self.setup_edit.setMaximumWidth(90)
        self.setup_edit.editingFinished.connect(self.refresh)
        row.addWidget(self.setup_edit)
        row.addStretch(1)
        self.job_pill = pill('No job running', MUTED)
        row.addWidget(self.job_pill)
        self.stop_button = button('Stop job')
        self.stop_button.clicked.connect(self.stop)
        row.addWidget(self.stop_button)
        self.flight_pill = pill('No flight running', MUTED)
        row.addWidget(self.flight_pill)
        return bar

    def _rail(self):
        rail = QtWidgets.QFrame(objectName='rail')
        rail.setFixedWidth(270)
        col = QtWidgets.QVBoxLayout(rail)
        col.setContentsMargins(12, 12, 12, 12)
        col.setSpacing(4)
        col.addWidget(label('THE LOOP', 'railTitle', wrap=False))
        group = QtWidgets.QButtonGroup(self)
        group.setExclusive(True)
        self.steps = []
        for i, (title, sub) in enumerate(STEPS):
            b = StepButton(i+1, title, sub)
            group.addButton(b, i)
            col.addWidget(b)
            self.steps.append(b)
        group.idToggled.connect(lambda i, on: on and self.step_pages.setCurrentIndex(i))
        col.addSpacing(10)
        col.addWidget(label('PLANNED FLIGHTS', 'railTitle', wrap=False))
        self.library = QtWidgets.QListWidget(objectName='library')
        self.library.currentRowChanged.connect(self.select)
        col.addWidget(self.library, 1)
        return rail

    def _centre(self, off_screen_plotter):
        centre = QtWidgets.QWidget(objectName='page')
        col = QtWidgets.QVBoxLayout(centre)
        col.setContentsMargins(14, 8, 14, 10)
        col.setSpacing(6)
        self.segments = QtWidgets.QTabBar(objectName='segments')
        for name in ('3D preview', 'Information', 'Safety', 'Envelope', 'Log'):
            self.segments.addTab(name)
        self.segments.setDrawBase(False)
        col.addWidget(self.segments)
        self.pages = QtWidgets.QStackedWidget()
        self.segments.currentChanged.connect(self.pages.setCurrentIndex)
        col.addWidget(self.pages, 1)

        view = QtWidgets.QWidget()
        v = QtWidgets.QVBoxLayout(view)
        v.setContentsMargins(0, 0, 0, 0)
        v.setSpacing(6)
        stage = QtWidgets.QWidget()
        grid = QtWidgets.QGridLayout(stage)
        grid.setContentsMargins(0, 0, 0, 0)
        self.scene = Scene3D(plotter=off_screen_plotter)
        grid.addWidget(self.scene, 0, 0)
        cameras = QtWidgets.QWidget()
        cameras.setAttribute(QtCore.Qt.WidgetAttribute.WA_TranslucentBackground)
        cr = QtWidgets.QHBoxLayout(cameras)
        cr.setContentsMargins(10, 10, 10, 10)
        for text, which in (('3D', 'iso'), ('Top', 'top'), ('Side', 'side'), ('Front', 'front'), ('Drone', 'close')):
            b = QtWidgets.QToolButton(objectName='camera')
            b.setText(text)
            b.clicked.connect(lambda _=False, w=which: self.scene.view(w))
            cr.addWidget(b)
        cr.addSpacing(12)
        self.room_button = QtWidgets.QToolButton(objectName='camera')
        self.room_button.setText('Lab')
        self.room_button.setCheckable(True)
        self.room_button.setChecked(bool(self.scene.lab))
        self.room_button.setEnabled(bool(self.scene.lab))
        self.room_button.setToolTip('The scanned lab (scripts/clean_lab_scan.py)' if self.scene.lab else 'No lab scan: see configs/lab_scan.yaml')
        self.room_button.toggled.connect(self.scene.set_lab_visible)
        cr.addWidget(self.room_button)
        cr.addStretch(1)
        grid.addWidget(cameras, 0, 0, QtCore.Qt.AlignmentFlag.AlignTop | QtCore.Qt.AlignmentFlag.AlignLeft)
        self.preview_note = label('', 'brandSub', wrap=False)
        self.preview_note.setStyleSheet('background: rgba(17,22,28,200); padding: 6px 10px; border-radius: 5px;')
        grid.addWidget(self.preview_note, 0, 0, QtCore.Qt.AlignmentFlag.AlignBottom | QtCore.Qt.AlignmentFlag.AlignLeft)
        v.addWidget(stage, 1)

        timeline = QtWidgets.QHBoxLayout()
        timeline.setSpacing(10)
        self.play_button = QtWidgets.QToolButton(objectName='play')
        self.play_button.setFixedSize(32, 32)
        self.play_button.setText('▶')
        self.play_button.clicked.connect(self.toggle_play)
        timeline.addWidget(self.play_button)
        self.slider = QtWidgets.QSlider(QtCore.Qt.Orientation.Horizontal)
        self.slider.valueChanged.connect(self.seek_index)
        timeline.addWidget(self.slider, 1)
        self.time_label = label('0.0 / 0.0 s', 'brandSub', wrap=False)
        self.time_label.setMinimumWidth(96)
        timeline.addWidget(self.time_label)
        self.speed_box = QtWidgets.QComboBox()
        for s in ('0.25×', '0.5×', '1×', '2×', '4×'):
            self.speed_box.addItem(s)
        self.speed_box.setCurrentIndex(2)
        self.speed_box.currentTextChanged.connect(lambda s: setattr(self, 'speed', float(s[:-1])))
        timeline.addWidget(self.speed_box)
        v.addLayout(timeline)
        self.strip = Strip()
        self.strip.seek_requested.connect(self.seek_time)
        v.addWidget(self.strip)
        self.pages.addWidget(view)

        self.information, self.safety, self.envelope = InformationView(), SafetyView(), EnvelopeView()
        for w in (self.information, self.safety, self.envelope):
            self.pages.addWidget(w)
        log = QtWidgets.QWidget(objectName='page')
        lv = QtWidgets.QVBoxLayout(log)
        lv.setContentsMargins(18, 14, 18, 14)
        self.job_label = label('', 'muted')
        lv.addWidget(self.job_label)
        self.log = QtWidgets.QPlainTextEdit(readOnly=True)
        self.log.setFont(QtGui.QFontDatabase.systemFont(QtGui.QFontDatabase.SystemFont.FixedFont))
        lv.addWidget(self.log, 1)
        self.pages.addWidget(log)
        return centre

    def _inspector(self):
        panel = QtWidgets.QFrame(objectName='inspector')
        panel.setFixedWidth(370)
        scroll = QtWidgets.QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setHorizontalScrollBarPolicy(QtCore.Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        inner = QtWidgets.QWidget(objectName='inspectorInner')
        inner.setStyleSheet('QWidget#inspectorInner { background: #181d23; }')
        col = QtWidgets.QVBoxLayout(inner)
        col.setContentsMargins(16, 14, 16, 14)
        col.setSpacing(10)
        head = QtWidgets.QHBoxLayout()
        self.title = label('No flight planned', 'title', wrap=False)
        head.addWidget(self.title, 1)
        self.state_pill = pill('—', MUTED)
        head.addWidget(self.state_pill)
        col.addLayout(head)
        self.subtitle = label('', 'muted')
        col.addWidget(self.subtitle)
        tiles = QtWidgets.QGridLayout()
        tiles.setSpacing(8)
        self.tiles = {}
        for k, (key, caption) in enumerate((('level', 'peak horizontal accel'), ('length', 'flight length'),
                                            ('rise', 'cable top, worst model'), ('motor', 'smallest motor, worst model'),
                                            ('speed', 'peak speed'), ('weakest', 'teaches least about'))):
            self.tiles[key] = Tile(caption)
            tiles.addWidget(self.tiles[key], k//2, k % 2)
        col.addLayout(tiles)
        col.addWidget(label('AT THE CURSOR (SIMULATED)', 'railTitle', wrap=False))
        cursor = QtWidgets.QFrame(objectName='panel')
        cg = QtWidgets.QGridLayout(cursor)
        cg.setContentsMargins(12, 8, 12, 8)
        self.readout = {}
        for k, (key, caption) in enumerate((('height', 'height'), ('tilt', 'tilt'), ('rise', 'cable top'),
                                            ('motor', 'smallest motor'), ('tip', 'tip speed'), ('accel', 'planned accel'))):
            cg.addWidget(label(caption, 'tileLabel', wrap=False), k//2, (k % 2)*2)
            self.readout[key] = label('—', None, wrap=False)
            cg.addWidget(self.readout[key], k//2, (k % 2)*2+1)
        col.addWidget(cursor)
        col.addWidget(label('STEP', 'railTitle', wrap=False))
        self.step_pages = QtWidgets.QStackedWidget()
        for build in (self._plan_page, self._fly_page, self._measure_page, self._fit_page, self._check_page):
            page = QtWidgets.QFrame(objectName='panel')
            lay = QtWidgets.QVBoxLayout(page)
            lay.setContentsMargins(14, 12, 14, 14)
            lay.setSpacing(8)
            build(lay)
            lay.addStretch(1)
            self.step_pages.addWidget(page)
        col.addWidget(self.step_pages)
        col.addStretch(1)
        scroll.setWidget(inner)
        lay = QtWidgets.QVBoxLayout(panel)
        lay.setContentsMargins(0, 0, 0, 0)
        lay.addWidget(scroll)
        return panel

    def _plan_page(self, lay):
        lay.addWidget(label('Plan the next flight', 'heading'))
        lay.addWidget(label('A free trajectory one level above what the cable has flown cleanly. Every model in the ensemble '
                            'must keep the cable below the clamp; among those, the one that teaches the fit most about the '
                            'parameter it knows least. About 6 minutes on every core.', 'muted'))
        form = QtWidgets.QFormLayout()
        self.level = QtWidgets.QDoubleSpinBox(decimals=1, minimum=0., maximum=12., singleStep=.5, specialValueText='envelope + 1')
        self.level.setSuffix(' m/s²')
        self.duration = QtWidgets.QSpinBox(minimum=10, maximum=120, singleStep=10, value=60, suffix=' s')
        form.addRow('Level', self.level)
        form.addRow('Length', self.duration)
        lay.addLayout(form)
        self.optimise_button = button('Optimise next flight', primary=True)
        self.optimise_button.clicked.connect(self.optimise)
        lay.addWidget(self.optimise_button)

    def _fly_page(self, lay):
        lay.addWidget(label('Fly it', 'heading'))
        lay.addWidget(label('In the flight shell (./docker/shell.sh --real), with the drone at the start point and the cable '
                            'hanging still. The flight lands itself on a lost link or with the cable over the drone.', 'muted'))
        self.check_text = label('', 'muted')
        lay.addWidget(self.check_text)
        self.command = QtWidgets.QPlainTextEdit(readOnly=True)
        self.command.setMaximumHeight(92)
        self.command.setFont(QtGui.QFontDatabase.systemFont(QtGui.QFontDatabase.SystemFont.FixedFont))
        lay.addWidget(self.command)
        row = QtWidgets.QHBoxLayout()
        self.copy_check = button('Copy preflight check')
        self.copy_check.clicked.connect(lambda: self.copy_command(check=True))
        self.copy_fly = button('Copy flight command', primary=True)
        self.copy_fly.clicked.connect(lambda: self.copy_command(check=False))
        row.addWidget(self.copy_check)
        row.addWidget(self.copy_fly)
        lay.addLayout(row)

    def _measure_page(self, lay):
        lay.addWidget(label('Measure the envelope', 'heading'))
        lay.addWidget(label('Reads every recorded part (hover start to hover end) from the raw marker clouds: how high the '
                            'cable went, how far the drone tilted, whether the flight completed. The next level is planned '
                            'from this, never from the simulator.', 'muted'))
        self.measure_text = label('', None)
        lay.addWidget(self.measure_text)
        self.envelope_button = button('Update envelope', primary=True)
        self.envelope_button.clicked.connect(self.update_envelope)
        lay.addWidget(self.envelope_button)

    def _fit_page(self, lay):
        lay.addWidget(label('Label and fit', 'heading'))
        lay.addWidget(label('Label the cable markers over the whole recording at once, then refit the drone (delay, motor '
                            'response, drag) and the cable (stroke windows weighed) in Model Fit, on every flight of the '
                            'drone.', 'muted'))
        self.fit_text = label('', None)
        lay.addWidget(self.fit_text)
        self.label_button = button('Label newest flight', primary=True)
        self.label_button.clicked.connect(self.label_newest)
        lay.addWidget(self.label_button)
        open_fit = button('Open Model Fit')
        open_fit.clicked.connect(self.open_model_fit)
        lay.addWidget(open_fit)

    def _check_page(self, lay):
        lay.addWidget(label('Check', 'heading'))
        lay.addWidget(label('The coupled validation on held-out flights (Model Fit, last stage): stroke tip path, tip peak '
                            'speed, cable rise. When the held-out score stops improving for two rounds, freeze the model, '
                            'rebuild the reset bank and train.', 'muted'))
        open_fit = button('Open Model Fit', primary=True)
        open_fit.clicked.connect(self.open_model_fit)
        lay.addWidget(open_fit)

    # ------------------------------------------------------------ state
    def refresh(self):
        self.setup = self.setup_edit.text().strip() or self.setup
        current = self.selected_path()
        self.paths = ld.trajectories()
        flights = ld.flown(self.setup)
        names = ld.flown_names(flights)
        self.library.blockSignals(True)
        self.library.clear()
        for p in self.paths:
            d = ld.read_json(p) or {}
            state = 'flown' if p.stem in names else ('copied to flight stack' if ld.copied(p) else 'not flown')
            item = QtWidgets.QListWidgetItem(f"{p.stem}\n{d.get('level', '—')} m/s² · {datetime.fromtimestamp(p.stat().st_mtime):%b %d %H:%M} · {state}")
            item.setForeground(QtGui.QColor(T['text']))
            self.library.addItem(item)
        row = self.paths.index(current) if current in self.paths else (0 if self.paths else -1)
        self.library.setCurrentRow(row)
        self.library.blockSignals(False)

        env = ld.envelope(self.setup)
        self.envelope.show(env)
        clean = ld.clean_level(env)
        self.level.setSpecialValueText(f'envelope + 1 ({clean+1:g})' if clean is not None else 'first level (2)')
        newest = self.paths[0] if self.paths else None
        self.steps[0].set_state(f'{newest.stem} ready' if newest else 'nothing planned', GOOD if newest else WARN)
        if newest is None:
            self.steps[1].set_state('plan first', MUTED)
        elif newest.stem in names:
            self.steps[1].set_state(f'{newest.stem} flown', GOOD)
        else:
            self.steps[1].set_state(f'{newest.stem} to fly', WARN)
        envelope_fresh = env is not None and (not flights or ld.envelope_path(self.setup).stat().st_mtime >= flights[0].stat().st_mtime)
        self.steps[2].set_state(f'clean up to {clean:g} m/s²' if clean is not None else 'no envelope', GOOD if envelope_fresh else WARN)
        self.measure_text.setText(('Clean up to ' + (f'{clean:g} m/s²' if clean is not None else '—')) +
                                  ('' if envelope_fresh else '. A newer flight is not in it yet.'))
        labelled = bool(flights) and (flights[0]/'cable_labels.npz').exists()
        self.steps[3].set_state('no trajectory flight yet' if not flights else (f'{flights[0].name} labelled' if labelled else f'{flights[0].name} to label'),
                                MUTED if not flights else (GOOD if labelled else WARN))
        self.fit_text.setText(f'{len(flights)} trajectory flight(s) of {self.setup}. ' +
                              (f'Newest {flights[0].name}: cable {"labelled" if labelled else "not labelled yet"}.' if flights else ''))
        self.steps[4].set_state('in Model Fit', MUTED)
        self.select()

    def selected_path(self):
        i = self.library.currentRow()
        return self.paths[i] if 0 <= i < len(self.paths) else None

    def select(self, *_):
        path = self.selected_path()
        self.data = ld.read_json(path) if path else None
        self.report = ld.report_of(path) if path else None
        self.preview = ld.load_preview(path) if path else None
        self.scene.set_trajectory(self.data, self.preview)
        self.strip.show(self.data, self.preview)
        self.information.show(self.data, self.report)
        self.safety.show(self.data, self.report)
        self._inspect(path)
        n = self.scene.frames()
        self.slider.blockSignals(True)
        self.slider.setRange(0, max(n-1, 0))
        self.slider.setValue(0)
        self.slider.blockSignals(False)
        self.play_t = 0.
        self._update_cursor(0)
        if path is not None and self.preview is None:
            if self.flying:
                self.preview_note.setText('3D preview not simulated: a flight is running')
            elif path not in self.auto_previewed and not (self.job and self.job.running):
                self.auto_previewed.add(path)
                self.preview_note.setText('Simulating the 3D preview…')
                self.run('preview', [ld.PYTHON, 'scripts/preview_trajectory.py', path], env='OMP_NUM_THREADS=1', show_log=False)
            else:
                self.preview_note.setText('No 3D preview yet')
        else:
            self.preview_note.setText('Simulated with the adopted model · drone, cable and the next 3 s of the plan'
                                      if self.preview is not None else '')
        self.preview_note.setVisible(bool(self.preview_note.text()))

    def _inspect(self, path):
        d = self.data
        if d is None:
            self.title.setText('No flight planned')
            self.subtitle.setText('Optimise the first flight (step 1).')
            for t in self.tiles.values():
                t.set('—')
            self.command.setPlainText('')
            return
        self.title.setText(path.stem)
        flights = ld.flown(self.setup)
        flown = path.stem in ld.flown_names(flights)
        set_pill(self.state_pill, 'FLOWN' if flown else ('READY TO FLY' if ld.copied(path) else 'PLANNED'), GOOD if flown else ACCENT)
        s = d.get('summary', {})
        members = d.get('member_rows') or ((self.report or {}).get('best') or {}).get('members') or []
        motor = min((m.get('motor', 1.) for m in members), default=None)
        self.subtitle.setText(f"Optimised {datetime.fromtimestamp(path.stat().st_mtime):%b %d %H:%M} · {len(members)} models · "
                              f"score {d.get('score', 0):.2f} · extent {' × '.join(f'{v:.2f}' for v in s.get('extent', []))} m")
        self.tiles['level'].set(f"{d.get('level', 0):g} m/s²")
        self.tiles['length'].set(f"{d['t'][-1]:.0f} s")
        rise = d.get('worst_rise')
        self.tiles['rise'].set('—' if rise is None else f'{rise*100:+.0f} cm', None if rise is None else (GOOD if rise < -.06 else WARN))
        self.tiles['motor'].set('—' if motor is None else f'{motor:.2f}', None if motor is None else (GOOD if motor > .15 else WARN))
        self.tiles['speed'].set(f"{s.get('peak_speed', 0):.2f} m/s")
        weakest = str(d.get('weakest', '—'))
        short = weakest.split(' (')[0]
        self.tiles['weakest'].set(short if len(short) < 17 else short[:16]+'…')
        self.tiles['weakest'].setToolTip(weakest)
        self.check_text.setText('Preflight check first (read-only: nothing is sent), then the flight.')
        self.command.setPlainText(self.flight_command(path, True)+'\n\n'+self.flight_command(path, False))

    def poll(self):
        self.flying = ld.flight_running()
        set_pill(self.flight_pill, 'FLIGHT RUNNING · computing locked' if self.flying else 'No flight running', BAD if self.flying else MUTED)
        busy = self.job is not None and self.job.running
        for b in (self.optimise_button, self.envelope_button, self.label_button):
            b.setEnabled(not self.flying and not busy)
            b.setToolTip('A flight is running: no computing during flights (it once stalled the host and dropped the drone)'
                         if self.flying else ('A job is running' if busy else ''))
        self.stop_button.setVisible(busy)
        if self.job is not None:
            if busy:
                minutes, seconds = divmod(int(time.time()-self.job.started), 60)
                set_pill(self.job_pill, f'{self.job.name} · {minutes}:{seconds:02d}', ACCENT)
            else:
                code = self.job.exit_code
                set_pill(self.job_pill, f'{self.job.name} · {"done" if code == 0 else f"failed ({code})"}', GOOD if code == 0 else BAD)
            self.job_label.setText(f'{self.job.name}: {"running" if busy else f"finished, exit {self.job.exit_code}"} · {self.job.log}')
            text = self.job.output()
            if text != self.log.toPlainText():
                self.log.setPlainText(text)
                self.log.verticalScrollBar().setValue(self.log.verticalScrollBar().maximum())
        if self.was_running and not busy:
            self.refresh()
        self.was_running = busy

    # ------------------------------------------------------------ playback
    def toggle_play(self):
        if self.scene.frames() == 0:
            return
        self.playing = not self.playing
        self.play_button.setText('❚❚' if self.playing else '▶')
        self.last_tick = time.monotonic()
        (self.player.start if self.playing else self.player.stop)()

    def tick(self):
        now = time.monotonic()
        self.play_t += (now-self.last_tick)*self.speed
        self.last_tick = now
        t = self.preview['t']
        if self.play_t > t[-1]:
            self.play_t = t[0]
        self.slider.setValue(int(np.searchsorted(t, self.play_t)))

    def seek_time(self, t):
        if self.preview is not None:
            self.play_t = float(t)
            self.slider.setValue(int(np.searchsorted(self.preview['t'], t)))

    def seek_index(self, i):
        if self.preview is None:
            return
        if not self.playing:
            self.play_t = float(self.preview['t'][i])
        self.scene.set_frame(i)
        self._update_cursor(i)

    def _update_cursor(self, i):
        pr = self.preview
        if pr is None:
            self.time_label.setText('— / — s')
            for w in self.readout.values():
                w.setText('—')
            return
        t = float(pr['t'][i])
        self.time_label.setText(f"{t:5.1f} / {pr['t'][-1]:.1f} s")
        self.strip.set_cursor(t)
        rise = float(pr['rise'][i])
        r = self.readout
        r['height'].setText(f"{pr['position'][i][2]:.2f} m")
        r['tilt'].setText(f"{pr['tilt'][i]:.0f}°")
        r['rise'].setText(f'{rise*100:+.0f} cm')
        r['rise'].setStyleSheet(f'color: {GOOD if rise < -.06 else (WARN if rise < 0 else BAD)};')
        r['motor'].setText(f"{pr['motor'][i]:.2f}")
        r['tip'].setText(f"{pr['tip_speed'][i]:.1f} m/s")
        if self.data is not None:
            dt, a = np.asarray(self.data['t']), np.asarray(self.data['acceleration'])
            j = min(int(np.searchsorted(dt, t)), len(dt)-1)
            r['accel'].setText(f'{np.linalg.norm(a[j, :2]):.1f} m/s²')

    # ------------------------------------------------------------ actions
    def run(self, name, command, cwd=ROOT, env='', show_log=True):
        if self.job is not None and self.job.running:
            self.statusBar().showMessage(f'{self.job.name} is still running')
            return
        if ld.flight_running():
            self.statusBar().showMessage('A flight is running: not starting computation')
            return
        job = ld.Job.start(name, command, cwd, env)
        if job is None:
            self.statusBar().showMessage(f'Could not start {name}')
            return
        self.job, self.was_running = job, True
        if show_log:
            self.segments.setCurrentIndex(4)
        self.poll()

    def optimise(self):
        name = ld.next_name()
        command = [ld.PYTHON, 'scripts/optimize_trajectory.py', '--setup', self.setup, '--duration', str(self.duration.value()),
                   '--output', ld.PLANNING/f'{name}.json', '--report', ld.PLANNING/f'{name}_report.json']
        if self.level.value() > 0:
            command += ['--level', f'{self.level.value():g}']
        self.run(f'optimise-{name}', command, env='OMP_NUM_THREADS=1')

    def flight_command(self, path, check):
        return (f'python3 {ld.CONTAINER}/scripts/acceleration_circle_cf2.py --trajectory '
                f'{ld.CONTAINER}/calibration_data/plans/{path.name}' + (' --check' if check else ''))

    def copy_command(self, check):
        path = self.selected_path()
        if path is None:
            return
        ld.PLANS.mkdir(parents=True, exist_ok=True)
        if not ld.copied(path):
            shutil.copyfile(path, ld.PLANS/path.name)
        QtWidgets.QApplication.clipboard().setText(self.flight_command(path, check))
        self.statusBar().showMessage(f'{path.name} is in {ld.PLANS}; the {"preflight check" if check else "flight command"} is on the clipboard')
        self.refresh()

    def update_envelope(self):
        self.run('envelope', ['python3', 'scripts/envelope.py', '--setup', self.setup], cwd=FLIGHT_STACK)

    def label_newest(self):
        flights = ld.flown(self.setup)
        if not flights:
            self.statusBar().showMessage('No trajectory flight to label yet')
            return
        self.run('label', [ld.PYTHON, 'scripts/label_sequence.py', flights[0]])

    def open_model_fit(self):
        QtCore.QProcess.startDetached(str(ld.PYTHON), ['-m', 'aerowhip.ui.fitting'], str(ROOT))

    def stop(self):
        if self.job is not None and self.job.running:
            self.job.stop()
            self.poll()


def main():
    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication(sys.argv)
    window = LoopWindow()
    window.resize(1680, 1000)
    window.show()
    return app.exec()


if __name__ == '__main__':
    sys.exit(main())
