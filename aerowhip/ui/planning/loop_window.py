"""AeroWhip Identification Loop: the planned-identification loop in one window.

    python -m aerowhip.ui.planning

Left: the five steps (plan, fly, measure, label and fit, check), each with its state and its action.
Right: Trajectory (the selected excitation flight: position, top view, accelerations), Information
(what it teaches each fitted parameter, which parameters it leaves coupled, and the search that
found it), Safety (each ensemble member's predicted cable rise, tilt, motors), Envelope (what the
real cable did per kind and level) and Log (the job running now).

Jobs run as their own processes (not children of the window), one at a time, with their output in
runs/planning/jobs/. The window never flies: the Fly step copies the trajectory into the flight
stack's calibration_data/plans/ (the container sees /workspace/calibration_data) and gives the
command to paste in the flight shell. While a flight script is running on this computer, the
computing actions are locked: a simulation running during a flight once stalled the host loop and
dropped the drone.
"""
from datetime import datetime
import json
import os
from pathlib import Path
import shlex
import shutil
import subprocess
import sys

import numpy as np
from PySide6 import QtCore, QtGui, QtWidgets

from aerowhip.fitting import calibration_set
from aerowhip.paths import ROOT
from aerowhip.planning.simulate import FLIGHT_STACK
from aerowhip.ui import style
from aerowhip.ui.plotting import PlotCanvas
from aerowhip.ui.style import AXIS, PALETTE, ROLE

PLANNING = ROOT/'runs'/'planning'
JOBS = PLANNING/'jobs'
PLANS = FLIGHT_STACK/'calibration_data'/'plans'
CONTAINER = '/workspace'
PYTHON = ROOT/'.venv'/'bin'/'python'
FLIGHT_PROCESS = 'acceleration_circle_cf2.py'
GOOD, BAD, WARN = PALETTE['green'], PALETTE['vermillion'], PALETTE['orange']


# ---------------------------------------------------------------- data

def read_json(path):
    try:
        return json.loads(Path(path).read_text())
    except (OSError, ValueError):
        return None


def trajectories():
    """Excitation trajectory files under runs/planning/, newest first."""
    found = [p for p in PLANNING.glob('*.json') if not p.stem.endswith('_report') and 'excitation' in p.stem]
    return sorted(found, key=lambda p: p.stat().st_mtime, reverse=True)


def next_name():
    numbers = [int(p.stem.split('_')[1]) for p in PLANNING.glob('excitation_[0-9][0-9][0-9].json')]
    return f'excitation_{max(numbers, default=0)+1:03d}'


def envelope(setup):
    return read_json(FLIGHT_STACK/'calibration_data'/'envelope'/f'{setup}.json')


def clean_level(env):
    """Highest level flown cleanly (no failure, cable below the threshold), or None."""
    levels = [r['level'] for r in (env or {}).get('summary', []) if r.get('level') is not None and r['clean'] and not r['failed']]
    return max(levels, default=None)


def flown(setup):
    """Flight folders of trajectory flights, newest first."""
    base = FLIGHT_STACK/'calibration_data'/setup/'trajectory'
    return sorted((p for p in base.glob('*') if (p/'flight.jsonl').exists()), key=lambda p: p.name, reverse=True) if base.exists() else []


def flight_running():
    return subprocess.run(['pgrep', '-f', FLIGHT_PROCESS], capture_output=True).returncode == 0


def pid_alive(pid):
    try:
        os.kill(int(pid), 0)
        return True
    except (OSError, ValueError):
        return False


class Job:
    """A detached process with its output in JOBS/<stamp>-<name>.log, its exit code in .exit and its
    shell's process id in .pid; found again from the files when the window is reopened."""
    def __init__(self, log):
        self.log = Path(log)
        self.name = self.log.stem.split('-', 2)[-1]

    @classmethod
    def start(cls, name, command, cwd, env=''):
        JOBS.mkdir(parents=True, exist_ok=True)
        log = JOBS/f'{datetime.now():%Y%m%d-%H%M%S}-{name}.log'
        q = lambda p: shlex.quote(str(p))          # noqa: E731
        script = (f'echo $$ > {q(log.with_suffix(".pid"))}; cd {q(cwd)}; {env} nice -n 10 '
                  f'{" ".join(shlex.quote(str(c)) for c in command)} > {q(log)} 2>&1; echo $? > {q(log.with_suffix(".exit"))}')
        ok, _ = QtCore.QProcess.startDetached('/bin/sh', ['-c', script], str(cwd))
        return cls(log) if ok else None

    @classmethod
    def latest(cls):
        logs = sorted(JOBS.glob('*.log')) if JOBS.exists() else []
        return cls(logs[-1]) if logs else None

    @property
    def exit_code(self):
        try:
            return int(self.log.with_suffix('.exit').read_text().strip())
        except (OSError, ValueError):
            return None

    @property
    def running(self):
        if self.exit_code is not None:
            return False
        try:
            return pid_alive(self.log.with_suffix('.pid').read_text().strip())
        except OSError:
            return True                            # just started: the shell has not written its id yet

    def output(self, lines=400):
        try:
            text = self.log.read_text(errors='replace').splitlines()
        except OSError:
            return ''
        return '\n'.join(l for l in text if 'Warning' not in l)[-lines*120:]

    def stop(self):
        try:
            pid = self.log.with_suffix('.pid').read_text().strip()
        except OSError:
            return
        subprocess.run(['pkill', '-TERM', '-P', pid], check=False)


# ---------------------------------------------------------------- small widgets

def label(text='', name=None, wrap=True):
    w = QtWidgets.QLabel(text)
    if name:
        w.setObjectName(name)
    w.setWordWrap(wrap)
    w.setTextInteractionFlags(QtCore.Qt.TextInteractionFlag.TextSelectableByMouse)
    return w


def chip(text, colour):
    w = QtWidgets.QLabel(text)
    w.setStyleSheet(f'color: {colour}; font-weight: 600;')
    return w


def table(columns):
    t = QtWidgets.QTableWidget(0, len(columns))
    t.setHorizontalHeaderLabels(columns)
    t.verticalHeader().setVisible(False)
    t.setEditTriggers(QtWidgets.QAbstractItemView.EditTrigger.NoEditTriggers)
    t.setShowGrid(False)
    t.horizontalHeader().setSectionResizeMode(QtWidgets.QHeaderView.ResizeMode.ResizeToContents)
    t.horizontalHeader().setDefaultAlignment(QtCore.Qt.AlignmentFlag.AlignLeft)
    return t


def fill(t, rows, colours=None):
    colours = colours or {}
    t.setRowCount(len(rows))
    for i, row in enumerate(rows):
        for j, v in enumerate(row):
            text = f'{v:.3g}' if isinstance(v, float) else ('—' if v is None else str(v))
            item = QtWidgets.QTableWidgetItem(text)
            if isinstance(v, (int, float)) and not isinstance(v, bool):
                item.setTextAlignment(QtCore.Qt.AlignmentFlag.AlignRight | QtCore.Qt.AlignmentFlag.AlignVCenter)
            if (i, j) in colours:
                item.setForeground(QtGui.QColor(colours[(i, j)]))
            t.setItem(i, j, item)


class StepCard(QtWidgets.QFrame):
    def __init__(self, number, title, text):
        super().__init__()
        self.setObjectName('card')
        layout = QtWidgets.QVBoxLayout(self)
        layout.setContentsMargins(12, 10, 12, 12)
        head = QtWidgets.QHBoxLayout()
        head.addWidget(label(f'{number}  {title}', 'heading', wrap=False))
        head.addStretch(1)
        self.state = chip('', PALETTE['grey'])
        head.addWidget(self.state)
        layout.addLayout(head)
        layout.addWidget(label(text, 'note'))
        self.status = label('', None)
        layout.addWidget(self.status)
        self.body = QtWidgets.QVBoxLayout()
        self.body.setSpacing(6)
        layout.addLayout(self.body)

    def set_state(self, text, colour):
        self.state.setText(text)
        self.state.setStyleSheet(f'color: {colour}; font-weight: 600;')

    def row(self, *widgets):
        r = QtWidgets.QHBoxLayout()
        for w in widgets:
            r.addWidget(w)
        r.addStretch(1)
        self.body.addLayout(r)


def button(text, primary=False):
    b = QtWidgets.QPushButton(text)
    if primary:
        b.setObjectName('primary')
    return b


# ---------------------------------------------------------------- the loop (left)

class LoopPanel(QtWidgets.QWidget):
    def __init__(self, window):
        super().__init__()
        self.w = window
        self.setObjectName('page')
        layout = QtWidgets.QVBoxLayout(self)
        layout.setSpacing(10)
        top = QtWidgets.QHBoxLayout()
        top.addWidget(label('Setup', 'cardTitle', wrap=False))
        self.setup = QtWidgets.QLineEdit(window.setup)
        self.setup.setMaximumWidth(120)
        self.setup.editingFinished.connect(window.refresh)
        top.addWidget(self.setup)
        top.addStretch(1)
        self.flying = chip('', PALETTE['grey'])
        top.addWidget(self.flying)
        layout.addLayout(top)

        self.plan = StepCard(1, 'Plan', 'Optimise a free trajectory one level above the clean envelope: every ensemble member '
                             'must keep the cable below the clamp; scored by what it teaches the parameter the fit knows least.')
        self.level = QtWidgets.QDoubleSpinBox(decimals=1, minimum=0., maximum=12., singleStep=.5, specialValueText='envelope + 1')
        self.level.setSuffix(' m/s²')
        self.duration = QtWidgets.QSpinBox(minimum=10, maximum=120, singleStep=10, value=60, suffix=' s')
        self.optimise = button('Optimise next', primary=True)
        self.optimise.clicked.connect(window.optimise)
        self.plan.row(label('Level', 'muted', wrap=False), self.level, label('Length', 'muted', wrap=False), self.duration)
        self.plan.row(self.optimise)
        layout.addWidget(self.plan)

        self.fly = StepCard(2, 'Fly', 'Copies the selected trajectory into the flight stack and puts the command on the '
                            'clipboard; paste it in the flight shell (./docker/shell.sh --real). Check first with --check.')
        self.selected = QtWidgets.QComboBox()
        self.selected.currentIndexChanged.connect(window.select)
        self.copy = button('Copy flight command', primary=True)
        self.copy.clicked.connect(window.copy_command)
        self.command = QtWidgets.QPlainTextEdit(readOnly=True)
        self.command.setMaximumHeight(64)
        self.fly.body.addWidget(self.selected)
        self.fly.row(self.copy)
        self.fly.body.addWidget(self.command)
        layout.addWidget(self.fly)

        self.measure = StepCard(3, 'Measure', 'What the real cable did, per kind and level, from the raw marker clouds '
                                '(the parts only: hover start to hover end).')
        self.envelope_button = button('Update envelope')
        self.envelope_button.clicked.connect(window.update_envelope)
        self.measure.row(self.envelope_button)
        layout.addWidget(self.measure)

        self.fit = StepCard(4, 'Label and fit', 'Label the cable in the new flight (whole recording at once, '
                            'scripts/label_sequence.py), then refit the drone (delay, response, drag) and the cable '
                            '(stroke windows weighed) in Model Fit, on all flights of the setup.')
        self.label_button = button('Label newest flight')
        self.label_button.clicked.connect(window.label_newest)
        self.fit_button = button('Open Model Fit')
        self.fit_button.clicked.connect(window.open_model_fit)
        self.fit.row(self.label_button, self.fit_button)
        layout.addWidget(self.fit)

        self.check = StepCard(5, 'Check', 'Held-out flights through the coupled validation (Model Fit); then plan the next '
                              'level from the new fit and envelope.')
        layout.addWidget(self.check)
        layout.addStretch(1)


# ---------------------------------------------------------------- views (right)

class TrajectoryView(QtWidgets.QWidget):
    def __init__(self):
        super().__init__()
        layout = QtWidgets.QVBoxLayout(self)
        self.summary = label('', 'muted')
        layout.addWidget(self.summary)
        self.canvas = PlotCanvas(minimum=(500, 420))
        layout.addWidget(self.canvas, 1)

    def show(self, data):
        self.canvas.reset()
        if data is None:
            self.summary.setText('No trajectory selected.')
            self.canvas.draw_idle()
            return
        t = np.asarray(data['t']); p = np.asarray(data['position']); a = np.asarray(data['acceleration']); v = np.asarray(data['velocity'])
        s = data.get('summary', {})
        self.summary.setText(f"{data.get('name')}: {t[-1]:.0f} s at level {data.get('level', '—')} m/s²; peak speed "
                             f"{s.get('peak_speed', 0):.2f} m/s, vertical {s.get('peak_vertical', 0):.2f} m/s², extent "
                             f"{s.get('extent')} m. Relative to the hover point.")
        with self.canvas.style():
            grid = self.canvas.figure.add_gridspec(3, 2, width_ratios=(2.2, 1))
            ax = self.canvas.figure.add_subplot(grid[0, 0])
            for k, name in enumerate('xyz'):
                ax.plot(t, p[:, k], color=AXIS[k], label=name)
            ax.set_ylabel('position (m)'); ax.legend(loc='upper right', ncols=3); ax.set_title('Position')
            ax2 = self.canvas.figure.add_subplot(grid[1, 0], sharex=ax)
            ax2.plot(t, np.linalg.norm(a[:, :2], axis=1), color=ROLE['command'], label='horizontal')
            ax2.plot(t, a[:, 2], color=AXIS[2], label='vertical')
            ax2.set_ylabel('m/s²'); ax2.legend(loc='upper right', ncols=2); ax2.set_title('Acceleration')
            ax3 = self.canvas.figure.add_subplot(grid[2, 0], sharex=ax)
            ax3.plot(t, np.linalg.norm(v, axis=1), color=ROLE['model'])
            ax3.set_ylabel('m/s'); ax3.set_xlabel('time (s)'); ax3.set_title('Speed')
            top = self.canvas.figure.add_subplot(grid[:, 1])
            top.plot(p[:, 0], p[:, 1], color=ROLE['measured'], linewidth=.9)
            top.plot([0], [0], 'o', color=ROLE['error'])
            top.set_aspect('equal', adjustable='datalim'); top.set_xlabel('x (m)'); top.set_ylabel('y (m)'); top.set_title('Top view')
        self.canvas.draw_idle()


class InformationView(QtWidgets.QScrollArea):
    def __init__(self):
        super().__init__()
        self.setWidgetResizable(True)
        page = QtWidgets.QWidget()
        page.setObjectName('page')
        layout = QtWidgets.QVBoxLayout(page)
        layout.addWidget(label('Per parameter: the information left about it after the other parameters are allowed to '
                               'move (log10 of 1/(F⁻¹)ᵢᵢ, in units of the fit\'s current uncertainty). Coupled parameters '
                               '(two that move the measurements alike) lower each other\'s value; the pairs a trajectory '
                               'leaves correlated beyond 0.8 are listed below.', 'note'))
        self.canvas = PlotCanvas(minimum=(500, 560))
        layout.addWidget(self.canvas)
        layout.addWidget(label('Coupled estimates', 'section'))
        self.coupled = label('', None)
        layout.addWidget(self.coupled)
        self.setWidget(page)

    def show(self, data, report):
        self.canvas.reset()
        if data is None or 'marginal_log10' not in data:
            self.coupled.setText('—')
            self.canvas.draw_idle()
            return
        marginal = sorted(data['marginal_log10'].items(), key=lambda kv: kv[1])
        correlation = data.get('correlation')
        candidates = (report or {}).get('candidates', [])
        with self.canvas.style():
            rows = 2 if candidates else 1
            grid = self.canvas.figure.add_gridspec(rows, 2, height_ratios=(1.6, 1)[:rows], width_ratios=(1, 1))
            ax = self.canvas.figure.add_subplot(grid[0, 0] if correlation else grid[0, :])
            names = [n for n, _ in marginal]
            values = [v for _, v in marginal]
            colours = [ROLE['error'] if i == 0 else ROLE['model'] for i in range(len(values))]
            ax.barh(range(len(values)), values, color=colours)
            ax.set_yticks(range(len(values)), names); ax.invert_yaxis()
            ax.set_xlabel('log10 marginal information'); ax.set_title(f"Weakest: {names[0]}")
            ax.grid(axis='y', visible=False)
            if correlation:
                order = data['parameters']
                C = np.asarray(correlation)
                cx = self.canvas.figure.add_subplot(grid[0, 1])
                image = cx.imshow(C, vmin=-1, vmax=1, cmap='RdBu_r')
                cx.set_xticks(range(len(order)), [str(i+1) for i in range(len(order))])
                cx.set_yticks(range(len(order)), [f'{i+1} {n[:22]}' for i, n in enumerate(order)])
                cx.grid(False); cx.set_title('Estimate correlation')
                self.canvas.figure.colorbar(image, ax=cx, shrink=.7)
            if candidates:
                hx = self.canvas.figure.add_subplot(grid[1, :])
                ok = [(i, c['score']) for i, c in enumerate(candidates) if c['feasible'] and c['score'] > -1e8]
                bad = [i for i, c in enumerate(candidates) if not c['feasible']]
                if ok:
                    hx.plot(*zip(*ok), 'o', color=ROLE['model'], label='feasible')
                    hx.plot([i for i, _ in ok], np.maximum.accumulate([s for _, s in ok]), color=ROLE['measured'], linewidth=.8, label='best so far')
                if bad:
                    low = min((s for _, s in ok), default=0.)
                    hx.plot(bad, [low]*len(bad), 'x', color=ROLE['error'], label='unsafe on a member')
                hx.set_xlabel('candidate'); hx.set_ylabel('score'); hx.set_title('Search'); hx.legend(loc='lower right', ncols=3)
        self.canvas.draw_idle()
        if 'coupled' not in data:
            self.coupled.setText('Not computed for this trajectory (optimised before the coupling report); optimise again to see it.')
        elif not data['coupled']:
            self.coupled.setText('None beyond 0.8: the trajectory separates every pair.')
        else:
            self.coupled.setText('\n'.join(f'{a}  ~  {b}:  {c:+.2f}' for a, b, c in data['coupled'][:12]))


class SafetyView(QtWidgets.QWidget):
    def __init__(self):
        super().__init__()
        layout = QtWidgets.QVBoxLayout(self)
        layout.addWidget(label('Each ensemble member (fitted-model samples and the brackets that reach the measured flip) '
                               'flown closed loop on the trajectory. Rise: the highest cable node above the attachment '
                               '(must stay below −0.04 m). Motor: the smallest motor as a share of its hover share.', 'note'))
        self.table = table(['member', 'rise (m)', 'tilt (deg)', 'error (m)', 'motor', 'verdict'])
        layout.addWidget(self.table, 1)

    def show(self, data, report):
        rows = (data or {}).get('member_rows') or ((report or {}).get('best') or {}).get('members') or []
        colours = {}
        out = []
        for i, m in enumerate(rows):
            out.append([m['label'], m.get('rise'), m.get('tilt'), m.get('error'), m.get('motor'), 'safe' if m['ok'] else m.get('reason', '')])
            colours[(i, 5)] = GOOD if m['ok'] else BAD
        fill(self.table, out, colours)


class EnvelopeView(QtWidgets.QWidget):
    def __init__(self):
        super().__init__()
        layout = QtWidgets.QVBoxLayout(self)
        self.note = label('', 'note')
        layout.addWidget(self.note)
        self.table = table(['kind', 'heading', 'level', 'flights', 'clean', 'failed', 'max rise (m)', 'over frames', 'max tilt'])
        layout.addWidget(self.table, 1)

    def show(self, env):
        if env is None:
            self.note.setText('No envelope yet: Update envelope after the first flight.')
            fill(self.table, [])
            return
        self.note.setText(f"Worst of all flights per kind, heading and level. Cable over the drone: a point more than "
                          f"{env['rise_threshold_m']:.2f} m above the attachment.")
        rows, colours = [], {}
        for i, r in enumerate(env['summary']):
            rows.append([r['kind'], r.get('heading'), r.get('level'), r['flights'], r['clean'], r['failed'],
                         r.get('max_rise_m'), r.get('max_over_fraction'), r.get('max_tilt_deg')])
            colours[(i, 5)] = BAD if r['failed'] else GOOD
            if r.get('max_rise_m') is not None:
                colours[(i, 6)] = BAD if r['max_rise_m'] > env['rise_threshold_m'] else (WARN if r['max_rise_m'] > -.04 else GOOD)
        fill(self.table, rows, colours)


# ---------------------------------------------------------------- window

class LoopWindow(QtWidgets.QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle('AeroWhip Identification Loop')
        style.apply(self)
        try:
            self.setup = calibration_set.default_setup()
        except Exception:
            self.setup = 'cf_6'
        self.job = Job.latest()
        self.was_running = self.job is not None and self.job.running
        self.paths = []
        self.loop = LoopPanel(self)
        left = QtWidgets.QScrollArea()
        left.setWidgetResizable(True)
        left.setWidget(self.loop)
        left.setMinimumWidth(400)
        self.trajectory, self.information, self.safety, self.envelope = TrajectoryView(), InformationView(), SafetyView(), EnvelopeView()
        self.log = QtWidgets.QPlainTextEdit(readOnly=True)
        self.log.setFont(QtGui.QFontDatabase.systemFont(QtGui.QFontDatabase.SystemFont.FixedFont))
        log_page = QtWidgets.QWidget()
        log_layout = QtWidgets.QVBoxLayout(log_page)
        head = QtWidgets.QHBoxLayout()
        self.job_label = label('', 'muted')
        head.addWidget(self.job_label, 1)
        self.stop_button = button('Stop')
        self.stop_button.clicked.connect(self.stop)
        head.addWidget(self.stop_button)
        log_layout.addLayout(head)
        log_layout.addWidget(self.log, 1)
        self.tabs = QtWidgets.QTabWidget()
        for widget, name in ((self.trajectory, 'Trajectory'), (self.information, 'Information'), (self.safety, 'Safety'),
                             (self.envelope, 'Envelope'), (log_page, 'Log')):
            self.tabs.addTab(widget, name)
        split = QtWidgets.QSplitter()
        split.addWidget(left)
        split.addWidget(self.tabs)
        split.setStretchFactor(1, 1)
        split.setSizes([430, 1100])
        self.setCentralWidget(split)
        self.statusBar().showMessage(f'Plans: {PLANNING}. Jobs run on their own: closing this window does not stop them.')
        self.timer = QtCore.QTimer(self, interval=1000, timeout=self.poll)
        self.timer.start()
        self.refresh()
        self.poll()

    # state
    def refresh(self):
        self.setup = self.loop.setup.text().strip() or self.setup
        current = self.selected_path()
        self.paths = trajectories()
        box = self.loop.selected
        box.blockSignals(True)
        box.clear()
        for p in self.paths:
            d = read_json(p.with_name(p.stem+'_report.json')) or {}
            box.addItem(f"{p.stem}  ({datetime.fromtimestamp(p.stat().st_mtime):%b %d %H:%M}, level {d.get('level', '—')})")
        if current in self.paths:
            box.setCurrentIndex(self.paths.index(current))
        box.blockSignals(False)
        env = envelope(self.setup)
        self.envelope.show(env)
        clean = clean_level(env)
        self.loop.level.setSpecialValueText(f'envelope + 1 ({clean+1:g})' if clean is not None else 'first level (2)')
        newest = self.paths[0] if self.paths else None
        nd = read_json(newest) if newest else None
        card = self.loop.plan
        if nd:
            card.status.setText(f"Newest: {newest.stem}, level {nd.get('level')} m/s², weakest parameter {nd.get('weakest')}, "
                                f"worst member rise {nd.get('worst_rise', 0):+.2f} m.")
            card.set_state('READY', GOOD)
        else:
            card.status.setText('No trajectory yet.')
            card.set_state('TO DO', PALETTE['grey'])
        flights = flown(self.setup)
        planned = {p.stem for p in PLANS.glob('*.json')} if PLANS.exists() else set()
        flown_names = {(read_json(f/'trajectory.json') or {}).get('name') for f in flights[:30]}
        self.loop.fly.status.setText(f'{len(flights)} trajectory flights of {self.setup}; newest: '
                                     f'{flights[0].relative_to(FLIGHT_STACK/"calibration_data") if flights else "—"}.')
        if newest is not None and newest.stem in flown_names:
            self.loop.fly.set_state('FLOWN', GOOD)
        elif newest is not None:
            self.loop.fly.set_state('COPIED' if newest.stem in planned else 'TO FLY', WARN)
        self.loop.measure.status.setText('Clean up to ' + (f'{clean:g} m/s²' if clean is not None else '—') +
                                         (f"; {sum(r['failed'] for r in env['summary'])} failed flight(s) at some level." if env else '.'))
        self.loop.measure.set_state('UP TO DATE' if env and flights and (FLIGHT_STACK/'calibration_data'/'envelope'/f'{self.setup}.json').stat().st_mtime
                                    >= flights[0].stat().st_mtime else ('TO DO' if flights else ''), GOOD if env else WARN)
        labelled = bool(flights) and (flights[0]/'cable_labels.npz').exists()
        self.loop.fit.status.setText(f'Newest flight: {flights[0].name if flights else "—"}' +
                                     (', cable labelled.' if labelled else (', cable not labelled yet.' if flights else '.')))
        if flights:
            self.loop.fit.set_state('LABELLED' if labelled else 'TO LABEL', GOOD if labelled else WARN)
        self.select()

    def selected_path(self):
        i = self.loop.selected.currentIndex()
        return self.paths[i] if 0 <= i < len(self.paths) else None

    def select(self, *_):
        path = self.selected_path()
        data = read_json(path) if path else None
        report = read_json(path.with_name(path.stem+'_report.json')) if path else None
        self.trajectory.show(data)
        self.information.show(data, report)
        self.safety.show(data, report)
        self.loop.command.setPlainText(self.flight_command(path) if path else '')

    def poll(self):
        flying = flight_running()
        self.loop.flying.setText('FLIGHT RUNNING: computing locked' if flying else 'no flight running')
        self.loop.flying.setStyleSheet(f"color: {BAD if flying else PALETTE['grey']}; font-weight: 600;")
        busy = self.job is not None and self.job.running
        for b in (self.loop.optimise, self.loop.envelope_button, self.loop.label_button):
            b.setEnabled(not flying and not busy)
            b.setToolTip('A flight script is running: no computing during flights (it once stalled the host and dropped the drone)'
                         if flying else ('A job is running' if busy else ''))
        self.stop_button.setEnabled(busy)
        if self.job is not None:
            state = 'running' if busy else f'finished (exit {self.job.exit_code})'
            self.job_label.setText(f'{self.job.name}: {state}   {self.job.log}')
            text = self.job.output()
            if text != self.log.toPlainText():
                self.log.setPlainText(text)
                self.log.verticalScrollBar().setValue(self.log.verticalScrollBar().maximum())
            if busy and self.job.name.startswith('optimise'):
                self.loop.plan.set_state('RUNNING', PALETTE['blue'])
        if self.was_running and not busy:
            self.refresh()
        self.was_running = busy

    # actions
    def run(self, name, command, cwd=ROOT, env=''):
        if self.job is not None and self.job.running:
            self.statusBar().showMessage(f'{self.job.name} is still running')
            return
        if flight_running():
            self.statusBar().showMessage('A flight script is running: not starting computation')
            return
        job = Job.start(name, command, cwd, env)
        if job is None:
            self.statusBar().showMessage(f'Could not start {name}')
            return
        self.job, self.was_running = job, True
        self.tabs.setCurrentIndex(self.tabs.count()-1)
        self.poll()

    def optimise(self):
        name = next_name()
        command = [PYTHON, 'scripts/optimize_trajectory.py', '--setup', self.setup, '--duration', str(self.loop.duration.value()),
                   '--output', PLANNING/f'{name}.json', '--report', PLANNING/f'{name}_report.json']
        if self.loop.level.value() > 0:
            command += ['--level', f'{self.loop.level.value():g}']
        self.run(f'optimise-{name}', command, env='OMP_NUM_THREADS=1')

    def flight_command(self, path):
        return (f'python3 {CONTAINER}/scripts/acceleration_circle_cf2.py --trajectory {CONTAINER}/calibration_data/plans/{path.name} --check\n'
                f'python3 {CONTAINER}/scripts/acceleration_circle_cf2.py --trajectory {CONTAINER}/calibration_data/plans/{path.name}')

    def copy_command(self):
        path = self.selected_path()
        if path is None:
            return
        PLANS.mkdir(parents=True, exist_ok=True)
        target = PLANS/path.name
        if not target.exists() or target.read_bytes() != path.read_bytes():
            shutil.copyfile(path, target)
        fly = self.flight_command(path).splitlines()[-1]
        QtWidgets.QApplication.clipboard().setText(fly)
        self.statusBar().showMessage(f'Copied {path.name} to {PLANS}; flight command on the clipboard')
        self.refresh()

    def update_envelope(self):
        self.run('envelope', ['python3', 'scripts/envelope.py', '--setup', self.setup], cwd=FLIGHT_STACK)

    def label_newest(self):
        flights = flown(self.setup)
        if not flights:
            self.statusBar().showMessage('No trajectory flight to label yet')
            return
        self.run('label', [PYTHON, 'scripts/label_sequence.py', flights[0]])

    def open_model_fit(self):
        QtCore.QProcess.startDetached(str(PYTHON), ['-m', 'aerowhip.ui.fitting'], str(ROOT))

    def stop(self):
        if self.job is not None and self.job.running:
            self.job.stop()
            self.poll()


def main():
    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication(sys.argv)
    window = LoopWindow()
    window.resize(1550, 980)
    window.show()
    return app.exec()


if __name__ == '__main__':
    sys.exit(main())
