"""AeroWhip Model Fit: run the model identification stages, watch a fit converge, compare a result
with the configuration and adopt it.

    python -m aerowhip.ui.fitting

Top: the activity bar, always visible: the fit running now (stage, step, progress, time left), what
is queued next, or the last run. Left: the pipeline (thrust map → drone → cable → coupled validation),
each stage with its state (RUNNING, QUEUED, result to review, in use, not run) and one action (Run,
Queue, Cancel, Stop). Right: Live (the running fit, or the last one), Result (before/after, per-flight
chart and the explicit Adopt), Data (the calibration manifest) and Runs (every result, old layout
included). Fits run one at a time as their own processes (`aerowhip.fitting.stages`), not as
children of the window: output in <result>.log, exit code in <result>.exit; the window finds a
running fit from its `.progress.jsonl` (process id), so it can be closed and reopened at any time,
and fits started from a terminal show the same way.
"""
from datetime import datetime
import json
import math
import os
import shlex
from pathlib import Path
import subprocess
import sys
import time

import numpy as np
from PySide6 import QtCore, QtGui, QtWidgets

from aerowhip.fitting import calibration_set, progress as progress_file, stages
from aerowhip.paths import ROOT
from aerowhip.ui import style
from aerowhip.ui.plotting import PlotCanvas
from aerowhip.ui.style import PALETTE, ROLE

CHECKER = calibration_set.DEFAULT_ROOT.parent/'scripts/check_calibration_data.py'
ADOPTIONS = stages.FITS/'adoptions.jsonl'
# Older results have no progress file; their parameter names come from here.
LEGACY_PARAMETERS = {'cable': ['log10 EI', 'log10 Cb', 'Cd', 'log10 linear drag']}
GOOD, BAD = PALETTE['green'], PALETTE['vermillion']
IN_USE = 1e-4            # relative difference below which a fitted value counts as the configured one


# ---------------------------------------------------------------- data

def display_unit(unit):
    """(factor, label) to show a loss of this unit."""
    return {'m': (1e3, 'mm'), 'log ratio': (100., '%')}.get(unit, (1., unit or ''))


def fmt(v, digits=4):
    if v is None or (isinstance(v, float) and not math.isfinite(v)):
        return '—'
    if isinstance(v, bool):
        return 'yes' if v else 'no'
    if isinstance(v, (int, np.integer)):
        return str(v)
    if isinstance(v, (list, tuple)):
        return ', '.join(fmt(x, digits) for x in v)
    if isinstance(v, str):
        return v
    return f'{v:.{digits}g}'


def all_results():
    """[(path, stage)] of every result file, newest first: runs/fits/<stage>/ and the older
    runs/calibration_*/ folders."""
    found = []
    for path in [*stages.FITS.glob('*/*.json'), *ROOT.glob('runs/calibration_*/*.json')]:
        stage = stages.stage_of(path)
        if stage is not None:
            found.append((path, stage))
    return sorted(found, key=lambda item: item[0].stat().st_mtime, reverse=True)


def adoptions():
    if not ADOPTIONS.exists():
        return []
    return [json.loads(line) for line in ADOPTIONS.read_text().splitlines() if line.strip()]


def load_result(path):
    return json.loads(Path(path).read_text())


def events_for(result_path, result=None):
    """Progress events of a run: its progress file, or rebuilt from the search record of an older
    result (evaluations and per-iteration best)."""
    path = progress_file.progress_path(result_path)
    if path.exists():
        return progress_file.read(path)
    result = result if result is not None else load_result(result_path)
    stage = stages.stage_of(result_path)
    search = result.get('search', result)
    history = search.get('history', [])
    evaluations = search.get('evaluations', [])
    if not history and not evaluations:
        return []
    x0 = evaluations[0][0] if evaluations else history[0][2]
    names = result.get('parameters') or LEGACY_PARAMETERS.get(stage.key) or [f'x{i}' for i in range(len(x0))]
    seconds = result.get('seconds', result.get('search', {}).get('seconds'))
    events = [dict(kind='start', stage=stage.key, parameters=names, x0=x0, unit=stage.unit, legacy=True)]
    if seconds is not None:
        events[0]['time'] = 0.
    events += [dict(kind='evaluation', n=i+1, x=x, value=v) for i, (x, v) in enumerate(evaluations)]
    events += [dict(kind='iteration', n=n, best=b, x=x) for n, b, x in history]
    events.append(dict(kind='done', result=str(result_path)))
    if seconds is not None:
        events[-1]['time'] = float(seconds)
    return events


def stage_status(stage, results):
    """('in use' | 'pending' | 'none' | 'validated', newest result path or None, summary or None).
    'in use': the newest result's fitted values equal the configuration now."""
    mine = [p for p, s in results if s is stage]
    if not mine:
        return 'none', None, None
    path = mine[0]
    try:
        summary = stage.summary(load_result(path))
    except Exception:
        return 'pending', path, None
    if not stage.adoptable:
        return 'validated', path, summary
    return ('in use' if in_use(stage, summary) else 'pending'), path, summary


def in_use(stage, summary):
    """The result's fitted values are the values in the configuration now."""
    current = stage.current()
    return bool(summary['parameters']) and all(
        after is not None and math.isclose(after, current[name], rel_tol=IN_USE, abs_tol=1e-9)
        for name, _, after, *_ in summary['parameters'])


def when(path):
    return datetime.fromtimestamp(Path(path).stat().st_mtime).strftime('%b %d %H:%M')


# ---------------------------------------------------------------- small widgets

def label(text='', name=None, wrap=False):
    w = QtWidgets.QLabel(text)
    if name:
        w.setObjectName(name)
    w.setWordWrap(wrap)
    return w


def table(columns, stretch=None):
    t = QtWidgets.QTableWidget(0, len(columns))
    t.setHorizontalHeaderLabels(columns)
    t.verticalHeader().setVisible(False)
    t.setEditTriggers(QtWidgets.QAbstractItemView.EditTrigger.NoEditTriggers)
    t.setSelectionMode(QtWidgets.QAbstractItemView.SelectionMode.NoSelection)
    t.setFocusPolicy(QtCore.Qt.FocusPolicy.NoFocus)
    t.setShowGrid(False)
    t.setSizeAdjustPolicy(QtWidgets.QAbstractScrollArea.SizeAdjustPolicy.AdjustToContents)
    t.setHorizontalScrollBarPolicy(QtCore.Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
    t.setVerticalScrollBarPolicy(QtCore.Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
    t.horizontalHeader().setSectionResizeMode(QtWidgets.QHeaderView.ResizeMode.ResizeToContents)
    t.horizontalHeader().setDefaultAlignment(QtCore.Qt.AlignmentFlag.AlignLeft)
    if stretch is not None:
        t.horizontalHeader().setSectionResizeMode(stretch, QtWidgets.QHeaderView.ResizeMode.Stretch)
    return t


def fill(t, rows, colours=None, tooltips=None, maximum_height=None):
    """rows: lists of cells (numbers are formatted and right-aligned); colours: {(row, col): hex}."""
    t.setRowCount(len(rows))
    for i, row in enumerate(rows):
        for j, v in enumerate(row):
            item = QtWidgets.QTableWidgetItem(fmt(v))
            if isinstance(v, (int, float)) and not isinstance(v, bool):
                item.setTextAlignment(QtCore.Qt.AlignmentFlag.AlignRight | QtCore.Qt.AlignmentFlag.AlignVCenter)
            if colours and (i, j) in colours:
                item.setForeground(QtGui.QColor(colours[(i, j)]))
            if tooltips and i in tooltips:
                item.setToolTip(tooltips[i])
            t.setItem(i, j, item)
    height = t.horizontalHeader().sizeHint().height()+sum(t.rowHeight(i) for i in range(len(rows)))+6
    t.setFixedHeight(min(height, maximum_height) if maximum_height else height)


def change(before, after):
    """Relative change (%) and its colour (an error going down is good)."""
    if before in (None, 0) or after is None or not all(map(math.isfinite, (before, after))):
        return None, None
    c = (after-before)/abs(before)*100
    return round(c, 1), (GOOD if c < -.5 else BAD if c > .5 else None)


def smooth(x, y, points=400):
    """Shape-preserving cubic (PCHIP, Fritsch–Carlson) through (x, y): smooth, no overshoot, so a
    best-so-far curve stays monotone."""
    x, y = np.asarray(x, float), np.asarray(y, float)
    keep = np.isfinite(y)
    x, y = x[keep], y[keep]
    if len(x) < 3:
        return x, y
    h, d = np.diff(x), np.diff(y)/np.diff(x)
    m = np.zeros_like(y)
    same = d[:-1]*d[1:] > 0
    w1, w2 = 2*h[1:]+h[:-1], h[1:]+2*h[:-1]
    m[1:-1][same] = (w1+w2)[same]/(w1[same]/d[1:][same]+w2[same]/d[:-1][same])
    m[0], m[-1] = d[0], d[-1]
    xs = np.linspace(x[0], x[-1], points)
    k = np.clip(np.searchsorted(x, xs)-1, 0, len(x)-2)
    t = (xs-x[k])/h[k]
    h00, h10, h01, h11 = 2*t**3-3*t**2+1, t**3-2*t**2+t, -2*t**3+3*t**2, t**3-t**2
    return xs, h00*y[k]+h10*h[k]*m[k]+h01*y[k+1]+h11*h[k]*m[k+1]


def improvements(values):
    """Indices of the first value and of every value below all before it."""
    keep = [0]
    for i in range(1, len(values)):
        if values[i] < values[keep[-1]]:
            keep.append(i)
    return keep


def status_chip(text, colour):
    return f'<span style="color:{colour}; font-weight:600">● {text}</span>' if colour else f'<span>○ {text}</span>'


# ---------------------------------------------------------------- runs on disk

def pid_alive(pid):
    try:
        os.kill(int(pid), 0)
        return True
    except ProcessLookupError:
        return False
    except PermissionError:
        return True


def run_state(path, events=None):
    """'running' | 'done' | 'failed' | 'stopped' for a progress file. A run is running while the
    process that writes it is alive; its wrapper's `.exit` file (runs started from the window)
    tells a stop (terminated) from a failure."""
    path = Path(path)
    events = progress_file.read(path) if events is None else events
    if events and events[-1]['kind'] == 'done':
        return 'done'
    if events and events[-1]['kind'] == 'failed':
        return 'failed'
    exit_file = path.with_name(path.name.replace('.progress.jsonl', '.exit'))
    if exit_file.exists():
        code = exit_file.read_text().strip()
        return 'stopped' if code in ('143', '137', '-15', '-9') else 'failed' if code != '0' else 'done'
    pid = events[0].get('pid') if events else None
    if pid is not None:
        return 'running' if pid_alive(pid) else 'stopped'
    # older runs record no process id: look for a process whose command line names this run's result
    result = path.name.replace('.progress.jsonl', '.json')
    found = subprocess.run(['pgrep', '-f', f'--output [^ ]*{result}'], capture_output=True).returncode == 0
    return 'running' if found else 'stopped'


def progress_files():
    """Every progress file under runs/fits, newest first."""
    return sorted(stages.FITS.glob('*/*.progress.jsonl'), key=lambda p: p.stat().st_mtime, reverse=True)


def stage_of_progress(path):
    return stages.stage_of(Path(str(path).replace('.progress.jsonl', '.json')))


# ---------------------------------------------------------------- pipeline

STATUS_TEXT = {'in use': ('in use', GOOD), 'pending': ('result to review', PALETTE['orange']),
               'none': ('not run', None), 'validated': ('validated', GOOD)}


class StageCard(QtWidgets.QFrame):
    action_clicked = QtCore.Signal(object, str)        # stage, 'run' | 'stop' | 'cancel'
    open_clicked = QtCore.Signal(object)

    def __init__(self, number, stage):
        super().__init__()
        self.stage, self.latest, self.action = stage, None, 'run'
        self.setObjectName('card')
        self.setToolTip(f'{stage.description}\nData: {stage.data}')
        layout = QtWidgets.QVBoxLayout(self)
        layout.setContentsMargins(12, 9, 12, 9)
        layout.setSpacing(3)
        top = QtWidgets.QHBoxLayout()
        top.addWidget(label(f'<span style="font-size:14px"><b>{number}&nbsp;&nbsp;{stage.title}</b></span>'))
        top.addStretch()
        self.status = label('')
        top.addWidget(self.status)
        layout.addLayout(top)
        self.bar = QtWidgets.QProgressBar()
        self.bar.setRange(0, 1000)
        self.bar.setTextVisible(False)
        self.bar.setFixedHeight(6)
        self.bar.setVisible(False)
        layout.addWidget(self.bar)
        self.latest_label = label('', 'note', wrap=True)
        layout.addWidget(self.latest_label)
        buttons = QtWidgets.QHBoxLayout()
        self.button = QtWidgets.QPushButton('Run')
        self.button.setToolTip(' '.join(stage.command(stages.FITS/stage.key/'<time>.json')))
        self.button.clicked.connect(lambda: self.action_clicked.emit(self.stage, self.action))
        self.open = QtWidgets.QPushButton('View result')
        self.open.clicked.connect(lambda: self.open_clicked.emit(self.latest))
        buttons.addWidget(self.button)
        buttons.addWidget(self.open)
        buttons.addStretch()
        layout.addLayout(buttons)
        self.result_status = ('none', None)

    def update_from(self, status, latest, summary, suggested):
        """Result side: status of the newest result, its headline; suggested: the next step."""
        self.latest, self.result_status, self.suggested = latest, STATUS_TEXT[status], suggested
        self.open.setEnabled(latest is not None)
        text = ''
        if summary and summary['headline']:
            name, before, after = summary['headline'][0]
            text = f'{name}: ' + (f'{fmt(before, 3)} → <b>{fmt(after, 3)}</b>' if before is not None else f'<b>{fmt(after, 3)}</b>')
        if latest is not None:
            text += f'<br>last result {when(latest)}'
        self.latest_label.setText(text)

    def set_activity(self, activity, busy, fraction=None):
        """activity: 'running' | 'queued' | None for this stage; busy: another fit is running."""
        if activity == 'running':
            self.status.setText(status_chip('RUNNING' + (f' {fraction*100:.0f} %' if fraction is not None else ''), PALETTE['blue']))
            self.action, text = 'stop', 'Stop'
        elif activity == 'queued':
            self.status.setText(status_chip('QUEUED', PALETTE['sky']))
            self.action, text = 'cancel', 'Cancel'
        else:
            self.status.setText(status_chip(*self.result_status))
            self.action, text = 'run', 'Queue' if busy else 'Run'
        self.button.setText(text)
        self.bar.setVisible(activity == 'running')
        if fraction is not None:
            self.bar.setValue(int(fraction*1000))
        primary = activity is None and not busy and getattr(self, 'suggested', False)
        self.button.setObjectName('primary' if primary else '')
        self.button.style().unpolish(self.button)
        self.button.style().polish(self.button)
        self.setStyleSheet(f'QFrame#card {{ border: 2px solid {PALETTE["blue"]}; }}' if activity == 'running' else '')


class Pipeline(QtWidgets.QWidget):
    action_clicked = QtCore.Signal(object, str)
    run_all_clicked = QtCore.Signal()
    open_clicked = QtCore.Signal(object)

    def __init__(self):
        super().__init__()
        layout = QtWidgets.QVBoxLayout(self)
        layout.setContentsMargins(10, 12, 10, 10)
        layout.setSpacing(8)
        self.next = label('', 'note', wrap=True)
        layout.addWidget(self.next)
        self.run_all = QtWidgets.QPushButton('Run stages 1–3 in turn')
        self.run_all.setToolTip('Thrust map, drone dynamics and cable, one after another (each starts from the '
                                'configuration as it is; nothing is adopted)')
        self.run_all.clicked.connect(self.run_all_clicked)
        layout.addWidget(self.run_all)
        self.cards = []
        number = 0
        for stage in stages.STAGES:
            if stage.key == 'coupled':                 # the coupled fit comes before its validation
                number += 1
                layout.addWidget(self._planned(number, 'Coupled refinement', 'planned',
                    'Not built yet. A few parameters (command delay, bending damping, attachment) fitted with the drone '
                    'and the cable simulated together on the cable flights, after the drone and cable fits.'))
            number += 1
            card = StageCard(number, stage)
            card.action_clicked.connect(self.action_clicked)
            card.open_clicked.connect(self.open_clicked)
            layout.addWidget(card)
            self.cards.append(card)
        layout.addWidget(self._planned(number+1, 'Residual', 'later',
            'A small learned correction on top of the physical model, fitted once the coupled validation is settled.'))
        layout.addStretch()

    @staticmethod
    def _planned(number, title, tag, text):
        frame = QtWidgets.QFrame()
        frame.setObjectName('card')
        box = QtWidgets.QVBoxLayout(frame)
        box.setContentsMargins(12, 9, 12, 9)
        top = QtWidgets.QHBoxLayout()
        top.addWidget(label(f'<b>{number}&nbsp;&nbsp;{title}</b>', 'muted'))
        top.addStretch()
        top.addWidget(label(tag, 'muted'))
        box.addLayout(top)
        box.addWidget(label(text, 'note', wrap=True))
        frame.setEnabled(False)
        return frame

    def refresh(self, results):
        statuses = [stage_status(card.stage, results) for card in self.cards]
        suggested = next((i for i, (s, _, _) in enumerate(statuses) if s in ('none', 'pending')), len(self.cards)-1)
        for i, (card, (status, latest, summary)) in enumerate(zip(self.cards, statuses)):
            card.update_from(status, latest, summary, i == suggested)
        stage, status = self.cards[suggested].stage, statuses[suggested][0]
        self.next.setText({'none': f'Suggested next: run <b>{stage.title.lower()}</b>.',
                           'pending': f'Suggested next: review the <b>{stage.title.lower()}</b> result.'}
                          .get(status, f'Suggested next: <b>{stage.title.lower()}</b> of the adopted model.'))

    def set_activity(self, running_key=None, fraction=None, queued=()):
        for card in self.cards:
            key = card.stage.key
            card.set_activity('running' if key == running_key else 'queued' if key in queued else None,
                              running_key is not None, fraction if key == running_key else None)
        self.run_all.setEnabled(running_key is None and not queued)


# ---------------------------------------------------------------- progress

def plan_progress(events):
    """(fraction done, label of the current phase, phase index, phase count, evaluations left at
    most) from the plan and phase events; None without a plan. A phase counts in full once it is
    finished; the running phase counts its evaluations against its budget."""
    plan = next((e for e in events if e['kind'] == 'plan'), None)
    if plan is None:
        return None
    budget = plan['evaluations']
    current, finished, used = 0, set(), {}
    for e in events:
        if e['kind'] == 'phase':
            current = e['index']
            finished.update(range(current))
        elif e['kind'] == 'phase_done':
            finished.add(e['index'])
        elif e['kind'] == 'evaluation' and e.get('phase') is not None:
            used[e['phase']] = used.get(e['phase'], 0)+1
        elif e['kind'] == 'done':
            finished.update(range(len(budget)))
    done = sum(budget[i] for i in finished)
    if current not in finished:
        done += min(used.get(current, 0), budget[current])
    total = sum(budget)
    return done/total, plan['phases'][current], current, len(budget), total-done


PARAMETER_WORDS = {                      # progress parameter name -> (plain words with unit, conversion)
    'inertia_scale_xy': ('roll/pitch inertia (× configured)', lambda v: v),
    'arm_scale': ('motor arm (× configured)', lambda v: v),
    'motor_response_s': ('motor response time (ms)', lambda v: v*1e3),
    'command_delay_s': ('command delay (ms)', lambda v: v*1e3),
    'body_drag_xy_s_inv': ('air drag, horizontal (1/s)', lambda v: v),
    'body_drag_z_s_inv': ('air drag, vertical (1/s)', lambda v: v),
    'log10 EI': ('bending stiffness EI (N m²)', lambda v: 10**v),
    'log10 Cb': ('bending damping Cb (N m² s)', lambda v: 10**v),
    'Cd': ('normal drag coefficient Cd', lambda v: v),
    'log10 linear drag': ('linear air drag (1/s)', lambda v: 10**v),
    'kappa': ('thrust scale κ', lambda v: v),
    'exponent p': ('thrust exponent p', lambda v: v),
    'voltage slope beta': ('voltage slope β (1/V)', lambda v: v),
}

PHASE_WORDS = {'loading flights': 'loading the flights', 'start point': 'scoring the configuration now',
               'least squares': 'least-squares search', 'final check': 'final check (scored twice)'}


def describe_progress(events, state='running'):
    """(fraction done or None, 'step 4 of 10: fit with delay 10 ms · 23 % · at most 15 min left')."""
    p = plan_progress(events)
    if p is None:
        return None, ''
    fraction, phase, index, count, left = p
    words = PHASE_WORDS.get(phase, f'fit with {phase}' if phase.startswith('delay') else phase)
    if state == 'done':
        return 1., 'complete'
    if state != 'running':
        return fraction, f'stopped at step {index+1} of {count} ({words})'
    times = [e['time'] for e in events if e['kind'] == 'evaluation'][-30:]
    eta = ''
    if len(times) >= 3:
        minutes = left*(times[-1]-times[0])/(len(times)-1)/60
        eta = f' · at most {minutes:.0f} min left' if minutes >= 1 else ' · under a minute left'
    return fraction, f'step {index+1} of {count}: {words} · {fraction*100:.0f} %{eta}'


class ProgressView(QtWidgets.QWidget):
    stop_clicked = QtCore.Signal()

    def __init__(self):
        super().__init__()
        layout = QtWidgets.QVBoxLayout(self)
        layout.setContentsMargins(14, 12, 14, 12)
        layout.setSpacing(6)
        top = QtWidgets.QHBoxLayout()
        self.title = label('No run selected', 'heading')
        top.addWidget(self.title)
        top.addStretch()
        self.output_toggle = QtWidgets.QPushButton('Show output')
        self.output_toggle.setCheckable(True)
        self.output_toggle.toggled.connect(self._toggle_output)
        top.addWidget(self.output_toggle)
        self.stop = QtWidgets.QPushButton('Stop')
        self.stop.setVisible(False)
        self.stop.clicked.connect(self.stop_clicked)
        top.addWidget(self.stop)
        layout.addLayout(top)
        self.summary = label('', wrap=True)
        layout.addWidget(self.summary)
        bar = QtWidgets.QHBoxLayout()
        self.bar = QtWidgets.QProgressBar()
        self.bar.setRange(0, 1000)
        self.bar.setTextVisible(False)
        self.bar.setFixedHeight(10)
        bar.addWidget(self.bar, 1)
        self.bar_label = label('', 'note')
        bar.addWidget(self.bar_label)
        layout.addLayout(bar)
        self.loss = PlotCanvas(minimum=(300, 260))
        layout.addWidget(self.loss, 5)
        self.table_title = label('Parameters: value at the start and in the best fit so far', 'section')
        layout.addWidget(self.table_title)
        self.table = table(['', 'start', 'best so far', 'change'])
        layout.addWidget(self.table)
        self.console = QtWidgets.QPlainTextEdit()
        self.console.setReadOnly(True)
        self.console.setMaximumBlockCount(20000)
        self.console.setFont(QtGui.QFontDatabase.systemFont(QtGui.QFontDatabase.SystemFont.FixedFont))
        self.console.setPlaceholderText('Output of a fit started from this window.')
        self.console.setVisible(False)
        layout.addWidget(self.console, 3)
        self.source = label('', 'note', wrap=True)
        self.source.setTextInteractionFlags(QtCore.Qt.TextInteractionFlag.TextSelectableByMouse)
        layout.addWidget(self.source)
        self.signature = None

    def _toggle_output(self, shown):
        self.console.setVisible(shown)
        self.output_toggle.setText('Hide output' if shown else 'Show output')

    def show_events(self, events, *, source='', state='running', title=None):
        """state: 'starting' | 'running' | 'done' | 'failed' | 'stopped' (see run_state)."""
        start = next((e for e in events if e['kind'] == 'start'), None)
        if start is None:
            self.title.setText(title or ('Starting…' if state in ('starting', 'running') else 'Nothing has run yet'))
            self.summary.setText(status_chip('starting: loading the program', PALETTE['blue']) if state == 'starting' else '')
            self.bar.setRange(0, 0 if state == 'starting' else 1000)
            self.bar.setVisible(state == 'starting')
            self.bar_label.setText('')
            self.source.setText(source)
            self.loss.reset()
            self.loss.draw_idle()
            self.table.setVisible(False)
            self.table_title.setVisible(False)
            self.signature = None
            return
        stage = stages.BY_KEY.get(start['stage']) or next((s for s in stages.STAGES if s.title.lower() == start['stage']), None)
        name = stage.title if stage else start['stage']
        self.title.setText(title or (f'Running now: {name}' if state == 'running' else f'Last run: {name}'))
        self._summary(events, start, state)
        self._bar(events, state)
        signature = (source, len(events))
        if signature == self.signature:
            return                                      # only the clock moved
        self.signature = signature
        self.source.setText(source+('  (rebuilt from the result file)' if start.get('legacy') else ''))
        flights = [e for e in events if e['kind'] == 'flight']
        self.state = state
        if flights:
            self._flights(flights)
        else:
            self._search(events, start)

    def _summary(self, events, start, state):
        last = events[-1]
        evaluations = [e for e in events if e['kind'] == 'evaluation']
        flights = sum(e['kind'] == 'flight' for e in events)
        factor, unit = display_unit(start.get('unit'))
        chip = {'done': status_chip('finished', GOOD), 'running': status_chip('running', PALETTE['blue']),
                'failed': status_chip(f'failed{": "+last["error"] if last.get("error") else " (see Show output)"}', BAD),
                'stopped': status_chip('stopped before the end (no result written)', BAD)}[state]
        parts = [chip]
        start_value = next((e['value'] for e in events if e['kind'] == 'phase_done' and e['value'] is not None), None)
        values = [e['value'] for e in evaluations if e['value'] is not None]
        if values:
            first = start_value if start_value is not None else values[0]
            best = min(values)
            c, colour = change(first, best)
            parts.append(f'best <b style="font-size:16px">{best*factor:.4g} {unit}</b>' +
                         (f' <span style="color:{colour or "inherit"}">({c:+.1f} % from {first*factor:.4g})</span>' if c is not None else ''))
        elif start_value is not None:
            parts.append(f'start {start_value*factor:.4g} {unit}')
        if evaluations:
            rejected = sum(1 for e in evaluations if e['value'] is None)
            parts.append(f'{len(evaluations)} evaluations' + (f', <span style="color:{BAD}">{rejected} with failed segments</span>' if rejected else ''))
        if flights:
            parts.append(f'{flights} flights scored')
        if 'time' in start:
            seconds = (time.time() if state == 'running' else last.get('time', start['time']))-start['time']
            parts.append(f'{seconds:.0f} s' if seconds < 60 else f'{seconds/60:.1f} min')
        self.summary.setText('&nbsp;&nbsp;·&nbsp;&nbsp;'.join(parts))

    def _bar(self, events, state):
        p = plan_progress(events)
        if p is None:
            self.bar.setVisible(state == 'running')
            self.bar.setRange(0, 0 if state == 'running' else 1000)      # busy indicator without a plan
            self.bar_label.setText('')
            return
        fraction, text = describe_progress(events, state)
        self.bar.setVisible(True)
        self.bar.setRange(0, 1000)
        self.bar.setValue(int(1000*fraction))
        self.bar_label.setText(text)

    def _search(self, events, start):
        """Error over time (best so far), the delay profile for the drone, and the parameter table."""
        factor, unit = display_unit(start.get('unit'))
        evaluations = [e for e in events if e['kind'] == 'evaluation' and e['value'] is not None]
        if not evaluations:                                 # older results: per-iteration best only
            evaluations = [dict(n=e['n'], value=e['best'], x=e['x']) for e in events if e['kind'] == 'iteration']
        start_value = next((e['value'] for e in events if e['kind'] == 'phase_done' and e['value'] is not None),
                           evaluations[0]['value'] if evaluations else None)
        plan = next((e for e in events if e['kind'] == 'plan'), None)
        delays = [p for p in plan['phases'] if p.startswith('delay')] if plan else []
        canvas = self.loss
        canvas.reset()
        with canvas.style():
            if delays:
                grid = canvas.figure.add_gridspec(1, 2, width_ratios=(2.2, 1))
                ax, side = canvas.figure.add_subplot(grid[0]), canvas.figure.add_subplot(grid[1])
            else:
                ax, side = canvas.figure.add_subplot(111), None
            if evaluations:
                timed = 'time' in evaluations[0] and 'time' in start
                t = np.array([(e['time']-start['time'])/60 if timed else e['n'] for e in evaluations])
                v = np.array([e['value'] for e in evaluations])*factor
                best = np.minimum.accumulate(v)
                keep = sorted(set(improvements(list(best))+[len(best)-1]))
                if timed and start_value is not None:      # the start point is scored before the search
                    t, best, keep = np.r_[0., t], np.r_[start_value*factor, best], [0]+[k+1 for k in keep]
                xs, ys = smooth(t[keep], best[keep])
                ax.plot(xs, ys, color=ROLE['model'], lw=2.4)
                ax.plot(t[keep[-1]], best[keep[-1]], 'o', color=ROLE['model'], ms=7)
                ax.annotate(f'best {best[-1]:.3g} {unit}', (t[keep[-1]], best[-1]), xytext=(-8, 10),
                            textcoords='offset points', ha='right', fontsize=9, color=ROLE['model'], fontweight='bold')
                if start_value is not None:
                    ax.axhline(start_value*factor, color=ROLE['reference'], ls='--', lw=1)
                    ax.annotate(f'start {start_value*factor:.3g} {unit}', (t[0], start_value*factor), xytext=(4, 4),
                                textcoords='offset points', fontsize=8.5, color=ROLE['reference'])
                ax.set_xlabel('minutes since the start' if timed else 'evaluation')
                span = max(best.max()-best.min(), 1e-9)
                ax.set_ylim(best.min()-.15*span, best.max()+.15*span)
            ax.set_ylabel(f'error ({unit})' if unit else 'error')
            ax.set_title('Model error on the flights, best so far (lower is better)')
            ax.grid(alpha=.3)
            if side is not None:
                self._delay_profile(side, events, delays, factor, unit, running=self.state == 'running')
            canvas.figure.tight_layout()
        canvas.draw_idle()
        self._parameter_table(start, evaluations)

    @staticmethod
    def _delay_profile(ax, events, delays, factor, unit, running=True):
        """Error at each command delay of the plan: finished ones filled, the one in progress open,
        the best circled; delays not tried yet are only ticks."""
        done = {e['label']: e['value'] for e in events if e['kind'] == 'phase_done' and e['label'].startswith('delay')}
        current = max((e for e in events if e['kind'] == 'phase'), key=lambda e: e['time'], default=None)
        ms = [float(d.split()[1]) for d in delays]
        xs = [m for d, m in zip(delays, ms) if done.get(d) is not None]
        ys = [done[d]*factor for d in delays if done.get(d) is not None]
        if xs:
            ax.plot(xs, ys, '-', color=ROLE['model'], lw=1.2, alpha=.6)
            ax.plot(xs, ys, 'o', color=ROLE['model'], ms=6)
            b = int(np.argmin(ys))
            ax.plot(xs[b], ys[b], 'o', ms=13, mfc='none', mew=1.6, color=GOOD)
            ax.annotate(f'best so far: {xs[b]:.0f} ms', (xs[b], ys[b]), xytext=(0, -16), textcoords='offset points',
                        ha='center', fontsize=8.5, color=GOOD)
        if current is not None and current['label'] in delays and current['label'] not in done and events[-1]['kind'] != 'done':
            vals = [e['value'] for e in events if e['kind'] == 'evaluation' and e.get('phase') == current['index']
                    and e['value'] is not None]
            if vals:
                m = float(current['label'].split()[1])
                ax.plot(m, min(vals)*factor, 'o', ms=6, mfc='none', color=ROLE['model'])
                ax.annotate('in progress' if running else 'stopped here', (m, min(vals)*factor), xytext=(0, 8), textcoords='offset points',
                            ha='center', fontsize=8, color=ROLE['model'])
        ax.set_xticks(ms)
        ax.set_xlim(min(ms)-5, max(ms)+5)
        ax.set_xlabel('command delay tried (ms)')
        ax.set_ylabel(f'error ({unit})')
        ax.set_title('Which command delay fits best?')
        ax.grid(alpha=.3)

    def _parameter_table(self, start, evaluations):
        names = start.get('parameters') or []
        if not names or not evaluations:
            self.table.setRowCount(0)
            self.table_title.setVisible(False)
            self.table.setVisible(False)
            return
        best = min(evaluations, key=lambda e: e['value'])['x']
        rows, colours = [], {}
        for i, name in enumerate(names):
            text, convert = PARAMETER_WORDS.get(name, (name.replace('_', ' '), lambda v: v))
            before = convert(start['x0'][i]) if i < len(start['x0']) else None
            after = convert(best[i]) if i < len(best) else None
            c = None if before in (None, 0) or after is None else (after-before)/abs(before)*100
            rows.append([text, before, after, ('new' if before == 0 and after else '') if c is None
                         else f'{c:+.0f} %' if abs(c) >= 1 else 'same'])
            if (c is not None and abs(c) >= 1) or (before == 0 and after):
                colours[(i, 3)] = PALETTE['orange']
        fill(self.table, rows, colours)
        self.table_title.setVisible(True)
        self.table.setVisible(True)

    def _flights(self, flights):
        canvas = self.loss
        canvas.reset()
        with canvas.style():
            ax = canvas.figure.add_subplot(111)
            y = np.arange(len(flights))
            for k, (key, colour, text) in enumerate((('drone_debiased_rms_m', ROLE['model'], 'drone, debiased'),
                                                     ('cable_shape_rms_m', ROLE['command'], 'cable shape'))):
                ax.barh(y+(k-.5)*.38, [(e.get(key) or np.nan)*1e3 for e in flights], height=.36, color=colour, label=text)
            ax.set_yticks(y, [e['name'] for e in flights])
            ax.invert_yaxis()
            ax.set_xlabel('RMS error over 2 s windows (mm)')
            ax.set_title('Per flight')
            ax.legend(loc='lower right', frameon=False)
            canvas.figure.tight_layout()
        canvas.draw_idle()
        self.table.setVisible(False)
        self.table_title.setVisible(False)


# ---------------------------------------------------------------- result

class ResultView(QtWidgets.QScrollArea):
    adopted = QtCore.Signal()

    def __init__(self):
        super().__init__()
        self.setWidgetResizable(True)
        page = QtWidgets.QWidget()
        page.setObjectName('page')
        self.setWidget(page)
        layout = QtWidgets.QVBoxLayout(page)
        layout.setContentsMargins(16, 14, 16, 14)
        layout.setSpacing(8)
        top = QtWidgets.QHBoxLayout()
        self.title = label('No result selected', 'heading')
        top.addWidget(self.title)
        top.addStretch()
        self.adopt = QtWidgets.QPushButton('Adopt…')
        self.adopt.setObjectName('primary')
        self.adopt.clicked.connect(self._adopt)
        top.addWidget(self.adopt)
        layout.addLayout(top)
        self.verdict = label('', wrap=True)
        layout.addWidget(self.verdict)
        row = QtWidgets.QHBoxLayout()
        row.setSpacing(32)
        for title, attribute, columns in (('Errors', 'headline', ['', 'before', 'after', 'change']),
                                          ('Parameters', 'parameters', ['', 'configuration now', 'this result', '95 % interval', 'change'])):
            column = QtWidgets.QVBoxLayout()
            heading = label(title, 'section')
            column.addWidget(heading)
            setattr(self, attribute, table(columns))
            column.addWidget(getattr(self, attribute))
            column.addStretch()
            row.addLayout(column)
            if attribute == 'parameters':
                self.parameter_title = heading
        row.addStretch()
        layout.addLayout(row)
        self.flights = PlotCanvas(minimum=(380, 260))
        layout.addWidget(self.flights)
        self.notes = label('', 'note', wrap=True)
        self.notes.setTextInteractionFlags(QtCore.Qt.TextInteractionFlag.TextSelectableByMouse)
        layout.addWidget(self.notes)
        layout.addStretch()
        self.stage = self.path = self.result = None
        self.adopt.setVisible(False)

    def show_result(self, path):
        self.path, self.stage = Path(path), stages.stage_of(path)
        self.title.setText(f'{self.stage.title}')
        source = f'{self.path.relative_to(ROOT) if self.path.is_relative_to(ROOT) else self.path} · {when(self.path)}'
        try:
            self.result = load_result(path)
            summary = self.stage.summary(self.result)
        except Exception as exc:
            self.verdict.setText(f'<span style="color:{BAD}">This result cannot be summarised ({type(exc).__name__}: {exc}).</span>')
            for t in (self.headline, self.parameters):
                t.setRowCount(0)
            self.flights.reset()
            self.flights.draw_idle()
            self.notes.setText(source)
            self.adopt.setVisible(False)
            return
        rows, colours = [], {}
        for i, (name, before, after) in enumerate(summary['headline']):
            c, colour = change(before, after)
            rows.append([name, before, after, f'{c:+.1f} %' if c is not None else ''])
            if colour:
                colours[(i, 3)] = colour
        fill(self.headline, rows, colours)
        current = self.stage.current()
        rows, colours, matches = [], {}, bool(summary['parameters'])
        for i, (name, _, after, *interval) in enumerate(summary['parameters']):
            now = current.get(name)
            same = after is not None and now is not None and math.isclose(after, now, rel_tol=IN_USE, abs_tol=1e-9)
            matches &= same
            text = '' if same else ('new' if not now else f'{(after-now)/abs(now)*100:+.1f} %' if after is not None else '')
            rows.append([name, now, after, interval[0] if interval else '', text])
            if not same:
                colours[(i, 4)] = PALETTE['orange']
        fill(self.parameters, rows, colours)
        self.parameters.setVisible(bool(rows))
        self.parameter_title.setVisible(bool(rows))
        self._flight_chart(summary['flights'])
        self.notes.setText('<br>'.join([*summary.get('notes', []), source]))
        adoptable = self.stage.adoptable and bool(summary['parameters'])
        self.adopt.setVisible(adoptable and not matches)
        if not self.stage.adoptable:
            self.verdict.setText('Validation of the adopted model: nothing to adopt.')
        elif matches:
            self.verdict.setText(status_chip('These values are the configuration now.', GOOD))
        else:
            self.verdict.setText(status_chip('Not adopted.', PALETTE['orange']) +
                                 f' Adopt writes the new values into {", ".join(p.name for p in self.stage.writes)}.')

    def _flight_chart(self, flights):
        """Per-flight error before (grey) and after (blue): the first before/after column pair."""
        canvas = self.flights
        canvas.reset()
        columns, rows = flights['columns'], flights['rows']
        pair = next((j for j in range(1, len(columns)-1) if 'before' in columns[j] and 'after' in columns[j+1]), None)
        with canvas.style():
            ax = canvas.figure.add_subplot(111)
            if pair is not None and rows:
                y = np.arange(len(rows))
                before = np.array([np.nan if r[pair] is None else r[pair] for r in rows], float)
                after = np.array([np.nan if r[pair+1] is None else r[pair+1] for r in rows], float)
                ax.hlines(y, np.fmin(before, after), np.fmax(before, after), color=ROLE['band'], lw=2)
                ax.plot(before, y, 'o', color=ROLE['reference'], ms=6, label='before')
                ax.plot(after, y, 'o', color=ROLE['model'], ms=6, label='after')
                ax.set_yticks(y, [str(r[0]) for r in rows], fontsize=7.5)
                ax.invert_yaxis()
                ax.set_title(f'Per flight: {columns[pair].replace(" before", "")}')
                ax.legend(loc='lower right', frameon=False)
                ax.grid(axis='x', alpha=.3)
            elif rows:                                   # validation: one value per flight
                y = np.arange(len(rows))
                for k, (j, colour) in enumerate(((2, ROLE['model']), (3, ROLE['command']))):
                    ax.barh(y+(k-.5)*.38, [np.nan if r[j] is None else r[j] for r in rows], height=.36, color=colour, label=columns[j])
                ax.set_yticks(y, [str(r[0]) for r in rows], fontsize=7.5)
                ax.invert_yaxis()
                ax.set_title('Per flight (mm)')
                ax.legend(loc='lower right', frameon=False)
            canvas.figure.set_figheight(max(2.6, .22*len(rows)+1.2))
            canvas.setMinimumHeight(int(canvas.figure.get_figheight()*canvas.figure.dpi))
            canvas.figure.tight_layout()
        canvas.draw_idle()

    def _adopt(self):
        try:
            preview = [f'{name}: {fmt(now)} → {fmt(after)}' for (name, _, after, *_), now
                       in zip(self.stage.summary(self.result)['parameters'], self.stage.current().values())]
        except Exception as exc:
            QtWidgets.QMessageBox.warning(self, 'Cannot adopt', str(exc))
            return
        files = '\n'.join(str(p.relative_to(ROOT)) for p in self.stage.writes)
        answer = QtWidgets.QMessageBox.question(
            self, f'Adopt {self.stage.title.lower()}',
            'Write this result into the configuration?\n\n' + '\n'.join(preview) +
            f'\n\nFiles changed:\n{files}\n\nThe adoption is recorded in runs/fits/adoptions.jsonl.')
        if answer != QtWidgets.QMessageBox.StandardButton.Yes:
            return
        try:
            values = self.stage.adopt(self.result)
        except Exception as exc:
            QtWidgets.QMessageBox.warning(self, 'Adoption failed', f'{type(exc).__name__}: {exc}')
            return
        stages.record_adoption(self.stage.key, self.path, values)
        self.show_result(self.path)
        self.adopted.emit()


# ---------------------------------------------------------------- data

def pretty(name):
    """'drone_mass_kg' → ('drone mass', 'kg')."""
    units = {'kg': 'kg', 'm': 'm', 'v': 'V', 's': 's'}
    parts = name.split('_')
    unit = units.get(parts[-1])
    return ' '.join(parts[:-1] if unit else parts).capitalize(), unit or ''


class DataView(QtWidgets.QScrollArea):
    def __init__(self):
        super().__init__()
        self.setWidgetResizable(True)
        page = QtWidgets.QWidget()
        page.setObjectName('page')
        self.setWidget(page)
        layout = QtWidgets.QVBoxLayout(page)
        layout.setContentsMargins(16, 14, 16, 14)
        layout.setSpacing(8)
        top = QtWidgets.QHBoxLayout()
        top.addWidget(label('Calibration data', 'heading'))
        top.addStretch()
        self.check = QtWidgets.QPushButton('Check data')
        self.check.setToolTip(str(CHECKER))
        self.check.clicked.connect(self._check)
        top.addWidget(self.check)
        layout.addLayout(top)
        self.output = QtWidgets.QPlainTextEdit()
        self.output.setReadOnly(True)
        self.output.setFont(QtGui.QFontDatabase.systemFont(QtGui.QFontDatabase.SystemFont.FixedFont))
        self.output.setVisible(False)
        self.output.setMinimumHeight(220)
        layout.addWidget(self.output)
        row = QtWidgets.QHBoxLayout()
        row.setSpacing(24)
        left = QtWidgets.QVBoxLayout()
        left.addWidget(label('Measured', 'section'))
        self.measurements = table(['', 'value', ''])
        left.addWidget(self.measurements)
        left.addStretch()
        row.addLayout(left)
        right = QtWidgets.QVBoxLayout()
        right.addWidget(label('Flights by category', 'section'))
        self.flights = table(['setup', 'motion', 'flights', 'used for', 'cable labels'])
        right.addWidget(self.flights)
        right.addStretch()
        row.addLayout(right, 1)
        layout.addLayout(row)
        self.source = label('', 'note', wrap=True)
        self.source.setTextInteractionFlags(QtCore.Qt.TextInteractionFlag.TextSelectableByMouse)
        layout.addWidget(self.source)
        layout.addStretch()
        self.manifest = None
        self.process = None

    def refresh(self):
        try:
            self.manifest = calibration_set.manifest()
        except Exception as exc:
            self.source.setText(f'<span style="color:{BAD}">Manifest not readable: {exc}</span>')
            return
        self.source.setText(str(self.manifest.path))
        rows, colours, tips = [], {}, {}
        for name, entry in self.manifest.vehicle().items():
            if not isinstance(entry, dict) or 'value' not in entry:
                continue
            text, unit = pretty(name)
            value = entry['value']
            if value is None:
                colours[(len(rows), 1)] = BAD
            tips[len(rows)] = ' · '.join(str(entry[k]) for k in ('method', 'date', 'note') if entry.get(k))
            rows.append([text, 'missing' if value is None else value, unit])
        fill(self.measurements, rows, colours, tips)
        groups = {}
        for e in self.manifest.data['flights']:
            groups.setdefault((e['setup'], e['motion']), []).append(e)
        rows, colours, tips = [], {}, {}
        for (setup, motion), entries in sorted(groups.items()):
            roles = sorted({e['role'] for e in entries})
            cable = setup == 'cf_2' and any(e['role'] in ('fit', 'held_out') for e in entries)
            labelled = sum((self.manifest.folder(e)/'cable_labels.npz').exists() for e in entries if e['role'] in ('fit', 'held_out'))
            used = sum(e['role'] in ('fit', 'held_out') for e in entries)
            i = len(rows)
            rows.append([setup, motion, len(entries), ', '.join(roles), f'{labelled}/{used}' if cable else ''])
            if cable and labelled < used:
                colours[(i, 4)] = BAD
            if roles == ['excluded']:
                for j in range(4):
                    colours[(i, j)] = ROLE['reference']
            tips[i] = '\n'.join(f"{e['folder']}  ({e['role']})" for e in entries)
        fill(self.flights, rows, colours, tips)

    def _check(self):
        self.output.clear()
        self.output.setVisible(True)
        self.process = QtCore.QProcess(self)
        self.process.setProcessChannelMode(QtCore.QProcess.ProcessChannelMode.MergedChannels)
        self.process.readyReadStandardOutput.connect(
            lambda: self.output.appendPlainText(bytes(self.process.readAllStandardOutput()).decode(errors='replace').rstrip()))
        self.process.finished.connect(lambda code, _: self.output.appendPlainText(f'\n[exit code {code}]'))
        self.process.start(str(stages.PYTHON), [str(CHECKER)])


# ---------------------------------------------------------------- runs

class RunsView(QtWidgets.QWidget):
    open_clicked = QtCore.Signal(object)

    def __init__(self):
        super().__init__()
        layout = QtWidgets.QVBoxLayout(self)
        layout.setContentsMargins(12, 12, 12, 12)
        layout.addWidget(label('Runs', 'heading'))
        layout.addWidget(label('Every result of the four stages, newest first (runs/fits/ and the older '
                               'runs/calibration_*/ folders). Double-click to open.', 'note', wrap=True))
        self.tree = QtWidgets.QTreeWidget()
        self.tree.setHeaderLabels(['stage / file', 'time', 'result', 'state'])
        for i in range(4):
            self.tree.header().setSectionResizeMode(i, QtWidgets.QHeaderView.ResizeMode.ResizeToContents)
        self.tree.itemDoubleClicked.connect(self._open)
        layout.addWidget(self.tree)

    def refresh(self, results):
        self.tree.clear()
        log = adoptions()
        adopted = {Path(a['result']).resolve() for a in log}
        groups = {}
        for stage in stages.STAGES:
            groups[stage.key] = QtWidgets.QTreeWidgetItem([stage.title])
            groups[stage.key].setFirstColumnSpanned(False)
            self.tree.addTopLevelItem(groups[stage.key])
        for path in progress_files():                  # runs without a result: running, stopped or failed
            if Path(str(path).replace('.progress.jsonl', '.json')).exists():
                continue
            stage = stage_of_progress(path)
            if stage:
                item = QtWidgets.QTreeWidgetItem([str(path.relative_to(ROOT/'runs')).replace('.progress.jsonl', ''),
                                                  when(path), '', run_state(path)])
                item.setForeground(3, QtGui.QColor(PALETTE['blue'] if run_state(path) == 'running' else BAD))
                groups[stage.key].addChild(item)
        for path, stage in results:
            try:
                summary = stage.summary(load_result(path))
                name, before, after = summary['headline'][0]
                state = 'in use' if stage.adoptable and in_use(stage, summary) else ''
            except Exception:
                summary, name, before, after, state = None, '(cannot summarise)', None, None, ''
            if path.resolve() in adopted:
                state = ', '.join(filter(None, [state, 'adopted here']))
            text = str(path.relative_to(ROOT/'runs'))
            item = QtWidgets.QTreeWidgetItem([text, when(path), fmt(after, 3), state])
            item.setToolTip(2, f'{name}: {fmt(before, 4)} → {fmt(after, 4)}')
            item.setData(0, QtCore.Qt.ItemDataRole.UserRole, str(path))
            item.setTextAlignment(2, QtCore.Qt.AlignmentFlag.AlignRight)
            if state:
                item.setForeground(3, QtGui.QColor(GOOD))
            groups[stage.key].addChild(item)
        for g in groups.values():
            g.setText(0, f'{g.text(0)}  ({g.childCount()})')
            g.setExpanded(True)

    def _open(self, item, _column):
        path = item.data(0, QtCore.Qt.ItemDataRole.UserRole)
        if path:
            self.open_clicked.emit(Path(path))


# ---------------------------------------------------------------- window

class ActivityBar(QtWidgets.QFrame):
    """Always visible: what is running now (stage, step, progress, time left), what is queued
    next, or, when nothing runs, the last run and its outcome."""
    stop_clicked = QtCore.Signal()
    live_clicked = QtCore.Signal()
    result_clicked = QtCore.Signal()

    def __init__(self):
        super().__init__()
        self.setObjectName('card')
        layout = QtWidgets.QHBoxLayout(self)
        layout.setContentsMargins(14, 8, 14, 8)
        text = QtWidgets.QVBoxLayout()
        text.setSpacing(3)
        self.headline = label('')
        self.headline.setStyleSheet('font-size: 15px;')
        text.addWidget(self.headline)
        self.bar = QtWidgets.QProgressBar()
        self.bar.setRange(0, 1000)
        self.bar.setTextVisible(False)
        self.bar.setFixedHeight(8)
        text.addWidget(self.bar)
        self.detail = label('', 'note')
        text.addWidget(self.detail)
        layout.addLayout(text, 1)
        self.live = QtWidgets.QPushButton('Watch live')
        self.live.clicked.connect(self.live_clicked)
        self.result = QtWidgets.QPushButton('View result')
        self.result.clicked.connect(self.result_clicked)
        self.stop = QtWidgets.QPushButton('Stop')
        self.stop.setToolTip('Stop the running fit (no result is written) and empty the queue')
        self.stop.clicked.connect(self.stop_clicked)
        for b in (self.live, self.result, self.stop):
            layout.addWidget(b)

    def show_running(self, name, state, describe, queue):
        fraction, words = describe
        dot = PALETTE['blue']
        self.headline.setText(f'<span style="color:{dot}">●</span> <b>{"Starting" if state == "starting" else "Running"}: '
                              f'{name}</b>' + (f'&nbsp;&nbsp;—&nbsp;&nbsp;{words}' if words else ''))
        self.bar.setVisible(True)
        if fraction is None:
            self.bar.setRange(0, 0)
        else:
            self.bar.setRange(0, 1000)
            self.bar.setValue(int(fraction*1000))
        self.detail.setText('Queued next: '+', then '.join(queue) if queue else 'Nothing queued after it.')
        self.live.setVisible(True)
        self.stop.setVisible(True)
        self.result.setVisible(False)

    def show_idle(self, last):
        """last: (name, state, finished time text, headline text) of the newest run, or None."""
        self.headline.setText('<span style="color:#8c8c8c">○</span> <b>Nothing running.</b>')
        self.bar.setVisible(False)
        self.stop.setVisible(False)
        if last is None:
            self.detail.setText('Press Run on a stage to start a fit.')
            self.live.setVisible(False)
            self.result.setVisible(False)
            return
        name, state, at, headline = last
        verb = {'done': 'finished', 'stopped': 'was stopped', 'failed': 'failed'}.get(state, state)
        self.detail.setText(f'Last run: {name} {verb} at {at}' + (f' · {headline}' if headline else ''))
        self.live.setVisible(True)
        self.live.setText('Show last run')
        self.result.setVisible(state == 'done')

    def reset_live_text(self):
        self.live.setText('Watch live')


class ModelFitWindow(QtWidgets.QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle('AeroWhip Model Fit')
        style.apply(self)
        self.queue = []              # stages waiting their turn (fits run one at a time)
        self.launching = None        # (stage, result path, wrapper pid, start time) until its progress file appears
        self.live_path = None        # progress file shown in the Live tab
        self.was_running = None      # progress file that was running at the last poll
        self.activity = ActivityBar()
        self.pipeline = Pipeline()
        left = QtWidgets.QScrollArea()
        left.setWidgetResizable(True)
        left.setWidget(self.pipeline)
        left.setMinimumWidth(360)
        self.pipeline.setObjectName('page')
        self.progress, self.result, self.data, self.runs = ProgressView(), ResultView(), DataView(), RunsView()
        self.tabs = QtWidgets.QTabWidget()
        for widget, name in ((self.progress, 'Live'), (self.result, 'Result'), (self.data, 'Data'), (self.runs, 'Runs')):
            self.tabs.addTab(widget, name)
        split = QtWidgets.QSplitter()
        split.addWidget(left)
        split.addWidget(self.tabs)
        split.setStretchFactor(1, 1)
        split.setSizes([380, 1100])
        page = QtWidgets.QWidget()
        page.setObjectName('page')
        outer = QtWidgets.QVBoxLayout(page)
        outer.setContentsMargins(8, 8, 8, 0)
        outer.setSpacing(8)
        outer.addWidget(self.activity)
        outer.addWidget(split, 1)
        self.setCentralWidget(page)
        self.statusBar().showMessage(f'Calibration data: {calibration_set.DEFAULT_ROOT}. Fits run on their own: '
                                     'closing this window does not stop them.')
        self.pipeline.action_clicked.connect(self.action)
        self.pipeline.run_all_clicked.connect(lambda: [self.start(s) for s in stages.STAGES if s.adoptable])
        self.pipeline.open_clicked.connect(self.open)
        self.runs.open_clicked.connect(self.open)
        self.progress.stop_clicked.connect(self.stop)
        self.activity.stop_clicked.connect(self.stop)
        self.activity.live_clicked.connect(lambda: self.tabs.setCurrentWidget(self.progress))
        self.activity.result_clicked.connect(self._open_last_result)
        self.result.adopted.connect(self.refresh)
        self.tabs.currentChanged.connect(lambda _: self.tabs.currentWidget() is self.data and self.data.refresh())
        self.timer = QtCore.QTimer(self, interval=700, timeout=self.poll)
        self.timer.start()
        self.refresh()
        self.data.refresh()
        results = all_results()
        if results:
            self.result.show_result(results[0][0])
        self.poll()

    def refresh(self):
        results = all_results()
        self.pipeline.refresh(results)
        self.runs.refresh(results)

    # what is running
    def running(self):
        """(stage, progress path or None, state, events) of the fit running now, or None."""
        if self.launching is not None:
            stage, output, wrapper, t0 = self.launching
            path = progress_file.progress_path(output)
            exit_file = output.with_suffix('.exit')
            if path.exists() and progress_file.read(path):
                self.launching = None
            elif exit_file.exists() or time.time()-t0 > 300:
                self.launching = None
                self.statusBar().showMessage(f'{stage.title} did not start: see {output.with_suffix(".log").relative_to(ROOT)}')
                return None
            else:
                return stage, None, 'starting', []
        for path in progress_files():
            if time.time()-path.stat().st_mtime > 3600:       # a running fit writes at least every few minutes
                break
            events = progress_file.read(path)
            if events and run_state(path, events) == 'running':
                return stage_of_progress(path), path, 'running', events
        return None

    def poll(self):
        active = self.running()
        if active is None and self.queue:
            self._launch(self.queue.pop(0))
            active = self.running()
        queued = [s.title for s in self.queue]
        if active is not None:
            stage, path, state, events = active
            describe = describe_progress(events, 'running') if events else (None, 'loading the program')
            self.activity.show_running(stage.title if stage else path.name, state, describe, queued)
            self.activity.reset_live_text()
            self.pipeline.set_activity(stage.key if stage else None, describe[0], [s.key for s in self.queue])
            self.live_path = path
            self.progress.stop.setVisible(True)
            if path is None:
                self.progress.show_events([], state='starting', title=f'Starting: {stage.title}')
            else:
                self.progress.show_events(events, source=str(path), state='running')
                self._console(path)
        else:
            newest = next(iter(progress_files()), None)
            last = None
            if newest is not None:
                events = progress_file.read(newest)
                state = run_state(newest, events)
                stage = stage_of_progress(newest)
                result = Path(str(newest).replace('.progress.jsonl', '.json'))
                headline = ''
                if state == 'done' and result.exists() and stage is not None:
                    try:
                        name, before, after = stage.summary(load_result(result))['headline'][0]
                        headline = f'{name}: {fmt(before, 3)} → {fmt(after, 3)}' if before is not None else f'{name}: {fmt(after, 3)}'
                    except Exception:
                        headline = ''
                at = datetime.fromtimestamp(newest.stat().st_mtime).strftime('%H:%M')
                last = (stage.title if stage else newest.name, state, at, headline)
                self.live_path = newest
                self.progress.show_events(events, source=str(newest), state=state)
                self._console(newest)
            self.activity.show_idle(last)
            self.pipeline.set_activity(None, None, [])
            self.progress.stop.setVisible(False)
        current = active[1] if active else None
        if self.was_running is not None and current != self.was_running:
            self._finished(self.was_running)
        self.was_running = current

    def _console(self, path):
        if not self.progress.console.isVisible():
            return
        log = Path(str(path).replace('.progress.jsonl', '.log'))
        text = log.read_text(errors='replace')[-20000:] if log.exists() else '(no output log for this run)'
        if text != self.progress.console.toPlainText():
            self.progress.console.setPlainText(text)
            self.progress.console.moveCursor(QtGui.QTextCursor.MoveOperation.End)

    def _finished(self, path):
        """A fit that was running has ended: refresh the pipeline and the run list, open its result."""
        self.refresh()
        result = Path(str(path).replace('.progress.jsonl', '.json'))
        stage = stage_of_progress(path)
        if result.exists() and run_state(path) == 'done':
            self.result.show_result(result)
            self.statusBar().showMessage(f'{stage.title if stage else ""} finished: {result.relative_to(ROOT)}')
        else:
            self.statusBar().showMessage(f'{stage.title if stage else path.name} ended without a result ({run_state(path)})')

    # starting and stopping
    def action(self, stage, what):
        if what == 'run':
            self.start(stage)
        elif what == 'stop':
            self.stop()
        elif what == 'cancel' and stage in self.queue:
            self.queue.remove(stage)
            self.poll()

    def start(self, stage):
        """Run a stage now, or queue it behind the running fit (two fits at once slow each other
        down several times over: each already uses every core)."""
        active = self.running()
        if active is not None or self.queue:
            if (active is None or active[0] is not stage) and stage not in self.queue:
                self.queue.append(stage)
        else:
            self._launch(stage)
        self.poll()

    def _launch(self, stage):
        """Start a fit as its own process (not a child of this window): output to <result>.log,
        exit code to <result>.exit, at a lower priority so the desktop stays responsive."""
        output = stage.output()
        output.parent.mkdir(parents=True, exist_ok=True)
        command = ' '.join(shlex.quote(c) for c in stage.command(output))
        script = (f'nice -n 10 {command} > {shlex.quote(str(output.with_suffix(".log")))} 2>&1; '
                  f'echo $? > {shlex.quote(str(output.with_suffix(".exit")))}')
        ok, pid = QtCore.QProcess.startDetached('/bin/sh', ['-c', script], str(ROOT))
        if not ok:
            self.statusBar().showMessage(f'Could not start {stage.title.lower()}')
            return
        self.launching = (stage, output, pid, time.time())
        self.tabs.setCurrentWidget(self.progress)

    def stop(self):
        """Stop the running fit and empty the queue."""
        self.queue.clear()
        active = self.running()
        if active is None:
            return
        stage, path, state, events = active
        pid = events[0].get('pid') if events else None
        if pid is None and self.launching is not None:
            subprocess.run(['pkill', '-TERM', '-P', str(self.launching[2])], check=False)
        elif pid is not None:
            answer = QtWidgets.QMessageBox.question(self, 'Stop the fit', f'Stop {stage.title.lower()}? No result is written.')
            if answer != QtWidgets.QMessageBox.StandardButton.Yes:
                return
            try:
                os.kill(int(pid), 15)
            except ProcessLookupError:
                pass
        else:
            self.statusBar().showMessage('This fit was started elsewhere without a process id; stop it from where it runs')
        self.poll()

    # viewing results
    def open(self, path):
        if path is None:
            return
        path = Path(path)
        if path.name.endswith('.progress.jsonl'):
            self.statusBar().showMessage(f'{path.name} has no result (the run did not finish)')
            return
        self.result.show_result(path)
        self.tabs.setCurrentWidget(self.result)

    def _open_last_result(self):
        newest = next(iter(progress_files()), None)
        if newest is not None:
            self.open(Path(str(newest).replace('.progress.jsonl', '.json')))

    def closeEvent(self, event):
        if self.queue:
            answer = QtWidgets.QMessageBox.question(
                self, 'Queued fits', f'{", ".join(s.title for s in self.queue)} will not start once this window is '
                'closed (the running fit continues). Close anyway?')
            if answer != QtWidgets.QMessageBox.StandardButton.Yes:
                event.ignore()
                return
        event.accept()


def main():
    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication(sys.argv)
    window = ModelFitWindow()
    window.resize(1500, 950)
    window.show()
    return app.exec()


if __name__ == '__main__':
    sys.exit(main())
