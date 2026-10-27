"""AeroWhip Model Fit: run the model identification stages, watch a fit converge, compare a result
with the configuration and adopt it.

    python -m aerowhip.ui.fitting

Left: the pipeline (thrust map → drone → cable → coupled validation), each stage with whether its
latest result is in use and a Run button (hover a stage for what it fits). Right: Progress (best loss
and parameters of the selected run, live while it runs, also for fits started from a terminal),
Result (before/after, per-flight chart and the explicit Adopt), Data (the calibration manifest) and
Runs (every result, old layout included). Fits run as their own processes
(`aerowhip.fitting.stages`); the window reads their `.progress.jsonl` and result files.
"""
from datetime import datetime
import json
import math
from pathlib import Path
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
STALLED_S = 600.          # a progress file silent this long without 'done' is reported as stopped
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


def running_progress():
    """Progress files under runs/fits without a result yet (fits running or stopped)."""
    return [p for p in stages.FITS.glob('*/*.progress.jsonl')
            if not p.with_name(p.name.replace('.progress.jsonl', '.json')).exists()]


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


# ---------------------------------------------------------------- pipeline

STATUS_TEXT = {'in use': ('in use', GOOD), 'pending': ('result to review', PALETTE['orange']),
               'none': ('not run', None), 'validated': ('validated', GOOD)}


class StageCard(QtWidgets.QFrame):
    run_clicked = QtCore.Signal(object)
    open_clicked = QtCore.Signal(object)

    def __init__(self, number, stage):
        super().__init__()
        self.stage, self.latest = stage, None
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
        self.latest_label = label('', 'note', wrap=True)
        layout.addWidget(self.latest_label)
        buttons = QtWidgets.QHBoxLayout()
        self.run = QtWidgets.QPushButton('Run')
        self.run.setToolTip(' '.join(stage.command(stages.FITS/stage.key/'<time>.json')))
        self.run.clicked.connect(lambda: self.run_clicked.emit(self.stage))
        self.open = QtWidgets.QPushButton('Open result')
        self.open.clicked.connect(lambda: self.open_clicked.emit(self.latest))
        buttons.addWidget(self.run)
        buttons.addWidget(self.open)
        buttons.addStretch()
        layout.addLayout(buttons)

    def update_from(self, status, latest, summary, suggested):
        """suggested: this stage is the next step; its primary button is Run, or Open result when a
        result waits for a decision."""
        self.latest = latest
        self.status.setText(status_chip(*STATUS_TEXT[status]))
        for button, primary in ((self.run, suggested and status != 'pending'), (self.open, suggested and status == 'pending')):
            button.setObjectName('primary' if primary else '')
            button.style().unpolish(button)
            button.style().polish(button)
        self.open.setEnabled(latest is not None)
        text = ''
        if summary and summary['headline']:
            name, before, after = summary['headline'][0]
            text = f'{name}: ' + (f'{fmt(before, 3)} → <b>{fmt(after, 3)}</b>' if before is not None else f'<b>{fmt(after, 3)}</b>')
        if latest is not None:
            text += f'<br>{when(latest)}'
        self.latest_label.setText(text)


class Pipeline(QtWidgets.QWidget):
    run_clicked = QtCore.Signal(object)
    run_all_clicked = QtCore.Signal()
    open_clicked = QtCore.Signal(object)

    def __init__(self):
        super().__init__()
        layout = QtWidgets.QVBoxLayout(self)
        layout.setContentsMargins(10, 12, 10, 10)
        layout.setSpacing(8)
        self.next = label('', wrap=True)
        layout.addWidget(self.next)
        self.run_all = QtWidgets.QPushButton('Run stages 1–3 in turn')
        self.run_all.setToolTip('Thrust map, drone dynamics and cable, one after another (each starts from the '
                                'configuration as it is; nothing is adopted)')
        self.run_all.clicked.connect(self.run_all_clicked)
        layout.addWidget(self.run_all)
        self.cards = []
        for i, stage in enumerate(stages.STAGES):
            card = StageCard(i+1, stage)
            card.run_clicked.connect(self.run_clicked)
            card.open_clicked.connect(self.open_clicked)
            layout.addWidget(card)
            self.cards.append(card)
        residual = QtWidgets.QFrame()
        residual.setObjectName('card')
        residual.setToolTip('A small learned correction on top of the physical model, fitted once the coupled '
                            'validation is settled.')
        r = QtWidgets.QHBoxLayout(residual)
        r.setContentsMargins(12, 9, 12, 9)
        r.addWidget(label(f'<b>{len(self.cards)+1}&nbsp;&nbsp;Residual</b>', 'muted'))
        r.addStretch()
        r.addWidget(label('later', 'muted'))
        residual.setEnabled(False)
        layout.addWidget(residual)
        layout.addStretch()

    def refresh(self, results):
        statuses = [stage_status(card.stage, results) for card in self.cards]
        suggested = next((i for i, (s, _, _) in enumerate(statuses) if s in ('none', 'pending')), len(self.cards)-1)
        for i, (card, (status, latest, summary)) in enumerate(zip(self.cards, statuses)):
            card.update_from(status, latest, summary, i == suggested)
        stage, status = self.cards[suggested].stage, statuses[suggested][0]
        self.next.setText({'none': f'Next: run <b>{stage.title.lower()}</b>.',
                           'pending': f'Next: review the <b>{stage.title.lower()}</b> result.'}
                          .get(status, f'Next: <b>{stage.title.lower()}</b> of the adopted model.'))

    def set_running(self, active=None, queued=()):
        """active: key of the stage running now (None if idle); queued: keys waiting their turn."""
        for card in self.cards:
            key = card.stage.key
            card.run.setEnabled(key != active and key not in queued)
            card.run.setText('Running…' if key == active else 'Queued' if key in queued else 'Queue' if active else 'Run')
        self.run_all.setEnabled(active is None)


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
        self.loss = PlotCanvas(minimum=(300, 220))
        self.parameters = PlotCanvas(minimum=(300, 180))
        layout.addWidget(self.loss, 5)
        layout.addWidget(self.parameters, 3)
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

    def show_events(self, events, *, source='', running=False, exit_code=None):
        """exit_code: the exit code of a fit started here that has ended (None if still running or
        started elsewhere)."""
        start = next((e for e in events if e['kind'] == 'start'), None)
        if start is None:
            self.title.setText('Starting…' if running else 'No progress recorded')
            self.summary.setText('')
            self.bar.setValue(0)
            self.bar_label.setText('')
            self.source.setText(source)
            return
        stage = stages.BY_KEY.get(start['stage']) or next((s for s in stages.STAGES if s.title.lower() == start['stage']), None)
        self.title.setText(stage.title if stage else start['stage'])
        state = self._summary(events, start, running, exit_code)
        self._bar(events, state)
        signature = (source, len(events))
        if signature == self.signature:
            return                                      # only the clock moved
        self.signature = signature
        self.source.setText(source+('  (rebuilt from the result file)' if start.get('legacy') else ''))
        flights = [e for e in events if e['kind'] == 'flight']
        if flights:
            self._flights(flights)
        else:
            self._search(events, start)

    def _summary(self, events, start, running, exit_code):
        last = events[-1]
        evaluations = [e for e in events if e['kind'] == 'evaluation']
        flights = sum(e['kind'] == 'flight' for e in events)
        factor, unit = display_unit(start.get('unit'))
        if last['kind'] == 'done':
            state = 'done'
            chip = status_chip('done', GOOD)
        elif last['kind'] == 'failed' or exit_code not in (None, 0, 'stopped'):
            state = 'failed'
            chip = status_chip(f'failed: {last.get("error", f"exit code {exit_code}")}', BAD)
        elif running or (exit_code is None and time.time()-last.get('time', 0) < STALLED_S):
            state = 'running'
            chip = status_chip('running', PALETTE['blue'])
        else:
            state = 'stopped'
            chip = status_chip('stopped (no result written)', BAD)
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
        return state

    def _bar(self, events, state):
        p = plan_progress(events)
        if p is None:
            self.bar.setVisible(state == 'running')
            self.bar.setRange(0, 0 if state == 'running' else 1000)      # busy indicator without a plan
            self.bar_label.setText('')
            return
        fraction, phase, index, count, left = p
        self.bar.setVisible(True)
        self.bar.setRange(0, 1000)
        self.bar.setValue(int(1000*(1. if state == 'done' else fraction)))
        if state != 'running':
            self.bar_label.setText({'done': 'complete'}.get(state, f'stopped in {phase}'))
            return
        times = [e['time'] for e in events if e['kind'] == 'evaluation'][-30:]
        eta = ''
        if len(times) >= 3:
            per = (times[-1]-times[0])/(len(times)-1)
            minutes = left*per/60
            eta = f' · at most {minutes:.0f} min left' if minutes >= 1 else ' · under a minute left'
        self.bar_label.setText(f'{phase} ({index+1}/{count}) · {fraction*100:.0f} %{eta}')

    @staticmethod
    def _phase_curves(events):
        """{phase: (evaluation numbers, best-so-far values, best x)} at the start of each phase and
        at every evaluation that improved its phase's best; phase None without phase events."""
        curves = {}
        if not any(e['kind'] == 'evaluation' for e in events):     # older results: per-iteration best only
            iterations = [e for e in events if e['kind'] == 'iteration']
            keep = improvements([e['best'] for e in iterations]) if iterations else []
            if keep and keep[-1] != len(iterations)-1:
                keep.append(len(iterations)-1)
            return {None: (np.array([iterations[i]['n'] for i in keep], float), np.array([iterations[i]['best'] for i in keep]),
                           np.array([iterations[i]['x'] for i in keep]))} if keep else {}
        for e in events:
            if e['kind'] != 'evaluation' or e['value'] is None:
                continue
            n, v, x = curves.setdefault(e.get('phase'), ([], [], []))
            if not v or e['value'] < v[-1]:
                n.append(e['n']); v.append(e['value']); x.append(e['x'])
        last = {}
        for e in events:                                   # extend each curve to its phase's last evaluation
            if e['kind'] == 'evaluation':
                last[e.get('phase')] = e['n']
        out = {}
        for phase, (n, v, x) in curves.items():
            if last[phase] > n[-1]:
                n, v, x = n+[last[phase]], v+[v[-1]], x+[x[-1]]
            out[phase] = (np.array(n, float), np.array(v), np.array(x))
        return out

    def _search(self, events, start):
        factor, unit = display_unit(start.get('unit'))
        names = start['parameters']
        curves = self._phase_curves(events)
        plan = next((e for e in events if e['kind'] == 'plan'), None)
        labels = plan['phases'] if plan else []
        current = max((e['index'] for e in events if e['kind'] == 'phase'), default=None)
        finished = events[-1]['kind'] == 'done'
        profile = [(e['label'], e['value'], e['x']) for e in events
                   if e['kind'] == 'phase_done' and e['label'].startswith('delay') and e['value'] is not None]
        start_value = next((e['value'] for e in events if e['kind'] == 'phase_done' and e['value'] is not None), None)
        for canvas in (self.loss, self.parameters):
            canvas.reset()
        with self.loss.style():
            if profile or (plan and any(l.startswith('delay') for l in labels)):
                grid = self.loss.figure.add_gridspec(1, 2, width_ratios=(3, 1))
                ax, side = self.loss.figure.add_subplot(grid[0]), self.loss.figure.add_subplot(grid[1])
            else:
                ax, side = self.loss.figure.add_subplot(111), None
            for phase, (n, v, _) in sorted(curves.items(), key=lambda item: -1 if item[0] is None else item[0]):
                live = phase == current and not finished
                xs, ys = smooth(n, v*factor)
                ax.plot(xs, ys, color=ROLE['model'] if live or phase is None else ROLE['reference'],
                        lw=2.2 if live or phase is None else 1.3, alpha=1 if live or phase is None else .8)
                ax.plot(n[-1], v[-1]*factor, 'o', ms=5 if live else 3.5,
                        color=ROLE['model'] if live or phase is None else ROLE['reference'])
                if phase is not None and len(labels) > phase and labels[phase].startswith('delay') and not live:
                    ax.annotate(labels[phase].replace('delay ', ''), (n[-1], v[-1]*factor), xytext=(3, 3),
                                textcoords='offset points', fontsize=7, color=ROLE['reference'])
            if start_value is not None:
                ax.axhline(start_value*factor, color=ROLE['reference'], ls='--', lw=1)
            ax.set_title(f'Loss ({unit}): best so far in each phase' if plan else f'Best loss ({unit})')
            ax.set_xlabel('evaluation')
            ax.grid(axis='y', alpha=.3)
            if side is not None:
                if profile:
                    d = [float(l.split()[1]) for l, _, _ in profile]
                    side.plot(d, [v*factor for _, v, _ in profile], 'o-', color=ROLE['model'], lw=1.5, ms=5)
                    best = int(np.argmin([v for _, v, _ in profile]))
                    side.plot(d[best], profile[best][1]*factor, 'o', ms=9, mfc='none', color=ROLE['highlight'])
                side.set_title('Delay profile')
                side.set_xlabel('command delay (ms)')
                side.grid(alpha=.3)
            self.loss.figure.tight_layout()
        self.loss.draw_idle()
        with self.parameters.style():
            if names and curves:
                axes = self.parameters.figure.subplots(1, len(names), squeeze=False).ravel()
                for i, axp in enumerate(axes):
                    for phase, (n, _, X) in curves.items():
                        live = phase == current and not finished
                        xs, ys = smooth(n, X[:, i])
                        axp.plot(xs, ys, color=ROLE['model'] if live or phase is None else ROLE['reference'],
                                 lw=1.8 if live or phase is None else 1., alpha=1 if live or phase is None else .7)
                    axp.axhline(start['x0'][i], color=ROLE['reference'], ls='--', lw=.9)
                    newest = max(curves.items(), key=lambda item: item[1][0][-1])[1][2][-1, i]
                    axp.set_title(f'{names[i]}\n{fmt(float(newest))}', fontsize=8.5)
                    axp.tick_params(labelsize=7)
                    axp.locator_params(axis='y', nbins=4)
                    axp.locator_params(axis='x', nbins=3)
                self.parameters.figure.tight_layout()
        self.parameters.draw_idle()

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
        self.parameters.reset()
        self.parameters.draw_idle()


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
        for path in running_progress():
            stage = stages.stage_of(path.with_name(path.name.replace('.progress.jsonl', '.json')))
            if stage:
                item = QtWidgets.QTreeWidgetItem([path.name, when(path), '', 'running or stopped'])
                item.setData(0, QtCore.Qt.ItemDataRole.UserRole, str(path))
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

class ModelFitWindow(QtWidgets.QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle('AeroWhip Model Fit')
        style.apply(self)
        self.process = None
        self.stopping = False
        self.running = None          # (stage, result path) of the fit started here
        self.queue = []              # stages waiting their turn (fits run one at a time)
        self.exits = {}              # progress file -> exit code of fits started here that have ended
        self.watched = None          # progress file shown in the Progress tab
        self.pipeline = Pipeline()
        left = QtWidgets.QScrollArea()
        left.setWidgetResizable(True)
        left.setWidget(self.pipeline)
        left.setMinimumWidth(360)
        self.pipeline.setObjectName('page')
        self.progress, self.result, self.data, self.runs = ProgressView(), ResultView(), DataView(), RunsView()
        self.tabs = QtWidgets.QTabWidget()
        for widget, name in ((self.progress, 'Progress'), (self.result, 'Result'), (self.data, 'Data'), (self.runs, 'Runs')):
            self.tabs.addTab(widget, name)
        split = QtWidgets.QSplitter()
        split.addWidget(left)
        split.addWidget(self.tabs)
        split.setStretchFactor(1, 1)
        split.setSizes([380, 1100])
        self.setCentralWidget(split)
        self.statusBar().showMessage(f'Calibration data: {calibration_set.DEFAULT_ROOT}')
        self.pipeline.run_clicked.connect(self.start)
        self.pipeline.run_all_clicked.connect(lambda: [self.start(s) for s in stages.STAGES if s.adoptable])
        self.pipeline.open_clicked.connect(self.open)
        self.runs.open_clicked.connect(self.open)
        self.progress.stop_clicked.connect(self.stop)
        self.result.adopted.connect(self.refresh)
        self.tabs.currentChanged.connect(lambda _: self.tabs.currentWidget() is self.data and self.data.refresh())
        self.timer = QtCore.QTimer(self, interval=700, timeout=self.poll)
        self.timer.start()
        self.refresh()
        self.data.refresh()
        self._open_newest()

    def refresh(self):
        results = all_results()
        self.pipeline.refresh(results)
        self.runs.refresh(results)

    def _open_newest(self):
        live = sorted(running_progress(), key=lambda p: p.stat().st_mtime, reverse=True)
        if live and time.time()-live[0].stat().st_mtime < STALLED_S:
            self.open(live[0])
            return
        results = all_results()
        if results:
            self.open(results[0][0])

    def open(self, path):
        """Show a result (Result and Progress tabs) or a progress file without a result yet."""
        if path is None:
            return
        path = Path(path)
        if path.name.endswith('.progress.jsonl'):
            self.watched = path
            self.tabs.setCurrentWidget(self.progress)
        else:
            self.watched = progress_file.progress_path(path)
            self.result.show_result(path)
            if self.tabs.currentWidget() not in (self.progress, self.result):
                self.tabs.setCurrentWidget(self.result)
            if not self.watched.exists():
                self.watched = None
                self.progress.signature = None
                try:
                    self.progress.show_events(events_for(path), source=str(path))
                except Exception as exc:
                    self.progress.show_events([], source=f'{path}: {exc}')
        self.poll()

    def poll(self):
        if self.watched is None or not self.watched.exists():
            return
        running = self.running is not None and progress_file.progress_path(self.running[1]) == self.watched
        self.progress.show_events(progress_file.read(self.watched), source=str(self.watched), running=running,
                                  exit_code=self.exits.get(self.watched))

    # fits started here, one at a time
    def start(self, stage):
        """Run a stage now, or queue it behind the running one (two fits at once slow each other
        down several times over: each already uses every core)."""
        if self.running is not None:
            if stage is not self.running[0] and stage not in self.queue:
                self.queue.append(stage)
                self.statusBar().showMessage(f'Queued: {", ".join(s.title.lower() for s in self.queue)}')
            self._buttons()
            return
        self._launch(stage)

    def _buttons(self):
        self.pipeline.set_running(self.running[0].key if self.running else None, [s.key for s in self.queue])

    def _launch(self, stage):
        output = stage.output()
        output.parent.mkdir(parents=True, exist_ok=True)
        command = stage.command(output)
        self.process = QtCore.QProcess(self)
        self.process.setWorkingDirectory(str(ROOT))
        self.process.setProcessChannelMode(QtCore.QProcess.ProcessChannelMode.MergedChannels)
        self.process.readyReadStandardOutput.connect(self._output)
        self.process.finished.connect(self._finished)
        self.running = (stage, output)
        self.progress.console.clear()
        self.progress.console.appendPlainText('$ '+' '.join(command))
        self.progress.stop.setVisible(True)
        self.stopping = False
        self._buttons()
        self.watched = progress_file.progress_path(output)
        self.progress.signature = None
        self.progress.title.setText(f'{stage.title}: starting…')
        self.tabs.setCurrentWidget(self.progress)
        self.statusBar().showMessage(f'Running {stage.title.lower()} → {output.relative_to(ROOT)}')
        self.process.start('nice', ['-n', '10', *command])      # the window stays responsive

    def _output(self):
        text = bytes(self.process.readAllStandardOutput()).decode(errors='replace')
        self.progress.console.moveCursor(QtGui.QTextCursor.MoveOperation.End)
        self.progress.console.insertPlainText(text)
        self.progress.console.ensureCursorVisible()

    def stop(self):
        """Stop the running fit and empty the queue."""
        self.queue.clear()
        if self.process is not None and self.process.state() != QtCore.QProcess.ProcessState.NotRunning:
            self.stopping = True
            self.process.terminate()
            QtCore.QTimer.singleShot(3000, self.process.kill)
        self._buttons()

    def _finished(self, code, _status):
        stage, output = self.running
        self.running = None
        self.exits[progress_file.progress_path(output)] = 'stopped' if self.stopping else code
        self.progress.stop.setVisible(False)
        self.progress.console.appendPlainText(f'\n[exit code {code}]')
        self.poll()
        self.refresh()
        if code == 0 and output.exists():
            self.statusBar().showMessage(f'{stage.title} finished: {output.relative_to(ROOT)}')
            self.result.show_result(output)
        else:
            self.statusBar().showMessage(f'{stage.title} {"stopped" if self.stopping else f"ended with exit code {code}"}; '
                                         'no result written')
        if self.queue:
            self._launch(self.queue.pop(0))
        else:
            self._buttons()

    def closeEvent(self, event):
        if self.process is not None and self.process.state() != QtCore.QProcess.ProcessState.NotRunning:
            answer = QtWidgets.QMessageBox.question(self, 'A fit is running', 'Stop the running fit and close?')
            if answer != QtWidgets.QMessageBox.StandardButton.Yes:
                event.ignore()
                return
            self.process.kill()
            self.process.waitForFinished(2000)
        event.accept()


def main():
    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication(sys.argv)
    window = ModelFitWindow()
    window.resize(1500, 950)
    window.show()
    return app.exec()


if __name__ == '__main__':
    sys.exit(main())
