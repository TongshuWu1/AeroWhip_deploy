"""The analysis pages of the Identification Loop window: the timeline strip under the 3D view, the
information a trajectory carries (per parameter, coupling, the optimiser's search), each ensemble
member's predicted safety, and the measured envelope."""
import numpy as np
from PySide6 import QtCore, QtGui, QtWidgets

from aerowhip.ui.plotting import PlotCanvas
from aerowhip.ui.style import PALETTE, ROLE

GOOD, BAD, WARN, MUTED = '#2fbf8f', '#ff7a59', '#E69F00', '#93a0ad'
RISE_LIMIT_M = -.04          # the optimiser's margin: the cable's highest node this far below the attachment


def label(text='', name=None, wrap=True):
    w = QtWidgets.QLabel(text)
    if name:
        w.setObjectName(name)
    w.setWordWrap(wrap)
    return w


def table(columns):
    t = QtWidgets.QTableWidget(0, len(columns))
    t.setHorizontalHeaderLabels(columns)
    t.verticalHeader().setVisible(False)
    t.setEditTriggers(QtWidgets.QAbstractItemView.EditTrigger.NoEditTriggers)
    t.setSelectionMode(QtWidgets.QAbstractItemView.SelectionMode.NoSelection)
    t.setFocusPolicy(QtCore.Qt.FocusPolicy.NoFocus)
    t.setShowGrid(False)
    t.setAlternatingRowColors(True)
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


class Strip(PlotCanvas):
    """Under the 3D view: the planned accelerations and the predicted cable rise against time, with
    the playback cursor; a click seeks."""
    def __init__(self):
        super().__init__(dark=True, minimum=(400, 150))
        self.setMaximumHeight(190)

    def show(self, data, preview):
        self.reset()
        with self.style():
            ax = self.figure.add_subplot(1, 1, 1)
            if data is None:
                ax.set_axis_off()
                self.draw_idle()
                return
            t = np.asarray(data['t']); a = np.asarray(data['acceleration'])
            ax.plot(t, np.linalg.norm(a[:, :2], axis=1), color=ROLE['command'], linewidth=.9, label='horizontal accel (m/s²)')
            ax.plot(t, a[:, 2], color=PALETTE['sky'], linewidth=.8, alpha=.8, label='vertical accel (m/s²)')
            ax.set_xlim(t[0], t[-1]+2)
            ax.set_xlabel('time (s)')
            if preview is not None:
                rx = ax.twinx()
                rx.plot(preview['t'], preview['rise']*100, color='#f2f2f2', linewidth=1., label='cable top above clamp (cm, right)')
                rx.axhline(RISE_LIMIT_M*100, color=BAD, linewidth=.8, linestyle='--')
                rx.set_ylabel('cable top (cm)')
                rx.grid(False)
                rx.spines['right'].set_visible(True)
                lines = ax.get_lines()+rx.get_lines()[:1]
                ax.legend(lines, [l.get_label() for l in lines], loc='lower left', bbox_to_anchor=(0, 1.0), ncols=3, fontsize=7.5,
                          frameon=False, borderaxespad=.1)
            else:
                ax.legend(loc='lower left', bbox_to_anchor=(0, 1.0), ncols=2, fontsize=7.5, frameon=False, borderaxespad=.1)
            self.add_cursor(ax, 0.)
        self.draw_idle()


class InformationView(QtWidgets.QScrollArea):
    def __init__(self):
        super().__init__()
        self.setWidgetResizable(True)
        page = QtWidgets.QWidget()
        page.setObjectName('page')
        layout = QtWidgets.QVBoxLayout(page)
        layout.setContentsMargins(18, 14, 18, 14)
        layout.addWidget(label('What this flight would teach the fit', 'heading'))
        layout.addWidget(label('Per parameter: the information left about it after every other parameter is allowed to move '
                               '(log10 of 1/(F⁻¹)ᵢᵢ, in units of the fit\'s present uncertainty). Coupled parameters, two '
                               'that move the cable alike, lower each other\'s value, so a high bar means the flight tells '
                               'that parameter apart from the rest. The flight is as good as its weakest bar.', 'muted'))
        self.canvas = PlotCanvas(dark=True, minimum=(600, 620))
        layout.addWidget(self.canvas)
        layout.addWidget(label('Coupled estimates', 'section'))
        self.coupled = label('', None)
        layout.addWidget(self.coupled)
        layout.addStretch(1)
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
            grid = self.canvas.figure.add_gridspec(rows, 2, height_ratios=(1.7, 1)[:rows])
            ax = self.canvas.figure.add_subplot(grid[0, 0] if correlation else grid[0, :])
            names = [n for n, _ in marginal]
            values = [v for _, v in marginal]
            ax.barh(range(len(values)), values, color=[BAD if i == 0 else PALETTE['sky'] for i in range(len(values))], height=.7)
            ax.set_yticks(range(len(values)), names)
            ax.invert_yaxis()
            ax.set_xlabel('log10 marginal information')
            ax.set_title(f'Weakest: {names[0]}')
            ax.grid(axis='y', visible=False)
            if correlation:
                order = data['parameters']
                cx = self.canvas.figure.add_subplot(grid[0, 1])
                image = cx.imshow(np.asarray(correlation), vmin=-1, vmax=1, cmap='RdBu_r')
                cx.set_xticks(range(len(order)), [str(i+1) for i in range(len(order))])
                cx.set_yticks(range(len(order)), [f'{i+1}  {n[:22]}' for i, n in enumerate(order)])
                cx.grid(False)
                cx.set_title('Correlation of the estimates')
                self.canvas.figure.colorbar(image, ax=cx, shrink=.7)
            if candidates:
                hx = self.canvas.figure.add_subplot(grid[1, :])
                ok = [(i, c['score']) for i, c in enumerate(candidates) if c['feasible'] and c['score'] > -1e8]
                bad = [i for i, c in enumerate(candidates) if not c['feasible']]
                low = min((s for _, s in ok), default=0.)
                if ok:
                    hx.plot(*zip(*ok), 'o', color=PALETTE['sky'], label='safe on every model')
                    hx.plot([i for i, _ in ok], np.maximum.accumulate([s for _, s in ok]), color='#f2f2f2', linewidth=.8, label='best so far')
                if bad:
                    hx.plot(bad, [low]*len(bad), 'x', color=BAD, label='unsafe on a model')
                hx.set_xlabel('candidate')
                hx.set_ylabel('score')
                hx.set_title('How the optimiser found it')
                hx.legend(loc='lower right', ncols=3)
        self.canvas.draw_idle()
        if 'coupled' not in data:
            self.coupled.setText('Not computed for this trajectory (optimised before the coupling report); optimise again to see it.')
        elif not data['coupled']:
            self.coupled.setText('None beyond 0.8: the flight separates every pair.')
        else:
            self.coupled.setText('\n'.join(f'{a}  ~  {b}:  {c:+.2f}' for a, b, c in data['coupled'][:12]))


class SafetyView(QtWidgets.QWidget):
    def __init__(self):
        super().__init__()
        self.setObjectName('page')
        layout = QtWidgets.QVBoxLayout(self)
        layout.setContentsMargins(18, 14, 18, 14)
        layout.addWidget(label('Is it safe on every model we might have?', 'heading'))
        layout.addWidget(label('Each member of the ensemble (samples of the fitted model, and the brackets that reach the '
                               'measured cable flip) flown closed loop on this trajectory. Cable top: the highest cable node '
                               f'above the clamp, must stay below {RISE_LIMIT_M*100:.0f} cm. Motor: the smallest motor as a '
                               'share of its hover share, must stay above 0.05.', 'muted'))
        self.table = table(['model', 'cable top (cm)', 'tilt (deg)', 'tracking error (cm)', 'smallest motor', 'verdict'])
        layout.addWidget(self.table, 1)

    def show(self, data, report):
        rows = (data or {}).get('member_rows') or ((report or {}).get('best') or {}).get('members') or []
        out, colours = [], {}
        for i, m in enumerate(rows):
            rise = None if m.get('rise') is None else m['rise']*100
            err = None if m.get('error') is None else m['error']*100
            out.append([m['label'], rise, m.get('tilt'), err, m.get('motor'), 'safe' if m['ok'] else m.get('reason', '')])
            colours[(i, 5)] = GOOD if m['ok'] else BAD
            if rise is not None:
                colours[(i, 1)] = GOOD if rise < RISE_LIMIT_M*100-2 else WARN
        fill(self.table, out, colours)


class EnvelopeView(QtWidgets.QWidget):
    def __init__(self):
        super().__init__()
        self.setObjectName('page')
        layout = QtWidgets.QVBoxLayout(self)
        layout.setContentsMargins(18, 14, 18, 14)
        layout.addWidget(label('What the real cable did', 'heading'))
        self.note = label('', 'muted')
        layout.addWidget(self.note)
        self.table = table(['kind', 'heading', 'level (m/s²)', 'flights', 'clean', 'failed', 'cable top (cm)', 'frames over', 'max tilt'])
        layout.addWidget(self.table, 1)

    def show(self, env):
        if env is None:
            self.note.setText('No envelope yet: Update envelope after the first flight.')
            fill(self.table, [])
            return
        threshold = env['rise_threshold_m']
        self.note.setText(f'Worst of all flights per kind, heading and level, from the raw marker clouds of the recorded '
                          f'parts (hover start to hover end). Cable over the drone: a point more than {threshold*100:.0f} cm '
                          'above the clamp.')
        rows, colours = [], {}
        for i, r in enumerate(env['summary']):
            rise = None if r.get('max_rise_m') is None else r['max_rise_m']*100
            rows.append([r['kind'], r.get('heading'), r.get('level'), r['flights'], r['clean'], r['failed'], rise,
                         r.get('max_over_fraction'), r.get('max_tilt_deg')])
            colours[(i, 5)] = BAD if r['failed'] else GOOD
            if rise is not None:
                colours[(i, 6)] = BAD if rise > threshold*100 else (WARN if rise > RISE_LIMIT_M*100 else GOOD)
        fill(self.table, rows, colours)


class OnlineView(QtWidgets.QWidget):
    """The online estimate of a flight (scripts/online_fit.py writes runs/online/*.jsonl): each
    parameter over the flight with its standard deviation, the prior (the offline fit) dashed, and
    what happened to every window: accepted, held as an outlier, applied as a change, or gated."""
    STATUS = dict(accepted=(GOOD, '.'), outlier=(WARN, 'x'), change=(BAD, 'D'), gated=(MUTED, '|'))

    def __init__(self, folder):
        super().__init__()
        self.setObjectName('page')
        self.folder, self.path, self.mtime = folder, None, None
        self.follow = True                     # show the newest run as runs appear
        layout = QtWidgets.QVBoxLayout(self)
        layout.setContentsMargins(18, 14, 18, 14)
        head = QtWidgets.QHBoxLayout()
        head.addWidget(label('Online identification', 'heading'), 1)
        self.runs = QtWidgets.QComboBox()
        self.runs.setMinimumWidth(360)
        self.runs.activated.connect(self._picked)
        head.addWidget(self.runs)
        layout.addLayout(head)
        self.status = label('', 'muted')
        layout.addWidget(self.status)
        self.canvas = PlotCanvas(dark=True, minimum=(600, 430))
        layout.addWidget(self.canvas, 1)
        layout.addWidget(label('Not used (newest first)', 'section'))
        self.events = label('', None)
        self.events.setTextInteractionFlags(QtCore.Qt.TextInteractionFlag.TextSelectableByMouse)
        layout.addWidget(self.events)

    def _picked(self, index):
        self.follow = index == 0
        self.load(self.runs.currentData(), force=True)

    def files(self):
        return sorted(self.folder.glob('*.jsonl'), key=lambda p: p.stat().st_mtime, reverse=True) if self.folder.exists() else []

    def poll(self):
        """Follow the newest run, unless an older one was picked; redraw when its file grew."""
        files = self.files()
        names = [str(p) for p in files]
        if names != [self.runs.itemData(i) for i in range(self.runs.count())]:
            current = self.runs.currentData()
            self.runs.blockSignals(True)
            self.runs.clear()
            for p in files:
                self.runs.addItem(p.stem, str(p))
            index = 0 if self.follow or current not in names else names.index(current)
            self.runs.setCurrentIndex(index)
            self.runs.blockSignals(False)
        if self.runs.currentData():
            self.load(self.runs.currentData())

    def load(self, path, force=False):
        from pathlib import Path
        path = Path(path)
        try:
            mtime = path.stat().st_mtime
        except OSError:
            return
        if not force and path == self.path and mtime == self.mtime:
            return
        self.path, self.mtime = path, mtime
        import json
        rows = []
        for line in path.read_text().splitlines():
            try:
                rows.append(json.loads(line))
            except ValueError:
                pass
        header = next((r for r in rows if r.get('kind') == 'header'), None)
        updates = [r for r in rows if r.get('kind') == 'update']
        done = any(r.get('kind') == 'summary' for r in rows)
        self.draw(header, updates)
        if header is None:
            self.status.setText('')
            return
        counts = {}
        for u in updates:
            for f, r in u['families'].items():
                counts.setdefault(f, {}).setdefault(r['status'], 0)
                counts[f][r['status']] += 1
        state = 'finished' if done else ('LIVE' if header.get('live') else 'replaying')
        self.status.setText(f"{state} · {header['flight']} · {len(updates)} updates · "
                            + ' · '.join(f"{f}: " + ', '.join(f'{v} {k}' for k, v in c.items()) for f, c in counts.items()))
        lines = []
        for u in reversed(updates):
            for f, r in u['families'].items():
                if r['status'] != 'accepted' and len(lines) < 10:
                    lines.append(f"{u['stamp']-updates[0]['stamp']:6.1f} s  {f:5s}  {r['status']:8s}  {r.get('reason', '')}")
        self.events.setText('\n'.join(lines) or 'every window used')

    def draw(self, header, updates):
        self.canvas.reset()
        if header is None or not updates:
            self.canvas.draw_idle()
            return
        names = header['parameters']
        t0 = updates[0]['stamp']
        t = np.array([u['stamp']-t0 for u in updates])
        with self.canvas.style():
            grid = self.canvas.figure.add_gridspec(2, 3)
            for k, name in enumerate(names):
                ax = self.canvas.figure.add_subplot(grid[k//3, k % 3])
                v = np.array([u['theta'][name] for u in updates]); s = np.array([u['sd'][name] for u in updates])
                prior = header['prior'][name]
                ax.axhspan(prior['value']-prior['sd'], prior['value']+prior['sd'], color=MUTED, alpha=.12, lw=0)
                ax.axhline(prior['value'], color=MUTED, lw=.8, ls='--')
                ax.fill_between(t, v-s, v+s, color=PALETTE['sky'], alpha=.25, lw=0)
                ax.plot(t, v, color=PALETTE['sky'], lw=1.3)
                family = 'cable' if k < 3 else 'drone'
                lo = min((v-s).min(), prior['value']-prior['sd']); hi = max((v+s).max(), prior['value']+prior['sd'])
                y = lo-.08*(hi-lo)
                for status, (colour, marker) in self.STATUS.items():
                    at = [ti for ti, u in zip(t, updates) if u['families'].get(family, {}).get('status') == status]
                    if at and status != 'accepted':
                        ax.plot(at, [y]*len(at), marker, color=colour, ms=5, ls='none', label=status)
                ax.set_title(name)
                if k >= 3:
                    ax.set_xlabel('flight time (s)')
                if k == 0:
                    handles = [ax.plot([], [], m, color=c, ls='none')[0] for c, m in (v_ for s_, v_ in self.STATUS.items() if s_ != 'accepted')]
                    ax.legend(handles, [s_ for s_ in self.STATUS if s_ != 'accepted'], loc='upper right', ncols=3, fontsize=7)
        self.canvas.draw_idle()
