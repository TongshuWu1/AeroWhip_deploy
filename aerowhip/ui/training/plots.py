"""Learning curves of a PPO run from its saved events and per-collection outcome arrays.

Training returns are drawn as the per-collection median with the 10-90 % band over all
episodes of that collection (from `collection-NNNNN.npz`; the logged mean is used when the
array is unavailable). Deterministic evaluations are separate markers: one fixed initial
case each, not a distribution.
"""
import math

import numpy as np
from matplotlib.ticker import MaxNLocator
from PySide6 import QtWidgets

from aerowhip.ui.plotting import PlotCanvas, message
from aerowhip.ui.style import PALETTE, ROLE


def _series(rows, key, scale=1.):
    x = [e['round'] for e in rows if _finite(_get(e, key))]
    y = [scale*_get(e, key) for e in rows if _finite(_get(e, key))]
    return np.asarray(x, float), np.asarray(y, float)


def _get(event, key):
    if '.' in key:
        group, name = key.split('.', 1)
        return (event.get(group) or {}).get(name)
    return event.get(key)


def _finite(value):
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)


class CollectionReturns:
    """Per-episode returns of each collection, read once per file version."""
    def __init__(self):
        self.cache = {}

    def quantiles(self, directory, rounds):
        out = []
        for number in rounds:
            path = directory/f'collection-{number:05d}.npz'
            try:
                stat = path.stat()
                key = (str(path), stat.st_mtime_ns, stat.st_size)
                if key not in self.cache:
                    with np.load(path) as data:
                        returns = np.asarray(data['returns'], float)
                    returns = returns[np.isfinite(returns)]
                    self.cache[key] = np.percentile(returns, (10, 50, 90)) if len(returns) else None
                value = self.cache[key]
            except (OSError, KeyError, ValueError):
                value = None
            out.append(value)
        return out


class LearningCurves(QtWidgets.QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        layout = QtWidgets.QVBoxLayout(self)
        layout.setContentsMargins(4, 4, 4, 4)
        self.canvas = PlotCanvas(self, minimum=(600, 420))
        layout.addWidget(self.canvas)
        self.returns = CollectionReturns()
        self.signature = None

    def set_history(self, directory, history, experiment):
        rounds = [e for e in history if e.get('phase') == 'round']
        evaluations = [e for e in history if e.get('phase') == 'evaluation']
        signature = (str(directory), len(history), history[-1].get('elapsed_s') if history else None)
        if signature == self.signature:
            return
        self.signature = signature
        scale = experiment['ppo']['reward_scale']
        target_kl = experiment['ppo'].get('target_kl')
        c = self.canvas
        with c.style():
            c.reset()
            axes = c.figure.subplots(3, 2, sharex=True)
            ret, outcome, decomposition, kl, critic, entropy = axes.ravel()
            if not rounds:
                for ax in axes.ravel():
                    message(ax, 'No completed collection yet')
                c.draw_idle()
                return
            x = np.array([e['round'] for e in rounds], float)
            # Episode return: per-collection median and 10-90 % band; evaluation markers.
            bands = self.returns.quantiles(directory, [e['round'] for e in rounds])
            have = np.array([b is not None for b in bands])
            if have.any():
                q = np.array([b if b is not None else (np.nan,)*3 for b in bands])*scale
                ret.fill_between(x, q[:, 0], q[:, 2], color=ROLE['band'], alpha=.45, linewidth=0,
                                 label='training 10–90 %')
                ret.plot(x, q[:, 1], color=ROLE['model'], label='training median')
            else:
                ret.plot(*_series(rounds, 'return_mean', scale), color=ROLE['model'], label='training mean')
            if evaluations:
                ex, ey = _series(evaluations, 'evaluation_return', scale)
                ret.plot(ex, ey, 'o', color=ROLE['evaluation'], markersize=3.5, label='evaluation (1 case)')
                bx, by = _series(evaluations, 'best_return', scale)
                ret.step(bx, by, where='post', color=ROLE['evaluation'], linewidth=.8, linestyle='--',
                         label='best evaluation')
            ret.set_title('Episode return')
            ret.set_ylabel(f'return ×{scale:g}')
            ret.legend(loc='lower right', ncols=2)
            # Outcomes: shares of the collection's episodes.
            plotted = False
            for key, label, color in (('valid_fraction', 'task-valid', ROLE['model']),
                                      ('components.completed_horizon', 'full horizon', PALETTE['green']),
                                      ('components.physical_violation', 'physical violation', ROLE['error']),
                                      ('components.numerically_valid', 'numerically valid', ROLE['reference'])):
                sx, sy = _series(rounds, key)
                if len(sx):
                    # Numerical validity is usually 1: dashed so overlapping curves stay visible.
                    outcome.plot(sx, sy, color=color, label=label,
                                 linestyle='--' if key.endswith('numerically_valid') else '-')
                    plotted = True
            outcome.set_ylim(-.03, 1.03)
            outcome.set_title('Episode outcomes')
            outcome.set_ylabel('fraction of episodes')
            if plotted:
                outcome.legend(loc='best')
            # Return decomposition in task units (mean over the collection).
            plotted = False
            for key, label, color in (('components.continuous_whip_credit', 'accrued credit', PALETTE['green']),
                                      ('components.workspace_cost', 'workspace cost (subtracted)', PALETTE['orange']),
                                      ('components.failure_cost', 'abort cost (subtracted)', ROLE['error']),
                                      ('components.net_episode_return', 'net return', ROLE['measured'])):
                sx, sy = _series(rounds, key, scale)
                if len(sx):
                    decomposition.plot(sx, sy, color=color, label=label)
                    plotted = True
            if plotted:
                decomposition.axhline(0, color=ROLE['reference'], linewidth=.6)
                decomposition.legend(loc='best')
                decomposition.set_title('Return decomposition (collection mean)')
                decomposition.set_ylabel(f'×{scale:g}')
            else:
                message(decomposition, 'Return components not logged by this task version')
            # Policy update size.
            kx, ky = _series(rounds, 'policy_kl_mean')
            if not len(kx):
                kx, ky = _series(rounds, 'approx_kl')
            kl.plot(kx, ky, color=ROLE['model'], label='KL after update')
            stops = [e['round'] for e in rounds if e.get('kl_early_stop')]
            if stops:
                ky_stop = [ky[list(kx).index(s)] for s in stops if s in list(kx)]
                kl.plot([s for s in stops if s in list(kx)], ky_stop, 'x', color=ROLE['error'],
                        markersize=4, label='early stop')
            if _finite(target_kl):
                kl.axhline(target_kl, color=ROLE['reference'], linestyle='--', linewidth=.8, label='target')
            kl.set_title('Policy update')
            kl.set_ylabel('mean KL (nats)')
            kl.legend(loc='best')
            # Critic quality: explained variance of the value targets.
            ev_x, ev_y = _series(rounds, 'value_explained_variance')
            if len(ev_x):
                critic.plot(ev_x, ev_y, color=ROLE['model'], label='after update')
                bx, by = _series(rounds, 'value_explained_variance_before')
                if len(bx):
                    critic.plot(bx, by, color=ROLE['reference'], linewidth=.9, label='before update')
                critic.axhline(1, color=ROLE['reference'], linewidth=.6)
                critic.set_ylim(min(-.1, float(np.nanmin(ev_y))-.05) if len(ev_y) else -.1, 1.05)
                critic.set_title('Critic fit')
                critic.set_ylabel('explained variance')
                critic.legend(loc='lower right')
            else:
                message(critic, 'Value explained variance not logged')
            # Exploration: measured entropy and its bonus coefficient.
            exploration = [e for e in history if e.get('phase') == 'exploration']
            nx, ny = _series(exploration or rounds, 'entropy')
            entropy.plot(nx, ny, color=ROLE['model'], label='policy entropy')
            entropy.set_title('Exploration')
            entropy.set_ylabel('entropy (nats)')
            wx, wy = _series(exploration or rounds, 'entropy_weight')
            if len(wx):
                twin = entropy.twinx()
                twin.plot(wx, wy, color=PALETTE['orange'], linewidth=.9, label='bonus coefficient')
                twin.set_ylabel('bonus coefficient', color=PALETTE['orange'])
                twin.tick_params(axis='y', colors=PALETTE['orange'])
                twin.grid(False)
                twin.spines['right'].set_visible(True)
                lines = entropy.get_lines()+twin.get_lines()
                entropy.legend(lines, [l.get_label() for l in lines], loc='best')
            for ax in (critic, entropy):
                ax.set_xlabel('collection')
                ax.xaxis.set_major_locator(MaxNLocator(integer=True))
            c.draw_idle()
