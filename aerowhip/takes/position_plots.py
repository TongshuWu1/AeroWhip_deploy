"""Tracked-origin position of a take against its command and reference, on native samples.

Three stacked axes (x, y, z) share time; a fourth shows the tracking error |measured −
command| with its RMS over the selected interval when commands exist. Commands are zero-
order holds of at most 0.2 s, as in playback; missing samples remain gaps. Click to seek.
"""
import numpy as np
from PySide6.QtCore import Signal

from aerowhip.ui.plotting import PlotCanvas, gapped, message, rms
from aerowhip.ui.style import ROLE

COLORS = {"Desired": ROLE['reference'], "Command": ROLE['command'], "Measured": ROLE['measured'],
          "Predicted": ROLE['model']}
STYLES = {"Desired": '--', "Command": '-', "Measured": '-', "Predicted": '-'}


def held(time, values, breaks=None, hold=.2):
    """Zero-order hold of commands, each held until the next or for at most `hold` seconds."""
    time, values = np.asarray(time, float), np.asarray(values, float)
    t, v = [], []
    for i in range(len(time)):
        end = time[i+1] if i+1 < len(time) else time[i]+hold
        broken = breaks is not None and i < len(breaks) and breaks[i]
        end = min(end, time[i]+hold)
        t += [time[i], end]
        v += [values[i], values[i]]
        if broken or end < (time[i+1] if i+1 < len(time) else end):
            t.append(np.nan)
            v.append(np.full(values.shape[1:], np.nan))
    return np.asarray(t), np.asarray(v)


def sample_hold(time, values, at, hold=.2):
    """Command in force at each `at` time (NaN when none was received within `hold`)."""
    j = np.searchsorted(time, at, side='right')-1
    ok = (j >= 0) & (at-time[np.clip(j, 0, None)] <= hold)
    out = np.full((len(at),)+values.shape[1:], np.nan)
    out[ok] = values[j[ok]]
    return out


class PositionPlots(PlotCanvas):
    def __init__(self):
        super().__init__(minimum=(300, 210))
        self.take = None
        self.prediction = None
        self.visible = dict.fromkeys(COLORS, True)
        self.interval = None
        self.setToolTip("Tracked-origin position. Click to seek. Commands: zero-order hold; gaps are missing data.")
        self.render_plots()

    def clear(self):
        self.take = None
        self.prediction = None
        self.render_plots()

    def invalidate(self):
        self.render_plots()

    def set_take(self, take):
        self.take = take
        self.prediction = None
        self.interval = (float(take.time[0]), float(take.time[-1]))
        self.render_plots()

    def set_interval(self, first, last):
        if self.take is not None:
            self.interval = (float(self.take.time[first]), float(self.take.time[last]))
            self.render_plots()

    def set_prediction(self, prediction):
        self.prediction = prediction
        self.render_plots()

    def set_layer(self, name, visible):
        self.visible[name] = visible
        self.render_plots()

    def series(self):
        take = self.take
        out = {}
        if take.position is not None:
            breaks = np.diff(take.time) > 1.5*np.median(np.diff(take.time))
            if take.frame is not None:
                breaks |= np.diff(take.frame) > 1
            t = np.asarray(take.time, float)
            p = np.asarray(take.position, float).copy()
            idx = np.flatnonzero(breaks)+1
            out["Measured"] = (np.insert(t, idx, np.nan), np.insert(p, idx, np.nan, axis=0))
        if take.desired is not None:
            out["Desired"] = gapped(take.desired_time, take.desired)
        if take.command is not None:
            out["Command"] = held(take.command_time, take.command, take.command_breaks)
        if self.prediction is not None:
            out["Predicted"] = gapped(self.prediction.time, self.prediction.position)
        return out

    def render_plots(self):
        with self.style():
            self.reset()
            if self.take is None:
                message(self.figure.add_subplot(), 'Tracked-origin position x / y / z')
                self.draw_idle()
                return
            take = self.take
            error = take.command is not None and self.visible["Command"] and self.visible["Measured"]
            axes = self.figure.subplots(4 if error else 3, 1, sharex=True)
            data = self.series()
            for axis, ax in enumerate(axes[:3]):
                for name in ("Desired", "Command", "Predicted", "Measured"):
                    if name in data and self.visible[name]:
                        t, v = data[name]
                        ax.plot(t, v[:, axis], STYLES[name], color=COLORS[name], label=name.lower(),
                                linewidth=1.0 if name != "Measured" else 1.2)
                ax.set_ylabel(f"{'xyz'[axis]} (m)")
            if error:
                command = sample_hold(np.asarray(take.command_time, float), np.asarray(take.command, float), take.time)
                e = np.linalg.norm(np.asarray(take.position, float)-command, axis=1)
                lo, hi = self.interval
                inside = (take.time >= lo) & (take.time <= hi)
                axes[3].plot(take.time, 100*e, color=ROLE['error'],
                             label=f'|measured − command|, RMS {100*rms(e[inside]):.1f} cm')
                axes[3].set_ylabel('error (cm)')
                axes[3].set_ylim(bottom=0)
                axes[3].legend(loc='upper right')
            handles, labels = axes[0].get_legend_handles_labels()
            if len(handles) > 1:
                axes[0].legend(handles, labels, loc='upper right', ncols=len(handles))
            lo, hi = self.interval
            if (lo, hi) != (float(take.time[0]), float(take.time[-1])):
                for ax in axes:
                    ax.axvspan(lo, hi, color=ROLE['band'], alpha=.18, linewidth=0)
            axes[-1].set_xlabel('time (s)')
            axes[-1].set_xlim(float(take.time[0]), float(take.time[-1]))
            for ax in axes:
                self.add_cursor(ax, float(take.time[0]))
        self.draw_idle()
