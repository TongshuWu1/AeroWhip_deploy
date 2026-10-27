"""Native-sample diagnostics, not fitting eligibility or automatic cleaning."""
import numpy as np
from matplotlib.colors import ListedColormap

from aerowhip.ui.plotting import PlotCanvas
from aerowhip.ui.style import ROLE


def masks(take):
    pose = (np.isfinite(take.position).all(1) & np.isfinite(take.quaternion).all(1)
            & (np.linalg.norm(take.quaternion, axis=1) > 1e-12))
    markers = np.isfinite(take.cable).all(2)
    dt = np.diff(take.time)
    gaps = dt > 1.5 * np.median(dt)
    if take.frame is not None:
        gaps |= np.diff(take.frame) > 1
    return pose, markers, gaps


def summarize(take, first, last):
    pose, markers, gaps = masks(take)
    window = slice(first, last+1)
    dt = np.diff(take.time)
    return {
        "frames": last-first+1,
        "duration": float(take.time[last]-take.time[first]),
        "rate": float(1/np.median(dt)),
        "pose_percent": float(100*pose[window].mean()),
        "marker_percent": 100*markers[window].mean(axis=0),
        "complete_percent": float(100*markers[window].all(1).mean()),
        "gaps": int(gaps[first:last].sum()),
        "largest_gap": float(dt[first:last][gaps[first:last]].max(initial=0)),
    }


class QualityTimeline(PlotCanvas):
    """Which samples were seen: drone pose and each cable marker (c1 at the drone), plus
    timestamp gaps. Hidden markers are normal (partial frames still constrain the cable).
    Click to seek."""

    def __init__(self):
        super().__init__(minimum=(300, 150))
        self.setMaximumHeight(230)
        self.take = None
        self.first = self.last = 0
        self.setToolTip(self.__doc__.replace('\n    ', ' '))
        self.clear()

    def clear(self):
        self.take = None
        with self.style():
            self.reset()
        self.draw_idle()

    def set_take(self, take):
        self.take = take
        self.pose, self.markers, self.gaps = masks(take)
        self.first, self.last = 0, len(take.time)-1
        self.render_plot()

    def set_interval(self, first, last):
        self.first, self.last = first, last
        if self.take is not None:
            self.render_plot()

    def render_plot(self):
        take = self.take
        rows = np.column_stack([self.pose, self.markers]).T.astype(float)
        names = ['pose']+[f'c{i+1}' for i in range(self.markers.shape[1])]
        percent = 100*rows.mean(1)
        with self.style():
            self.reset()
            ax = self.figure.add_subplot()
            t = np.asarray(take.time, float)
            ax.imshow(rows, aspect='auto', interpolation='nearest', origin='upper',
                      cmap=ListedColormap(['#f2c4b6', '#d5dde5']), vmin=0, vmax=1,
                      extent=(t[0], t[-1], len(names)-.5, -.5))
            for i in np.flatnonzero(self.gaps):
                ax.axvspan(t[i], t[i+1], color=ROLE['error'], alpha=.8, linewidth=0)
            ax.set_yticks(range(len(names)), names)
            ax.tick_params(axis='y', labelsize=7, length=0)
            ax.grid(False)
            if (self.first, self.last) != (0, len(t)-1):
                for edge in (t[self.first], t[self.last]):
                    ax.axvline(edge, color=ROLE['command'], linewidth=1.2)
            seen = self.markers.sum(1).mean()
            ax.set_title(f'Visibility · pose {percent[0]:.0f} % · cable {seen:.2f} of {self.markers.shape[1]} '
                         f'markers per frame (hidden: red)', fontsize=8.5)
            self.add_cursor(ax, float(t[0]))
        self.draw_idle()
