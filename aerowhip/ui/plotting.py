"""Matplotlib canvas for Qt windows: shared style, playback cursor by blitting, click-to-seek.

Create axes and artists inside `with canvas.style():` so they take the shared defaults
(`aerowhip.ui.style`). Playback cursors redraw without re-rendering the static plot.
"""
from contextlib import contextmanager

import matplotlib
import numpy as np
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg
from matplotlib.figure import Figure
from PySide6 import QtCore, QtWidgets

from aerowhip.ui.style import ROLE, matplotlib_defaults


class PlotCanvas(FigureCanvasQTAgg):
    seek_requested = QtCore.Signal(float)

    def __init__(self, parent=None, *, dark=False, minimum=(300, 180)):
        self.rc = matplotlib_defaults(dark)
        with matplotlib.rc_context(self.rc):
            figure = Figure()
        super().__init__(figure)
        self.setParent(parent)
        self.setMinimumSize(*minimum)
        self.setSizePolicy(QtWidgets.QSizePolicy.Policy.Expanding, QtWidgets.QSizePolicy.Policy.Expanding)
        self.cursors = []
        self.dynamic = []
        self._background = None
        self.mpl_connect('draw_event', self._drawn)
        self.mpl_connect('button_press_event', self._clicked)

    @contextmanager
    def style(self):
        with matplotlib.rc_context(self.rc):
            yield

    def reset(self):
        """Clear the figure (inside the style context, so new axes take the defaults)."""
        self.figure.clear()
        self.cursors = []
        self.dynamic = []
        self._background = None

    def add_cursor(self, ax, x=0.):
        line = ax.axvline(x, color=ROLE['measured'], linewidth=.9, alpha=.75, animated=True)
        self.cursors.append(line)
        return line

    def animate(self, artist):
        """Register an artist that changes during playback (redrawn by blitting)."""
        artist.set_animated(True)
        self.dynamic.append(artist)
        return artist

    def set_cursor(self, x):
        for line in self.cursors:
            line.set_xdata([x, x])
        self.refresh()

    def refresh(self):
        """Redraw only the cursors and animated artists over the cached static plot."""
        if not (self.cursors or self.dynamic):
            return
        if self._background is None:
            self.draw_idle()
            return
        self.restore_region(self._background)
        self._draw_dynamic()
        self.blit(self.figure.bbox)

    def _draw_dynamic(self):
        for artist in self.dynamic+self.cursors:
            if artist.axes is not None and artist.figure is not None:
                artist.axes.draw_artist(artist)

    def _drawn(self, _event):
        self._background = self.copy_from_bbox(self.figure.bbox)
        self._draw_dynamic()

    def _clicked(self, event):
        if event.inaxes is not None and event.button == 1 and event.xdata is not None \
                and self.toolbar_mode() == '':
            self.seek_requested.emit(float(event.xdata))

    def toolbar_mode(self):
        toolbar = getattr(self, 'toolbar', None)
        return '' if toolbar is None else str(toolbar.mode)

    def resizeEvent(self, event):
        self._background = None
        super().resizeEvent(event)


def gapped(time, values, *, max_gap=None):
    """Insert NaN where samples are further apart than max_gap (default 1.5 x median step)."""
    time = np.asarray(time, float)
    values = np.asarray(values, float)
    if len(time) < 2:
        return time, values
    step = np.diff(time)
    limit = 1.5*np.median(step) if max_gap is None else max_gap
    breaks = np.flatnonzero(step > limit)+1
    if not len(breaks):
        return time, values
    t = np.insert(time, breaks, np.nan)
    v = np.insert(values, breaks, np.nan, axis=0)
    return t, v


def rms(values):
    values = np.asarray(values, float)
    values = values[np.isfinite(values)]
    return float(np.sqrt(np.mean(values**2))) if len(values) else float('nan')


def message(ax, text):
    """An empty-state note in place of data (axes hidden)."""
    ax.set_axis_off()
    ax.text(.5, .5, text, transform=ax.transAxes, ha='center', va='center', color=ROLE['reference'])
