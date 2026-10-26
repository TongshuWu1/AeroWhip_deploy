"""Cached native-sample plots; playback only repaints the cursor overlay."""
import numpy as np
from PySide6.QtCore import Qt, QRectF, Signal
from PySide6.QtGui import QColor, QPainter, QPainterPath, QPen, QPixmap
from PySide6.QtWidgets import QWidget

COLORS = {"Desired": "#4d91ff", "Command": "#e88724", "Measured": "#35a66c", "Predicted": "#ad70dc"}


def native_segments(time, values, breaks=None, *, step=False):
    """Edges in native time. Commands hold at most 0.2 s, as in playback."""
    time, values = np.asarray(time), np.asarray(values)
    valid = np.isfinite(values) & np.isfinite(time)
    edges = []
    for i in range(len(time)-1):
        broken = breaks is not None and breaks[i]
        if not valid[i]:
            continue
        if step:
            if broken:
                continue
            end = min(time[i+1], time[i]+.2)
            edges.append(((time[i], values[i]), (end, values[i])))
            if valid[i+1] and not broken and time[i+1]-time[i] <= .2:
                edges.append(((end, values[i]), (end, values[i+1])))
        elif valid[i+1] and not broken:
            edges.append(((time[i], values[i]), (time[i+1], values[i+1])))
    return np.asarray(edges, dtype=float).reshape(-1, 2, 2)


class PositionPlots(QWidget):
    seek_requested = Signal(float)

    def __init__(self):
        super().__init__()
        self.setMinimumHeight(210)
        self.take = None
        self.current = 0.
        self.interval = (0., 0.)
        self.visible = dict.fromkeys(COLORS, True)
        self.series = {}
        self.cache = None
        self.setToolTip("Tracked-origin position in metres. Click a plot to seek. Commands use zero-order hold; missing data remain gaps.")

    def clear(self):
        self.take = None
        self.series = {}
        self.invalidate()

    def invalidate(self):
        self.cache = None
        self.update()

    def set_take(self, take):
        self.take = take
        self.current = 0.
        self.interval = (float(take.time[0]), float(take.time[-1]))
        self.series = {}
        for name, time, points in (("Measured", take.time, take.position),
                ("Desired", take.desired_time, take.desired),
                ("Command", take.command_time, take.command)):
            if points is None:
                continue
            breaks = np.diff(time) > 1.5*np.median(np.diff(time))
            if name == "Measured" and take.frame is not None:
                breaks |= np.diff(take.frame) > 1
            if name == "Command":
                breaks = take.command_breaks
            self.series[name] = [native_segments(time, points[:, i], breaks, step=name == "Command")
                                 for i in range(3)]
        self.invalidate()

    def set_interval(self, first, last):
        if self.take is not None:
            self.interval = (float(self.take.time[first]), float(self.take.time[last]))
            self.invalidate()

    def set_prediction(self, prediction):
        self.series.pop('Predicted',None)
        if prediction is not None:
            self.series['Predicted']=[native_segments(prediction.time,prediction.position[:,i]) for i in range(3)]
        self.invalidate()

    def set_cursor(self, time):
        self.current = time
        self.update()

    def set_layer(self, name, visible):
        self.visible[name] = visible
        self.invalidate()

    def rectangles(self):
        height = max(30., (self.height()-28)/3)
        return [QRectF(61, 4+i*height, max(1, self.width()-76), height-12) for i in range(3)]

    def x(self, time, rect):
        return rect.left()+rect.width()*time/self.take.time[-1]

    def render_cache(self):
        ratio = self.devicePixelRatioF()
        pixmap = QPixmap(round(self.width()*ratio), round(self.height()*ratio))
        pixmap.setDevicePixelRatio(ratio)
        pixmap.fill(self.palette().window().color())
        painter = QPainter(pixmap)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        foreground = self.palette().text().color()
        if self.take is None:
            painter.setPen(foreground)
            painter.drawText(self.rect(), Qt.AlignmentFlag.AlignCenter, "Position · X / Y / Z")
            painter.end()
            return pixmap
        for axis, rect in enumerate(self.rectangles()):
            arrays = [rows[axis] for name, rows in self.series.items() if self.visible[name] and len(rows[axis])]
            values = []
            for edges in arrays:
                overlap = (edges[:, 1, 0] >= 0) & (edges[:, 0, 0] <= self.take.time[-1])
                values.extend(edges[overlap, :, 1].ravel())
            lo, hi = (min(values), max(values)) if len(values) else (-1., 1.)
            pad = max(.02, (hi-lo)*.08)
            lo, hi = lo-pad, hi+pad
            def y(value):
                return rect.bottom()-rect.height()*(value-lo)/(hi-lo)
            painter.fillRect(rect, self.palette().base())
            if self.interval[0] <= self.interval[1]:
                painter.fillRect(QRectF(self.x(self.interval[0], rect), rect.top(),
                    rect.width()*(self.interval[1]-self.interval[0])/self.take.time[-1], rect.height()), QColor(225, 174, 53, 28))
            painter.setPen(QPen(QColor(125, 125, 125, 70), 1))
            for t in np.linspace(0, self.take.time[-1], 5):
                x = self.x(t, rect)
                painter.drawLine(int(x), int(rect.top()), int(x), int(rect.bottom()))
            painter.setPen(foreground)
            painter.drawText(QRectF(0, rect.top(), 55, 16), Qt.AlignmentFlag.AlignRight, f"{hi:.2f}")
            painter.drawText(QRectF(0, rect.bottom()-16, 55, 16), Qt.AlignmentFlag.AlignRight, f"{lo:.2f}")
            painter.drawText(QRectF(0, rect.center().y()-8, 55, 16), Qt.AlignmentFlag.AlignRight, f"{'XYZ'[axis]} (m)")
            painter.save()
            painter.setClipRect(rect)
            for name in ("Desired", "Command", "Measured", "Predicted"):
                if name not in self.series or not self.visible[name]:
                    continue
                path = QPainterPath()
                previous = None
                for a, b in self.series[name][axis]:
                    if previous is None or not np.array_equal(a, previous):
                        path.moveTo(self.x(a[0], rect), y(a[1]))
                    path.lineTo(self.x(b[0], rect), y(b[1]))
                    previous = b
                pen = QPen(QColor(COLORS[name]), 1.4)
                if name == "Desired":
                    pen.setStyle(Qt.PenStyle.DashLine)
                painter.setPen(pen)
                painter.drawPath(path)
            painter.restore()
        rect = self.rectangles()[-1]
        painter.setPen(foreground)
        for t in np.linspace(0, self.take.time[-1], 5):
            painter.drawText(QRectF(self.x(t, rect)-28, rect.bottom()+4, 56, 18),
                             Qt.AlignmentFlag.AlignCenter, f"{t:.1f}")
        painter.drawText(QRectF(0, self.height()-20, 58, 18), Qt.AlignmentFlag.AlignRight, "Time (s)")
        painter.end()
        return pixmap

    def paintEvent(self, event):
        if self.cache is None:
            self.cache = self.render_cache()
        painter = QPainter(self)
        painter.drawPixmap(0, 0, self.cache)
        if self.take is not None:
            painter.setPen(QPen(self.palette().text().color(), 1))
            for rect in self.rectangles():
                x = int(self.x(self.current, rect))
                painter.drawLine(x, int(rect.top()), x, int(rect.bottom()))

    def resizeEvent(self, event):
        self.invalidate()
        super().resizeEvent(event)

    def mousePressEvent(self, event):
        if self.take is not None and event.button() == Qt.MouseButton.LeftButton:
            for rect in self.rectangles():
                if rect.contains(event.position()):
                    self.seek_requested.emit(float((event.position().x()-rect.left())/rect.width()*self.take.time[-1]))
                    break
