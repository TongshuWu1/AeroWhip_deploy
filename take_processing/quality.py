"""Native-sample diagnostics, not fitting eligibility or automatic cleaning."""
import numpy as np
from PySide6.QtCore import Qt, Signal
from PySide6.QtGui import QColor, QPainter, QPen
from PySide6.QtWidgets import QWidget


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


class QualityTimeline(QWidget):
    seek_requested = Signal(float)

    def __init__(self):
        super().__init__()
        self.take = None
        self.current = 0.
        self.first = self.last = 0
        self.setMinimumHeight(72)
        self.setMaximumHeight(90)
        self.setToolTip("Pose and cable validity; red marks show missing samples or timestamp gaps. Click to seek.")

    def set_take(self, take):
        self.take = take
        self.pose, self.markers, self.gaps = masks(take)
        self.first, self.last = 0, len(take.time)-1
        self.update()

    def paintEvent(self, event):
        painter = QPainter(self)
        painter.fillRect(self.rect(), self.palette().window())
        if self.take is None:
            return
        left, width = 52, max(1, self.width()-64)
        time = self.take.time
        def x(t):
            return int(left + width*t/time[-1])
        for row, (label, valid) in enumerate((("Pose", self.pose), ("Cable", self.markers.all(1)))):
            y = 13+row*23
            painter.setPen(self.palette().text().color())
            painter.drawText(0, y+9, label)
            painter.fillRect(left, y, width, 10, QColor("#5e9f8b"))
            for i in np.flatnonzero(~valid):
                end = time[min(i+1, len(time)-1)]
                painter.fillRect(x(time[i]), y, max(2, x(end)-x(time[i])), 10, QColor("#cf6464"))
            for i in np.flatnonzero(self.gaps):
                painter.fillRect(x(time[i]), y, max(2, x(time[i+1])-x(time[i])), 10, QColor("#cf6464"))
        painter.fillRect(left, 8, max(0, x(time[self.first])-left), 43, QColor(100, 100, 100, 110))
        painter.fillRect(x(time[self.last]), 8, max(0, left+width-x(time[self.last])), 43, QColor(100, 100, 100, 110))
        painter.setPen(QPen(QColor("#e4b13e"), 2))
        for t in (time[self.first], time[self.last]):
            painter.drawLine(x(t), 7, x(t), 53)
        painter.setPen(QPen(self.palette().text().color(), 2))
        painter.drawLine(x(self.current), 5, x(self.current), 56)

    def mousePressEvent(self, event):
        if self.take is not None and event.button() == Qt.MouseButton.LeftButton:
            fraction = np.clip((event.position().x()-52)/max(1, self.width()-64), 0, 1)
            self.seek_requested.emit(float(fraction*self.take.time[-1]))
