"""Small Qt line chart for recorded training metrics; no plotting dependency."""

import math
from PySide6 import QtCore, QtGui, QtWidgets


class MetricPlot(QtWidgets.QWidget):
    COLORS = ('#2877c7', '#269b76', '#cc7a28', '#9761bc')

    def __init__(self, title, *, limits=None, parent=None):
        super().__init__(parent)
        self.title, self.limits = title, limits
        self.series = []
        self.setMinimumSize(320, 175)
        self.setSizePolicy(QtWidgets.QSizePolicy.Policy.Expanding, QtWidgets.QSizePolicy.Policy.Expanding)

    def set_series(self, series):
        # Each series contains (label, [(iteration, value)], dashed). None is a gap.
        self.series = series
        self.update()

    def paintEvent(self, event):
        painter = QtGui.QPainter(self)
        painter.setRenderHint(QtGui.QPainter.RenderHint.Antialiasing)
        painter.fillRect(self.rect(), self.palette().base())
        painter.setPen(self.palette().text().color())
        font = painter.font()
        font.setBold(True)
        painter.setFont(font)
        painter.drawText(14, 23, self.title)
        font.setBold(False)
        painter.setFont(font)
        legend_x = 14
        for i, (label, rows, dashed) in enumerate(self.series):
            latest = next((y for _, y in reversed(rows) if y is not None and math.isfinite(y)), None)
            legend = f'{label} {latest:.4g}' if latest is not None else label
            color = QtGui.QColor(self.COLORS[i % len(self.COLORS)])
            pen = QtGui.QPen(color, 2)
            if dashed:
                pen.setStyle(QtCore.Qt.PenStyle.DashLine)
            painter.setPen(pen)
            painter.drawLine(legend_x, 43, legend_x+17, 43)
            painter.setPen(self.palette().text().color())
            painter.drawText(legend_x+23, 47, legend)
            legend_x += painter.fontMetrics().horizontalAdvance(legend)+42
        rect = QtCore.QRectF(66, 66, max(1, self.width()-86), max(1, self.height()-104))
        points = [(x, y) for _, rows, _ in self.series for x, y in rows
                  if y is not None and math.isfinite(y)]
        if not points:
            painter.setPen(self.palette().placeholderText().color())
            painter.drawText(rect, QtCore.Qt.AlignmentFlag.AlignCenter, 'No completed samples')
            return
        xmax = max(1, max(x for x, _ in points))
        if self.limits is None:
            lo, hi = min(y for _, y in points), max(y for _, y in points)
            margin = max((hi-lo)*.12, 1e-5, abs(hi)*.02)
            lo, hi = lo-margin, hi+margin
        else:
            lo, hi = self.limits
        for i in range(5):
            y = rect.bottom()-rect.height()*i/4
            painter.setPen(self.palette().midlight().color())
            painter.drawLine(QtCore.QPointF(rect.left(), y), QtCore.QPointF(rect.right(), y))
            painter.setPen(self.palette().text().color())
            painter.drawText(QtCore.QRectF(0, y-9, 58, 18), QtCore.Qt.AlignmentFlag.AlignRight,
                             f'{lo+(hi-lo)*i/4:.3g}')
        ticks = range(int(xmax)+1) if xmax <= 5 else sorted({round(xmax*i/4) for i in range(5)})
        for tick in ticks:
            x = rect.left()+rect.width()*tick/xmax
            painter.drawText(QtCore.QRectF(x-22, rect.bottom()+8, 44, 18),
                             QtCore.Qt.AlignmentFlag.AlignCenter, str(tick))
        painter.drawText(QtCore.QRectF(rect.left(), self.height()-20, rect.width(), 18),
                         QtCore.Qt.AlignmentFlag.AlignCenter, 'Training iteration')
        for i, (_, rows, dashed) in enumerate(self.series):
            path, active = QtGui.QPainterPath(), False
            color = QtGui.QColor(self.COLORS[i % len(self.COLORS)])
            painter.setPen(QtGui.QPen(color, 1.8, QtCore.Qt.PenStyle.DashLine if dashed else QtCore.Qt.PenStyle.SolidLine))
            for x, y in rows:
                if y is None or not math.isfinite(y):
                    active = False
                    continue
                point = QtCore.QPointF(rect.left()+rect.width()*x/xmax, rect.bottom()-rect.height()*(y-lo)/(hi-lo))
                path.lineTo(point) if active else path.moveTo(point)
                active = True
                painter.drawEllipse(point, 2.6, 2.6)
            painter.drawPath(path)
