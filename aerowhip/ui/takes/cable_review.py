"""Review and correct automatic cable-marker labels of one calibration flight.

A 3D view shows the frame's marker cloud: labelled cable markers c1..c10 as coloured
spheres joined from the attachment, unlabelled points grey, the drone as a box in its
measured attitude. Markers missing in the frame are drawn as open spheres at an estimate
between their visible neighbours; the estimate is for display only and never saved. The
camera follows the attachment and keeps the angle you turn it to. Below, a timeline shows
which markers are labelled in each frame (missing ones are normal: the cable is still
constrained by the others), the flagged runs (red) and the chain-length error of the labels;
the runs are listed on the right.

Markers corrected by a person are ringed blue. Mouse: left drag rotates, right drag or wheel zooms, middle drag pans; a left click on a
point selects it. Keys: Left/Right one frame, Shift +10, PageUp/PageDown 100, N/P next/
previous flagged run, Space play/pause, 1..9, 0 label the selected point c1..c10, Delete
marks the selected marker not visible in this frame, Backspace returns it to the automatic
label, A accepts the current flagged run as checked, R resets the view, Ctrl+S saves.
"""
import numpy as np
import pyvista as pv
import vtk
from matplotlib import colormaps
from matplotlib.colors import ListedColormap
from pyvistaqt import QtInteractor
from PySide6 import QtCore, QtGui, QtWidgets

from aerowhip.takes import cable_labels as cl
from aerowhip.ui import style
from aerowhip.ui.plotting import PlotCanvas
from aerowhip.ui.style import ROLE

# Marker colour runs along the cable (c1 at the drone dark, c10 at the tip light).
COLOURS = [colormaps['viridis'](x, bytes=False) for x in np.linspace(0, .92, 10)]
COLOURS = ['#%02x%02x%02x' % tuple(round(255*c) for c in rgba[:3]) for rgba in COLOURS]
MARKER_RADIUS_M = .014
PICK_RADIUS_M = .05
DRONE_BOX_M = (.1, .1, .03)


class CableReview(QtWidgets.QMainWindow):
    def __init__(self, folder):
        super().__init__()
        self.flight = cl.load_flight(folder)
        labels = cl.load_labels(self.flight)
        self.labeller = cl.AutoLabeller(self.flight, labels)
        if labels is None:
            self.labeller.run()  # saved labels open as they are; tracker states are rebuilt on the first edit
        self.frame, self.selected, self.dirty = 0, None, False
        self.setWindowTitle(f'Cable labels · {self.flight.folder.name}')
        self._build()
        self._build_scene()
        self._refresh_flags()
        self._draw()
        self._reset_view()

    # ------------------------------------------------------------------ layout
    def _build(self):
        self.plotter = QtInteractor(self, auto_update=False, multi_samples=4)
        # The view's own one-letter keys (w, s, a, q, ...) would fight the shortcuts below.
        self.plotter.iren.interactor.RemoveObservers('CharEvent')
        self.plotter.track_click_position(self._click, side='left')

        self.timeline_canvas = PlotCanvas(self, minimum=(400, 210))
        self.timeline_canvas.setFixedHeight(230)
        self.timeline_canvas.seek_requested.connect(self._timeline_seek)

        self.info = QtWidgets.QLabel()
        self.info.setTextFormat(QtCore.Qt.RichText)
        self.runs = QtWidgets.QListWidget()
        self.runs.itemClicked.connect(self._jump_to_run)
        self.runs.itemActivated.connect(self._jump_to_run)
        accept = QtWidgets.QPushButton('Accept run (A)')
        accept.clicked.connect(self._accept_run)
        save = QtWidgets.QPushButton('Save (Ctrl+S)')
        save.clicked.connect(self._save)
        reset = QtWidgets.QPushButton('Reset view (R)')
        reset.clicked.connect(self._reset_view)
        self.show_free = QtWidgets.QCheckBox('Show unlabelled points')
        self.show_free.setChecked(True)
        self.show_free.toggled.connect(lambda _: self._draw())
        self.show_estimate = QtWidgets.QCheckBox('Show estimates for missing markers')
        self.show_estimate.setChecked(True)
        self.show_estimate.toggled.connect(lambda _: self._draw())
        legend = QtWidgets.QLabel(' '.join(f'<span style="color:{c}">●</span>c{k+1}' for k, c in enumerate(COLOURS)))
        legend.setTextFormat(QtCore.Qt.RichText)
        help_text = QtWidgets.QLabel('Left drag rotate · wheel zoom · middle drag pan · click selects a point<br>'
                                     '1–9,0 label c1–c10 · Del not visible · Backspace automatic<br>'
                                     '←/→ frame · Shift ±10 · PgUp/PgDn ±100 · N/P runs · Space play · A accept')
        help_text.setTextFormat(QtCore.Qt.RichText)
        help_text.setWordWrap(True)
        side = QtWidgets.QVBoxLayout()
        for widget in (self.info, legend, self.show_free, self.show_estimate, reset):
            side.addWidget(widget)
        side.addWidget(QtWidgets.QLabel('Flagged runs, not yet checked'))
        side.addWidget(self.runs, 1)
        for widget in (accept, save, help_text):
            side.addWidget(widget)
        panel = QtWidgets.QWidget(); panel.setLayout(side); panel.setFixedWidth(360)

        self.slider = QtWidgets.QSlider(QtCore.Qt.Horizontal)
        self.slider.setRange(0, self.flight.frames-1)
        self.slider.valueChanged.connect(self._goto)
        left = QtWidgets.QVBoxLayout()
        left.addWidget(self.plotter.interactor, 1)
        left.addWidget(self.timeline_canvas)
        left.addWidget(self.slider)
        holder = QtWidgets.QWidget(); holder.setLayout(left)
        main = QtWidgets.QHBoxLayout()
        main.addWidget(holder, 1); main.addWidget(panel)
        central = QtWidgets.QWidget(); central.setLayout(main)
        self.setCentralWidget(central)
        self.resize(1600, 1000)
        style.apply(self)

        self.timer = QtCore.QTimer(self)
        self.timer.timeout.connect(lambda: self._goto(min(self.frame+3, self.flight.frames-1)))
        shortcuts = {'Right': lambda: self._goto(self.frame+1), 'Left': lambda: self._goto(self.frame-1),
                     'Shift+Right': lambda: self._goto(self.frame+10), 'Shift+Left': lambda: self._goto(self.frame-10),
                     'PgDown': lambda: self._goto(self.frame+100), 'PgUp': lambda: self._goto(self.frame-100),
                     'N': lambda: self._next_run(1), 'P': lambda: self._next_run(-1), 'R': self._reset_view,
                     'Space': self._play, 'A': self._accept_run, 'Ctrl+S': self._save,
                     'Delete': lambda: self._edit(None), 'Backspace': self._clear, 'Escape': self._deselect}
        for key in range(10):
            shortcuts[str(key)] = (lambda k: lambda: self._edit((k-1) % 10))(key)
        for key, action in shortcuts.items():
            shortcut = QtGui.QShortcut(QtGui.QKeySequence(key), self, action)
            shortcut.setContext(QtCore.Qt.WindowShortcut)

    def _build_scene(self):
        p = self.plotter
        p.set_background('#f6f7f9')
        floor = pv.Plane(center=(0, 0, 0), i_size=4, j_size=3, i_resolution=8, j_resolution=6)
        p.add_mesh(floor, style='wireframe', color='#c8c8c8', line_width=1, pickable=False)
        p.add_mesh(pv.Plane(center=(0, 0, -.002), i_size=4, j_size=3), color='#e9ecef', lighting=False,
                   pickable=False)
        p.add_axes(color='black')
        self.free = pv.PolyData(np.zeros((1, 3)))
        self.free_actor = p.add_mesh(self.free, color='#7a7a7a', point_size=9, render_points_as_spheres=True)
        self.chain = pv.PolyData(np.zeros((2, 3)), lines=[2, 0, 1])
        p.add_mesh(self.chain, color='#333333', line_width=3, pickable=False)
        sphere = pv.Sphere(radius=MARKER_RADIUS_M, theta_resolution=16, phi_resolution=16)
        self.markers, self.estimates, self.texts = [], [], []
        for k, colour in enumerate(COLOURS):
            self.markers.append(p.add_mesh(sphere.copy(), color=colour, smooth_shading=True))
            self.estimates.append(p.add_mesh(sphere.copy(), color=colour, style='wireframe', line_width=1.5,
                                             opacity=.8, pickable=False))
            text = vtk.vtkBillboardTextActor3D()
            text.SetInput(f'c{k+1}')
            text.GetTextProperty().SetFontSize(14)
            text.GetTextProperty().SetColor(0, 0, 0)
            text.SetDisplayOffset(10, 0)
            text.PickableOff()
            p.renderer.AddActor(text)
            self.texts.append(text)
        ring = pv.Sphere(radius=1.8*MARKER_RADIUS_M, theta_resolution=12, phi_resolution=12)
        self.ring_actors = [p.add_mesh(ring.copy(), color='#0050ff', style='wireframe', line_width=1, opacity=.7,
                                       pickable=False) for _ in range(10)]
        self.selected_actor = p.add_mesh(pv.Sphere(radius=2.4*MARKER_RADIUS_M), color='black', style='wireframe',
                                         line_width=2, pickable=False)
        self.root_actor = p.add_mesh(pv.Cube(x_length=.02, y_length=.02, z_length=.02), color='black', pickable=False)
        self.drone_actor = p.add_mesh(pv.Box(bounds=(-DRONE_BOX_M[0]/2, DRONE_BOX_M[0]/2, -DRONE_BOX_M[1]/2,
                                                     DRONE_BOX_M[1]/2, -DRONE_BOX_M[2], 0.)),
                                      color='#4a6fa5', opacity=.85, pickable=False)
        self.camera_anchor = self.flight.root[0].copy()

    # ------------------------------------------------------------------ data
    def _refresh_flags(self):
        self.reasons = cl.flags(self.flight, self.labeller.labels)
        self.segments = cl.segments(self.reasons, self.labeller.labels.reviewed)
        current = self.runs.currentRow()
        self.runs.clear()
        for first, last, why in self.segments:
            t = self.flight.stamps[first]-self.flight.stamps[0]
            self.runs.addItem(f'{t:6.2f} s  frames {first}–{last} ({last-first+1})  {", ".join(why)[:60]}')
        if 0 <= current < self.runs.count():
            self.runs.setCurrentRow(current)
        self._plot_timeline()

    def _plot_timeline(self):
        """Per-marker label raster and the chain-length error of the labels, over flight time."""
        fl, lab = self.flight, self.labeller.labels
        t = fl.stamps-fl.stamps[0]
        index = lab.index
        seen = index >= 0
        c = self.timeline_canvas
        with c.style():
            c.reset()
            raster, error = c.figure.subplots(2, 1, sharex=True, height_ratios=(1.25, 1))
            raster.imshow(seen.T.astype(float), aspect='auto', interpolation='nearest', origin='upper',
                          cmap=ListedColormap(['#ffffff', '#9fb0c0']), vmin=0, vmax=1,
                          extent=(t[0], t[-1], 9.5, -.5))
            for first, last, _ in self.segments:
                raster.axvspan(t[first], t[min(last+1, len(t)-1)], color=ROLE['error'], alpha=.3, linewidth=0)
            manual = sorted({f for f, _ in lab.manual})
            if manual:
                raster.plot(t[manual], [-.9]*len(manual), 'v', color=ROLE['model'], markersize=4, clip_on=False)
            raster.set_yticks([0, 4, 9], ['c1', 'c5', 'c10'])
            raster.tick_params(axis='y', labelsize=6.5, length=0)
            raster.grid(False)
            raster.set_title(f'Labelled markers (grey) · {seen.sum(1).mean():.2f} of 10 per frame · '
                             'flagged runs red · corrections ▼', fontsize=8.5)
            # Adjacent labelled markers: |distance − cable interval| (both directions; flags use 'too far').
            arc = np.r_[0., fl.arc]
            worst = np.full(fl.frames, np.nan)
            for f in range(fl.frames):
                k = np.flatnonzero(seen[f])
                if len(k) < 2:
                    continue
                q = fl.points[f][index[f, k]]
                gaps = np.linalg.norm(np.diff(q, axis=0), axis=1)
                along = np.diff(arc[k+1])
                adjacent = np.diff(k) == 1
                if adjacent.any():
                    worst[f] = 1000*np.max(np.abs(gaps-along)[adjacent])
            error.plot(t, worst, color=ROLE['measured'], linewidth=.6)
            error.axhline(1000*cl.LENGTH_TOLERANCE_M, color=ROLE['error'], linestyle='--', linewidth=.8,
                          label=f'flag tolerance {1000*cl.LENGTH_TOLERANCE_M:.0f} mm')
            error.set_ylabel('chain error (mm)')
            error.set_xlabel('flight time (s)')
            error.set_ylim(0, max(30., float(np.nanpercentile(worst, 99.5)) if np.isfinite(worst).any() else 30.))
            error.legend(loc='upper right')
            for ax in (raster, error):
                c.add_cursor(ax, float(t[self.frame]))
        c.draw_idle()

    # ------------------------------------------------------------------ drawing
    def _draw(self):
        f, lab, fl = self.frame, self.labeller.labels, self.flight
        points, root = fl.points[f], fl.root[f]
        index = lab.index[f]
        labelled = set(int(i) for i in index if i >= 0)
        free = np.array([q for i, q in enumerate(points) if i not in labelled]).reshape(-1, 3)
        self.free.copy_from(pv.PolyData(free if len(free) else np.zeros((1, 3))))
        self.free_actor.SetVisibility(bool(len(free)) and self.show_free.isChecked())
        estimate = cl.estimate(fl, lab, f)
        chain = np.vstack([root, estimate])
        self.chain.copy_from(pv.lines_from_points(chain))
        for k in range(10):
            seen = index[k] >= 0
            position = points[index[k]] if seen else estimate[k]
            self.markers[k].SetPosition(*position)
            self.markers[k].SetVisibility(bool(seen))
            self.estimates[k].SetPosition(*position)
            self.estimates[k].SetVisibility(bool(not seen and self.show_estimate.isChecked()))
            self.texts[k].SetPosition(*position)
            self.texts[k].SetVisibility(bool(seen or self.show_estimate.isChecked()))
            corrected = seen and (f, k) in lab.manual
            self.ring_actors[k].SetVisibility(bool(corrected))
            if corrected:  # blue ring: corrected by a person
                self.ring_actors[k].SetPosition(*position)
        selected = self.selected is not None and self.selected < len(points)
        self.selected_actor.SetVisibility(selected)
        if selected:
            self.selected_actor.SetPosition(*points[self.selected])
        self.root_actor.SetPosition(*root)
        pose = np.eye(4); pose[:3, :3], pose[:3, 3] = fl.rotation[f], fl.drone[f]
        self.drone_actor.user_matrix = pose
        # Follow the attachment, keeping the camera angle the user chose.
        shift = root-self.camera_anchor
        camera = self.plotter.camera
        camera.position = tuple(np.add(camera.position, shift))
        camera.focal_point = tuple(np.add(camera.focal_point, shift))
        self.camera_anchor = root.copy()
        self.plotter.render()

        self.timeline_canvas.set_cursor(float(fl.stamps[f]-fl.stamps[0]))
        self.slider.blockSignals(True); self.slider.setValue(f); self.slider.blockSignals(False)
        t = fl.stamps[f]-fl.stamps[0]
        missing = [f'c{k+1}' for k in range(10) if index[k] < 0]
        manual_here = sorted(k+1 for (fr, k) in lab.manual if fr == f)
        selection = ''
        if selected:
            k = [m for m in range(10) if index[m] == self.selected]
            selection = f'<br>selected: point {self.selected}' + (f' = <b>c{k[0]+1}</b>' if k else ' (unlabelled)')
        self.info.setText(f'<b>frame {f}</b> / {fl.frames-1} · {t:.2f} s<br>phase: {fl.phases[f]}<br>'
                          f'points: {len(points)} · labelled {10-len(missing)}/10<br>'
                          f'missing: {", ".join(missing) or "–"}<br>manual here: {manual_here or "–"}<br>'
                          f'flags: {"; ".join(self.reasons[f]) or "–"}{selection}<br>'
                          f'{"<b>unsaved changes</b>" if self.dirty else "saved"}')

    def _reset_view(self):
        root = self.flight.root[self.frame]
        centre = root-[0, 0, .5]
        camera = self.plotter.camera
        camera.focal_point = tuple(centre)
        camera.position = tuple(centre+np.array([2.2, -2.6, 1.0]))
        camera.up = (0, 0, 1)
        self.camera_anchor = root.copy()
        self.plotter.render()

    # ------------------------------------------------------------------ actions
    def _goto(self, frame):
        frame = int(np.clip(frame, 0, self.flight.frames-1))
        if frame != self.frame:
            self.selected = None
        self.frame = frame
        if frame == self.flight.frames-1:
            self.timer.stop()
        self._draw()

    def _play(self):
        if self.timer.isActive():
            self.timer.stop()
        else:
            self.timer.start(30)

    def _click(self, position):
        points = self.flight.points[self.frame]
        if position is None or not len(points):
            return
        d = np.linalg.norm(points-np.asarray(position), axis=1)
        i = int(np.argmin(d))
        if d[i] < PICK_RADIUS_M:  # a click on empty space (or the start of a rotation) keeps the selection
            self.selected = i
            self._draw()

    def _timeline_seek(self, time_s):
        t = self.flight.stamps-self.flight.stamps[0]
        self._goto(int(np.clip(np.searchsorted(t, time_s), 0, self.flight.frames-1)))

    def _selected_marker(self):
        index = self.labeller.labels.index[self.frame]
        k = [m for m in range(10) if self.selected is not None and index[m] == self.selected]
        return k[0] if k else None

    def _edit(self, marker):
        """marker = 0..9: label the selected point; None: selected marker not visible here."""
        if marker is None:
            marker = self._selected_marker()
            if marker is None:
                return
            point = -1
        else:
            if self.selected is None:
                return
            point = self.selected
        self._busy(lambda: self.labeller.set_manual(self.frame, marker, point))

    def _clear(self):
        marker = self._selected_marker()
        if marker is None:
            index = self.labeller.labels.index[self.frame]
            manual = [k for (fr, k) in self.labeller.labels.manual if fr == self.frame and index[k] < 0]
            if not manual:
                return
            marker = manual[0]
        self._busy(lambda: self.labeller.clear_manual(self.frame, marker))

    def _busy(self, action):
        QtWidgets.QApplication.setOverrideCursor(QtCore.Qt.WaitCursor)
        try:
            action()
        finally:
            QtWidgets.QApplication.restoreOverrideCursor()
        self.dirty = True
        self._refresh_flags()
        self._draw()

    def _deselect(self):
        self.selected = None
        self._draw()

    def _run_at(self, frame):
        return next((i for i, (a, b, _) in enumerate(self.segments) if a <= frame <= b), None)

    def _next_run(self, step):
        if not self.segments:
            return
        starts = [a for a, _, _ in self.segments]
        if step > 0:
            later = [i for i, a in enumerate(starts) if a > self.frame]
            i = later[0] if later else 0
        else:
            current = self._run_at(self.frame)
            earlier = [i for i, a in enumerate(starts) if a < self.frame and i != current]
            i = earlier[-1] if earlier else len(starts)-1
        self.runs.setCurrentRow(i)
        self._goto(starts[i])

    def _jump_to_run(self, item):
        self._goto(self.segments[self.runs.row(item)][0])

    def _accept_run(self):
        i = self._run_at(self.frame)
        if i is None:
            return
        first, last, _ = self.segments[i]
        self.labeller.labels.reviewed.update(range(first, last+1))
        self.dirty = True
        self._refresh_flags()
        if self.segments:
            self._next_run(1)
        else:
            self._draw()

    def _save(self):
        summary = cl.save(self.flight, self.labeller.labels)
        self.dirty = False
        self.statusBar().showMessage(f"Saved {cl.LABELS_FILE}: {summary['complete_frames']}/{summary['frames']} frames complete, "
                                     f"{summary['unreviewed_flagged_frames']} flagged frames unchecked", 8000)
        self._draw()

    def closeEvent(self, event):
        if self.dirty:
            answer = QtWidgets.QMessageBox.question(self, 'Unsaved labels', 'Save the labels before closing?',
                QtWidgets.QMessageBox.Save | QtWidgets.QMessageBox.Discard | QtWidgets.QMessageBox.Cancel)
            if answer == QtWidgets.QMessageBox.Cancel:
                event.ignore()
                return
            if answer == QtWidgets.QMessageBox.Save:
                self._save()
        self.timer.stop()
        self.plotter.close()
        event.accept()
