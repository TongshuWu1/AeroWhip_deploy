"""Whipping diagnostics tab: saved arrays only, with CPU-only Qt rendering."""
from time import perf_counter

import numpy as np
from PySide6 import QtCore, QtGui, QtWidgets

from training.ui.whip_diagnostics_data import WhipDiagnosticsData, recording_choices


class DiagnosticCanvas(QtWidgets.QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.data = None
        self.index = 0
        self.heat = None
        self.setMinimumSize(580, 470)
        self.setSizePolicy(QtWidgets.QSizePolicy.Policy.Expanding, QtWidgets.QSizePolicy.Policy.Expanding)

    def set_data(self, data):
        self.data, self.index, self.heat = data, 0, None
        if data is not None:
            values = data.density if data.density is not None else data.joint_turn_deg
            self.heat_max = max(float(np.max(values)), 1e-6)
            colors = np.array([[15, 20, 40], [53, 40, 102], [136, 70, 134], [225, 125, 89], [255, 234, 150]], float)
            x = np.clip(values / self.heat_max, 0, 1) * 4
            lo = np.minimum(x.astype(int), 3)
            rgb = colors[lo]*(1-(x-lo))[..., None] + colors[lo+1]*(x-lo)[..., None]
            rgb = np.ascontiguousarray(rgb.transpose(1, 0, 2)[::-1].astype(np.uint8))
            self.heat = QtGui.QImage(rgb.data, rgb.shape[1], rgb.shape[0], rgb.strides[0], QtGui.QImage.Format.Format_RGB888).copy()
            # Cache material-track polylines; IDs are geometry associations, not whip numbers.
            self.tracks = {}
            for i, rows in enumerate(data.regions):
                for region in rows:
                    if set(region['flags']) & {'plateau', 'ambiguous_identity'}:
                        continue
                    self.tracks.setdefault(region['track_id'], []).append((i, region['centre_m']))
        self.update()

    def paintEvent(self, event):
        p = QtGui.QPainter(self)
        p.setRenderHint(QtGui.QPainter.RenderHint.Antialiasing)
        p.fillRect(self.rect(), QtGui.QColor('white'))
        if self.data is None:
            p.setPen(QtGui.QColor('#607486'))
            p.drawText(self.rect(), QtCore.Qt.AlignmentFlag.AlignCenter, 'Select a saved motion to inspect whipping diagnostics.')
            return
        d, i = self.data, self.index
        w, h = self.width(), self.height()
        def panel(x, y, width, height, title):
            p.setPen(QtGui.QColor('#213547'))
            p.drawText(QtCore.QRectF(x+5, y, width-10, 24), QtCore.Qt.AlignmentFlag.AlignCenter, title)
            return QtCore.QRectF(x+46, y+30, width-62, height-64)
        for col, dim, title in [(0, 2, 'Side view · X–Z (m)'), (1, 1, 'Top view · X–Y (m)')]:
            r = panel(col*w/2, 0, w/2, h*.46, title)
            span = max(1.05*d.length, float(np.max(abs(d.relative_positions)))*1.05)
            unit = min(r.width(), r.height())/(2*span)
            r = QtCore.QRectF(r.center().x()-unit*span, r.center().y()-unit*span, 2*unit*span, 2*unit*span)
            def xy(x, y):
                return QtCore.QPointF(r.center().x()+unit*x, r.center().y()-unit*y)
            for k in (-1, -.5, 0, .5, 1):
                p.setPen(QtGui.QPen(QtGui.QColor('#e2e8ef'), 1))
                p.drawLine(xy(k*span,-span),xy(k*span,span))
                p.drawLine(xy(-span,k*span),xy(span,k*span))
                p.setPen(QtGui.QColor('#607486'))
                pt = xy(k*span,-span)
                p.drawText(QtCore.QRectF(pt.x()-20,pt.y()+3,40,18), QtCore.Qt.AlignmentFlag.AlignCenter, f'{k*span:.1f}')
            p.setPen(QtGui.QColor('#607486'))
            p.drawText(QtCore.QRectF(r.left()-42,r.top()-6,39,18),QtCore.Qt.AlignmentFlag.AlignRight,f'{span:.1f}')
            p.drawText(QtCore.QRectF(r.left()-42,r.bottom()-12,39,18),QtCore.Qt.AlignmentFlag.AlignRight,f'{-span:.1f}')
            p.save();p.setClipRect(r.adjusted(-2,-2,2,2))
            pts = [xy(v[0],v[dim]) for v in d.relative_positions[i]]
            p.setPen(QtGui.QPen(QtGui.QColor('#087d79'), 2.5))
            p.drawPolyline(QtGui.QPolygonF(pts))
            p.setBrush(QtGui.QColor('#087d79'))
            for pt in pts: p.drawEllipse(pt, 2.6, 2.6)
            p.setBrush(QtGui.QColor('#213547')); p.drawRect(QtCore.QRectF(pts[0].x()-4,pts[0].y()-4,8,8))
            p.setPen(QtGui.QPen(QtGui.QColor('#df9137'), 1.7));p.setBrush(QtCore.Qt.BrushStyle.NoBrush)
            nodes = np.r_[0.,d.rest.cumsum()]
            for b in d.regions[i]:
                center = [np.interp(b['centre_m'],nodes,d.relative_positions[i,:,axis]) for axis in (0,dim)]
                p.drawEllipse(xy(*center), 6, 6)
            p.setPen(QtGui.QPen(QtGui.QColor('#c94b52'),2));p.setBrush(QtGui.QColor('#c94b52'))
            p.drawEllipse(pts[-1],4,4)
            tip=d.relative_positions[i,-1];v=d.relative_velocities[i,-1]
            end=xy(tip[0]+.05*v[0],tip[dim]+.05*v[dim]);p.drawLine(pts[-1],end)
            delta=end-pts[-1];norm=np.hypot(delta.x(),delta.y())
            if norm>2:
                ux,uy=delta.x()/norm,delta.y()/norm
                p.drawPolygon(QtGui.QPolygonF([end, end-QtCore.QPointF(7*ux+3*uy,7*uy-3*ux),end-QtCore.QPointF(7*ux-3*uy,7*uy+3*ux)]))
            p.restore()
        unit_label='bend density (rad/m)' if d.density is not None else 'joint turning (deg)'
        r=panel(0,h*.46,w,h*.25,f'Material bending map · {unit_label} · 0 → {self.heat_max:.2g}')
        p.drawImage(r,self.heat)
        start,end=float(d.time[0]),float(d.time[-1]);duration=max(end-start,1e-9)
        def tx(time): return r.left()+r.width()*(time-start)/duration
        grid = d.grid if d.grid is not None else np.r_[0.,d.rest.cumsum()][1:-1]
        low,high=float(grid[0]),float(grid[-1])
        def sy(s): return r.bottom()-r.height()*(s-low)/max(high-low,1e-9)
        p.save();p.setClipRect(r)
        p.setPen(QtGui.QPen(QtGui.QColor('#ffffff'), .9))
        for rows in self.tracks.values():
            path=QtGui.QPainterPath();previous=-2
            for index,s in rows:
                pt=QtCore.QPointF(tx(d.time[index]),sy(s))
                path.lineTo(pt) if index==previous+1 else path.moveTo(pt)
                previous=index
            p.drawPath(path)
        for e in d.events:
            if start<=e['time_s']<=end and e['kind'] in ('armed','propagated','opening_confirmed'):
                x=tx(e['time_s']);p.setPen(QtGui.QPen(QtGui.QColor('#fcbe64'),1,QtCore.Qt.PenStyle.DashLine))
                p.drawLine(QtCore.QPointF(x,r.top()),QtCore.QPointF(x,r.bottom()))
        p.setPen(QtGui.QPen(QtGui.QColor('#3cf4d6'),2))
        p.drawLine(QtCore.QPointF(tx(d.time[i]),r.top()),QtCore.QPointF(tx(d.time[i]),r.bottom()))
        p.restore();p.setPen(QtGui.QColor('#607486'))
        for time in np.linspace(start,end,5):
            p.drawText(QtCore.QRectF(tx(time)-22,r.bottom()+2,44,18),QtCore.Qt.AlignmentFlag.AlignCenter,f'{time:.2f}')
        p.drawText(QtCore.QRectF(2,r.top(),40,18),QtCore.Qt.AlignmentFlag.AlignRight,f'{high:.2f} m')
        p.drawText(QtCore.QRectF(2,r.bottom()-18,40,18),QtCore.Qt.AlignmentFlag.AlignRight,f'{low:.2f} m')
        self.curves(p,panel(0,h*.71,w/2,h*.29,'Reach / height · attachment-relative (m)'),
                    [(d.reach*d.length,'#087d79'),(d.tip_height,'#7352bd')], ('teal: positive +X reach   purple: tip height' if d.mode in ('continuous_directional_whip_full_horizon_v5', 'continuous_directional_whip_four_second_v6') else 'teal: horizontal reach   purple: tip height'))
        self.curves(p,panel(w/2,h*.71,w/2,h*.29,('+X velocity (m/s)' if d.mode in ('continuous_directional_whip_full_horizon_v5', 'continuous_directional_whip_four_second_v6') else 'Outward radial velocity (m/s)')),
                    [(d.world_outward,'#2877c7'),(d.relative_outward,'#d17b25')], 'blue: world   orange: relative   negative = inward')

    def curves(self,p,r,series,legend):
        d=self.data;finite=np.concatenate([a[np.isfinite(a)] for a,_ in series])
        low=min(0.,float(finite.min())) if len(finite) else 0.
        high=max(0.,float(finite.max())) if len(finite) else 0.
        margin=max((high-low)*.1,.05);low-=margin;high+=margin
        start,end=d.time[0],d.time[-1];duration=max(end-start,1e-9)
        def point(t,value):return QtCore.QPointF(r.left()+r.width()*(t-start)/duration,r.bottom()-r.height()*(value-low)/(high-low))
        p.setPen(QtGui.QPen(QtGui.QColor('#ccd8e2'),1));p.drawRect(r)
        p.drawLine(point(start,0),point(end,0))
        for values,color in series:
            path=QtGui.QPainterPath();active=False
            for t,value in zip(d.time,values):
                if not np.isfinite(value):active=False;continue
                pt=point(t,value)
                path.lineTo(pt) if active else path.moveTo(pt)
                active=True
            p.setPen(QtGui.QPen(QtGui.QColor(color),1.7));p.drawPath(path)
        p.setPen(QtGui.QPen(QtGui.QColor('#213547'),1,QtCore.Qt.PenStyle.DashLine))
        p.drawLine(point(d.time[self.index],low),point(d.time[self.index],high))
        p.setPen(QtGui.QColor('#607486'))
        for value in (low,high):
            pt=point(start,value);p.drawText(QtCore.QRectF(r.left()-43,pt.y()-8,38,18),QtCore.Qt.AlignmentFlag.AlignRight,f'{value:.2g}')
        p.drawText(QtCore.QRectF(r.left(),r.bottom()+2,r.width(),18),QtCore.Qt.AlignmentFlag.AlignCenter,f'{start:.2f} — {end:.2f} s')
        font=p.font();font.setPointSize(8);p.setFont(font)
        p.drawText(QtCore.QRectF(r.left()-30,r.bottom()+18,r.width()+60,16),QtCore.Qt.AlignmentFlag.AlignCenter,legend)


class WhippingDiagnostics(QtWidgets.QWidget):
    def __init__(self,parent=None):
        super().__init__(parent)
        self.directory=None;self.data=None;self.loaded_key=None;self.playing=False;self.index=0
        layout=QtWidgets.QVBoxLayout(self)
        row=QtWidgets.QHBoxLayout()
        self.choices=QtWidgets.QComboBox();self.choices.setMinimumWidth(320)
        self.choices.setSizeAdjustPolicy(QtWidgets.QComboBox.SizeAdjustPolicy.AdjustToMinimumContentsLengthWithIcon)
        self.choices.currentIndexChanged.connect(self.load_selected)
        self.follow=QtWidgets.QCheckBox('Follow latest evaluation');self.follow.setChecked(True)
        self.follow.setToolTip('Prefer the latest saved evaluation; use the newest training sample if no evaluation exists.')
        self.follow.toggled.connect(self.follow_latest)
        row.addWidget(self.choices,1);row.addWidget(self.follow)
        refresh=QtWidgets.QPushButton('Reload');refresh.clicked.connect(lambda:self.load_selected(force=True));row.addWidget(refresh)
        layout.addLayout(row)
        self.caption=QtWidgets.QLabel('Saved-state diagnostics · not live simulation');self.caption.setWordWrap(True);layout.addWidget(self.caption)
        body=QtWidgets.QHBoxLayout();self.canvas=DiagnosticCanvas();body.addWidget(self.canvas,1)
        side=QtWidgets.QWidget();sidebar=QtWidgets.QVBoxLayout(side);sidebar.setContentsMargins(4,0,4,0)
        sidebar.setSizeConstraint(QtWidgets.QLayout.SizeConstraint.SetMinimumSize)
        self.outcome=QtWidgets.QLabel();self.outcome.setWordWrap(True);self.outcome.setTextInteractionFlags(QtCore.Qt.TextInteractionFlag.TextSelectableByMouse);sidebar.addWidget(self.outcome)
        self.telemetry=QtWidgets.QLabel();self.telemetry.setWordWrap(True);sidebar.addWidget(self.telemetry)
        sidebar.addWidget(QtWidgets.QLabel('Saved detector timeline · click to seek'))
        self.timeline=QtWidgets.QTableWidget(0,2);self.timeline.setHorizontalHeaderLabels(['Time (s)','Recorded decision'])
        self.timeline.setEditTriggers(QtWidgets.QAbstractItemView.EditTrigger.NoEditTriggers)
        self.timeline.setSelectionBehavior(QtWidgets.QAbstractItemView.SelectionBehavior.SelectRows)
        self.timeline.horizontalHeader().setSectionResizeMode(0,QtWidgets.QHeaderView.ResizeMode.ResizeToContents)
        self.timeline.horizontalHeader().setSectionResizeMode(1,QtWidgets.QHeaderView.ResizeMode.Stretch)
        self.timeline.verticalHeader().hide();self.timeline.cellClicked.connect(self.jump_event)
        self.timeline.setMinimumHeight(175);sidebar.addWidget(self.timeline,1)
        details=QtWidgets.QScrollArea();details.setWidgetResizable(True);details.setFixedWidth(290)
        details.setFrameShape(QtWidgets.QFrame.Shape.NoFrame);details.setWidget(side)
        body.addWidget(details);layout.addLayout(body,1)
        controls=QtWidgets.QHBoxLayout()
        self.play=QtWidgets.QPushButton('Play');self.play.clicked.connect(self.toggle);controls.addWidget(self.play)
        for label,callback in [('Step',lambda:self.seek(self.index+1)),('Reset',lambda:self.seek(0))]:
            button=QtWidgets.QPushButton(label);button.clicked.connect(callback);controls.addWidget(button)
        self.speed=QtWidgets.QComboBox()
        for speed in (.1,.25,.5,1.,2.):self.speed.addItem(f'{speed:g}×',speed)
        self.speed.setCurrentIndex(1);controls.addWidget(self.speed)
        self.slider=QtWidgets.QSlider(QtCore.Qt.Orientation.Horizontal);self.slider.valueChanged.connect(self.seek);controls.addWidget(self.slider,1)
        self.clock=QtWidgets.QLabel('—');controls.addWidget(self.clock);layout.addLayout(controls)
        note=QtWidgets.QLabel('Attachment-centred views; red arrow = relative tip velocity. Orange circles / white tracks = bend centres, not automatically folds.\nFull-history plots are offline context. Recorded opening or validity is not proof of useful horizontal whipping.')
        note.setWordWrap(True);note.setStyleSheet('color:#607486;font-size:11px');layout.addWidget(note)
        self.timer=QtCore.QTimer(self);self.timer.setInterval(30);self.timer.timeout.connect(self.tick)

    def set_run(self,directory,history):
        choices=recording_choices(directory,history)
        previous=self.choices.currentData() if directory==self.directory else None
        signature=(str(directory),tuple(choices))
        if signature==getattr(self,'signature',None):return
        self.signature=signature;self.directory=directory
        self.choices.blockSignals(True);self.choices.clear()
        for label,name in choices:self.choices.addItem(label,name)
        index=0 if self.follow.isChecked() else next((i for i,(_,name) in enumerate(choices) if name==previous),0)
        self.choices.setCurrentIndex(index if choices else -1);self.choices.blockSignals(False)
        if self.isVisible():self.load_selected()

    def follow_latest(self,checked):
        if checked:self.choices.setCurrentIndex(0)

    def load_selected(self,*_,force=False):
        if not self.isVisible():return
        filename=self.choices.currentData()
        if self.choices.currentIndex()>0:self.follow.setChecked(False)
        key=(str(self.directory),filename)
        if key==self.loaded_key and not force:return
        self.pause();self.data=None;self.canvas.set_data(None);self.timeline.setRowCount(0)
        self.outcome.clear();self.telemetry.clear();self.clock.setText('—')
        self.loaded_key=None
        if not filename:
            self.caption.setText('No individual saved motions in this run yet. No physics or evaluation will be started.');return
        self.caption.setText('Reading saved motion…')
        try:
            self.data=WhipDiagnosticsData.load(self.directory,filename)
        except (OSError,ValueError,KeyError,TypeError) as exc:
            self.caption.setText(f'Cannot display this recording: {exc}');return
        self.loaded_key=key;d=self.data;self.canvas.set_data(d)
        self.caption.setText(f'{self.choices.currentText()} · {filename} · {d.mode}\n'+ ' '.join(d.notices))
        self.caption.setToolTip(str(d.path))
        valid='valid' if d.validity is True else 'INVALID' if d.validity is False else 'validity not recorded'
        qualified=d.record.get('qualified')
        qualification='yes' if qualified is True else 'no' if qualified is False else 'not recorded'
        selected=d.record.get('selected')
        self.outcome.setText(f"Saved final outcome: {d.record.get('terminal','not recorded')} · {valid}\nSaved task qualification: {qualification}\nSelected output: {'recorded' if selected is not None else 'none recorded'}\n{len(d.rest)} links · rest length {d.length:.4f} m")
        self.outcome.setMinimumHeight(max(self.outcome.heightForWidth(250),len(self.outcome.text().splitlines())*self.outcome.fontMetrics().lineSpacing())+12)
        for row,e in enumerate(d.events):
            self.timeline.insertRow(row)
            self.timeline.setItem(row,0,QtWidgets.QTableWidgetItem(f"{e['time_s']:.3f}"))
            self.timeline.setItem(row,1,QtWidgets.QTableWidgetItem(e['kind'].replace('_',' ')))
        self.slider.setRange(0,len(d.time)-1);self.seek(0)

    def seek(self,index):
        if self.data is None:return
        self.index=max(0,min(int(index),len(self.data.time)-1));d=self.data;i=self.index
        self.slider.blockSignals(True);self.slider.setValue(i);self.slider.blockSignals(False)
        self.canvas.index=i;self.canvas.update();self.clock.setText(f'{d.time[i]:.2f} / {d.time[-1]:.2f} s')
        def number(value):return f'{value:+.2f}' if np.isfinite(value) else 'undefined'
        def angle(cosine):return f'{np.degrees(np.arccos(np.clip(cosine,-1,1))):.1f}°' if np.isfinite(cosine) else 'undefined'
        rows=d.regions[i];b=max(rows,key=lambda b:b['theta']) if rows else None
        bend=(f"Largest bend: {np.degrees(b['theta']):.1f}° total turn\nCentre: {b['centre_m']:.3f} m · coherence {b['coherence']:.2f}" if b else 'No reconstructed bend centre')
        self.telemetry.setText(f"At cursor: {d.phase(i)}\n\nHorizontal reach: {100*d.reach[i]:.1f}% L\nTip height vs attachment: {d.tip_height[i]:+.3f} m\nTip speed world / relative:\n{d.world_speed[i]:.2f} / {d.relative_speed[i]:.2f} m/s\nOutward world / relative:\n{number(d.world_outward[i])} / {number(d.relative_outward[i])} m/s\nAngle to horizontal outward:\nworld {angle(d.world_cosine[i])} · relative {angle(d.relative_cosine[i])}\n\n{bend}")
        if d.continuous:
            extra='\n\nSaved continuous reward trace'
            for key,label in [('fold_support','Fold support'),('straightening_mps','Endpoint extension (m/s)'),
                              ('reward_rate','Reward rate'),('accrued_credit','Accrued credit'),
                              ('workspace_cost','Workspace cost'),('net_return','Net return')]:
                if key in d.continuous:extra+=f'\n{label}: {number(d.continuous[key][i])}'
            self.telemetry.setText(self.telemetry.text()+extra)
        self.telemetry.setMinimumHeight(max(self.telemetry.heightForWidth(250),len(self.telemetry.text().splitlines())*self.telemetry.fontMetrics().lineSpacing())+12)
        self.play_time=float(d.time[i]);self.last_tick=perf_counter()

    def jump_event(self,row,_column):
        if self.data is not None:self.pause();self.seek(np.argmin(abs(self.data.time-self.data.events[row]['time_s'])))

    def toggle(self):
        if self.playing:self.pause();return
        if self.data is None:return
        if self.index==len(self.data.time)-1:self.seek(0)
        self.playing=True;self.play.setText('Pause');self.last_tick=perf_counter();self.timer.start()

    def pause(self):
        self.playing=False;self.play.setText('Play');self.timer.stop()

    def tick(self):
        if self.data is None or not self.playing:return
        now=perf_counter();target=self.play_time+(now-self.last_tick)*self.speed.currentData()
        index=min(len(self.data.time)-1,int(np.searchsorted(self.data.time,target,side='right')-1))
        self.seek(index);self.play_time=target;self.last_tick=now
        if target>=self.data.time[-1]:self.pause()

    def showEvent(self,event):
        super().showEvent(event);self.load_selected()

    def hideEvent(self,event):
        self.pause();super().hideEvent(event)
