"""Whipping diagnostics tab: saved arrays only, with CPU-only Qt rendering."""
from time import perf_counter

import numpy as np
from PySide6 import QtCore, QtWidgets

from aerowhip.ui.plotting import PlotCanvas, message
from aerowhip.ui.style import PALETTE, ROLE

from aerowhip.ui.training.whip_diagnostics_data import WhipDiagnosticsData, recording_choices


class DiagnosticCanvas(PlotCanvas):
    """Side and top views (attachment-centred), bending map over time, reach and +X velocity."""

    def __init__(self, parent=None):
        super().__init__(parent, minimum=(580, 470))
        self.data = None
        self.index = 0
        self.set_data(None)

    def set_data(self, data):
        self.data, self.index = data, 0
        with self.style():
            self.reset()
            if data is None:
                message(self.figure.add_subplot(), 'Select a saved motion to inspect whipping diagnostics.')
                self.draw_idle()
                return
            grid = self.figure.add_gridspec(3, 2, height_ratios=(1.25, .9, .8))
            side, top = self.figure.add_subplot(grid[0, 0]), self.figure.add_subplot(grid[0, 1])
            heat = self.figure.add_subplot(grid[1, :])
            reach = self.figure.add_subplot(grid[2, 0], sharex=heat)
            velocity = self.figure.add_subplot(grid[2, 1], sharex=heat)
            length = float(data.rest.sum())
            span = max(1.05*length, float(np.max(abs(data.relative_positions)))*1.05)
            self.views = []
            nodes = np.r_[0., data.rest.cumsum()]
            for ax, dim, title, ylabel in ((side, 2, 'Side view', 'z (m)'), (top, 1, 'Top view', 'y (m)')):
                ax.set_aspect('equal', adjustable='box')
                ax.set_xlim(-span, span)
                ax.set_ylim(-span, span)
                ax.set_title(title+' · attachment-centred')
                ax.set_xlabel('x (m)')
                ax.set_ylabel(ylabel)
                ax.plot(0, 0, 's', color=ROLE['measured'], markersize=5)
                cable, = ax.plot([], [], '-o', color=ROLE['model'], linewidth=1.6, markersize=2.5)
                tip, = ax.plot([], [], 'o', color=ROLE['error'], markersize=4.5)
                arrow, = ax.plot([], [], '-', color=ROLE['error'], linewidth=1.3)
                bends, = ax.plot([], [], 'o', markerfacecolor='none', markeredgecolor=PALETTE['orange'],
                                 markersize=8, markeredgewidth=1.2)
                for artist in (cable, tip, arrow, bends):
                    self.animate(artist)
                self.views.append((dim, cable, tip, arrow, bends, nodes))
            t = data.time
            if data.density is not None:
                values, s = data.density.T, data.grid
                label = 'bend density (rad/m)'
            else:
                values, s = data.joint_turn_deg.T, nodes[1:-1]
                label = 'joint turn (deg)'
            image = heat.imshow(values, aspect='auto', origin='lower', cmap='magma', interpolation='nearest',
                                extent=(t[0], t[-1], float(s[0]), float(s[-1])), vmin=0)
            colorbar = self.figure.colorbar(image, ax=heat, pad=.01, fraction=.04)
            colorbar.set_label(label)
            tracks = {}
            for i, rows in enumerate(data.regions):
                for region in rows:
                    if set(region['flags']) & {'plateau', 'ambiguous_identity'}:
                        continue
                    tracks.setdefault(region['track_id'], []).append((t[i], region['centre_m']))
            for rows in tracks.values():
                rows = np.asarray(rows)
                heat.plot(rows[:, 0], rows[:, 1], color='white', linewidth=.6, alpha=.8)
            for event in data.events:
                if t[0] <= event['time_s'] <= t[-1] and event['kind'] in ('armed', 'propagated', 'opening_confirmed'):
                    heat.axvline(event['time_s'], color=PALETTE['yellow'], linewidth=.8, linestyle='--')
            heat.set_title('Bending along the cable (white: tracked bend centres)')
            heat.set_ylabel('arc length (m)')
            heat.grid(False)
            reach.plot(t, data.reach*length, color=ROLE['model'], label='+x reach')
            reach.plot(t, data.tip_height, color=PALETTE['purple'], label='tip height')
            reach.axhline(0, color=ROLE['reference'], linewidth=.6)
            reach.set_title('Tip position rel. attachment')
            reach.set_ylabel('m')
            reach.set_xlabel('time (s)')
            reach.legend(loc='best')
            velocity.plot(t, data.world_outward, color=ROLE['model'], label='world')
            velocity.plot(t, data.relative_outward, color=PALETTE['orange'], label='rel. attachment')
            velocity.axhline(0, color=ROLE['reference'], linewidth=.6)
            velocity.set_title('Tip +x velocity')
            velocity.set_ylabel('m/s')
            velocity.set_xlabel('time (s)')
            velocity.legend(loc='best')
            for ax in (heat, reach, velocity):
                self.add_cursor(ax, t[0])
            heat.tick_params(labelbottom=False)
        self.set_index(0)
        self.draw_idle()

    def set_index(self, index):
        d = self.data
        if d is None:
            return
        self.index = i = index
        tip, velocity = d.relative_positions[i, -1], d.relative_velocities[i, -1]
        for dim, cable, tip_dot, arrow, bends, nodes in self.views:
            cable.set_data(d.relative_positions[i, :, 0], d.relative_positions[i, :, dim])
            tip_dot.set_data([tip[0]], [tip[dim]])
            # 50 ms of relative tip velocity, drawn as a segment from the tip.
            arrow.set_data([tip[0], tip[0]+.05*velocity[0]], [tip[dim], tip[dim]+.05*velocity[dim]])
            centres = [b['centre_m'] for b in d.regions[i]]
            bends.set_data([np.interp(c, nodes, d.relative_positions[i, :, 0]) for c in centres],
                           [np.interp(c, nodes, d.relative_positions[i, :, dim]) for c in centres])
        self.set_cursor(float(d.time[i]))


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
        note=QtWidgets.QLabel('Red segment: 50 ms of tip velocity relative to the attachment. Orange circles / white tracks: bend centres (not detected folds).')
        note.setWordWrap(True);note.setObjectName('note');layout.addWidget(note)
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
        self.canvas.set_index(i);self.clock.setText(f'{d.time[i]:.2f} / {d.time[-1]:.2f} s')
        def number(value):return f'{value:+.2f}' if np.isfinite(value) else 'undefined'
        def angle(cosine):return f'{np.degrees(np.arccos(np.clip(cosine,-1,1))):.1f}°' if np.isfinite(cosine) else 'undefined'
        rows=d.regions[i];b=max(rows,key=lambda b:b['theta']) if rows else None
        bend=(f"Largest bend: {np.degrees(b['theta']):.1f}° total turn\nCentre: {b['centre_m']:.3f} m · coherence {b['coherence']:.2f}" if b else 'No reconstructed bend centre')
        self.telemetry.setText(f"At cursor: {d.phase(i)}\n\n+X reach: {100*d.reach[i]:.1f}% L\nTip height vs attachment: {d.tip_height[i]:+.3f} m\nTip speed world / relative:\n{d.world_speed[i]:.2f} / {d.relative_speed[i]:.2f} m/s\n+X velocity world / relative:\n{number(d.world_outward[i])} / {number(d.relative_outward[i])} m/s\nAngle to +X:\nworld {angle(d.world_cosine[i])} · relative {angle(d.relative_cosine[i])}\n\n{bend}")
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
