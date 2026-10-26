"""Visible Stage 1 PPO orchestration; manual and automatic start share one handler."""
from copy import deepcopy
from pathlib import Path
import sys
from PySide6 import QtCore, QtWidgets

from project_paths import ROOT, project_path
from training.configuration import DEFAULT, experiment, load
from training.artifacts import PPO_RUNS, create_run, read_json, save_json, events
from training.ui.plots import MetricPlot
from training.ui.preview import EvaluationPreview
from training.ui.whip_diagnostics import WhippingDiagnostics


def unchanged(control, value):
    """Keep the exact stored value unless the user edited the (display-rounded) spin box."""
    control.blockSignals(True)
    shown=control.value(); control.setValue(value); initial=control.value(); control.setValue(shown)
    control.blockSignals(False)
    return value if shown==initial else shown


class SettingsDialog(QtWidgets.QDialog):
    def __init__(self, settings, task, parent=None, *, read_only=False):
        super().__init__(parent)
        self.setWindowTitle('Configure training experiment'); self.resize(820,780)
        self.settings,self.task = deepcopy(settings),deepcopy(task)
        self.fields = []; self.controls = {}
        layout = QtWidgets.QVBoxLayout(self)
        note = QtWidgets.QLabel('Read-only saved configuration. Active runs and exact resumes cannot be retuned.' if read_only else
            'Changes apply to the next experiment only. Active runs are frozen. Resume keeps the saved configuration; changing reward or task creates a new experiment.')
        note.setWordWrap(True); layout.addWidget(note)
        tabs = QtWidgets.QTabWidget(); layout.addWidget(tabs)
        used=set()
        def subset(keys):
            used.update(keys)
            return {k:self.settings[k] for k in keys if k in self.settings}
        groups = [
            ('Run & compute', subset(('device','seed','parallel_environments','evaluation_environments','training_episodes','evaluation_interval_episodes','policy_rate_hz','acceleration_limit_mps2'))),
            ('Learner & exploration', subset(('learning_rate','batch_size','hidden_units','initial_log_std','discount','reward_scale','ppo'))),
            ('Task & reward', self.task),
            ('Configuration sources', {k:v for k,v in self.settings.items() if k not in used})]
        # Top-level learner keys are shown through copies; edits write back to settings.
        learning_owners={id(groups[i][1]):self.settings for i in (0,1,3)}
        for name,values in groups:
            scroll = QtWidgets.QScrollArea(); scroll.setWidgetResizable(True)
            contents = QtWidgets.QWidget(); form = QtWidgets.QFormLayout(contents)
            form.setRowWrapPolicy(QtWidgets.QFormLayout.RowWrapPolicy.WrapLongRows)
            scroll.setWidget(contents); tabs.addTab(scroll,name)
            def add(mapping,form,prefix=''):
                for key,value in mapping.items():
                    path=prefix+key
                    owner=learning_owners.get(id(mapping),mapping)
                    label = {
                        'training_episodes': 'Training episodes',
                        'parallel_environments': 'Collection environments',
                        'evaluation_environments': 'Evaluation actor lanes (one physical case)',
                        'evaluation_interval_episodes': 'Evaluation interval (episodes)',
                        'batch_size': 'Optimizer minibatch size',
                        'initial_weight': 'Initial entropy bonus weight',
                        'final_weight': 'Final entropy bonus weight',
                        'learning_rate': 'Learning rate', 'batch_size': 'Minibatch transitions',
                        'policy_rate_hz': 'Action frequency [Hz]',
                        'acceleration_limit_mps2': 'Acceleration limits X / Y / Z [m/s²]',
                        'reward_scale': 'Reward multiplier', 'radius_m': 'Horizontal radius [m]',
                        'height_lower_world_m': 'Minimum drone height [m]', 'height_upper_world_m': 'Maximum drone height [m]',
                        'duration_s': 'Maneuver duration [s]',
                        'target_kl': 'Maximum cohort-mean policy KL',
                    }.get(key, key.replace('_',' ').capitalize())
                    help_text = {
                        'parallel_environments':'Episodes simulated together in each collection. More lanes require more memory; speed is not guaranteed to improve.',
                        'evaluation_environments':'Actor batch layout only. Deterministic evaluation runs one native physical lane from a fixed initial condition, not independent trials.',
                        'training_episodes':'Total episode budget including earlier completed episodes when resuming; not an iteration count.',
                        'evaluation_interval_episodes':'Evaluation happens when this cumulative episode boundary is crossed, plus first and final collections.',
                        'reward_scale':'Multiplies task returns for optimization. It does not change physical units.',
                        'workspace_cost_weight':'Outside-only soft cost weight. Zero inside/on the workspace; no action filter or enforced confinement.',
                        'reward':'Frozen reward version. Changing reward settings creates a new experiment; old checkpoints retain their original objective.',
                        'target_kl':'With the enabled post-update safeguard, trial actor updates are checked against cohort-mean KL; not a pointwise action guarantee.',
                        'initial_weight':'PPO entropy bonus coefficient, not a guaranteed achieved entropy.',
                    }.get(key,'Stored configuration key: '+path)
                    if isinstance(value,dict):
                        box = QtWidgets.QGroupBox(label); inner=QtWidgets.QFormLayout(box)
                        box.setToolTip(help_text)
                        form.addRow(box); add(value,inner,path+'.'); continue
                    if value is None:
                        form.addRow(label,QtWidgets.QLabel('Pending identification / approval'))
                        continue
                    if isinstance(value,list):
                        row = QtWidgets.QWidget(); items=QtWidgets.QHBoxLayout(row); items.setContentsMargins(0,0,0,0)
                        controls=[]
                        for number in value:
                            control=QtWidgets.QDoubleSpinBox(); control.setDecimals(9); control.setRange(-1e6,1e6); control.setValue(number)
                            items.addWidget(control); controls.append(control)
                        row.setToolTip(help_text); self.controls[path]=controls
                        form.addRow(label,row); self.fields.append((owner,key,lambda controls=controls,value=value:
                            [unchanged(c,v) for c,v in zip(controls,value)]))
                        continue
                    elif isinstance(value,bool):
                        control=QtWidgets.QCheckBox(); control.setChecked(value)
                        form.addRow(label,control); self.fields.append((owner,key,control.isChecked))
                    elif type(value) is int:
                        # Signed values must round trip unchanged. The configuration validator sets semantics.
                        control=QtWidgets.QSpinBox(); control.setRange(-2**31+1,2**31-1); control.setValue(value)
                        form.addRow(label,control); self.fields.append((owner,key,control.value))
                    elif isinstance(value,float):
                        control=QtWidgets.QDoubleSpinBox(); control.setDecimals(9); control.setRange(-1e9,1e9); control.setValue(value)
                        form.addRow(label,control); self.fields.append((owner,key,lambda c=control,v=value:unchanged(c,v)))
                    elif key=='device':
                        control=QtWidgets.QComboBox(); control.addItems(['cuda','cpu']); control.setCurrentText(value)
                        form.addRow(label,control); self.fields.append((owner,key,control.currentText))
                    else:
                        control=QtWidgets.QLineEdit(str(value))
                        if key in ('task','model','scene','action_interface','version'): control.setReadOnly(True)
                        form.addRow(label,control); self.fields.append((owner,key,control.text))
                    control.setToolTip(help_text); self.controls[path]=control
            add(values,form)
        if read_only:
            for control in self.controls.values():
                for widget in (control if isinstance(control,list) else [control]):
                    if hasattr(widget,'setReadOnly'): widget.setReadOnly(True)
                    else: widget.setEnabled(False)
        buttons=QtWidgets.QDialogButtonBox(QtWidgets.QDialogButtonBox.StandardButton.Close if read_only else
            QtWidgets.QDialogButtonBox.StandardButton.Ok|QtWidgets.QDialogButtonBox.StandardButton.Cancel)
        buttons.accepted.connect(self.accept); buttons.rejected.connect(self.reject); layout.addWidget(buttons)

    def accept(self):
        try:
            for mapping,key,getter in self.fields: mapping[key]=getter()
            load(experiment(self.settings,self.task))
        except Exception as exc:
            QtWidgets.QMessageBox.warning(self,'Invalid settings',str(exc)); return
        super().accept()


class TrainingWindow(QtWidgets.QMainWindow):
    def __init__(self, settings_path=DEFAULT):
        super().__init__()
        self.setWindowTitle('Training'); self.resize(1320,960)
        self.settings,self.snapshot,_,_ = load(settings_path)
        self.task = self.snapshot['task']
        self.config_path=project_path(settings_path); self.current_run=None; self.process=None; self.resume_path=None
        self.replays=[]; self.close_pending=False; self.last_event_count=-1
        central=QtWidgets.QWidget(objectName='trainingPage'); self.setCentralWidget(central)
        self.setStyleSheet('''
            QWidget#trainingPage { background: #edf2f6; }
            QLabel, QGroupBox, QCheckBox { color: #213547; }
            QGroupBox { border: 1px solid #dce5ec; border-radius: 8px; margin-top: 8px; padding-top: 16px; }
            QPushButton, QToolButton, QComboBox, QSpinBox { background: white; color: #213547; border: 1px solid #ccd8e2; border-radius: 4px; padding: 6px; }
            QPushButton:disabled { color: #96a3ad; }
            QPushButton:hover, QToolButton:hover { background: #e1f2f0; }
            QComboBox QAbstractItemView, QPlainTextEdit, QTableWidget { background: white; color: #213547; }
            QTabBar::tab { background: #dce5ec; color: #213547; padding: 9px; }
            QTabBar::tab:selected { background: white; color: #087d79; }
            QTabWidget::pane { border: 1px solid #dce5ec; background: #edf2f6; }
            QProgressBar { border: 1px solid #ccd8e2; border-radius: 4px; color: #213547; background: white; text-align: center; }
            QProgressBar::chunk { background: #8bd5ca; }
        ''')
        layout=QtWidgets.QVBoxLayout(central); layout.setContentsMargins(20,16,20,16); layout.setSpacing(12)
        top=QtWidgets.QHBoxLayout()
        title=QtWidgets.QLabel('AeroWhip · Training'); title.setStyleSheet('font-size:24px;font-weight:600')
        top.addWidget(title); top.addStretch()
        self.status=QtWidgets.QLabel('Ready'); top.addWidget(self.status)
        self.status.setStyleSheet('font-weight: 600; padding: 6px 12px; border: 1px solid palette(mid); border-radius: 6px;')
        self.start_button=QtWidgets.QPushButton('Start'); self.start_button.clicked.connect(self.start)
        self.stop_button=QtWidgets.QPushButton('Stop'); self.stop_button.clicked.connect(self.stop)
        self.save_button=QtWidgets.QPushButton('Save'); self.save_button.clicked.connect(self.save_checkpoint)
        self.review_button=QtWidgets.QPushButton('Approve motion'); self.review_button.clicked.connect(self.approve_motion)
        self.review_button.setEnabled(False)
        self.save_button.setToolTip('Request a complete checkpoint at the next training-iteration boundary.')
        for b in (self.start_button,self.stop_button,self.save_button,self.review_button): b.setMinimumHeight(34); top.addWidget(b)
        self.stop_button.setEnabled(False); self.save_button.setEnabled(False)
        layout.addLayout(top)
        self.setup_toggle=QtWidgets.QToolButton(); self.setup_toggle.setCheckable(True)
        self.setup_toggle.setToolButtonStyle(QtCore.Qt.ToolButtonStyle.ToolButtonTextBesideIcon)
        self.setup_toggle.setArrowType(QtCore.Qt.ArrowType.RightArrow)
        layout.addWidget(self.setup_toggle)
        self.setup=setup=QtWidgets.QGroupBox(); grid=QtWidgets.QGridLayout(setup)
        self.setup_toggle.toggled.connect(lambda expanded: (setup.setVisible(expanded),self.setup_toggle.setArrowType(QtCore.Qt.ArrowType.DownArrow if expanded else QtCore.Qt.ArrowType.RightArrow)))
        setup.hide()
        self.config_label=QtWidgets.QLabel(); self.config_label.setWordWrap(True); grid.addWidget(self.config_label,0,0,1,3)
        details=QtWidgets.QPushButton('Experiment details…')
        details.clicked.connect(lambda:self.show_text_details('Experiment definition',self.description_details))
        grid.addWidget(details,0,3)
        self.settings_button=QtWidgets.QPushButton('Configure experiment…'); self.settings_button.clicked.connect(self.edit_settings)
        self.open_button=QtWidgets.QPushButton('Load configuration…'); self.open_button.clicked.connect(self.open_settings)
        self.preset_button=QtWidgets.QPushButton('Export configuration…'); self.preset_button.clicked.connect(self.save_settings)
        for i,b in enumerate((self.settings_button,self.open_button,self.preset_button)): grid.addWidget(b,1,i)
        self.rounds=QtWidgets.QSpinBox(); self.rounds.setRange(1,10000000)
        self.sync_budget_control(); grid.addWidget(self.rounds,1,3)
        self.rounds.setToolTip('Total training budget, excluding evaluation. PPO updates use current-cohort numerical prefixes and may stop on KL; task-invalid trajectories remain in the data.')
        self.mode=QtWidgets.QComboBox(); self.mode.addItems(['New training','Resume checkpoint']); self.mode.currentIndexChanged.connect(self.mode_changed)
        self.checkpoint_label=QtWidgets.QLabel('New policy'); self.checkpoint_label.setWordWrap(True)
        self.checkpoint_button=QtWidgets.QPushButton('Choose checkpoint'); self.checkpoint_button.clicked.connect(self.choose_checkpoint); self.checkpoint_button.setEnabled(False)
        grid.addWidget(self.mode,2,0); grid.addWidget(self.checkpoint_label,2,1,1,2); grid.addWidget(self.checkpoint_button,2,3)
        layout.addWidget(setup)
        cards=QtWidgets.QHBoxLayout(); self.cards={}
        for key,label in (('round','Collections'),('episodes','Episodes'),('steps','Environment steps'),('gradient_updates','Gradient updates'),('elapsed','Elapsed')):
            box=QtWidgets.QGroupBox(label); inner=QtWidgets.QVBoxLayout(box)
            value=QtWidgets.QLabel('—'); value.setStyleSheet('font-size:21px;font-weight:600'); inner.addWidget(value); cards.addWidget(box); self.cards[key]=value
        layout.addLayout(cards)
        self.live_detail=QtWidgets.QLabel('No active worker. Start a new experiment or resume a checkpoint.')
        self.live_detail.setWordWrap(True); layout.addWidget(self.live_detail)
        self.budget_progress=QtWidgets.QProgressBar(); self.budget_progress.setRange(0,1000)
        self.budget_progress.setMinimumHeight(22)
        self.budget_progress.setFormat('No run selected'); layout.addWidget(self.budget_progress)
        browser=QtWidgets.QHBoxLayout(); self.runs=QtWidgets.QComboBox(); self.runs.currentIndexChanged.connect(self.select_run)
        self.runs.setMinimumWidth(360)
        refresh=QtWidgets.QPushButton('Refresh runs'); refresh.clicked.connect(lambda:self.refresh_runs())
        browser.addWidget(QtWidgets.QLabel('Dashboard / charts / saved results')); browser.addWidget(self.runs,1); browser.addWidget(refresh)
        replay_browser=QtWidgets.QPushButton('Replay browser…'); replay_browser.clicked.connect(self.open_replay_browser)
        browser.addWidget(replay_browser)
        layout.addLayout(browser)
        self.tabs=QtWidgets.QTabWidget(); layout.addWidget(self.tabs,1)
        overview=QtWidgets.QWidget(); overview_layout=QtWidgets.QVBoxLayout(overview)
        result_box=QtWidgets.QWidget(); result_layout=QtWidgets.QHBoxLayout(result_box)
        result_layout.setContentsMargins(0,0,0,0)
        self.latest_result=QtWidgets.QLabel('No saved evaluation yet.'); self.latest_result.setWordWrap(True)
        self.latest_result.hide()  # Full diagnostics remain available through Details.
        self.latest_summary=QtWidgets.QLabel('No saved evaluation yet.'); self.latest_summary.setWordWrap(True)
        result_layout.addWidget(self.latest_summary,1)
        result_details=QtWidgets.QPushButton('Evaluation details…')
        result_details.clicked.connect(lambda:self.show_text_details('Latest saved evaluation',self.latest_result.text()))
        result_layout.addWidget(result_details)
        overview_layout.addWidget(result_box)
        self.preview=EvaluationPreview(); overview_layout.addWidget(self.preview,1)
        self.tabs.addTab(overview,'Overview · latest motion')
        self.whip_diagnostics=WhippingDiagnostics()
        self.tabs.addTab(self.whip_diagnostics,'Whipping diagnostics')
        curves=QtWidgets.QWidget(); chartgrid=QtWidgets.QGridLayout(curves)
        self.return_plot=MetricPlot('Episode return'); self.mechanics_plot=MetricPlot('Whipping event')
        self.valid_plot=MetricPlot('Valid episodes',limits=(0,1)); self.loss_plot=MetricPlot('Learner losses')
        for i,p in enumerate((self.return_plot,self.mechanics_plot,self.valid_plot,self.loss_plot)): chartgrid.addWidget(p,i//2,i%2)
        self.tabs.addTab(curves,'Learning curves')
        self.reward_plot=MetricPlot('Event reward components')
        self.tabs.addTab(self.reward_plot,'Reward breakdown')
        exploration=QtWidgets.QWidget(); exploration_layout=QtWidgets.QVBoxLayout(exploration)
        self.entropy_plot=MetricPlot('Normalized-action entropy (nats)')
        self.entropy_plot.setToolTip('Measured on an optimizer minibatch, not a success metric. Continuous-action entropy can be negative. A coefficient or scheduled target does not guarantee achieved entropy.')
        self.temperature_plot=MetricPlot('Entropy weight')
        exploration_layout.addWidget(self.entropy_plot); exploration_layout.addWidget(self.temperature_plot)
        self.tabs.addTab(exploration,'Exploration')
        results=QtWidgets.QWidget(); result_layout=QtWidgets.QVBoxLayout(results)
        self.evaluations=QtWidgets.QTableWidget(0,7); self.evaluations.setHorizontalHeaderLabels(['Iteration','Return','Mechanics','Strike','Valid','Outside workspace (s)','Workspace cost'])
        self.evaluations.setSelectionBehavior(QtWidgets.QAbstractItemView.SelectionBehavior.SelectRows)
        self.evaluations.setEditTriggers(QtWidgets.QAbstractItemView.EditTrigger.NoEditTriggers)
        self.evaluations.horizontalHeader().setSectionResizeMode(QtWidgets.QHeaderView.ResizeMode.Stretch)
        result_layout.addWidget(self.evaluations)
        replayrow=QtWidgets.QHBoxLayout()
        for label,slot in (('Replay selected',lambda:self.open_replay('selected')),('Replay latest',lambda:self.open_replay('latest')),('Replay best',lambda:self.open_replay('best'))):
            button=QtWidgets.QPushButton(label); button.clicked.connect(slot); replayrow.addWidget(button)
        result_layout.addLayout(replayrow); self.tabs.addTab(results,'Results')
        self.log=QtWidgets.QPlainTextEdit(); self.log.setReadOnly(True); self.log.setMaximumBlockCount(6000); self.tabs.addTab(self.log,'Log')
        self.timer=QtCore.QTimer(self); self.timer.timeout.connect(self.poll); self.timer.start(750)
        self.update_description(); self.refresh_runs()

    def error(self,exc): QtWidgets.QMessageBox.warning(self,'Training',str(exc))

    def show_text_details(self,title,text):
        dialog=QtWidgets.QDialog(self); dialog.setWindowTitle(title); dialog.resize(850,650)
        layout=QtWidgets.QVBoxLayout(dialog)
        contents=QtWidgets.QPlainTextEdit(); contents.setReadOnly(True); contents.setPlainText(text)
        layout.addWidget(contents)
        close=QtWidgets.QPushButton('Close'); close.clicked.connect(dialog.accept); layout.addWidget(close)
        dialog.exec()

    def open_replay_browser(self):
        from viewer.replay_browser import ReplayBrowser
        window=ReplayBrowser(self.runs.currentData()); self.replays.append(window); window.show()

    def sync_budget_control(self, settings=None):
        settings = self.settings if settings is None else settings
        self.rounds.setPrefix('Training episodes  ')
        self.rounds.setValue(settings['training_episodes'])

    def update_description(self):
        s,t=self.settings,self.task
        w,f=t['workspace'],t['fold']
        self.setup_toggle.setText(f"Training setup · Stage 1 · PPO · {s['device'].upper()} · {s['parallel_environments']} lanes")
        self.setWindowTitle(f"Training — PPO — {self.config_path.stem}")
        e=s['ppo']['entropy']
        self.config_label.setText(
            f"Stage 1 ({t['reward']}): continuous fold transport and outward opening toward world +X. No target-hit reward.\n"
            f"{s['parallel_environments']} training lanes · {s['training_episodes']} episodes · {t['duration_s']:g} s horizon · 10 links · {s['policy_rate_hz']} Hz acceleration actions.\n"
            f"Return Q − C. Q = integral of {t['rate_weight']:g} × fold support × forward reach^{t['reach_power']:g} × alignment^{t['alignment_power']:g} × outward speed / cable length.\n"
            f"C = {t['workspace_cost_weight']:g} × fixed-horizon mean of squared workspace excess in metres. Radius {w['radius_m']:g} m; world height {w['height_lower_world_m']:g}–{w['height_upper_world_m']:g} m.\n"
            f"Fold support: turn {f['turn_lower_deg']:g}–{f['turn_upper_deg']:g}°, coherence {f['coherence_reference']:g}, {1000*f['history_window_s']:g} ms window.\n"
            f"Tilt >{t['maximum_tilt_deg']:g}° or cable height <{t['minimum_cable_height_m']:g} m is recorded as invalid but continues. Numerical corruption aborts and costs {t['failure_cost_per_remaining_s']:g} per remaining second.\n"
            f"Entropy coefficient = {e['final_weight']:g} + {e['initial_weight']-e['final_weight']:g} exp(−{e['decay_exponent']:g} n/N), original episode budget retained on resume. AR(1) correlation {1000*s['ppo']['action_noise']['correlation_time_s']:g} ms.\n"
            f"Acceleration limits {s['acceleration_limit_mps2']} m/s² excluding gravity; zero action does not brake or hold position.\n"
            "Deterministic evaluation is one fixed physical case; the saved actor batch layout is not independent trials.\n"
            "Continuous score is an experimental guide, not a validated binary whipping-success definition.")

    def edit_settings(self):
        if self.process is not None or self.mode.currentIndex()==1:
            SettingsDialog(self.settings,self.task,self,read_only=True).exec()
            return
        self.settings['training_episodes']=self.rounds.value()
        dialog=SettingsDialog(self.settings,self.task,self)
        if dialog.exec():
            self.settings,self.task=dialog.settings,dialog.task
            self.sync_budget_control(); self.update_description()

    def open_settings(self):
        filename,_=QtWidgets.QFileDialog.getOpenFileName(self,'Training settings',str(DEFAULT.parent),'JSON (*.json)')
        if filename:
            try:
                settings,snapshot,_,_=load(filename)
                self.settings,self.snapshot,self.task=settings,snapshot,snapshot['task']; self.config_path=project_path(filename)
                self.sync_budget_control(); self.update_description()
            except Exception as exc: self.error(exc)

    def save_settings(self):
        filename,_=QtWidgets.QFileDialog.getSaveFileName(self,'Save training settings',str(self.config_path),'JSON (*.json)')
        if not filename: return False
        try:
            path=project_path(filename)
            self.settings['training_episodes']=self.rounds.value()
            save_json(path,experiment(self.settings,self.task)); self.config_path=path
            return True
        except Exception as exc: self.error(exc); return False

    def mode_changed(self,index):
        self.checkpoint_button.setEnabled(index==1)
        self.settings_button.setEnabled(True)
        self.settings_button.setText('Inspect configuration…' if index==1 else 'Configure experiment…')
        for b in (self.open_button,self.preset_button): b.setEnabled(index==0)
        if index==0:
            self.resume_path=None
            self.checkpoint_label.setText('New policy'); self.sync_budget_control(); self.update_description()

    def choose_checkpoint(self):
        path,_=QtWidgets.QFileDialog.getOpenFileName(self,'Select learner checkpoint',str(PPO_RUNS),'Checkpoint (*.pt)')
        if path:
            try:
                path=project_path(path)
                import torch
                saved = torch.load(path,map_location='cpu',weights_only=True)
                settings = saved['snapshot']['settings']
                self.settings=deepcopy(settings);self.task=deepcopy(saved['snapshot']['task'])
                self.update_description()
                self.sync_budget_control(settings)
                self.resume_path=Path(path); self.checkpoint_label.setText(self.resume_path.parent.name+' / '+self.resume_path.name)
                self.config_label.setText('Resume uses the checkpoint’s saved task, model and learner settings. Only the total training budget changes.')
            except Exception as exc: self.error(exc)

    def start(self):
        if self.process is not None: return
        try:
            if self.mode.currentIndex()==1 and self.resume_path is None: raise ValueError('Choose a learner checkpoint first')
            settings=deepcopy(self.settings)
            settings['training_episodes']=self.rounds.value()
            # Unsaved UI edits are frozen into this run, never written into defaults.
            directory=create_run(settings,self.resume_path,task=self.task)
            self.current_run=directory; self.last_event_count=-1
            frozen=read_json(directory/'configuration.json')
            self.settings,self.task=deepcopy(frozen['settings']),deepcopy(frozen['task'])
            self.process=QtCore.QProcess(self); self.process.setWorkingDirectory(str(ROOT))
            self.process.setProcessChannelMode(QtCore.QProcess.ProcessChannelMode.MergedChannels)
            self.process.readyReadStandardOutput.connect(self.read_output)
            self.process.finished.connect(self.finished)
            self.process.errorOccurred.connect(self.process_error)
            self.log.clear(); self.status.setText('Starting')
            self.set_running(True)
            self.process.start(sys.executable,['-u','-m','training.worker','--run',str(directory)])
            self.refresh_runs(directory)
        except Exception as exc: self.error(exc)

    def set_running(self,running):
        self.start_button.setEnabled(not running); self.stop_button.setEnabled(running); self.save_button.setEnabled(running)
        for control in (self.mode,self.rounds,self.open_button,self.preset_button,self.checkpoint_button): control.setEnabled(not running)
        self.settings_button.setEnabled(True)
        self.settings_button.setText('Inspect active configuration…' if running else 'Configure experiment…')
        if not running: self.mode_changed(self.mode.currentIndex())

    def stop(self):
        if self.current_run and self.process:
            (self.current_run/'stop.request').touch(); self.status.setText('Stopping safely'); self.stop_button.setEnabled(False)

    def save_checkpoint(self):
        if self.current_run and self.process:
            (self.current_run/'save.request').touch(); self.status.setText('Save requested · next iteration boundary')

    def approve_motion(self):
        self.show_text_details('No automatic curriculum',
            'Current Stage 1 has one fixed reward and does not advance automatically.')

    def read_output(self):
        if self.process:
            text=bytes(self.process.readAllStandardOutput()).decode('utf-8',errors='replace')
            with (self.current_run/'worker.log').open('a',encoding='utf-8') as stream: stream.write(text)
            self.log.appendPlainText(text.rstrip())

    def process_error(self,error):
        if error==QtCore.QProcess.ProcessError.FailedToStart:
            self.log.appendPlainText(self.process.errorString()); self.finished(-1,QtCore.QProcess.ExitStatus.CrashExit)

    def finished(self,code,status):
        if self.process is None: return
        self.read_output(); self.process.deleteLater(); self.process=None; self.set_running(False)
        self.last_event_count=-1; self.poll()
        summary=self.current_run/'summary.json'
        self.status.setText(read_json(summary)['phase'].capitalize() if summary.exists() else f'Worker exited ({code})')
        if self.close_pending: self.close()

    def refresh_runs(self,selected=None):
        current=selected or self.runs.currentData()
        self.runs.blockSignals(True); self.runs.clear()
        paths=[p for root in (PPO_RUNS,) if root.exists()
               for p in root.iterdir() if p.is_dir() and (p/'configuration.json').exists()]
        for path in sorted(paths,key=lambda p:p.name,reverse=True):
            self.runs.addItem('PPO Stage 1 · '+path.name,path)
        # Qt's QVariant lookup does not compare separately created Path objects
        # by value. Keep Path data for replay callers, but select in Python.
        index=next((i for i in range(self.runs.count()) if self.runs.itemData(i)==current),-1)
        if index>=0: self.runs.setCurrentIndex(index)
        self.runs.blockSignals(False); self.select_run()

    def select_run(self,*_):
        self.last_event_count=-1
        self._event_signature=None
        self.poll()

    def poll(self):
        self.review_button.setEnabled(False)
        if self.current_run and self.process:
            try:
                progress=read_json(self.current_run/'progress.json')
                phase=progress.get('phase')
                pending=phase=='curriculum_review'
                self.review_button.setEnabled(pending)
                if phase:
                    label='Iteration completed' if phase=='round' else phase.replace('_',' ').capitalize()
                    if (self.current_run/'stop.request').exists() and phase not in ('stopped','failed','completed','plateau_stopped'):
                        label='Stopping safely'
                    self.status.setText(label)
                self.live_detail.setText(f"Active: {self.current_run.name} · {label if phase else 'Starting'} · iteration {progress.get('round','—')} / {progress.get('total_rounds','—')}"
                    + (f" · simulation {progress['simulation_s']:.1f} s · {progress.get('active','—')} active lanes" if 'simulation_s' in progress else ''))
            except (OSError,ValueError): pass
        directory=self.runs.currentData()
        if directory is None: return
        try:
            stat=(directory/'events.jsonl').stat()
            signature=(directory,stat.st_mtime_ns,stat.st_size,stat.st_ino)
        except FileNotFoundError:
            signature=(directory,None)  # A new run must still clear old charts.
        except OSError: return
        if signature==getattr(self,'_event_signature',None) and self.last_event_count>=0: return
        try:
            history=events(directory)
        except OSError: return
        self._event_signature=signature
        if len(history)==self.last_event_count: return
        self.last_event_count=len(history)
        # Always use the selected run's immutable units, not today's UI preset.
        selected_snapshot=read_json(directory/'configuration.json')
        selected_settings=selected_snapshot['settings']
        self.update_overview(directory,history,selected_settings)
        self.whip_diagnostics.set_run(directory,history)
        scale=selected_settings.get('reward_scale',1.0)
        self.loss_plot.title='PPO losses (last minibatch)'
        self.return_plot.title=f'Episode return (reward ×{scale:g})'
        self.reward_plot.title=f'Event reward components (×{scale:g})'
        self.return_plot.setToolTip('Training reward units. Original task returns are retained in the log; compare runs using the same scale.')
        self.evaluations.setHorizontalHeaderLabels(['Iteration',f'Return ×{scale:g}','Mechanics','Strike','Valid','Outside workspace (s)',f'Workspace cost ×{scale:g}'])
        rounds=[e for e in history if e['phase']=='round']; evaluations=[e for e in history if e['phase']=='evaluation']
        rows=evaluations
        self.evaluation_rows=rows
        if history:
            latest=history[-1]
            for key in ('round','episodes','steps','gradient_updates'):
                item=next((e[key] for e in reversed(history) if key in e),None)
                self.cards[key].setText(str(item) if item is not None else '—')
            seconds=int(latest.get('elapsed_s',0)); self.cards['elapsed'].setText(f'{seconds//3600:02}:{seconds//60%60:02}:{seconds%60:02}')
        self.return_plot.set_series([
            ('Training',[(e['round'],scale*e['return_mean']) for e in rounds],False),
            ('Evaluation',[(e['round'],scale*e['evaluation_return']) for e in evaluations],False),
            ('Best',[(e['round'],scale*e['best_return']) for e in evaluations],True)])
        self.mechanics_plot.title='Continuous output diagnostics · not success rate'
        self.mechanics_plot.set_series([(label,[(e['round'],e['components'].get(key)) for e in rounds],False)
            for key,label in (('peak_reward_rate','Peak rate'),('fold_support_integral_s','Fold support (s)'),
                              ('positive_output_duration_s','Output duration (s)'))])
        self.reward_plot.title='Continuous reward and costs'
        self.reward_plot.set_series([(label,[(e['round'],e['components'].get(key)) for e in rounds],False)
            for key,label in (('continuous_whip_credit','Accrued output'),('workspace_cost','Workspace cost'),
                              ('failure_cost','Numerical abort cost'),('net_episode_return','Net return'))])
        self.valid_plot.title='Completion and physical-envelope diagnostics'
        self.valid_plot.set_series([(label,[(e['round'],e['components'].get(key)) for e in rounds],False)
            for key,label in (('completed_horizon','Full horizon'),('physical_violation','Physical violation'),
                              ('numerically_valid','Numerically valid'))])
        self.evaluations.setHorizontalHeaderLabels(['Collection',f'Return ×{scale:g}','Accrued output',
            'Full horizon','Valid','Outside workspace (s)','Workspace cost'])
        self.loss_plot.set_series([(label,[(e['round'],e.get(key)) for e in rounds],False) for key,label in (('critic_loss','Critic'),('actor_loss','Actor'))])
        exploration=[e for e in history if e['phase']=='exploration']
        self.entropy_plot.set_series([('Measured',[(e['round'],e.get('entropy')) for e in exploration],False)])
        self.temperature_plot.title='PPO entropy bonus coefficient'
        self.temperature_plot.set_series([('Coefficient',[(e['round'],e.get('entropy_weight')) for e in exploration],False)])
        self.evaluations.setRowCount(len(rows))
        for i,e in enumerate(rows):
            outside=e.get('workspace',{}).get('outside_duration_s')
            iteration=f"{e['round']} · Lane {e['lane']}" if 'lane' in e else e['round']
            charge=e['components'].get('workspace_cost')
            values=(iteration,scale*e['evaluation_return'],e['components'].get('continuous_whip_credit'),
                    e['components'].get('completed_horizon'),e['valid'],outside,charge)
            for j,value in enumerate(values):
                text='—' if value is None else (f'{value:.4g}' if isinstance(value,float) else str(value))
                self.evaluations.setItem(i,j,QtWidgets.QTableWidgetItem(text))

    def update_overview(self,directory,history,settings):
        episodes=next((e['episodes'] for e in reversed(history) if 'episodes' in e),0)
        budget=settings['training_episodes']
        self.budget_progress.setValue(min(1000,round(1000*episodes/budget)))
        self.budget_progress.setFormat(f'Stage 1 · {episodes:,} / {budget:,} training episodes · {100*episodes/budget:.1f}%')
        if self.process is None:
            phase=history[-1]['phase'] if history else 'no events'
            self.live_detail.setText(f'Viewing saved run: {directory.name} · {phase.replace("_"," ")} · no worker owned by this window')
        latest=next((e for e in reversed(history) if e['phase']=='evaluation'),None)
        # Do not show an old valid replay as if it were the newest failed result.
        self.preview.set_result(directory,latest if latest and latest.get('replay') else None)
        if latest is None:
            self.latest_result.setText('No completed evaluation yet.\n\nCollection and optimization progress is shown above. Evaluation is separate from stochastic training episodes.')
            self.latest_summary.setText('No saved evaluation yet · training progress is shown above')
            return
        def number(value): return '—' if value is None else f'{value:.5g}'
        c=latest.get('components',{}); ws=latest.get('workspace',{})
        raw=latest.get('evaluation_return'); scale=settings.get('reward_scale',1.)
        if 'continuous_whip_credit' in c:
            outcome=('Full horizon completed' if latest.get('valid') else 'INVALID physical envelope · full horizon completed') if c.get('completed_horizon')==1 else 'INVALID numerical abort / saved prefix'
            self.latest_result.setText(
                f"Stage1 · collection {latest['round']} · {outcome}\nReturn: {number(raw)}\n"
                f"Accrued output: {number(c['continuous_whip_credit'])}\nWorkspace cost: {number(c.get('workspace_cost'))}\nFailure correction: {number(c.get('failure_adjustment'))}\n"
                f"Peak reward rate: {number(c.get('peak_reward_rate'))}\nFold support: {number(c.get('fold_support_integral_s'))} s\n"
                f"Positive output duration: {number(c.get('positive_output_duration_s'))} s\n"
                "No target or binary whip success gate. See Whipping diagnostics for recorded rate and fold support.")
            self.latest_summary.setText(f"Stage1 · collection {latest['round']} · {outcome} · return {number(raw)} · continuous score is not success")
            return
        self.latest_result.setText('No current continuous-reward measurements in this evaluation. Historical task support is retired.')
        self.latest_summary.setText('Unsupported historical evaluation')

    def open_replay(self,selection):
        try:
            rows=getattr(self,'evaluation_rows',[])
            if not rows: raise ValueError('No completed evaluation is saved yet')
            deterministic=[e for e in rows if e['phase']=='evaluation']
            if selection=='best': event=max(deterministic,key=lambda e:e['evaluation_return'])
            elif selection=='latest': event=deterministic[-1]
            else:
                index=self.evaluations.currentRow()
                if index<0: raise ValueError('Select an evaluation row')
                event=rows[index]
            if not event.get('replay'): raise ValueError('This evaluation has no valid numerical replay prefix')
            from training.ui.replay_window import ReplayWindow
            window=ReplayWindow(self.runs.currentData(),event['replay'],event,self)
            self.replays.append(window); window.show()
        except Exception as exc: self.error(exc)

    def closeEvent(self,event):
        if self.process:
            answer=QtWidgets.QMessageBox.question(self,'Training is running','Stop safely and close after saving?',
                QtWidgets.QMessageBox.StandardButton.Yes|QtWidgets.QMessageBox.StandardButton.No)
            if answer==QtWidgets.QMessageBox.StandardButton.Yes: self.close_pending=True; self.stop()
            event.ignore(); return
        self.timer.stop(); self.whip_diagnostics.pause(); self.preview.dispose(); super().closeEvent(event)


def configure_resume(window, checkpoint, saved, requested_settings=None):
    """Restore saved configuration, accepting only an explicit budget override.

    Execution/identity checks still occur in create_run before dispatch. No
    checkpoint, optimizer, RNG or original entropy budget is changed here.
    """
    settings=deepcopy(saved['snapshot']['settings'])
    if requested_settings is not None:
        requested=deepcopy(requested_settings)
        budget=requested.pop('training_episodes')
        original=deepcopy(settings); original.pop('training_episodes')
        if requested != original:
            raise ValueError('Exact resume settings may change ONLY training_episodes; reward/source changes require a fresh experiment')
        settings['training_episodes']=budget
    window.settings=settings;window.task=deepcopy(saved['snapshot']['task'])
    window.mode.setCurrentIndex(1)
    window.resume_path=checkpoint
    window.sync_budget_control(settings)
    window.update_description()
    window.checkpoint_label.setText(str(checkpoint))


def main(*, auto_start=False, settings_path=None, resume_checkpoint=None):
    app=QtWidgets.QApplication.instance() or QtWidgets.QApplication(sys.argv)
    requested_settings=load(settings_path)[0] if settings_path is not None else None
    window=TrainingWindow(settings_path or DEFAULT); window.show()
    if resume_checkpoint is not None:
        import torch
        checkpoint=project_path(resume_checkpoint)
        saved=torch.load(checkpoint,map_location='cpu',weights_only=True)
        configure_resume(window,checkpoint,saved,requested_settings)
    if auto_start:
        # Enter the GUI event loop first. Never launch a detached headless worker.
        QtCore.QTimer.singleShot(0, lambda: window.start() if window.isVisible() else None)
    return app.exec()
