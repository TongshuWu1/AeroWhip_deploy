"""Training window: configure, start, resume and monitor PPO runs; manual and automatic start share one handler."""
from copy import deepcopy
from pathlib import Path
import sys
from PySide6 import QtCore, QtWidgets

from aerowhip.paths import ROOT, project_path
from aerowhip.experiment import DEFAULT, load
from aerowhip.learning.artifacts import RUNS, create_run, read_json, save_json, events
from aerowhip.ui import style
from aerowhip.ui.training.plots import LearningCurves
from aerowhip.ui.training.preview import EvaluationPreview


def unchanged(control, value):
    """Keep the exact stored value unless the user edited the (display-rounded) spin box."""
    control.blockSignals(True)
    shown=control.value(); control.setValue(value); initial=control.value(); control.setValue(shown)
    control.blockSignals(False)
    return value if shown==initial else shown


class SettingsDialog(QtWidgets.QDialog):
    def __init__(self, experiment, parent=None, *, read_only=False):
        super().__init__(parent)
        self.setWindowTitle('Configure training experiment'); self.resize(820,780)
        self.experiment = deepcopy(experiment)
        self.fields = []; self.controls = {}
        layout = QtWidgets.QVBoxLayout(self)
        note = QtWidgets.QLabel('Read-only saved configuration. Active runs and exact resumes cannot be retuned.' if read_only else
            'Changes apply to the next experiment only. Active runs are frozen. Resume keeps the saved configuration; changing reward or task creates a new experiment.')
        note.setWordWrap(True); layout.addWidget(note)
        tabs = QtWidgets.QTabWidget(); layout.addWidget(tabs)
        e=self.experiment
        sources={k:e[k] for k in ('model','scene')}
        groups = [('Run & compute', e['run']), ('Environment', e['environment']), ('Task & reward', e['task']),
                  ('Learner & exploration', e['ppo']), ('Configuration sources', sources)]
        # Sources are shown through a copy; their edits write back to the experiment.
        owners={id(sources):e}
        for name,values in groups:
            scroll = QtWidgets.QScrollArea(); scroll.setWidgetResizable(True)
            contents = QtWidgets.QWidget(); form = QtWidgets.QFormLayout(contents)
            form.setRowWrapPolicy(QtWidgets.QFormLayout.RowWrapPolicy.WrapLongRows)
            scroll.setWidget(contents); tabs.addTab(scroll,name)
            def add(mapping,form,prefix=''):
                for key,value in mapping.items():
                    path=prefix+key
                    owner=owners.get(id(mapping),mapping)
                    label = {
                        'training_episodes': 'Training episodes',
                        'parallel_environments': 'Collection environments',
                        'evaluation_actor_batch': 'Evaluation actor batch (one physical case)',
                        'evaluation_interval_episodes': 'Evaluation interval (episodes)',
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
                        'evaluation_actor_batch':'Actor batch layout only. Deterministic evaluation runs one native physical lane from a fixed initial condition, not independent trials.',
                        'training_episodes':'Total episode budget including earlier completed episodes when resuming; not an iteration count.',
                        'evaluation_interval_episodes':'Evaluation happens when this cumulative episode boundary is crossed, plus first and final collections.',
                        'reward_scale':'Multiplies task returns for optimization. It does not change physical units.',
                        'workspace_cost_weight':'Outside-only soft cost weight. Zero inside/on the workspace; no action filter or enforced confinement.',
                        'name':'Registered task (reward definition). Changing task settings creates a new experiment; old checkpoints retain their original objective.',
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
                        if key in ('model','scene','name'): control.setReadOnly(True)
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
            load(self.experiment)
        except Exception as exc:
            QtWidgets.QMessageBox.warning(self,'Invalid settings',str(exc)); return
        super().accept()


class TrainingWindow(QtWidgets.QMainWindow):
    def __init__(self, settings_path=DEFAULT):
        super().__init__()
        self.setWindowTitle('Training'); self.resize(1320,960)
        _,snapshot,_ = load(settings_path)
        self.experiment = snapshot['experiment']
        self.config_path=project_path(settings_path); self.current_run=None; self.process=None; self.resume_path=None
        self.replays=[]; self.close_pending=False; self.last_event_count=-1
        central=QtWidgets.QWidget(objectName='trainingPage'); self.setCentralWidget(central)
        style.apply(self)
        layout=QtWidgets.QVBoxLayout(central); layout.setContentsMargins(20,16,20,16); layout.setSpacing(12)
        top=QtWidgets.QHBoxLayout()
        title=QtWidgets.QLabel('PPO training'); title.setObjectName('heading')
        top.addWidget(title); top.addStretch()
        self.status=QtWidgets.QLabel('Ready'); top.addWidget(self.status)
        self.status.setObjectName('section')
        self.start_button=QtWidgets.QPushButton('Start'); self.start_button.clicked.connect(self.start)
        self.stop_button=QtWidgets.QPushButton('Stop'); self.stop_button.clicked.connect(self.stop)
        self.save_button=QtWidgets.QPushButton('Save'); self.save_button.clicked.connect(self.save_checkpoint)
        self.save_button.setToolTip('Request a complete checkpoint at the next training-iteration boundary.')
        self.start_button.setObjectName('primary')
        for b in (self.start_button,self.stop_button,self.save_button): top.addWidget(b)
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
            box=QtWidgets.QGroupBox(label); inner=QtWidgets.QVBoxLayout(box); inner.setContentsMargins(6,2,6,2)
            value=QtWidgets.QLabel('—'); value.setObjectName('value'); inner.addWidget(value); cards.addWidget(box); self.cards[key]=value
        layout.addLayout(cards)
        self.live_detail=QtWidgets.QLabel('No active worker. Start a new experiment or resume a checkpoint.')
        self.live_detail.setWordWrap(True); self.live_detail.setObjectName('muted'); layout.addWidget(self.live_detail)
        self.budget_progress=QtWidgets.QProgressBar(); self.budget_progress.setRange(0,1000)
        self.budget_progress.setFormat('No run selected'); layout.addWidget(self.budget_progress)
        browser=QtWidgets.QHBoxLayout(); self.runs=QtWidgets.QComboBox(); self.runs.currentIndexChanged.connect(self.select_run)
        self.runs.setMinimumWidth(360)
        refresh=QtWidgets.QPushButton('Refresh runs'); refresh.clicked.connect(lambda:self.refresh_runs())
        browser.addWidget(QtWidgets.QLabel('Run')); browser.addWidget(self.runs,1); browser.addWidget(refresh)
        replay_browser=QtWidgets.QPushButton('Replay browser…'); replay_browser.clicked.connect(self.open_replay_browser)
        browser.addWidget(replay_browser)
        layout.addLayout(browser)
        self.tabs=QtWidgets.QTabWidget(); layout.addWidget(self.tabs,1)
        self.curves=LearningCurves()
        self.curves.setToolTip('Training: median and 10–90 % of all episodes of each collection. '
                               'Evaluation: one deterministic fixed initial case per marker.')
        self.tabs.addTab(self.curves,'Learning curves')
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
        self.tabs.addTab(overview,'Latest evaluation')
        results=QtWidgets.QWidget(); result_layout=QtWidgets.QVBoxLayout(results)
        self.evaluations=QtWidgets.QTableWidget(0,7); self.evaluations.setHorizontalHeaderLabels(['Collection','Return','Whip events','Strikes','Valid','Outside workspace (s)','Workspace cost'])
        self.evaluations.setSelectionBehavior(QtWidgets.QAbstractItemView.SelectionBehavior.SelectRows)
        self.evaluations.setEditTriggers(QtWidgets.QAbstractItemView.EditTrigger.NoEditTriggers)
        self.evaluations.horizontalHeader().setSectionResizeMode(QtWidgets.QHeaderView.ResizeMode.Stretch)
        result_layout.addWidget(self.evaluations)
        replayrow=QtWidgets.QHBoxLayout()
        for label,slot in (('Replay selected',lambda:self.open_replay('selected')),('Replay latest',lambda:self.open_replay('latest')),('Replay best',lambda:self.open_replay('best'))):
            button=QtWidgets.QPushButton(label); button.clicked.connect(slot); replayrow.addWidget(button)
        result_layout.addLayout(replayrow); self.tabs.addTab(results,'Evaluations')
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
        from aerowhip.viewer.replay_browser import ReplayBrowser
        window=ReplayBrowser(self.runs.currentData()); self.replays.append(window); window.show()

    def sync_budget_control(self, experiment=None):
        experiment = self.experiment if experiment is None else experiment
        self.rounds.setPrefix('Training episodes  ')
        self.rounds.setValue(experiment['run']['training_episodes'])

    def update_description(self):
        r,v,t,ppo=(self.experiment[k] for k in ('run','environment','task','ppo'))
        w,e=v['workspace'],ppo['entropy']
        self.setup_toggle.setText(f"Training setup · PPO · {r['device'].upper()} · {r['parallel_environments']} lanes")
        self.setWindowTitle(f"Training — PPO — {self.config_path.stem}")
        stages=t['curriculum']['stages']; final=stages[-1]; rnd=v.get('randomisation',{})
        if True:
            self.config_label.setText(
                f"Task {t['name']}: strike a target with the cable tip, then recover. Target {final['target_distance_m']:g} m along +X, "
                f"{t['target_below_origin_m']:g} m below the origin; hit within {t['hit_radius_m']:g} m at ≥ {final['speed_threshold_mps']:g} m/s.\n"
                f"{r['parallel_environments']} training lanes · {r['training_episodes']} episodes · {v['duration_s']:g} s horizon, a hit ends the episode "
                f"{t['recovery_s']:g} s later · {v['policy_rate_hz']} Hz acceleration actions, limits {v['acceleration_limit_mps2']} m/s².\n"
                f"Reward: strike {t['strike_bonus']:g} + {t['speed_bonus_per_mps']:g}/(m/s) above the threshold; progress {t['progress_weight']:g} per cable length closed "
                f"(potential-based); approach ≤ {t['approach_weight']:g}/s; recovery {t['recovery_bonus']:g}. Costs: workspace {t['workspace_cost_weight']:g} × mean squared excess, "
                f"action {t['action_cost_weight']:g}, termination {t['failure_cost_per_remaining_s']:g}/s remaining.\n"
                f"Curriculum {len(stages)} stages (" + ', '.join(f"{s['target_distance_m']:g} m / {s['speed_threshold_mps']:g} m/s / r {s['workspace_radius_m']:g} m" for s in stages) +
                f"), advance at {t['curriculum']['advance_strike_rate']:.0%} strikes over {t['curriculum']['window_episodes']} episodes; "
                f"{t['curriculum']['reference_state_fraction']:.0%} of lanes start from recorded whip states.\n"
                f"Randomisation: {rnd.get('groups',0)} model groups per cohort, voltage {rnd.get('battery_voltage_v')} V, drag ×{rnd.get('body_drag_scale')}, "
                f"Cb ×{rnd.get('bending_damping_scale')}, response {rnd.get('motor_response_s')} s, delay {rnd.get('command_delay_s')} s, bias sd {rnd.get('acceleration_bias_sd_mps2')} m/s².\n"
                f"Tilt >{v['maximum_tilt_deg']:g}° or cable below {v['minimum_cable_height_m']:g} m {'ends the episode' if v.get('terminate_on_violation') else 'is recorded as invalid'}. "
                f"Heights {w['height_lower_world_m']:g}–{w['height_upper_world_m']:g} m.\n"
                f"Evaluation: {r.get('evaluation_episodes',1)} independent deterministic episodes at the final stage every {r['evaluation_interval_episodes']} training episodes. "
                f"Entropy {e['final_weight']:g} + {e['initial_weight']-e['final_weight']:g} exp(−{e['decay_exponent']:g} n/N); AR(1) action noise {1000*ppo['action_noise_correlation_s']:g} ms.\n"
                "Whip event (reported, not rewarded): peak tip speed ≥ threshold, ≥ 2× the attachment's speed, tip at ≥ 90 % of the cable length.")

    def edit_settings(self):
        if self.process is not None or self.mode.currentIndex()==1:
            SettingsDialog(self.experiment,self,read_only=True).exec()
            return
        self.experiment['run']['training_episodes']=self.rounds.value()
        dialog=SettingsDialog(self.experiment,self)
        if dialog.exec():
            self.experiment=dialog.experiment
            self.sync_budget_control(); self.update_description()

    def open_settings(self):
        filename,_=QtWidgets.QFileDialog.getOpenFileName(self,'Training settings',str(DEFAULT.parent),'JSON (*.json)')
        if filename:
            try:
                _,snapshot,_=load(filename)
                self.experiment=snapshot['experiment']; self.config_path=project_path(filename)
                self.sync_budget_control(); self.update_description()
            except Exception as exc: self.error(exc)

    def save_settings(self):
        filename,_=QtWidgets.QFileDialog.getSaveFileName(self,'Save training settings',str(self.config_path),'JSON (*.json)')
        if not filename: return False
        try:
            path=project_path(filename)
            self.experiment['run']['training_episodes']=self.rounds.value()
            save_json(path,self.experiment); self.config_path=path
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
        path,_=QtWidgets.QFileDialog.getOpenFileName(self,'Select learner checkpoint',str(RUNS),'Checkpoint (*.pt)')
        if path:
            try:
                path=project_path(path)
                import torch
                saved = torch.load(path,map_location='cpu',weights_only=True)
                self.experiment=deepcopy(saved['snapshot']['experiment'])
                self.update_description()
                self.sync_budget_control()
                self.resume_path=Path(path); self.checkpoint_label.setText(self.resume_path.parent.name+' / '+self.resume_path.name)
                self.config_label.setText('Resume uses the checkpoint’s saved task, model and learner settings. Only the total training budget changes.')
            except Exception as exc: self.error(exc)

    def start(self):
        if self.process is not None: return
        try:
            if self.mode.currentIndex()==1 and self.resume_path is None: raise ValueError('Choose a learner checkpoint first')
            experiment=deepcopy(self.experiment)
            experiment['run']['training_episodes']=self.rounds.value()
            # Unsaved UI edits are frozen into this run, never written into defaults.
            directory=create_run(experiment,self.resume_path)
            self.current_run=directory; self.last_event_count=-1
            frozen=read_json(directory/'configuration.json')
            self.experiment=deepcopy(frozen['experiment'])
            self.process=QtCore.QProcess(self); self.process.setWorkingDirectory(str(ROOT))
            self.process.setProcessChannelMode(QtCore.QProcess.ProcessChannelMode.MergedChannels)
            self.process.readyReadStandardOutput.connect(self.read_output)
            self.process.finished.connect(self.finished)
            self.process.errorOccurred.connect(self.process_error)
            self.log.clear(); self.status.setText('Starting')
            self.set_running(True)
            self.process.start(sys.executable,['-u','-m','aerowhip.learning.worker','--run',str(directory)])
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
        paths=[p for root in (RUNS,) if root.exists()
               for p in root.iterdir() if p.is_dir() and (p/'configuration.json').exists()]
        for path in sorted(paths,key=lambda p:p.name,reverse=True):
            self.runs.addItem('PPO · '+path.name,path)
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
        if self.current_run and self.process:
            try:
                progress=read_json(self.current_run/'progress.json')
                phase=progress.get('phase')
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
        selected=selected_snapshot['experiment']
        self.update_overview(directory,history,selected)
        scale=selected['ppo']['reward_scale']
        rounds=[e for e in history if e['phase']=='round']; evaluations=[e for e in history if e['phase']=='evaluation']
        rows=evaluations
        self.evaluation_rows=rows
        if history:
            latest=history[-1]
            for key in ('round','episodes','steps','gradient_updates'):
                item=next((e[key] for e in reversed(history) if key in e),None)
                self.cards[key].setText(f'{item:,}' if isinstance(item,int) else str(item) if item is not None else '—')
            seconds=int(latest.get('elapsed_s',0)); self.cards['elapsed'].setText(f'{seconds//3600:02}:{seconds//60%60:02}:{seconds%60:02}')
        self.curves.set_history(directory,history,selected)
        self.evaluations.setHorizontalHeaderLabels(['Collection',f'Return ×{scale:g}','Whip events',
            'Strikes','Valid','Outside workspace (s)','Workspace cost'])
        self.evaluations.setRowCount(len(rows))
        for i,e in enumerate(rows):
            outside=e.get('workspace',{}).get('outside_duration_s')
            iteration=f"{e['round']} · Lane {e['lane']}" if 'lane' in e else e['round']
            charge=e['components'].get('workspace_cost')
            c=e['components']
            values=(iteration,scale*e['evaluation_return'],c.get('whip_event'),c.get('strike'),
                    e.get('valid_fraction',e['valid']),outside,charge)
            for j,value in enumerate(values):
                text='—' if value is None else (f'{value:.4g}' if isinstance(value,float) else str(value))
                self.evaluations.setItem(i,j,QtWidgets.QTableWidgetItem(text))

    def update_overview(self,directory,history,experiment):
        episodes=next((e['episodes'] for e in reversed(history) if 'episodes' in e),0)
        budget=experiment['run']['training_episodes']
        self.budget_progress.setValue(min(1000,round(1000*episodes/budget)))
        self.budget_progress.setFormat(f'{episodes:,} / {budget:,} training episodes · {100*episodes/budget:.1f}%')
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
        c=latest.get('components',{})
        raw=latest.get('evaluation_return')
        if 'strike' in c:
            n=latest.get('evaluation_episodes',1)
            self.latest_result.setText(
                f"Collection {latest['round']} · {n} evaluation episodes · mean return {number(raw)}\n"
                f"Strike rate: {number(c.get('strike'))}\nWhip-event rate: {number(c.get('whip_event'))}\nRecovered: {number(c.get('recovered'))}\n"
                f"Peak tip speed: {number(c.get('peak_tip_speed_mps'))} m/s · amplification {number(c.get('amplification_at_peak'))} · extension {number(c.get('extension_at_peak'))}\n"
                f"Closest approach: {number(c.get('closest_approach_m'))} m · sharpest joint {number(c.get('max_joint_turn_deg'))}°\n"
                f"Workspace cost: {number(c.get('workspace_cost'))} · termination cost: {number(c.get('failure_cost'))} · valid {number(latest.get('valid_fraction'))}")
            self.latest_summary.setText(f"Collection {latest['round']} · strike rate {number(c.get('strike'))} · whip events {number(c.get('whip_event'))} · mean return {number(raw)}")
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
            from aerowhip.ui.training.replay_window import ReplayWindow
            window=ReplayWindow(self.runs.currentData(),event['replay'],event,self)
            self.replays.append(window); window.show()
        except Exception as exc: self.error(exc)

    def closeEvent(self,event):
        if self.process:
            answer=QtWidgets.QMessageBox.question(self,'Training is running','Stop safely and close after saving?',
                QtWidgets.QMessageBox.StandardButton.Yes|QtWidgets.QMessageBox.StandardButton.No)
            if answer==QtWidgets.QMessageBox.StandardButton.Yes: self.close_pending=True; self.stop()
            event.ignore(); return
        self.timer.stop(); self.preview.dispose(); super().closeEvent(event)


def configure_resume(window, checkpoint, saved, requested=None):
    """Restore the saved experiment, accepting only an explicit episode-budget override.

    Execution/identity checks still occur in create_run before dispatch. No
    checkpoint, optimizer, RNG or original entropy budget is changed here.
    """
    experiment=deepcopy(saved['snapshot']['experiment'])
    if requested is not None:
        requested=deepcopy(requested)
        budget=requested['run'].pop('training_episodes')
        original=deepcopy(experiment); original['run'].pop('training_episodes')
        if requested != original:
            raise ValueError('Exact resume may change ONLY run.training_episodes; other changes require a fresh experiment')
        experiment['run']['training_episodes']=budget
    window.experiment=experiment
    window.mode.setCurrentIndex(1)
    window.resume_path=checkpoint
    window.sync_budget_control()
    window.update_description()
    window.checkpoint_label.setText(str(checkpoint))


def main(*, auto_start=False, settings_path=None, resume_checkpoint=None):
    app=QtWidgets.QApplication.instance() or QtWidgets.QApplication(sys.argv)
    requested=load(settings_path)[1]['experiment'] if settings_path is not None else None
    window=TrainingWindow(settings_path or DEFAULT); window.show()
    if resume_checkpoint is not None:
        import torch
        checkpoint=project_path(resume_checkpoint)
        saved=torch.load(checkpoint,map_location='cpu',weights_only=True)
        configure_resume(window,checkpoint,saved,requested)
    if auto_start:
        # Enter the GUI event loop first. Never launch a detached headless worker.
        QtCore.QTimer.singleShot(0, lambda: window.start() if window.isVisible() else None)
    return app.exec()
