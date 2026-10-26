"""Visible PPO/SAC orchestration; manual and automatic start share one handler."""
from copy import deepcopy
from pathlib import Path
import sys
from PySide6 import QtCore, QtWidgets

from project_paths import ROOT, project_path
from training.configuration import DEFAULT, CURRENT_PRESETS, load, task_configuration, evaluation_settings, algorithm, training_stage
from training.artifacts import RUNS, PPO_RUNS, PPO_STAGE2_RUNS, PPO_STAGE3_RUNS, PPO_STAGE4_RUNS, create_run, read_json, save_json, events
from training.ui.plots import MetricPlot
from training.ui.preview import EvaluationPreview
from training.ui.whip_diagnostics import WhippingDiagnostics


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
            ('Run & compute',subset(('device','seed','parallel_environments','evaluation_environments','training_episodes','evaluation_interval_episodes','policy_rate_hz','velocity_limit_mps','acceleration_limit_mps2','plateau_detection'))),
            ('Learner & exploration',subset(('learning_rate','batch_size','hidden_units','initial_log_std','discount','ppo','exploration','initial_temperature','target_entropy','target_update_tau','replay_episodes','gradient_updates_per_round','minimum_replay_transitions'))),
            ('Reward & boundary',dict(learning=subset(('reward_scale','workspace_penalty')),return_costs=self.task['return'],matching=self.task['matching'])),
            ('Target & episode',dict(episode=self.task['episode'],workspace=self.task['workspace'],strike_surface=self.task['strike_surface'],request=self.task['request'],**subset(('target_curriculum',)))),
            ('Advanced / diagnostics',dict(sources={k:v for k,v in self.settings.items() if k not in used},mechanics=self.task['mechanics']))]
        # Subset dictionaries share nested values; scalar edits are written back
        # through their owner, not left in the temporary presentation grouping.
        learning_owners={id(values):(self.settings) for _,values in groups[:2]}
        learning_owners[id(groups[2][1]['learning'])]=self.settings
        learning_owners[id(groups[4][1]['sources'])]=self.settings
        learning_owners[id(groups[3][1])]=self.task
        for name,values in groups:
            scroll = QtWidgets.QScrollArea(); scroll.setWidgetResizable(True)
            contents = QtWidgets.QWidget(); form = QtWidgets.QFormLayout(contents)
            form.setRowWrapPolicy(QtWidgets.QFormLayout.RowWrapPolicy.WrapLongRows)
            scroll.setWidget(contents); tabs.addTab(scroll,name)
            def add(mapping,form,prefix=''):
                for key,value in mapping.items():
                    path=prefix+key
                    owner=learning_owners.get(id(mapping),mapping)
                    if key=='target_curriculum': owner=self.settings
                    label = {
                        'training_episodes': 'Training episodes',
                        'parallel_environments': 'Collection environments',
                        'evaluation_environments': 'Evaluation actor lanes (one physical case)',
                        'gradient_updates_per_round': 'Gradient updates per full collection',
                        'evaluation_interval_episodes': 'Evaluation interval (episodes)',
                        'batch_size': 'Optimizer minibatch size',
                        'initial_weight': 'Initial entropy bonus weight',
                        'final_weight': 'Final entropy bonus weight',
                        'target_entropy': 'Final target entropy (nats)',
                        'initial_target_entropy': 'Initial target entropy (nats)',
                        'hold_transitions': 'Initial exploration hold (training steps)',
                        'anneal_end_transitions': 'Exploration decay ends (training steps)',
                        'hold_fraction': 'Entropy hold (fraction of episode budget)',
                        'anneal_end_fraction': 'Entropy decay ends (fraction of episode budget)',
                        'learning_rate': 'Learning rate', 'batch_size': 'Minibatch transitions',
                        'policy_rate_hz': 'Action frequency [Hz]', 'velocity_limit_mps': 'Velocity limits X / Y / Z [m/s]',
                        'acceleration_limit_mps2': 'Acceleration limits X / Y / Z [m/s²]',
                        'reward_scale': 'Reward multiplier', 'radius_m': 'Horizontal radius [m]',
                        'height_lower_world_m': 'Minimum drone height [m]', 'height_upper_world_m': 'Maximum drone height [m]',
                        'duration_s': 'Maneuver duration [s]', 'settling_s': 'Maximum settling interval [s]',
                        'distance_m': 'Wall distance [m]', 'width_m': 'Target full width [m]', 'height_m': 'Target full height [m]',
                        'center_sigma_m': 'Center accuracy scale [m]', 'center_weight': 'Center accuracy weight',
                        'target_kl': 'Maximum cohort-mean policy KL', 'time_preference': 'Earlier-strike preference',
                        'request': 'Heading [rad] / reserved / target height [m] / forward speed [m/s]',
                    }.get(key, key.replace('_',' ').capitalize())
                    help_text = {
                        'parallel_environments':'Episodes simulated together in each collection. More lanes require more memory; speed is not guaranteed to improve.',
                        'evaluation_environments':'Actor batch layout only. Deterministic evaluation runs one native physical lane from a fixed initial condition, not independent trials.',
                        'training_episodes':'Total episode budget including earlier completed episodes when resuming; not an iteration count.',
                        'evaluation_interval_episodes':'Evaluation happens when this cumulative episode boundary is crossed, plus first and final collections.',
                        'reward_scale':'Multiplies task returns for optimization. It does not change strike qualification or physical units.',
                        'workspace_penalty':'Outside-only soft cost. Zero inside/on the workspace; no action filter or enforced confinement.',
                        'time_preference':'Legacy: earlier-event factor. Simple hit modes: total elapsed-time cost over a full episode, not a movement charge.',
                        'reward_mode':'Frozen reward version. Changing it creates a new experiment; old checkpoints retain their original objective.',
                        'center_weight':'Accuracy shaping weight. Reward changes define a new experiment, not an exact continuation.',
                        'outreach_base_fraction':'Fraction of existing whip credit retained before joint reach/height shaping. Changing it defines a new experiment.',
                        'outreach_height_width_fraction':'Gaussian width for tip height minus tracked-drone height at the same release, divided by rest cable length. A smooth preference, not a success cutoff.',
                        'outreach_reach_weight':'Weight of normalized forward reach within one mechanics-gated event.',
                        'outreach_height_weight':'Weight of linear tip-to-drone height quality within the same event.',
                        'outreach_joint_weight':'Weight of forward reach times linear height quality at that event.',
                        'target_kl':'With the enabled post-update safeguard, trial actor updates are checked against cohort-mean KL; not a pointwise action guarantee.',
                        'initial_weight':'PPO entropy bonus coefficient, not a guaranteed achieved entropy.',
                        'hold_transitions':'Exploration hold measured in collected transitions, not episodes or wall-clock seconds.',
                        'anneal_end_transitions':'Transition count where linear exploration decay reaches its final value.',
                        'hold_fraction':'0.1 means hold initial entropy for the first 10% of completed training episodes. Evaluation and discarded collections do not count.',
                        'anneal_end_fraction':'0.8 means reach final entropy at 80% of the original episode budget; checkpoint resume preserves that budget, even if extended.',
                        'plateau_detection':'Stage-aware ALL-lane rolling-window heuristic, not proof of convergence. Warn or safely stop at a completed collection boundary. Never before40 collections; hard episode budget remains the cap.',
                        'minimum_collections':'Minimum completed training collections before plateau stopping. Cannot be less than40; failures/user stops remain possible earlier.',
                        'mechanics':'Research diagnostic definitions; changing them changes the interpretation of a whip, not merely the display.',
                        'termination':'Geometric quality mode stops at the first geometric forward target hit; retained strict tasks stop at the first qualified hit. No unexecuted suffix is learned.',
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
                        form.addRow(label,row); self.fields.append((owner,key,lambda controls=controls:[c.value() for c in controls]))
                        continue
                    elif key=='target_observation':
                        control=QtWidgets.QLabel('Target position relative to initial drone position (world axes)')
                    elif key=='stage3_actor_training':
                        control=QtWidgets.QLabel('Retired combined adaptation (historical checkpoint metadata; training blocked)')
                    elif key=='target_sampler':
                        control=QtWidgets.QLabel('Retired target-grid sampler (historical checkpoint metadata; training blocked)')
                    elif key=='training_stage':
                        control=QtWidgets.QLabel(f'Stage {value} (select stage in the main window)')
                    elif isinstance(value,bool):
                        control=QtWidgets.QCheckBox(); control.setChecked(value)
                        form.addRow(label,control); self.fields.append((owner,key,control.isChecked))
                    elif type(value) is int:
                        # Signed task values (e.g. failure_return=-1) must round
                        # trip unchanged. The configuration validator sets semantics.
                        control=QtWidgets.QSpinBox(); control.setRange(-10000000,10000000); control.setValue(value)
                        form.addRow(label,control); self.fields.append((owner,key,control.value))
                    elif isinstance(value,float):
                        control=QtWidgets.QDoubleSpinBox(); control.setDecimals(9); control.setRange(-1e9,1e9); control.setValue(value)
                        form.addRow(label,control); self.fields.append((owner,key,control.value))
                    elif key=='device':
                        control=QtWidgets.QComboBox(); control.addItems(['cuda','cpu']); control.setCurrentText(value)
                        form.addRow(label,control); self.fields.append((owner,key,control.currentText))
                    else:
                        control=QtWidgets.QLineEdit(str(value))
                        if key in ('task','model','scene','speed_mode','action_interface','algorithm','termination','status'): control.setReadOnly(True)
                        form.addRow(label,control); self.fields.append((owner,key,control.text))
                    if key in ('training_stage','target_observation','stage3_actor_training','target_sampler'): form.addRow(label,control)
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
            settings,snapshot,_,scene=load(self.settings)
            task_configuration(self.task,settings,snapshot['model'],scene)
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
        self.initialize_actor_path=None
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
        self.rounds.setToolTip('Total training budget, excluding evaluation. PPO updates use valid current-cohort transitions and may stop on KL; SAC allocates updates by episode count.')
        self.mode=QtWidgets.QComboBox(); self.mode.addItems(['New training','Resume checkpoint','Fine-tune actor (fresh critic)']); self.mode.currentIndexChanged.connect(self.mode_changed)
        self.checkpoint_label=QtWidgets.QLabel('New policy'); self.checkpoint_label.setWordWrap(True)
        self.checkpoint_button=QtWidgets.QPushButton('Choose checkpoint'); self.checkpoint_button.clicked.connect(self.choose_checkpoint); self.checkpoint_button.setEnabled(False)
        grid.addWidget(self.mode,2,0); grid.addWidget(self.checkpoint_label,2,1,1,2); grid.addWidget(self.checkpoint_button,2,3)
        self.stage_presets = QtWidgets.QComboBox()
        self.stage_presets.addItem('Select current training stage…', None)
        for stage,label in ((1,'Stage 1 · compact whip · acceleration'),(2,'Stage 2 · broad delivery · acceleration'),(3,'Stage 3 · precision ±5 cm · acceleration'),(4,'Stage 4 · recovery · retained velocity')):
            self.stage_presets.addItem(label,stage)
        self.stage_presets.setCurrentIndex(self.stage_presets.findData(training_stage(self.settings)))
        self.stage_presets.currentIndexChanged.connect(self.select_stage_preset)
        grid.addWidget(self.stage_presets,3,0,1,4)
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
        t=self.task; wall=t['strike_surface']; ws=t['workspace']
        method=algorithm(self.settings).upper(); stage=training_stage(self.settings)
        self.setup.setTitle(f'Stage {stage} · '+method)
        self.setup_toggle.setText(f'Training setup · Stage {stage} · {method} · {self.settings["device"].upper()} · {self.settings["parallel_environments"]} lanes')
        self.setWindowTitle(f"Training — {method} — {self.config_path.stem} — reward ×{self.settings.get('reward_scale', 1):g}")
        goal={1:'Learn a broad directed strike',2:'Improve accuracy with staged target narrowing',3:'Robustness — blocked until uncertainty is identified',4:'Stage4 learned recovery'}[stage]
        quality_mode=t['matching'].get('reward_mode') == 'geometric_hit_quality_v1'
        compact_mode=t['matching'].get('reward_mode') == 'directed_compact_whip_v1'
        level_mode=t['matching'].get('reward_mode') == 'directed_level_outreach_whip_v1'
        linear_height_mode=t['matching'].get('reward_mode') == 'directed_outreach_height_whip_v1'
        whip_mode=t['matching'].get('reward_mode') in ('directed_whip_v1', 'directed_discovery_bounded_v1', 'directed_shape_release_v1', 'directed_unified_fold_v1', 'directed_fold_formation_v1', 'directed_fold_release_aligned_v1', 'directed_compact_whip_v1', 'directed_level_outreach_whip_v1', 'directed_outreach_speed_whip_v1', 'directed_outreach_height_whip_v1')
        speed_delivery_mode=t['matching'].get('reward_mode') == 'directed_outreach_speed_delivery_v1'
        crossing_priority_mode=t['matching'].get('reward_mode') == 'crossing_priority_delivery_v1'
        delivery_mode=t['matching'].get('reward_mode') in ('directed_delivery_v1', 'directed_outreach_speed_delivery_v1')
        outreach_precision_mode=t['matching'].get('reward_mode') == 'directed_outreach_speed_precision_v1'
        precision_mode=t['matching'].get('reward_mode') in ('directed_precision_v1', 'directed_outreach_speed_precision_v1')
        if whip_mode: goal='Learn directed bend propagation and unfolding'
        if t['matching'].get('reward_mode') == 'directed_shape_delivery_v1': goal='Experimental Stage 2: same-release broad-target delivery'
        if t['matching'].get('reward_mode') == 'directed_shape_release_v1': goal='Experimental Stage 1: outward bend and directed shape release'
        if t['matching'].get('reward_mode') == 'directed_fold_formation_v1': goal='Stage 1 warm-up: form a fold; release skill remains unlearned'
        if t['matching'].get('reward_mode') == 'directed_unified_fold_v1': goal='Stage 1: form, propagate and release a fold toward +X'
        excitation_mode=t['matching'].get('reward_mode') == 'flexible_excitation_v1'
        if t['matching'].get('reward_mode') == 'flexible_delivery_v1': goal='Stage 2: fixed +X broad delivery; no internal curriculum'
        if t['matching'].get('reward_mode') == 'first_hit_flexible_delivery_v2': goal='Stage 2: first useful +X hit; five-second deadline'
        if t['matching'].get('reward_mode') == 'retained_fold_delivery_v1': goal='Stage 2: retained fold progress toward a useful +X strike'
        if t['matching'].get('reward_mode') in ('continuous_horizontal_whip_v1', 'continuous_horizontal_whip_soft_workspace_v2','continuous_horizontal_whip_distance_v3','continuous_horizontal_whip_full_horizon_v4', 'continuous_directional_whip_full_horizon_v5', 'continuous_directional_whip_four_second_v6'): goal='Stage 1: continuous horizontal fold transport and opening, any heading'
        if t['matching'].get('reward_mode') in ('continuous_directional_whip_full_horizon_v5', 'continuous_directional_whip_four_second_v6'): goal='Stage 1: fold transport and opening toward fixed world +X'
        if excitation_mode: goal='Experimental Stage 1: flexible cable excitation and distal output'
        if t['matching'].get('reward_mode') == 'directed_fold_release_aligned_v1': goal='Stage 1: outward fold opening with aligned forward release'
        if compact_mode: goal='Learn a compact directed whip with useful forward reach'
        if level_mode: goal='Learn forward whip outreach near the drone\'s current height'
        if linear_height_mode: goal='Improve whip outreach and tip height together'
        if delivery_mode: goal='Deliver the learned whip toward a fixed broad target; no narrowing'
        if crossing_priority_mode: goal='Hit the fixed broad target and bring actual crossings toward its center'
        if precision_mode: goal='Precision Stage 3: narrow actual whip crossings to +/-5 cm; no randomization'
        if stage==4: goal='Stage4 learned recovery: precision whip, return and sustained hover; no randomization'
        terminal=('first geometric hit or timeout' if quality_mode else
            'first qualified hit or timeout' if t['episode'].get('termination')=='first_qualified_hit_v1' else 'full horizon')
        impact=(f"speed/straightness rewarded; {t['request'][3]:g} m/s / {wall['maximum_entry_angle_deg']:g}° diagnostic only" if quality_mode else
            f"forward speed ≥ {t['request'][3]:g} m/s · entry ≤ {wall['maximum_entry_angle_deg']:g}°")
        if whip_mode or excitation_mode: impact='target, 3 m/s and 30° are diagnostics; no hit reward or hit termination'
        if t['matching'].get('reward_mode') == 'flexible_delivery_v1': impact='confirmed forward crossings; q>=1 quality measured separately; 3 m/s / 30° diagnostic only'
        if t['matching'].get('reward_mode') == 'first_hit_flexible_delivery_v2':
            impact='first confirmed forward crossing with q>=1; timeout at5s;3m/s/30° diagnostic only'
            terminal='first useful hit, physical failure, or5second deadline'
        if delivery_mode: impact='same-event spatial guidance; 3 m/s / 30° diagnostic only; no hit termination'
        if t['matching'].get('reward_mode') == 'directed_shape_delivery_v1': impact='actual forward rectangle hit in the selected release; 3 m/s / 30° separately diagnostic; full horizon'
        if precision_mode: impact='actual forward crossing center accuracy; 3 m/s / 30° diagnostic only; no hit termination'
        if outreach_precision_mode: impact='same-release rectangle and world XYZ center guidance; 3 m/s / 30° diagnostic only'
        if crossing_priority_mode: impact='geometric hit bonus plus strict bonus at ≥3 m/s and ≤30°; full horizon'
        self.config_label.setText(f"{goal}\n"
            f"Compute: {self.settings['device'].upper()} · {self.settings['parallel_environments']} collection lanes · {self.settings['policy_rate_hz']} Hz {self.settings['action_interface'].split('_')[0]} actions · 1 fixed evaluation case\n"
            f"Task: target {wall['width_m']:g} × {wall['height_m']:g} m at wall {wall['distance_m']:g} m / height {t['request'][2]:g} m · {impact} · stop at {terminal}\n"
            f"Workspace: radius {ws['radius_m']:g} m / height {ws['height_lower_world_m']:g}–{ws['height_upper_world_m']:g} m · outside-only cost {self.settings['workspace_penalty']['weight']:g} · reward ×{self.settings.get('reward_scale',1):g}")
        entropy=self.settings.get('ppo',{}).get('entropy',{})
        if self.settings['action_interface'] == 'acceleration_direct_v1':
            self.config_label.setText(self.config_label.text()+
                f"\nAcceleration: world XYZ limits {self.settings['acceleration_limit_mps2']} m/s², excluding gravity; fixed yaw. Attitude stabilization only; zero action does not brake or hold position.")
        if 'hold_fraction' in entropy:
            self.config_label.setText(self.config_label.text()+
                f"\nEntropy: {entropy['initial_weight']:g} → {entropy['final_weight']:g} · completed training episodes · hold first {100*entropy['hold_fraction']:g}% · decay ends at {100*entropy['anneal_end_fraction']:g}% · original budget preserved on resume")
        if entropy.get('mode') == 'exponential_episodes_v1':
            self.config_label.setText(self.config_label.text()+
                f"\nEntropy: {entropy['initial_weight']:g} → {entropy['final_weight']:g} asymptotically · exponential decay over {entropy['decay_episodes']} completed episodes · no hold · elapsed clock preserved on resume")
        if entropy.get('mode') == 'exponential_budget_v1':
            self.config_label.setText(self.config_label.text()+
                f"\nEntropy: {entropy['final_weight']:g} + ({entropy['initial_weight']:g} − {entropy['final_weight']:g}) × exp(−{entropy['decay_exponent']:g} n/N); N is the saved planned episode budget; no reset on resume")
        if t['matching'].get('reward_mode') in ('continuous_horizontal_whip_v1', 'continuous_horizontal_whip_soft_workspace_v2','continuous_horizontal_whip_distance_v3','continuous_horizontal_whip_full_horizon_v4', 'continuous_directional_whip_full_horizon_v5', 'continuous_directional_whip_four_second_v6'):
            self.config_label.setText(
                "Continuous Stage1: recent fold travel × horizontal reach² × outward alignment² × extending output.\n"
                "Full7s except safety failure; no target, count bonus, early completion, time/effort/path cost.\n"
                "Rate = 10 F H² A² min(positive world radial, relative radial, endpoint-extension speed) / L.\n"
                "Failure total −10, cancelling all accrued output. Continuous score is not validated whip success.\n"
                f"{self.settings['parallel_environments']} lanes · ten links · 50Hz acceleration · entropy .001+.019exp(−8n/N), N={self.settings['training_episodes']}.")
        if t['matching'].get('reward_mode') == 'continuous_horizontal_whip_soft_workspace_v2':
            self.config_label.setText(self.config_label.text().replace(f"N={self.settings['training_episodes']}","N=25600 (original budget)")+
                "\nSoft workspace: C=10×mean(radial excess² + height excess²), metres, fixed7s denominator. Crossing boundary does not terminate."
                "\nNet x=Q−C; return=x if x≥0, else5x/(5−x). Physical/numerical failure remains−10."
                "\nActor and original entropy clock continued; critic/optimizers fresh because reward changed. Remaining178collections of200.")
        if t['matching'].get('reward_mode') == 'continuous_horizontal_whip_distance_v3':
            self.config_label.setText(
                "Stage1: continuous horizontal-whip output Q minus squared-distance workspace cost C.\n"
                "C=10×fixed7s mean(radial excess² + height excess²), metres. Boundary crossing does not stop an episode.\n"
                "Valid return Q−C, without compression. Physical/numerical failure keeps Q−C and subtracts1×remaining seconds. No fixed−10.\n"
                "Safe horizon7s; tilt60°/cable-height0.05m/numerical guards retained. No time reward or whip-count limit.\n"
                "128lanes · remaining175collections (25already spent). Actor retained; critic/optimizers fresh.\n"
                "Entropy .001+.019exp(−8×(3200+n)/25600); original budget and elapsed clock preserved.")
        if t['matching'].get('reward_mode') == 'continuous_horizontal_whip_full_horizon_v4':
            self.config_label.setText(
                "Stage1: continuous horizontal-whip output Q minus squared-distance workspace cost C.\n"
                "Finite, numerically admissible trajectories run7s. Tilt60° and cable-height0.05m violations are recorded, not termination conditions.\n"
                "Q−C is unchanged; these physical violations have no extra reward penalty. No early completion or whip-count limit.\n"
                "Numerical corruption still aborts and charges1×remaining seconds. No fixed−10.\n"
                "Full horizon does not imply physical validity; no ground-contact model is added.\n"
                f"{self.settings['parallel_environments']}lanes · episode budget {self.settings['training_episodes']}. Checkpoint continuation preserves the existing learning state and entropy clock.")
        if t['matching'].get('reward_mode') in ('continuous_directional_whip_full_horizon_v5', 'continuous_directional_whip_four_second_v6'):
            self.config_label.setText(
                "Stage1: continuous fold-supported output toward fixed WORLD +X.\n"
                "Q = integral 10 F H² A² U/L dt. H = positive attachment-relative X reach/L; A = both-frame +X alignment.\n"
                "U = positive minimum(world X speed, relative X speed, cable extension speed). No sideways/backward output credit.\n"
                f"Return Q−C; squared-distance workspace mean over {t['episode']['duration_s']:g}s. No completion/count/time/effort cost.\n"
                "Tilt/low cable remain diagnostics; only numerical failure aborts with remaining-seconds cost.\n"
                f"{self.settings['parallel_environments']} lanes · {self.settings['training_episodes']//self.settings['parallel_environments']} collections · fresh actor/critic.\n"
                f"Entropy .001+.019exp(−8n/{self.settings['training_episodes']}); near floor at75% of this budget.")
        if t['matching'].get('reward_mode') == 'horizontal_fold_outreach_v2':
            self.config_label.setText(
                f"Experimental Stage1: form, propagate and open a fold into horizontal outreach in any heading.\n"
                f"{self.settings['parallel_environments']} lanes · 10 links · 50 Hz world acceleration · maximum 7 s.\n"
                f"Hard drone workspace: radius {ws['radius_m']:g} m, world height {ws['height_lower_world_m']:g}–{ws['height_upper_world_m']:g} m; checked at native substeps.\n"
                "Completed event: 5H² − 0.1T; timeout: −2.7; failure: −10 − 0.1T. No target-hit reward.\n"
                f"Entropy: 0.001 + 0.019 exp(−8n/N), N = saved planned budget ({self.settings['training_episodes']} episodes for a fresh run).")
        if t['matching'].get('reward_mode') == 'retained_fold_delivery_v1':
            self.config_label.setText(
                "Experimental Stage2: form, propagate and open the SAME fold toward a broad +X target.\n"
                f"{self.settings['parallel_environments']} lanes · 10 links · 50 Hz world acceleration · maximum 7 s.\n"
                f"Hard drone workspace: radius {ws['radius_m']:g} m, world height {ws['height_lower_world_m']:g}–{ws['height_upper_world_m']:g} m; checked at native substeps.\n"
                "Progress: 5 × increases in episode-best coherent progress P. Verified useful hit: total15. Completed miss or timeout: 5P−2. Failure: 5P−10, valid-prefix credit only.\n"
                "No time, acceleration-effort or travelled-distance cost. Progress is not hit success.\n"
                "Hit: same-opening forward rectangle crossing, H≥0.75, world forward speed≥1m/s, both outward radial/cone conditions.\n"
                f"Entropy: 0.001 + 0.019 exp(−8n/N), N = saved planned budget ({self.settings['training_episodes']} episodes for a fresh run).")
        if t['matching'].get('reward_mode') in ('directed_hit_v1', 'geometric_hit_quality_v1'):
            self.config_label.setText(self.config_label.text()+
                f"\nEXPERIMENT: fixed hit bonus {wall['strike_bonus_weight']:g} · spatial record credit ≤ {t['matching']['shaping_weight']:g} · full-horizon time cost {t['matching']['time_preference']:g} · mechanics diagnostic only")
        if quality_mode:
            m=t['matching']
            self.config_label.setText(self.config_label.text()+
                f"\nHit quality: speed bonus ≤ {m['hit_speed_weight']:g} (scale {m['hit_speed_scale_mps']:g} m/s) · straightness bonus ≤ {m['hit_straightness_weight']:g} · neither is a hit threshold")
        if t['matching'].get('reward_mode') in ('directed_outreach_speed_whip_v1', 'directed_outreach_speed_delivery_v1', 'directed_outreach_speed_precision_v1'):
            m=t['matching']
            self.config_label.setText(self.config_label.text()+
                f"\nEvent: MU/(MU+{m['outreach_mechanics_scale']:g}) × D × reach/L × height × speed² × path factor · speed scale {m['outreach_speed_scale_mps']:g} m/s · height width {m['outreach_height_width_fraction']:g} L")
        if linear_height_mode:
            m=t['matching']
            bound=m['outreach_reach_weight']+m['outreach_height_weight']+m['outreach_joint_weight']
            self.config_label.setText(self.config_label.text()+
                f"\nStage1 event: K × ({m['outreach_reach_weight']:g}r + {m['outreach_height_weight']:g}h + {m['outreach_joint_weight']:g}rh) ≤ {bound:g}. r=clipped forward reach/L; h=clip(1−|tip_z−drone_z|/L,0,1). K keeps mechanics, alignment, speed and path factors. Same release; workspace cost is separate.")
        if level_mode:
            m=t['matching']
            self.config_label.setText(self.config_label.text()+
                f"\nStage1 experiment: best W × ({m['outreach_base_fraction']:g} + {1-m['outreach_base_fraction']:g} × reach/L × H) × exp(-{m['preparation_path_weight']:g} × drone path/L) ≤ 1. H is Gaussian tip-to-drone height quality, width {m['outreach_height_width_fraction']:g} L. All terms use the same release; workspace cost is separate.")
        elif compact_mode:
            m=t['matching']
            self.config_label.setText(self.config_label.text()+
                f"\nStage1 experiment: best W × (1 + {m['reach_weight']:g} × forward reach/L) × exp(-{m['preparation_path_weight']:g} × drone path/L) ≤ {1+m['reach_weight']:g}. Path is measured before the same release; workspace cost is separate. No new hit/whip threshold.")
        elif t['matching'].get('reward_mode') == 'directed_shape_delivery_v1':
            self.config_label.setText(self.config_label.text()+
                '\nShape delivery: max finalized S × (1 + exp(-0.5 × (rectangle distance at SAME peak-shape sample / 0.70 m)^2)). Actual forward hits inside that SAME selected release are checked separately. Failure −10; zero additional workspace/acceleration/time costs. Full-episode movement cost retained. Inherited AR(1) exploration; complete actor transfer preserves learned scale.')
        elif t['matching'].get('reward_mode') == 'first_hit_flexible_delivery_v2':
            self.config_label.setText(self.config_label.text()+
                '\nFirst-hit delivery: time cost0.1/s. Hit F(B-C-0.1T); timeout -2+F(A-C-0.5); failure -10+F(-C-0.1T). B=5q/(1+q), A<1. Workspace charge keeps450sample reference. No post-hit tail, effort or path cost. Actor/clock inherited;451sample history retained and rolled beyond4.5s.')
        elif t['matching'].get('reward_mode') == 'flexible_delivery_v1':
            self.config_label.setText(self.config_label.text()+
                '\nFixed delivery: q=min(positive world-X tip speed, shape-residual speed)/(1 m/s). Approach <1; crossing 5q/(1+q). Valid F(max utility minus full-episode workspace cost); failure -10+F(-accrued cost). No path/effort/time cost. Stage1 actor and exploration clock inherited. Full velocity commands deferred to Stage3.')
        elif excitation_mode:
            if self.settings.get('stage1_return_transform') == 'negative_net_rational_v1':
                reward_text = (' Uncapped quadratic workspace cost C. Valid return F(Q−C); failure −10+F(−C accrued). '
                    'F(x)=x for x≥0, otherwise 5x/(5−x). Valid returns >−5; failures ≤−10. '
                    'Farther excursions increase raw cost; very negative returns are compressed. No containment or learning guarantee.')
            elif self.settings['workspace_penalty']['mode'] == 'bounded_density_v1':
                reward_text = (' Historical bounded workspace cost: valid Q−C, failure −10−C accrued; C≤5. '
                    'Far-out instantaneous cost saturates.')
            else:
                reward_text = ' Historical unbounded quadratic cost: valid Q−C, failure −10−C accrued.'
            self.config_label.setText(self.config_label.text()+
                '\nExcitation guide Q: best same-sample min(world tip speed, nonrigid tip speed) / (1 m/s). '
                'Sidearm motion allowed; no fold, direction, path or acceleration term. Experimental guide, not whipping success; '
                '120 collections, no automatic stage transition.'+reward_text)
        elif t['matching'].get('reward_mode') == 'directed_unified_fold_v1':
            self.config_label.setText(self.config_label.text()+
                '\nUnified fold: best SAME attempt (0.5 formation + 0.5 propagation + aligned release). Preparation credit is not success. Quadratic outside-workspace cost, coefficient0.1 / (0.1m)^2, averaged over4.5s; no path cost. Failure −10 minus accrued workspace cost. Fresh120-collection run; no automatic stage transition.')
        elif t['matching'].get('reward_mode') in ('directed_fold_formation_v1','directed_fold_release_aligned_v1'):
            formation=t['matching'].get('reward_mode')=='directed_fold_formation_v1'
            self.config_label.setText(self.config_label.text()+
                ('\nFormation guide: best attachment-side tangent reversal angle in radians. Stops at30 collections for independent motion review; full120-collection entropy budget retained. This is not a successful whip.' if formation else
                 '\nFold release: same tracked outward fold must open and gain forward tip speed. Multiply output by squared3D alignment of BOTH shape and world tip velocities with intended direction. Lateral/backward output has no bonus. Reviewed warm-up actor required; remaining total120 budget only.')+
                '\nMovement cost0.25 × full-episode drone path/cable length; failure replaces entire return with−10. Physics/control unchanged; no automatic later-stage promotion.')
        elif t['matching'].get('reward_mode') == 'directed_shape_release_v1':
            self.config_label.setText(self.config_label.text()+
                '\nShape release: best finalized min(positive shape-forward speed, positive world-forward speed) / (1 m/s), at peak shape speed; shape gain ≥0.25 m/s. Failure return −10. Zero workspace/acceleration/time costs. Experimental kinematics, not a validated whip classifier.'+
                f"\nExploration: conditional AR(1), 100-ms correlation time; initial marginal latent std 0.370409. Budget {self.settings['training_episodes'] // self.settings['parallel_environments']} collections; no automatic plateau stop or stage advance.")
            if 'movement_penalty' in self.settings:
                weight=self.settings['movement_penalty']['weight_per_cable_length']
                self.config_label.setText(self.config_label.text()+
                    f'\nMovement cost: {weight:g} × full-episode drone path / cable length, including post-release drift. Failure still replaces the entire return with −10. Travel is not physical energy.')
        elif t['matching'].get('reward_mode') == 'directed_discovery_bounded_v1':
            self.config_label.setText(self.config_label.text()+
                '\nStage1 discovery: 0.5 × best directed whip proxy − bounded workspace cost (≤0.25). Failure replaces the total return with −1. Hits are diagnostics. Motion review is required before later-stage transfer; no physical whip classifier is implied.')
        elif t['matching'].get('reward_mode') == 'directed_whip_v1':
            self.config_label.setText(self.config_label.text()+
                '\nStage1: best directed mechanics × unfolding event ≤ 1; outside-only cost. No acceleration, hit, accuracy or time cost/bonus.')
        if speed_delivery_mode:
            self.config_label.setText(self.config_label.text()+
                '\nStage2: best speed/level outreach event × (1 + target proximity) ≤ 2; path cost retained; fixed broad target, no narrowing or hit bonus.')
        elif crossing_priority_mode:
            m=t['matching']
            self.config_label.setText(self.config_label.text()+
                f"\nStage2: {m['crossing_whip_weight']:g} × normalized Stage1 whip + best crossing [C + {m['crossing_geometric_weight']:g} × geometric hit + {m['crossing_strict_weight']:g} × strict hit]. C=1/[1+(side/{m['crossing_side_scale_m']:g} m)²+(height/{m['crossing_height_scale_m']:g} m)²]. Separate whip/crossing records; repeated crossings do not accumulate.")
        elif delivery_mode:
            self.config_label.setText(self.config_label.text()+
                '\nStage2: best W × (1 + target proximity) ≤ 2; same-event mechanics preserved; no hit, hardness, centering or time bonus.')
        if outreach_precision_mode:
            self.config_label.setText(self.config_label.text()+
                '\nPrecision event: best B × (1 + rectangle proximity + XYZ center Gaussian) ≤ 3. Same observed release; sigma 0.25/0.15/0.075/0.05 m across target ladder. 80% geometric hits for 3 all-valid cohorts + fixed replay; workspace excess measured, not a progression gate.')
        if precision_mode and not outreach_precision_mode and stage!=4:
            self.config_label.setText(self.config_label.text()+
                '\nStage3: best W × (1 + crossing center Gaussian) ≤ 2; 80% hits for 3 cohorts + fixed replay; workspace excess measured, not a progression gate.')
        if stage==4:
            r=t['recovery']
            settling=r.get('quality_mode')=='return_and_cable_settle_v1'
            gate='actual-hit gate' if settling else 'same-event W*C gate'
            self.config_label.setText(self.config_label.text()+
                f"\nStage4: first actual hit (miss timeout {r['whip_duration_s']:g}s), then {r['recovery_duration_s']:g}s recovery; weight {r['weight']:g}, {gate}; positive bound {(3 if outreach_precision_mode else 2)+r['weight']:g}. No controller switch. Return to episode-start drone position.\nHold diagnostic: position ≤{r['position_tolerance_m']:g}m, speed ≤{r['speed_tolerance_mps']:g}m/s for final {r['final_hold_s']:g}s; separate workspace charges, no RMS gate.")
            if settling:
                self.config_label.setText(self.config_label.text()+f"\nFrozen whip + internal PD return + learned recovery residual. Cable hold: relative RMS speed ≤{r['cable_speed_tolerance_mps']:g}m/s, horizontal RMS shape ≤{r['cable_horizontal_tolerance_m']:g}m. Faster settling improves window quality; fixed recovery duration.")
        if t['matching'].get('alignment_mode')=='cone_plateau_v1':
            self.config_label.setText(self.config_label.text()+f"\nCone reward version: alignment plateau inside {t['matching']['alignment_cone_deg']:g}°; continuous tail outside, no geometric-hit angle gate. Speed factors unchanged.")
        if 'plateau_detection' in self.settings:
            c=self.settings['plateau_detection']
            guard='stable entropy floor required' if c['require_entropy_floor'] else 'entropy guard explicitly disabled'
            self.config_label.setText(self.config_label.text()+f"\nPlateau mode {c['mode']}: NEVER before {c['minimum_collections']} completed collections; compare {c['window_collections']}+{c['window_collections']} same-target cohorts, patience {c['patience_checks']} checks. Final target required; {guard}; not proof of convergence. Hard budget remains a cap.")
        self.description_details=self.config_label.text()
        self.config_label.setToolTip(self.description_details)
        self.config_label.setText(f"{goal}\n{self.settings['device'].upper()} · {self.settings['parallel_environments']} lanes · {self.settings['policy_rate_hz']} Hz {self.settings['action_interface'].split('_')[0]} actions · {self.settings['training_episodes']:,} training episodes")

    def select_stage_preset(self, index):
        stage = self.stage_presets.itemData(index)
        if stage is None or stage not in CURRENT_PRESETS or self.process is not None or self.mode.currentIndex() == 1:
            return
        try:
            self.settings,self.snapshot,_,_ = load(CURRENT_PRESETS[stage])
            self.task = self.snapshot['task']; self.config_path = CURRENT_PRESETS[stage]
            self.initialize_actor_path = self.resume_path = None
            self.mode.setCurrentIndex(0 if stage == 1 else 2)
            self.checkpoint_label.setText('New policy' if stage == 1 else f'Choose Stage {stage-1} actor checkpoint')
            self.sync_budget_control(); self.update_description()
        except Exception as exc:
            self.error(exc)

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
            path=project_path(filename); task_path=path.with_name(path.stem+'_task.json')
            if task_path==path: raise ValueError('Separate task filename required')
            self.settings['training_episodes']=self.rounds.value()
            self.settings['task']=task_path.relative_to(ROOT).as_posix()
            save_json(task_path,self.task); save_json(path,self.settings); self.config_path=path
            return True
        except Exception as exc: self.error(exc); return False

    def mode_changed(self,index):
        self.checkpoint_button.setEnabled(index in (1,2))
        self.settings_button.setEnabled(True)
        self.settings_button.setText('Inspect configuration…' if index==1 else 'Configure experiment…')
        for b in (self.open_button,self.preset_button): b.setEnabled(index in (0,2))
        if index==0:
            self.resume_path=None
            self.initialize_actor_path=None
            self.checkpoint_label.setText('New policy'); self.sync_budget_control(); self.update_description()
        elif index==1:
            self.initialize_actor_path=None
        else:
            self.resume_path=None
            self.checkpoint_label.setText(str(self.initialize_actor_path or 'Choose Stage 1 actor checkpoint'))
            clock=self.settings.get('ppo',{}).get('transfer_entropy_clock','inherit')
            self.config_label.setText(f'Actor-only transfer; fresh value and optimizers. Entropy clock: {clock}. Not exact resume.')

    def choose_checkpoint(self):
        path,_=QtWidgets.QFileDialog.getOpenFileName(self,'Select learner checkpoint',str(ROOT/'models'),'Checkpoint (*.pt)')
        if path:
            try:
                path=project_path(path)
                import torch
                saved = torch.load(path,map_location='cpu',weights_only=True)
                settings = saved['snapshot']['settings']
                if self.mode.currentIndex()==2:
                    self.initialize_actor_path=Path(path)
                    self.checkpoint_label.setText('Actor only: '+str(path))
                else:
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
            if self.mode.currentIndex()==2 and self.initialize_actor_path is None: raise ValueError('Choose the Stage 1 actor checkpoint first')
            settings=deepcopy(self.settings)
            settings['training_episodes']=self.rounds.value()
            # Unsaved UI edits are frozen into this run, never written into defaults.
            directory=create_run(settings,self.resume_path,task=self.task,initialize_actor=self.initialize_actor_path)
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
        for control in (self.mode,self.rounds,self.open_button,self.preset_button,self.checkpoint_button,self.stage_presets): control.setEnabled(not running)
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
        if not self.current_run or not self.process: return
        try:
            progress=read_json(self.current_run/'progress.json')
            if progress.get('phase') != 'curriculum_review': return
            review=read_json(self.current_run/'curriculum_review.json')
            answer=QtWidgets.QMessageBox.question(self,'Motion review',
                'Have you reviewed the saved stochastic motions in Results and confirmed directed whipping mechanics, not just wall hits? Advance to curriculum stage 2?')
            if answer != QtWidgets.QMessageBox.StandardButton.Yes: return
            save_json(self.current_run/'curriculum_review_approved.json',
                      dict(identity=review['identity'],round=review['round']))
            self.review_button.setEnabled(False)
        except Exception as exc: self.error(exc)

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
        paths=[p for root in (PPO_STAGE4_RUNS,PPO_STAGE3_RUNS,PPO_STAGE2_RUNS,PPO_RUNS,RUNS) if root.exists()
               for p in root.iterdir() if p.is_dir() and (p/'configuration.json').exists()]
        for path in sorted(paths,key=lambda p:p.name,reverse=True):
            saved_settings=read_json(path/'configuration.json')['settings']
            label=f'{algorithm(saved_settings).upper()} Stage {training_stage(saved_settings)}'
            self.runs.addItem(label+' · '+path.name,path)
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
                    if 'stage' in progress: label+=f" · Stage {progress['stage']}"
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
        retained_progress_mode=selected_snapshot['task']['matching'].get('reward_mode')=='retained_fold_delivery_v1'
        continuous_mode=selected_snapshot['task']['matching'].get('reward_mode')in ('continuous_horizontal_whip_v1', 'continuous_horizontal_whip_soft_workspace_v2','continuous_horizontal_whip_distance_v3','continuous_horizontal_whip_full_horizon_v4', 'continuous_directional_whip_full_horizon_v5', 'continuous_directional_whip_four_second_v6')
        self.update_overview(directory,history,selected_settings)
        self.whip_diagnostics.set_run(directory,history)
        method=algorithm(selected_settings)
        scale=selected_settings.get('reward_scale',1.0)
        self.loss_plot.title=method.upper()+' losses (last minibatch)'
        self.return_plot.title=f'Episode return (reward ×{scale:g})'
        self.reward_plot.title=f'Event reward components (×{scale:g})'
        self.return_plot.setToolTip('Training reward units. Original task returns are retained in the log; compare runs using the same scale.')
        self.evaluations.setHorizontalHeaderLabels(['Iteration',f'Return ×{scale:g}','Mechanics','Strike','Valid','Outside workspace (s)',f'Workspace cost ×{scale:g}'])
        rounds=[e for e in history if e['phase']=='round']; evaluations=[e for e in history if e['phase']=='evaluation']
        rows=[e for e in history if e['phase'] in ('evaluation','curriculum_sample')]
        self.evaluation_rows=rows
        if history:
            latest=history[-1]
            for key in ('round','episodes','steps','gradient_updates'):
                item=next((e[key] for e in reversed(history) if key in e),None)
                self.cards[key].setText(str(item) if item is not None else '—')
            seconds=int(latest.get('elapsed_s',0)); self.cards['elapsed'].setText(f'{seconds//3600:02}:{seconds//60%60:02}:{seconds%60:02}')
        curriculum=('target_curriculum' in selected_settings or any(e['phase']=='curriculum_stage' for e in history))
        self.return_plot.set_series([('Training (stage reward)' if curriculum else 'Training',[(e['round'],scale*e['return_mean']) for e in rounds],False),
            ('Strict evaluation' if curriculum else 'Evaluation',[(e['round'],scale*e['evaluation_return']) for e in evaluations],False),
            ('Best',[(e['round'],scale*e['best_return']) for e in evaluations],True)])
        quality_history=any('geometric_hit_success' in e.get('components',{}) for e in rounds+evaluations)
        motion_terms=(('mechanical','Mechanics'),('matching','Matching'),('geometric_hit_success' if quality_history else 'strike_success','Geometric hit' if quality_history else 'Strike'))
        if quality_history:motion_terms+=(('strict_strike_success','Strict 3 m/s / 30° diagnostic'),)
        unified_history=any('unified_fold_score' in e.get('components',{}) for e in rounds+evaluations)
        excitation_history=any('excitation_score' in e.get('components',{}) for e in rounds+evaluations)
        if unified_history:
            motion_terms=(('formation_fraction','Selected formation'),('propagation_fraction','Selected propagation'),
                          ('completed_release_fraction_all','Completed aligned release / all episodes'))
        if excitation_history:
            motion_terms=(('world_speed_at_q','World tip speed at selected sample (m/s)'),
                          ('shape_speed_at_q','Nonrigid tip speed at selected sample (m/s)'))
        self.mechanics_plot.set_series([(label,[(e['round'],e['components'].get(key)) for e in rounds],False)
            for key,label in motion_terms])
        speed_outreach_history=any('outreach_credit' in e.get('components',{}) for e in rounds+evaluations)
        base_key=('outreach_credit' if speed_outreach_history else
            'base_mechanics_credit' if any('base_mechanics_credit' in e.get('components',{}) for e in rounds+evaluations) else 'mechanics_credit')
        base_label='Outreach credit before path' if speed_outreach_history else 'Base mechanics credit'
        if unified_history:base_key,base_label='unified_fold_score','Unified attempt utility before workspace cost'
        if excitation_history:base_key,base_label='excitation_score','Excitation guide before workspace cost'
        self.reward_plot.set_series([(label,[(e['round'],scale*e['components'][key] if e['components'].get(key) is not None else None) for e in rounds],False)
            for key,label in ((base_key,base_label),('outreach_reach_credit','Outreach contribution'),('outreach_height_credit','Height contribution'),('outreach_joint_credit','Joint outreach/height contribution'),('reach_bonus','Reach bonus'),('level_outreach_bonus','Level outreach credit'),('preparation_cost','Preparation reduction'),('approach_reward','Approach'),('center_reward','Center guidance'),('recovery_credit','Recovery credit'),('strike_bonus','Strike bonus'),('hit_speed_bonus','Hit speed bonus'),('hit_straightness_bonus','Hit straightness bonus'),('time_cost','Time cost'),('workspace_cost','Workspace cost'))])
        if any('crossing_reward' in e.get('components',{}) for e in rounds+evaluations):
            self.reward_plot.set_series([(label,[(e['round'],scale*e['components'][key]
                if e['components'].get(key) is not None else None) for e in rounds],False) for key,label in (
                ('whip_support_reward','Whip support'),('crossing_center_reward','Crossing center guidance'),
                ('crossing_geometric_bonus','Geometric hit bonus'),('crossing_strict_bonus','Strict hit bonus'),
                ('workspace_cost','Workspace cost'))])
        if any('shape_release_score' in e.get('components',{}) for e in rounds+evaluations):
            self.mechanics_plot.set_series([(label,[(e['round'],e['components'].get(key)) for e in rounds],False)
                for key,label in (('q_shape_mps','Selected peak shape speed (m/s)'),
                                  ('shape_gain_mps','Selected shape speed gain (m/s)'),
                                  ('geometric_hit_success','Geometric hit diagnostic'))])
            self.reward_plot.set_series([(label,[(e['round'],e['components'].get(key)) for e in rounds],False)
                for key,label in (('shape_release_score','Raw shape release'),('shape_delivery_score','Same-event delivery utility'),('selected_event_geometric_hit_success','Selected release broad hit'),('movement_cost','Movement cost (subtracted)'))])
        if any('fold_formation_score' in e.get('components',{}) or 'fold_release_score' in e.get('components',{}) for e in rounds+evaluations):
            self.reward_plot.set_series([(label,[(e['round'],e['components'].get(key)) for e in rounds],False)
                for key,label in (('fold_formation_score','Formation guide (rad)'),('fold_release_score','Full fold release'),('movement_cost','Movement cost (subtracted)'))])
            self.mechanics_plot.set_series([(label,[(e['round'],e['components'].get(key)) for e in rounds],False)
                for key,label in (('root_fold_formed','Persistent root fold fraction'),('release_direction_alignment','Release direction alignment'))])
        self.valid_plot.set_series([('Fraction',[(e['round'],e['valid_fraction']) for e in rounds],False)])
        if retained_progress_mode:
            self.mechanics_plot.title='Fold phases · fraction of all attempts'
            self.mechanics_plot.set_series([(label,[(e['round'],e['components'].get(key)) for e in rounds],False)
                for key,label in (('fold_formed','Formed'),('fold_propagated','Travelled'),
                                  ('fold_opened','Opened'),('qualified_hit','Hit'))])
            self.reward_plot.title='Retained progress and terminal outcome'
            self.reward_plot.set_series([(label,[(e['round'],scale*e['components'][key]
                if e['components'].get(key) is not None else None) for e in rounds],False)
                for key,label in (('progress_credit','Progress'),('outcome_credit','Outcome'),
                                  ('net_episode_return','Net return'))])
            self.valid_plot.title='Validity and mean progress · all attempts'
            self.valid_plot.set_series([
                ('Valid',[(e['round'],e['valid_fraction']) for e in rounds],False),
                ('Mean P',[(e['round'],e['components'].get('progress')) for e in rounds],False),
                ('Failure',[(e['round'],e['components'].get('failure')) for e in rounds],False)])
            self.evaluations.setHorizontalHeaderLabels(['Collection',f'Return ×{scale:g}','Progress P',
                'Verified hit','Valid','Outside workspace (s)',f'Workspace cost ×{scale:g}'])
        else:
            self.mechanics_plot.title='Whipping event'
            self.valid_plot.title='Valid episodes'
        if continuous_mode:
            self.mechanics_plot.title='Continuous output diagnostics · not success rate'
            self.mechanics_plot.set_series([(label,[(e['round'],e['components'].get(key)) for e in rounds],False)
                for key,label in (('peak_reward_rate','Peak rate'),('fold_support_integral_s','Fold support (s)'),
                                  ('positive_output_duration_s','Output duration (s)'))])
            self.reward_plot.set_series([(label,[(e['round'],e['components'].get(key)) for e in rounds],False)
                for key,label in (('continuous_whip_credit','Accrued output'),('workspace_cost','Workspace cost'),('return_transform_adjustment','Return transform (legacy modes)'),('failure_adjustment','Failure cost / correction'),('net_episode_return','Net return'))])
            self.valid_plot.set_series([('Valid',[(e['round'],e['valid_fraction']) for e in rounds],False),
                ('Full horizon',[(e['round'],e['components'].get('completed_horizon')) for e in rounds],False)])
            self.evaluations.setHorizontalHeaderLabels(['Collection',f'Return ×{scale:g}','Accrued output',
                'Full horizon','Valid','Outside workspace (s)','Workspace cost'])
            if selected_snapshot['task']['matching'].get('reward_mode') in ('continuous_horizontal_whip_full_horizon_v4','continuous_directional_whip_full_horizon_v5', 'continuous_directional_whip_four_second_v6'):
                self.valid_plot.title='Completion and physical-envelope diagnostics'
                self.valid_plot.set_series([(label,[(e['round'],e['components'].get(key)) for e in rounds],False)
                    for key,label in (('completed_horizon','Full horizon'),('physical_violation','Physical violation'),
                                      ('numerically_valid','Numerically valid'))])
        self.loss_plot.set_series([(label,[(e['round'],e.get(key)) for e in rounds],False) for key,label in (('critic_loss','Critic'),('actor_loss','Actor'))])
        exploration=[e for e in history if e['phase']=='exploration']
        self.entropy_plot.set_series([(label,[(e['round'],e.get(key)) for e in exploration],dashed)
            for key,label,dashed in ((('entropy','Measured',False),) if method=='ppo' else
                                    (('entropy','Measured',False),('target_entropy','Target',True)))])
        self.temperature_plot.title='PPO entropy bonus coefficient' if method=='ppo' else 'SAC entropy temperature'
        self.temperature_plot.set_series([('Coefficient' if method=='ppo' else 'Temperature',
            [(e['round'],e.get('entropy_weight' if method=='ppo' else 'temperature')) for e in exploration],False)])
        self.evaluations.setRowCount(len(rows))
        for i,e in enumerate(rows):
            outside=e.get('workspace',{}).get('outside_duration_s')
            iteration=f"{e['round']} · Lane {e['lane']}" if 'lane' in e else e['round']
            charge=e['components'].get('workspace_cost')
            values=(iteration,scale*e['evaluation_return'],e['components'].get('progress' if retained_progress_mode else 'mechanical'),
                    e['components'].get('qualified_hit' if retained_progress_mode else 'strike_success'),e['valid'],
                    outside,scale*charge if charge is not None else None)
            if continuous_mode:
                values=(iteration,scale*e['evaluation_return'],e['components'].get('continuous_whip_credit'),
                        e['components'].get('completed_horizon'),e['valid'],outside,charge)
            for j,value in enumerate(values):
                text='—' if value is None else (f'{value:.4g}' if isinstance(value,float) else str(value))
                self.evaluations.setItem(i,j,QtWidgets.QTableWidgetItem(text))

    def update_overview(self,directory,history,settings):
        episodes=next((e['episodes'] for e in reversed(history) if 'episodes' in e),0)
        budget=settings['training_episodes']
        self.budget_progress.setValue(min(1000,round(1000*episodes/budget)))
        self.budget_progress.setFormat(f'Stage {training_stage(settings)} · {episodes:,} / {budget:,} training episodes · {100*episodes/budget:.1f}%')
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
            outcome=('Full horizon completed' if latest.get('valid') else 'Full horizon completed · physical violation') if c.get('completed_horizon')==1 else 'Aborted / saved prefix'
            self.latest_result.setText(
                f"Stage1 · collection {latest['round']} · {outcome}\nReturn: {number(raw)}\n"
                f"Accrued output: {number(c['continuous_whip_credit'])}\nWorkspace cost: {number(c.get('workspace_cost'))}\nFailure correction: {number(c.get('failure_adjustment'))}\n"
                f"Peak reward rate: {number(c.get('peak_reward_rate'))}\nFold support: {number(c.get('fold_support_integral_s'))} s\n"
                f"Positive output duration: {number(c.get('positive_output_duration_s'))} s\n"
                "No target or binary whip success gate. See Whipping diagnostics for recorded rate and fold support.")
            self.latest_summary.setText(f"Stage1 · collection {latest['round']} · {outcome} · return {number(raw)} · continuous score is not success")
            return
        if 'progress_credit' in c and 'outcome_credit' in c:
            valid=latest.get('valid')
            hit=c.get('qualified_hit')==1
            outcome=('VERIFIED useful hit' if hit and valid else 'INVALID attempt' if valid is False
                     else 'Safe timeout' if valid and c.get('timeout')==1
                     else 'Completed miss' if valid and c.get('event_completed')==1
                     else 'No verified outcome recorded')
            self.latest_result.setText(
                f"Stage 2 · collection {latest['round']} · {outcome}\n\n"
                f"Task return: {number(raw)}\nOptimizer return (×{scale:g}): {number(None if raw is None else scale*raw)}\n"
                f"Best coherent progress P: {number(c.get('progress'))}\n"
                f"Retained progress credit: {number(c.get('progress_credit'))}\n"
                f"Terminal outcome adjustment: {number(c.get('outcome_credit'))}\n\n"
                f"Formed / propagated / opened: {number(c.get('fold_formed'))} / {number(c.get('fold_propagated'))} / {number(c.get('fold_opened'))}\n"
                f"Verified same-opening useful hit: {number(c.get('qualified_hit'))}\n"
                f"Elapsed episode time: {number(c.get('executed_duration_s'))} s\n"
                "No elapsed-time, effort or path cost. Partial progress is not hit success.\n"
                "One saved deterministic motion; not a robustness test or scientific promotion.")
            self.latest_summary.setText(f"Stage 2 · collection {latest['round']} · return {number(raw)} · "
                f"progress {number(c.get('progress'))} · {outcome}")
            return
        hit_label='Geometric target hit' if 'geometric_hit_success' in c else 'Qualified strike'
        text=(f"Stage {training_stage(settings)} · iteration {latest['round']}\n"
              f"{'Valid numerical rollout' if latest.get('valid') else 'Numerical failure / invalid rollout'}\n\n"
              f"Task return: {number(raw)}\nOptimizer return (×{scale:g}): {number(None if raw is None else scale*raw)}\n\n"
              f"{hit_label}: {number(c.get('geometric_hit_success', c.get('strike_success')))}\nMechanics diagnostic: {number(c.get('mechanical'))}\nMatching: {number(c.get('matching'))}\n"
              f"Outside workspace: {number(ws.get('outside_duration_s'))} s\nWorkspace cost (raw): {number(c.get('workspace_cost'))}")
        if 'excitation_score' in c:
            text+=(f"\nExcitation guide: {number(c.get('excitation_score'))}"
                   f"\nSame-sample world / nonrigid tip speed: {number(c.get('world_speed_at_q'))} / {number(c.get('shape_speed_at_q'))} m/s"
                   '\nTarget crossings are diagnostic only. Positive guide is not whipping success; full motion review remains required.')
        elif 'unified_fold_score' in c:
            text+=(f"\nUnified attempt utility: {number(c.get('unified_fold_score'))}"
                   f"\nFormation / propagation: {number(c.get('formation_fraction'))} / {number(c.get('propagation_fraction'))}"
                   f"\nAligned release output: {number(c.get('aligned_release_score'))}"
                   f"\nCompleted aligned release: {number(c.get('completed_release_fraction_all'))}"
                   '\nPreparation credit is not whipping success; motion review remains required.')
        elif 'geometric_hit_success' in c and 'unfolding_fraction' not in c:
            angle=c.get('direction_error_rad')
            text+=f"\nStrict 3 m/s / 30° diagnostic: {number(c.get('strict_strike_success'))}"
            text+=f"\nSelected event forward speed: {number(c.get('forward_mps'))} m/s"
            text+=f"\nSelected event entry angle: {number(None if angle is None else angle*180/3.141592653589793)}°"
            text+='\nSelected approach measurements are not hits unless Geometric target hit = 1.'
        if 'unfolding_fraction' in c:
            text+='\nSelected mechanics event is not necessarily a target crossing.'
            text+=f"\nRecovered unfolding fraction: {number(c.get('unfolding_fraction'))}"
            text+=f"\nForward alignment factor: {number(c.get('forward_alignment'))}"
            credit_label=('Whip quality before shaping' if 'base_mechanics_credit' in c else
                'Base whip credit' if 'forward_reach_fraction' in c else 'Event credit')
            text+=f"\n{credit_label} (raw): {number(c.get('mechanics_credit'))}"
        if 'forward_reach_fraction' in c:
            text+=f"\nForward reach / cable length: {number(c.get('forward_reach_fraction'))}"
            text+=f"\nDrone path before release: {number(c.get('preparation_path_m'))} m"
            if 'level_outreach_quality' in c:
                text+=f"\nTip height relative to tracked drone: {number(c.get('tip_drone_height_error_m'))} m"
                text+=f"\nHeight quality / joint level outreach: {number(c.get('level_height_factor'))} / {number(c.get('level_outreach_quality'))}"
                if 'outreach_credit' in c:
                    text+=f"\nMechanics gate / speed factor: {number(c.get('outreach_mechanics_gate'))} / {number(c.get('outreach_speed_factor'))}"
                    text+=f"\nOutreach credit before path reduction: {number(c.get('outreach_credit'))}"
                else:
                    text+=f"\nBase mechanics / level outreach credit: {number(c.get('base_mechanics_credit'))} / {number(c.get('level_outreach_bonus'))}"
                text+=f"\nPreparation reduction: {number(c.get('preparation_cost'))}"
            else:
                text+=f"\nReach bonus / preparation reduction: {number(c.get('reach_bonus'))} / {number(c.get('preparation_cost'))}"
            text+=f"\nPreparation multiplier: {number(c.get('preparation_factor'))}"
        if 'crossing_reward' in c:
            text+=f"\nWhip support / complete crossing credit: {number(c.get('whip_support_reward'))} / {number(c.get('crossing_reward'))}"
            text+=f"\nRewarded crossing side / height error: {number(c.get('crossing_reward_lateral_error_m'))} / {number(c.get('crossing_reward_vertical_error_m'))} m"
        elif 'target_proximity' in c:
            text+=f"\nSame-event current-target proximity: {number(c.get('target_proximity'))}"
            text+=f"\nWeighted delivery credit: {number(c.get('approach_reward'))}"
        if 'center_proximity' in c:
            text+=f"\nSame-event center distance / sigma: {number(c.get('center_distance_m'))} / {number(c.get('precision_sigma_m'))} m"
            text+=f"\nCenter proximity / weighted credit: {number(c.get('center_proximity'))} / {number(c.get('center_reward'))}"
        target=next((e for e in reversed(history) if e['phase']=='target_curriculum_evaluation' and e['round']==latest['round']),None)
        if target:
            text+=f"\n\nCurrent target: {target['target_width_m']:g} × {target['target_height_m']:g} m\nCurrent-target qualified: {target.get('qualified','—')}\nAbove strike / return: strict final target."
        if latest.get('inherited'): text+='\n\nInherited evaluation from the parent checkpoint, not a new trial.'
        if not latest.get('replay'): text+='\n\nNo valid saved replay for this evaluation.'
        self.latest_result.setText(text)
        self.latest_summary.setText(f"Stage {training_stage(settings)} · collection {latest['round']} · task return {number(raw)} · "
            + ('valid saved evaluation' if latest.get('valid') else 'INVALID evaluation / no successful result substituted')
            + (' · inherited from parent' if latest.get('inherited') else ''))

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
            raise ValueError('Exact resume settings may change ONLY training_episodes; reward/source changes require actor-only transfer')
        settings['training_episodes']=budget
    window.settings=settings;window.task=deepcopy(saved['snapshot']['task'])
    window.mode.setCurrentIndex(1)
    window.resume_path=checkpoint
    window.sync_budget_control(settings)
    window.update_description()
    window.checkpoint_label.setText(str(checkpoint))


def main(*, auto_start=False, settings_path=None, resume_checkpoint=None, initialize_actor=None):
    if resume_checkpoint and initialize_actor:
        raise ValueError('Select exact resume OR actor-only initialization')
    app=QtWidgets.QApplication.instance() or QtWidgets.QApplication(sys.argv)
    requested_settings=load(settings_path)[0] if settings_path is not None else None
    window=TrainingWindow(settings_path or CURRENT_PRESETS[1]); window.show()
    if resume_checkpoint is not None:
        import torch
        checkpoint=project_path(resume_checkpoint)
        saved=torch.load(checkpoint,map_location='cpu',weights_only=True)
        configure_resume(window,checkpoint,saved,requested_settings)
    if initialize_actor is not None:
        window.mode.setCurrentIndex(2)
        window.initialize_actor_path=project_path(initialize_actor)
        window.checkpoint_label.setText('Actor only: '+str(window.initialize_actor_path))
    if auto_start:
        # Enter the GUI event loop first. Never launch a detached headless worker.
        QtCore.QTimer.singleShot(0, lambda: window.start() if window.isVisible() else None)
    return app.exec()
