"""Editable single-case policy tester. No trainer, optimizer or model promotion."""
import argparse
from pathlib import Path
import sys

import numpy as np
from PySide6 import QtCore, QtWidgets

from evaluation.policy_test import (prepare, checkpoint_context, initial_case, case_warnings,
                                    rpy_quaternion, PROTOCOL)
from evaluation.records import discover_stages, task_stage
from project_paths import ROOT
from simulation.state import quaternion_xyzw_to_rotation_matrix
from training.artifacts import read_json
from training.ui.preview import EvaluationPreview


class PolicyTesterWindow(QtWidgets.QMainWindow):
    def __init__(self, checkpoint=None):
        super().__init__()
        self.setWindowTitle('AeroWhip · Policy simulation tester')
        self.resize(1480,960)
        self.setStyleSheet('QWidget#testerControls {background:#edf2f6;} '
            'QWidget#testerControls QLabel {color:#213547;} '
            'QWidget#testerControls QDoubleSpinBox, QWidget#testerControls QLineEdit, '
            'QWidget#testerControls QComboBox {background:white;color:#213547;padding:4px;} '
            'QWidget#testerControls QPushButton {background:white;color:#213547;padding:7px;border:1px solid #ccd8e2;}')
        self.info=self.scene=self.cfg=self.directory=None
        self.closing=False
        self.process=QtCore.QProcess(self)
        self.process.setWorkingDirectory(str(ROOT))
        self.process.finished.connect(self.finished)
        self.process.errorOccurred.connect(self.process_error)
        self.timer=QtCore.QTimer(self); self.timer.setInterval(500); self.timer.timeout.connect(self.poll)
        central=QtWidgets.QWidget(); self.setCentralWidget(central)
        outer=QtWidgets.QHBoxLayout(central)
        controls=QtWidgets.QWidget(objectName='testerControls')
        panel=QtWidgets.QVBoxLayout(controls)
        panel.addWidget(QtWidgets.QLabel('<h2>Test a learned policy</h2>'))
        self.stage=QtWidgets.QComboBox(); self.stage.addItem('All stages',None)
        for i in (1,2,3,4): self.stage.addItem(f'Stage {i}',i)
        self.runs=QtWidgets.QComboBox()
        self.runs.setSizeAdjustPolicy(QtWidgets.QComboBox.SizeAdjustPolicy.AdjustToMinimumContentsLengthWithIcon)
        self.runs.setMinimumContentsLength(10)
        self.runs.setSizePolicy(QtWidgets.QSizePolicy.Policy.Ignored,QtWidgets.QSizePolicy.Policy.Fixed)
        self.runs.setPlaceholderText('Custom checkpoint')
        self.stage.currentIndexChanged.connect(self.refresh_runs)
        self.runs.currentIndexChanged.connect(self.select_run)
        row=QtWidgets.QHBoxLayout(); row.addWidget(self.stage); row.addWidget(self.runs,1)
        refresh=QtWidgets.QPushButton('Refresh'); refresh.clicked.connect(self.refresh_runs); row.addWidget(refresh)
        panel.addLayout(row)
        self.checkpoint=QtWidgets.QLineEdit(); self.checkpoint.setReadOnly(True)
        browse=QtWidgets.QPushButton('Choose checkpoint…'); browse.clicked.connect(self.browse)
        panel.addWidget(self.checkpoint); panel.addWidget(browse)
        self.source=QtWidgets.QLabel(); self.source.setWordWrap(True); panel.addWidget(self.source)
        self.inputs=QtWidgets.QWidget(); form=QtWidgets.QFormLayout(self.inputs)
        self.values={}
        def triple(key,title):
            row=QtWidgets.QWidget(); layout=QtWidgets.QHBoxLayout(row); layout.setContentsMargins(0,0,0,0)
            self.values[key]=[]
            for axis in ('Roll','Pitch','Yaw') if key=='attitude' else 'XYZ':
                column=QtWidgets.QVBoxLayout(); column.addWidget(QtWidgets.QLabel(axis))
                spin=QtWidgets.QDoubleSpinBox(); spin.setDecimals(5); spin.setRange(-1000,1000)
                spin.setMinimumWidth(0)
                spin.setSizePolicy(QtWidgets.QSizePolicy.Policy.Ignored,QtWidgets.QSizePolicy.Policy.Fixed)
                spin.setSingleStep(.05 if key!='attitude' else 5)
                spin.valueChanged.connect(self.update_warnings)
                column.addWidget(spin); layout.addLayout(column); self.values[key].append(spin)
            form.addRow(QtWidgets.QLabel(title)); form.addRow(row)
        triple('target','Target world position [m]')
        triple('initial','Drone initial position [m]')
        triple('attitude','Initial body→world [deg]')
        triple('velocity','Initial world velocity [m/s]')
        triple('omega','Initial body rate [rad/s]')
        panel.addWidget(self.inputs)
        self.normal=QtWidgets.QLabel(); self.normal.setWordWrap(True); panel.addWidget(self.normal)
        self.warning=QtWidgets.QLabel(); self.warning.setWordWrap(True); panel.addWidget(self.warning)
        self.test=QtWidgets.QPushButton('Run simulation test'); self.test.clicked.connect(self.start)
        self.stop=QtWidgets.QPushButton('Stop this test'); self.stop.clicked.connect(self.request_stop); self.stop.setEnabled(False)
        row=QtWidgets.QHBoxLayout(); row.addWidget(self.test); row.addWidget(self.stop); panel.addLayout(row)
        self.status=QtWidgets.QLabel('Choose a checkpoint. Nothing runs until you press Run.'); self.status.setWordWrap(True)
        panel.addWidget(self.status)
        saved=QtWidgets.QPushButton('Open saved test…'); saved.clicked.connect(self.open_saved); panel.addWidget(saved)
        self.result_path=QtWidgets.QLabel(); self.result_path.setWordWrap(True); panel.addWidget(self.result_path)
        for widget in (self.source,self.normal,self.warning,self.status,self.result_path):
            widget.setSizePolicy(QtWidgets.QSizePolicy.Policy.Ignored,QtWidgets.QSizePolicy.Policy.Preferred)
        panel.addStretch()
        scroll=QtWidgets.QScrollArea(); scroll.setWidgetResizable(True); scroll.setWidget(controls)
        scroll.setObjectName('testerScroll')
        scroll.setMinimumWidth(370); scroll.setMaximumWidth(440)
        outer.addWidget(scroll)
        self.preview=EvaluationPreview(); outer.addWidget(self.preview,1)
        self.preview.caption.setText('Test result')
        self.preview.placeholder.setText('Run a case to see its recorded motion, strike speed, direction and target error.')
        self.preview.note.setText('Stationary target · frozen checkpoint · native simulation · no learning or hardware')
        self.selectors=[self.stage,self.runs,refresh,browse,saved]
        self.refresh_runs()
        if checkpoint: self.load_source(Path(checkpoint))

    def refresh_runs(self):
        self.runs.blockSignals(True); self.runs.clear()
        paths,_=discover_stages(self.stage.currentData())
        for path in paths:
            if (path/'latest.pt').exists():
                self.runs.addItem(f'S{task_stage(read_json(path/"configuration.json"))} · {path.name}',str(path/'latest.pt'))
        self.runs.blockSignals(False)
        self.select_run()

    def select_run(self):
        if self.runs.currentData(): self.load_source(Path(self.runs.currentData()))

    def browse(self):
        filename,_=QtWidgets.QFileDialog.getOpenFileName(self,'Learned policy checkpoint',str(ROOT/'runs'),'Checkpoint (*.pt)')
        if filename: self.load_source(Path(filename))

    def load_source(self,path):
        try:
            info,scene,cfg=checkpoint_context(path)
            self.info,self.scene,self.cfg=info,scene,cfg
            self.checkpoint.setText(str(path.resolve()))
            self.runs.blockSignals(True)
            self.runs.setCurrentIndex(self.runs.findData(str(path.resolve())))
            self.runs.blockSignals(False)
            conditioned=info['snapshot']['settings'].get('target_observation')=='initial_relative_position_v1'
            self.source.setText(f'Stage {task_stage(info["snapshot"])} · checkpoint iteration {info["round"]}\n'
                                +('Target-conditioned policy' if conditioned else 'Historical fixed-target policy (no goal input)'))
            self.set_case_inputs(initial_case(scene,cfg))
            normal=cfg.strike_surface.description(scene.initial_state.position_world_m,cfg.request)['normal_world']
            self.normal.setText('Strike-plane normal stays fixed in world axes: '+str(normal)+'\n'
                                'Changing yaw does not rotate the target. Velocity-action controller is unchanged.')
            self.update_warnings()
        except Exception as exc:
            self.info=None; self.test.setEnabled(False); self.status.setText(str(exc))

    def set_case_inputs(self,case):
        rotation=quaternion_xyzw_to_rotation_matrix(np.asarray(case['initial_quaternion_body_to_world_xyzw']))
        pitch=np.arcsin(np.clip(-rotation[2,0],-1,1))
        rpy=np.degrees([np.arctan2(rotation[2,1],rotation[2,2]),pitch,np.arctan2(rotation[1,0],rotation[0,0])])
        values={'target':case['target_position_world_m'],'initial':case['initial_position_world_m'],
                'attitude':rpy,'velocity':case['initial_linear_velocity_world_mps'],
                'omega':case['initial_angular_velocity_body_radps']}
        for key,vector in values.items():
            for spin,v in zip(self.values[key],vector):
                spin.blockSignals(True); spin.setValue(float(v)); spin.blockSignals(False)

    def case(self):
        v={key:[spin.value() for spin in spins] for key,spins in self.values.items()}
        return dict(target_position_world_m=v['target'],initial_position_world_m=v['initial'],
                    initial_quaternion_body_to_world_xyzw=rpy_quaternion(v['attitude']).tolist(),
                    initial_linear_velocity_world_mps=v['velocity'],initial_angular_velocity_body_radps=v['omega'])

    def update_warnings(self):
        if self.info is None: return
        try:
            warnings=case_warnings(self.info['snapshot'],self.scene,self.cfg,self.case())
            self.warning.setText('\n\n'.join(warnings))
            self.test.setEnabled(self.process.state()==QtCore.QProcess.ProcessState.NotRunning)
        except Exception as exc:
            self.warning.setText(str(exc)); self.test.setEnabled(False)

    def busy(self,active):
        for widget in self.selectors+[self.inputs]: widget.setEnabled(not active)
        self.test.setEnabled(not active and self.info is not None); self.stop.setEnabled(active)

    def clear_preview(self,message):
        self.preview.dispose();self.preview.pending=None;self.preview.loaded_key=None
        self.preview.placeholder.setText(message);self.preview.placeholder.show()
        self.preview.caption.setText('Policy simulation test');self.preview.caption.show();self.preview.note.show()

    def start(self):
        if self.process.state()!=QtCore.QProcess.ProcessState.NotRunning: return
        try:
            self.directory=prepare(Path(self.checkpoint.text()),self.case())
            self.clear_preview('Simulating this case. Its recorded motion will appear after completion.')
            self.result_path.setText('Results: '+str(self.directory))
            self.process.setStandardOutputFile(str(self.directory/'stdout.log'))
            self.process.setStandardErrorFile(str(self.directory/'stderr.log'))
            self.process.start(sys.executable,['-m','evaluation.policy_test','--run',str(self.directory)])
            self.busy(True); self.status.setText('Initializing frozen actor and native simulator…'); self.timer.start()
        except Exception as exc: self.status.setText(str(exc))

    def request_stop(self):
        if self.directory is not None and self.process.state()!=QtCore.QProcess.ProcessState.NotRunning:
            (self.directory/'stop.request').touch(exist_ok=True)
            self.stop.setEnabled(False); self.status.setText('Stop requested; incomplete episodes will not be scored.')

    def poll(self):
        if self.directory is None: return
        from training.artifacts import events
        rows=events(self.directory)
        if rows:
            e=rows[-1]
            if e.get('phase')=='evaluating':
                self.status.setText(f'Simulating {e["simulation_s"]:.2f} / {e["duration_s"]:.2f} s')

    def show_summary(self,directory):
        summary=read_json(directory/'summary.json')
        self.result_path.setText('Results: '+str(directory))
        if summary.get('case'):
            self.set_case_inputs(summary['case']); self.update_warnings()
        if summary['phase']=='completed':
            c=summary.get('components',{})
            hit=bool(summary.get('valid') and c.get('geometric_hit_success',c.get('strike_success',0))==1)
            error=c.get('first_crossing_center_error_m')
            error_text='No valid crossing' if error is None else f'First crossing error {100*error:.2f} cm'
            self.status.setText(f'Valid: {summary["valid"]} · Target hit: {hit}\n{error_text}')
            if summary.get('replay'): self.preview.set_result(directory,summary)
            else: self.clear_preview('No valid numerical replay prefix; see the retained validity/failure evidence.')
        else:
            self.status.setText(summary.get('error',summary.get('message',summary['phase'])))
            self.clear_preview(self.status.text())

    def finished(self,code,status):
        self.timer.stop(); self.busy(False)
        if self.directory is not None and (self.directory/'summary.json').exists():
            self.show_summary(self.directory)
        elif code!=0: self.status.setText('Worker failed; see retained stderr.log. No retry was started.')
        if self.closing: self.close()

    def process_error(self,error):
        if error==QtCore.QProcess.ProcessError.FailedToStart:
            self.timer.stop(); self.busy(False); self.status.setText('Worker could not start: '+self.process.errorString())

    def open_saved(self):
        filename,_=QtWidgets.QFileDialog.getOpenFileName(self,'Saved policy test',str(ROOT/'runs/policy_tests'),'Summary (summary.json)')
        if not filename: return
        try:
            directory=Path(filename).parent
            request=read_json(directory/'request.json')
            if request['protocol']!=PROTOCOL: raise ValueError('Select a policy-test result')
            self.load_source(Path(request['checkpoint']))
            if self.info is not None and self.info['sha256']!=request['checkpoint_sha256']:
                self.info=None;self.test.setEnabled(False)
                self.source.setText('Recorded checkpoint differs from the current file. Saved motion remains viewable; select a source for new tests.')
            self.show_summary(directory)
        except Exception as exc: self.status.setText(str(exc))

    def closeEvent(self,event):
        if self.process.state()!=QtCore.QProcess.ProcessState.NotRunning:
            answer=QtWidgets.QMessageBox.question(self,'Test is active','Stop this test safely and close after the worker exits?',
                QtWidgets.QMessageBox.StandardButton.Yes|QtWidgets.QMessageBox.StandardButton.No,
                QtWidgets.QMessageBox.StandardButton.No)
            if answer==QtWidgets.QMessageBox.StandardButton.Yes:
                self.closing=True; self.request_stop()
            event.ignore(); return
        self.preview.dispose(); super().closeEvent(event)


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint',type=Path)
    args=parser.parse_args(argv)
    app=QtWidgets.QApplication.instance() or QtWidgets.QApplication(sys.argv)
    window=PolicyTesterWindow(args.checkpoint); window.show()
    return app.exec()
