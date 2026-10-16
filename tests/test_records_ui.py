"""Evaluation provenance and essential UI behavior, without live workers."""
from copy import deepcopy
import json
import os
from pathlib import Path
import unittest
from unittest.mock import patch, MagicMock
import torch
from helpers import project_temporary

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
from PySide6 import QtWidgets
from training.configuration import FROZEN_PRESETS, load
from training.artifacts import save_json, read_json
from training.ui.window import TrainingWindow, SettingsDialog
from training.ui.preview import EvaluationPreview
from evaluation import records


class ProvenanceTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.base = load(FROZEN_PRESETS[1])[1]
        cls.precision = load(FROZEN_PRESETS[2])[1]
        cls.hashes = {"simulation/model.py": "physical", "training/ppo.py": "old"}

    def setUp(self):
        self.temporary = project_temporary()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)

    def checkpoint(self, stage=1, name='latest.pt', directory=None):
        directory = directory or self.root
        directory.mkdir(parents=True, exist_ok=True)
        snapshot = deepcopy(self.base if stage == 1 else self.precision if stage == 2
            else load(FROZEN_PRESETS[3])[1])
        snapshot['settings']['device'] = 'cpu'
        snapshot['execution_hashes'] = deepcopy(self.hashes)
        snapshot['identity'] = records.checkpoint_identity(snapshot)
        save_json(directory/'configuration.json', snapshot)
        (directory/'scene.toml').write_text(snapshot['scene_text'], encoding='utf-8')
        saved = dict(schema='ppo_stage1_checkpoint_v1', snapshot=snapshot,
            identity=snapshot['identity'], round=3, agent={'actor':{'encoder.columns':torch.tensor([0])}})
        if stage == 2:
            saved['target_curriculum'] = dict(level=1, streak=0)
        torch.save(saved, directory/name)
        return directory/name, snapshot

    def test_checkpoint_identity_and_replay_path_guard_remain_strict(self):
        path, snapshot = self.checkpoint()
        saved = torch.load(path, weights_only=True)
        saved['snapshot']['task']['request'][2] += 1
        torch.save(saved, path)
        with self.assertRaisesRegex(ValueError, 'identity'):
            records.load_checkpoint(path)
        with self.assertRaises(ValueError):
            records.child(self.root, '../outside.npz')

    def test_exact_mode_still_rejects_every_changed_source(self):
        _, snapshot = self.checkpoint()
        with patch.object(records, 'execution_hashes', return_value={**self.hashes, 'training/ppo.py':'new'}):
            with self.assertRaisesRegex(ValueError, 'saved implementation'):
                records.evaluation_contract(snapshot, 'exact_source')

    def test_current_mode_pins_both_sources_without_snapshot_migration(self):
        path, snapshot = self.checkpoint(2)
        before = path.read_bytes()
        current = {**self.hashes, 'training/ppo.py':'new'}
        with patch.object(records, 'execution_hashes', return_value=current), patch.object(records, 'active_workers', return_value=[]):
            result = records.prepare(path, root=self.root/'outputs', mode='current_code')
            request = read_json(result/'request.json')
            self.assertEqual(read_json(result/'configuration.json'), snapshot)
            self.assertEqual(request['changed_files'], ['training/ppo.py'])
            self.assertEqual(request['execution_hashes'], current)
            self.assertEqual(request['source_execution_hashes'], self.hashes)
            self.assertFalse(request['exact_source_reproduction'])
            self.assertEqual(records.verify_contract(snapshot, request)['source_mode'], 'current_code')
        self.assertEqual(path.read_bytes(), before)
        with patch.object(records, 'execution_hashes', return_value={**current, 'training/ppo.py':'later'}):
            with self.assertRaisesRegex(ValueError, 'changed after confirmation'):
                records.verify_contract(snapshot, request)
        with patch.object(records, 'execution_hashes', return_value=current), patch.object(records, 'evaluator_hashes', return_value={}):
            with self.assertRaisesRegex(ValueError, 'evaluator_hashes'):
                records.verify_contract(snapshot, request)

    def test_current_mode_does_not_ignore_physics_removed_or_added_paths(self):
        _, snapshot = self.checkpoint()
        for current in ({**self.hashes, 'simulation/model.py':'new'},
                        {'training/ppo.py':'old'}, {**self.hashes, 'native/new.cpp':'new'}):
            with patch.object(records, 'execution_hashes', return_value=current):
                with self.assertRaisesRegex(ValueError, 'unchanged physical'):
                    records.evaluation_contract(snapshot, 'current_code')

    def test_busy_evaluation_blocks_before_output_creation(self):
        path, _ = self.checkpoint()
        with patch.object(records, 'execution_hashes', return_value=self.hashes), patch.object(
                records, 'active_workers', return_value=[dict(ProcessId=123, CommandLine='python -m evaluation.worker')]):
            with self.assertRaisesRegex(ValueError, 'idle'):
                records.prepare(path, root=self.root/'outputs')
        self.assertFalse((self.root/'outputs').exists())

    def test_policy_test_worker_also_blocks_evaluation(self):
        with patch.object(records, 'active_workers', return_value=[
                dict(ProcessId=123, CommandLine='python -m evaluation.policy_test --run runs/test')]):
            with self.assertRaisesRegex(ValueError, 'idle'):
                records.require_evaluation_idle()


class InterfaceTests(unittest.TestCase):
    def setUp(self):
        self.app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
        self.temp = project_temporary()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        _, snapshot, _, _ = load(FROZEN_PRESETS[1])
        self.live = self.root/'stage1'/'new'
        self.old = self.root/'stage2'/'old'
        for directory in (self.live, self.old):
            directory.mkdir(parents=True)
            save_json(directory/'configuration.json', snapshot)
        self.patches = [patch('training.ui.window.'+name, value) for name, value in (
            ('PPO_RUNS', self.root/'stage1'), ('PPO_STAGE2_RUNS', self.root/'stage2'),
            ('PPO_STAGE3_RUNS', self.root/'stage3'), ('RUNS', self.root/'sac'))]
        for p in self.patches:
            p.start()
            self.addCleanup(p.stop)
        self.window = TrainingWindow()
        self.window.timer.stop()
        self.addCleanup(self.close_window)

    def close_window(self):
        self.window.process = None
        self.window.close()

    def test_refresh_selects_path_by_value_and_keeps_it(self):
        self.window.refresh_runs(Path(str(self.live)))
        self.assertEqual(self.window.runs.currentData(), self.live)
        self.window.refresh_runs()
        self.assertEqual(self.window.runs.currentData(), self.live)

    def test_stop_request_status_is_not_overwritten_by_collection(self):
        self.window.current_run = self.live
        self.window.process = object()
        save_json(self.live/'progress.json', dict(phase='collecting'))
        (self.live/'stop.request').touch()
        self.window.poll()
        self.assertEqual(self.window.status.text(), 'Stopping safely')
        save_json(self.live/'progress.json', dict(phase='stopped'))
        self.window.poll()
        self.assertEqual(self.window.status.text(), 'Stopped')

    def test_settings_no_edit_roundtrip_keeps_every_value(self):
        for path in (*FROZEN_PRESETS.values(), 'configs/learning/whip_stage1_ppo_level_outreach_v6.json'):
            settings, snapshot, _, _ = load(path)
            dialog=SettingsDialog(settings,snapshot['task'])
            self.addCleanup(dialog.close)
            with patch.object(QtWidgets.QMessageBox,'warning') as warning:
                dialog.accept()
                warning.assert_not_called()
            self.assertEqual(dialog.settings,settings)
            self.assertEqual(dialog.task,snapshot['task'])
            self.assertEqual(dialog.result(),QtWidgets.QDialog.DialogCode.Accepted)

    def test_settings_save_rejects_outside_project_without_writing(self):
        from project_paths import ROOT
        destination = ROOT.parent/'outside-training-settings.json'
        with patch.object(QtWidgets.QFileDialog, 'getSaveFileName', return_value=(str(destination), '')), \
             patch.object(QtWidgets.QMessageBox, 'warning') as warning, \
             patch('training.ui.window.save_json') as write:
            self.window.save_settings()
            write.assert_not_called()
            warning.assert_called_once()

    def test_active_or_resume_configuration_can_be_inspected_but_not_retuned(self):
        dialog=SettingsDialog(self.window.settings,self.window.task,read_only=True)
        self.addCleanup(dialog.close)
        self.assertTrue(dialog.controls['parallel_environments'].isReadOnly())
        self.assertFalse(dialog.controls['device'].isEnabled())
        self.window.process=object()
        with patch('training.ui.window.SettingsDialog') as make_dialog:
            self.window.edit_settings()
            self.assertTrue(make_dialog.call_args.kwargs['read_only'])
        self.window.process=None
        self.window.mode.setCurrentIndex(1)
        with patch('training.ui.window.SettingsDialog') as make_dialog:
            self.window.edit_settings()
            self.assertTrue(make_dialog.call_args.kwargs['read_only'])

    def test_latest_evaluation_does_not_substitute_best_or_old_valid_result(self):
        a=dict(phase='evaluation',round=1,evaluation_return=.5,valid=True,replay='one.npz',components={})
        b=dict(a,round=2,evaluation_return=.1,replay='two.npz')
        with patch.object(self.window.preview,'set_result') as result:
            self.window.update_overview(self.live,[a,b],self.window.settings)
            self.assertEqual(result.call_args.args[1],b)
            self.window.update_overview(self.live,[a,b,dict(b,round=3,valid=False,replay=None)],self.window.settings)
            self.assertIsNone(result.call_args.args[1])
            self.assertIn('Numerical failure',self.window.latest_result.text())

    def test_preview_is_lazy_single_load_per_replay_and_disposes_on_switch(self):
        preview=EvaluationPreview();self.addCleanup(preview.close)
        event=dict(round=4,replay='round-00004.npz')
        def fake_replay(*args):
            from PySide6 import QtCore
            widget=QtWidgets.QWidget(preview)
            widget.telemetry=QtWidgets.QLabel();widget.result=QtWidgets.QLabel()
            widget.timer=QtCore.QTimer(widget)
            widget.toggle=MagicMock();widget.pause=MagicMock()
            return widget
        with patch.object(preview,'_create_replay',side_effect=fake_replay) as create:
            preview.set_result(self.live,event);create.assert_not_called()
            preview.show();self.app.processEvents()
            create.assert_called_once()
            self.assertEqual(preview.replay_window.timer.interval(),50)
            preview.set_result(self.live,event);create.assert_called_once()
            previous=preview.replay_window
            preview.hide();self.assertFalse(previous.timer.isActive())
            previous.pause.assert_called_once()
            preview.set_result(self.old,None);preview.show();self.app.processEvents()
            self.assertIsNone(preview.replay_window)
            self.assertTrue(preview.placeholder.isVisible())

    def test_preview_render_error_is_explicit_and_does_not_retry_each_poll(self):
        preview=EvaluationPreview();self.addCleanup(preview.close)
        event=dict(round=1,replay='one.npz')
        with patch.object(preview,'_create_replay',side_effect=RuntimeError('renderer unavailable')) as create:
            preview.show();preview.set_result(self.live,event);self.app.processEvents()
            self.assertIn('renderer unavailable',preview.placeholder.text())
            preview.set_result(self.live,event);create.assert_called_once()
