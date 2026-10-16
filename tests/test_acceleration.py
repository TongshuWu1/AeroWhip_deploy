"""Direct acceleration semantics and compatibility; not learned-flight evidence."""
from copy import deepcopy
from dataclasses import replace
from pathlib import Path
import unittest
from unittest.mock import patch

import numpy as np
import torch

from helpers import project_temporary
from controllers.full_state_controller.tensor_controller import GeometricController
from evaluation.records import load_checkpoint, discover
from evaluation.recovery import StrikeReturnEnvironment
from simulation.tensor_model import ControlledState, controlled_values
from tasks.whip.deterministic_evaluation import DeterministicEvaluationEnvironment
from tasks.whip.sac_environment import Stage1Environment
from training.artifacts import (identity, validate_actor_source, create_run,
                                checkpoint_replay, replay_from_checkpoint)
from training.configuration import DEFAULT, CURRENT_PRESETS, FROZEN_PRESETS, load, validate
from training.demonstration import load_demonstration
from training.ppo import PPO
from viewer.replay import StateReplay


class AccelerationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)
        cls.settings, cls.snapshot, cls.cfg, cls.scene = load(DEFAULT)

    def env(self, native=False):
        settings = dict(self.settings, device='cpu', parallel_environments=1, evaluation_environments=1)
        if native:
            return DeterministicEvaluationEnvironment(settings,self.snapshot,self.cfg,self.scene)
        return Stage1Environment(settings,self.snapshot,self.cfg,self.scene,capture=False)

    def agent(self, velocity=False):
        settings = load(FROZEN_PRESETS[1])[0] if velocity else self.settings
        e = self.env()
        return PPO(dict(settings,device='cpu'),e.actor_columns,e.frame_size,'cpu',
                   privileged_size=e.critic_state_size,encoder_options=e.policy_encoder_options)

    def checkpoint(self):
        snap=deepcopy(self.snapshot)
        snap['execution_hashes']={}
        snap['identity']=identity(snap)
        return dict(schema='ppo_stage1_checkpoint_v1',snapshot=snap,identity=snap['identity'],
                    agent=self.agent().state_dict(),round=0,episodes=0,steps=0)

    def test_new_default_preserves_task_and_legacy_presets(self):
        old,snap,cfg,_=load(FROZEN_PRESETS[1])
        self.assertEqual(self.snapshot['task'],snap['task'])
        self.assertEqual(self.snapshot['model'],snap['model'])
        differences={k for k in old.keys()|self.settings.keys() if old.get(k)!=self.settings.get(k)}
        self.assertEqual(differences,{'action_interface','policy_rate_hz','velocity_limit_mps','acceleration_limit_mps2'})
        self.assertEqual(self.settings['acceleration_limit_mps2'],[8,8,4])
        self.assertEqual(self.settings['policy_rate_hz'],50)
        self.assertEqual(old['action_interface'],'velocity_v1')

    def test_acceleration_stage_handoffs_keep_action_units_and_rewards(self):
        previous=self.agent()
        for stage in (2,3):
            settings,snapshot,cfg,scene=load(CURRENT_PRESETS[stage])
            old,old_snapshot,_,_=load(FROZEN_PRESETS[stage])
            self.assertEqual(snapshot['task'],old_snapshot['task'])
            self.assertEqual(settings.get('target_curriculum'),old.get('target_curriculum'))
            env=Stage1Environment(dict(settings,device='cpu',parallel_environments=1),snapshot,cfg,scene,capture=False)
            current=PPO(dict(settings,device='cpu'),env.actor_columns,env.frame_size,'cpu',
                        privileged_size=env.critic_state_size,encoder_options=env.policy_encoder_options)
            current.initialize_actor(previous.state_dict())
            for key,value in previous.actor.state_dict().items():
                torch.testing.assert_close(current.actor.state_dict()[key],value,rtol=0,atol=0)
            self.assertFalse(current.actor_optimizer.state)
            self.assertFalse(current.value_optimizer.state)
            previous=current

    def test_training_ui_exposes_acceleration_units_and_stage_selection(self):
        import os
        os.environ.setdefault('QT_QPA_PLATFORM','offscreen')
        from PySide6 import QtWidgets
        from training.ui.window import TrainingWindow, SettingsDialog
        app=QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
        window=TrainingWindow()
        try:
            self.assertIn('50 Hz acceleration',window.config_label.text())
            self.assertIn('m/s²',window.description_details)
            dialog=SettingsDialog(window.settings,window.task,window)
            self.assertIn('acceleration_limit_mps2',dialog.controls)
            dialog.close()
            for stage in (2,3):
                window.select_stage_preset(window.stage_presets.findData(stage))
                self.assertEqual(window.settings['action_interface'],'acceleration_direct_v1')
        finally:
            window.close();app.processEvents()

    def test_acceleration_has_no_position_or_velocity_feedback(self):
        e=self.env(); m=e.model; c=GeometricController(m)
        state=ControlledState(*(v[0] for v in controlled_values(e.state)))
        zero=m.tensor([0,0,0,0]); thrust,torque=c.acceleration(state,zero)
        self.assertAlmostEqual(float(thrust),float(-(m.mass+m.node_mass.sum())*m.gravity[2]),places=12)
        torch.testing.assert_close(torque,torch.zeros_like(torque),rtol=0,atol=1e-14)
        moved=replace(state,position_com=state.position_com+m.tensor([.1,.2,.3]),
                      velocity_com=state.velocity_com+m.tensor([1,2,3]))
        for a,b in zip(c.acceleration(state,zero),c.acceleration(moved,zero)):
            torch.testing.assert_close(a,b,rtol=0,atol=0)
        # Desired tilted frame makes collective thrust equal the requested force
        # norm and eliminates attitude error without bypassing the controller.
        command=m.tensor([3,-2,1,.4])
        force=m.mass*command[:3]-(m.mass+m.node_mass.sum())*m.gravity
        aligned=replace(state,rotation=c.desired_rotation(force,command[3]))
        thrust,torque=c.acceleration(aligned,command)
        torch.testing.assert_close(thrust,force.norm(),rtol=0,atol=1e-13)
        torch.testing.assert_close(torque,torch.zeros_like(torque),rtol=0,atol=1e-13)

    def test_action_scaling_delay_and_replay_units(self):
        e=self.env(native=True)
        action=torch.tensor([[.25,-.125,.125]])
        e.step(action);e.step(torch.zeros_like(action))
        expected=torch.tensor([2,-1,.5],dtype=torch.float64)
        torch.testing.assert_close(e.history['command'][0,1,:3],expected,rtol=0,atol=0)
        torch.testing.assert_close(e.history['applied_command'][0,1:4,:3],torch.zeros(3,3,dtype=torch.float64),rtol=0,atol=0)
        torch.testing.assert_close(e.history['applied_command'][0,4,:3],expected,rtol=0,atol=0)
        self.assertEqual(e.index,4)
        self.assertEqual(e.stride,2)
        self.assertEqual(e.queue_scale[:3].tolist(),[8,8,4])
        arrays=e.replay_arrays();StateReplay(arrays)
        self.assertEqual(str(arrays['action_interface']),'acceleration_direct_v1')
        self.assertEqual(int(arrays['policy_rate_hz']),50)
        np.testing.assert_array_equal(arrays['acceleration_limit_mps2'],[8,8,4])

    def test_native_tensor_agree_with_motor_and_history_state(self):
        native,tensor=self.env(native=True),self.env()
        for action in ([.25,-.125,.125],[.25,-.125,.125],[-.25,.125,-.125],[0,0,0]):
            for e in (native,tensor):e.step(torch.tensor([action]))
            for a,b in zip(controlled_values(native.state),controlled_values(tensor.state)):
                torch.testing.assert_close(a,b,rtol=0,atol=1e-9)

    def test_invalid_limits_and_velocity_recovery_are_rejected(self):
        for changes in ({'velocity_limit_mps':[2,2,.8]}, {'acceleration_limit_mps2':[8,8,float('nan')]},
                        {'acceleration_limit_mps2':[True,8,4]}, {'acceleration_limit_mps2':[8,8,20]},
                        {'policy_rate_hz':60}):
            with self.subTest(changes=changes),self.assertRaises(ValueError):load(dict(self.settings,**changes))
        s=load(FROZEN_PRESETS[4])[0];s.pop('velocity_limit_mps')
        s.update(action_interface='acceleration_direct_v1',acceleration_limit_mps2=[8,8,4])
        with self.assertRaisesRegex(ValueError,'recovery'):validate(s)
        with self.assertRaisesRegex(ValueError,'velocity'):StrikeReturnEnvironment(self.settings,self.snapshot,self.cfg,self.scene)
        with self.assertRaisesRegex(ValueError,'velocity'):load_demonstration('unused',self.env())

    def test_actor_and_resume_cannot_reinterpret_velocity_outputs(self):
        acc,vel=self.agent(),self.agent(velocity=True)
        for target,source in ((acc,vel),(vel,acc)):
            with self.assertRaisesRegex(ValueError,'interface'):target.initialize_actor(source.state_dict())
            with self.assertRaisesRegex(ValueError,'interface'):target.load_state_dict(source.state_dict())
        saved=self.checkpoint()
        with patch('training.artifacts.execution_hashes',return_value={}):
            validate_actor_source(saved,self.snapshot)
            with self.assertRaisesRegex(ValueError,'interface'):
                validate_actor_source(saved,load(FROZEN_PRESETS[1])[1])
        with project_temporary() as d:
            p=Path(d)/'checkpoint.pt';torch.save(saved,p)
            with self.assertRaisesRegex(ValueError,'interface'):
                create_run(load(FROZEN_PRESETS[1])[0],resume=p,root=Path(d)/'new')
            self.assertFalse((Path(d)/'new').exists())

    def test_acceleration_checkpoint_roundtrip_and_tamper_guard(self):
        saved=self.checkpoint()
        with project_temporary() as d:
            p=Path(d)/'checkpoint.pt';torch.save(saved,p)
            info=load_checkpoint(p)
            self.assertEqual(info['snapshot']['schema'],'ppo_stage1_acceleration_direct_v1')
            import json
            (Path(d)/'configuration.json').write_text(json.dumps(saved['snapshot']))
            self.assertEqual(discover(d)[0],[Path(d)])
            restored=self.agent();restored.load_state_dict(saved['agent'])
            saved['agent']['interface']['acceleration_limit_mps2']=[1,1,1]
            torch.save(saved,p)
            with self.assertRaisesRegex(ValueError,'control interface'):load_checkpoint(p)

    def test_best_replay_checkpoint_roundtrip_with_text_and_legacy_arrays(self):
        env=self.env(native=True)
        env.step(torch.zeros(1,3))
        arrays=env.replay_arrays()
        legacy={k:v for k,v in arrays.items() if v.dtype.kind!='U'}
        with project_temporary() as directory:
            for original in (arrays,legacy):
                path=Path(directory)/'checkpoint.pt'
                saved=self.checkpoint()
                saved['best_rollout']=checkpoint_replay(original)
                torch.save(saved,path)
                restored=replay_from_checkpoint(torch.load(path,weights_only=True)['best_rollout'])
                StateReplay(restored)
                self.assertEqual(set(original),set(restored))
                for key in original:
                    np.testing.assert_array_equal(restored[key],original[key])
                    self.assertEqual(restored[key].dtype,original[key].dtype)
                # Resume writes an inherited NPZ from the embedded best rollout.
                np.savez_compressed(Path(directory)/'inherited-best.npz',**restored)
                with np.load(Path(directory)/'inherited-best.npz',allow_pickle=False) as replay:
                    StateReplay(dict(replay))
                    for key in original:np.testing.assert_array_equal(replay[key],original[key])


if __name__=='__main__':unittest.main()
