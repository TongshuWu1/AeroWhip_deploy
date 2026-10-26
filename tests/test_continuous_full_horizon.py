"""Constructed finite/invalid native-adapter states, not dynamics validation."""
from dataclasses import replace
import json
import math
from pathlib import Path
import unittest
import numpy as np
import torch
from training.configuration import load
from tasks.whip.sac_environment import make_environment

PRESET='configs/learning/whip_stage1_ppo_continuous_horizontal_full_horizon_v4.json'

class FullHorizonTests(unittest.TestCase):
    def env(self):
        settings,snapshot,cfg,scene=load(PRESET)
        return make_environment(dict(settings,device='cpu',parallel_environments=1),snapshot,cfg,scene,capture=False)

    def test_tilt_and_low_cable_continue_all_seven_seconds_and_preserve_reward(self):
        e=self.env();initial=e.state
        theta=math.radians(75);rotation=e.model.tensor([[math.cos(theta),0,math.sin(theta)],[0,1,0],[-math.sin(theta),0,math.cos(theta)]]).unsqueeze(0)
        bad=replace(initial,position_com=initial.position_com+e.model.tensor([.8,0,-2.3]),rotation=rotation)
        def fixed(state,command,controller,**kw):kw['observer'](bad,.01);return bad
        e.model.step_unchecked=fixed
        for step in range(350):
            _,_,done=e.step(torch.zeros(1,3))
            self.assertEqual(done,step==349)
            self.assertFalse(bool(e.failed[0]))
        returns,valid,score,cost=e.finish();c=score.components
        self.assertFalse(bool(valid[0]));self.assertEqual(c['completed_horizon'][0],1)
        self.assertEqual(c['physical_violation'][0],1);self.assertEqual(c['numerically_valid'][0],1)
        self.assertEqual(c['failure_cost'][0],0);self.assertEqual(c['failure_adjustment'][0],0)
        self.assertLess(c['minimum_cable_height_m'][0],.05);self.assertAlmostEqual(float(c['maximum_tilt_deg'][0]),75)
        torch.testing.assert_close(returns.double(),c['continuous_whip_credit']-c['workspace_cost'],rtol=1e-6,atol=1e-6)
        torch.testing.assert_close(e.transition_rewards.sum(1),returns)
        row=e.event_records()[0];self.assertEqual(row['terminal'],'horizon');self.assertEqual(row['terminal_time_s'],7)
        self.assertEqual(row['contract'],'continuous_horizontal_whip_full_horizon_v4');self.assertTrue(row['physical_violation'])
        replay=e.replay_arrays();self.assertEqual(len(replay['time_s']),701)
        self.assertFalse(bool(replay['task_valid']));self.assertTrue(bool(replay['numerically_valid']))
        self.assertTrue(json.loads(str(replay['fold_event_json']))['physical_violation'])
        e.reset();self.assertFalse(bool(e.physical_violation[0]))

    def test_numerical_corruption_still_aborts_before_geometry(self):
        e=self.env();bad=replace(e.state,position_com=torch.full_like(e.state.position_com,float('nan')))
        def invalid(state,command,controller,**kw):kw['observer'](bad,1/600);return bad
        e.model.step_unchecked=invalid
        _,_,done=e.step(torch.zeros(1,3));self.assertTrue(done)
        returns,valid,score,_=e.finish();self.assertFalse(bool(valid[0]));self.assertEqual(e.attempts[0].terminal,'numerical_failure')
        self.assertEqual(len(e.attempts[0].trace),1)
        self.assertAlmostEqual(float(score.components['failure_cost'][0]),7-1/600)
        self.assertTrue(torch.isfinite(returns).all())

    def test_invalid_rotation_still_aborts(self):
        e=self.env();bad=replace(e.state,rotation=torch.zeros_like(e.state.rotation))
        def invalid(state,command,controller,**kw):kw['observer'](bad,1/600);return bad
        e.model.step_unchecked=invalid;e.step(torch.zeros(1,3))
        self.assertEqual(e.attempts[0].terminal,'numerical_failure')

    def test_old_physical_guard_remains_and_new_contract_cannot_hide_termination(self):
        from copy import deepcopy
        from training.configuration import task_configuration
        s,snapshot,cfg,scene=load('configs/learning/whip_stage1_ppo_continuous_horizontal_distance_v3.json')
        e=make_environment(dict(s,device='cpu',parallel_environments=1),snapshot,cfg,scene,capture=False)
        self.assertTrue(e.physical_termination)
        s,snapshot,cfg,scene=load(PRESET);changed=deepcopy(snapshot['task'])
        changed['continuous_whip']['physical_termination']=True
        with self.assertRaises(ValueError):task_configuration(changed,s,snapshot['model'],scene)
        self.assertEqual(s['training_episodes'],9920)
        self.assertEqual(s['parallel_environments'],64)

    def test_stopped_actor_continuation_clock_and_guard(self):
        from training.ppo import PPO
        from training.artifacts import validate_actor_source
        path=Path('runs/ppo_stage1/20261025-163957-732295/latest.pt')
        if not path.exists():self.skipTest('Local stopped v3 checkpoint missing')
        saved=torch.load(path,map_location='cpu',weights_only=True)
        # Reconstruct the original handoff, not the subsequently resized preset.
        settings,_,_,_=load(PRESET)
        settings.update(parallel_environments=128,training_episodes=153*128)
        settings,snapshot,cfg,scene=load(settings)
        validate_actor_source(saved,snapshot)
        e=make_environment(dict(settings,device='cpu',parallel_environments=1),snapshot,cfg,scene,capture=False)
        a=PPO(e.settings,e.actor_columns,e.frame_size,'cpu',privileged_size=e.critic_state_size,encoder_options=e.policy_encoder_options)
        a.initialize_actor(saved['agent'])
        self.assertEqual(a.entropy_episode_offset,6016);self.assertEqual(a.entropy_episode_budget,25600)
        self.assertEqual(a.entropy_episode_offset+e.settings['training_episodes'],25600)
        for k,v in saved['agent']['actor'].items():torch.testing.assert_close(a.actor.state_dict()[k],v,rtol=0,atol=0)
        self.assertFalse(a.actor_optimizer.state_dict()['state']);self.assertFalse(a.value_optimizer.state_dict()['state'])

if __name__=='__main__':unittest.main()
