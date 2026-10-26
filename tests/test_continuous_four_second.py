"""Four-second task schedule, full native-adapter accounting and config validation."""
from copy import deepcopy
from dataclasses import replace
import unittest
import numpy as np
import torch
from helpers import TASK
from tasks.whip.configuration import REWARD_VERSION
from tasks.whip.continuous_whip import ContinuousWhipAttempt
from tasks.whip.environment import make_environment
from training.configuration import DEFAULT, load, task_configuration
from training.ppo import PPO

PRESET=DEFAULT

class FourSecondTests(unittest.TestCase):
    def env(self):
        s,snap,cfg,scene=load(PRESET)
        return make_environment(dict(s,device='cpu',parallel_environments=1),snap,cfg,scene,capture=False)

    def test_exact_horizon_history_learner_and_physical_violation(self):
        e=self.env()
        self.assertEqual((e.samples,e.decisions),(401,200))
        self.assertEqual(e.policy_encoder_options,dict(history_samples=401))
        a=PPO(e.settings,e.actor_columns,e.frame_size,'cpu',privileged_size=e.critic_state_size,encoder_options=e.policy_encoder_options)
        self.assertEqual(a.interface['history_samples'],401)
        self.assertEqual(a.entropy_episode_budget,12800)
        self.assertFalse(a.actor_optimizer.state_dict()['state'])
        bad=replace(e.state,position_com=e.state.position_com+e.model.tensor([.8,0,-2.3]))
        def fixed(state,command,controller,**kw):
            kw['observer'](bad,.01)
            return bad
        e.model.step_unchecked=fixed
        for i in range(200):
            _,_,done=e.step(torch.zeros(1,3));self.assertEqual(done,i==199)
        returns,valid,score,cost=e.finish()
        self.assertFalse(bool(valid[0]));self.assertFalse(bool(e.failed[0]))
        self.assertEqual(score.components['executed_duration_s'][0],4.)
        self.assertEqual(score.components['failure_cost'][0],0.)
        torch.testing.assert_close(e.transition_rewards.sum(1),returns)
        self.assertEqual(e.event_records()[0]['contract'],REWARD_VERSION)
        self.assertEqual(len(e.replay_arrays()['time_s']),401)
        self.assertAlmostEqual(float(e.attempts[0].features()[0]),1.)
        with self.assertRaises(RuntimeError):e.step(torch.zeros(1,3))

    def test_mean_workspace_cost_and_rate_are_separate(self):
        # Known constant output with synthetic fold support isolates integration.
        p=np.linspace([0,0,0],[.8,0,0],11);v=np.zeros_like(p);v[-1,0]=2
        result=[]
        for horizon in (TASK.duration_s,):
            a=ContinuousWhipAttempt(np.full(10,.1),TASK,workspace_origin=np.zeros(3))
            a.initialize(p,v);a._fold_support=lambda *args:(1.,1.,np.pi)
            rewards=[a.observe(k*.01,p,v,drone_position=[.65,0,2]) for k in range(1,int(horizon*100)+1)]
            self.assertAlmostEqual(a.cost,.1)
            self.assertAlmostEqual(a.accrued,12.8*horizon)
            self.assertAlmostEqual(sum(rewards),a.total)
            result.append(a)

    def test_numerical_abort_uses_four_second_deadline(self):
        e=self.env();bad=replace(e.state,position_com=torch.full_like(e.state.position_com,float('nan')))
        def invalid(state,command,controller,**kw):
            kw['observer'](bad,1/600)
            return bad
        e.model.step_unchecked=invalid
        _,_,done=e.step(torch.zeros(1,3));self.assertTrue(done)
        returns,_,score,_=e.finish()
        self.assertAlmostEqual(float(score.components['failure_cost'][0]),4-1/600)
        self.assertTrue(torch.isfinite(returns).all())

    def test_invalid_task_values_rejected(self):
        s,snap,cfg,scene=load(PRESET)
        edits=[('duration_s',4.005),('duration_s',-1),('reward','unknown_v1'),('rate_weight',0),
               ('maximum_tilt_deg',90),('workspace',dict(radius_m=.55)),('extra',1)]
        for key,value in edits:
            bad=deepcopy(snap['task']);bad[key]=value
            with self.subTest(key=key,value=value),self.assertRaises(ValueError):
                task_configuration(bad,s,snap['model'],scene)
        bad=deepcopy(snap['task']);del bad['fold']
        with self.assertRaises(ValueError):task_configuration(bad,s,snap['model'],scene)
        bad=deepcopy(snap['task']);bad['fold']['turn_upper_deg']=80
        with self.assertRaises(ValueError):task_configuration(bad,s,snap['model'],scene)
        for key,value in (('policy_rate_hz',30),('acceleration_limit_mps2',[8,8]),('discount',0),('action_interface','velocity_v1')):
            with self.subTest(key=key),self.assertRaises(ValueError):load(dict(s,task=snap['task'],**{key:value}))
        e=self.env()
        with self.assertRaises(ValueError):PPO(e.settings,e.actor_columns,e.frame_size,'cpu',privileged_size=e.critic_state_size,encoder_options=dict(phase=1))

if __name__=='__main__':unittest.main()
