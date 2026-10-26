"""Directed-output counterexamples and adapter accounting; not physical validation."""
from copy import deepcopy
from dataclasses import replace
import math
import unittest
import numpy as np
import torch
from helpers import TASK
from tasks.whip.configuration import REWARD_VERSION
from tasks.whip.continuous_whip import ContinuousWhipAttempt, output_metrics
from tasks.whip.environment import make_environment
from training.configuration import DEFAULT, load, task_configuration
from training.ppo import PPO

PRESET=DEFAULT


class DirectionalTests(unittest.TestCase):
    def sample(self, direction=(1.,0.,0.), velocity=None, root_velocity=(0.,0.,0.)):
        p=np.linspace(np.zeros(3),np.array(direction)*.8,11)
        v=np.tile(root_velocity,(11,1)).astype(float)
        v[-1]=np.array(direction if velocity is None else velocity)*2
        return p,v

    def rate(self,p,v):
        m=output_metrics(p,v,1.)
        return 10*m['horizontal_reach_fraction']**2*m['alignment']**2*m['effective_outward_mps']

    def test_forward_exact(self):
        p,v=self.sample()
        self.assertAlmostEqual(self.rate(p,v),12.8)

    def test_sideways_backwards_downward_zero(self):
        for direction in ((0.,1.,0.),(-1.,0.,0.),(0.,0.,-1.)):
            p,v=self.sample(direction)
            self.assertEqual(self.rate(p,v),0.)

    def test_oblique_receives_partial_credit(self):
        p,v=self.sample((math.sqrt(.5),math.sqrt(.5),0.))
        self.assertAlmostEqual(self.rate(p,v),12.8*math.sqrt(.5)**5)

    def test_both_frames_and_extension_required(self):
        p,v=self.sample(root_velocity=(2.,0.,0.))
        self.assertEqual(self.rate(p,v),0.)  # rigid translation
        p,v=self.sample(velocity=(-1.,0.,0.),root_velocity=(-3.,0.,0.))
        self.assertEqual(self.rate(p,v),0.)  # relative forward, world backward
        p,v=self.sample((.5,0.,-.8),velocity=(1.,0.,2.))
        self.assertEqual(self.rate(p,v),0.)  # forward but shortening
        p,v=self.sample((-1.,0.,0.),velocity=(1.,0.,0.))
        self.assertEqual(self.rate(p,v),0.)  # behind attachment

    def test_no_fold_zero_credit_repetition_and_workspace_accounting(self):
        p,v=self.sample()
        a=ContinuousWhipAttempt(np.full(10,.1),TASK,workspace_origin=np.zeros(3))
        a.initialize(p,v)
        self.assertEqual(a.observe(.01,p,v,drone_position=[0,0,2]),0.)
        # Isolate scoring from fold geometry with explicitly synthetic support.
        a._fold_support=lambda *args:(1.,1.,math.pi)
        delta=[]
        for k in range(2,401):
            delta.append(a.observe(k*.01,p,v,drone_position=[.65,0,2]))
        expected_q=399*.01*12.8
        expected_c=399*10*.1**2/400
        self.assertAlmostEqual(a.total,expected_q-expected_c)
        self.assertAlmostEqual(sum(delta),a.total)
        self.assertEqual(a.terminal,'horizon')
        self.assertEqual(a.record()['contract'],REWARD_VERSION)

    def env(self):
        s,snapshot,cfg,scene=load(PRESET)
        return make_environment(dict(s,device='cpu',parallel_environments=1),snapshot,cfg,scene,capture=False)

    def test_adapter_full_horizon_despite_physical_violation(self):
        e=self.env();bad=replace(e.state,position_com=e.state.position_com+e.model.tensor([.8,0,-2.3]))
        def fixed(state,command,controller,**kw):
            kw['observer'](bad,.01)
            return bad
        e.model.step_unchecked=fixed
        for i in range(200):
            _,_,done=e.step(torch.zeros(1,3))
            self.assertEqual(done,i==199)
        returns,valid,score,_=e.finish()
        self.assertFalse(bool(valid[0]));self.assertFalse(bool(e.failed[0]))
        self.assertEqual(e.event_records()[0]['contract'],REWARD_VERSION)
        self.assertEqual(e.event_records()[0]['output_direction_world'],[1.,0.,0.])
        torch.testing.assert_close(e.transition_rewards.sum(1),returns)
        self.assertEqual(score.components['executed_duration_s'][0],4.)
        self.assertEqual(score.components['failure_cost'][0],0.)

    def test_numerical_guard_retained(self):
        e=self.env();bad=replace(e.state,position_com=torch.full_like(e.state.position_com,float('nan')))
        def invalid(state,command,controller,**kw):
            kw['observer'](bad,1/600)
            return bad
        e.model.step_unchecked=invalid
        _,_,done=e.step(torch.zeros(1,3));self.assertTrue(done)
        returns,_,score,_=e.finish()
        self.assertAlmostEqual(float(score.components['failure_cost'][0]),4-1/600)
        self.assertTrue(torch.isfinite(returns).all())

    def test_registered_direction_and_fresh_entropy_budget(self):
        s,snapshot,cfg,scene=load(PRESET)
        self.assertEqual(s['training_episodes'],200*64)
        bad=deepcopy(snapshot['task']);bad['output_direction_world']=[0.,1.,0.]
        with self.assertRaises(ValueError):task_configuration(bad,s,snapshot['model'],scene)
        e=self.env()
        a=PPO(e.settings,e.actor_columns,e.frame_size,'cpu',privileged_size=e.critic_state_size,encoder_options=e.policy_encoder_options)
        self.assertEqual(a.entropy_episode_budget,12800)
        self.assertFalse(a.actor_optimizer.state_dict()['state'])
        self.assertFalse(a.value_optimizer.state_dict()['state'])


if __name__=='__main__':unittest.main()
