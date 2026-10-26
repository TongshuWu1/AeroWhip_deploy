"""Approved soft boundary: accounting and constructed guard/transfer checks."""
from dataclasses import replace
import math
from pathlib import Path
import unittest
import numpy as np
import torch
from tasks.whip.continuous_whip import ContinuousWhipAttempt,net_return
from tasks.whip.sac_environment import make_environment
from training.configuration import load
from training.ppo import PPO

PRESET='configs/learning/whip_stage1_ppo_continuous_horizontal_soft_workspace_v2.json'

class SoftWorkspaceTests(unittest.TestCase):
    def test_distance_scaling_fixed_horizon_and_strict_monotonicity(self):
        p=np.zeros((11,3));p[:,2]=np.linspace(2,1,11);v=np.zeros_like(p)
        results=[]
        for outside in (0,.1,.5,1.,10.):
            a=ContinuousWhipAttempt(np.full(10,.1),workspace_origin=[0,0,2]);a.initialize(p,v)
            rewards=[a.observe(i*.01,p,v,drone_position=[.55+outside,0,2]) for i in range(1,701)]
            self.assertAlmostEqual(a.cost,10*outside**2,places=9)
            self.assertAlmostEqual(sum(rewards),a.total,places=12)
            self.assertEqual(a.terminal,'horizon');self.assertGreater(a.total,-5)
            results.append(a.total)
        self.assertTrue(all(x>y for x,y in zip(results,results[1:])))

    def test_physical_failure_is_worse_than_valid_net_and_no_early_bonus(self):
        p=np.zeros((11,3));p[:,2]=np.linspace(2,1,11);v=np.zeros_like(p)
        for n in (10,150):
            a=ContinuousWhipAttempt(np.full(10,.1),workspace_origin=[0,0,2]);a.initialize(p,v)
            rewards=[a.observe(i*.01,p,v,drone_position=[2,0,4]) for i in range(1,n+1)]
            self.assertGreater(a.total,-5);self.assertGreater(a.cost,0)
            rewards.append(a.fail(n*.01+.001,'physical_failure'))
            self.assertAlmostEqual(sum(rewards),-10,places=12)
        self.assertEqual(net_return(2.34),2.34)

    def env(self):
        settings,snapshot,cfg,scene=load(PRESET)
        return make_environment(dict(settings,device='cpu',parallel_environments=1),snapshot,cfg,scene,capture=False)

    def test_boundary_crossing_continues_full_horizon_and_preserves_ledger(self):
        e=self.env();initial=e.state
        outside=replace(initial,position_com=initial.position_com+e.model.tensor([.7,0,0]))
        def fake(state,command,controller,**kwargs):
            kwargs['observer'](outside,.01)
            return outside
        e.model.step_unchecked=fake
        for i in range(350):
            _,_,done=e.step(torch.zeros(1,3))
            self.assertEqual(done,i==349)
        returns,valid,score,cost=e.finish()
        self.assertTrue(valid[0]);self.assertTrue(e.workspace_breached[0]);self.assertGreater(cost[0],0)
        self.assertAlmostEqual(e.terminal_times[0].item(),7.)
        self.assertEqual(e.critic_state_size,219)
        self.assertTrue(e.history['workspace_violation'].any())
        torch.testing.assert_close(e.transition_rewards.sum(1),returns)
        c=score.components
        torch.testing.assert_close(c['continuous_whip_credit']-c['workspace_cost']+c['return_transform_adjustment']+c['failure_adjustment'],returns.double())
        self.assertEqual(e.replay_arrays()['continuous_whip_diagnostics'].shape,(701,15))

    def test_outside_workspace_does_not_hide_physical_failure(self):
        e=self.env();initial=e.state;rotation=torch.eye(3,dtype=initial.rotation.dtype).unsqueeze(0);rotation[:,1,1]=-1;rotation[:,2,2]=-1
        bad=replace(initial,position_com=initial.position_com+e.model.tensor([.7,0,0]),rotation=rotation)
        def fake(state,command,controller,**kwargs):
            kwargs['observer'](bad,1/600);return bad
        e.model.step_unchecked=fake
        _,_,done=e.step(torch.zeros(1,3));self.assertTrue(done)
        returns,valid,score,_=e.finish()
        self.assertFalse(valid[0]);self.assertEqual(returns[0].item(),-10)
        self.assertEqual(e.failure_reasons[0],'physical_failure')

    def test_actor_and_entropy_transfer_without_critic_or_optimizer_transfer(self):
        source=Path('runs/ppo_stage1/20261025-153330-415723/latest.pt')
        if not source.exists():self.skipTest('Local stopped checkpoint unavailable')
        saved=torch.load(source,map_location='cpu',weights_only=True)
        self.assertEqual(saved['episodes'],2816)
        e=self.env();a=PPO(e.settings,e.actor_columns,e.frame_size,'cpu',privileged_size=e.critic_state_size,encoder_options=e.policy_encoder_options)
        a.initialize_actor(saved['agent'])
        for k,v in saved['agent']['actor'].items():torch.testing.assert_close(a.actor.state_dict()[k],v,rtol=0,atol=0)
        self.assertEqual(a.entropy_episode_budget,25600);self.assertEqual(a.entropy_episode_offset,2816)
        self.assertAlmostEqual(a.entropy_weight,.001+.019*math.exp(-8*2816/25600),places=14)
        self.assertEqual(a.actor_optimizer.state_dict()['state'],{})
        self.assertEqual(a.value_optimizer.state_dict()['state'],{})
        self.assertEqual(a.completed_episodes,0)

if __name__=='__main__':unittest.main()
