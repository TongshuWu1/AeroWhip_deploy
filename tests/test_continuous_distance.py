"""Direct cost and remaining-time failure accounting; fixtures are not dynamics."""
from dataclasses import replace
from pathlib import Path
import math
import unittest
import numpy as np
import torch
from tasks.whip.continuous_whip import ContinuousWhipAttempt
from tasks.whip.sac_environment import make_environment
from training.configuration import load,task_configuration
from training.ppo import PPO
from tests.test_continuous_whip import state,region

PRESET='configs/learning/whip_stage1_ppo_continuous_horizontal_distance_v3.json'

class DirectDistanceTests(unittest.TestCase):
    def env(self):
        s,snapshot,cfg,scene=load(PRESET)
        return make_environment(dict(s,device='cpu',parallel_environments=1),snapshot,cfg,scene,capture=False)

    def test_no_cost_compression_or_fixed_failure_total(self):
        p,v=state(tip_velocity=(0.,0.,0.))
        for excess in (.1,.5,1.,10.):
            a=ContinuousWhipAttempt(np.full(10,.1),workspace_origin=[0,0,2],plain_net=True);a.initialize(p,v)
            rewards=[a.observe(i*.01,p,v,drone_position=[.55+excess,0,2]) for i in range(1,701)]
            self.assertAlmostEqual(a.total,-10*excess**2,places=9)
            self.assertAlmostEqual(sum(rewards),a.total,places=10)
            self.assertEqual(a.failure_cost,0)

    def test_failure_keeps_both_credit_and_cost_then_charges_remaining_time(self):
        for n,expected_cost in ((100,6.),(600,1.)):
            p,v=state();a=ContinuousWhipAttempt(np.full(10,.1),workspace_origin=[0,0,2],plain_net=True)
            rows=[region(.1)];a.geometry.observe=lambda _:(np.zeros(9),rows)
            a.initialize(p,v);rewards=[]
            for i in range(1,n+1):
                rows[:]=[region(.1+.01*i)] if i<40 else []
                rewards.append(a.observe(i*.01,p,v,drone_position=[.75,0,2]))
            q,c=a.accrued,a.cost;self.assertGreater(q,0);self.assertGreater(c,0)
            rewards.append(a.fail(n*.01,'physical_failure'))
            self.assertEqual(a.failure_cost,expected_cost)
            self.assertEqual((a.accrued,a.cost),(q,c))
            self.assertAlmostEqual(sum(rewards),q-c-expected_cost,places=12)
            self.assertNotEqual(a.total,-10)
            self.assertEqual(a.record()['events'][-1]['preserved_credit'],q)

    def test_numerical_abort_uses_same_accounting_and_actual_substep_time(self):
        p,v=state(tip_velocity=(0,0,0));a=ContinuousWhipAttempt(np.full(10,.1),workspace_origin=[0,0,2],plain_net=True);a.initialize(p,v)
        prefix=a.observe(.01,p,v,drone_position=[1,0,2]);correction=a.fail(.013,'numerical_failure')
        self.assertAlmostEqual(correction,-6.987,places=12)
        self.assertAlmostEqual(prefix+correction,-a.cost-6.987,places=12)

    def test_native_adapter_constructed_physical_abort_preserves_boundary_cost(self):
        e=self.env();initial=e.state
        outside=replace(initial,position_com=initial.position_com+e.model.tensor([.7,0,0]))
        def safe(state,command,controller,**kw):kw['observer'](outside,.01);return outside
        e.model.step_unchecked=safe;e.step(torch.zeros(1,3));self.assertTrue(e.active[0])
        rotation=torch.eye(3,dtype=initial.rotation.dtype).unsqueeze(0);rotation[:,1,1]=-1;rotation[:,2,2]=-1
        bad=replace(outside,rotation=rotation)
        def fail(state,command,controller,**kw):kw['observer'](bad,1/600);return bad
        e.model.step_unchecked=fail;_,_,done=e.step(torch.zeros(1,3));self.assertTrue(done)
        returns,valid,score,cost=e.finish();c=score.components
        self.assertFalse(valid[0]);self.assertGreater(cost[0],0);self.assertEqual(c['return_transform_adjustment'][0],0)
        expected=c['continuous_whip_credit']-c['workspace_cost']-c['failure_cost']
        torch.testing.assert_close(expected,returns.double());torch.testing.assert_close(e.transition_rewards.sum(1),returns)
        self.assertAlmostEqual(c['failure_cost'][0].item(),7-.02-1/600,places=12)

    def test_source_actor_and_total_campaign_clock_preserved(self):
        source=Path('runs/ppo_stage1/20261025-161103-645210/latest.pt')
        if not source.exists():self.skipTest('Local stopped source not available')
        saved=torch.load(source,map_location='cpu',weights_only=True);e=self.env()
        a=PPO(e.settings,e.actor_columns,e.frame_size,'cpu',privileged_size=e.critic_state_size,encoder_options=e.policy_encoder_options);a.initialize_actor(saved['agent'])
        for k,v in saved['agent']['actor'].items():torch.testing.assert_close(a.actor.state_dict()[k],v,atol=0,rtol=0)
        self.assertEqual(a.entropy_episode_budget,25600);self.assertEqual(a.entropy_episode_offset,3200)
        self.assertAlmostEqual(a.entropy_weight,.001+.019*math.exp(-1),places=14)
        self.assertEqual(a.actor_optimizer.state_dict()['state'],{});self.assertEqual(a.value_optimizer.state_dict()['state'],{})
        self.assertEqual(e.settings['training_episodes']+a.entropy_episode_offset,25600)

    def test_failure_modes_cannot_be_combined_or_used_by_old_reward(self):
        from copy import deepcopy
        s,snapshot,cfg,scene=load(PRESET)
        changed=deepcopy(snapshot['task']);changed['return']['failure_return']=-10
        with self.assertRaises(ValueError):task_configuration(changed,s,snapshot['model'],scene)
        s,old,cfg,scene=load('configs/learning/whip_stage1_ppo_continuous_horizontal_v1.json')
        changed=deepcopy(old['task']);changed['return']['failure_return']=None;changed['return']['failure_cost_per_remaining_s']=1
        with self.assertRaises(ValueError):task_configuration(changed,s,old['model'],scene)

if __name__=='__main__':unittest.main()
