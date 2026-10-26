"""First-hit semantics and policy handoff; analytic histories, not flight evidence."""
import unittest
from copy import deepcopy
import numpy as np
import torch
from tasks.whip.first_hit_delivery import FirstHitTracker
from tasks.whip.flexible_delivery import score_trace
from tasks.whip.sac_environment import Stage1Environment
from training.configuration import load
from training.ppo import PPO, advantages

PRESET='configs/learning/whip_stage2_ppo_first_hit_delivery_v2.json'
OLD='configs/learning/whip_stage2_ppo_flexible_delivery_v1.json'


class FirstHitTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)
        cls.s,cls.snap,cls.cfg,cls.scene=load(PRESET)

    def env(self,n=1):
        return Stage1Environment(dict(self.s,device='cpu',parallel_environments=n),self.snap,self.cfg,self.scene,capture=False)

    def inject(self,e,end=500,hit=False,failed=False):
        z=2+torch.flip(torch.cat((torch.zeros_like(e.model.rest[:1]),e.model.rest.cumsum(0))),[0])
        e.history['cable_positions'][:]=0;e.history['cable_positions'][...,2]=z
        e.history['cable_positions'][...,0]=1.
        if hit:e.history['cable_positions'][:,end:,:,0]=1.2
        e.history['cable_velocities'][:]=0
        e.history['cable_velocities'][:,:,-1]=e.time.new_tensor([4,0,3])
        e.history['drone_positions'][:]=e.origin;e.history['drone_velocities'][:]=0
        e.numerical_valid[:]=False;e.numerical_valid[:,:end+1]=True
        e.failed[:]=failed;e.terminal_hit[:]=hit and not failed;e.active[:]=False
        e.executed_samples[:]=end;e.lengths[:]=(end+1)//2;e.decision=(end+1)//2
        return e

    def test_first_hit_odd_substep_return_and_mask(self):
        e=self.inject(self.env(),end=101,hit=True)
        r,valid,s,_=e.finish();b=float(s.components['crossing_utility'][0])
        self.assertTrue(valid[0]);self.assertAlmostEqual(float(r[0]),b-.101,places=6)
        self.assertEqual(float(s.components['timeout'][0]),0)
        self.assertAlmostEqual(float(s.components['executed_duration_s'][0]),1.01)
        self.assertTrue(bool((e.transition_rewards[:,51:]==0).all()))
        torch.testing.assert_close(e.transition_rewards.sum(1),r)

    def test_timeout_retains_guide_and_pays_deadline_cost(self):
        e=self.inject(self.env());r,valid,s,_=e.finish();a=float(s.components['approach_utility'][0])
        net=a-.5;expected=-2+(net if net>=0 else 5*net/(5-net))
        self.assertTrue(valid[0]);self.assertEqual(float(s.components['timeout'][0]),1)
        self.assertAlmostEqual(float(r[0]),expected,places=6)
        self.assertAlmostEqual(float(s.components['time_cost'][0]),.5,places=12)

    def test_hit_at_deadline_beats_timeout_and_failure_takes_priority(self):
        e=self.inject(self.env(),hit=True);r,valid,s,_=e.finish()
        self.assertEqual(float(s.components['terminal_quality_hit'][0]),1)
        self.assertEqual(float(s.components['timeout'][0]),0)
        e.failed[:]=True;e.terminal_hit[:]=False
        r,valid,s,_=e.finish();self.assertFalse(valid[0])
        self.assertAlmostEqual(float(r[0]),-10-2.5/5.5,places=6)

    def test_workspace_rate_preserved_and_stops_at_execution(self):
        e=self.inject(self.env(),end=100,failed=True)
        e.history['drone_positions'][:,1:101,2]=3.1
        e.history['drone_positions'][:,101:,2]=100
        r,valid,s,_=e.finish();c=2.5*100/450
        self.assertAlmostEqual(float(s.components['workspace_cost'][0]),c,places=12)
        self.assertAlmostEqual(float(r[0]),-10-5*(c+.1)/(5+c+.1),places=6)

    def test_numerical_failure_time_includes_failed_endpoint(self):
        e=self.inject(self.env(),end=100,failed=True)
        e.numerical_valid[:,100:]=False
        r,valid,s,_=e.finish()
        self.assertFalse(valid[0]);self.assertAlmostEqual(float(s.components['time_cost'][0]),.1,places=12)

    def test_online_tracker_matches_contact_and_safety_rules(self):
        rest=torch.tensor([.5,.5],dtype=torch.float64);center=torch.tensor([1.1,0.,2.],dtype=torch.float64)
        p=torch.zeros(1,3,3,dtype=torch.float64);p[:,:,2]=torch.tensor([3,2.5,2]);p[:,:,0]=1.
        v=torch.zeros_like(p);v[:,-1]=torch.tensor([4.,0,3]);active=torch.tensor([True])
        tracker=FirstHitTracker(p,rest,center)
        zero=p.clone();zero[:,:,0]=1.1
        self.assertFalse(tracker.observe(p,zero,v,v,active)[0])
        positive=p.clone();positive[:,:,0]=1.2
        self.assertTrue(tracker.observe(zero,positive,v,v,active)[0])
        self.assertFalse(tracker.observe(zero,positive,v,v,active)[0])
        tracker=FirstHitTracker(p,rest,center)
        self.assertFalse(tracker.observe(p,positive,v,v,torch.tensor([False]))[0])

    def test_weak_crossing_has_no_extra_bonus(self):
        p=np.zeros((2,3,3));p[:,:,2]=[3,2.5,2];p[:,:,0]=np.array([1,1.2])[:,None]
        v=np.zeros_like(p);v[:,-1]=[.2,0,.3]
        r=score_trace(p,v,[0,.01],[.5,.5],[1.1,0,2],1,.6,first_hit=True)
        self.assertFalse(r['crossings'][0]['quality_hit'])
        self.assertAlmostEqual(r['best_event_score'],r['approach_max'])

    def test_step_stops_hit_lane_at_first_substep_while_other_lane_continues(self):
        e=self.inject(self.env(n=2))
        e.active[:]=True;e.decision=e.index=0;e.lengths[:]=e.executed_samples[:]=0
        e.numerical_valid[:]=False;e.numerical_valid[:,0]=True
        before=e.history['cable_positions'][:,0].clone()
        current=before.clone();current[0,:,0]=1.2
        velocity=e.history['cable_velocities'][:,0].clone()
        e.delivery_tracker=FirstHitTracker(before,e.model.rest,e.scorer.center)
        e._advance=lambda command:e.state
        e._result_finite=lambda state:torch.ones(2,dtype=torch.bool)
        e._observe=lambda state:(e.origin.expand(2,3),torch.zeros(2,3,dtype=torch.float64),current,velocity)
        _,_,done=e.step(torch.zeros(2,3))
        self.assertFalse(done)
        self.assertEqual(e.terminal_hit.tolist(),[True,False])
        self.assertEqual(e.active.tolist(),[False,True])
        self.assertEqual(e.executed_samples.tolist(),[1,2])
        self.assertEqual(e.numerical_valid[:,:3].tolist(),[[True,True,False],[True,True,True]])

    def test_history_transfer_prefix_bit_equivalence(self):
        old,osnap,ocfg,oscene=load(OLD)
        oe=Stage1Environment(dict(old,device='cpu',parallel_environments=2),osnap,ocfg,oscene,capture=False)
        ne=self.env(n=2);torch.manual_seed(42)
        a=PPO(old,oe.actor_columns,oe.frame_size,'cpu',oe.action_size,privileged_size=oe.critic_state_size,encoder_options=oe.policy_encoder_options)
        b=PPO(self.s,ne.actor_columns,ne.frame_size,'cpu',ne.action_size,privileged_size=ne.critic_state_size,encoder_options=ne.policy_encoder_options)
        b.initialize_actor(a.state_dict())
        h=torch.randn(2,451,oe.frame_size);extended=torch.cat((h,torch.randn(2,50,oe.frame_size)),1)
        for step in (0,100,224,450):
            index=torch.tensor([step,step])
            pa=a.actor.encoder.prepare(h,index);pb=b.actor.encoder.prepare(extended,index)
            for x,y in zip(pa,pb):self.assertTrue(torch.equal(x,y))
            with torch.no_grad():
                for x,y in zip(a.actor.distribution(h,index),b.actor.distribution(extended,index)):
                    torch.testing.assert_close(x,y,rtol=0,atol=0)
        self.assertEqual(ne.time_samples,450);self.assertEqual(ne.samples,501)
        prepared=b.actor.encoder.prepare(extended,torch.tensor([500,500]))
        self.assertEqual(prepared[0].shape[1],451)

    def test_task_terminal_does_not_bootstrap(self):
        rewards=torch.tensor([[2.,0.,0.],[-2.,0.,0.]])
        values=torch.tensor([[.5,100.,100.],[.5,100.,100.]])
        adv,targets,_=advantages(rewards,values,torch.tensor([1,1]),1.,.95)
        torch.testing.assert_close(targets[:,0],rewards[:,0])

    def test_contract_rejects_unapproved_coefficients(self):
        bad=deepcopy(self.s);bad['stage2_first_hit']['time_cost_per_second']=1
        with self.assertRaises(ValueError):load(bad)
        self.assertEqual(self.s['training_episodes'],7680)


if __name__=='__main__':unittest.main()
