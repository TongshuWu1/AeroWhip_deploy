"""Approved fold contract: causal geometry, reward, boundary and PPO clocks."""
import math
from copy import deepcopy
from dataclasses import replace
from types import SimpleNamespace
import unittest
import numpy as np
import torch
from tasks.whip.horizontal_fold import HorizontalFoldAttempt,FoldGeometry,output_sample,CONTRACT
from tasks.whip.sac_environment import make_environment
from training.configuration import load
from training.ppo import PPO
from simulation.tensor_model import controlled_values

PRESET='configs/learning/whip_stage1_ppo_horizontal_fold_v2.json'
REST=np.array([.063,.087,.1,.1,.1,.1,.1,.1025,.1,.1])
def chain(turn):
    angle=np.r_[0.,np.cumsum(turn)];u=np.c_[np.sin(angle),np.zeros(10),-np.cos(angle)]
    return np.vstack((np.zeros(3),np.cumsum(REST[:,None]*u,axis=0)))

class HorizontalFoldTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2);cls.s,cls.snap,cls.cfg,cls.scene=load(PRESET)
    def env(self,n=1):return make_environment(dict(self.s,device='cpu',parallel_environments=n),self.snap,self.cfg,self.scene,capture=False)
    def test_u_s_coherence_and_static_timeout(self):
        u=np.array([.05,.15,.30,.50,.50,.30,.15,.05,0])*np.pi/2
        for theta,coherent in [(u,True),(u*np.array([1,1,1,1,-1,-1,-1,-1,1]),False)]:
            p=chain(theta);g=FoldGeometry(REST);_,regions=g.observe(p)
            c=max((r['coherence'] for r in regions),default=0.)
            self.assertEqual(c>.8,coherent)
            a=HorizontalFoldAttempt(REST);rewards=[a.observe(k*.01,p,np.zeros_like(p)) for k in range(1,701)]
            self.assertEqual(a.terminal,'timeout');self.assertAlmostEqual(sum(rewards),-2.7,places=12)
            self.assertFalse(a.qualified)
    def test_straight_cable_never_forms(self):
        p=chain(np.zeros(9));a=HorizontalFoldAttempt(REST)
        for k in range(1,701):a.observe(k*.01,p,np.ones_like(p))
        self.assertFalse(a.formed);self.assertEqual(a.terminal,'timeout')
    def test_output_frames_and_alignment(self):
        p=chain(np.zeros(9));p[-1]=[.8,0,0];v=np.zeros_like(p)
        for tip,root,expected in [([1,0,0],[0,0,0],True),([.1,0,2],[0,0,0],False),
                ([-.2,0,0],[-2,0,0],False),([1,0,0],[2,0,0],False),([1,0,0],[0,0,-2],False),([.05,0,0],[0,0,0],False)]:
            v[-1]=tip;v[0]=root;s=output_sample(p,v,1.)
            self.assertEqual(s['outward'] and s['aligned'],expected)
    def test_geometry_rotation_and_prefix(self):
        theta=np.linspace(.05,.5,9);p=chain(theta);Q=np.array([[0,-1,0],[1,0,0],[0,0,1.]])
        ga,gb=FoldGeometry(REST),FoldGeometry(REST)
        for k in range(6):
            ta,ra=ga.observe(p);tb,rb=gb.observe(p@Q.T)
            np.testing.assert_allclose(ta,tb,atol=1e-14)
            self.assertEqual(ra,rb)
    def test_potential_terminal_failure_claws_back(self):
        a=HorizontalFoldAttempt(REST);p=chain(np.full(9,.3))
        rewards=[a.observe(k*.01,p,np.zeros_like(p)) for k in range(1,8)]
        rewards.append(a.fail(.071,'workspace_failure'))
        self.assertAlmostEqual(sum(rewards),-10.0071,places=12)
        self.assertEqual(a.phi,0.);self.assertFalse(a.qualified)
    def test_no_after_terminal_or_offgrid_updates(self):
        a=HorizontalFoldAttempt(REST);p=chain(np.zeros(9));v=np.zeros_like(p)
        with self.assertRaises(ValueError):a.observe(.02,p,v)
        a.fail(.005)
        with self.assertRaises(RuntimeError):a.observe(.01,p,v)
    def test_configuration_rejects_criteria_and_mesh_changes(self):
        from training.configuration import task_configuration
        for key,value in [('output_cone_deg',90),('radial_floor_mps',0),('horizon_s',4.5)]:
            raw=deepcopy(self.snap['task']);raw['horizontal_fold'][key]=value
            with self.assertRaises(ValueError):task_configuration(raw,self.s,self.snap['model'],self.scene)
        model=deepcopy(self.snap['model']);model['simulation']['segments_per_marker_interval']=[2]*10
        with self.assertRaises(ValueError):task_configuration(self.snap['task'],self.s,model,self.scene)
    def test_seven_second_history_and_entropy(self):
        e=self.env();self.assertEqual((e.samples,e.decisions),(701,350))
        p=PPO(self.s,e.actor_columns,e.frame_size,'cpu',privileged_size=e.critic_state_size,encoder_options=e.policy_encoder_options)
        for n in (0,6400,12800,19200,25600):
            p.set_completed_episodes(n);self.assertAlmostEqual(p.entropy_weight,.001+.019*math.exp(-8*n/25600),places=15)
        saved=p.state_dict();q=PPO(dict(self.s,training_episodes=51200),e.actor_columns,e.frame_size,'cpu',privileged_size=e.critic_state_size,encoder_options=e.policy_encoder_options)
        q.load_state_dict(saved)
        self.assertEqual(q.entropy_episode_budget,25600);self.assertEqual(q.entropy_weight,p.entropy_weight)
    def test_read_only_substep_observer_preserves_native_state(self):
        e=self.env();m=e.model;command=e.hold[None];seen=[]
        plain=m.step_unchecked(e.state,command,e.controller,batch=True)
        watched=m.step_unchecked(e.state,command,e.controller,batch=True,observer=lambda s,t:seen.append(t))
        self.assertEqual(len(seen),6);self.assertAlmostEqual(seen[-1],.01)
        for a,b in zip(controlled_values(plain),controlled_values(watched)):torch.testing.assert_close(a,b,rtol=0,atol=0)
    def test_first_substep_breach_is_not_hidden_by_endpoint(self):
        e=self.env();initial=e.state
        outside=replace(initial,position_com=initial.position_com+e.model.tensor([.6,0,0]))
        def fake(state,command,controller,**kwargs):
            kwargs['observer'](outside,1/600);kwargs['observer'](initial,2/600);return initial
        e.model.step_unchecked=fake
        _,_,done=e.step(torch.zeros(1,3));self.assertTrue(done)
        r,valid,score,_=e.finish();self.assertFalse(valid[0])
        self.assertAlmostEqual(float(e.terminal_times[0]),1/600)
        self.assertAlmostEqual(float(r[0]),-10-.1/600,places=5)
        self.assertTrue(e.workspace_breached[0]);self.assertAlmostEqual(float(e.replay_arrays()['time_s'][-1]),1/600)
        self.assertAlmostEqual(float(e.transition_rewards.sum()),float(r[0]),places=6)
    def test_failed_lane_frozen_while_other_lane_continues(self):
        e=self.env(2);initial=e.state
        outside=replace(initial,position_com=initial.position_com+e.model.tensor([[.6,0,0],[0,0,0]]))
        def fake(state,command,controller,**kwargs):kwargs['observer'](outside,1/600);return outside
        e.model.step_unchecked=fake;e.step(torch.zeros(2,3))
        self.assertEqual(e.active.tolist(),[False,True]);self.assertEqual(e.lengths.tolist(),[1,1])
        reward=e.ledger[0].sum().item();e.step(torch.zeros(2,3));self.assertEqual(e.ledger[0].sum().item(),reward)
        self.assertEqual(e.lengths.tolist(),[1,2])

if __name__=='__main__':unittest.main()
