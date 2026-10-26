"""Analytic crossing/accounting counterexamples, not physical validation."""
from copy import deepcopy
import json
from pathlib import Path
import unittest
import numpy as np
import torch

from tasks.whip.flexible_delivery import confirmed_passages, score_trace
from tasks.whip.sac_environment import Stage1Environment
from training.configuration import load, task_configuration
from training.worker import record_simple_collection
from helpers import project_temporary

PRESET='configs/learning/whip_stage2_ppo_flexible_delivery_v1.json'


class DeliveryTests(unittest.TestCase):
    def trace(self,x):
        # Analytic positions/velocities deliberately independent for scorer tests.
        p=np.zeros((len(x),3,3));p[:,:,2]=[3,2.5,2];p[:,:,0]=np.asarray(x)[:,None]
        v=np.zeros_like(p);v[:,-1]=[4,0,3]
        return p,v,np.arange(len(x))*.01,np.array([.5,.5]),np.array([1.1,0,2])

    def score(self,p,v,t,rest,center):return score_trace(p,v,t,rest,center,1.,.6)

    def test_passage_contact_conventions(self):
        for x,expected in [([-1,1],[(0,1,.5,1)]),([-1,0,0,1],[(2,2,0.,3)]),
                           ([-1,0,-1],[]),([-1,0],[]),([0,1],[]),([1,-1],[])]:
            self.assertEqual(list(confirmed_passages(x)),expected)
        self.assertEqual(list(confirmed_passages([-1,0,-1,1])),[(2,3,.5,3)])

    def test_crossing_credit_waits_for_confirmation_and_matches_same_state(self):
        a=self.trace([1.,1.2]);r=self.score(*a)
        self.assertEqual(r['causal_best'][0],0)
        c=r['crossings'][0]
        # Vertical collinear geometry: weights[1/4,1/2,1/4] give fitted tip X=8/3: (1/4*4)/(1/2*.5**2+1/4)=8/3.
        expected_shape=np.hypot(4/3,3.)
        self.assertAlmostEqual(c['shape_speed_mps'],expected_shape,places=12)
        self.assertAlmostEqual(c['quality'],expected_shape,places=12)
        self.assertAlmostEqual(r['crossing_max'],5*expected_shape/(1+expected_shape),places=12)
        self.assertTrue(c['quality_hit']);self.assertEqual(c['point_world_m'],[1.1,0,2])

    def test_zero_plateau_uses_departure_geometry(self):
        a=list(self.trace([1.,1.1,1.1,1.2]));a[0][2:,:,1]=2
        r=self.score(*a);self.assertFalse(r['crossings'][0]['geometric_hit'])
        self.assertEqual(r['crossing_max'],0)
        self.assertEqual(r['crossings'][0]['left_index'],2)

    def test_rigid_translation_and_rotation_receive_no_delivery(self):
        a=list(self.trace([1.,1.2]))
        for rotation in (False,True):
            a[1][:]=[4,0,0]
            if rotation:a[1][:]+=np.cross([0,1,0],a[0]-a[0][:,:1])
            r=self.score(*a)
            self.assertTrue(r['crossings'][0]['geometric_hit'])
            self.assertLess(r['crossing_max'],1e-12)
            self.assertFalse(r['crossings'][0]['quality_hit'])

    def test_velocity_interpolated_before_quality_not_peak_or_endpoint_speed(self):
        a=list(self.trace([1.,1.2]));a[1][0,-1]=[8,0,6];a[1][1,-1]=[0,0,-6]
        r=self.score(*a);c=r['crossings'][0]
        np.testing.assert_allclose(c['velocity_world_mps'],[4,0,0],atol=1e-12)
        self.assertAlmostEqual(c['quality'],4/3,places=12)

    def test_nonpositive_forward_velocity_cannot_qualify(self):
        a=list(self.trace([1.,1.2]));a[1][:,-1,0]=-1
        r=self.score(*a);self.assertEqual(r['crossing_max'],0)
        self.assertFalse(r['crossings'][0]['geometric_hit'])

    def test_maximum_not_sum_and_prefix_causality(self):
        a=self.trace([1.,1.2,1.,1.2]);r=self.score(*a)
        self.assertEqual(len(r['crossings']),2)
        self.assertAlmostEqual(r['best_event_score'],r['crossings'][0]['utility'])
        for n in range(1,5):
            prefix=self.score(a[0][:n],a[1][:n],a[2][:n],a[3],a[4])
            np.testing.assert_array_equal(prefix['causal_best'],r['causal_best'][:n])


class DeliveryIntegrationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)
        cls.s,cls.snap,cls.cfg,cls.scene=load(PRESET)

    def env(self):
        e=Stage1Environment(dict(self.s,device='cpu',parallel_environments=1),self.snap,self.cfg,self.scene,capture=False)
        z=2.+torch.flip(torch.cat((torch.zeros_like(e.model.rest[:1]),e.model.rest.cumsum(0))),[0])
        e.history['cable_positions'][:]=0.;e.history['cable_positions'][...,2]=z
        e.history['cable_positions'][...,0]=torch.linspace(1.,1.2,e.samples)[None,:,None]
        e.history['cable_velocities'][:]=0.;e.history['cable_velocities'][:,:,-1]=e.time.new_tensor([4,0,3])
        e.history['drone_positions'][:]=e.origin;e.history['drone_velocities'][:]=0
        e.numerical_valid[:]=True;e.failed[:]=e.active[:]=False
        e.decision=e.decisions;e.lengths[:]=e.decisions
        return e

    def test_accounting_cost_failure_and_recording(self):
        e=self.env();e.history['drone_positions'][:,1:,2]=3.1
        r,valid,s,_=e.finish();self.assertTrue(valid[0]);self.assertEqual(s.components['quality_hit_success'][0],1)
        c=2.5;u=float(s.causal_best[0,-1]);x=u-c;expected=x if x>=0 else 5*x/(5-x)
        self.assertAlmostEqual(float(r[0]),expected,places=6)
        self.assertAlmostEqual(float(s.components['workspace_cost'][0]),c,places=12)
        self.assertEqual(float(s.components['movement_cost'][0]),0)
        torch.testing.assert_close(e.transition_rewards.sum(1),r)
        e.failed[:]=True;r,valid,s,_=e.finish()
        self.assertAlmostEqual(float(r[0]),-10-5*c/(5+c),places=6)
        self.assertFalse(valid[0]);self.assertGreater(s.causal_best[0,-1],0)
        with project_temporary() as folder:
            record_simple_collection(folder,e,r,valid,s,1)
            row=json.loads((Path(folder)/'collection-00001-events.json').read_text())['lanes'][0]
            self.assertIn('raw_net_delivery_return',row)
            self.assertEqual(row['episode_return'],float(r[0]))
            with np.load(Path(folder)/'collection-00001.npz') as data:
                self.assertFalse(data['quality_hit'][0]);self.assertFalse(data['geometric_hit'][0])

    def test_partial_failure_ignores_future_hit_and_future_workspace_cost(self):
        e=self.env();e.numerical_valid[:,101:]=False;e.failed[:]=True;e.lengths[:]=50
        e.history['drone_positions'][:,1:101,2]=3.1;e.history['drone_positions'][:,101:,2]=100
        r,valid,s,_=e.finish();c=2.5*100/450
        self.assertAlmostEqual(float(r[0]),-10-5*c/(5+c),places=6)
        self.assertEqual(e.scorer.event_records[0]['crossings'],[])
        self.assertTrue(bool((e.transition_rewards[:,50:]==0).all()))

    def test_registration_rejects_unapproved_task_changes(self):
        for update in ({'training_stage':1},{'stage2_return_transform':None},{'training_episodes':16000},
                       {'workspace_penalty':dict(mode='squared_excess_v1',weight=0,distance_scale_m=.1)}):
            with self.subTest(update=update),self.assertRaises(ValueError):load(dict(self.s,**update))
        raw=deepcopy(self.snap['task']);raw['strike_surface']['distance_m']=.6
        with self.assertRaises(ValueError):task_configuration(raw,self.s,self.snap['model'],self.scene)
        self.assertEqual(self.snap['reward_delivery'],'causal_transformed_net_differences_v1')


if __name__=='__main__':unittest.main()
