"""Unified-attempt accounting and counterexamples; analytic motions are not physics validation."""
from copy import deepcopy
from pathlib import Path
import json, unittest
import numpy as np
import torch
from helpers import project_temporary
from test_fold_release import analytic_fold
from evaluation.whip_measurement_examples import kinematic_controls
from evaluation.whip_measurements import extract_measurements
from evaluation.whip_output_decomposition import decompose
from tasks.whip.unified_fold import score_events
from tasks.whip.sac_environment import Stage1Environment
from training.configuration import load,collection_limit
from training.worker import record_simple_collection

PRESET='configs/learning/whip_stage1_ppo_unified_fold_ar1_v1.json'
class UnifiedFoldTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)
        cls.settings,cls.snapshot,cls.cfg,cls.scene=load(PRESET)
        cls.rest=np.array(cls.snapshot['model']['cable']['marker_interval_lengths_m'])
    def score(self,a,complete=True,failed=False):
        m=extract_measurements(a,self.rest,[1,0,0]);d=decompose(a['cable_positions'],a['cable_velocities'],self.rest)
        return score_events(a,m,d,[1,0,0],complete=complete,failed=failed)
    def env(self):
        env=Stage1Environment(dict(self.settings,device='cpu',parallel_environments=1),self.snapshot,self.cfg,self.scene,capture=False)
        a=analytic_fold(self.rest)
        for key in ('cable_positions','cable_velocities'):env.history[key][0].copy_(torch.tensor(a[key]))
        env.history['drone_positions'][:]=env.origin;env.numerical_valid[:]=True;env.failed[:]=False;env.active[:]=False
        env.decision=env.decisions;env.lengths[:]=env.decisions
        return env
    def test_complete_release_uses_one_attempt_and_preparation_is_bounded(self):
        r=self.score(analytic_fold(self.rest));self.assertTrue(r['completed_aligned_release'])
        self.assertEqual(r['best_event_score'],max(e['score'] for e in r['events']))
        for e in r['events']:
            self.assertAlmostEqual(e['score'],.5*e['formation_fraction']+.5*e['propagation_fraction']+e['aligned_release_score'])
            self.assertLessEqual(.5*e['formation_fraction']+.5*e['propagation_fraction'],1.)
            if e['aligned_release_score']>0:
                old=e['release_evidence']
                self.assertEqual((e['segment_id'],e['baseline_index'],e['arming_index']),(old['segment_id'],old['baseline_index'],old['arming_index']))
                self.assertEqual(e['q_shape_index'],old['q_shape_index'])
                self.assertEqual(e['aligned_release_score'],old['score'])
        self.assertAlmostEqual(r['causal_best'][-1],r['best_event_score'])
    def test_lateral_backward_and_reversed_wave_never_complete(self):
        a=analytic_fold(self.rest)
        for R in (np.array([[0.,-1,0],[1,0,0],[0,0,1]]),np.diag([-1.,-1.,1.])):
            b=dict(a,cable_positions=a['cable_positions']@R.T,cable_velocities=a['cable_velocities']@R.T)
            r=self.score(b);self.assertFalse(r['completed_aligned_release']);self.assertLessEqual(r['best_event_score'],1.)
        self.assertFalse(self.score(analytic_fold(self.rest,reverse=True))['completed_aligned_release'])
    def test_rigid_controls_can_earn_only_formation(self):
        for control in kinematic_controls(self.rest):
            r=self.score(control['arrays'])
            self.assertFalse(r['completed_aligned_release'])
            self.assertLessEqual(r['best_event_score'],.5+1e-12)
    def test_prefix_scores_do_not_anticipate_future_release(self):
        a=analytic_fold(self.rest);full=self.score(a)
        for n in (51,151,251,350,450):
            b={k:v[:n] for k,v in a.items()};r=self.score(b,complete=False)
            np.testing.assert_allclose(r['causal_best'],full['causal_best'][:n],rtol=0,atol=1e-12)
    def test_contained_motion_free_and_outside_failure_does_not_refund_cost(self):
        env=self.env();env.history['drone_positions'][0,:,0]+=.4*torch.sin(torch.arange(451)*.04)
        ret,valid,s,_=env.finish();self.assertEqual(float(s.components['workspace_cost'][0]),0.)
        self.assertEqual(float(s.components['movement_cost'][0]),0.);self.assertGreater(float(s.components['drone_path_length_m'][0]),1.)
        torch.testing.assert_close(ret,env.transition_rewards.sum(1))
        env=self.env();env.history['drone_positions'][0,1:,2]=3.1
        ret,valid,s,_=env.finish();self.assertAlmostEqual(float(s.components['workspace_cost'][0]),2.5)
        self.assertAlmostEqual(float(ret[0]),float(s.score[0])-2.5,places=5)
        env.failed[:]=True;ret,valid,s,_=env.finish();self.assertEqual(float(ret[0]),-12.5)
        torch.testing.assert_close(ret,env.transition_rewards.sum(1))
        with project_temporary() as directory:
            record_simple_collection(Path(directory),env,ret,valid,s,1)
            e=json.loads((Path(directory)/'collection-00001-events.json').read_text())['lanes'][0]
            self.assertEqual(e['episode_return'],-12.5);self.assertEqual(e['event_only_episode_return'],-10.)
            self.assertEqual(e['workspace_cost'],float(s.components['workspace_cost'][0]));self.assertEqual(e['movement_cost'],0.)
    def test_failed_prefix_cost_uses_fixed_horizon_and_excludes_padding(self):
        env=self.env();env.history['drone_positions'][0,1:101,2]=3.1
        env.history['drone_positions'][0,101:,2]=100.;env.numerical_valid[:,101:]=False
        env.failed[:]=True;env.lengths[:]=50
        ret,valid,s,_=env.finish()
        self.assertAlmostEqual(float(s.components['workspace_cost'][0]),2.5*100/450)
        self.assertAlmostEqual(float(ret[0]),-10-2.5*100/450,places=5)
        torch.testing.assert_close(ret,env.transition_rewards.sum(1))
    def test_budget_and_cost_contract(self):
        self.assertEqual(collection_limit(self.settings),120);self.assertEqual(self.settings['training_episodes'],15360)
        self.assertNotIn('fold_curriculum',self.settings);self.assertNotIn('movement_penalty',self.settings)
        for key,value in [('training_episodes',3840),('workspace_penalty',dict(mode='squared_excess_v1',weight=.01,distance_scale_m=.1))]:
            bad=deepcopy(self.settings);bad[key]=value
            with self.assertRaises(ValueError):load(bad)

if __name__=='__main__':unittest.main()
