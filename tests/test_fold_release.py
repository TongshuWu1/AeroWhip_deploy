"""Fold-goal kinematic and accounting checks; not physical ground truth."""
from pathlib import Path
from copy import deepcopy
import unittest
import numpy as np

from test_whip_candidate_review import outward_packet
from evaluation.whip_measurement_examples import kinematic_controls
from evaluation.whip_measurements import extract_measurements
from evaluation.whip_output_decomposition import decompose
from tasks.whip.fold_release import score_events,fold_geometry

ROOT=Path(__file__).resolve().parents[1]


def analytic_fold(rest, *, reverse=False):
    # A larger geometric fold than the earlier small packet, oriented so its
    # opening output goes +X. Neither construction is a dynamic rollout.
    a=outward_packet(rest,amplitude=-1.,reverse=reverse)
    R=np.array([[0.,0.,-1.],[0.,1.,0.],[1.,0.,0.]])
    return dict(a,cable_positions=(a['cable_positions']-[0,0,2])@R.T+[0,0,2],
                cable_velocities=a['cable_velocities']@R.T)


class FoldReleaseTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        with np.load(ROOT/'evaluation/fixtures/stage1_side_swing_v1.npz') as z:
            cls.negative={k:z[k] for k in z.files}
        cls.rest=cls.negative['rest_lengths_m'];cls.direction=np.array([1.,0.,0.])

    def score(self,a,*,direction=None,complete=True,failed=False):
        direction=self.direction if direction is None else direction
        m=extract_measurements(a,self.rest,direction)
        d=decompose(a['cable_positions'],a['cable_velocities'],self.rest)
        return score_events(a,m,d,direction,complete=complete,failed=failed)

    def test_recorded_rejected_motion_receives_zero(self):
        r=self.score(self.negative)
        self.assertEqual(r['best_event_score'],0.)
        self.assertEqual(r['mechanism_status'],'unresolved')

    def test_outward_fold_opening_and_same_sample_output(self):
        a=analytic_fold(self.rest);r=self.score(a)
        self.assertGreater(r['best_event_score'],0.)
        e=r['events'][r['selected_event']]
        self.assertTrue(e['opening_verified'])
        self.assertLessEqual(e['fold_tail_turn_after_rad'],.5*e['fold_tail_turn_at_arrival_rad'])
        self.assertEqual(e['evidence_node_indices'][-1],len(self.rest)-1)
        self.assertGreaterEqual(len(set(e['evidence_node_indices'])),5)
        self.assertAlmostEqual(e['score'],min(e['shape_gain_mps'],e['shape_forward_mps'],e['world_forward_mps']))
        self.assertEqual(np.flatnonzero(np.diff(r['causal_best'])).tolist(),[e['end_index']-1])

    def test_backward_small_rigid_and_unrelated_sweep_do_not_score(self):
        cases=[analytic_fold(self.rest,reverse=True),outward_packet(self.rest)]
        cases.extend(c['arrays'] for c in kinematic_controls(self.rest))
        for i,a in enumerate(cases):
            with self.subTest(i=i):self.assertEqual(self.score(a)['best_event_score'],0.)

    def test_fold_must_open_and_full_window_is_required(self):
        a=analytic_fold(self.rest);r=self.score(a);e=r['events'][0];arm=e['arming_index']
        held=deepcopy(a);held['cable_positions'][arm+1:]=held['cable_positions'][arm];held['cable_velocities'][arm+1:]=0
        h=self.score(held)
        self.assertEqual(h['best_event_score'],0.)
        self.assertEqual(h['events'][0]['rejection_reason'],'fold_did_not_open')
        for complete in (True,False):
            clipped={k:v[:arm+10] for k,v in a.items()}
            c=self.score(clipped,complete=complete)
            self.assertEqual(c['best_event_score'],0.)
            self.assertFalse(c['events'][0]['finalized'])

    def test_distal_only_fold_is_not_root_to_tip_transfer(self):
        a=analytic_fold(self.rest);cut=40
        a={k:v[cut:].copy() for k,v in a.items()};a['time_s']-=a['time_s'][0]
        self.assertEqual(self.score(a)['best_event_score'],0.)

    def test_failure_keeps_evidence_but_entire_return_is_negative_ten(self):
        r=self.score(analytic_fold(self.rest),failed=True,complete=False)
        self.assertGreater(r['best_event_score'],0.)
        self.assertEqual(r['episode_return'],-10.)

    def test_origin_and_horizontal_corotation_covariance(self):
        a=analytic_fold(self.rest);r=self.score(a)
        angle=.71;R=np.array([[np.cos(angle),-np.sin(angle),0],[np.sin(angle),np.cos(angle),0],[0,0,1]])
        b=dict(a,cable_positions=a['cable_positions']@R.T+[3,-4,7],cable_velocities=a['cable_velocities']@R.T)
        s=self.score(b,direction=self.direction@R.T)
        self.assertAlmostEqual(r['best_event_score'],s['best_event_score'],places=10)
        # A lateral translation cannot become rewarded forward output.
        b=dict(a,cable_positions=a['cable_positions']+np.array([0,10,0])[None,None]*a['time_s'][:,None,None],
               cable_velocities=a['cable_velocities']+[0,10,0])
        self.assertAlmostEqual(r['best_event_score'],self.score(b)['best_event_score'],places=10)

    def test_invalid_geometry_and_sample_period_refused(self):
        with self.assertRaises(ValueError):fold_geometry(np.zeros((3,11,3)))
        a=analytic_fold(self.rest);a['time_s']*=2
        with self.assertRaises(ValueError):self.score(a)

    def test_batch_adapter_preserves_path_cost_and_failure_accounting(self):
        # Explicit test-only scorer substitution; no new reward mode is enabled.
        import torch
        from training.configuration import load
        from tasks.whip.sac_environment import Stage1Environment
        from tasks.whip.fold_release import FoldReleaseScorer
        torch.set_num_threads(2)
        settings,snapshot,cfg,scene=load('configs/learning/whip_stage1_ppo_shape_release_path_ar1_v3.json')
        env=Stage1Environment(dict(settings,device='cpu',parallel_environments=1),snapshot,cfg,scene,capture=False)
        env.scorer=FoldReleaseScorer(env.model.rest,env.time,cfg,initial_position=env.origin)
        a=analytic_fold(self.rest)
        env.history['cable_positions'][0].copy_(torch.from_numpy(a['cable_positions']))
        env.history['cable_velocities'][0].copy_(torch.from_numpy(a['cable_velocities']))
        env.history['drone_positions'][:]=env.origin
        env.history['drone_positions'][0,:,1]+=torch.arange(451,dtype=torch.float64)*.001
        env.numerical_valid[:]=True;env.failed[:]=False;env.active[:]=False
        env.decision=env.decisions;env.lengths[:]=env.decisions
        ret,valid,score,_=env.finish()
        self.assertTrue(bool(valid[0]));self.assertGreater(float(score.score[0]),0.)
        self.assertAlmostEqual(float(score.components['movement_accrued_cost'][0]),.25*.45/sum(self.rest),places=10)
        torch.testing.assert_close(env.transition_rewards.sum(1),ret)
        self.assertAlmostEqual(float(ret[0]),float(score.score[0]-score.components['movement_cost'][0]),places=6)
        env.failed[:]=True
        ret,valid,score,_=env.finish()
        self.assertFalse(bool(valid[0]));self.assertEqual(float(ret[0]),-10.)
        self.assertEqual(float(score.components['movement_cost'][0]),0.)
        self.assertGreater(float(score.causal_best.max()),0.)
        torch.testing.assert_close(env.transition_rewards.sum(1),ret)
