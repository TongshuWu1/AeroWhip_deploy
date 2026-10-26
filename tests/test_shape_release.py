"""Approved utility/accounting integration; no physical-whip validation claim."""
from copy import deepcopy
from dataclasses import replace
import unittest
import numpy as np
import torch

from test_stage1_candidate import measurements
from helpers import wave, project_temporary
from tasks.whip.shape_release import score_events
from tasks.whip.learned_recovery import make_scorer
from tasks.whip.sac_environment import Stage1Environment
from training.configuration import load
from training.worker import record_simple_collection

PRESET='configs/learning/whip_stage1_ppo_shape_release_ar1_v1.json'


def decomposition(m):
    u=np.zeros((*m['relative_speed_mps'].shape,3));u[...,0]=.1;u[15,-1,0]=2.
    return dict(shape_velocity_world_mps=u,shape_speed_mps=np.linalg.norm(u,axis=-1))


class ShapeReleaseTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)
        cls.settings,cls.snapshot,cls.cfg,cls.scene=load(PRESET)
        cls.rest=torch.tensor(cls.snapshot['model']['cable']['marker_interval_lengths_m'],dtype=torch.float64)
        cls.time=torch.arange(451,dtype=torch.float64)*.01
        cls.p,cls.v=wave(cls.rest,cls.time)

    def test_shape_peak_and_gain_replace_original_output(self):
        m=measurements();d=decomposition(m)
        # Original output loses its gain and peaks at a DIFFERENT sample.
        m['relative_speed_mps'][0,-1]=100
        m['world_forward_mps'][12,-1]=-5
        m['world_forward_mps'][15,-1]=1.5
        r=score_events(m,d,[1,0,0],complete=True,failed=False)
        e=r['events'][0]
        self.assertEqual(e['q_shape_index'],15)
        self.assertEqual(e['score'],1.5)
        self.assertEqual(np.flatnonzero(np.diff(r['causal_best'])).tolist(),[25])
        self.assertAlmostEqual(e['shape_gain_mps'],1.9)
        self.assertEqual(r['mechanism_status'],'unresolved')

    def test_earliest_shape_tie_and_both_same_sample_directions(self):
        for which in ('world','shape'):
            m=measurements();d=decomposition(m)
            d['shape_speed_mps'][17,-1]=2
            d['shape_velocity_world_mps'][17,-1,0]=2
            if which=='world':m['world_forward_mps'][15,-1]=-1
            else:d['shape_velocity_world_mps'][15,-1,0]=-2
            r=score_events(m,d,[1,0,0],complete=True,failed=False)
            self.assertEqual(r['events'][0]['q_shape_index'],15)
            self.assertEqual(r['best_event_score'],0)

    def test_shape_gain_and_original_free_fit_gate_are_independent(self):
        m=measurements();d=decomposition(m);d['shape_speed_mps'][0,-1]=2
        self.assertEqual(score_events(m,d,[1,0,0],complete=True,failed=False)['best_event_score'],0)
        m['nonrigid_speed_mps'][:]=0;d=decomposition(m)
        self.assertEqual(score_events(m,d,[1,0,0],complete=True,failed=False)['events'],[])

    def test_full_finite_failure_cannot_clip_unfinished_window(self):
        m={k:v[:20] if k!='node_s_m' else v for k,v in measurements().items()};d=decomposition(m)
        good=score_events(m,d,[1,0,0],complete=True,failed=False)
        bad=score_events(m,d,[1,0,0],complete=True,failed=True)
        partial=score_events(m,d,[1,0,0],complete=False,failed=False)
        self.assertTrue(good['events'][0]['truncated']);self.assertEqual(good['best_event_score'],1)
        self.assertFalse(bad['events'][0]['finalized']);self.assertEqual(bad['episode_return'],-10)
        self.assertFalse(partial['events'][0]['finalized']);self.assertIsNone(partial['episode_return'])

    def test_repeated_events_pay_only_record_and_failed_prefix_preserved(self):
        m=measurements();m['curvature_per_m'][28:]=m['curvature_per_m'][:23];d=decomposition(m)
        d['shape_speed_mps'][40,-1]=2;d['shape_velocity_world_mps'][40,-1,0]=2
        good=score_events(m,d,[1,0,0],complete=True,failed=False)
        bad=score_events(m,d,[1,0,0],complete=False,failed=True)
        self.assertEqual(good['best_event_score'],1)
        self.assertEqual(bad['causal_best'][-1],1);self.assertEqual(bad['episode_return'],-10)

    def environment(self):
        env=Stage1Environment(dict(self.settings,device='cpu',parallel_environments=1),self.snapshot,self.cfg,self.scene,capture=False)
        env.history['cable_positions'].copy_(self.p);env.history['cable_velocities'].copy_(self.v)
        env.numerical_valid[:]=True;env.failed[:]=False;env.active[:]=False
        env.decision=env.decisions;env.lengths[:]=env.decisions
        return env

    def test_runtime_rewards_sum_and_failure_claws_back_positive_credit(self):
        env=self.environment();ret,valid,score,_=env.finish()
        self.assertTrue(bool(valid[0]));self.assertGreater(float(ret[0]),0)
        torch.testing.assert_close(env.transition_rewards.sum(1),ret)
        self.assertEqual(float(score.components['workspace_cost'][0]),0)
        expected=score.causal_best[:,::2].diff(1)
        torch.testing.assert_close(env.transition_rewards,expected.float())
        env.failed[:]=True;env.lengths[:]=150;env.numerical_valid[:,301:]=False
        ret,valid,score,_=env.finish()
        self.assertFalse(bool(valid[0]));self.assertEqual(float(ret[0]),-10)
        self.assertGreater(float(score.causal_best[:,:300].max()),0)
        self.assertLess(float(env.transition_rewards[0,149]),-10)
        self.assertEqual(int(env.transition_rewards[0,150:].count_nonzero()),0)
        torch.testing.assert_close(env.transition_rewards.sum(1),ret)

    def test_target_geometry_diagnostic_only_and_event_records_retained(self):
        env=self.environment();ret,valid,score,_=env.finish()
        alternate=replace(self.cfg,strike_surface=replace(self.cfg.strike_surface,distance_m=100,width_m=.01))
        scorer=make_scorer(self.rest,self.time,alternate,initial_position=env.origin)
        other=scorer(self.p,self.v,env.numerical_valid,failed=env.failed)
        torch.testing.assert_close(score.score,other.score,rtol=0,atol=0)
        with project_temporary() as folder:
            from pathlib import Path
            import json
            record_simple_collection(folder,env,ret,valid,score,1)
            events=json.loads((Path(folder)/'collection-00001-events.json').read_text())
            self.assertEqual(events['lanes'][0]['schema'],'directed_shape_release_v1')

    def test_configuration_refuses_costs_old_stopping_and_other_stage(self):
        for key,value in (('workspace_penalty',dict(weight=.1,distance_scale_m=.1)),('reward_scale',100),
                          ('training_stage',2),('plateau_detection',load('configs/learning/whip_stage1_ppo_acceleration_discovery_v1.json')[0]['plateau_detection'])):
            settings=deepcopy(self.settings);settings[key]=value
            with self.subTest(key=key),self.assertRaises(ValueError):load(settings)
        old,_,_,_=load('configs/learning/whip_stage1_ppo_acceleration_discovery_v1.json')
        self.assertNotIn('action_noise',old['ppo']);self.assertEqual(old['reward_scale'],100)
