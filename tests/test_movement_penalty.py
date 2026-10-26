"""Travel-charge mathematics and full-horizon reward/failure integration."""
from copy import deepcopy
import unittest
import numpy as np
import torch
from tasks.whip.movement import MovementPenalty
from training.configuration import load
from test_shape_release import ShapeReleaseTests, PRESET

NEW='configs/learning/whip_stage1_ppo_shape_release_path_ar1_v2.json'


class MovementTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        ShapeReleaseTests.setUpClass()
        for name in ('settings','snapshot','cfg','scene','rest','time','p','v'):
            setattr(cls,name,getattr(ShapeReleaseTests,name))
        cls.penalty=MovementPenalty('path_length_v1',.05)

    def environment(self):return ShapeReleaseTests.environment(self)

    def test_round_trip_cost_not_endpoint_displacement(self):
        p=torch.tensor([[[0.,0,0],[1.,0,0],[0.,0,0]]],dtype=torch.float64)
        d,c=self.penalty.measure(p,torch.ones(1,3,dtype=torch.bool),1.)
        self.assertEqual(float(d.sum()),2);self.assertEqual(float(c.sum()),.1)
        shifted=p+torch.tensor([5.,2.,3.]);d2,c2=self.penalty.measure(shifted,torch.ones(1,3,dtype=torch.bool),1.)
        torch.testing.assert_close(c,c2,rtol=0,atol=0)

    def test_prefix_nan_suffix_and_no_gap_bridging(self):
        p=torch.tensor([[[0.,0,0],[1.,0,0],[float('nan'),0,0],[100.,0,0]]],dtype=torch.float64)
        mask=torch.tensor([[True,True,False,True]])
        d,c=self.penalty.measure(p,mask,2.)
        self.assertEqual(d.tolist(),[[1.,0.,0.]]);self.assertEqual(float(c.sum()),.025)
        mask[:]=True
        with self.assertRaises(ValueError):self.penalty.measure(p,mask,2.)

    def test_normalization_no_time_average_or_path_cap(self):
        p=torch.zeros(1,5,3,dtype=torch.float64);p[0,:,0]=torch.arange(5)*10
        d,c=self.penalty.measure(p,torch.ones(1,5,dtype=torch.bool),2.)
        self.assertEqual(float(d.sum()),40);self.assertEqual(float(c.sum()),1)
        _,other=self.penalty.measure(p[:,::2],torch.ones(1,3,dtype=torch.bool),2.)
        torch.testing.assert_close(c.sum(),other.sum(),rtol=0,atol=0)

    def test_raw_event_unchanged_and_dense_credit_exact(self):
        env=self.environment()
        env.history['drone_positions'][:]=env.origin
        env.history['drone_positions'][0,:,0]=torch.arange(451)*.02
        baseline,_,original,_=env.finish();rewards=env.transition_rewards.clone()
        env.settings=dict(env.settings,movement_penalty=dict(mode='path_length_v1',weight_per_cable_length=.05))
        returns,valid,score,_=env.finish()
        expected=.05*9/float(env.model.rest.sum())
        self.assertTrue(bool(valid[0]))
        torch.testing.assert_close(score.score,original.score,rtol=0,atol=0)
        self.assertAlmostEqual(float(score.components['movement_cost'][0]),expected,places=7)
        torch.testing.assert_close(returns,(original.score-expected).float())
        torch.testing.assert_close(env.transition_rewards.sum(1),returns)
        self.assertLess(float(env.transition_rewards[0,-1]),float(rewards[0,-1]))
        self.assertAlmostEqual(float(score.components['drone_path_length_m'][0]),9,places=7)

    def test_cost_after_release_is_not_free(self):
        env=self.environment();env.finish()
        env.settings=dict(env.settings,movement_penalty=dict(mode='path_length_v1',weight_per_cable_length=.05))
        p=env.history['drone_positions'];p[:]=env.origin
        base,_,original,_=env.finish()
        self.assertGreater(float(original.causal_best[0,300]),0)
        p[0,351:,0]=torch.arange(1,101,dtype=torch.float64)*.02
        out,_,score,_=env.finish()
        torch.testing.assert_close(score.score,original.score,rtol=0,atol=0)
        self.assertAlmostEqual(float((base-out)[0]),.05*2/float(env.model.rest.sum()),places=6)

    def test_failure_replaces_accrued_travel_and_claws_back(self):
        env=self.environment();env.settings=dict(env.settings,movement_penalty=dict(mode='path_length_v1',weight_per_cable_length=.05))
        env.history['drone_positions'][0,:,0]=torch.arange(451)*.02
        env.failed[:]=True;env.lengths[:]=150;env.numerical_valid[:,301:]=False
        out,valid,score,_=env.finish()
        self.assertFalse(bool(valid[0]));self.assertEqual(float(out[0]),-10)
        self.assertGreater(float(score.components['movement_accrued_cost'][0]),0)
        self.assertEqual(float(score.components['movement_cost'][0]),0)
        torch.testing.assert_close(env.transition_rewards.sum(1),out)
        self.assertEqual(int(env.transition_rewards[0,150:].count_nonzero()),0)

    def test_hover_zero_and_no_event_travel_negative(self):
        env=self.environment();env.settings=dict(env.settings,movement_penalty=dict(mode='path_length_v1',weight_per_cable_length=.05))
        env.history['cable_positions'][:]=self.p[:,:1];env.history['cable_velocities'].zero_()
        env.history['drone_positions'][:]=env.origin
        out,_,_,_=env.finish();self.assertEqual(float(out[0]),0)
        env.history['drone_positions'][0,:,0]=torch.arange(451)*.01
        out,_,score,_=env.finish();self.assertLess(float(out[0]),0)
        self.assertEqual(float(score.score[0]),0)

    def test_separate_preset_and_invalid_settings(self):
        new,_,_,_=load(NEW);old=load(PRESET)[0]
        self.assertEqual(new['training_episodes'],15360)
        self.assertEqual(new['movement_penalty'],dict(mode='path_length_v1',weight_per_cable_length=.05))
        for k in old:
            if k!='training_episodes':self.assertEqual(new[k],old[k])
        for value in (-.1,0,float('nan'),True):
            s=deepcopy(new);s['movement_penalty']['weight_per_cable_length']=value
            with self.assertRaises(ValueError):load(s)
        s=deepcopy(new);s['task']='configs/tasks/whip_stage1_discovery_bounded_v1.json'
        with self.assertRaises(ValueError):load(s)

    def test_recorded_event_return_includes_cost_without_overwriting_core(self):
        import json
        from pathlib import Path
        from helpers import project_temporary
        from training.worker import record_simple_collection
        env=self.environment()
        env.settings=dict(env.settings,movement_penalty=dict(mode='path_length_v1',weight_per_cable_length=.05))
        env.history['drone_positions'][:]=env.origin
        env.history['drone_positions'][0,:,0]=torch.arange(451)*.01
        returns,valid,score,_=env.finish()
        original=deepcopy(env.scorer.event_records)
        with project_temporary() as folder:
            record_simple_collection(folder,env,returns,valid,score,1)
            row=json.loads((Path(folder)/'collection-00001-events.json').read_text())['lanes'][0]
            self.assertEqual(row['episode_return'],float(returns[0]))
            self.assertEqual(row['shape_only_episode_return'],original[0]['episode_return'])
            self.assertGreater(row['movement_cost'],0)
        self.assertEqual(env.scorer.event_records,original)
