"""Approved net-return transformation: analytic histories, not policy evidence."""
from copy import deepcopy
import json
from pathlib import Path
import unittest

import numpy as np
import torch

from evaluation.whip_measurement_examples import kinematic_controls
from tasks.whip.excitation import negative_net_rational
from tasks.whip.sac_environment import Stage1Environment
from training.configuration import load
from training.worker import record_simple_collection
from helpers import project_temporary

PRESET = 'configs/learning/whip_stage1_ppo_flexible_excitation_rational_ar1_v3.json'
OLD = 'configs/learning/whip_stage1_ppo_flexible_excitation_ar1_v1.json'


def reference(x):
    x = np.asarray(x, dtype=np.float64)
    return np.where(x >= 0, x, 5*x/(5-np.minimum(x, 0)))


class RationalExcitationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)
        cls.settings, cls.snapshot, cls.cfg, cls.scene = load(PRESET)
        rest = np.asarray(cls.snapshot['model']['cable']['marker_interval_lengths_m'])
        cls.arrays = next(c['arrays'] for c in kinematic_controls(rest)
                          if c['name'] == 'small_travelling_wave')

    def env(self):
        env = Stage1Environment(dict(self.settings, device='cpu', parallel_environments=1),
                                self.snapshot, self.cfg, self.scene, capture=False)
        for key in ('cable_positions', 'cable_velocities'):
            env.history[key][0].copy_(torch.from_numpy(self.arrays[key]))
        env.history['drone_positions'][:] = env.origin
        env.history['drone_velocities'][:] = 0.
        env.numerical_valid[:] = True
        env.failed[:] = env.active[:] = False
        env.decision = env.decisions
        env.lengths[:] = env.decisions
        return env

    def test_preset_changes_only_return_map_from_uncapped_original(self):
        old, *_ = load(OLD)
        new = deepcopy(self.settings)
        self.assertEqual(new.pop('stage1_return_transform'), 'negative_net_rational_v1')
        self.assertEqual(old, new)
        self.assertEqual(self.snapshot['reward_delivery'], 'causal_transformed_net_differences_v1')
        for update in ({'stage1_return_transform': 'other'},
                       {'stage1_return_transform': None},
                       {'workspace_penalty':dict(mode='bounded_density_v1',weight=5.,distance_scale_m=2**-.5)},
                       {'task':'configs/tasks/whip_stage1_unified_fold_v1.json'}):
            with self.subTest(update=update), self.assertRaises(ValueError):
                load(dict(self.settings, **update))

    def test_transform_monotone_identity_and_no_positive_branch_singularity(self):
        x = torch.tensor([-1000.,-100.,-10.,-5.,-1.,0.,1.,5.,10.,100.],dtype=torch.float64)
        y = negative_net_rational(x)
        self.assertTrue(bool(torch.isfinite(y).all()))
        self.assertTrue(bool((y.diff()>0).all()))
        self.assertTrue(bool((y>-5).all()))
        np.testing.assert_allclose(y.numpy(), reference(x.numpy()), rtol=0, atol=1e-12)
        torch.testing.assert_close(y[x>=0],x[x>=0],rtol=0,atol=0)

    def test_contained_excitation_unchanged_and_hover_zero(self):
        env = self.env()
        env.history['drone_positions'][0,:,0] += .4*torch.sin(env.time*4)
        env.history['drone_velocities'][0,:,0] = 20*torch.sin(env.time*8)
        r,valid,s,effort=env.finish()
        self.assertTrue(bool(valid[0]));self.assertGreater(float(s.score[0]),0)
        torch.testing.assert_close(r,s.score.float(),rtol=0,atol=0)
        for key in ('workspace_cost','movement_cost','return_compression_adjustment'):
            self.assertEqual(float(s.components[key][0]),0.)
        self.assertEqual(float(effort[0]),0.)
        env.history['cable_velocities'].zero_()
        self.assertEqual(float(env.finish()[0][0]),0.)

    def test_farther_height_increases_cost_and_reduces_valid_return(self):
        results=[]
        for excess in [.5,1.,2.,5.,10.]:
            env=self.env();env.history['drone_positions'][0,1:,2]=2.6+excess
            r,valid,s,_=env.finish();self.assertTrue(bool(valid[0]))
            cost=10*excess**2
            self.assertAlmostEqual(float(s.components['workspace_cost'][0]),cost,places=10)
            expected=np.float32(reference(float(s.score[0])-cost))
            self.assertEqual(float(r[0]),float(expected))
            self.assertGreater(float(r[0]),-5.)
            results.append(float(r[0]))
        self.assertTrue(all(a>b for a,b in zip(results,results[1:])))

    def test_causal_rewards_are_transformed_net_differences(self):
        env=self.env()
        env.history['drone_positions'][0,:,2]=2+2*env.time
        r,valid,s,_=env.finish()
        p=env.history['drone_positions'][0].numpy()
        c=np.r_[0,np.cumsum(10*np.maximum(p[1:,2]-2.6,0)**2/450)]
        potential=reference(s.causal_best[0].numpy()-c)[::env.stride]
        expected=np.diff(potential).astype(np.float32)
        np.testing.assert_allclose(env.transition_rewards[0].numpy(),expected,rtol=0,atol=1e-7)
        self.assertAlmostEqual(float(env.transition_rewards.double().sum()),float(potential[-1]),places=6)
        self.assertAlmostEqual(float(s.components['net_episode_return'][0]),float(potential[-1]),places=12)

    def test_failed_prefix_no_suffix_charge_and_full_excitation_clawback(self):
        env=self.env();env.history['drone_positions'][0,1:,2]=3.6
        env.finish();before=env.transition_rewards.clone()
        env.failed[:]=True;env.lengths[:]=50
        env.numerical_valid[:,101:]=False
        env.history['drone_positions'][0,101:]=float('nan')
        r,valid,s,_=env.finish()
        self.assertFalse(bool(valid[0]))
        cost=10*100/450
        expected=-10+float(reference(-cost))
        self.assertAlmostEqual(float(s.components['workspace_accrued_cost'][0]),cost,places=12)
        self.assertEqual(float(r[0]),float(np.float32(expected)))
        self.assertGreater(float(s.causal_best[0,100]),0.)
        torch.testing.assert_close(env.transition_rewards[:,:49],before[:,:49],rtol=0,atol=0)
        self.assertEqual(int(env.transition_rewards[:,50:].count_nonzero()),0)
        self.assertTrue(bool(torch.isfinite(env.transition_rewards).all()))
        # Compare to the independent real-valued objective, not a second
        # independently rounded float32 quantity. Terminal cast is checked above.
        self.assertAlmostEqual(float(env.transition_rewards.double().sum()),expected,places=6)

    def test_severe_valid_outcome_always_outranks_cheap_failure(self):
        env=self.env();env.history['cable_velocities'].zero_()
        env.history['drone_positions'][0,1:,2]=100.
        valid_return=env.finish()[0]
        env.history['drone_positions'][:]=env.origin
        env.failed[:]=True;env.numerical_valid[:,3:]=False;env.lengths[:]=1
        failed_return=env.finish()[0]
        self.assertEqual(float(failed_return[0]),-10.)
        self.assertGreater(float(valid_return[0]),float(failed_return[0]))

    def test_recorder_separates_uncapped_cost_and_transformed_return(self):
        env=self.env();env.history['drone_positions'][0,1:,2]=4.6
        with project_temporary() as folder:
            folder=Path(folder)
            for index,failed in enumerate((False,True),1):
                env.failed[:]=failed
                r,valid,s,_=env.finish()
                record_simple_collection(folder,env,r,valid,s,index)
                row=json.loads((folder/f'collection-{index:05}-events.json').read_text())['lanes'][0]
                self.assertEqual(row['return_transform'],'negative_net_rational_v1')
                self.assertAlmostEqual(row['workspace_cost'],40.,places=12)
                self.assertEqual(row['episode_return'],float(r[0]))
                expected=returned=(-10+float(reference(-40.)) if failed else
                                  float(reference(row['best_event_score']-40.)))
                self.assertAlmostEqual(row['net_episode_return'],expected,places=12)
                with np.load(folder/f'collection-{index:05}.npz') as z:
                    self.assertAlmostEqual(float(z['component_workspace_cost'][0]),40.,places=12)
                    self.assertEqual(float(z['returns'][0]),float(r[0]))


if __name__=='__main__':
    unittest.main()
