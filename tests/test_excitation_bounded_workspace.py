"""Approved bounded-cost restart accounting, using injected CPU histories.

These analytic histories isolate the task objective. They are not simulated
flights, evidence of policy learning, or physical-whip validation.
"""
from copy import deepcopy
import unittest

import numpy as np
import torch

from evaluation.whip_measurement_examples import kinematic_controls
from tasks.whip.sac_environment import Stage1Environment
from training.configuration import load


OLD_PRESET = 'configs/learning/whip_stage1_ppo_flexible_excitation_ar1_v1.json'
PRESET = 'configs/learning/whip_stage1_ppo_flexible_excitation_bounded_ar1_v2.json'


class BoundedExcitationWorkspaceTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)
        cls.settings, cls.snapshot, cls.cfg, cls.scene = load(PRESET)
        cls.old_settings, cls.old_snapshot, cls.old_cfg, cls.old_scene = load(OLD_PRESET)
        rest = np.asarray(cls.snapshot['model']['cable']['marker_interval_lengths_m'])
        cls.arrays = next(case['arrays'] for case in kinematic_controls(rest)
                          if case['name'] == 'small_travelling_wave')

    def environment(self, *, old=False):
        settings, snapshot, cfg, scene = (
            (self.old_settings, self.old_snapshot, self.old_cfg, self.old_scene)
            if old else (self.settings, self.snapshot, self.cfg, self.scene))
        env = Stage1Environment(dict(settings, device='cpu', parallel_environments=1),
                                snapshot, cfg, scene, capture=False)
        for key in ('cable_positions', 'cable_velocities'):
            env.history[key][0].copy_(torch.from_numpy(self.arrays[key]))
        env.history['drone_positions'][:] = env.origin
        env.history['drone_velocities'][:] = 0.
        env.numerical_valid[:] = True
        env.failed[:] = False
        env.active[:] = False
        env.decision = env.decisions
        env.lengths[:] = env.decisions
        return env

    def test_restart_changes_only_workspace_profile_and_preserves_old_preset(self):
        old, new = deepcopy(self.old_settings), deepcopy(self.settings)
        old_cost, new_cost = old.pop('workspace_penalty'), new.pop('workspace_penalty')
        self.assertEqual(old, new)
        self.assertEqual(old_cost, dict(mode='squared_excess_v1', weight=.1,
                                       distance_scale_m=.1))
        self.assertEqual(new_cost['mode'], 'bounded_density_v1')
        self.assertEqual(new_cost['weight'], 5.)
        self.assertAlmostEqual(new_cost['distance_scale_m'] ** 2, .5, places=15)
        for key in ('task', 'model', 'scene_text'):
            self.assertEqual(self.snapshot[key], self.old_snapshot[key])

    def test_contained_motion_remains_free_of_workspace_path_and_effort_cost(self):
        env = self.environment()
        env.history['drone_positions'][0, :, 0] += .4 * torch.sin(4 * env.time)
        # An independently injected velocity tests cost accounting only.
        env.history['drone_velocities'][0, :, 0] = 20 * torch.sin(8 * env.time)
        returns, valid, score, effort = env.finish()
        self.assertTrue(bool(valid[0]))
        self.assertGreater(float(score.components['drone_path_length_m'][0]), 1.)
        for field in ('workspace_cost', 'workspace_accrued_cost', 'movement_cost',
                      'movement_accrued_cost'):
            self.assertEqual(float(score.components[field][0]), 0.)
        self.assertEqual(float(effort[0]), 0.)
        torch.testing.assert_close(returns, score.score.float(), rtol=0, atol=0)

    def test_unsaturated_combined_excess_retains_original_quadratic_cost(self):
        outcomes = []
        for old in (False, True):
            env = self.environment(old=old)
            # Radial excess 0.3m and vertical excess 0.5m give density 3.4.
            env.history['drone_positions'][0, 1:, 0] = .85
            env.history['drone_positions'][0, 1:, 2] = 3.1
            returns, valid, score, _ = env.finish()
            self.assertTrue(bool(valid[0]))
            self.assertAlmostEqual(float(score.components['workspace_cost'][0]),
                                   3.4, places=12)
            outcomes.append((returns, score.score, env.transition_rewards))
        for new, old in zip(outcomes[0], outcomes[1]):
            torch.testing.assert_close(new, old, rtol=0, atol=1e-12)

    def test_saturated_full_episode_cost_and_failed_credit_clawback(self):
        env = self.environment()
        env.history['drone_positions'][0, 1:, 2] = 20.
        returns, valid, score, effort = env.finish()
        self.assertTrue(bool(valid[0]))
        self.assertGreater(float(score.score[0]), 0.)
        self.assertAlmostEqual(float(score.components['workspace_cost'][0]), 5., places=12)
        torch.testing.assert_close(returns, (score.score - 5.).float(), rtol=0, atol=0)
        self.assertGreaterEqual(float(returns[0]), -5.)
        self.assertEqual(float(effort[0]), 0.)
        env.failed[:] = True
        returns, valid, score, _ = env.finish()
        self.assertFalse(bool(valid[0]))
        self.assertGreater(float(score.causal_best[0, -1]), 0.)
        self.assertAlmostEqual(float(score.components['workspace_accrued_cost'][0]),
                               5., places=12)
        self.assertEqual(float(returns[0]), -15.)

    def test_failed_prefix_keeps_accrued_cost_and_ignores_unexecuted_suffix(self):
        env = self.environment()
        env.history['drone_positions'][0, 1:, 2] = 3.6
        env.finish()
        before = env.transition_rewards.clone()
        env.history['drone_positions'][0, 101:] = float('nan')
        env.numerical_valid[:, 101:] = False
        env.failed[:] = True
        env.lengths[:] = 50
        returns, valid, score, _ = env.finish()
        expected_cost = 5. * 100 / 450
        self.assertFalse(bool(valid[0]))
        self.assertGreater(float(score.causal_best[0, 100]), 0.)
        self.assertAlmostEqual(float(score.components['workspace_cost'][0]),
                               expected_cost, places=12)
        self.assertAlmostEqual(float(score.components['workspace_accrued_cost'][0]),
                               expected_cost, places=12)
        self.assertEqual(float(returns[0]), float(np.float32(-10. - expected_cost)))
        torch.testing.assert_close(env.transition_rewards[:, :49], before[:, :49],
                                   rtol=0, atol=0)
        self.assertEqual(int(env.transition_rewards[:, 50:].count_nonzero()), 0)
        self.assertTrue(bool(torch.isfinite(env.transition_rewards).all()))

    def test_worst_zero_excitation_valid_outcome_outranks_cheapest_failure(self):
        env = self.environment()
        env.history['cable_velocities'].zero_()
        env.history['drone_positions'][0, 1:, 2] = 20.
        valid_return, valid, score, _ = env.finish()
        self.assertTrue(bool(valid[0]))
        self.assertEqual(float(score.score[0]), 0.)
        self.assertEqual(float(valid_return[0]), -5.)
        env.history['drone_positions'][:] = env.origin
        env.failed[:] = True
        env.numerical_valid[:, 3:] = False
        env.lengths[:] = 1
        failed_return, valid, _, _ = env.finish()
        self.assertFalse(bool(valid[0]))
        self.assertEqual(float(failed_return[0]), -10.)
        self.assertGreater(float(valid_return[0]), float(failed_return[0]))

    def test_bounded_approval_does_not_accept_other_bounds_scales_or_tasks(self):
        for field, value in (('weight', 4.), ('weight', 10.),
                             ('distance_scale_m', .1), ('distance_scale_m', 1.)):
            penalty = dict(self.settings['workspace_penalty'], **{field: value})
            with self.subTest(field=field, value=value), self.assertRaises(ValueError):
                load(dict(self.settings, workspace_penalty=penalty))
        for task in ('configs/tasks/whip_stage1_unified_fold_v1.json',
                     'configs/tasks/whip_stage1_discovery_bounded_v1.json'):
            with self.subTest(task=task), self.assertRaises(ValueError):
                load(dict(self.settings, task=task))


if __name__ == '__main__':
    unittest.main()
