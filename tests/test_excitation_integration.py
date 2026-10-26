"""CPU accounting/registration checks for the fresh flexible-excitation run.

Injected histories are analytic measurement fixtures, not physically simulated
flights. These checks do not validate the reward as a whipping definition or
demonstrate policy learning.
"""
from copy import deepcopy
import json
import math
from pathlib import Path
from types import SimpleNamespace
import unittest

import numpy as np
import torch

from helpers import project_temporary
from evaluation.whip_measurement_examples import kinematic_controls
from evaluation.whip_output_decomposition import decompose
from tasks.whip.sac_environment import Stage1Environment
from training.artifacts import create_run, identity, validate_actor_source
from training.configuration import collection_limit, load, task_configuration
from training.ppo import PPO
from training.worker import record_simple_collection


PRESET = 'configs/learning/whip_stage1_ppo_flexible_excitation_ar1_v1.json'


class ExcitationIntegrationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)
        cls.settings, cls.snapshot, cls.cfg, cls.scene = load(PRESET)
        cls.rest = np.asarray(cls.snapshot['model']['cable']['marker_interval_lengths_m'])
        cls.arrays = next(c['arrays'] for c in kinematic_controls(cls.rest)
                          if c['name'] == 'small_travelling_wave')

    def env(self):
        settings = dict(self.settings, device='cpu', parallel_environments=1)
        env = Stage1Environment(settings, self.snapshot, self.cfg, self.scene, capture=False)
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

    def test_valid_return_matches_raw_same_sample_maximum_and_reward_sum(self):
        env = self.env()
        shape = decompose(self.arrays['cable_positions'], self.arrays['cable_velocities'], self.rest)
        world = np.linalg.norm(self.arrays['cable_velocities'][:, -1], axis=1)
        q = np.minimum(world, shape['shape_speed_mps'][:, -1])
        expected = float(q[1:].max())  # initial sample is unpaid
        returns, valid, score, cost = env.finish()
        self.assertTrue(bool(valid[0]))
        self.assertGreater(expected, 0.)
        self.assertAlmostEqual(float(score.score[0]), expected, places=12)
        self.assertAlmostEqual(float(returns[0]), expected, places=8)
        self.assertEqual(float(cost[0]), 0.)
        torch.testing.assert_close(returns, env.transition_rewards.sum(1))
        self.assertEqual(self.settings['discount'], 1)

    def test_contained_movement_and_acceleration_have_no_cost(self):
        env = self.env()
        t = env.time
        env.history['drone_positions'][0, :, 0] += .4*torch.sin(4*t)
        # Deliberately inconsistent injected velocity isolates cost accounting;
        # this is not an executed controller/physics trajectory.
        env.history['drone_velocities'][0, :, 0] = 20*torch.sin(8*t)
        returns, valid, score, cost = env.finish()
        self.assertTrue(bool(valid[0]))
        self.assertGreater(float(score.components['drone_path_length_m'][0]), 1.)
        for field in ('workspace_cost', 'movement_cost', 'movement_accrued_cost'):
            self.assertEqual(float(score.components[field][0]), 0.)
        self.assertEqual(float(cost[0]), 0.)
        self.assertAlmostEqual(float(returns[0]), float(score.score[0]), places=8)
        torch.testing.assert_close(returns, env.transition_rewards.sum(1))

    def test_valid_workspace_cost_and_failed_credit_clawback_are_exact(self):
        env = self.env()
        env.history['drone_positions'][0, 1:, 2] = 3.1
        returns, valid, score, _ = env.finish()
        self.assertTrue(bool(valid[0]))
        self.assertGreater(float(score.score[0]), 0.)
        self.assertAlmostEqual(float(score.components['workspace_cost'][0]), 2.5, places=12)
        self.assertAlmostEqual(float(returns[0]), float(score.score[0])-2.5, places=6)
        torch.testing.assert_close(returns, env.transition_rewards.sum(1))
        env.failed[:] = True
        returns, valid, score, _ = env.finish()
        self.assertFalse(bool(valid[0]))
        self.assertGreater(float(score.causal_best[0, -1]), 0.)
        self.assertEqual(float(returns[0]), -12.5)
        self.assertAlmostEqual(float(score.components['workspace_accrued_cost'][0]), 2.5, places=12)
        torch.testing.assert_close(returns, env.transition_rewards.sum(1))

    def test_failed_prefix_cost_uses_full_horizon_and_ignores_padded_suffix(self):
        env = self.env()
        env.history['drone_positions'][0, 1:101, 2] = 3.1
        env.history['drone_positions'][0, 101:, 2] = 100.
        env.numerical_valid[:, 101:] = False
        env.failed[:] = True
        env.lengths[:] = 50
        returns, valid, score, _ = env.finish()
        expected_cost = 2.5*100/450
        self.assertFalse(bool(valid[0]))
        self.assertAlmostEqual(float(score.components['workspace_cost'][0]), expected_cost, places=12)
        self.assertAlmostEqual(float(returns[0]), -10-expected_cost, places=5)
        self.assertEqual(float(score.components['movement_cost'][0]), 0.)
        self.assertTrue(bool((env.transition_rewards[0, 50:] == 0).all()))
        torch.testing.assert_close(returns, env.transition_rewards.sum(1))

    def test_positive_guide_is_not_a_hit_and_recorder_preserves_separate_accounting(self):
        env = self.env()
        returns, valid, score, _ = env.finish()
        self.assertGreater(float(score.score[0]), 0.)
        for field in ('strike_success', 'geometric_hit_success', 'strict_strike_success'):
            self.assertEqual(float(score.components[field][0]), 0.)
        with project_temporary() as folder:
            root = Path(folder)
            record_simple_collection(root, env, returns, valid, score, 1)
            report = json.loads((root/'collection-00001-events.json').read_text())
            self.assertEqual(report['schema'], 'flexible_excitation_v1')
            row = report['lanes'][0]
            self.assertAlmostEqual(row['excitation_only_episode_return'], float(score.score[0]), places=12)
            self.assertEqual(row['episode_return'], float(returns[0]))
            self.assertEqual(row['mechanism_status'], 'unresolved')
            with np.load(root/'collection-00001.npz', allow_pickle=False) as data:
                self.assertFalse(bool(data['qualified'][0]))
                self.assertFalse(bool(data['geometric_hit'][0]))
                self.assertEqual(str(data['hit_definition']), 'diagnostic_only_full_horizon_crossings')
                np.testing.assert_allclose(data['stored_reward_sum'], data['returns'], atol=1e-6, rtol=0)
            env.history['drone_positions'][0, 1:, 2] = 3.1
            env.failed[:] = True
            returns, valid, score, _ = env.finish()
            record_simple_collection(root, env, returns, valid, score, 2)
            row = json.loads((root/'collection-00002-events.json').read_text())['lanes'][0]
            self.assertEqual(row['episode_return'], -12.5)
            self.assertEqual(row['excitation_only_episode_return'], -10.)
            self.assertGreater(row['best_event_score'], 0.)
            self.assertAlmostEqual(row['workspace_cost'], 2.5, places=12)
            self.assertEqual(row['movement_cost'], 0.)

    def test_fresh_preset_budget_control_and_exponential_exploration(self):
        self.assertEqual(self.settings['seed'], 5)
        self.assertEqual(self.settings['parallel_environments'], 128)
        self.assertEqual(self.settings['training_episodes'], 15360)
        self.assertEqual(collection_limit(self.settings), 120)
        self.assertEqual(self.settings['action_interface'], 'acceleration_direct_v1')
        self.assertEqual(self.settings['policy_rate_hz'], 50)
        self.assertEqual(self.settings['acceleration_limit_mps2'], [8., 8., 4.])
        self.assertEqual(self.settings['reward_scale'], 1)
        self.assertEqual(self.cfg.termination, 'full_horizon_v1')
        self.assertEqual(self.settings['ppo']['action_noise'], dict(mode='ar1_v1', correlation_time_s=.1))
        for field in ('movement_penalty', 'fold_curriculum', 'curriculum', 'target_curriculum', 'plateau_detection'):
            self.assertNotIn(field, self.settings)
        for collection in (0, 1, 20, 60, 120):
            state = SimpleNamespace(settings=self.settings, completed_episodes=128*collection,
                entropy_episode_offset=0, entropy_episode_budget=15360)
            self.assertAlmostEqual(PPO.entropy_weight.fget(state),
                                   .001+.019*math.exp(-collection/20), places=15)

    def test_registered_contract_rejects_reward_accounting_and_cost_changes(self):
        for key, value in (
                ('training_episodes', 3840), ('parallel_environments', 64), ('discount', .99),
                ('workspace_penalty', dict(mode='squared_excess_v1', weight=.01, distance_scale_m=.1)),
                ('movement_penalty', dict(mode='path_length_v1', weight_per_cable_length=.25)),
                ('training_stage', 2)):
            with self.subTest(key=key), self.assertRaises(ValueError):
                load(dict(self.settings, **{key: value}))
        task = deepcopy(self.snapshot['task'])
        task['return']['acceleration_cost_weight'] = .1
        with self.assertRaises(ValueError):
            task_configuration(task, self.settings, self.snapshot['model'], self.scene)

    def test_actor_initialization_is_forbidden_before_reading_source_or_creating_run(self):
        with project_temporary() as folder:
            root = Path(folder)/'new-runs'
            with self.assertRaisesRegex(ValueError, 'random actor'):
                create_run(self.settings, root=root, initialize_actor=Path(folder)/'nonexistent.pt')
            self.assertFalse(root.exists())

    def test_excitation_actor_has_no_registered_automatic_later_stage_handoff(self):
        old = deepcopy(self.snapshot)
        old['execution_hashes'] = {}
        old['identity'] = identity(old)
        saved = dict(schema='ppo_stage1_checkpoint_v1', snapshot=old, identity=old['identity'],
                     agent=dict(actor={'weight': torch.zeros(1)}))
        target = deepcopy(self.snapshot)
        target['settings']['training_stage'] = 2
        target['task']['matching']['reward_mode'] = 'directed_shape_delivery_v1'
        with self.assertRaisesRegex(ValueError, 'motion review'):
            validate_actor_source(saved, target)


if __name__ == '__main__':
    unittest.main()
