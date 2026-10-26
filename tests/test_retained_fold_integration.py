"""Approved retained-progress task plumbing; constructed states, not learned success."""
from copy import deepcopy
from dataclasses import replace
import json
import math
from pathlib import Path
import tempfile
import unittest

import torch

from tasks.whip.sac_environment import make_environment
from training.configuration import load, task_configuration
from training.artifacts import create_run
from training.ppo import PPO


PRESET = 'configs/learning/whip_stage2_ppo_retained_fold_delivery_v1.json'


class RetainedFoldIntegrationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)
        cls.settings, cls.snapshot, cls.cfg, cls.scene = load(PRESET)

    def env(self, lanes=1):
        return make_environment(dict(self.settings, device='cpu', parallel_environments=lanes),
                                self.snapshot, self.cfg, self.scene, capture=False)

    def test_fresh_stage2_is_narrowly_registered_without_actor_source(self):
        with tempfile.TemporaryDirectory(dir='tmp') as folder:
            run = create_run(self.settings, root=folder)
            request = json.loads((run/'request.json').read_text())
            self.assertIsNone(request['resume'])
            self.assertIsNone(request['initialize_actor'])
            snap = json.loads((run/'configuration.json').read_text())
            self.assertNotIn('policy_initialization', snap)
            self.assertEqual(snap['task']['matching']['reward_mode'], 'retained_fold_delivery_v1')
            self.assertEqual(snap['settings']['training_episodes'], 25600)
        old = load('configs/learning/whip_stage2_ppo_first_hit_delivery_v2.json')[0]
        with tempfile.TemporaryDirectory(dir='tmp') as folder:
            with self.assertRaisesRegex(ValueError, 'actor checkpoint'):
                create_run(old, root=folder)

    def test_rejects_model_resolution_change(self):
        model = deepcopy(self.snapshot['model'])
        model['simulation']['segments_per_marker_interval'] = [2]*10
        with self.assertRaises(ValueError):
            task_configuration(self.snapshot['task'], self.settings, model, self.scene)

    def test_full_history_progress_critic_and_saved_entropy_budget(self):
        e = self.env()
        self.assertEqual((e.samples, e.decisions), (701, 350))
        self.assertEqual(self.settings['discount'], 1)
        self.assertEqual(e.attempts[0].features().shape, (24,))
        a = PPO(self.settings, e.actor_columns, e.frame_size, 'cpu',
                privileged_size=e.critic_state_size, encoder_options=e.policy_encoder_options)
        for episodes in (0, 6400, 12800, 19200, 25600):
            a.set_completed_episodes(episodes)
            self.assertAlmostEqual(a.entropy_weight, .001+.019*math.exp(-8*episodes/25600), places=15)
        saved = a.state_dict()
        b = PPO(dict(self.settings, training_episodes=51200), e.actor_columns, e.frame_size, 'cpu',
                privileged_size=e.critic_state_size, encoder_options=e.policy_encoder_options)
        b.load_state_dict(saved)
        self.assertEqual(b.entropy_episode_budget, 25600)
        self.assertEqual(b.completed_episodes, 25600)
        self.assertEqual(b.entropy_weight, a.entropy_weight)

    def test_substep_failure_cannot_be_hidden_by_safe_endpoint(self):
        e = self.env()
        initial = e.state
        outside = replace(initial, position_com=initial.position_com+e.model.tensor([.6, 0, 0]))
        def fake(state, command, controller, **kwargs):
            kwargs['observer'](outside, 1/600)
            kwargs['observer'](initial, 2/600)
            return initial
        e.model.step_unchecked = fake
        _, _, done = e.step(torch.zeros(1, 3))
        self.assertTrue(done)
        returns, valid, score, _ = e.finish()
        self.assertFalse(valid[0])
        self.assertEqual(float(returns[0]), -10)
        self.assertEqual(e.attempts[0].progress, 0)
        self.assertAlmostEqual(float(e.terminal_times[0]), 1/600)
        torch.testing.assert_close(e.transition_rewards.sum(1), returns)
        self.assertFalse(e.attempts[0].qualified)

    def test_failed_lane_freezes_while_other_lane_observes(self):
        e = self.env(2)
        initial = e.state
        outside = replace(initial, position_com=initial.position_com+e.model.tensor([[.6, 0, 0], [0, 0, 0]]))
        def fake(state, command, controller, **kwargs):
            kwargs['observer'](outside, 1/600)
            return outside
        e.model.step_unchecked = fake
        e.step(torch.zeros(2, 3))
        self.assertEqual(e.active.tolist(), [False, True])
        self.assertEqual(e.lengths.tolist(), [1, 1])
        first_ledger = float(e.ledger[0].sum())
        e.step(torch.zeros(2, 3))
        self.assertEqual(e.lengths.tolist(), [1, 2])
        self.assertEqual(float(e.ledger[0].sum()), first_ledger)
        self.assertEqual(first_ledger, -10)

    def test_stationary_constructed_attempt_times_out_without_time_cost(self):
        e = self.env()
        def fixed_state(state, command, controller, **kwargs):
            kwargs['observer'](state, .01)
            return state
        e.model.step_unchecked = fixed_state
        for _ in range(350):
            _, _, done = e.step(torch.zeros(1, 3))
        self.assertTrue(done)
        returns, valid, score, _ = e.finish()
        self.assertTrue(valid[0])
        self.assertEqual(e.attempts[0].terminal, 'timeout')
        self.assertEqual(float(returns[0]), -2)
        self.assertEqual(float(e.terminal_times[0]), 7)
        torch.testing.assert_close(e.transition_rewards.sum(1), returns)
        record = e.event_records()[0]
        self.assertEqual(record['episode_return'], -2)


if __name__ == '__main__':
    unittest.main()
