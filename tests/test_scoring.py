"""Current scoring and return contracts; synthetic histories are not flight evidence."""
from dataclasses import replace
import math
import unittest
from unittest.mock import patch

import torch

from helpers import wave
from training.configuration import FROZEN_PRESETS, load
from tasks.whip.learned_recovery import make_scorer
from tasks.whip.sac_environment import Stage1Environment
from tasks.whip.stage1 import Stage1Scorer
from tasks.whip.strike import surface_events, crossing_precision_diagnostics


class ScoringTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)
        cls.presets = {stage: load(path) for stage, path in FROZEN_PRESETS.items()}
        cls.settings, cls.snapshot, cls.cfg, cls.scene = cls.presets[1]
        cls.rest = torch.tensor(cls.snapshot['model']['cable']['marker_interval_lengths_m'],
                                dtype=torch.float64)
        cls.time = torch.arange(451, dtype=torch.float64)*.01
        cls.positions, cls.velocities = wave(cls.rest, cls.time)
        cls.valid = torch.ones(1, 451, dtype=torch.bool)

    def score(self, positions=None, velocities=None, valid=None):
        return Stage1Scorer(self.rest, self.time, self.cfg, initial_position=[0, 0, 2])(
            self.positions if positions is None else positions,
            self.velocities if velocities is None else velocities,
            self.valid if valid is None else valid)

    def crossing(self, lateral, vertical, velocity=None, crossing=True):
        lateral = torch.as_tensor(lateral, dtype=torch.float64)
        vertical = torch.as_tensor(vertical, dtype=torch.float64)
        batch = len(lateral)
        positions = torch.zeros(batch, 2, 2, 3, dtype=torch.float64)
        positions[:, :, :, 2] = 2
        positions[:, :, -1, 0] = torch.tensor([1., 1.2] if crossing else [1.2, 1.2],
                                             dtype=torch.float64)
        positions[:, :, -1, 1] = lateral[:, None]
        positions[:, :, -1, 2] += vertical[:, None]
        velocities = torch.zeros_like(positions)
        velocities[:, :, -1] = (torch.tensor([3., 0., 0.], dtype=torch.float64)
                                if velocity is None else velocity[:, None])
        cfg = self.presets[3][2]
        return surface_events(positions, velocities, torch.tensor([0., .1], dtype=torch.float64),
            torch.tensor([1.1, 0., 2.], dtype=torch.float64),
            torch.tensor([1., 0., 0.], dtype=torch.float64), cfg.strike_surface, cfg.request[3])

    def recovery_history(self):
        # Synthetic observed crossing followed by a quiet vertical cable/hover.
        # This deliberately tests the scorer and is not a dynamically feasible rollout.
        time = torch.arange(851, dtype=torch.float64)*.01
        positions, velocities = wave(self.rest, time)
        start = 255
        shift = (1.1+1e-6-positions[0, start, -1, 0])*torch.arange(851)/start
        positions[..., 0] += shift[None, :, None]
        velocities[..., 0] += float((shift[1]-shift[0])/.01)
        origin = torch.tensor([0., 0., 2.], dtype=torch.float64)
        offsets = torch.zeros(len(self.rest)+1, 3, dtype=torch.float64)
        offsets[1:, 2] = -self.rest.cumsum(0)
        positions[:, start+1:] = origin+offsets
        velocities[:, start+1:] = 0
        drone_positions = positions[:, :, 0].clone()
        drone_velocities = velocities[:, :, 0].clone()
        valid = torch.arange(851)[None] <= start+400
        return positions, velocities, valid, drone_positions, drone_velocities

    def recovery_score(self, history=None, hit=True):
        positions, velocities, valid, drone_positions, drone_velocities = (
            self.recovery_history() if history is None else history)
        return make_scorer(self.rest, torch.arange(851, dtype=torch.float64)*.01,
                           self.presets[4][2], initial_position=[0, 0, 2])(
            positions, velocities, valid, drone_positions=drone_positions,
            drone_velocities=drone_velocities, recovery_start_samples=torch.tensor([255]),
            recovery_hit=torch.tensor([hit]), terminal_samples=valid.long().sum(1))

    def test_stage1_same_event_bound_components_and_causal_accounting(self):
        result = self.score()
        components = result.components
        self.assertTrue(bool(result.valid.all()))
        self.assertGreater(float(result.score[0]), 0.)
        self.assertLessEqual(float(result.score[0]), 1.)
        torch.testing.assert_close(result.score, components['mechanical']
            *components['unfolding_fraction']*components['forward_alignment'])
        torch.testing.assert_close(result.score, components['mechanics_credit']
            +components['approach_reward']+components['strike_bonus'])
        self.assertEqual(float(components['approach_reward'][0]+components['strike_bonus'][0]), 0.)
        torch.testing.assert_close(result.causal_best[:, -1], result.score)
        self.assertTrue(bool((result.causal_best.diff(dim=1) >= 0).all()))
        torch.testing.assert_close(result.causal_best.diff(
            prepend=torch.zeros(1, 1, dtype=torch.float64)).sum(1), result.score)

    def test_forward_wave_outscores_sideways_backward_and_rigid_motion(self):
        forward = float(self.score().score[0])
        for rotation in [torch.tensor([[0., -1., 0.], [1., 0., 0.], [0., 0., 1.]],
                                       dtype=torch.float64),
                         torch.diag(torch.tensor([-1., -1., 1.], dtype=torch.float64))]:
            changed = float(self.score(self.positions@rotation.T,
                                       self.velocities@rotation.T).score[0])
            self.assertGreater(forward, 5*changed)
        positions, velocities = torch.zeros_like(self.positions), torch.zeros_like(self.velocities)
        positions[..., 2] = 2-torch.cat((self.rest.new_zeros(1), self.rest.cumsum(0)))
        positions[..., 0] = self.time[None, :, None]*5
        velocities[..., 0] = 5
        self.assertLess(float(self.score(positions, velocities).score[0]), 1e-8)

    def test_no_cycle_accumulation_and_invalid_future_preserves_only_causal_prefix(self):
        original = self.score()
        positions, velocities = self.positions.clone(), self.velocities.clone()
        for start in [150, 300]:
            positions[:, start:start+150] = self.positions[:, :150]
            velocities[:, start:start+150] = self.velocities[:, :150]
        positions[:, 450], velocities[:, 450] = positions[:, 449], velocities[:, 449]
        torch.testing.assert_close(self.score(positions, velocities).score,
                                   original.score, rtol=1e-8, atol=1e-8)
        positions, velocities, valid = self.positions.clone(), self.velocities.clone(), self.valid.clone()
        positions[:, 201:], velocities[:, 201:], valid[:, 201:] = float('nan'), float('nan'), False
        invalid = self.score(positions, velocities, valid)
        self.assertFalse(bool(invalid.valid[0]))
        self.assertTrue(torch.isnan(invalid.score[0]))
        torch.testing.assert_close(invalid.causal_best[:, :200],
                                   original.causal_best[:, :200], rtol=0, atol=0)

    def test_environment_returns_telescope_and_workspace_remains_a_soft_cost(self):
        env = Stage1Environment(dict(self.settings, device='cpu', parallel_environments=1),
                                 self.snapshot, self.cfg, self.scene, capture=False)
        translation = torch.tensor([.7, 0., 0.], dtype=torch.float64)
        endpoint = replace(env.state, position_com=env.state.position_com+translation,
                           free_positions=env.state.free_positions+translation)
        with patch.object(env, '_advance', return_value=endpoint):
            env.step(torch.zeros(1, 3))
        self.assertTrue(bool(env.workspace_breached[0]))
        self.assertTrue(bool(env.active[0]))
        torch.testing.assert_close(env.state.position_com, endpoint.position_com, rtol=0, atol=0)
        env.history['cable_positions'].copy_(self.positions+torch.tensor([.7, 0., 0.]))
        env.history['cable_velocities'].copy_(self.velocities)
        env.history['drone_positions'][:] = env.origin+torch.tensor([.7, 0., 0.])
        env.history['drone_velocities'].zero_()
        env.numerical_valid[:] = True
        env.failed[:] = False
        env.active[:] = False
        env.decision = env.decisions
        env.lengths[:] = env.decisions
        returns, valid, score, _ = env.finish()
        self.assertTrue(bool(valid[0]))
        penalty = self.settings['workspace_penalty']
        cost = penalty['weight']*((.7-self.cfg.workspace.radius_m)/penalty['distance_scale_m'])**2
        self.assertAlmostEqual(float(returns[0]), float(score.score[0])-cost, places=6)
        torch.testing.assert_close(env.transition_rewards.sum(1), returns, rtol=0, atol=1e-5)
        env.numerical_valid[:, 301:] = False
        env.failed[:] = True
        env.lengths[:] = 31
        returns, valid, _, _ = env.finish()
        self.assertFalse(bool(valid[0]))
        self.assertAlmostEqual(float(returns[0]), self.cfg.returns.failure_return-cost*300/450, places=6)
        torch.testing.assert_close(env.transition_rewards.sum(1), returns, rtol=0, atol=1e-5)
        self.assertEqual(int(env.transition_rewards[:, 31:].count_nonzero()), 0)

    def test_precision_rectangle_is_inclusive_per_axis_and_interpolates_crossing(self):
        events = self.crossing([.05, -.05, .05001], [.05, -.05, 0.])
        self.assertEqual(events['geometric_hit'][:, 0].tolist(), [True, True, False])
        self.assertEqual(events['qualified'][:, 0].tolist(), [True, True, False])
        torch.testing.assert_close(events['fraction'][:, 0],
                                   torch.full((3,), .5, dtype=torch.float64), rtol=0, atol=1e-14)
        diagnostics = crossing_precision_diagnostics(events, torch.ones(3, dtype=torch.bool))
        self.assertEqual(diagnostics['final_precision_hit_success'].tolist(), [1., 1., 0.])
        self.assertGreater(float(diagnostics['first_crossing_center_error_m'][0]), .05)
        self.assertAlmostEqual(float(diagnostics['first_crossing_time_s'][0]), .05, places=14)

    def test_missing_crossing_or_invalid_episode_never_becomes_zero_error_success(self):
        missing = crossing_precision_diagnostics(self.crossing([0.], [0.], crossing=False),
                                                torch.tensor([True]))
        self.assertEqual(float(missing['forward_plane_crossing_success'][0]), 0.)
        self.assertEqual(float(missing['final_precision_hit_success'][0]), 0.)
        self.assertTrue(torch.isnan(missing['first_crossing_center_error_m'][0]))
        invalid = crossing_precision_diagnostics(self.crossing([0.], [0.]), torch.tensor([False]))
        for key in ['forward_plane_crossing_success', 'final_precision_hit_success',
                    'first_crossing_center_error_m']:
            self.assertTrue(torch.isnan(invalid[key][0]))

    def test_geometric_hits_and_original_strict_speed_angle_are_distinct(self):
        velocity = torch.tensor([[3., 0., 0.], [2.99, 0., 0.],
                                 [3., 3*math.tan(math.radians(31)), 0.],
                                 [3., 3*math.tan(math.radians(30)), 0.]], dtype=torch.float64)
        events = self.crossing([0.]*4, [0.]*4, velocity)
        self.assertEqual(events['geometric_hit'][:, 0].tolist(), [True]*4)
        self.assertEqual(events['qualified'][:, 0].tolist(), [True, False, False, True])

    def test_recovery_hit_gate_quality_bound_and_causal_return(self):
        result = self.recovery_score()
        self.assertTrue(bool(result.valid[0]))
        self.assertAlmostEqual(float(result.components['recovery_quality'][0]), 1., places=12)
        self.assertAlmostEqual(float(result.components['recovery_credit'][0]), .5, places=12)
        self.assertEqual(float(result.components['whip_hover_and_cable_success'][0]), 1.)
        self.assertLessEqual(float(result.score[0]), 2.5)
        self.assertEqual(float(self.recovery_score(hit=False).components['recovery_credit'][0]), 0.)
        torch.testing.assert_close(result.causal_best.diff(
            prepend=torch.zeros(1, 1, dtype=torch.float64)).sum(1), result.score,
            rtol=1e-12, atol=1e-12)

    def test_drone_hover_does_not_hide_cable_motion_or_drift(self):
        for kind in ['wide_cable', 'moving_cable', 'drone_drift']:
            with self.subTest(kind=kind):
                history = self.recovery_history()
                if kind == 'wide_cable':
                    history[0][:, 256:, 1:, 0] += .5
                elif kind == 'moving_cable':
                    history[1][:, 256:, 1:, 0] = 2
                else:
                    history[3][:, 256:, 0] = .5
                result = self.recovery_score(history)
                self.assertEqual(float(result.components['whip_hover_and_cable_success'][0]), 0.)
                if kind != 'drone_drift':
                    self.assertEqual(float(result.components['recovery_success'][0]), 1.)
                    self.assertEqual(float(result.components['cable_settling_success'][0]), 0.)
                else:
                    self.assertEqual(float(result.components['recovery_success'][0]), 0.)

    def test_recovery_nan_padding_is_ignored_but_incomplete_hold_is_invalid(self):
        original = self.recovery_score()
        history = self.recovery_history()
        for field in [0, 1, 3, 4]:
            history[field][:, 656:] = float('nan')
        padded = self.recovery_score(history)
        torch.testing.assert_close(padded.score, original.score, rtol=0, atol=0)
        torch.testing.assert_close(padded.causal_best, original.causal_best, rtol=0, atol=0)
        history = self.recovery_history()
        history[2][:, 350:] = False
        incomplete = self.recovery_score(history)
        self.assertFalse(bool(incomplete.valid[0]))
        self.assertTrue(torch.isnan(incomplete.score[0]))
        self.assertEqual(float(incomplete.components['whip_hover_and_cable_success'][0]), 0.)


if __name__ == '__main__':
    unittest.main()
