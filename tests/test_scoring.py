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
        # Keep the prior reward arithmetic under regression independently of
        # the newly selected compact Stage1 experiment.
        cls.settings, cls.snapshot, cls.cfg, cls.scene = load(
            'configs/learning/whip_stage1_ppo_cone25_v4.json')
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


class CompactStage1Tests(unittest.TestCase):
    """Task-only kinematic probes; these are not simulated or measured whips."""
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)
        cls.settings, cls.snapshot, cls.cfg, cls.scene = load(FROZEN_PRESETS[1])
        cls.legacy = load('configs/learning/whip_stage1_ppo_cone25_v4.json')[2]
        cls.rest = torch.tensor(cls.snapshot['model']['cable']['marker_interval_lengths_m'], dtype=torch.float64)
        cls.time = torch.arange(451, dtype=torch.float64)*.01
        positions, velocities = wave(cls.rest, cls.time)
        angle = -math.pi/3
        rotation = torch.tensor([[math.cos(angle), 0., math.sin(angle)], [0., 1., 0.],
                                 [-math.sin(angle), 0., math.cos(angle)]], dtype=torch.float64)
        cls.origin = torch.tensor([0., 0., 2.], dtype=torch.float64)
        cls.positions = (positions-cls.origin)@rotation.T+cls.origin
        cls.velocities = velocities@rotation.T
        cls.valid = torch.ones(1, 451, dtype=torch.bool)
        cls.drone = cls.positions[:, :, 0].clone()

    def score(self, *, cfg=None, positions=None, velocities=None, drone=None, valid=None):
        return Stage1Scorer(self.rest, self.time, cfg or self.cfg, initial_position=self.origin)(
            self.positions if positions is None else positions,
            self.velocities if velocities is None else velocities,
            self.valid if valid is None else valid,
            drone_positions=self.drone if drone is None else drone)

    def loop(self, amplitude):
        # Isolate the travel measurement while holding cable kinematics fixed.
        # A closed loop before the earliest eligible event has zero net displacement.
        drone = self.drone.clone()
        drone[:, :21, 1] += amplitude*torch.sin(math.pi*self.time[:21]/.2).square()
        return drone

    def test_same_event_reach_path_accounting_and_bound(self):
        drone = self.loop(.08)
        result = self.score(drone=drone)
        c = result.components
        selected = int(result.event_indices[0, -1])
        measured_path = drone[:, :selected+1].diff(dim=1).norm(dim=-1).sum(1)
        measured_reach = ((self.positions[:, selected, -1]-self.positions[:, selected, 0])
                          *torch.tensor([1., 0., 0.])).sum(-1)
        torch.testing.assert_close(c['preparation_path_m'], measured_path)
        torch.testing.assert_close(c['forward_reach_m'], measured_reach)
        self.assertGreater(float(c['forward_reach_fraction'][0]), 0.)
        self.assertTrue(bool(((c['forward_reach_fraction'] >= 0) & (c['forward_reach_fraction'] <= 1)).all()))
        self.assertGreater(float(result.score[0]), 0.)
        self.assertLessEqual(float(result.score[0]), 1.25)
        torch.testing.assert_close(result.score, c['mechanical']*c['unfolding_fraction']
            *c['forward_alignment']*(1+.25*c['forward_reach_fraction'])
            *torch.exp(-.15*measured_path/self.rest.sum()))
        torch.testing.assert_close(result.score, c['mechanics_credit']+c['reach_bonus']-c['preparation_cost'])
        torch.testing.assert_close(result.causal_best[:, -1], result.score)
        self.assertTrue(bool((result.causal_best.diff(dim=1) >= 0).all()))

    def test_closed_preparation_loops_reduce_credit_without_net_displacement(self):
        original = self.score()
        previous = float(original.score[0])
        for amplitude in (.08, .16):
            drone = self.loop(amplitude)
            torch.testing.assert_close(drone[:, 20:], self.drone[:, 20:], atol=1e-15, rtol=0)
            result = self.score(drone=drone)
            torch.testing.assert_close(result.event_indices, original.event_indices, atol=0, rtol=0)
            torch.testing.assert_close(result.score,
                original.score*math.exp(-.15*2*amplitude/float(self.rest.sum())))
            self.assertLess(float(result.score[0]), previous)
            previous = float(result.score[0])

    def test_reach_increases_credit_only_with_existing_mechanics(self):
        no_reach = replace(self.cfg, matching=replace(self.cfg.matching, reach_weight=0.))
        self.assertGreater(float(self.score().score[0]), float(self.score(cfg=no_reach).score[0]))
        material = torch.cat((self.rest.new_zeros(1), self.rest.cumsum(0)))
        angle = .6*torch.sin(2*math.pi*self.time)
        rate = .6*2*math.pi*torch.cos(2*math.pi*self.time)
        directions = torch.stack((angle.sin(), torch.zeros_like(angle), -angle.cos()), -1)
        tangents = torch.stack((angle.cos(), torch.zeros_like(angle), angle.sin()), -1)
        positions = self.origin+directions[:, None]*material[None, :, None]
        velocities = rate[:, None, None]*tangents[:, None]*material[None, :, None]
        swing = self.score(positions=positions[None], velocities=velocities[None])
        self.assertLess(float(swing.score[0]), 1e-8)
        # Purely translating a rigid cable also earns no reach-only credit.
        positions = positions[:1].expand(451, -1, -1).clone()
        positions[..., 0] += 5*self.time[:, None]
        velocities = torch.zeros_like(positions)
        velocities[..., 0] = 5
        rigid = self.score(positions=positions[None], velocities=velocities[None], drone=positions[None, :, 0])
        self.assertLess(float(rigid.score[0]), 1e-8)

    def test_future_travel_and_bad_drone_suffix_cannot_rewrite_earlier_credit(self):
        original = self.score()
        selected = int(original.event_indices[0, -1])
        drone = self.drone.clone()
        drone[:, selected+1:, 1] += self.time[selected+1:]-self.time[selected]
        later = self.score(drone=drone)
        torch.testing.assert_close(later.causal_best[:, :selected+1],
                                   original.causal_best[:, :selected+1], rtol=0, atol=0)
        torch.testing.assert_close(later.score, original.score, rtol=0, atol=0)
        drone = self.drone.clone()
        drone[:, 201:] = float('nan')
        invalid = self.score(drone=drone)
        self.assertFalse(bool(invalid.valid[0]))
        self.assertTrue(torch.isnan(invalid.score[0]))
        self.assertFalse(bool(invalid.has_event[0]))
        torch.testing.assert_close(invalid.causal_best[:, :201], original.causal_best[:, :201], rtol=0, atol=0)

    def test_zero_weights_preserve_legacy_arithmetic_and_repeated_event_maximum(self):
        zero = replace(self.cfg, matching=replace(self.cfg.matching,
                       reach_weight=0., preparation_path_weight=0.))
        legacy = Stage1Scorer(self.rest, self.time, self.legacy, initial_position=self.origin)(
            self.positions, self.velocities, self.valid)
        result = self.score(cfg=zero)
        for key in legacy.components:
            torch.testing.assert_close(result.components[key], legacy.components[key], rtol=0, atol=0, equal_nan=True)
        torch.testing.assert_close(result.score, legacy.score, rtol=0, atol=0)
        torch.testing.assert_close(result.causal_best, legacy.causal_best, rtol=0, atol=0)
        positions, velocities = self.positions.clone(), self.velocities.clone()
        for start in (150, 300):
            positions[:, start:start+150] = self.positions[:, :150]
            velocities[:, start:start+150] = self.velocities[:, :150]
        positions[:, 450], velocities[:, 450] = positions[:, 449], velocities[:, 449]
        torch.testing.assert_close(self.score(positions=positions, velocities=velocities).score,
                                   self.score().score, rtol=1e-8, atol=1e-8)

    def test_explicit_weights_and_real_drone_history_are_required(self):
        scorer = Stage1Scorer(self.rest, self.time, self.cfg, initial_position=self.origin)
        for drone in (None, self.drone[:, :-1], self.drone.float()):
            with self.subTest(drone_type=type(drone).__name__), self.assertRaisesRegex(ValueError, 'tracked drone_positions'):
                scorer(self.positions, self.velocities, self.valid, drone_positions=drone)
        for key in ('reach_weight', 'preparation_path_weight'):
            for value in (None, -1., float('nan'), float('inf'), True):
                with self.subTest(key=key, value=value), self.assertRaisesRegex(ValueError, 'Compact Stage1'):
                    replace(self.cfg.matching, **{key: value})
            with self.assertRaisesRegex(ValueError, 'compact Stage1'):
                replace(self.legacy.matching, **{key: .1})
        with self.assertRaisesRegex(ValueError, 'Stage1 only'):
            load(dict(self.settings, training_stage=2))

    def test_environment_reward_delivery_and_failure_correction(self):
        env = Stage1Environment(dict(self.settings, device='cpu', parallel_environments=1),
                                self.snapshot, self.cfg, self.scene, capture=False)
        env.history['cable_positions'].copy_(self.positions)
        env.history['cable_velocities'].copy_(self.velocities)
        env.history['drone_positions'].copy_(self.loop(.08))
        env.history['drone_velocities'].zero_()
        env.numerical_valid[:] = True
        env.failed[:] = False
        env.active[:] = False
        env.decision = env.decisions
        env.lengths[:] = env.decisions
        returns, valid, score, _ = env.finish()
        self.assertTrue(bool(valid[0]))
        # The task scores in float64; the established learner return is float32.
        torch.testing.assert_close(returns, score.score.float(), rtol=0, atol=0)
        torch.testing.assert_close(env.transition_rewards.sum(1), returns, rtol=0, atol=1e-6)
        env.numerical_valid[:, 301:] = False
        env.failed[:] = True
        env.lengths[:] = 31
        returns, valid, _, _ = env.finish()
        self.assertFalse(bool(valid[0]))
        self.assertAlmostEqual(float(returns[0]), self.cfg.returns.failure_return, places=12)
        torch.testing.assert_close(env.transition_rewards.sum(1), returns, rtol=0, atol=1e-6)
        self.assertEqual(int(env.transition_rewards[:, 31:].count_nonzero()), 0)

    @unittest.skipUnless(torch.cuda.is_available(), 'CUDA not available')
    def test_cuda_same_event_score_and_causal_history_match_cpu(self):
        drone = self.loop(.08)
        expected = self.score(drone=drone)
        actual = Stage1Scorer(self.rest.cuda(), self.time.cuda(), self.cfg,
                              initial_position=self.origin.cuda())(
            self.positions.cuda(), self.velocities.cuda(), self.valid.cuda(), drone_positions=drone.cuda())
        torch.testing.assert_close(actual.score.cpu(), expected.score, rtol=1e-8, atol=1e-10)
        torch.testing.assert_close(actual.causal_best.cpu(), expected.causal_best, rtol=1e-8, atol=1e-10)
        torch.testing.assert_close(actual.event_indices.cpu(), expected.event_indices, rtol=0, atol=0)
        for key in ('forward_reach_fraction', 'preparation_path_m', 'reach_bonus', 'preparation_cost'):
            torch.testing.assert_close(actual.components[key].cpu(), expected.components[key], rtol=1e-8, atol=1e-10)


class LevelOutreachStage1Tests(unittest.TestCase):
    """Approved experimental scoring on synthetic, not feasible-flight, histories."""
    @classmethod
    def setUpClass(cls):
        CompactStage1Tests.setUpClass()
        cls.settings, cls.snapshot, cls.cfg, cls.scene = load(
            'configs/learning/whip_stage1_ppo_level_outreach_v6.json')
        for name in ('legacy', 'rest', 'time', 'origin', 'positions', 'velocities', 'valid', 'drone'):
            setattr(cls, name, getattr(CompactStage1Tests, name))

    score = CompactStage1Tests.score
    loop = CompactStage1Tests.loop

    def test_same_release_formula_drone_reference_accounting_and_bound(self):
        drone = self.loop(.08)
        drone[..., 2] -= .12
        result = self.score(drone=drone)
        c = result.components
        selected = int(result.event_indices[0, -1])
        height = self.positions[:, selected, -1, 2]-drone[:, selected, 2]
        path = drone[:, :selected+1].diff(dim=1).norm(dim=-1).sum(1)
        reach = ((self.positions[:, selected, -1]-self.positions[:, selected, 0])
                 *self.rest.new_tensor([1., 0., 0.])).sum(-1)
        torch.testing.assert_close(c['tip_drone_height_error_m'], height)
        torch.testing.assert_close(c['preparation_path_m'], path)
        torch.testing.assert_close(c['forward_reach_m'], reach)
        h = torch.exp(-.5*(height/(.35*self.rest.sum())).square())
        torch.testing.assert_close(c['level_height_factor'], h)
        torch.testing.assert_close(c['level_outreach_quality'], (reach/self.rest.sum()).clamp(0, 1)*h)
        torch.testing.assert_close(result.score, c['mechanical']*c['unfolding_fraction']
            *c['forward_alignment']*(.25+.75*c['level_outreach_quality'])
            *torch.exp(-.15*path/self.rest.sum()))
        torch.testing.assert_close(result.score,
            c['base_mechanics_credit']+c['level_outreach_bonus']-c['preparation_cost'])
        self.assertGreater(float(result.score[0]), 0.)
        self.assertLessEqual(float(result.score[0]), 1.)
        torch.testing.assert_close(result.causal_best[:, -1], result.score)
        self.assertTrue(bool((result.causal_best.diff(dim=1) >= 0).all()))

    def test_height_preference_symmetry_and_world_translation(self):
        # Hold cable kinematics and path fixed to isolate the height factor.
        # Varying the tracked-drone history independently is a scorer probe only.
        level = self.drone.clone()
        level[..., 2] = self.positions[..., -1, 2]
        uncharged = replace(self.cfg, matching=replace(self.cfg.matching, preparation_path_weight=0.))
        expected = self.score(cfg=uncharged, drone=level)
        for sign in (-1, 1):
            displaced = level.clone()
            displaced[..., 2] += sign*.35*self.rest.sum()
            result = self.score(cfg=uncharged, drone=displaced)
            torch.testing.assert_close(result.components['level_height_factor'],
                                       self.rest.new_tensor([math.exp(-.5)]))
            self.assertLess(float(result.score[0]), float(expected.score[0]))
        shift = self.rest.new_tensor([0., 0., 3.])
        original = self.score()
        moved = self.score(positions=self.positions+shift, drone=self.drone+shift)
        for name in ('score', 'causal_best', 'event_indices'):
            torch.testing.assert_close(getattr(moved, name), getattr(original, name), rtol=1e-9, atol=1e-10)
        torch.testing.assert_close(moved.components['tip_drone_height_error_m'],
                                   original.components['tip_drone_height_error_m'])

    def test_preparation_and_future_height_are_causal_with_invalid_suffix(self):
        original = self.score()
        loop = self.score(drone=self.loop(.08))
        torch.testing.assert_close(loop.score, original.score*math.exp(-.15*.16/float(self.rest.sum())))
        torch.testing.assert_close(loop.event_indices, original.event_indices, rtol=0, atol=0)
        selected = int(original.event_indices[0, -1])
        drone = self.drone.clone()
        drone[:, selected+1:, 2] -= .5
        later = self.score(drone=drone)
        torch.testing.assert_close(later.causal_best[:, :selected+1],
                                   original.causal_best[:, :selected+1], rtol=0, atol=0)
        drone = self.drone.clone()
        drone[:, 201:] = float('nan')
        bad = self.score(drone=drone)
        self.assertFalse(bool(bad.valid[0]))
        self.assertFalse(bool(bad.has_event[0]))
        self.assertTrue(torch.isnan(bad.score[0]))
        torch.testing.assert_close(bad.causal_best[:, :201], original.causal_best[:, :201], rtol=0, atol=0)

    def test_base_only_preserves_legacy_and_rigid_motion_has_no_outreach_credit(self):
        cfg = replace(self.cfg, matching=replace(self.cfg.matching,
                      outreach_base_fraction=1., preparation_path_weight=0.))
        legacy = Stage1Scorer(self.rest, self.time, self.legacy, initial_position=self.origin)(
            self.positions, self.velocities, self.valid)
        result = self.score(cfg=cfg)
        torch.testing.assert_close(result.score, legacy.score, rtol=0, atol=0)
        torch.testing.assert_close(result.causal_best, legacy.causal_best, rtol=0, atol=0)
        for key in legacy.components:
            torch.testing.assert_close(result.components[key], legacy.components[key], rtol=0, atol=0, equal_nan=True)
        positions = self.positions[:, :1].expand(-1, 451, -1, -1).clone()
        positions[..., 0] += 5*self.time[None, :, None]
        velocities = torch.zeros_like(positions)
        velocities[..., 0] = 5
        rigid = self.score(positions=positions, velocities=velocities, drone=positions[:, :, 0])
        self.assertLess(float(rigid.score[0]), 1e-8)

    def test_profile_guards_and_required_tracked_history(self):
        for key, values in {
            'outreach_base_fraction': (None, -.01, 1.01, float('nan'), float('inf'), True),
            'outreach_height_width_fraction': (None, 0., -.1, float('nan'), float('inf'), True),
            'preparation_path_weight': (None, -.1, float('nan'), float('inf'), True),
        }.items():
            for value in values:
                with self.subTest(key=key, value=value), self.assertRaises(ValueError):
                    replace(self.cfg.matching, **{key: value})
        for key in ('outreach_base_fraction', 'outreach_height_width_fraction'):
            with self.assertRaises(ValueError):
                replace(self.legacy.matching, **{key: .1})
        with self.assertRaises(ValueError):
            replace(self.cfg.matching, reach_weight=.25)
        with self.assertRaisesRegex(ValueError, 'Stage1 only'):
            load(dict(self.settings, training_stage=2))
        with self.assertRaisesRegex(ValueError, 'tracked drone_positions'):
            Stage1Scorer(self.rest, self.time, self.cfg, initial_position=self.origin)(
                self.positions, self.velocities, self.valid)

    test_environment_reward_delivery_and_failure_correction = CompactStage1Tests.test_environment_reward_delivery_and_failure_correction

    @unittest.skipUnless(torch.cuda.is_available(), 'CUDA not available')
    def test_cuda_same_event_score_and_causal_history_match_cpu(self):
        drone = self.loop(.08)
        expected = self.score(drone=drone)
        actual = Stage1Scorer(self.rest.cuda(), self.time.cuda(), self.cfg,
                              initial_position=self.origin.cuda())(
            self.positions.cuda(), self.velocities.cuda(), self.valid.cuda(), drone_positions=drone.cuda())
        for name in ('score', 'causal_best', 'event_indices'):
            torch.testing.assert_close(getattr(actual, name).cpu(), getattr(expected, name), rtol=1e-8, atol=1e-10)
        for key in ('tip_drone_height_error_m', 'level_height_factor', 'level_outreach_quality',
                    'base_mechanics_credit', 'level_outreach_bonus', 'preparation_cost'):
            torch.testing.assert_close(actual.components[key].cpu(), expected.components[key], rtol=1e-8, atol=1e-10)


class SpeedOutreachStage1Tests(unittest.TestCase):
    """Formula, causal accounting and plant-independent scoring probes."""
    @classmethod
    def setUpClass(cls):
        CompactStage1Tests.setUpClass()
        cls.settings,cls.snapshot,cls.cfg,cls.scene = load('configs/learning/whip_stage1_ppo_demo_outreach_v7.json')
        for name in ('legacy','rest','time','origin','positions','velocities','valid','drone'):
            setattr(cls,name,getattr(CompactStage1Tests,name))

    score = CompactStage1Tests.score
    loop = CompactStage1Tests.loop
    test_environment_reward_delivery_and_failure_correction = CompactStage1Tests.test_environment_reward_delivery_and_failure_correction
    test_height_preference_symmetry_and_world_translation = LevelOutreachStage1Tests.test_height_preference_symmetry_and_world_translation
    test_preparation_and_future_height_are_causal_with_invalid_suffix = LevelOutreachStage1Tests.test_preparation_and_future_height_are_causal_with_invalid_suffix

    def test_same_event_approved_formula_and_additive_accounting(self):
        result = self.score(drone=self.loop(.08))
        c = result.components
        mu = c['mechanical']*c['unfolding_fraction']
        speed = c['forward_mps'].clamp_min(0)
        expected = mu/(mu+.05)*c['forward_alignment']*c['forward_reach_fraction']*c['level_height_factor']*(speed/(speed+3)).square()*c['preparation_factor']
        torch.testing.assert_close(result.score,expected)
        torch.testing.assert_close(result.score,c['outreach_credit']-c['preparation_cost'])
        torch.testing.assert_close(result.causal_best[:,-1],result.score)
        self.assertLessEqual(float(result.score[0]),1.)
        self.assertGreater(float(result.score[0]),0.)

    def test_explicit_scales_stage_and_no_base_allowance_guards(self):
        for key in ('outreach_mechanics_scale','outreach_speed_scale_mps','outreach_speed_power'):
            for value in (None,0.,-.1,float('nan'),float('inf'),True):
                with self.subTest(key=key,value=value),self.assertRaises(ValueError):
                    replace(self.cfg.matching,**{key:value})
            with self.assertRaises(ValueError):
                replace(self.legacy.matching,**{key:.1})
        with self.assertRaises(ValueError):
            replace(self.cfg.matching,outreach_base_fraction=.25)
        with self.assertRaisesRegex(ValueError,'Stage1 only'):
            load(dict(self.settings,training_stage=2))
        with self.assertRaises(ValueError):
            load(dict(self.settings,target_curriculum={}))

    @unittest.skipUnless(torch.cuda.is_available(),'CUDA not available')
    def test_cuda_formula_selection_and_causality_match_cpu(self):
        drone = self.loop(.08)
        expected = self.score(drone=drone)
        actual = Stage1Scorer(self.rest.cuda(),self.time.cuda(),self.cfg,initial_position=self.origin.cuda())(
            self.positions.cuda(),self.velocities.cuda(),self.valid.cuda(),drone_positions=drone.cuda())
        for name in ('score','causal_best','event_indices'):
            torch.testing.assert_close(getattr(actual,name).cpu(),getattr(expected,name),rtol=1e-8,atol=1e-10)
        for key in ('outreach_mechanics_gate','outreach_speed_factor','outreach_credit','preparation_cost'):
            torch.testing.assert_close(actual.components[key].cpu(),expected.components[key],rtol=1e-8,atol=1e-10)


class LinearOutreachHeightStage1Tests(unittest.TestCase):
    """Approved reward, causal/failure delivery and CPU/CUDA behavior probes."""
    @classmethod
    def setUpClass(cls):
        CompactStage1Tests.setUpClass()
        cls.settings,cls.snapshot,cls.cfg,cls.scene = load(
            'configs/learning/whip_stage1_ppo_outreach_height_radius055_150_v12.json')
        for name in ('legacy','rest','time','origin','positions','velocities','valid','drone'):
            setattr(cls,name,getattr(CompactStage1Tests,name))

    score = CompactStage1Tests.score
    loop = CompactStage1Tests.loop
    test_environment_reward_delivery_and_failure_correction = CompactStage1Tests.test_environment_reward_delivery_and_failure_correction
    test_future_drone_motion_and_invalid_suffix_are_causal = CompactStage1Tests.test_future_travel_and_bad_drone_suffix_cannot_rewrite_earlier_credit
    test_preparation_and_future_height_are_causal_with_invalid_suffix = LevelOutreachStage1Tests.test_preparation_and_future_height_are_causal_with_invalid_suffix

    def test_rigid_translation_and_rotation_cannot_earn_outreach(self):
        material=torch.cat((self.rest.new_zeros(1),self.rest.cumsum(0)))
        angle=.6*torch.sin(2*math.pi*self.time)
        rate=.6*2*math.pi*torch.cos(2*math.pi*self.time)
        direction=torch.stack((angle.sin(),torch.zeros_like(angle),-angle.cos()),-1)
        tangent=torch.stack((angle.cos(),torch.zeros_like(angle),angle.sin()),-1)
        p=self.origin+direction[:,None]*material[None,:,None]
        v=rate[:,None,None]*tangent[:,None]*material[None,:,None]
        self.assertLess(float(self.score(positions=p[None],velocities=v[None]).score[0]),1e-8)
        p=p[:1].expand(451,-1,-1).clone();p[...,0]+=5*self.time[:,None]
        v=torch.zeros_like(p);v[...,0]=5
        self.assertLess(float(self.score(positions=p[None],velocities=v[None],drone=p[None,:,0]).score[0]),1e-8)

    def test_selected_release_linear_height_components_and_bound(self):
        result = self.score(drone=self.loop(.08))
        c = result.components
        i = int(result.event_indices[0,-1])
        dz = self.positions[:,i,-1,2]-self.loop(.08)[:,i,2]
        h = (1-dz.abs()/self.rest.sum()).clamp(0,1)
        r = c['forward_reach_fraction']
        mu = c['mechanical']*c['unfolding_fraction']
        v = c['forward_mps'].clamp_min(0)
        k = mu/(mu+.05)*c['forward_alignment']*(v/(v+3)).square()*c['preparation_factor']
        torch.testing.assert_close(c['tip_drone_height_error_m'],dz)
        torch.testing.assert_close(c['level_height_factor'],h)
        torch.testing.assert_close(c['level_outreach_quality'],r*h)
        torch.testing.assert_close(result.score,k*(.5*r+.5*h+2*r*h))
        torch.testing.assert_close(c['outreach_credit'],
            c['outreach_reach_credit']+c['outreach_height_credit']+c['outreach_joint_credit'])
        torch.testing.assert_close(result.score,c['outreach_credit']-c['preparation_cost'])
        torch.testing.assert_close(result.causal_best[:,-1],result.score)
        self.assertTrue(bool((result.causal_best.diff(dim=1)>=0).all()))
        self.assertGreater(float(result.score[0]),0.)
        self.assertLessEqual(float(result.score[0]),3.)

    def test_height_progress_symmetry_clipping_translation_and_speed_gate(self):
        cfg = replace(self.cfg,matching=replace(self.cfg.matching,preparation_path_weight=0.))
        level = self.drone.clone();level[...,2] = self.positions[..., -1,2]
        previous = float('inf')
        for fraction in (0.,.25,.5,1.,1.5):
            values=[]
            for sign in (-1,1):
                drone = level.clone();drone[...,2] += sign*fraction*self.rest.sum()
                result = self.score(cfg=cfg,drone=drone)
                torch.testing.assert_close(result.components['level_height_factor'],self.rest.new_tensor([max(0,1-fraction)]))
                values.append(result.score)
            torch.testing.assert_close(values[0],values[1])
            self.assertLessEqual(float(values[0][0]),previous)
            previous = float(values[0][0])
        shift = self.rest.new_tensor([0.,0.,3.])
        a=self.score();b=self.score(positions=self.positions+shift,drone=self.drone+shift)
        for key in ('score','causal_best','event_indices'):
            torch.testing.assert_close(getattr(a,key),getattr(b,key),rtol=1e-9,atol=1e-10)
        stopped = self.score(velocities=torch.zeros_like(self.velocities))
        self.assertEqual(float(stopped.score[0]),0.)

    def test_explicit_weights_and_stage_gaussian_guards(self):
        for key in ('outreach_reach_weight','outreach_height_weight','outreach_joint_weight'):
            for value in (None,-.1,float('nan'),float('inf'),True):
                with self.subTest(key=key,value=value),self.assertRaises(ValueError):
                    replace(self.cfg.matching,**{key:value})
            with self.assertRaises(ValueError):replace(self.legacy.matching,**{key:.5})
        with self.assertRaises(ValueError):
            replace(self.cfg.matching,outreach_reach_weight=0.,outreach_height_weight=0.,outreach_joint_weight=0.)
        with self.assertRaises(ValueError):replace(self.cfg.matching,outreach_height_width_fraction=.35)
        with self.assertRaisesRegex(ValueError,'Stage1 only'):load(dict(self.settings,training_stage=2))

    @unittest.skipUnless(torch.cuda.is_available(),'CUDA not available')
    def test_cuda_selected_event_components_and_causality_match_cpu(self):
        drone = self.loop(.08);expected=self.score(drone=drone)
        actual = Stage1Scorer(self.rest.cuda(),self.time.cuda(),self.cfg,initial_position=self.origin.cuda())(
            self.positions.cuda(),self.velocities.cuda(),self.valid.cuda(),drone_positions=drone.cuda())
        for key in ('score','causal_best','event_indices'):
            torch.testing.assert_close(getattr(actual,key).cpu(),getattr(expected,key),rtol=1e-8,atol=1e-10)
        for key in ('outreach_reach_credit','outreach_height_credit','outreach_joint_credit','outreach_credit','preparation_cost'):
            torch.testing.assert_close(actual.components[key].cpu(),expected.components[key],rtol=1e-8,atol=1e-10)


class SpeedOutreachDeliveryStage2Tests(unittest.TestCase):
    """Same-event target guidance, causal accounting and fixed-stage guards."""
    @classmethod
    def setUpClass(cls):
        CompactStage1Tests.setUpClass()
        cls.settings,cls.snapshot,cls.cfg,cls.scene = load(
            'configs/learning/whip_stage2_ppo_outreach_speed_radius040_v5.json')
        for name in ('legacy','rest','time','origin','positions','velocities','valid','drone'):
            setattr(cls,name,getattr(CompactStage1Tests,name))

    score = CompactStage1Tests.score
    loop = CompactStage1Tests.loop
    test_environment_reward_delivery_and_failure_correction = CompactStage1Tests.test_environment_reward_delivery_and_failure_correction
    test_preparation_and_future_height_are_causal_with_invalid_suffix = LevelOutreachStage1Tests.test_preparation_and_future_height_are_causal_with_invalid_suffix

    def test_target_guidance_uses_actual_selected_release_and_discounted_credit(self):
        result = self.score(drone=self.loop(.08))
        c = result.components
        sample = int(result.event_indices[0,-1])
        tip = self.positions[:,sample,-1]
        # Independent distance to the axis-aligned fixed rectangle in world XYZ.
        dx = tip[:,0]-1.1
        dy = (tip[:,1].abs()-.5).clamp_min(0)
        dz = ((tip[:,2]-2.).abs()-.3).clamp_min(0)
        proximity = torch.exp(-.5*(dx.square()+dy.square()+dz.square())/.35**2)
        torch.testing.assert_close(c['target_proximity'],proximity)
        base = c['outreach_credit']-c['preparation_cost']
        torch.testing.assert_close(c['approach_reward'],base*proximity)
        torch.testing.assert_close(result.score,base+c['approach_reward'])
        torch.testing.assert_close(result.causal_best[:,-1],result.score)
        self.assertGreater(float(result.score[0]),0.)
        self.assertLessEqual(float(result.score[0]),2.)
        # Diagnostic W must not replace the preserved outreach objective.
        self.assertGreater(float((result.score-c['mechanics_credit']*(1+proximity)).abs()[0]),1e-5)

    def test_fixed_stage_and_explicit_scale_requirements(self):
        with self.assertRaisesRegex(ValueError,'Stage2 only'):
            load(dict(self.settings,training_stage=1))
        with self.assertRaises(ValueError):
            replace(self.cfg.matching,outreach_base_fraction=.25)
        with self.assertRaises(ValueError):
            replace(self.cfg.matching,outreach_speed_scale_mps=None)
        with self.assertRaises(ValueError):
            replace(self.cfg,strike_surface=None)

    @unittest.skipUnless(torch.cuda.is_available(),'CUDA not available')
    def test_cuda_target_guidance_selection_and_causality_match_cpu(self):
        drone = self.loop(.08)
        expected = self.score(drone=drone)
        actual = Stage1Scorer(self.rest.cuda(),self.time.cuda(),self.cfg,initial_position=self.origin.cuda())(
            self.positions.cuda(),self.velocities.cuda(),self.valid.cuda(),drone_positions=drone.cuda())
        for name in ('score','causal_best','event_indices'):
            torch.testing.assert_close(getattr(actual,name).cpu(),getattr(expected,name),rtol=1e-8,atol=1e-10)
        for key in ('target_proximity','approach_reward','outreach_credit','preparation_cost'):
            torch.testing.assert_close(actual.components[key].cpu(),expected.components[key],rtol=1e-8,atol=1e-10)


class SpeedOutreachPrecisionTests(unittest.TestCase):
    """Approved dense center proxy; synthetic histories are not flight evidence."""
    @classmethod
    def setUpClass(cls):
        from training.curriculum import TargetCurriculum
        CompactStage1Tests.setUpClass()
        cls.settings,cls.snapshot,final,cls.scene=load(
            'configs/learning/whip_stage3_ppo_outreach_speed_radius040_80_v5.json')
        cls.final=final
        cls.cfg=TargetCurriculum().configuration(final,cls.settings)
        for name in ('rest','time','origin','positions','velocities','valid','drone'):
            setattr(cls,name,getattr(CompactStage1Tests,name))

    score=CompactStage1Tests.score
    loop=CompactStage1Tests.loop
    test_environment_reward_delivery_and_failure_correction=CompactStage1Tests.test_environment_reward_delivery_and_failure_correction
    test_preparation_and_future_height_are_causal_with_invalid_suffix=LevelOutreachStage1Tests.test_preparation_and_future_height_are_causal_with_invalid_suffix

    def test_same_observed_release_xyz_center_rectangle_and_accounting(self):
        result=self.score(drone=self.loop(.08));c=result.components
        sample=int(result.event_indices[0,-1]);tip=self.positions[:,sample,-1]
        center=tip.new_tensor([1.1,0,2])
        distance=(tip-center).norm(dim=-1)
        gaussian=torch.exp(-.5*(distance/.25).square())
        dx=tip[:,0]-1.1;dy=(tip[:,1].abs()-.5).clamp_min(0)
        dz=((tip[:,2]-2).abs()-.3).clamp_min(0)
        proximity=torch.exp(-.5*(dx.square()+dy.square()+dz.square())/.35**2)
        base=c['outreach_credit']-c['preparation_cost']
        torch.testing.assert_close(c['center_distance_m'],distance)
        torch.testing.assert_close(c['center_proximity'],gaussian)
        torch.testing.assert_close(c['target_proximity'],proximity)
        torch.testing.assert_close(c['approach_reward'],base*proximity)
        torch.testing.assert_close(c['center_reward'],base*gaussian)
        torch.testing.assert_close(result.score,base*(1+proximity+gaussian))
        torch.testing.assert_close(result.causal_best[:,-1],result.score)
        self.assertTrue(bool((result.causal_best.diff(dim=1)>=0).all()))
        self.assertGreater(float(c['center_reward'][0]),0.)
        self.assertLessEqual(float(result.score[0]),3.)
        # Center guidance is present without relabeling this non-crossing wave.
        self.assertEqual(float(c['geometric_hit_success'][0]),0.)
        self.assertEqual(float(c['crossing_center_quality'][0]),0.)

    def test_roles_and_unchanged_target_sigma_progression(self):
        from training.curriculum import TargetCurriculum
        for stage in (1,2):
            with self.assertRaises(ValueError):load(dict(self.settings,training_stage=stage))
        for level,dimensions,sigma in zip(range(4),[(1,.6),(.6,.4),(.3,.2),(.1,.1)],[.25,.15,.075,.05]):
            cfg=TargetCurriculum(level=level).configuration(self.final,self.settings)
            self.assertEqual((cfg.strike_surface.width_m,cfg.strike_surface.height_m),dimensions)
            self.assertEqual(cfg.matching.center_sigma_m,sigma)
        gate=TargetCurriculum()
        for _ in range(3):gate.observe(self.settings,.8,0.,valid_fraction=1.)
        self.assertTrue(gate.ready(self.settings,valid=True,qualified=True,workspace_cost=0.))
        gate.observe(self.settings,.8,0.,valid_fraction=127/128)
        self.assertFalse(gate.ready(self.settings,valid=True,qualified=True,workspace_cost=0.))
        with self.assertRaises(ValueError):replace(self.cfg.matching,outreach_speed_scale_mps=None)
        with self.assertRaises(ValueError):replace(self.cfg,strike_surface=None)

    def test_recovery_freezes_path_height_center_at_observed_hit(self):
        ScoringTests.setUpClass();fixture=ScoringTests()
        history=list(fixture.recovery_history())
        # Use the positive forward-outreach fixture for the whip prefix. The
        # post-hit quiet fixture still tests event-relative padded recovery.
        history[0][:,:256]=self.positions[:,:256]
        history[1][:,:256]=self.velocities[:,:256]
        history[3][:,:256]=self.loop(.08)[:,:256]
        history[4][:,:256]=0.
        _,_,cfg,_=load('configs/learning/whip_stage4_ppo_outreach_speed_radius040_80_v5.json')
        scorer=make_scorer(self.rest,torch.arange(851,dtype=torch.float64)*.01,cfg,initial_position=self.origin)
        def run(values):
            p,v,valid,dp,dv=values
            return scorer(p,v,valid,drone_positions=dp,drone_velocities=dv,
                recovery_start_samples=torch.tensor([255]),recovery_hit=torch.tensor([True]),
                terminal_samples=valid.long().sum(1))
        result=run(history)
        self.assertTrue(bool(result.valid[0]))
        p,v,valid,dp,dv=history
        base=scorer.base(p[:,:451],v[:,:451],valid[:,:451],
            terminal_samples=torch.tensor([256]),drone_positions=dp[:,:451])
        torch.testing.assert_close(result.components['whip_event_score'],base.score)
        self.assertGreater(float(base.score[0]),0.)
        for key in ('outreach_credit','preparation_cost','approach_reward','center_reward'):
            torch.testing.assert_close(result.components[key],base.components[key])
        changed=[x.clone() for x in history]
        changed[3][:,256:,0]+=.5
        # Recovery can change; frozen whip and its causal prefix cannot.
        other=run(changed)
        torch.testing.assert_close(other.components['whip_event_score'],result.components['whip_event_score'])
        torch.testing.assert_close(other.causal_best[:,:256],result.causal_best[:,:256])
        for index in (0,1,3,4):changed[index][:,656:]=float('nan')
        padded=run(changed)
        self.assertTrue(bool(padded.valid[0]))
        torch.testing.assert_close(padded.score,other.score)
        torch.testing.assert_close(result.score,result.components['whip_event_score']+result.components['recovery_credit'])

    @unittest.skipUnless(torch.cuda.is_available(),'CUDA not available')
    def test_cuda_center_guidance_selection_and_causality_match_cpu(self):
        drone=self.loop(.08);expected=self.score(drone=drone)
        actual=Stage1Scorer(self.rest.cuda(),self.time.cuda(),self.cfg,initial_position=self.origin.cuda())(
            self.positions.cuda(),self.velocities.cuda(),self.valid.cuda(),drone_positions=drone.cuda())
        for name in ('score','causal_best','event_indices'):
            torch.testing.assert_close(getattr(actual,name).cpu(),getattr(expected,name),rtol=1e-8,atol=1e-10)
        for key in ('center_reward','center_proximity','target_proximity','outreach_credit','preparation_cost'):
            torch.testing.assert_close(actual.components[key].cpu(),expected.components[key],rtol=1e-8,atol=1e-10)




class CrossingPriorityStage2Tests(unittest.TestCase):
    """Synthetic reward/timing checks; these trajectories are not physics evidence."""
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)
        CompactStage1Tests.setUpClass()
        cls.settings,cls.snapshot,cls.cfg,cls.scene=load(
            'configs/learning/whip_stage2_ppo_crossing_priority_radius055_150_v10.json')
        cls.rest=CompactStage1Tests.rest
        cls.time=CompactStage1Tests.time
        cls.origin=CompactStage1Tests.origin

    def history(self, events=()):
        material=torch.cat((self.rest.new_zeros(1),self.rest.cumsum(0)))
        p=torch.zeros(1,len(self.time),len(material),3,dtype=torch.float64)
        p[...,0]=.9
        p[...,2]=2+self.rest.sum()-material
        v=torch.zeros_like(p);v[...,0]=6.
        for index,side,height,speed in events:
            p[:,index+1,:,0]=1.3
            p[:,index:index+2,:,1]=side
            p[:,index:index+2,:,2]+=height
            v[:,index:index+2,:,0]=speed
        return p,v,torch.ones(p.shape[:2],dtype=torch.bool),p[:,:,0].clone()

    def score(self, arrays, device='cpu'):
        p,v,valid,drone=[x.to(device) for x in arrays]
        return Stage1Scorer(self.rest.to(device),self.time.to(device),self.cfg,
            initial_position=self.origin.to(device))(p,v,valid,drone_positions=drone)

    def test_hit_miss_strict_and_no_crossing_credit(self):
        cases=[((),0.),(((20,0.,0.,6.),),4.),(((20,0.,0.,2.),),3.),
               (((20,.6,0.,6.),),1/(1+(.6/.25)**2))]
        for events,expected in cases:
            with self.subTest(events=events):
                r=self.score(self.history(events))
                self.assertAlmostEqual(float(r.score[0]),expected,places=12)
                torch.testing.assert_close(r.causal_best[:,-1],r.score)
                c=r.components
                torch.testing.assert_close(r.score,c['whip_support_reward']+c['crossing_center_reward']
                    +c['crossing_geometric_bonus']+c['crossing_strict_bonus'])
                self.assertTrue(bool((r.causal_best.diff(dim=1)>=0).all()))

    def test_credits_one_complete_crossing_not_sum_or_mixed_components(self):
        r=self.score(self.history(((20,0.,0.,2.),(40,.49,.29,6.))))
        expected=3+1/(1+(.49/.25)**2+(.29/.15)**2)
        self.assertAlmostEqual(float(r.score[0]),expected,places=12)
        self.assertAlmostEqual(float(r.components['crossing_reward_lateral_error_m'][0]),.49)
        repeated=self.score(self.history(((20,0.,0.,6.),(40,0.,0.,6.))))
        self.assertEqual(float(repeated.score[0]),4.)

    def test_crossing_waits_for_endpoint_and_ignores_invalid_suffix(self):
        arrays=self.history(((20,0.,0.,6.),))
        full=self.score(arrays)
        self.assertEqual(float(full.causal_best[0,20]),0.)
        self.assertEqual(float(full.causal_best[0,21]),4.)
        p,v,valid,drone=[x.clone() for x in arrays]
        p[:,40:]=float('nan');valid[:,40:]=False
        failed=self.score((p,v,valid,drone))
        self.assertFalse(bool(failed.valid[0]))
        self.assertTrue(bool(torch.isnan(failed.score).all()))
        torch.testing.assert_close(failed.causal_best[:,:40],full.causal_best[:,:40],rtol=0,atol=0)
        valid[:,21:]=False
        unseen=self.score((p,v,valid,drone))
        self.assertTrue(bool((unseen.causal_best==0).all()))

    def test_center_guidance_improves_symmetrically(self):
        scores=[]
        for side,height in ((.6,.4),(.3,.2),(.1,.05),(0.,0.)):
            a=self.score(self.history(((20,side,height,6.),)))
            b=self.score(self.history(((20,-side,-height,6.),)))
            torch.testing.assert_close(a.score,b.score)
            scores.append(float(a.components['crossing_center_reward'][0]))
        self.assertEqual(scores,sorted(scores))
        self.assertEqual(scores[-1],1.)

    def test_support_retains_approved_linear_stage1_record(self):
        source=CompactStage1Tests
        a=(source.positions,source.velocities,source.valid,source.drone)
        r=self.score(a)
        cfg=load('configs/learning/whip_stage1_ppo_outreach_height_radius055_150_v12.json')[2]
        w=Stage1Scorer(self.rest,self.time,cfg,initial_position=self.origin)(
            *a[:3],drone_positions=a[3])
        torch.testing.assert_close(r.components['whip_support_reward'],.2*w.score/3)
        torch.testing.assert_close(r.event_indices,w.event_indices,rtol=0,atol=0)
        self.assertGreater(float(w.score[0]),0.)

    def test_configuration_requires_explicit_weights_and_stage2(self):
        for key in ('crossing_whip_weight','crossing_side_scale_m','crossing_height_scale_m',
                    'crossing_geometric_weight','crossing_strict_weight'):
            for value in (None,0.,-.1,float('nan'),float('inf'),True):
                with self.subTest(key=key,value=value),self.assertRaises(ValueError):
                    replace(self.cfg.matching,**{key:value})
        with self.assertRaisesRegex(ValueError,'Stage2 only'):
            load(dict(self.settings,training_stage=1))
        with self.assertRaises(ValueError):replace(self.cfg,termination='first_qualified_hit_v1')
        with self.assertRaises(ValueError):replace(self.cfg,strike_surface=None)

    @unittest.skipUnless(torch.cuda.is_available(),'CUDA not available')
    def test_cpu_cuda_crossing_accounting_matches(self):
        arrays=self.history(((20,0.,0.,2.),(40,.49,.29,6.)))
        a=self.score(arrays);b=self.score(arrays,'cuda')
        for key in ('score','causal_best','event_indices'):
            torch.testing.assert_close(getattr(a,key),getattr(b,key).cpu(),rtol=1e-8,atol=1e-10)
        for key in ('crossing_reward','whip_support_reward','crossing_center_reward',
                    'crossing_geometric_bonus','crossing_strict_bonus'):
            torch.testing.assert_close(a.components[key],b.components[key].cpu(),rtol=1e-8,atol=1e-10)


if __name__ == '__main__':
    unittest.main()
