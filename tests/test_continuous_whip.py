"""Continuous-reward accounting and kinematic counterexamples, not dynamics tests.

Mocked region rows below isolate causal bend-history/accounting rules. Their
combination with endpoint fixtures is deliberately not a physically simulated
cable trajectory. Retained-recording tests exercise actual saved geometry.
"""
from hashlib import sha256
import math
from pathlib import Path
import unittest

import numpy as np

from tasks.whip.continuous_whip import (
    ContinuousWhipAttempt, FEATURE_SIZE, TRACE_FIELDS, output_metrics,
)


REST = np.full(10, .1)
ROOT = Path(__file__).resolve().parents[1]


def state(reach=.8, tip_velocity=(1., 0., 0.), root_velocity=(0., 0., 0.)):
    p = np.zeros((11, 3))
    p[:, 0] = np.linspace(0., reach, 11)
    p[:, 2] = 2.
    v = np.zeros_like(p)
    v[0] = root_velocity
    v[-1] = tip_velocity
    return p, v


def region(center, *, ident=1, angle=150., coherence=.8, flags=(), unresolved=False):
    return dict(track_id=ident, centre_m=center, theta=math.radians(angle),
                coherence=coherence, flags=list(flags), unresolved=unresolved)


class Harness:
    def __init__(self, rows=None, p=None, v=None):
        self.attempt = ContinuousWhipAttempt(REST)
        self.n = 0
        self.rewards = []
        self.rows = [region(.1)] if rows is None else rows
        self.attempt.geometry.observe = lambda _: (np.zeros(9), self.rows)
        p, v = state() if p is None else (p, v)
        self.attempt.initialize(p, v)

    def step(self, rows=None, p=None, v=None):
        self.n += 1
        self.rows = [region(.1 + self.n * .01)] if rows is None else rows
        p, v = state() if p is None else (p, v)
        reward = self.attempt.observe(self.n * .01, p, v)
        self.rewards.append(reward)
        return reward


class ContinuousWhipTests(unittest.TestCase):
    def test_no_early_completion_and_exact_full_horizon_ledger(self):
        h = Harness()
        for n in range(1, 701):
            # A short pulse of transport then no bend; no event ends the episode.
            h.step(rows=[region(.1+n*.01)] if n <= 35 else [])
            if n < 700:
                self.assertIsNone(h.attempt.terminal)
        a = h.attempt
        self.assertEqual(a.terminal, 'horizon')
        self.assertEqual(a.time, 7.)
        self.assertGreater(a.total, 0.)
        self.assertAlmostEqual(sum(h.rewards), a.total, places=12)
        self.assertAlmostEqual(a.accrued, a.total, places=12)
        self.assertEqual(len(a.trace), 701)
        self.assertEqual(np.asarray(a.trace).shape, (701, len(TRACE_FIELDS)))
        self.assertEqual(a.features().shape, (FEATURE_SIZE,))
        self.assertIsNone(a.record()['qualified'])
        self.assertEqual([e['kind'] for e in a.log], ['horizon'])
        with self.assertRaises(RuntimeError):
            h.step()

    def test_same_outward_motion_can_repeat_without_count_cap(self):
        h = Harness()
        totals = []
        for n in range(1, 151):
            cycle = n // 50
            phase = (n % 50) * .01
            # Outward opening then return, with fresh measured bend identity.
            if phase <= .25:
                reach = .3 + 2*phase
                velocity = 2.
            else:
                reach = .8 - 2*(phase-.25)
                velocity = -2.
            p, v = state(reach=reach, tip_velocity=(velocity, 0., 0.))
            # The fold leaves during recovery; the gap exceeds support memory,
            # so each repeated outbound segment has the same initial history.
            rows = [region(.1+phase, ident=cycle+1)] if phase <= .25 else []
            h.step(rows=rows, p=p, v=v)
            if n % 50 == 0:
                totals.append(h.attempt.total)
        first, second, third = np.diff([0.] + totals)
        self.assertGreater(first, 0.)
        self.assertAlmostEqual(second, first, places=12)
        self.assertAlmostEqual(third, first, places=12)
        self.assertIsNone(h.attempt.terminal)

    def test_static_extension_and_rigid_translation_earn_zero_even_with_support(self):
        for velocity in ((0., 0., 0.), (3., -2., .4)):
            p, v = state(reach=1., tip_velocity=velocity, root_velocity=velocity)
            h = Harness(p=p, v=v)
            for _ in range(20):
                h.step(p=p, v=v)
            self.assertGreater(h.attempt.last['fold_support'], .99)
            self.assertEqual(h.attempt.total, 0.)
            self.assertEqual(h.attempt.last['straightening_mps'], 0.)

    def test_rigid_rotation_is_rejected_by_endpoint_opening_not_just_fold_history(self):
        angles = np.linspace(-.5, .5, 10)
        base = np.vstack([np.zeros(3), (REST[:, None]*np.c_[np.sin(angles),
                    np.zeros(10), -np.cos(angles)]).cumsum(axis=0)])
        maximum_radial = 0.
        accumulated = 0.
        for theta in np.linspace(-1., 1., 101):
            rotation = np.array([[math.cos(theta), 0., math.sin(theta)],
                                 [0., 1., 0.], [-math.sin(theta), 0., math.cos(theta)]])
            p = base @ rotation.T
            v = np.cross(np.array([0., 2., 0.]), p)
            m = output_metrics(p, v, 1.)
            maximum_radial = max(maximum_radial, m['relative_outward_mps'])
            accumulated += m['effective_outward_mps']
            self.assertAlmostEqual(m['straightening_mps'], 0., places=12)
        self.assertGreater(maximum_radial, .5)  # A horizontal-speed-only reward would pay.
        self.assertLess(accumulated, 1e-12)

    def test_azimuth_rotation_preserves_output_metrics(self):
        p, v = state()
        p[-1] = [.6, .3, 2.2]
        v[-1] = [1.4, .8, .2]
        v[0] = [.1, -.2, .1]
        expected = output_metrics(p, v, 1.)
        for angle in (.3, 1.2, math.pi, -2.3):
            rotation = np.array([[math.cos(angle), -math.sin(angle), 0.],
                                 [math.sin(angle), math.cos(angle), 0.], [0., 0., 1.]])
            actual = output_metrics(p @ rotation.T, v @ rotation.T, 1.)
            for key in expected:
                self.assertAlmostEqual(actual[key], expected[key], places=12, msg=key)

    def test_output_progress_is_continuous_across_old_45_degree_boundary(self):
        values = []
        for vz in (.99, 1., 1.01):
            p, v = state(tip_velocity=(1., 0., vz))
            m = output_metrics(p, v, 1.)
            values.append(10*m['horizontal_reach_fraction']**2*m['alignment']**2*m['effective_outward_mps'])
        self.assertTrue(all(value > 0 for value in values))
        self.assertLess(max(values)-min(values), .1)
        self.assertGreater(values[0], values[1])
        self.assertGreater(values[1], values[2])

    def test_soft_fold_strength_has_no_bonus_jump_at_90_degrees(self):
        credits = []
        for angle in (89.999, 90., 90.001, 120., 150.):
            h = Harness(rows=[region(.1, angle=angle)])
            for n in range(1, 11):
                h.step(rows=[region(.1+n*.01, angle=angle)])
            credits.append(h.attempt.total)
        self.assertEqual(credits[:2], [0., 0.])
        self.assertGreater(credits[2], 0.)
        self.assertLess(credits[2], 1e-5)
        self.assertLess(credits[2], credits[3])
        self.assertLess(credits[3], credits[4])

    def test_support_fades_in_100ms_without_replenishment(self):
        h = Harness()
        for _ in range(10):
            h.step()
        self.assertAlmostEqual(h.attempt.last['fold_support'], 1.)
        values = []
        for _ in range(10):
            h.step(rows=[])
            values.append(h.attempt.last['fold_support'])
        self.assertAlmostEqual(values[4], .5, places=12)
        self.assertAlmostEqual(values[-1], 0., places=12)
        self.assertTrue(all(a >= b for a, b in zip(values, values[1:])))

    def test_ambiguous_or_reappearing_track_cannot_reuse_transport_history(self):
        for interruption in ('ambiguous_identity', 'plateau', 'missing', 'new_id'):
            h = Harness()
            for _ in range(9):
                h.step()
            rows = [] if interruption == 'missing' else [region(.2, ident=2 if interruption == 'new_id' else 1,
                     flags=() if interruption == 'new_id' else [interruption])]
            h.step(rows=rows)
            self.assertEqual(h.attempt.last['transport_signal'], 0.)
            for n in range(11, 20):
                h.step(rows=[region(.1+n*.01)])
                self.assertEqual(h.attempt.last['transport_signal'], 0.)

    def test_backward_or_incoherent_bend_does_not_build_support(self):
        for backward, coherence in ((True, .8), (False, 0.)):
            h = Harness(rows=[region(.5, coherence=coherence)])
            for n in range(1, 21):
                h.step(rows=[region(.5+(-1 if backward else 1)*n*.01, coherence=coherence)])
            self.assertEqual(h.attempt.total, 0.)
            self.assertEqual(h.attempt.last['fold_support'], 0.)

    def test_failure_cancels_all_prefix_credit_and_failure_time_has_no_bonus(self):
        totals = []
        for steps in (12, 30):
            h = Harness()
            for _ in range(steps):
                h.step()
            self.assertGreater(h.attempt.accrued, 0.)
            accrued = h.attempt.accrued
            correction = h.attempt.fail(h.n*.01+.004, 'workspace_failure')
            totals.append(sum(h.rewards)+correction)
            self.assertAlmostEqual(correction, -10.-accrued, places=12)
            self.assertEqual(h.attempt.total, -10.)
            self.assertEqual(h.attempt.accrued, accrued)
            self.assertEqual(h.attempt.record()['events'][-1]['cancelled_credit'], accrued)
            with self.assertRaises(RuntimeError):
                h.step()
        np.testing.assert_allclose(totals, [-10., -10.], atol=1e-12, rtol=0)

    def test_time_and_finite_input_guards(self):
        p, v = state()
        a = ContinuousWhipAttempt(REST)
        with self.assertRaises(RuntimeError):
            a.observe(.01, p, v)
        a.initialize(p, v)
        with self.assertRaises(RuntimeError):
            a.initialize(p, v)
        with self.assertRaises(ValueError):
            a.observe(.02, p, v)
        p[3, 0] = np.nan
        with self.assertRaises(ValueError):
            a.observe(.01, p, v)

    def test_retained_downward_trace_and_rigid_loop_are_not_relabelled_success(self):
        paths = [ROOT/'runs/ppo_stage2/20261025-133812-161285/round-00052.npz',
                 ROOT/'tmp/tangent-evolution-review-20261025/review/rigid_loop.npz',
                 ROOT/'tmp/stage4-tangent-review-20261025/review/stage4_full.npz']
        if not all(path.exists() for path in paths):
            self.skipTest('Retained local diagnostic fixtures are not installed')
        totals = []
        for index, path in enumerate(paths):
            before = sha256(path.read_bytes()).hexdigest()
            with np.load(path, allow_pickle=False) as z:
                time = z['time_s'] if 'time_s' in z else z['time']
                p = z['cable_positions'] if 'cable_positions' in z else z['positions']
                v = z['cable_velocities'] if 'cable_velocities' in z else z['velocities']
                rest = np.linalg.norm(np.diff(p[0], axis=0), axis=1) if 'rest' not in z else z['rest']
            a = ContinuousWhipAttempt(rest)
            a.initialize(p[0], v[0])
            rewards = [a.observe(float(t), pp, vv) for t, pp, vv in zip(time[1:], p[1:], v[1:])]
            self.assertAlmostEqual(a.total, sum(rewards), places=12)
            self.assertIsNone(a.record()['qualified'])
            self.assertEqual(sha256(path.read_bytes()).hexdigest(), before)
            totals.append(a.total)
        self.assertGreater(totals[0], 0.)
        self.assertLess(totals[0], .02)
        self.assertLess(abs(totals[1]), 1e-12)
        self.assertGreater(totals[2], .1)  # Nonzero on retained fold reference, not policy validation.


if __name__ == '__main__':
    unittest.main()
