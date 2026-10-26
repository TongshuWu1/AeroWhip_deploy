"""Constructed detector/accounting examples, not physical whip validation."""
import math
import unittest
import numpy as np
from tasks.whip.retained_fold_delivery import (
    RetainedFoldDeliveryAttempt, delivery_sample, FEATURE_SIZE)


class RetainedDeliveryTests(unittest.TestCase):
    def setUp(self):
        self.a = RetainedFoldDeliveryAttempt(np.full(10, .1))
        self.t = 0
        self.rewards = []
        self.positions, self.velocities = self.state(.9)
        self.a.initialize(self.positions, self.velocities)

    @staticmethod
    def state(tip_x, root_x=.25, tip_y=0., tip_z=2., vx=2., root_vx=0., vz=0.):
        p = np.linspace([root_x, 0., 2.], [tip_x, tip_y, tip_z], 11)
        v = np.zeros((11, 3))
        v[0, 0] = root_vx
        v[-1] = [vx, 0., vz]
        return p, v

    def step(self, centre=.3, turn=3., ident=1, flags=(), rows=True,
             coherence=1., tip_x=.9, **state):
        row = dict(track_id=ident, centre_m=centre, theta=turn, coherence=coherence,
                   joint_indices=[1], unresolved=False, flags=list(flags))
        self.a.geometry.observe = lambda _: (np.array([turn] + [0.] * 8), [row] if rows else [])
        self.t += 1
        self.positions, self.velocities = self.state(tip_x, **state)
        r = self.a.observe(self.t * .01, self.positions, self.velocities)
        self.rewards.append(r)
        return r

    def arm(self, centre=.3):
        for _ in range(3):
            self.step(centre=centre)
        self.assertTrue(self.a.formed)

    def propagate(self):
        self.arm()
        self.step(centre=.7, turn=2.8)
        self.assertTrue(self.a.propagated)

    def opening(self, **kw):
        self.propagate()
        for _ in range(3):
            self.step(centre=.8, turn=.5, **kw)
        self.assertTrue(self.a.opened)

    def assert_ledger(self):
        self.assertAlmostEqual(sum(self.rewards), self.a.total, places=12)
        self.assertAlmostEqual(self.a.total, self.a.progress_credit + self.a.outcome_credit, places=12)

    def test_reset_and_time_validation(self):
        b = RetainedFoldDeliveryAttempt(np.full(10, .1))
        with self.assertRaises(RuntimeError):
            b.observe(.01, self.positions, self.velocities)
        with self.assertRaises(RuntimeError):
            self.a.initialize(self.positions, self.velocities)
        with self.assertRaises(ValueError):
            self.a.observe(.02, self.positions, self.velocities)

    def test_formation_partial_is_retained_but_not_formed(self):
        self.step(turn=math.radians(75), coherence=.8)
        self.assertAlmostEqual(self.a.progress, .5 / 3 / 4)
        self.assertFalse(self.a.formed)
        for _ in range(2):
            self.step(turn=math.radians(75), coherence=.8)
        self.assertAlmostEqual(self.a.progress, .125)
        self.assertFalse(self.a.formed)
        self.assert_ledger()

    def test_formation_identity_change_restarts_persistence(self):
        self.step()
        self.step()
        previous = self.a.progress
        self.step(ident=2)
        self.assertFalse(self.a.formed)
        self.assertEqual(self.a.progress, previous)
        self.step(ident=2)
        self.step(ident=2)
        self.assertTrue(self.a.formed)
        self.assertEqual(self.a.active['id'], 2)

    def test_nonproximal_or_ambiguous_bend_has_no_formation_credit(self):
        for _ in range(3):
            self.step(centre=.6)
        self.assertEqual(self.a.progress, 0.)
        for _ in range(3):
            self.step(flags=['ambiguous_identity'])
        self.assertEqual(self.a.progress, 0.)

    def test_propagation_guide_needs_both_gates(self):
        self.arm(centre=.2)
        self.step(centre=.4)
        self.assertFalse(self.a.propagated)
        self.assertAlmostEqual(self.a.p, .2 / .45)
        self.step(centre=.65)
        self.assertTrue(self.a.propagated)

    def test_distal_position_cannot_replace_required_travel(self):
        self.arm(centre=.5)
        self.step(centre=.65)
        self.assertFalse(self.a.propagated)
        self.assertAlmostEqual(self.a.p, .75)
        self.step(centre=.71)
        self.assertTrue(self.a.propagated)

    def test_candidate_loss_keeps_credit_and_new_candidate_cannot_repay(self):
        self.arm()
        before = self.a.total
        self.step(ident=2, flags=['ambiguous_identity'])
        self.assertIsNone(self.a.active)
        for _ in range(3):
            self.step(ident=2)
        self.assertEqual(self.a.total, before)
        self.assertEqual(self.a.progress_source_event_id, 1)

    def test_backward_travel_does_not_replenish_credit(self):
        self.arm()
        self.step(centre=.45)
        before = self.a.total
        self.step(centre=.35)
        self.step(centre=.45)
        self.assertEqual(self.a.total, before)

    def test_lost_identity_low_samples_are_provisional(self):
        self.propagate()
        before = self.a.progress
        self.step(centre=.8, turn=.5, rows=False)
        self.step(centre=.8, turn=.5, rows=False)
        self.assertEqual(self.a.progress, before)
        self.step(centre=.8, turn=2.8)
        self.assertIsNone(self.a.active)
        self.assertEqual(self.a.progress, before)
        self.assertFalse(self.a.opened)

    def test_bend_may_disappear_during_confirmed_opening(self):
        self.propagate()
        for _ in range(3):
            self.step(centre=.8, turn=.5, rows=False)
        self.assertTrue(self.a.opened)
        self.assertGreaterEqual(self.a.progress, .75)
        self.assert_ledger()

    def test_opening_guide_has_no_running_peak_inflation(self):
        self.propagate()
        for _ in range(3):
            self.step(centre=.8, turn=math.radians(300))
        for _ in range(3):
            self.step(centre=.8, turn=math.radians(105))
        self.assertAlmostEqual(self.a.o, .5)
        self.assertFalse(self.a.opened)

    def test_output_requires_both_velocity_frames(self):
        self.opening(vx=2., root_vx=3.)
        self.assertIsNone(self.a.selected)
        self.assertEqual(self.a.progress, .75)
        self.assertEqual(self.a.terminal, 'event_end')
        self.assertAlmostEqual(self.a.total, 1.75)

    def test_same_sample_alignment_and_distance_guide(self):
        p, v = self.state(1.05)
        a = delivery_sample(p, v, 1.)
        v[-1] = [2., 0., 3.]
        b = delivery_sample(p, v, 1.)
        self.assertGreater(a['delivery_guide'], b['delivery_guide'])
        v[-1] = [-2., 0., 0.]
        self.assertEqual(delivery_sample(p, v, 1.)['delivery_guide'], 0.)

    def test_hit_in_opening_buffer_waits_for_confirmation(self):
        self.propagate()
        self.step(centre=.8, turn=.5, tip_x=1.05)
        self.step(centre=.8, turn=.5, tip_x=1.15)
        self.assertIsNone(self.a.terminal)
        self.assertFalse(self.a.crossings[-1]['opening_confirmed'])
        self.step(centre=.8, turn=.5, tip_x=1.16)
        self.assertTrue(self.a.qualified)
        self.assertEqual(self.a.total, 15.)
        self.assertAlmostEqual(self.a.selected['time_s'], .055)
        self.assertEqual(self.a.time, .07)
        self.assert_ledger()

    def test_hit_within_confirmed_opening(self):
        self.opening(tip_x=1.05)
        self.step(centre=.8, turn=.5, tip_x=1.15)
        self.assertTrue(self.a.qualified)
        self.assertEqual(self.a.total, 15.)
        self.assert_ledger()

    def test_unconfirmed_opening_crossing_does_not_become_hit(self):
        self.propagate()
        self.step(centre=.8, turn=.5, tip_x=1.05)
        self.step(centre=.8, turn=.5, tip_x=1.15)
        self.step(centre=.8, turn=2.8, tip_x=1.16)
        self.assertFalse(self.a.qualified)
        self.assertEqual(self.a.active['pending_hits'], [])

    def test_safety_failure_preempts_pending_hit(self):
        self.propagate()
        self.step(centre=.8, turn=.5, tip_x=1.05)
        self.step(centre=.8, turn=.5, tip_x=1.15)
        progress = self.a.progress
        self.rewards.append(self.a.fail(.061, 'workspace_failure'))
        self.assertFalse(self.a.qualified)
        self.assertEqual(self.a.progress, progress)
        self.assertAlmostEqual(self.a.total, 5 * progress - 10)
        self.assert_ledger()

    def test_crossing_outside_fold_not_counted(self):
        self.step(turn=.1, tip_x=1.15)
        self.assertFalse(self.a.qualified)
        self.assertIn('outside_same_opening', self.a.crossings[-1]['reasons'])

    def test_crossing_into_first_low_sample_precedes_output_interval(self):
        self.propagate()
        self.step(centre=.8, turn=.5, tip_x=1.15)
        self.assertIn('outside_same_opening', self.a.crossings[-1]['reasons'])
        for _ in range(2):
            self.step(centre=.8, turn=.5, tip_x=1.16)
        self.assertFalse(self.a.qualified)

    def test_slow_touch_and_reverse_are_not_hits(self):
        self.opening(tip_x=1.05)
        self.step(centre=.8, turn=.5, tip_x=1.1)
        self.assertIsNone(self.a.terminal)
        self.step(centre=.8, turn=.5, tip_x=1.05)
        self.assertEqual(self.a.crossings, [])

    def test_speed_floor_uses_interpolated_velocity(self):
        self.opening(tip_x=1.05, vx=.5)
        self.step(centre=.8, turn=.5, tip_x=1.15, vx=.5)
        self.assertFalse(self.a.qualified)
        self.assertIn('forward_speed_below_floor', self.a.crossings[-1]['reasons'])

    def test_plane_touch_followed_by_positive_passage_is_confirmed(self):
        self.opening(tip_x=1.05)
        self.step(centre=.8, turn=.5, tip_x=1.1)
        self.assertIsNone(self.a.terminal)
        self.step(centre=.8, turn=.5, tip_x=1.15)
        self.assertTrue(self.a.qualified)
        self.assertAlmostEqual(self.a.selected['time_s'], .08)

    def test_starting_on_plane_is_not_a_forward_crossing(self):
        self.a = RetainedFoldDeliveryAttempt(np.full(10, .1))
        self.a.initialize(*self.state(1.1))
        for _ in range(3):
            self.step(tip_x=1.1)
        self.step(centre=.7, turn=2.8, tip_x=1.1)
        for _ in range(3):
            self.step(centre=.8, turn=.5, tip_x=1.1)
        self.step(centre=.8, turn=.5, tip_x=1.15)
        self.assertFalse(self.a.qualified)

    def test_preopening_touch_cannot_be_retimed_by_holding_on_plane(self):
        self.arm()
        self.step(centre=.7, turn=2.8, tip_x=1.1)
        for _ in range(4):
            self.step(centre=.8, turn=.5, tip_x=1.1)
        self.step(centre=.8, turn=.5, tip_x=1.15)
        self.assertFalse(self.a.qualified)
        self.assertAlmostEqual(self.a.crossings[-1]['time_s'], .04)
        self.assertIn('outside_same_opening', self.a.crossings[-1]['reasons'])

    def test_touch_velocity_is_not_replaced_by_later_peak(self):
        self.opening(tip_x=1.05, vx=.5)
        self.step(centre=.8, turn=.5, tip_x=1.1, vx=.5)
        self.step(centre=.8, turn=.5, tip_x=1.1, vx=4.)
        self.step(centre=.8, turn=.5, tip_x=1.15, vx=4.)
        self.assertFalse(self.a.qualified)
        self.assertIn('forward_speed_below_floor', self.a.crossings[-1]['reasons'])

    def test_hit_at_deadline_precedes_timeout(self):
        self.opening(tip_x=1.05)
        while self.t < 699:
            self.step(centre=.8, turn=.5, tip_x=1.05)
        self.step(centre=.8, turn=.5, tip_x=1.15)
        self.assertEqual(self.a.terminal, 'hit')
        self.assertEqual(self.a.time, 7.)
        self.assertEqual(self.a.total, 15.)
        self.assert_ledger()

    def test_rectangle_miss_does_not_count(self):
        self.opening(tip_x=1.05, tip_y=.7)
        self.step(centre=.8, turn=.5, tip_x=1.15, tip_y=.7)
        self.assertFalse(self.a.qualified)
        self.assertIn('outside_rectangle', self.a.crossings[-1]['reasons'])

    def test_reach_floor_is_not_replaced_by_target_proximity(self):
        self.opening(tip_x=1.05, root_x=.5)
        self.step(centre=.8, turn=.5, tip_x=1.15, root_x=.5)
        self.assertFalse(self.a.qualified)
        self.assertIn('insufficient_reach', self.a.crossings[-1]['reasons'])

    def test_failure_has_no_time_cost_and_retains_progress(self):
        self.arm()
        before = self.a.progress
        self.rewards.append(self.a.fail(.031, 'workspace_failure'))
        self.assertEqual(self.a.total, 5 * before - 10)
        self.assert_ledger()
        with self.assertRaises(RuntimeError):
            self.step()

    def test_completed_miss_and_timeout_share_terminal_cost(self):
        self.opening()
        progress = self.a.progress
        for _ in range(3):
            self.step(centre=.8, turn=.5, vx=-1.)
        self.assertEqual(self.a.terminal, 'event_end')
        self.assertEqual(self.a.total, 5 * progress - 2)
        self.assert_ledger()

    def test_safe_no_progress_timeout_is_minus2(self):
        for _ in range(700):
            self.step(rows=False)
        self.assertEqual(self.a.terminal, 'timeout')
        self.assertEqual(self.a.total, -2.)
        self.assert_ledger()

    def test_features_and_record_are_finite_and_include_ledger(self):
        self.arm()
        self.assertEqual(self.a.features().shape, (FEATURE_SIZE,))
        self.assertTrue(np.isfinite(self.a.features()).all())
        self.rewards.append(self.a.fail(.031))
        record = self.a.record()
        self.assertEqual(record['progress_credit'], 5 * record['progress'])
        self.assertEqual(record['progress_source_event_id'], 1)
        self.assertEqual(record['terminal_event_id'], 1)


if __name__ == '__main__':
    unittest.main()
