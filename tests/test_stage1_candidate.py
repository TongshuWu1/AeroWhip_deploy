"""Accounting/causality checks, not evidence that the candidate identifies whips."""
import unittest

import numpy as np

from evaluation.stage1_candidate import detect, evaluate
from evaluation.whip_measurement_examples import kinematic_controls


def measurements():
    count = 51
    curvature = np.zeros((count, 9))
    for frame in range(count):
        curvature[frame, min(1+frame, 7)] = 3.
    speed = np.full((count, 11), .1)
    speed[12, -1] = 3.
    return dict(time_s=np.arange(count)*.01, node_s_m=np.arange(11)*.1,
                curvature_per_m=curvature, curvature_dual_length_m=np.full_like(curvature, .1),
                relative_speed_mps=speed, world_forward_mps=np.ones_like(speed),
                nonrigid_speed_mps=np.full_like(speed, .2))


class CandidateTests(unittest.TestCase):
    def test_fixed_window_gain_and_directed_score(self):
        r = detect(measurements())
        self.assertEqual(len(r['events']), 1)
        e = r['events'][0]
        self.assertEqual((e['arming_index'], e['end_index'], e['q_index']), (6, 26, 12))
        self.assertEqual(e['score'], 1.)
        self.assertEqual(e['baseline_index'], 0)
        self.assertAlmostEqual(e['signed_gain_mps'], 2.9)
        self.assertEqual(np.flatnonzero(r['reward_increments_100hz']).tolist(), [26])
        self.assertEqual(r['mechanism_status'], 'unresolved')

    def test_prefix_does_not_finalize_early(self):
        m = measurements()
        m = {k: v[:20] if k != 'node_s_m' else v for k, v in m.items()}
        r = detect(m, complete=False)
        self.assertFalse(r['events'][0]['finalized'])
        self.assertEqual(sum(r['reward_increments_100hz']), 0.)
        self.assertIsNone(r['episode_return'])

    def test_valid_horizon_clips_but_failure_does_not(self):
        m = measurements()
        m = {k: v[:20] if k != 'node_s_m' else v for k, v in m.items()}
        good = detect(m)
        self.assertTrue(good['events'][0]['truncated'])
        self.assertEqual(good['events'][0]['end_index'], 19)
        bad = detect(m, failed=True)
        self.assertFalse(bad['events'][0]['finalized'])
        self.assertEqual(bad['episode_return'], -10.)

    def test_failure_claws_back_prior_credit(self):
        r = detect(measurements(), failed=True)
        self.assertEqual(r['reward_increments_100hz'][26], 1.)
        self.assertEqual(r['reward_increments_100hz'][-1], -11.)
        self.assertEqual(sum(r['reward_increments_100hz']), -10.)

    def test_future_ambiguity_cannot_erase_frozen_event(self):
        m = measurements()
        before = detect(m)['events'][0]
        m['curvature_per_m'][8:10, 6:9] = 3.
        after = detect(m)['events'][0]
        self.assertEqual(before, after)

    def test_historical_nonrigid_evidence_is_sufficient(self):
        m = measurements()
        m['nonrigid_speed_mps'][2:] = 0.
        self.assertTrue(detect(m)['events'][0]['provisional_eligible'])

    def test_backward_and_tiny_tracks_do_not_arm(self):
        for mode in ('backward', 'tiny', 'rigid', 'stationary'):
            m = measurements()
            if mode == 'backward':
                m['curvature_per_m'] = m['curvature_per_m'][:, ::-1]
            elif mode == 'tiny':
                m['curvature_per_m'] *= .01
            elif mode == 'rigid':
                m['nonrigid_speed_mps'][:] = 0.
            else:
                m['curvature_per_m'][:] = m['curvature_per_m'][0]
            with self.subTest(mode=mode):
                self.assertEqual(detect(m)['events'], [])

    def test_q_sample_controls_direction_and_earliest_exact_tie(self):
        m = measurements()
        m['relative_speed_mps'][14, -1] = 3.
        m['world_forward_mps'][12, -1] = -1.
        m['world_forward_mps'][14, -1] = 5.
        e = detect(m)['events'][0]
        self.assertEqual(e['q_index'], 12)
        self.assertEqual(e['score'], 0.)

    def test_signed_gain_is_required(self):
        m = measurements()
        m['relative_speed_mps'][0, -1] = 3.
        e = detect(m)['events'][0]
        self.assertFalse(e['provisional_eligible'])
        self.assertEqual(e['score'], 0.)

    def test_repeated_events_only_pay_record_improvements(self):
        m = measurements()
        m['curvature_per_m'][28:] = m['curvature_per_m'][:23]
        m['relative_speed_mps'][40, -1] = 3.
        r = detect(m)
        self.assertEqual(len(r['events']), 2)
        self.assertEqual(sum(r['reward_increments_100hz']), 1.)

    def test_existing_kinematic_negative_controls(self):
        for case in kinematic_controls(np.full(10, .09525)):
            with self.subTest(case=case['name']):
                r = evaluate(case['arrays'], np.full(10, .09525), [1, 0, 0])
                self.assertEqual(r['episode_return'], 0.)

    def test_observation_gaps_are_not_silently_bridged(self):
        m = measurements()
        m['time_s'][20:] += .01
        with self.assertRaises(ValueError):
            detect(m)


if __name__ == '__main__':
    unittest.main()
