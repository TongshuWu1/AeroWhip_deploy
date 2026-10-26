"""Event-level measurement counterexamples, not physical-whip success tests.

All motions below are analytic kinematic fixtures. No physics, policy rollout,
training reward or acceptance threshold is validated by these checks.
"""
from copy import deepcopy
import unittest

import numpy as np

from evaluation.whip_attempt_review import evaluate_attempt
from evaluation.whip_measurement_examples import kinematic_controls


PROVENANCE = dict(reviewer='analytic_test_fixture', method='construction_interval',
                  notes='Bounds come from fixture construction, not peak searching.')
SURFACE = dict(center_world_m=[1.1, 0., 2.], normal_world=[1., 0., 0.],
               width_m=1., height_m=.6)


def release(start, end):
    return dict(start_index=start, end_index=end, provenance=deepcopy(PROVENANCE))


class WhipAttemptReviewTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.rest = np.full(10, .1)
        cls.controls = {case['name']: case['arrays'] for case in kinematic_controls(cls.rest)}

    def read(self, arrays, start=0, end=None, **kwargs):
        return evaluate_attempt(arrays, self.rest, SURFACE, start_index=start,
            end_index=len(arrays['time_s'])-1 if end is None else end,
            selection_provenance=deepcopy(PROVENANCE), **kwargs)

    def crossing(self, x):
        """Vertical chain translated in X; tip sits at target Z and Y."""
        x = np.asarray(x, float)
        time = .1*np.arange(len(x))
        p = np.broadcast_to(np.asarray(SURFACE['center_world_m']),
                            (len(x), len(self.rest)+1, 3)).copy()
        p[:, :, 0] += x[:, None]
        p[:, :, 2] += self.rest.sum()-np.r_[0., np.cumsum(self.rest)]
        v = np.zeros_like(p)
        v[:, :, 0] = np.gradient(x, time)[:, None]
        return dict(time_s=time, cable_positions=p, cable_velocities=v)

    def test_rigid_rotation_keeps_tip_output_but_removes_shape_motion(self):
        report, trace = self.read(self.controls['rigid_curved_rotation'])
        self.assertGreater(trace['relative_speed_mps'][:, -1].max(), 1.)
        np.testing.assert_allclose(trace['shape_speed_mps'], 0., atol=1e-13, rtol=0)
        for record in (report, report['excitation'], report['distal_response'], report['delivery']):
            for key in ('score', 'reward', 'is_whip', 'whip_success', 'task_success'):
                self.assertNotIn(key, record)

    def test_whole_cable_translation_is_not_relative_response(self):
        report, trace = self.read(self.controls['translated_curved_shape'], start=5, end=30)
        np.testing.assert_allclose(trace['world_speed_mps'], .8, atol=1e-14, rtol=0)
        np.testing.assert_array_equal(trace['relative_speed_mps'], 0.)
        np.testing.assert_array_equal(trace['shape_speed_mps'], 0.)
        output = report['distal_response']['output_sample']
        self.assertEqual(output['index'], 5)
        self.assertEqual(output['world_speed_mps'], .8)
        self.assertEqual(output['relative_speed_mps'], 0.)

    def test_backward_or_off_rectangle_crossing_is_not_forward_geometric_hit(self):
        backward = self.crossing([.2, .1, -.1, -.2])
        off_rectangle = self.crossing([-.2, -.1, .1, .2])
        off_rectangle['cable_positions'][:, :, 1] = 2.
        for arrays in (backward, off_rectangle):
            with self.subTest(backward=arrays is backward):
                report, _ = self.read(arrays, release_interval=release(0, 3))
                self.assertEqual(report['delivery']['release_crossing_ids'], [0])
                self.assertEqual(report['delivery']['geometric_hit_ids'], [])

    def test_failed_trial_is_measured_and_retained_without_promotion(self):
        outcome = dict(status='failed', reason='Synthetic later failure retained in denominator.')
        report, trace = self.read(self.crossing([-.2, -.1, .1, .2]),
                                 release_interval=release(0, 3), trial_outcome=outcome)
        self.assertEqual(report['trial_outcome'], outcome)
        self.assertEqual(report['delivery']['geometric_hit_ids'], [0])
        self.assertEqual(len(trace['time_s']), 4)
        self.assertNotIn('success', report)

if __name__ == '__main__':
    unittest.main()
