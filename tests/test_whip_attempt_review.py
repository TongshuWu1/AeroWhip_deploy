"""Event-level measurement counterexamples, not physical-whip success tests.

All motions below are analytic kinematic fixtures. No physics, policy rollout,
training reward or acceptance threshold is validated by these checks.
"""
from copy import deepcopy
import json
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

    def test_sidearm_rotation_preserves_kinematic_amplitude(self):
        arrays = self.controls['small_travelling_wave']
        # Rotate an XZ motion into XY about X. The target normal remains +X;
        # no assertion that its target intersection remains the same is made.
        rotation = np.array([[1., 0., 0.], [0., 0., -1.], [0., 1., 0.]])
        changed = dict(arrays,
            cable_positions=arrays['cable_positions']@rotation.T,
            cable_velocities=arrays['cable_velocities']@rotation.T)
        a, trace_a = self.read(arrays)
        b, trace_b = self.read(changed)
        for field in ('world_speed_mps', 'relative_speed_mps', 'shape_speed_mps', 'curvature_per_m'):
            np.testing.assert_allclose(trace_a[field], trace_b[field], atol=1e-13, rtol=0)
        for frame in ('world', 'attachment_relative', 'attachment_anchored_shape'):
            values_a = [n['peak_speed_mps'] for n in a['distal_response']['peak_timing'][frame]['nodes']]
            values_b = [n['peak_speed_mps'] for n in b['distal_response']['peak_timing'][frame]['nodes']]
            np.testing.assert_allclose(values_a, values_b, atol=1e-13, rtol=0)

    def test_small_wave_retains_raw_amplitude_and_does_not_become_success(self):
        report, trace = self.read(self.controls['small_travelling_wave'])
        output = report['distal_response']['output_sample']
        self.assertGreater(output['relative_speed_mps'], 0.)
        self.assertLess(output['relative_speed_mps'], .1)
        self.assertLess(trace['shape_speed_mps'].max(), .01)
        self.assertEqual(report['delivery']['geometric_hit_ids'], [])
        json.dumps(report, allow_nan=False)

    def test_all_frames_share_explicit_attempt_window_and_recorded_clock(self):
        arrays = dict(self.controls['small_travelling_wave'])
        arrays['time_s'] = 7.3+np.cumsum(np.linspace(.009, .012, len(arrays['time_s'])))
        report, trace = self.read(arrays, start=30, end=140)
        np.testing.assert_array_equal(trace['time_s'], arrays['time_s'])
        self.assertEqual(len(trace['world_speed_mps']), len(arrays['time_s']))
        for timing in report['distal_response']['peak_timing'].values():
            self.assertEqual(timing['start_index'], 30)
            self.assertEqual(timing['end_index'], 140)
            self.assertEqual(timing['start_time_s'], arrays['time_s'][30])
            self.assertEqual(timing['end_time_s'], arrays['time_s'][140])
            for node in timing['nodes']:
                self.assertTrue(all(30 <= i <= 140 for i in node['peak_indices']))
                np.testing.assert_array_equal(node['peak_times_s'],
                                              arrays['time_s'][node['peak_indices']])

    def test_release_output_uses_only_release_and_same_sample_vectors(self):
        arrays = self.controls['small_travelling_wave']
        report, trace = self.read(arrays, release_interval=release(10, 40))
        output = report['distal_response']['output_sample']
        index = 10+int(np.argmax(trace['relative_speed_mps'][10:41, -1]))
        self.assertEqual(output['index'], index)
        self.assertEqual(output['time_s'], arrays['time_s'][index])
        np.testing.assert_array_equal(output['world_velocity_mps'], arrays['cable_velocities'][index, -1])
        np.testing.assert_allclose(output['relative_velocity_mps'],
            arrays['cable_velocities'][index, -1]-arrays['cable_velocities'][index, 0], atol=1e-14)
        self.assertAlmostEqual(np.linalg.norm(output['shape_velocity_mps']), output['shape_speed_mps'])
        self.assertGreater(trace['relative_speed_mps'][:, -1].max(), output['relative_speed_mps'])

    def test_shape_tracks_retain_motion_after_display_peak_and_global_indices(self):
        arrays = self.controls['small_travelling_wave']
        report, trace = self.read(arrays, start=20, end=400,
                                 release_interval=release(30, 45))
        response = report['distal_response']
        tracks = response['curvature_tracks']
        self.assertTrue(tracks)
        self.assertGreater(max(max(track['indices']) for track in tracks),
                           response['output_sample']['index'])
        for track in tracks:
            self.assertTrue(all(20 <= i <= 400 for i in track['indices']))
            expected = [trace['curvature_per_m'][i, node-1]
                        for i, node in zip(track['indices'], track['node_indices'])]
            np.testing.assert_array_equal(track['amplitude_per_m'], expected)

    def test_early_event_cannot_borrow_later_rigid_crossing(self):
        arrays = self.controls['earlier_wave_later_rigid_crossing']
        report, _ = self.read(arrays, start=0, end=180, release_interval=release(80, 120))
        self.assertTrue(report['delivery']['crossings'])
        self.assertEqual(report['delivery']['attempt_crossing_ids'], [])
        self.assertEqual(report['delivery']['release_crossing_ids'], [])
        self.assertEqual(report['delivery']['geometric_hit_ids'], [])

    def test_attempt_passage_does_not_invent_release_without_annotation(self):
        report, _ = self.read(self.crossing([-.2, -.1, .1, .2]))
        self.assertEqual(report['delivery']['attempt_crossing_ids'], [0])
        self.assertEqual(report['delivery']['release_crossing_ids'], [])
        self.assertEqual(report['delivery']['geometric_hit_ids'], [])

    def test_plane_touch_and_terminal_contact_are_not_passages(self):
        for x in ([-.2, -.1, 0., -.1], [-.2, -.1, 0., 0.]):
            with self.subTest(x=x):
                report, _ = self.read(self.crossing(x), release_interval=release(0, 3))
                self.assertEqual(report['delivery']['crossings'], [])
                self.assertEqual(report['delivery']['geometric_hit_ids'], [])

    def test_confirmation_must_lie_inside_release_bounds(self):
        arrays = self.crossing([-.2, -.1, 0., 0., 0., .1])
        report, _ = self.read(arrays, release_interval=release(0, 4))
        self.assertEqual(report['delivery']['attempt_crossing_ids'], [0])
        self.assertEqual(report['delivery']['release_crossing_ids'], [])
        completed, _ = self.read(arrays, release_interval=release(0, 5))
        self.assertEqual(completed['delivery']['release_crossing_ids'], [0])
        self.assertEqual(completed['delivery']['geometric_hit_ids'], [0])

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

    def test_inputs_are_not_mutated(self):
        arrays = deepcopy(self.controls['small_travelling_wave'])
        before = deepcopy(arrays)
        annotation = release(30, 70)
        saved_annotation = deepcopy(annotation)
        self.read(arrays, release_interval=annotation)
        for key in arrays:
            np.testing.assert_array_equal(arrays[key], before[key])
        self.assertEqual(annotation, saved_annotation)

    def test_invalid_bounds_provenance_and_nonfinite_data_are_rejected(self):
        arrays = self.controls['small_travelling_wave']
        for start, end in [(-1, 20), (30, 20), (0, 451), (False, 20), (0., 20), (0, True)]:
            with self.subTest(start=start, end=end), self.assertRaises(ValueError):
                self.read(arrays, start=start, end=end)
        for annotation in (release(-1, 20), release(10, 451), release(30, 20), release(True, 20),
                           dict(start_index=10, end_index=20),
                           dict(release(10, 20), provenance=dict(PROVENANCE, notes=' '))):
            with self.subTest(annotation=annotation), self.assertRaises(ValueError):
                self.read(arrays, release_interval=annotation)
        with self.assertRaises(ValueError):
            evaluate_attempt(arrays, self.rest, SURFACE, start_index=0, end_index=20,
                             selection_provenance=dict(PROVENANCE, method=''))
        invalid = deepcopy(arrays)
        invalid['cable_velocities'][2, 0, 0] = np.nan
        with self.assertRaises(ValueError):
            self.read(invalid)


if __name__ == '__main__':
    unittest.main()
