"""Numerical/kinematic checks, not validation of a physical whip definition."""
import unittest

import numpy as np

from evaluation.whip_measurement_examples import kinematic_controls
from evaluation.whip_measurements import extract_measurements, track_curvature


class WhipMeasurementTests(unittest.TestCase):
    def setUp(self):
        self.rest = np.array([.063, .087, .1, .1, .1, .1, .1, .1025, .1, .1])
        self.controls = {case['name']: case for case in kinematic_controls(self.rest)}

    def measure(self, name):
        return extract_measurements(self.controls[name]['arrays'], self.rest, [1., 0., 0.])

    @staticmethod
    def ridge(rows):
        curvature = np.asarray(rows, float)
        nodes = np.arange(curvature.shape[1]+2)*.1
        return dict(time_s=np.arange(len(curvature))*.01, node_s_m=nodes,
                    material_s_m=nodes[1:-1], curvature_per_m=curvature)

    def test_curvature_and_residual_are_invariant_to_rigid_frame_change(self):
        arrays = self.controls['small_travelling_wave']['arrays']
        original = extract_measurements(arrays, self.rest, [1., 0., 0.])
        # Proper 3-D rotation plus position shift and constant velocity boost.
        rotation = np.array([[0., 0., 1.], [1., 0., 0.], [0., 1., 0.]])
        transformed = dict(arrays,
            cable_positions=arrays['cable_positions']@rotation.T+[3., -2., 1.],
            cable_velocities=arrays['cable_velocities']@rotation.T+[.7, -.2, .1])
        changed = extract_measurements(transformed, self.rest, rotation@np.array([1., 0., 0.]))
        for key in ('curvature_per_m', 'turn_rad', 'arc_length_m', 'chord_ratio',
                    'relative_speed_mps', 'relative_forward_mps', 'nonrigid_speed_mps'):
            np.testing.assert_allclose(changed[key], original[key], atol=2e-12, rtol=1e-10,
                                       err_msg=key)

    def test_rigid_curved_rotation_has_speed_but_zero_rigid_residual(self):
        measured = self.measure('rigid_curved_rotation')
        self.assertGreater(measured['world_speed_mps'][:, -1].min(), 1.)
        self.assertLess(measured['nonrigid_speed_mps'].max(), 2e-14)
        np.testing.assert_allclose(measured['curvature_per_m'],
            np.broadcast_to(measured['curvature_per_m'][0], (451, 9)), atol=2e-13, rtol=0)
        translation = self.measure('translated_curved_shape')
        self.assertLess(translation['nonrigid_speed_mps'].max(), 2e-14)
        self.assertEqual(translation['relative_speed_mps'].max(), 0.)

    def test_small_wave_remains_small_in_absolute_units(self):
        measured = self.measure('small_travelling_wave')
        # Each of the nine prescribed turns is at most 0.01 rad. These are
        # construction bounds, not acceptance thresholds for a physical whip.
        self.assertGreater(measured['turn_rad'].max(), 0.)
        self.assertLessEqual(measured['turn_rad'].max(), 9*.01+1e-13)
        expected_integral = (measured['curvature_per_m']*
                             measured['curvature_dual_length_m']).sum(axis=1)
        np.testing.assert_allclose(expected_integral, measured['turn_rad'], atol=1e-15)
        self.assertGreater(measured['nonrigid_speed_mps'].max(), 1e-5)

    def test_prescribed_forward_and_backward_packets_retain_direction(self):
        for name, sign in [('small_travelling_wave', 1), ('small_backward_wave', -1)]:
            tracks = track_curvature(self.measure(name))
            longest = max(tracks, key=lambda row: len(row['indices']))
            self.assertGreater(len(longest['indices']), 200)
            self.assertGreater(sign*(longest['position_m'][-1]-longest['position_m'][0]), .5)

    def test_split_and_merge_are_reported_without_joining_branches(self):
        measured = self.ridge([[0, 0, 4, 0, 0], [0, 3, 0, 3, 0], [0, 0, 4, 0, 0]])
        tracks = track_curvature(measured)
        self.assertEqual(len(tracks), 4)
        self.assertTrue(all(len(track['indices']) == 1 for track in tracks))
        flags = set(flag for track in tracks for flag in track['flags'])
        self.assertIn('ambiguous_split', flags)
        self.assertIn('ambiguous_merge', flags)

    def test_gaps_and_large_jumps_cannot_invent_continuity(self):
        for rows in ([[0, 4, 0, 0, 0], [0, 0, 0, 0, 0], [0, 4, 0, 0, 0]],
                     [[4, 0, 0, 0, 0], [0, 0, 0, 0, 4]]):
            tracks = track_curvature(self.ridge(rows))
            self.assertEqual(len(tracks), 2)
            self.assertTrue(all(len(track['indices']) == 1 for track in tracks))

    def test_curvature_uses_measured_lengths_and_interior_material_coordinates(self):
        arrays = dict(time_s=np.array([0.]),
                      cable_positions=np.array([[[0., 0., 0.], [1., 0., 0.], [1., 2., 0.]]]),
                      cable_velocities=np.zeros((1, 3, 3)))
        measured = extract_measurements(arrays, [.5, .5], [1., 0., 0.])
        np.testing.assert_allclose(measured['curvature_per_m'], [[np.pi/3]])
        np.testing.assert_array_equal(measured['material_s_m'], [.5])
        np.testing.assert_array_equal(measured['node_s_m'], [0., .5, 1.])
        np.testing.assert_allclose(measured['arc_length_m'], [3.])
        np.testing.assert_allclose(measured['chord_ratio'], [np.sqrt(5)/3])


class WhipEventReviewTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        from helpers import SNAPSHOT
        cls.snapshot = SNAPSHOT
        rest = cls.snapshot['model']['cable']['marker_interval_lengths_m']
        cls.case = next(c for c in kinematic_controls(rest)
                        if c['name'] == 'earlier_wave_later_rigid_crossing')

    def test_plane_touches_dwells_and_unfinished_passages_are_not_crossings(self):
        from evaluation.whip_event_review import measure_crossings
        surface = dict(normal_world=[1., 0., 0.], center_world_m=[0., 0., 0.],
                       width_m=1., height_m=1.)

        def measure(signed):
            positions = np.zeros((len(signed), 3, 3))
            positions[:, :, 0] = np.asarray(signed)[:, None]
            positions[:, :, 2] = [-.2, -.1, 0.]
            # Unequal sampling intervals test that confirmation uses the clock.
            times = 4.+np.cumsum(np.arange(1, len(signed)+1)*.07)
            arrays = dict(time_s=times, cable_positions=positions,
                          cable_velocities=np.zeros_like(positions))
            return measure_crossings(arrays, surface), times

        for signed in ([-1., 0., -1.], [1., 0., 1.], [-1., 0., 0., -1.],
                       [-1., 0.], [-1., 0., 0.], [0., 0., 1.], [0., 0., 0.]):
            with self.subTest(signed=signed):
                self.assertEqual(measure(signed)[0], [])
        for signed, passage in (([-1., 0., 1.], 'forward'),
                                ([-1., 0., 0., 1.], 'forward'),
                                ([1., 0., 0., -1.], 'backward')):
            with self.subTest(signed=signed):
                crossings, times = measure(signed)
                self.assertEqual(len(crossings), 1)
                crossing = crossings[0]
                self.assertEqual(crossing['passage'], passage)
                self.assertEqual(crossing['time_s'], times[1])
                self.assertEqual(crossing['confirmed_at_s'], times[-1])
                self.assertTrue(crossing['inside_target_rectangle'])

if __name__ == '__main__':
    unittest.main()
