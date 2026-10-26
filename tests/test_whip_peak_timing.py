"""Sampled timing/ambiguity checks; no physical-whip classification claims."""
import unittest

import numpy as np

from evaluation.whip_peak_timing import describe_speed_peaks


class WhipPeakTimingTests(unittest.TestCase):
    def describe(self, speed, time=None, start=0, end=None):
        speed = np.asarray(speed)
        if time is None:
            time = np.arange(len(speed), dtype=float)
        return describe_speed_peaks(time, speed, start_index=start,
                                    end_index=len(speed)-1 if end is None else end)

    def test_common_window_preserves_nonuniform_seconds_and_amplitudes(self):
        time = np.array([8., 8.1, 8.4, 8.5, 8.9, 9.2])
        speed = np.array([[100., 100.], [0., 0.], [2., 1.],
                          [1., 3.], [0., 0.], [200., 200.]])
        result = self.describe(speed, time, start=1, end=4)
        self.assertEqual(result['start_time_s'], 8.1)
        self.assertEqual(result['end_time_s'], 8.9)
        self.assertEqual(result['sample_count'], 4)
        self.assertEqual([n['peak_indices'] for n in result['nodes']], [[2], [3]])
        self.assertEqual([n['peak_times_s'] for n in result['nodes']], [[8.4], [8.5]])
        self.assertEqual([n['peak_speed_mps'] for n in result['nodes']], [2., 3.])
        pair = result['adjacent'][0]
        self.assertEqual(pair['order'], 'distal_later')
        self.assertAlmostEqual(pair['minimum_peak_lag_s'], .1)
        self.assertAlmostEqual(pair['maximum_peak_lag_s'], .1)
        self.assertEqual(pair['peak_amplitude_difference_mps'], 1.)

    def test_plateaus_and_disjoint_equal_maxima_keep_every_index(self):
        result = self.describe([[0, 0, 0], [2, 0, 0], [2, 3, 1],
                                [0, 3, 0], [0, 0, 1], [0, 0, 0]])
        self.assertEqual([n['peak_indices'] for n in result['nodes']],
                         [[1, 2], [2, 3], [2, 4]])
        self.assertTrue(all(n['multiple_maxima'] for n in result['nodes']))
        self.assertEqual([p['order'] for p in result['adjacent']],
                         ['overlapping', 'overlapping'])
        self.assertEqual(result['adjacent'][0]['minimum_peak_lag_s'], 0.)
        self.assertEqual(result['adjacent'][0]['maximum_peak_lag_s'], 2.)

    def test_later_order_requires_all_distal_peaks_after_all_proximal_peaks(self):
        result = self.describe([[0, 0], [2, 0], [2, 0], [0, 3], [0, 3], [0, 0]])
        self.assertEqual(result['adjacent'][0]['order'], 'distal_later')
        self.assertEqual(result['adjacent'][0]['minimum_peak_lag_s'], 1.)
        self.assertEqual(result['adjacent'][0]['maximum_peak_lag_s'], 3.)

    def test_reverse_order_and_unique_tie_are_distinct(self):
        result = self.describe([[0, 0, 0], [0, 4, 2], [3, 0, 0], [0, 0, 0]])
        self.assertEqual([p['order'] for p in result['adjacent']],
                         ['distal_earlier', 'tied'])
        self.assertEqual(result['adjacent'][0]['minimum_peak_lag_s'], -1.)

    def test_rest_and_flat_profiles_have_no_identifiable_peak_time(self):
        result = self.describe([[0, 2, 0], [0, 2, 1], [0, 2, 0]])
        self.assertTrue(result['nodes'][0]['zero'])
        self.assertTrue(result['nodes'][1]['flat'])
        self.assertFalse(result['nodes'][1]['zero'])
        self.assertEqual(result['nodes'][0]['peak_indices'], [0, 1, 2])
        self.assertEqual([p['order'] for p in result['adjacent']],
                         ['not_identifiable', 'not_identifiable'])
        self.assertIsNone(result['adjacent'][0]['minimum_peak_lag_s'])

    def test_machine_precision_constancy_does_not_erase_raw_maxima(self):
        epsilon = np.finfo(float).eps
        speed = np.array([[1., 0.], [1.+8*epsilon, 0.], [1., 1.], [1., 0.]])
        result = self.describe(speed)
        node = result['nodes'][0]
        self.assertFalse(node['flat'])
        self.assertTrue(node['numerically_constant'])
        self.assertEqual(node['peak_indices'], [1])
        self.assertEqual(node['peak_speed_mps'], speed[1, 0])
        self.assertEqual(node['speed_range_mps'], 8*epsilon)
        self.assertEqual(result['adjacent'][0]['order'], 'not_identifiable')

    def test_tiny_resolved_variation_retains_amplitude_and_order(self):
        speed = np.array([[0., 0.], [1e-10, 0.], [0., 2e-10], [0., 0.]])
        result = self.describe(speed)
        self.assertEqual(result['numerical_constancy_tolerance_mps'], 64*np.finfo(float).eps)
        self.assertFalse(any(n['numerically_constant'] for n in result['nodes']))
        self.assertEqual([n['peak_speed_mps'] for n in result['nodes']], [1e-10, 2e-10])
        self.assertEqual(result['adjacent'][0]['order'], 'distal_later')

    def test_boundary_censoring_and_single_sample_are_explicit(self):
        result = self.describe([[3., 0.], [2., 1.], [0., 4.]])
        self.assertTrue(all(n['boundary_peak'] for n in result['nodes']))
        single = self.describe([[2., 3.]])
        self.assertTrue(all(n['flat'] and n['boundary_peak'] for n in single['nodes']))
        self.assertEqual(single['adjacent'][0]['order'], 'not_identifiable')

    def test_inputs_are_preserved(self):
        time = np.array([2., 2.2, 2.5])
        speed = np.array([[0., 0.], [3., 1.], [0., 4.]])
        original_time, original_speed = time.copy(), speed.copy()
        self.describe(speed, time)
        np.testing.assert_array_equal(time, original_time)
        np.testing.assert_array_equal(speed, original_speed)

    def test_invalid_shapes_times_values_and_bounds_are_rejected(self):
        speed = np.array([[0., 0.], [1., 2.], [0., 0.]])
        for time, values in [([], np.empty((0, 2))), ([0., 0., 1.], speed),
                             ([2., 1., 0.], speed), ([0., np.nan, 1.], speed),
                             ([[0., 1., 2.]], speed), ([0., 1.], speed),
                             ([0., 1., 2.], np.empty((3, 0))),
                             ([0., 1., 2.], [0., 1., 2.]),
                             ([0., 1., 2.], speed-1),
                             ([0., 1., 2.], speed+np.inf),
                             ([0., 1., 2.], speed.astype(complex)),
                             ([0., 1., 2.], speed.astype(str)),
                             ([0., 1., 2.], speed.astype(bool))]:
            with self.subTest(time=time, shape=np.shape(values)):
                with self.assertRaises(ValueError):
                    describe_speed_peaks(time, values, start_index=0, end_index=2)
        for start, end in [(-1, 2), (0, 3), (2, 1), (0., 2), (False, 2), (0, True)]:
            with self.subTest(start=start, end=end):
                with self.assertRaises(ValueError):
                    describe_speed_peaks([0., 1., 2.], speed,
                                         start_index=start, end_index=end)


if __name__ == '__main__':
    unittest.main()
