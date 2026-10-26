"""Sampled timing/ambiguity checks; no physical-whip classification claims."""
import unittest

import numpy as np

from aerowhip.evaluation.whip_peak_timing import describe_speed_peaks


class WhipPeakTimingTests(unittest.TestCase):
    def describe(self, speed, time=None, start=0, end=None):
        speed = np.asarray(speed)
        if time is None:
            time = np.arange(len(speed), dtype=float)
        return describe_speed_peaks(time, speed, start_index=start,
                                    end_index=len(speed)-1 if end is None else end)

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

    def test_boundary_censoring_and_single_sample_are_explicit(self):
        result = self.describe([[3., 0.], [2., 1.], [0., 4.]])
        self.assertTrue(all(n['boundary_peak'] for n in result['nodes']))
        single = self.describe([[2., 3.]])
        self.assertTrue(all(n['flat'] and n['boundary_peak'] for n in single['nodes']))
        self.assertEqual(single['adjacent'][0]['order'], 'not_identifiable')

if __name__ == '__main__':
    unittest.main()
