"""Explicit-release measurement contracts on analytic fixtures, never whip labels.

Review annotations below exercise schema/serialization only. In particular, the
late rigid crossing is deliberately annotated to ensure the evaluator does not
invent a physical or task-success conclusion from user-supplied review text.
No physics rollout, policy evaluation, or training is used.
"""
from copy import deepcopy
import unittest

import numpy as np

from evaluation.whip_event_review import evaluate, snapshot_geometry
from evaluation.whip_measurement_examples import kinematic_controls
from helpers import SNAPSHOT


PROVENANCE = dict(reviewer='synthetic_test_fixture', method='schema_contract_test',
                  notes='Constructed annotation for testing only; no physical whip label.')


def release(start, end, baseline):
    return dict(start_index=start, end_index=end, baseline_index=baseline,
                provenance=deepcopy(PROVENANCE))


def mechanism(status):
    return dict(status=status, provenance=deepcopy(PROVENANCE))


def delivery(ids):
    return dict(linked_crossing_ids=ids, provenance=deepcopy(PROVENANCE))


class WhipEventCoreTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.snapshot = SNAPSHOT
        rest, _ = snapshot_geometry(cls.snapshot)
        cls.controls = {case['name']: case for case in kinematic_controls(rest)}
        cls.wave = cls.controls['earlier_wave_later_rigid_crossing']

    def early(self, **updates):
        return dict(self.wave['events'][0], **updates)

    def late(self, **updates):
        return dict(self.wave['events'][1], **updates)

    def read(self, windows, arrays=None, surface=None):
        result, trace = evaluate(self.wave['arrays'] if arrays is None else arrays,
                                 self.snapshot, event_windows=windows, target_surface=surface)
        return result, trace

    def test_q_uses_only_explicit_release_not_larger_candidate_peak(self):
        window = self.early(release_interval=release(20, 45, 0))
        result, trace = self.read([window])
        record = result['events'][0]['release_measurement']
        values = trace['relative_speed_mps'][:, -1]
        index = 20+int(np.argmax(values[20:46]))
        self.assertGreater(values[:181].max(), values[20:46].max())
        self.assertEqual(record['q_index'], index)
        self.assertEqual(record['q_relative_tip_speed_mps'], float(values[index]))
        self.assertEqual(record['q_time_s'], float(trace['time_s'][index]))
        self.assertEqual(record['selection_provenance'], PROVENANCE)
        self.assertEqual(result['events'][0]['review_status'], 'unreviewed')

    def test_global_velocity_boost_preserves_q_gain_and_relative_velocity(self):
        arrays = self.wave['arrays']
        boost = np.array([1.7, -.4, .2])
        changed = dict(arrays,
            cable_positions=arrays['cable_positions']+arrays['time_s'][:, None, None]*boost,
            cable_velocities=arrays['cable_velocities']+boost,
            drone_positions=arrays['drone_positions']+arrays['time_s'][:, None]*boost,
            drone_velocities=arrays['drone_velocities']+boost)
        windows = [self.early(release_interval=release(80, 120, 0))]
        original, _ = self.read(windows)
        transformed, _ = self.read(windows, changed)
        a, b = (report['events'][0]['release_measurement'] for report in (original, transformed))
        for field in ('q_relative_tip_speed_mps', 'baseline_relative_tip_speed_mps', 'signed_gain_mps'):
            self.assertAlmostEqual(a[field], b[field], places=12)
        np.testing.assert_allclose(a['peak_velocity']['velocity_relative_mps'],
                                   b['peak_velocity']['velocity_relative_mps'], atol=1e-12, rtol=0)
        np.testing.assert_allclose(np.asarray(b['peak_velocity']['velocity_world_mps'])-
                                   a['peak_velocity']['velocity_world_mps'], boost, atol=1e-12, rtol=0)

    def test_late_unrelated_crossing_cannot_be_linked_to_early_release(self):
        with self.assertRaises(ValueError):
            self.read([self.early(release_interval=release(80, 120, 0),
                                 mechanism_review=mechanism('supported'), delivery_review=delivery([0]))])
        # Being inside the broader candidate is insufficient: release ends
        # before the actual crossing, so the claimed link must still fail.
        with self.assertRaises(ValueError):
            self.read([self.late(release_interval=release(300, 320, 293),
                                mechanism_review=mechanism('supported'), delivery_review=delivery([0]))])

if __name__ == '__main__':
    unittest.main()
