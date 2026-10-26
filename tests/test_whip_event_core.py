"""Explicit-release measurement contracts on analytic fixtures, never whip labels.

Review annotations below exercise schema/serialization only. In particular, the
late rigid crossing is deliberately annotated to ensure the evaluator does not
invent a physical or task-success conclusion from user-supplied review text.
No physics rollout, policy evaluation, or training is used.
"""
from copy import deepcopy
import csv
import json
from pathlib import Path
import unittest

import numpy as np

from helpers import project_temporary
from evaluation.whip_event_review import evaluate, INPUT_SCHEMA, run_manifest, snapshot_geometry
from evaluation.whip_measurement_examples import kinematic_controls
from training.configuration import DEFAULT, load
from viewer.replay_data import PresentationData, sha256


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
        cls.snapshot = load(DEFAULT)[1]
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

    def test_explicit_release_bounds_are_inclusive_and_ties_choose_first(self):
        # Entire late segment is exactly rigid, hence Q is exactly zero.
        result, _ = self.read([self.late(release_interval=release(330, 350, 293))])
        record = result['events'][0]['release_measurement']
        self.assertEqual(record['q_index'], 330)
        self.assertEqual(record['q_relative_tip_speed_mps'], 0.)
        result, trace = self.read([self.early(release_interval=release(45, 45, 0))])
        self.assertEqual(result['events'][0]['release_measurement']['q_index'], 45)
        self.assertEqual(result['events'][0]['release_measurement']['q_relative_tip_speed_mps'],
                         float(trace['relative_speed_mps'][45, -1]))

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

    def test_gain_preserves_negative_difference_and_raw_baseline(self):
        result, trace = self.read([self.early(release_interval=release(160, 175, 99))])
        record = result['events'][0]['release_measurement']
        self.assertEqual(record['baseline_index'], 99)
        self.assertEqual(record['baseline_time_s'], float(trace['time_s'][99]))
        self.assertEqual(record['baseline_relative_tip_speed_mps'],
                         float(trace['relative_speed_mps'][99, -1]))
        self.assertLess(record['signed_gain_mps'], 0.)
        self.assertAlmostEqual(record['signed_gain_mps'], record['q_relative_tip_speed_mps']-
                               record['baseline_relative_tip_speed_mps'], places=14)

    def test_release_and_baseline_times_use_the_recorded_clock(self):
        arrays = dict(self.wave['arrays'], time_s=7.3+np.cumsum(np.linspace(.009, .012, 451)))
        result, _ = self.read([self.early(release_interval=release(20, 45, 0))], arrays)
        record = result['events'][0]['release_measurement']
        for field, index in (('start_s', 20), ('end_s', 45), ('baseline_time_s', 0),
                             ('q_time_s', record['q_index'])):
            self.assertEqual(record[field], float(arrays['time_s'][index]))

    def test_automatic_candidates_remain_unresolved_without_release_q(self):
        result, _ = evaluate(self.wave['arrays'], self.snapshot)
        self.assertTrue(result['events'])
        for event in result['events']:
            self.assertNotIn('release_measurement', event)
            self.assertEqual(event['mechanism_review'], dict(status='unresolved', provenance=None))
            self.assertEqual(event['review_status'], 'unreviewed')
            self.assertEqual(event['delivery']['link_status'], 'unresolved')
            self.assertEqual(event['delivery']['reviewed_linked_crossing_ids'], [])
            self.assertNotIn('is_whip', event)
            self.assertNotIn('reward', event)

    def test_release_annotations_reject_bad_types_bounds_keys_and_provenance(self):
        valid = release(20, 45, 0)
        changes = [dict(start_index=True), dict(start_index=20.), dict(end_index=451),
                   dict(end_index=19), dict(start_index=-1), dict(baseline_index=20),
                   dict(baseline_index=-1), dict(extra='not permitted'),
                   dict(provenance=None), dict(provenance=dict(PROVENANCE, reviewer=' ')),
                   dict(provenance=dict(PROVENANCE, method=7)),
                   dict(provenance=dict(PROVENANCE, notes='')),
                   dict(provenance=dict(PROVENANCE, extra='not permitted'))]
        for change in changes:
            with self.subTest(change=change), self.assertRaises(ValueError):
                self.read([self.early(release_interval=dict(valid, **change))])
        for missing in ('start_index', 'end_index', 'baseline_index', 'provenance'):
            annotation = dict(valid); del annotation[missing]
            with self.subTest(missing=missing), self.assertRaises(ValueError):
                self.read([self.early(release_interval=annotation)])

    def test_mechanism_review_requires_release_and_explicit_valid_provenance(self):
        with self.assertRaises(ValueError):
            self.read([self.early(mechanism_review=mechanism('supported'))])
        for annotation in (mechanism('unknown'), dict(mechanism('supported'), extra=True),
                           dict(status='supported'), dict(status='supported', provenance=None)):
            with self.subTest(annotation=annotation), self.assertRaises(ValueError):
                self.read([self.early(release_interval=release(20, 45, 0), mechanism_review=annotation)])
        for status in ('supported', 'unsupported', 'unresolved'):
            annotation = mechanism(status)
            result, _ = self.read([self.early(release_interval=release(20, 45, 0),
                                             mechanism_review=annotation)])
            self.assertEqual(result['events'][0]['mechanism_review'], annotation)
            self.assertEqual(result['events'][0]['review_status'], 'unreviewed')
            self.assertIn('release_measurement', result['events'][0])

    def test_late_unrelated_crossing_cannot_be_linked_to_early_release(self):
        with self.assertRaises(ValueError):
            self.read([self.early(release_interval=release(80, 120, 0),
                                 mechanism_review=mechanism('supported'), delivery_review=delivery([0]))])
        # Being inside the broader candidate is insufficient: release ends
        # before the actual crossing, so the claimed link must still fail.
        with self.assertRaises(ValueError):
            self.read([self.late(release_interval=release(300, 320, 293),
                                mechanism_review=mechanism('supported'), delivery_review=delivery([0]))])

    def test_delivery_annotations_require_supported_review_and_unique_real_ids(self):
        for status in (None, 'unresolved', 'unsupported'):
            window = self.late(release_interval=release(330, 350, 293), delivery_review=delivery([0]))
            if status is not None:
                window['mechanism_review'] = mechanism(status)
            with self.subTest(status=status), self.assertRaises(ValueError):
                self.read([window])
        for ids in ([], [0, 0], [False], [0.], [-1], [99], '0'):
            with self.subTest(ids=ids), self.assertRaises(ValueError):
                self.read([self.late(release_interval=release(330, 350, 293),
                                    mechanism_review=mechanism('supported'), delivery_review=delivery(ids))])
        for annotation in (dict(linked_crossing_ids=[0]), dict(delivery([0]), extra=True),
                           dict(delivery([0]), provenance=None),
                           dict(delivery([0]), provenance=dict(PROVENANCE, notes=' '))):
            with self.subTest(annotation=annotation), self.assertRaises(ValueError):
                self.read([self.late(release_interval=release(330, 350, 293),
                                    mechanism_review=mechanism('supported'), delivery_review=annotation)])

    def test_crossing_confirmation_must_be_inside_release_not_only_candidate(self):
        rest, surface = snapshot_geometry(self.snapshot)
        positions = np.broadcast_to(np.asarray(surface['center_world_m']), (6, len(rest)+1, 3)).copy()
        material = np.r_[0., np.cumsum(rest)]
        positions[:, :, 2] += material[-1]-material
        positions[:, :, 0] += np.asarray([-.2, -.1, 0., 0., 0., .1])[:, None]
        arrays = dict(time_s=np.arange(6)*.1, cable_positions=positions,
                      cable_velocities=np.zeros_like(positions))
        window = dict(start_index=0, peak_index=2, end_index=5,
                      release_interval=release(1, 4, 0), mechanism_review=mechanism('supported'))
        result, _ = self.read([window], arrays)
        self.assertEqual(result['events'][0]['delivery']['crossing_ids'], [0])
        self.assertEqual(result['events'][0]['delivery']['release_crossing_ids'], [])
        with self.assertRaises(ValueError):
            self.read([dict(window, delivery_review=delivery([0]))], arrays)

    def test_review_link_records_raw_backward_direction_without_inventing_success(self):
        _, surface = snapshot_geometry(self.snapshot)
        surface = dict(surface, normal_world=[-1., 0., 0.])
        window = self.late(release_interval=release(330, 350, 293),
                           mechanism_review=mechanism('supported'), delivery_review=delivery([0]))
        result, _ = self.read([window], surface=surface)
        event = result['events'][0]
        link = event['delivery']
        self.assertEqual(link['link_status'], 'review_linked')
        self.assertEqual(link['release_crossing_ids'], [0])
        self.assertEqual(link['reviewed_linked_crossing_ids'], [0])
        self.assertEqual(link['review_provenance'], PROVENANCE)
        self.assertEqual(len(link['linked_quality']), 1)
        crossing = link['linked_quality'][0]
        self.assertLess(crossing['forward_world_mps'], 0.)
        self.assertEqual(crossing['speed_relative_mps'], 0.)
        self.assertEqual(crossing['entry_angle_deg'], 180.)
        self.assertEqual(event['review_status'], 'unreviewed')
        for record in (event, link, crossing, result):
            for key in ('success', 'task_success', 'whip_success', 'is_whip', 'reward'):
                self.assertNotIn(key, record)

    def test_legacy_v1_values_and_fields_are_preserved(self):
        result, _ = self.read(self.wave['events'])
        self.assertAlmostEqual(result['summary']['maximum_curvature_per_m'], 2.981738866954887, places=12)
        self.assertAlmostEqual(result['summary']['peak_relative_tip_speed_mps'], 1.2748268273946093, places=12)
        earlier, later = result['events']
        self.assertEqual([earlier['start_s'], earlier['peak_s'], earlier['end_s']], [0., .99, 1.8])
        self.assertAlmostEqual(earlier['response']['relative_tip_increase_mps'], .9497887445619259, places=12)
        self.assertEqual(earlier['delivery']['crossing_ids'], [])
        self.assertEqual(later['delivery']['crossing_ids'], [0])
        self.assertEqual(later['response']['speed_relative_mps'], 0.)
        self.assertEqual(result['default_event_index'], 0)
        self.assertAlmostEqual(result['crossings'][0]['time_s'], 3.429051093169276, places=12)

    def test_manifest_roundtrip_exports_annotations_and_preserves_old_viewer_loading(self):
        temporary = project_temporary()
        self.addCleanup(temporary.cleanup)
        folder = Path(temporary.name)
        source = folder/'source_snapshot.json'
        source.write_text(json.dumps(self.snapshot))
        (folder/'scene.toml').write_text(self.snapshot['scene_text'])
        replay = folder/'synthetic.npz'
        np.savez_compressed(replay, **self.wave['arrays'])
        events = [self.early(release_interval=release(80, 120, 0),
                             mechanism_review=mechanism('unresolved')),
                  self.late(release_interval=release(330, 350, 293),
                            mechanism_review=mechanism('supported'), delivery_review=delivery([0]))]
        manifest = dict(schema=INPUT_SCHEMA, cases=[dict(name='core_schema_fixture',
            path=replay.name, sha256=sha256(replay), source_snapshot=dict(path=source.name, sha256=sha256(source)),
            evidence_kind='synthetic_kinematic_control', events=events,
            notes='Synthetic contract fixture. Review text is not a physical-whip label.')])
        manifest_path = folder/'manifest.json'; manifest_path.write_text(json.dumps(manifest))
        output = folder/'review'
        report = run_manifest(manifest_path, output)
        self.assertEqual(json.loads((output/'report.json').read_text()), report)
        emitted = report['cases'][0]['events']
        self.assertEqual(emitted[1]['mechanism_review'], mechanism('supported'))
        self.assertEqual(emitted[1]['delivery']['reviewed_linked_crossing_ids'], [0])
        with (output/'events.csv').open(newline='') as stream:
            rows = list(csv.DictReader(stream))
        self.assertEqual(len(rows), 2)
        self.assertAlmostEqual(float(rows[0]['q_relative_tip_speed_mps']),
                               emitted[0]['release_measurement']['q_relative_tip_speed_mps'])
        self.assertAlmostEqual(float(rows[0]['signed_gain_mps']),
                               emitted[0]['release_measurement']['signed_gain_mps'])
        markdown = (output/'review.md').read_text()
        self.assertIn('synthetic_kinematic_control', markdown)
        self.assertIn('supported', markdown)
        self.assertIn('unresolved', markdown)
        data = PresentationData.load_measurement(output/'report.json', 0, 1)
        self.assertEqual(data.measurement_event['event']['release_measurement'], emitted[1]['release_measurement'])
        self.assertEqual(data.event, {})
        self.assertIsNone(data.strike)
        # A report written before the additive fields existed still loads.
        old = deepcopy(report)
        for event in old['cases'][0]['events']:
            event.pop('release_measurement', None)
            event.pop('mechanism_review', None)
            for key in ('release_crossing_ids', 'reviewed_linked_crossing_ids', 'review_provenance',
                        'link_status', 'linked_quality'):
                event['delivery'].pop(key, None)
        old_path = folder/'old-v1.json'; old_path.write_text(json.dumps(old))
        old_data = PresentationData.load_measurement(old_path, 0, 1)
        self.assertNotIn('release_measurement', old_data.measurement_event['event'])
        self.assertEqual(sha256(replay), manifest['cases'][0]['sha256'])
        self.assertEqual(sha256(source), manifest['cases'][0]['source_snapshot']['sha256'])


if __name__ == '__main__':
    unittest.main()
