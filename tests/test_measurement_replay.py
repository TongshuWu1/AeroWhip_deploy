"""Pinned offline review presentation; synthetic states only, no physics or scoring."""
from copy import deepcopy
import json
from pathlib import Path
import unittest

import numpy as np

from helpers import project_temporary
from training.configuration import DEFAULT, load
from viewer.replay_data import PresentationData, sha256


class MeasurementReplayTests(unittest.TestCase):
    def setUp(self):
        self.temp = project_temporary()
        self.addCleanup(self.temp.cleanup)
        self.directory = Path(self.temp.name)
        self.snapshot = load(DEFAULT)[1]
        self.snapshot_path = self.directory/'configuration.json'
        self.snapshot_path.write_text(json.dumps(self.snapshot))
        (self.directory/'scene.toml').write_text(self.snapshot['scene_text'])
        self.replay_path = self.directory/'saved.npz'
        rest = np.asarray(self.snapshot['model']['cable']['marker_interval_lengths_m'])
        positions = np.zeros((5, len(rest)+1, 3))
        positions[:, :, 2] = 2.-np.r_[0., rest.cumsum()]
        np.savez(self.replay_path, time_s=np.arange(5)*.01,
                 drone_positions=np.tile([0., 0., 2.], (5, 1)), drone_velocities=np.zeros((5, 3)),
                 drone_rotations=np.tile(np.eye(3), (5, 1, 1)), drone_angular_velocities=np.zeros((5, 3)),
                 cable_positions=positions, cable_velocities=np.zeros_like(positions),
                 cable_marker_indices=np.arange(len(rest)+1), state_valid=np.ones(5, dtype=bool))
        self.recorded = dict(phase='evaluation', round=7, replay=self.replay_path.name,
                             valid=True, components={'strict_strike_success': 0.})
        (self.directory/'events.jsonl').write_text(json.dumps(self.recorded)+'\n')
        self.candidate = dict(id=0, start_index=1, peak_index=2, end_index=3,
                              start_s=.01, peak_s=.02, end_s=.03,
                              review_status='unreviewed', notes=['Synthetic test candidate'])
        self.report = dict(schema='offline_whip_event_review_v1', source_hashes={}, cases=[dict(
            name='fixture', evidence_kind='synthetic_kinematic_control',
            source_replay=str(self.replay_path), replay_sha256=sha256(self.replay_path),
            source_snapshot=str(self.snapshot_path), source_snapshot_sha256=sha256(self.snapshot_path),
            events=[self.candidate], default_event_index=0)])
        self.report_path = self.directory/'report.json'

    def read(self, *, case=0, event=None):
        self.report_path.write_text(json.dumps(self.report))
        return PresentationData.load_measurement(self.report_path, case, event)

    def test_candidate_does_not_replace_recorded_event_or_outcome(self):
        before = {path: sha256(path) for path in (self.replay_path, self.snapshot_path,
                                                self.directory/'events.jsonl')}
        data = self.read()
        self.assertEqual(data.event, self.recorded)
        self.assertIsNone(data.strike)
        self.assertEqual(data.measurement_event['event'], self.candidate)
        self.assertEqual(data.evidence()['recorded_event'], self.recorded)
        self.assertTrue(all(sha256(path) == value for path, value in before.items()))

    def test_no_events_still_loads_full_replay(self):
        self.report['cases'][0].update(events=[], default_event_index=None)
        data = self.read()
        self.assertIsNone(data.measurement_event['event'])
        self.assertEqual(len(data.replay.time), 5)
        with self.assertRaisesRegex(ValueError, 'event index'):
            self.read(event=0)

    def test_detached_replay_uses_pinned_snapshot_without_inventing_outcomes(self):
        detached = self.directory/'detached'
        detached.mkdir()
        destination = detached/'copy.npz'
        destination.write_bytes(self.replay_path.read_bytes())
        self.report['cases'][0]['source_replay'] = str(destination)
        data = self.read()
        self.assertEqual(data.directory, detached)
        self.assertEqual(data.event, {})
        self.assertIsNone(data.strike)

    def test_hash_mismatch_rejected_for_both_sources(self):
        for field in ('replay_sha256', 'source_snapshot_sha256'):
            with self.subTest(field=field):
                original = self.report['cases'][0][field]
                self.report['cases'][0][field] = '0'*64
                with self.assertRaisesRegex(ValueError, 'SHA256 mismatch'):
                    self.read()
                self.report['cases'][0][field] = original

    def test_scene_must_match_pinned_snapshot(self):
        with (self.directory/'scene.toml').open('a') as stream:
            stream.write('\n# Changed after measurement\n')
        with self.assertRaisesRegex(ValueError, 'scene differs'):
            self.read()

    def test_invalid_selection_and_event_timing_rejected(self):
        for case, event in ((-1, None), (1, None), (0, -1), (0, 1)):
            with self.subTest(case=case, event=event), self.assertRaises(ValueError):
                self.read(case=case, event=event)
        for updates in (dict(start_index=3), dict(peak_index=20), dict(peak_s=.025),
                        dict(review_status='validated'), dict(notes='not a list')):
            with self.subTest(updates=updates), self.assertRaises(ValueError):
                self.report['cases'][0]['events'] = [dict(self.candidate, **updates)]
                self.read()

    def test_target_geometry_mismatch_rejected(self):
        surface = deepcopy(self.read().surface)
        surface['center_world_m'][0] += 1.
        self.report['cases'][0]['target_surface'] = surface
        with self.assertRaisesRegex(ValueError, 'target geometry differs'):
            self.read()


if __name__ == '__main__':
    unittest.main()
