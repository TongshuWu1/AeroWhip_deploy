"""Numerical/kinematic checks, not validation of a physical whip definition."""
import json
from pathlib import Path
import tempfile
import unittest

import numpy as np

from evaluation.whip_measurement_examples import kinematic_controls
from evaluation.whip_measurements import extract_measurements, track_curvature
from project_paths import ROOT


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

    def test_times_are_preserved_and_invalid_clocks_rejected(self):
        arrays = self.controls['small_travelling_wave']['arrays']
        irregular = dict(arrays, time_s=1.2+np.cumsum(np.linspace(.009, .012, 451)))
        measured = extract_measurements(irregular, self.rest, [2., 0., 0.])
        np.testing.assert_array_equal(measured['time_s'], irregular['time_s'])
        for invalid in (np.zeros(451), -arrays['time_s'],
                        np.r_[np.nan, arrays['time_s'][1:]]):
            with self.subTest(clock=invalid[:3]):
                with self.assertRaises(ValueError):
                    extract_measurements(dict(arrays, time_s=invalid), self.rest, [1., 0., 0.])

    def test_invalid_or_padded_measurements_are_rejected(self):
        arrays = self.controls['translated_curved_shape']['arrays']
        valid = arrays['state_valid'].copy(); valid[-1] = False
        positions = arrays['cable_positions'].copy(); positions[5, 2] = positions[5, 1]
        velocity = arrays['cable_velocities'].copy(); velocity[-1, -1, 0] = np.inf
        for replacement in (dict(state_valid=valid), dict(cable_positions=positions),
                            dict(cable_velocities=velocity)):
            with self.assertRaises(ValueError):
                extract_measurements(dict(arrays, **replacement), self.rest, [1., 0., 0.])
        with self.assertRaises(ValueError):
            extract_measurements(arrays, self.rest, [0., 0., 0.])
        with self.assertRaises(ValueError):
            extract_measurements(arrays, np.zeros_like(self.rest), [1., 0., 0.])

    def test_single_frame_has_measurements_without_fabricated_transport(self):
        arrays = self.controls['small_travelling_wave']['arrays']
        single = {key: arrays[key][70:71] for key in
                  ('time_s', 'cable_positions', 'cable_velocities', 'state_valid')}
        measured = extract_measurements(single, self.rest, [1., 0., 0.])
        self.assertEqual(measured['curvature_per_m'].shape, (1, 9))
        self.assertTrue(all(track['indices'] == [0] for track in track_curvature(measured)))

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

    def test_stationary_bends_do_not_become_one_transported_centroid(self):
        tracks = track_curvature(self.measure('separated_bends_fade_and_grow'))
        self.assertGreaterEqual(len(tracks), 2)
        for track in tracks:
            self.assertEqual(len(set(track['node_indices'])), 1)

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

    def test_reflection_is_retained_without_forcing_outward_direction(self):
        rows = np.zeros((5, 5))
        rows[np.arange(5), [1, 2, 3, 2, 1]] = 4.
        tracks = track_curvature(self.ridge(rows))
        self.assertEqual(len(tracks), 1)
        self.assertEqual(tracks[0]['node_indices'], [2, 3, 4, 3, 2])

    def test_plateau_has_no_silently_chosen_peak(self):
        self.assertEqual(track_curvature(self.ridge(np.ones((3, 5)))), [])
        tracks = track_curvature(self.ridge([[0, 3, 3, 0, 0], [0, 3, 3, 0, 0]]))
        self.assertEqual(len(tracks), 4)
        self.assertTrue(all('ambiguous_plateau' in row['flags'] for row in tracks))
        self.assertTrue(all(len(row['indices']) == 1 for row in tracks))

    def test_fixtures_preserve_lengths_and_late_motion_is_exactly_rigid(self):
        for case in self.controls.values():
            lengths = np.linalg.norm(np.diff(case['arrays']['cable_positions'], axis=1), axis=-1)
            np.testing.assert_allclose(lengths, np.tile(self.rest, (451, 1)), atol=1e-15, rtol=0)
        late = self.controls['earlier_wave_later_rigid_crossing']['arrays']
        measured = extract_measurements(late, self.rest, [1., 0., 0.])
        self.assertGreater(measured['turn_rad'][:180].max(), .1)
        self.assertLess(measured['turn_rad'][181:].max(), 1e-13)
        self.assertEqual(measured['relative_speed_mps'][181:].max(), 0.)
        self.assertGreater(measured['world_speed_mps'][300:, -1].max(), 1.)

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
        from training.configuration import DEFAULT, load
        _, cls.snapshot, _, _ = load(DEFAULT)
        rest = cls.snapshot['model']['cable']['marker_interval_lengths_m']
        cls.case = next(c for c in kinematic_controls(rest)
                        if c['name'] == 'earlier_wave_later_rigid_crossing')

    def test_automatic_basins_stop_at_quiet_plateau_edges(self):
        from evaluation.whip_event_review import candidate_windows
        speed = np.array([0., 0., 0., 1., 0., 0., 0., 2., 0., 0., 0.])
        windows = candidate_windows({'relative_speed_mps': speed[:, None]})
        self.assertEqual(windows, [dict(start_index=2, peak_index=3, end_index=4),
                                   dict(start_index=6, peak_index=7, end_index=8)])
        self.assertEqual(candidate_windows({'relative_speed_mps': np.ones((5, 1))}), [])

    def test_automatic_event_cannot_borrow_a_crossing_during_later_quiet_motion(self):
        from evaluation.whip_event_review import evaluate
        arrays = {key: value.copy() for key, value in self.case['arrays'].items()}
        # Keep the early prescribed packet, then move the now-straight cable
        # rigidly during its relative-speed zero plateau. The old midpoint
        # trough boundary included this passage in the earlier speed basin.
        time = arrays['time_s']
        phase = np.clip((time-2.1)/.4, 0., 1.)
        shift = .6*(3*phase**2-2*phase**3)
        rate = .6*(6*phase-6*phase**2)/.4
        arrays['cable_positions'][181:] = arrays['cable_positions'][181]
        arrays['cable_positions'][181:, :, 0] += shift[181:, None]
        arrays['cable_velocities'][181:] = 0.
        arrays['cable_velocities'][181:, :, 0] = rate[181:, None]
        result, _ = evaluate(arrays, self.snapshot)  # Automatic intervals only.
        self.assertTrue(result['events'])
        self.assertTrue(all(event['end_index'] <= 181 for event in result['events']))
        self.assertEqual(len(result['crossings']), 1)
        crossing = result['crossings'][0]
        self.assertGreater(crossing['time_s'], 2.1)
        self.assertLess(crossing['time_s'], 2.5)
        self.assertEqual(crossing['associated_event_ids'], [])
        self.assertTrue(all(event['delivery']['crossing_ids'] == []
                            for event in result['events']))

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

    def test_event_review_rejects_single_frame_and_nonunit_target_normal(self):
        from evaluation.whip_event_review import evaluate, snapshot_geometry
        single = {key: self.case['arrays'][key][:1] for key in
                  ('time_s', 'cable_positions', 'cable_velocities', 'state_valid')}
        with self.assertRaisesRegex(ValueError, 'at least two'):
            evaluate(single, self.snapshot)
        _, surface = snapshot_geometry(self.snapshot)
        surface = dict(surface, normal_world=[1.000001, 0., 0.])
        with self.assertRaisesRegex(ValueError, 'unit normal'):
            evaluate(self.case['arrays'], self.snapshot, target_surface=surface)

    def test_dwell_confirmation_outside_event_cannot_supply_its_crossing(self):
        from evaluation.whip_event_review import evaluate, snapshot_geometry
        rest, surface = snapshot_geometry(self.snapshot)
        surface = dict(surface, normal_world=[1., 0., 0.])
        material = np.r_[0., np.cumsum(rest)]
        positions = np.broadcast_to(np.asarray(surface['center_world_m']),
                                     (5, len(material), 3)).copy()
        positions[:, :, 2] += material[-1]-material
        positions[:, :, 0] += np.array([-.1, 0., 0., 0., .1])[:, None]
        arrays = dict(time_s=np.arange(5)*.1, cable_positions=positions,
                      cable_velocities=np.zeros_like(positions))
        # First arrival is inside this explicit review interval; opposite-side
        # evidence occurs only after it ends. The global passage is retained.
        window = dict(start_index=0, peak_index=1, end_index=3)
        result, _ = evaluate(arrays, self.snapshot, event_windows=[window],
                             target_surface=surface)
        self.assertEqual(len(result['crossings']), 1)
        crossing = result['crossings'][0]
        self.assertEqual(crossing['time_s'], arrays['time_s'][1])
        self.assertEqual(crossing['confirmed_at_s'], arrays['time_s'][4])
        self.assertEqual(crossing['associated_event_ids'], [])
        self.assertEqual(result['events'][0]['delivery']['crossing_ids'], [])
        self.assertEqual(result['events'][0]['delivery']['closest_approach']['center_distance_m'], 0.)

    def test_earlier_wave_cannot_support_later_rigid_crossing(self):
        from evaluation.whip_event_review import evaluate
        arrays, windows = self.case['arrays'], self.case['events']
        result, _ = evaluate(arrays, self.snapshot, event_windows=windows)
        earlier, later = result['events']
        self.assertGreater(earlier['deformation']['maximum_turn_rad'], .1)
        self.assertGreater(earlier['deformation']['outward_track_count'], 0)
        self.assertEqual(later['deformation']['tracks'], [])
        self.assertEqual(later['deformation']['distal_outward_track_count'], 0)
        self.assertEqual(later['response']['speed_relative_mps'], 0.)
        self.assertEqual(earlier['delivery']['crossing_ids'], [])
        self.assertEqual(len(result['crossings']), 1)
        crossing = result['crossings'][0]
        self.assertEqual(crossing['associated_event_ids'], [later['id']])
        self.assertEqual(later['delivery']['crossing_ids'], [crossing['id']])
        self.assertEqual(later['review_status'], 'unreviewed')
        self.assertNotIn('score', result)
        self.assertNotIn('is_whip', later)
        # The late report must stay identical when the distant early packet is
        # removed; it cannot borrow the earlier event's deformation evidence.
        altered = {key: value.copy() for key, value in arrays.items()}
        altered['cable_positions'][:181] = arrays['cable_positions'][181]
        altered['cable_velocities'][:181] = 0.
        changed, _ = evaluate(altered, self.snapshot, event_windows=windows)
        self.assertEqual(changed['events'][1], later)

    def test_event_times_use_recorded_clock_and_reject_bad_bounds(self):
        from evaluation.whip_event_review import evaluate
        arrays = dict(self.case['arrays'], time_s=self.case['arrays']['time_s']+7.3)
        result, _ = evaluate(arrays, self.snapshot, event_windows=self.case['events'])
        for event in result['events']:
            for field in ('start', 'peak', 'end'):
                self.assertEqual(event[field+'_s'], arrays['time_s'][event[field+'_index']])
        for indices in ((-1, 5, 10), (3, 2, 7), (0, 450, 451), (2, 2, 2), (False, 2, 4)):
            window = dict(zip(('start_index', 'peak_index', 'end_index'), indices))
            with self.subTest(indices=indices), self.assertRaises(ValueError):
                evaluate(arrays, self.snapshot, event_windows=[window])
        with self.assertRaisesRegex(ValueError, 'Duplicate'):
            evaluate(arrays, self.snapshot, event_windows=[self.case['events'][0]]*2)

    def test_manifest_pins_inputs_and_never_overwrites_review(self):
        from evaluation.whip_event_review import INPUT_SCHEMA, run_manifest
        from viewer.replay_data import sha256
        (ROOT/'tmp').mkdir(exist_ok=True)
        with tempfile.TemporaryDirectory(prefix='whip-measurement-test-', dir=ROOT/'tmp') as tmp:
            folder = Path(tmp)
            snapshot = folder/'source_snapshot.json'
            snapshot.write_text(json.dumps(self.snapshot))
            (folder/'scene.toml').write_text(self.snapshot['scene_text'])
            replay = folder/'kinematic.npz'
            np.savez_compressed(replay, **self.case['arrays'])
            manifest = dict(schema=INPUT_SCHEMA, cases=[dict(name='synthetic_event_linkage',
                path=replay.name, sha256=sha256(replay),
                source_snapshot=dict(path=snapshot.name, sha256=sha256(snapshot)),
                evidence_kind='synthetic_kinematic_control', events=self.case['events'],
                notes='Known kinematic construction; no physics or whip label.')])
            path = folder/'manifest.json'; path.write_text(json.dumps(manifest))
            report = run_manifest(path, folder/'review')
            self.assertEqual(report['cases'][0]['source_replay'], str(replay))
            self.assertTrue((folder/'review/report.json').exists())
            self.assertTrue((folder/'review/events.csv').exists())
            self.assertTrue((folder/'review/case-000.png').exists())
            before = sha256(folder/'review/report.json')
            with self.assertRaises(FileExistsError):
                run_manifest(path, folder/'review')
            self.assertEqual(sha256(folder/'review/report.json'), before)
            self.assertEqual(sha256(replay), manifest['cases'][0]['sha256'])
            self.assertEqual(sha256(snapshot), manifest['cases'][0]['source_snapshot']['sha256'])
            for entry in (manifest['cases'][0], manifest['cases'][0]['source_snapshot']):
                old = entry['sha256']; entry['sha256'] = 'wrong'
                path.write_text(json.dumps(manifest))
                with self.assertRaisesRegex(ValueError, 'hash mismatch'):
                    run_manifest(path, folder/'bad-hash')
                self.assertFalse((folder/'bad-hash').exists())
                entry['sha256'] = old


if __name__ == '__main__':
    unittest.main()
