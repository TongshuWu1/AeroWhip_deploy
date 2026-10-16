"""Synthetic association/transport checks, not physical cable validation."""
import struct
from types import SimpleNamespace
import unittest

import numpy as np

from deployment.cable_tracking import CableTracker, TrackingSettings
from deployment.real_flight import RealFlightFeed
from deployment.ros_real_bridge import cloud_xyz
from simulation.configuration import load_configuration


class CableTrackingTests(unittest.TestCase):
    def setUp(self):
        self.arc = np.array([.063, .15, .25, .35, .45, .55, .65, .7525, .8525, .9525])
        self.settings = TrackingSettings(tuple(self.arc))
        self.tracker = CableTracker(self.settings)
        self.root = np.array([1., 2., 1.2])
        self.points = self.root+[0., 0., -1.]*self.arc[:, None]

    def observe(self, t, points, root=None):
        return self.tracker.observe(t, self.root if root is None else root, points)

    def test_order_changes_duplicates_and_clutter_do_not_define_identity(self):
        rng = np.random.default_rng(4)
        for step in range(80):
            shift = np.array([.003*step, 0., 0.])
            expected = self.points+shift
            # Include distant clutter, rigid-body markers above the attachment,
            # and a duplicated cable point as produced by overlapping lists.
            cloud = np.vstack((expected, expected[4], self.root+shift+[0., 0., .1], [5., 5., 5.]))
            cloud = cloud[rng.permutation(len(cloud))]
            result = self.observe(step*.01, cloud, self.root+shift)
            self.assertEqual(result['observed_count'], len(self.arc))
            np.testing.assert_allclose(result['positions_world_m'], expected)
            for position, index in zip(expected, result['point_indices']):
                np.testing.assert_allclose(cloud[index], position)

    def test_occlusion_is_missing_and_reacquisition_uses_geometry_and_history(self):
        self.observe(0., self.points)
        result = self.observe(.01, np.delete(self.points, 4, axis=0))
        self.assertEqual(result['observed_count'], 9)
        self.assertIsNone(result['positions_world_m'][4])
        self.assertEqual(result['point_indices'][4], -1)
        result = self.observe(.02, self.points[::-1])
        self.assertEqual(result['observed_count'], 10)
        np.testing.assert_allclose(result['positions_world_m'], self.points)

    def test_indistinguishable_points_remain_ambiguous(self):
        self.observe(0., self.points)
        cloud = self.points.copy()
        cloud[4, 0] -= .005
        cloud = np.vstack((cloud, self.points[4]+[.005, 0., 0.]))
        result = self.observe(.01, cloud)
        self.assertIsNone(result['positions_world_m'][4])
        self.assertEqual(result['confidence'][4], 0.)

    def test_folding_shortens_chords_without_changing_material_spacing(self):
        self.observe(0., self.points)
        lengths = np.diff(np.r_[0., self.arc])
        for step in range(1, 61):
            theta = .025*step*np.linspace(0., 1., len(self.arc))
            edges = lengths[:, None]*np.column_stack((np.sin(theta), np.zeros(len(theta)), -np.cos(theta)))
            cloud = self.root+np.cumsum(edges, axis=0)
            result = self.observe(step*.01, cloud[::-1])
            self.assertEqual(result['observed_count'], 10, result)
            np.testing.assert_allclose(result['positions_world_m'], cloud)

    def test_nonhanging_start_long_gap_and_capacity_are_explicit(self):
        result = self.observe(0., self.points+[.2, 0., 0.])
        self.assertFalse(result['initialized'])
        self.assertEqual(result['observed_count'], 0)
        self.observe(.01, self.points)
        result = self.observe(.5, self.points+[.2, 0., 0.])
        self.assertFalse(result['initialized'])
        result = self.observe(.51, np.zeros((2049, 3)))
        self.assertIn('capacity', result['status'])
        with self.assertRaisesRegex(ValueError, 'increase'):
            self.observe(.51, self.points)
        result = self.observe(.52, self.points)
        self.assertTrue(result['initialized'])
        self.assertEqual(result['observed_count'], 10)

    def test_crowding_and_invalid_points_cannot_force_assignments(self):
        cloud = np.vstack((self.points, np.full((1, 3), np.nan)))
        self.assertEqual(self.observe(0., cloud)['observed_count'], 10)
        cloud = np.vstack((self.points, self.points[4]+np.arange(1, 16)[:, None]*[.002, 0., 0.]))
        result = self.observe(.01, cloud)
        self.assertGreater(result['observed_count'], 0)
        self.assertIsNone(result['positions_world_m'][4])
        self.assertIn(5, result['crowded_markers'])
        self.assertIn('crowded', result['status'])

    def test_bunched_receipt_timestamps_do_not_destroy_velocity_prediction(self):
        self.observe(0., self.points)
        self.observe(.01, self.points)
        shifted = self.points+[.002, 0., 0.]
        result = self.observe(.0102, shifted)
        self.assertEqual(result['observed_count'], 10)
        result = self.observe(.0362, shifted+[.002, 0., 0.])
        self.assertEqual(result['observed_count'], 10, result)

    def test_impossible_root_distances_do_not_consume_candidate_capacity(self):
        self.observe(0., self.points)
        clutter = self.root+[0., 0., .10]+np.arange(16)[:, None]*[.002, 0., 0.]
        result = self.observe(.02, np.vstack((self.points, clutter)))
        self.assertNotIn(1, result['crowded_markers'])
        np.testing.assert_allclose(result['positions_world_m'][0], self.points[0])

    def test_long_partial_occlusion_preserves_visible_tracks_and_recovers_while_bent(self):
        self.observe(0., self.points)
        lengths = np.diff(np.r_[0., self.arc])
        for step in range(1, 121):
            theta = min(step, 50)*.012*np.linspace(0., 1., len(self.arc))
            edges = lengths[:, None]*np.column_stack((np.sin(theta), np.zeros(10), -np.cos(theta)))
            full = self.root+np.cumsum(edges, axis=0)
            result = self.observe(step*.01, np.delete(full, 4, axis=0)[::-1])
            self.assertTrue(result['initialized'])
            self.assertEqual(result['observed_count'], 9, result)
            self.assertIsNone(result['positions_world_m'][4])
            np.testing.assert_allclose(np.asarray([p for i, p in enumerate(result['positions_world_m']) if i != 4]), np.delete(full, 4, axis=0))
        result = self.observe(1.21, full[::-1])
        self.assertEqual(result['observed_count'], 10, result)
        self.assertEqual(result['reacquired_markers'], [5])
        np.testing.assert_allclose(result['positions_world_m'], full)

    def test_long_occlusion_with_two_possible_returning_points_stays_missing(self):
        self.observe(0., self.points)
        for step in range(1, 61):
            self.observe(step*.01, np.delete(self.points, 4, axis=0))
        ambiguous = np.vstack((np.delete(self.points, 4, axis=0),
                               self.points[4]+[.005, 0., 0.], self.points[4]-[.005, 0., 0.]))
        result = self.observe(.61, ambiguous)
        self.assertEqual(result['observed_count'], 9)
        self.assertIsNone(result['positions_world_m'][4])
        self.assertEqual(result['reacquired_markers'], [])
        result = self.observe(.62, self.points)
        self.assertEqual(result['observed_count'], 10)

    def test_pointcloud_layout_endianness_padding_and_nonfinite(self):
        for big in (False, True):
            fmt = '>fff' if big else '<fff'
            fields = [SimpleNamespace(name=name, offset=i*4, datatype=7, count=1)
                      for i, name in enumerate(('x', 'y', 'z'))]
            message = SimpleNamespace(width=1, height=2, fields=fields, point_step=16,
                row_step=20, is_bigendian=big,
                data=struct.pack(fmt, 1., 2., 3.)+bytes(8)+struct.pack(fmt, 4., float('nan'), 6.)+bytes(8))
            self.assertEqual(cloud_xyz(message), [[1., 2., 3.], [4., None, 6.]])
            message.row_step = 8
            with self.assertRaisesRegex(ValueError, 'layout'):
                cloud_xyz(message)

    def test_feed_exact_matching_timestamps_and_generation(self):
        config = load_configuration('configs/scenes/default.toml')
        feed = RealFlightFeed(config, [dict(name='cf_2', provenance='Measured')], 'mocap', 'test',
                              tracking_settings=self.settings)
        self.addCleanup(feed.close)
        pose = dict(name='cf_2', time_s=1., stamp_ns=1_000_000_000,
            position_world_m=(self.root-config.platform.attachment_offset_body_m).tolist(),
            quaternion_xyzw=[0., 0., 0., 1.])
        cloud = dict(kind='markers', session='test', frame_id='mocap', generation=0,
            time_s=1., stamp_ns=pose['stamp_ns'], robots=[pose], points_world_m=self.points.tolist())
        feed.receive(cloud)
        state = dict(kind='state', session='test', frame_id='mocap', generation=0,
            sequence=1, time_s=1.01, capture_type='optitrack', robots=[pose])
        feed.receive(state)
        self.assertEqual(feed.latest.frame.robots[0].cable_tracking.observed.sum(), 10)
        self.assertIsNone(feed.latest.frame.robots[0].cable)
        self.assertIn('10/10', feed.latest.frame.source_label)
        with self.assertRaisesRegex(ValueError, 'timestamp mismatch'):
            feed.receive(dict(cloud, stamp_ns=9))
        feed.receive(dict(cloud, robots=[], time_s=1.02))
        self.assertIsNone(feed.tracking_display)
        feed.receive(dict(cloud, generation=1, time_s=.1))
        self.assertTrue(feed.tracker.initialized)
        feed.receive(dict(state, generation=1, sequence=2, time_s=.11, robots=[]))
        self.assertTrue(feed.tracker.initialized)  # First marker precedes new heartbeat.


if __name__ == '__main__':
    unittest.main()
