"""Real-view isolation, causal capture validation and display transport lifecycle."""
import socket
import time
import unittest
from types import SimpleNamespace
from unittest.mock import Mock, patch

import numpy as np

from deployment.real_flight import RealFlightFeed, RealFlightTransport, select_real_robots
from deployment.ros_real_bridge import encode
from simulation.configuration import load_configuration


class EvidenceStub:
    def write(self, *args, **kwargs):
        pass


class RealViewerTests(unittest.TestCase):
    def test_native_worker_crash_keeps_cleanup_in_supervisor_and_attach_preserves_stack_without_stop(self):
        import run_custom_real
        for attached in (False, True):
            child = Mock(returncode=-11)
            child.poll.return_value = -11
            stop = Mock()
            stop.is_set.return_value = False
            stop.wait.return_value = True  # User stops the surviving owner.
            with patch.object(run_custom_real, 'Event', return_value=stop), \
                 patch.object(run_custom_real.signal, 'signal'), \
                 patch.object(run_custom_real.subprocess, 'Popen', return_value=child), \
                 patch.object(run_custom_real, 'capture_real_stack', return_value={'root': 'captured'}), \
                 patch.object(run_custom_real, 'close_captured_stack') as attached_cleanup, \
                 patch.object(run_custom_real, 'stop_real_stack', return_value=False) as cleanup:
                code = run_custom_real.supervise_viewer(SimpleNamespace(attach=attached, container='mock', domain=42), [])
            self.assertEqual(code, 139)
            self.assertEqual(cleanup.call_count, 0 if attached else 1)
            attached_cleanup.assert_not_called()

    def test_ctrl_c_in_attached_supervisor_stops_captured_stack_even_after_worker_crash(self):
        import run_custom_real
        for returncode in (0, -11):
            child = Mock(returncode=returncode)
            child.poll.return_value = returncode
            stop = Mock()
            stop.is_set.return_value = True
            target = {'root': 'captured-at-attach'}
            with patch.object(run_custom_real, 'Event', return_value=stop), \
                 patch.object(run_custom_real.signal, 'signal'), \
                 patch.object(run_custom_real.subprocess, 'Popen', return_value=child), \
                 patch.object(run_custom_real, 'capture_real_stack', return_value=target), \
                 patch.object(run_custom_real, 'close_captured_stack') as cleanup:
                run_custom_real.supervise_viewer(SimpleNamespace(attach=True, container='mock', domain=42), [])
            cleanup.assert_called_once_with('mock', 42, target)

    def setUp(self):
        self.configuration = load_configuration('configs/scenes/default.toml')
        self.robots = [dict(name='generic_robot', provenance='Measured')]
        self.feed = RealFlightFeed(self.configuration, self.robots, 'mocap', 'session')
        self.addCleanup(self.feed.close)

    def packet(self, sequence=1, time_s=1.):
        return dict(kind='state', session='session', frame_id='mocap', sequence=sequence,
            generation=0, time_s=time_s, capture_type='optitrack', issues={}, missing_robots=[],
            robots=[dict(name='generic_robot', time_s=.9, position_world_m=[1., 2., 3.],
                         quaternion_xyzw=[0., 0., 0., 1.])])

    def test_capture_only_without_synthetic_state_or_control(self):
        self.assertEqual(self.feed.latest.frame.robots, ())
        self.feed.receive(self.packet())
        robot = self.feed.latest.frame.robots[0]
        np.testing.assert_array_equal(robot.drone.position_world_m, [1., 2., 3.])
        self.assertIsNone(robot.drone.linear_velocity_world_mps)
        self.assertIsNone(robot.drone.angular_velocity_body_radps)
        self.assertIsNone(robot.drone.linear_acceleration_world_mps2)
        self.assertIsNone(robot.cable)
        self.assertIsNone(robot.forces)
        self.assertIsNone(self.feed.latest.state)
        self.assertIsNone(self.feed.command_source)
        self.assertFalse(self.feed.has_playback or self.feed.has_trajectory)

    def test_new_flight_changes_display_epoch_without_resetting_measured_pose(self):
        first = self.packet()
        first['flight_epochs'] = {'generic_robot': 1}
        self.feed.receive(first)
        before = self.feed.latest.frame.robots[0]
        second = self.packet(sequence=2, time_s=2.)
        second['flight_epochs'] = {'generic_robot': 2}
        self.feed.receive(second)
        after = self.feed.latest.frame.robots[0]
        self.assertEqual((before.trajectory_epoch, after.trajectory_epoch), (1, 2))
        np.testing.assert_array_equal(after.drone.position_world_m, before.drone.position_world_m)
        self.assertEqual(after.time_s, before.time_s)
        self.assertIsNone(after.cable)
        self.assertFalse(self.feed.has_playback)

    def test_heartbeat_cannot_make_a_frozen_capture_pose_fresh(self):
        self.feed.receive(self.packet())
        self.feed.receive(self.packet(sequence=2, time_s=2.))
        self.assertEqual(self.feed.latest.status()[0], 'Stale')

    def test_invalid_capture_cannot_replace_last_valid_display(self):
        self.feed.receive(self.packet())
        before = self.feed.latest
        invalid = self.packet(sequence=2)
        invalid['robots'][0]['quaternion_xyzw'] = [0., 0., 0., 0.]
        with self.assertRaisesRegex(ValueError, 'quaternion'):
            self.feed.receive(invalid)
        self.assertIs(self.feed.latest, before)
        invalid = self.packet(sequence=2)
        invalid['frame_id'] = 'world'
        with self.assertRaisesRegex(ValueError, 'frame mismatch'):
            self.feed.receive(invalid)
        self.assertIs(self.feed.latest, before)

    def test_clock_reset_needs_generation_change_and_clears_display(self):
        self.feed.receive(self.packet())
        reset = self.packet(sequence=2, time_s=0.)
        reset['robots'] = []
        with self.assertRaisesRegex(ValueError, 'clock regressed'):
            self.feed.receive(reset)
        reset['generation'] = 1
        self.feed.receive(reset)
        self.assertEqual(self.feed.latest.frame.robots, ())
        self.assertEqual(self.feed.generation, 1)

    def test_transport_accepts_split_packets_and_stops_on_close(self):
        transport = RealFlightTransport(self.feed, EvidenceStub())
        self.addCleanup(transport.close)
        peer = socket.create_connection(('127.0.0.1', transport.port))
        self.addCleanup(peer.close)
        ready = encode(dict(kind='ready', session='session', frame_id='mocap'))
        peer.sendall(ready[:5]); peer.sendall(ready[5:])
        self.assertTrue(transport.connected.wait(2.))
        peer.sendall(encode(self.packet()))
        end = time.monotonic()+2.
        while self.feed.sequence < 1 and time.monotonic() < end:
            time.sleep(.005)
        self.assertEqual(self.feed.sequence, 1)
        self.feed.close()
        self.assertFalse(transport.thread.is_alive())

    def test_configured_names_all_enabled_and_tracker_provenance(self):
        config = dict(robots={
            'generic_robot': dict(enabled=True, type='vendor', initial_position=[0, 0, 0]),
            'other_robot': dict(enabled=True, type='tracker', initial_position=[3, -2, 1]),
            'disabled': dict(enabled=False, type='vendor')},
            robot_types=dict(vendor=dict(motion_capture=dict(tracking='vendor')),
                             tracker=dict(motion_capture=dict(tracking='librigidbodytracker'))))
        selected = select_real_robots(config)
        self.assertEqual([r['name'] for r in selected], ['generic_robot', 'other_robot'])
        self.assertEqual([r['provenance'] for r in selected], ['Measured', 'Estimated'])
        self.assertEqual(len(select_real_robots(config, ['other_robot'])), 1)
        with self.assertRaisesRegex(ValueError, 'Select enabled'):
            select_real_robots(config, ['missing'])


if __name__ == '__main__':
    unittest.main()
