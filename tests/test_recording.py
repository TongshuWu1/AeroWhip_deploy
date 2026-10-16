"""Display recording/replay and capture-only launch checks; no real hardware."""
from dataclasses import replace
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import Mock, patch

import numpy as np

from deployment.real_flight import RealFlightFeed
from deployment.real_stack import capture_configuration, stack_command
from simulation.configuration import load_configuration
from simulation.state import DroneState
from viewer.display_state import CableDisplay, CableTrackingDisplay, DisplayFrame, Provenance, RobotDisplay
from viewer.recording import DisplayRecorder, RecordingSource, decode_frame, json_value
from viewer.sources import DisplayFeed


class RecordingTests(unittest.TestCase):
    def setUp(self):
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        self.directory = Path(directory.name)
        self.configuration = load_configuration('configs/scenes/default.toml')

    def frame(self, t=10.):
        tracking = CableTrackingDisplay(t-.03, np.array([0., 0., 1.]),
            np.array([[0., 0., .9], [0., 0., 0.], [0., 0., .7]]),
            np.array([True, False, True]), np.array([.7, 0., .9]),
            np.array([[0., 0., .9], [0., 0., .7]]), 'partial')
        robot = RobotDisplay('cf_2', t-.01, Provenance.MEASURED,
            DroneState(np.array([0., 0., 1.05]), np.array([0., 0., 0., 1.]), None, None),
            np.array([0., 0., 1.]), CableDisplay(t-.02, Provenance.SIMULATED,
                np.array([[0., 0., 1.], [0., 0., .5]]), np.array([0, 1])),
            trajectory_epoch=2, cable_tracking=tracking)
        return DisplayFrame(t, (robot,), frame_id='mocap', source_label='Real flight')

    def test_complete_round_trip_preserves_missing_fields_and_provenance(self):
        frame = self.frame()
        self.assertEqual(json_value(decode_frame(json_value(frame))), json_value(frame))
        feed = DisplayFeed(frame, self.configuration)
        recorder = DisplayRecorder(self.directory, self.configuration, dict(model='fixed'), feed.latest)
        recorder.event('capture', packet=dict(points=[[None, 1., 2.]], stamp_ns=123))
        feed.publish(self.frame(10.1), error='Retained tracking error')
        recorder.frame(feed.latest)
        # Original ROS clock regression remains valid on a monotonic clip timeline.
        feed.publish(self.frame(.1))
        recorder.frame(feed.latest)
        path = recorder.close()
        source = RecordingSource(path)
        self.addCleanup(source.close)
        self.assertTrue(source.replay.metadata['complete'])
        self.assertEqual(len(source.replay.time), 3)
        for index, expected in enumerate((frame, self.frame(10.1), self.frame(.1))):
            source.seek(index)
            actual = replace(source.latest.frame, recorded=False, source_label='Real flight')
            self.assertEqual(json_value(actual), json_value(expected))
        source.seek(1)
        self.assertEqual(source.latest.error, 'Retained tracking error')
        self.assertIsNone(source.latest.frame.robots[0].drone.linear_velocity_world_mps)
        self.assertEqual(source.configuration.viewer.visual_rotor_rpm, self.configuration.viewer.visual_rotor_rpm)
        self.assertEqual(json.loads(path.with_name('events.jsonl').read_text())['packet']['stamp_ns'], 123)

    def test_record_controls_do_not_reset_live_state_and_multiple_clips_are_unique(self):
        feed = RealFlightFeed(self.configuration, [dict(name='cf_2', provenance='Measured')], 'mocap', 's')
        self.addCleanup(feed.close)
        feed.configure_recording(self.directory, {})
        packet = dict(kind='state', session='s', frame_id='mocap', generation=0,
            sequence=1, time_s=1., capture_type='optitrack',
            robots=[dict(name='cf_2', time_s=1., position_world_m=[0., 0., 1.], quaternion_xyzw=[0., 0., 0., 1.])])
        feed.receive(packet)
        before = json_value(feed.latest.frame)
        first = feed.start_recording()
        with self.assertRaises(RuntimeError):
            feed.start_recording()
        feed.receive(dict(packet, sequence=2, time_s=1.1))
        feed.stop_recording()
        self.assertEqual(feed.sequence, 2)
        self.assertEqual(json_value(feed.latest.frame)['robots'], before['robots'])
        second = feed.start_recording()
        feed.close()  # Finalizes an active recording.
        self.assertNotEqual(first, second)
        for path in (first, second):
            self.assertTrue(json.loads(path.read_text())['complete'])
            with RecordingSource(path).replay.file:
                pass

    def test_incomplete_final_line_is_recoverable_but_complete_corruption_is_rejected(self):
        feed = DisplayFeed(self.frame(), self.configuration)
        recorder = DisplayRecorder(self.directory, self.configuration, {}, feed.latest)
        path = recorder.close()
        metadata = json.loads(path.read_text())
        metadata['complete'] = False
        path.write_text(json.dumps(metadata))
        with path.with_name('frames.jsonl').open('ab') as file:
            file.write(b'{"elapsed_s":')
        source = RecordingSource(path)
        self.assertIn('Incomplete', source.latest.error)
        self.assertEqual(len(source.replay.time), 1)
        source.close()
        metadata['complete'] = True
        path.write_text(json.dumps(metadata))
        with self.assertRaises(ValueError):
            RecordingSource(path)

    def test_write_failure_is_explicit_and_never_marks_clip_complete(self):
        feed = DisplayFeed(self.frame(), self.configuration)
        recorder = DisplayRecorder(self.directory, self.configuration, {}, feed.latest)
        original = recorder.events
        recorder.events = Mock(wraps=original)
        recorder.events.write.side_effect = OSError('disk full')
        recorder.event('capture', packet={})
        self.assertFalse(recorder.active)
        self.assertIn('disk full', recorder.error)
        path = recorder.close()
        self.assertFalse(json.loads(path.read_text())['complete'])

    def test_play_step_seek_speed_and_end_preserve_recorded_errors(self):
        feed = DisplayFeed(self.frame(), self.configuration)
        with patch('viewer.recording.perf_counter', side_effect=[100., 101., 102.]):
            recorder = DisplayRecorder(self.directory, self.configuration, {}, feed.latest)
            feed.publish(self.frame(11.), error='capture gap')
            recorder.frame(feed.latest)
            feed.publish(self.frame(12.))
            recorder.frame(feed.latest)
        source = RecordingSource(recorder.close())
        self.addCleanup(source.close)
        source.set_speed(.5)
        source.toggle_running()
        source.tick(2.)
        self.assertEqual(source.step_count, 1)
        self.assertEqual(source.latest.error, 'capture gap')
        source.tick(2.)
        self.assertEqual(source.step_count, 2)
        self.assertFalse(source.running)
        source.toggle_running()
        self.assertEqual(source.step_count, 0)
        source.step()
        self.assertEqual(source.step_count, 1)
        self.assertFalse(source.running)
        source.seek(0)
        self.assertEqual(source.playback_time_s, 0.)

    def test_capture_only_command_has_no_flight_server_and_preserves_tracking_config(self):
        paths = dict(crazyflies=Path('/a/cf.yaml'), motion_capture=Path('/a/mocap.yaml'))
        command = stack_command(paths, True)
        self.assertEqual(command[:4], ['ros2', 'run', 'motion_capture_tracking', 'motion_capture_tracking_node'])
        self.assertNotIn('launch.py', command)
        # YAML parsing belongs to the ROS container; exercise the mapping here
        # with JSON (also a YAML subset), without adding host dependencies.
        import sys
        base = dict(fileversion=2, robots={
            'cf_2': dict(enabled=True, type='vendor', initial_position=[0, 0, 0]),
            'tracked': dict(enabled=True, type='tracker', initial_position=[1, 2, 3])},
            robot_types={'vendor': dict(motion_capture=dict(tracking='vendor')),
                         'tracker': dict(motion_capture=dict(tracking='librigidbodytracker', marker='test', dynamics='default'))})
        original = dict(crazyflies=json.dumps(base), motion_capture=json.dumps({
            '/motion_capture_tracking': {'ros__parameters': dict(type='optitrack', hostname='unchanged')}}))
        with patch.dict(sys.modules, yaml=Mock(safe_load=json.loads, safe_dump=json.dumps)):
            result = json.loads(capture_configuration(original))['/motion_capture_tracking']['ros__parameters']
        self.assertEqual(result['hostname'], 'unchanged')
        self.assertEqual(result['rigid_bodies'], {'tracked': dict(initial_position=[1, 2, 3], marker='test', dynamics='default')})
        base['robots']['tracked']['enabled'] = False
        original['crazyflies'] = json.dumps(base)
        with patch.dict(sys.modules, yaml=Mock(safe_load=json.loads, safe_dump=json.dumps)):
            result = json.loads(capture_configuration(original))['/motion_capture_tracking']['ros__parameters']
        self.assertNotIn('rigid_bodies', result)

    def test_capture_only_worker_crash_cleans_up_without_waiting_for_flight_stop(self):
        import run_custom_real
        from types import SimpleNamespace
        child = Mock(returncode=-11)
        child.poll.return_value = -11
        stop = Mock()
        stop.is_set.return_value = False
        with patch.object(run_custom_real, 'Event', return_value=stop), \
             patch.object(run_custom_real.signal, 'signal'), \
             patch.object(run_custom_real.subprocess, 'Popen', return_value=child), \
             patch.object(run_custom_real, 'stop_real_stack', return_value=False) as cleanup:
            result = run_custom_real.supervise_viewer(SimpleNamespace(
                attach=False, capture_only=True, container='fixture', domain=42), [])
        self.assertEqual(result, 139)
        stop.wait.assert_not_called()
        cleanup.assert_called_once()


if __name__ == '__main__':
    unittest.main()
