"""Display ownership, replay isolation and short native timing/parity checks."""
from dataclasses import replace
import threading
import time
import unittest
from unittest.mock import patch

import numpy as np

from simulation.configuration import DEFAULT_MODEL, load_configuration
from simulation.runtime import Simulation
from simulation.state import DroneState
from trajectories.presets import simple_motion
from viewer.display_state import CableDisplay, DisplayFrame, Provenance, RobotDisplay
from viewer.replay import StateReplay
from viewer.sources import DisplayFeed, ReplaySource, SimulationSource, SourceSnapshot


class DisplayTests(unittest.TestCase):
    def robot(self, name='cf_2', cable=None):
        return RobotDisplay(name, 1., Provenance.MEASURED,
            DroneState(np.array([0., 0., 2.]), np.array([0., 0., 0., 1.]), None, None),
            np.array([0., 0., 1.949]), cable)

    def test_optional_cable_missing_telemetry_and_provenance(self):
        cable = CableDisplay(.8, Provenance.ESTIMATED, np.array([[0., 0., 1.9], [0., 0., .9]]), np.array([0, 1]))
        frame = DisplayFrame(1., (self.robot(), self.robot('cf_3', cable)), source_label='Real flight')
        self.assertIsNone(frame.robots[0].cable)
        self.assertIsNone(frame.robots[0].drone.linear_velocity_world_mps)
        self.assertEqual(frame.robots[1].cable.provenance, Provenance.ESTIMATED)
        self.assertEqual(frame.robots[1].cable.time_s, .8)
        with self.assertRaisesRegex(ValueError, 'unique'):
            DisplayFrame(1., (self.robot(), self.robot()))
        with self.assertRaisesRegex(ValueError, 'timeline'):
            DisplayFrame(.9, (self.robot(),))

    def test_published_arrays_cannot_mutate_producer_or_shared_frame(self):
        raw = np.array([0., 0., 2.])
        robot = replace(self.robot(), drone=DroneState(raw, np.array([0., 0., 0., 1.]), None, None))
        raw[:] = 99
        np.testing.assert_array_equal(robot.drone.position_world_m, [0., 0., 2.])
        with self.assertRaises(ValueError):
            robot.drone.position_world_m[0] = 1
        with self.assertRaises(ValueError):
            robot.drone.position_world_m.flags.writeable = True

    def test_freshness_cannot_label_errors_or_old_running_snapshots_fresh(self):
        frame = DisplayFrame(1., (self.robot(),))
        snapshot = SourceSnapshot(1, None, frame, True, None, 1, 1., 10.)
        self.assertEqual(snapshot.status(10.3)[0], 'Stale')
        self.assertEqual(snapshot.status(10.1)[0], 'Running')
        self.assertEqual(replace(snapshot, running=False).status(15.)[0], 'Paused')
        self.assertEqual(replace(snapshot, error='solver rejected').status(10.1)[0], 'Error')

    def test_external_feed_has_no_plant_or_playback_and_rejects_publish_after_close(self):
        frame = DisplayFrame(1., (self.robot(),), source_label='Synthetic display fixture')
        feed = DisplayFeed(frame, None)
        self.assertFalse(feed.has_playback)
        self.assertIsNone(feed.latest.state)
        self.assertIs(feed.tick(.5).frame, frame)
        self.assertEqual(feed.latest.status(feed.latest.published_monotonic_s+.1)[0], 'Live')
        before = feed.latest
        feed.publish(replace(frame, time_s=2.))
        self.assertEqual(feed.latest.status(feed.latest.published_monotonic_s+.1)[0], 'Stale')
        self.assertEqual(before.frame.time_s, 1.)
        with self.assertRaisesRegex(ValueError, 'world frame'):
            feed.publish(replace(frame, frame_id='other'))
        feed.close()
        self.assertEqual(feed.latest.status()[0], 'Error')
        with self.assertRaisesRegex(RuntimeError, 'closed'):
            feed.publish(frame)


class SimulationSourceTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.scene = load_configuration('configs/scenes/default.toml')

    def simulation(self, source=None):
        return Simulation(self.scene, model_config=DEFAULT_MODEL, command_source=source)

    def test_native_checkpoints_and_command_clocks_match_without_rendering(self):
        trajectory = simple_motion('Move', self.scene.initial_state.position_world_m,
                                   duration_s=2., timestep_s=1/30)
        direct = self.simulation(trajectory)
        source = SimulationSource(self.simulation(trajectory))
        try:
            for i in range(30):
                direct.step(); source.step()
                if i in (8, 20):
                    source.set_force_telemetry(i == 8)
                np.testing.assert_array_equal(source.checkpoint()['state'], direct.dynamics.checkpoint()['state'])
            self.assertEqual(source.step_count, 30)
            self.assertEqual(source.state.time_s, .3)
            np.testing.assert_array_equal(source.state.cable_node_positions_world_m,
                                          direct.state.cable_node_positions_world_m)
        finally:
            source.close()
        self.assertFalse(source._thread.is_alive())

    def test_simulation_advances_while_render_consumer_is_idle(self):
        owner = threading.get_ident()
        polling_threads = set()
        from simulation.controller_interface import ControlInput
        def controls():
            polling_threads.add(threading.get_ident())
            return ControlInput()
        source = SimulationSource(self.simulation(), control_source=controls)
        try:
            first = source.latest
            source.running = True
            time.sleep(.12)  # Represents blocked/no rendering; no Qt event processing.
            source.running = False
            self.assertGreaterEqual(source.step_count, 4)
            self.assertEqual(polling_threads, {source._thread.ident})
            self.assertNotIn(owner, polling_threads)
            self.assertEqual(first.state.time_s, 0.)
            held = source.step_count
            time.sleep(.025)
            self.assertEqual(source.step_count, held)
            source.reset()
            self.assertEqual(source.step_count, 0)
        finally:
            source.close()

    def test_failed_step_is_visible_and_reset_recovers(self):
        def fail(index):
            raise ValueError('Rejected command fixture')
        source = SimulationSource(self.simulation(fail))
        try:
            with self.assertRaisesRegex(ValueError, 'Rejected command'):
                source.step()
            self.assertIn('Rejected command', source.latest.error)
            self.assertFalse(source.running)
            self.assertEqual(source.step_count, 0)
            source.command_source = None
            source.reset(); source.step()
            self.assertIsNone(source.latest.error)
            self.assertEqual(source.step_count, 1)
        finally:
            source.close()

    def test_invalid_publication_retains_valid_prefix_and_keeps_worker_responsive(self):
        sim = self.simulation()
        source = SimulationSource(sim)
        try:
            initial = source.state
            bad = replace(initial, drone=replace(initial.drone, quaternion_body_to_world_xyzw=np.zeros(4)))
            with patch.object(sim.dynamics, 'snapshot', return_value=bad):
                source.step()
            self.assertIn('Display publication failed', source.latest.error)
            self.assertFalse(source.running)
            np.testing.assert_array_equal(source.state.drone.quaternion_body_to_world_xyzw,
                                          initial.drone.quaternion_body_to_world_xyzw)
            source.reset(); source.step()
            self.assertIsNone(source.latest.error)
        finally:
            source.close()

    def test_reference_export_and_source_switch_preserve_control_interface(self):
        source = SimulationSource(self.simulation())
        try:
            command = np.r_[self.scene.initial_state.position_world_m, np.zeros(8)]
            np.testing.assert_array_equal(source.dynamics.controller.prepare_reference(command),
                                          self.simulation().dynamics.controller.prepare_reference(command))
            source.running = True
            with self.assertRaisesRegex(ValueError, 'Pause'):
                source.command_source = None
            source.running = False
            trajectory = simple_motion('Hover', self.scene.initial_state.position_world_m)
            source.command_source = trajectory; source.command_reset = None
            source.reset(); source.step()
            self.assertIs(source.command_source, trajectory)
        finally:
            source.close()
        with self.assertRaisesRegex(RuntimeError, 'closed'):
            source.step()


class ReplaySourceTests(unittest.TestCase):
    def test_replay_seek_step_speed_and_eof_use_only_saved_arrays(self):
        scene = load_configuration('configs/scenes/default.toml')
        arrays = dict(time_s=np.array([0., .01, .02]), drone_positions=np.array([[0., 0., 2.], [.1, 0., 2.], [.2, 0., 2.]]),
            drone_velocities=np.zeros((3, 3)), drone_rotations=np.tile(np.eye(3), (3, 1, 1)),
            drone_angular_velocities=np.zeros((3, 3)), cable_positions=np.tile([[0., 0., 1.949], [0., 0., 1.]], (3, 1, 1)),
            cable_velocities=np.zeros((3, 2, 3)), cable_marker_indices=np.array([0, 1]), state_valid=np.ones(3, dtype=bool))
        originals = {key: value.copy() for key, value in arrays.items()}
        source = ReplaySource(StateReplay(arrays), scene)
        source.step(); self.assertEqual(source.state.time_s, .01)
        source.seek(0); source.set_speed(.5); source.toggle_running(); source.tick(.04)
        self.assertEqual(source.state.time_s, .02)
        self.assertFalse(source.latest.running)
        self.assertTrue(source.latest.frame.recorded)
        source.toggle_running(); self.assertEqual(source.state.time_s, 0.)
        source.reset(); source.close()
        for key, value in originals.items(): np.testing.assert_array_equal(arrays[key], value)


if __name__ == '__main__':
    unittest.main()
