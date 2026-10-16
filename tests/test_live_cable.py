"""Native cable driven by timestamped poses; no command path or hardware."""
from dataclasses import replace
from types import SimpleNamespace
import unittest

import numpy as np

from deployment.real_flight import RealFlightFeed
from simulation.configuration import load_configuration, DEFAULT_MODEL
from simulation.live_cable import LiveCable
from simulation.model import Model
from viewer.display_state import Provenance


class LiveCableTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.model = Model(DEFAULT_MODEL)

    def test_stationary_boundary_preserves_lengths_and_duplicate_stamp_preserves_history(self):
        cable = LiveCable(self.model)
        for i in range(20):
            state = cable.observe(i*.01, [0, 0, 1], [0, 0, 0, 1])
            self.assertIsNone(cable.error)
            np.testing.assert_allclose(np.linalg.norm(np.diff(state.positions_world_m, axis=0), axis=1),
                self.model.rest, atol=1e-10, rtol=0)
        previous = cable._previous.copy()
        steps = cable.steps
        self.assertIs(cable.observe(state.time_s, [0,0,1], [0,0,0,1]), state)
        self.assertEqual(cable.steps, steps)
        np.testing.assert_array_equal(cable._previous, previous)

    def test_moving_rotating_boundary_follows_attachment_without_resetting_cable(self):
        cable = LiveCable(self.model)
        for i in range(101):
            t = i*.01
            q = np.array([np.sin(.05*t), 0, 0, np.cos(.05*t)])
            state = cable.observe(t, [.1*t,0,1], q)
            self.assertIsNone(cable.error)
        np.testing.assert_allclose(state.positions_world_m[0],
            np.array([.1,0,1])+cable._rotation@self.model.attachment_tracking, atol=1e-12, rtol=0)
        self.assertEqual(cable.steps, 600)
        self.assertGreater(np.linalg.norm(state.velocities_world_mps[1:]), .01)
        self.assertGreater(abs(state.positions_world_m[-1,0]-.1), .001)
        np.testing.assert_allclose(np.linalg.norm(np.diff(state.positions_world_m,axis=0),axis=1),
            self.model.rest, atol=1e-10, rtol=0)

    def test_quaternion_sign_change_does_not_drive_spurious_motion(self):
        positive, negative = LiveCable(self.model), LiveCable(self.model)
        for i in range(11):
            positive.observe(i*.01, [0,0,1], [0,0,0,1])
            negative.observe(i*.01, [0,0,1], [0,0,0,(-1)**i])
        np.testing.assert_array_equal(positive._packed, negative._packed)

    def test_rejected_interval_retries_from_unchanged_history_with_smaller_steps(self):
        model = SimpleNamespace(**vars(self.model), initial_state=self.model.initial_state)
        cable = LiveCable(model)
        cable.observe(0., [0,0,1], [0,0,0,1])
        state, history = cable._packed.copy(), cable._previous.copy()
        calls = []
        original = self.model.native.prescribed_advance
        def advance(packed,samples,previous,previous_step):
            np.testing.assert_array_equal(packed,state)
            np.testing.assert_array_equal(previous,history)
            self.assertEqual(previous_step,0.)
            calls.append(samples.copy())
            if len(calls)==1:
                raise RuntimeError('Prescribed implicit endpoint iteration failed after 20 iterations')
            return original(packed,samples,previous,previous_step)
        model.native = SimpleNamespace(prescribed_advance=advance)
        result = cable.observe(.01, [.001,0,1], [0,0,0,1])
        self.assertIsNone(cable.error)
        self.assertEqual(cable.refinement_retries,1)
        self.assertEqual(cable.maximum_refinement,2)
        self.assertEqual(len(calls[1]),2*len(calls[0]))
        np.testing.assert_allclose(np.linalg.norm(np.diff(result.positions_world_m,axis=0),axis=1),
            self.model.rest,atol=1e-10,rtol=0)

    def test_tracking_gap_freezes_until_explicit_reset(self):
        cable = LiveCable(self.model)
        old = cable.observe(0., [0,0,1], [0,0,0,1])
        self.assertIs(cable.observe(.5, [1,0,1], [0,0,0,1]), old)
        self.assertIn('tracking gap', cable.error)
        self.assertIs(cable.observe(.51, [1,0,1], [0,0,0,1]), old)
        cable.reset()
        self.assertEqual(cable.observe(.51, [1,0,1], [0,0,0,1]).time_s, .51)

    def test_short_capture_interval_actually_subdivides_each_numerical_retry(self):
        for rejection in LiveCable._numerical_rejections:
            with self.subTest(rejection=rejection):
                model = SimpleNamespace(**vars(self.model), initial_state=self.model.initial_state)
                cable = LiveCable(model)
                cable.observe(0., [0,0,1], [0,0,0,1])
                state, history = cable._packed.copy(), cable._previous.copy()
                calls = []
                original = self.model.native.prescribed_advance
                def advance(packed, samples, previous, previous_step):
                    np.testing.assert_array_equal(packed, state)
                    np.testing.assert_array_equal(previous, history)
                    self.assertEqual(previous_step, 0.)
                    calls.append(samples.copy())
                    if len(calls) < 4:
                        raise RuntimeError(rejection)
                    return original(packed, samples, previous, previous_step)
                model.native = SimpleNamespace(prescribed_advance=advance)
                duration = self.model.physics_dt/16
                result = cable.observe(duration, [.000001,0,1], [0,0,0,1])
                self.assertIsNone(cable.error)
                self.assertEqual([len(rows) for rows in calls], [1,2,4,8])
                np.testing.assert_allclose([rows[0,0] for rows in calls],
                    duration/np.array([1,2,4,8]), atol=1e-15, rtol=0)
                self.assertEqual(cable.refinement_retries, 3)
                self.assertEqual(cable.maximum_refinement, 8)
                np.testing.assert_array_equal(result.positions_world_m[0], [.000001,0,.949])
                np.testing.assert_allclose(np.linalg.norm(np.diff(result.positions_world_m,axis=0),axis=1),
                    self.model.rest, atol=1e-10, rtol=0)

    def test_exhausted_or_invalid_input_failure_preserves_last_accepted_state(self):
        for message, expected_calls in (
                ('Prescribed implicit position projection failed', 4),
                ('Invalid boundary rotation', 1)):
            with self.subTest(message=message):
                model = SimpleNamespace(**vars(self.model), initial_state=self.model.initial_state)
                cable = LiveCable(model)
                accepted = cable.observe(0., [0,0,1], [0,0,0,1])
                state, history = cable._packed.copy(), cable._previous.copy()
                calls = []
                def advance(packed, samples, previous, previous_step):
                    np.testing.assert_array_equal(packed, state)
                    np.testing.assert_array_equal(previous, history)
                    calls.append(samples.copy())
                    raise RuntimeError(message)
                model.native = SimpleNamespace(prescribed_advance=advance)
                self.assertIs(cable.observe(.01, [.001,0,1], [0,0,0,1]), accepted)
                self.assertEqual(len(calls), expected_calls)
                self.assertIn(message, cable.error)
                self.assertIs(cable.observe(.02, [.002,0,1], [0,0,0,1]), accepted)
                self.assertEqual(len(calls), expected_calls)
                np.testing.assert_array_equal(cable._packed, state)
                np.testing.assert_array_equal(cable._previous, history)

    def test_feed_advances_all_capture_samples_independent_of_display_heartbeat_and_flight_epochs(self):
        configuration = load_configuration('configs/scenes/default.toml')
        robots = [dict(name='cf_2',provenance='Measured'),dict(name='other',provenance='Measured')]
        feed = RealFlightFeed(configuration,robots,'mocap','test',cable_model=self.model)
        self.addCleanup(feed.close)
        def sample(name, t):
            return dict(name=name,time_s=t,position_world_m=[t*.1 if name=='cf_2' else 2,0,1],quaternion_xyzw=[0,0,0,1])
        first = [sample(name,0.) for name in ('cf_2','other')]
        packet = dict(kind='state',session='test',frame_id='mocap',generation=0,
            capture_type='optitrack',sequence=1,time_s=.001,robots=first,pose_samples=first)
        feed.receive(packet)
        samples = [sample(name, i*.01) for i in range(1,11) for name in ('cf_2','other')]
        packet.update(sequence=2,time_s=.101,robots=samples[-2:],pose_samples=samples,flight_epochs={'cf_2':1})
        feed.receive(packet)
        reference = LiveCable(self.model)
        for s in [first[0], *[s for s in samples if s['name']=='cf_2']]:
            reference.observe(s['time_s'], s['position_world_m'],s['quaternion_xyzw'])
        np.testing.assert_array_equal(feed.cables['cf_2']._packed, reference._packed)
        self.assertEqual(feed.cables['cf_2'].steps,60)
        displayed = feed.latest.frame.robots[0]
        self.assertEqual(displayed.provenance, Provenance.MEASURED)
        self.assertEqual(displayed.cable.provenance, Provenance.SIMULATED)
        self.assertIsNone(displayed.drone.linear_velocity_world_mps)
        packet.update(sequence=3,time_s=.2,pose_samples=[],flight_epochs={'cf_2':2})
        feed.receive(packet)
        self.assertEqual(feed.cables['cf_2'].steps,60)
        self.assertEqual(feed.latest.frame.robots[0].trajectory_epoch,2)
        self.assertEqual(feed.cables['other'].steps,60)
        # The heartbeat may lose source verification after an already accepted
        # capture callback; its buffered prefix must still advance exactly once.
        accepted = sample('cf_2',.11)
        packet.update(sequence=4,time_s=.205,capture_type=None,robots=[accepted],pose_samples=[accepted])
        feed.receive(packet)
        self.assertEqual(feed.cables['cf_2'].steps,66)
        np.testing.assert_array_equal(feed.latest.frame.robots[0].attachment_position_world_m,
            feed.latest.frame.robots[0].cable.positions_world_m[0])
        packet.update(sequence=5,time_s=.21,pose_samples_complete=False)
        feed.receive(packet)
        self.assertIn('buffer overflowed', feed.latest.error)
        packet.update(sequence=6,time_s=0.,generation=1,robots=first,pose_samples=first,pose_samples_complete=True,capture_type='optitrack')
        feed.receive(packet)
        self.assertIsNone(feed.latest.error)
        self.assertEqual(feed.cables['cf_2'].steps,0)


if __name__ == '__main__':
    unittest.main()
