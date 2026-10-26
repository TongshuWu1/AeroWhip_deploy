"""Native cable driven by timestamped poses; no command path or hardware."""
import unittest

import numpy as np

from aerowhip.physics.configuration import DEFAULT_MODEL
from aerowhip.physics.live_cable import LiveCable
from aerowhip.physics.model import Model


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

if __name__ == '__main__':
    unittest.main()
