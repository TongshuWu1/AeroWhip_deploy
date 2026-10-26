"""Projection/invariance checks; no physical-whip classification assertions."""
import unittest
import numpy as np

from evaluation.whip_measurement_examples import _chain
from evaluation.whip_output_decomposition import decompose


class OutputDecompositionTests(unittest.TestCase):
    def setUp(self):
        self.rest = np.linspace(.06, .12, 10)
        self.t = np.arange(41)*.01
        phase = self.t[:, None]*3+np.arange(10)[None]*.3
        self.p, self.v = _chain(self.rest, .4*np.sin(phase), 1.2*np.cos(phase))

    def test_reconstruction_and_weighted_orthogonality(self):
        r = decompose(self.p, self.v, self.rest)
        np.testing.assert_allclose(r['fitted_rigid_velocity_world_mps']+r['shape_velocity_world_mps'],
                                   self.v-self.v[:, :1], atol=1e-14)
        moment = (np.cross(self.p-self.p[:, :1], r['shape_velocity_world_mps'])*
                  r['rest_length_weights'][None, :, None]).sum(1)
        np.testing.assert_allclose(moment, 0., atol=1e-14)

    def test_arbitrary_superposed_rigid_motion_covariance(self):
        axis = np.array([1., 2., 3.]); axis /= np.linalg.norm(axis)
        x, y, z = axis
        cross = np.array([[0., -z, y], [z, 0., -x], [-y, x, 0.]])
        angle = .3+7*self.t**2
        rotation = (np.eye(3)[None]+np.sin(angle)[:, None, None]*cross+
                    (1-np.cos(angle))[:, None, None]*(cross@cross))
        rp = np.einsum('tij,tnj->tni', rotation, self.p)
        rv = np.einsum('tij,tnj->tni', rotation, self.v)
        omega = 14*self.t[:, None]*axis
        shift = np.column_stack((self.t**2, np.sin(self.t), -3*self.t))
        rate = np.column_stack((2*self.t, np.cos(self.t), np.full_like(self.t, -3)))
        original = decompose(self.p, self.v, self.rest)
        transformed = decompose(rp+shift[:, None], rv+np.cross(omega[:, None], rp)+rate[:, None], self.rest)
        expected = np.einsum('tij,tnj->tni', rotation, original['shape_velocity_world_mps'])
        np.testing.assert_allclose(transformed['shape_velocity_world_mps'], expected, atol=2e-14)
        np.testing.assert_allclose(transformed['shape_speed_mps'], original['shape_speed_mps'], atol=2e-14)

    def test_pure_rigid_motion_is_removed(self):
        p = np.broadcast_to(self.p[0], self.p.shape)
        omega = np.column_stack((self.t*4, self.t*7, np.ones_like(self.t)))
        v = np.cross(omega[:, None], p)+np.array([4., -3., 2.])
        r = decompose(p, v, self.rest)
        np.testing.assert_allclose(r['shape_speed_mps'], 0., atol=1e-14)

    def test_straight_chain_rank_deficiency_is_handled(self):
        p, v = _chain(self.rest, np.zeros((1, 10)), np.full((1, 10), 2.))
        r = decompose(p, v, self.rest)
        self.assertEqual(r['fit_rank'].tolist(), [2])
        np.testing.assert_allclose(r['shape_speed_mps'], 0., atol=1e-14)

    def test_localized_terminal_rotation_is_not_removed(self):
        angles = np.zeros((len(self.t), 10)); rates = np.zeros_like(angles)
        angles[:, 8:] = 4*self.t[:, None]; rates[:, 8:] = 4
        p, v = _chain(self.rest, angles, rates)
        r = decompose(p, v, self.rest)
        self.assertGreater(r['shape_speed_mps'][:, -1].max(), .4)
        self.assertGreater(r['shape_speed_mps'][:, -1].max()/np.linalg.norm(v[:, -1], axis=-1).max(), .5)

    def test_input_arrays_are_preserved(self):
        p, v = self.p.copy(), self.v.copy()
        decompose(p, v, self.rest)
        np.testing.assert_array_equal(p, self.p)
        np.testing.assert_array_equal(v, self.v)

    def test_invalid_measurements_rejected(self):
        for p, v, rest in [(self.p, self.v[:, :-1], self.rest),
                           (self.p, self.v, np.zeros(10)),
                           (self.p*np.nan, self.v, self.rest),
                           (self.p*0, self.v, self.rest)]:
            with self.assertRaises(ValueError):
                decompose(p, v, rest)


if __name__ == '__main__':
    unittest.main()
