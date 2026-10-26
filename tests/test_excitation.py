"""Flexible-excitation arithmetic and analytic controls, not whip validation."""
from copy import deepcopy
import unittest

import numpy as np
import torch

from evaluation.whip_measurements import extract_measurements
from evaluation.whip_output_decomposition import decompose
from evaluation.whip_measurement_examples import kinematic_controls
from tasks.whip.excitation import MODE, ExcitationScorer, score_events
from training.configuration import load


def inputs(world, shape):
    world, shape = np.asarray(world, float), np.asarray(shape, float)
    n = len(world)
    w = np.zeros((n, 3)); w[:, -1] = world
    s = np.zeros_like(w); s[:, -1] = shape
    sv = np.zeros((n, 3, 3)); sv[..., 0] = s
    return dict(time_s=np.arange(n)*.01, world_speed_mps=w,
                world_forward_mps=w.copy(), relative_speed_mps=w.copy()), dict(
                shape_speed_mps=s, shape_velocity_world_mps=sv)


class ExcitationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)
        cls.rest = np.full(10, .1)
        cls.controls = {c['name']: c['arrays'] for c in kinematic_controls(cls.rest)}

    def score(self, m, d, **kwargs):
        return score_events(m, d, [1, 0, 0], complete=kwargs.get('complete', True),
                            failed=kwargs.get('failed', False))

    def analytic(self, arrays):
        return self.score(extract_measurements(arrays, self.rest, [1, 0, 0]),
                          decompose(arrays['cable_positions'], arrays['cable_velocities'], self.rest))

    def test_same_sample_minimum_not_minimum_of_independent_peaks(self):
        m, d = inputs([0, 8, 1, 3], [0, 1, 9, 2])
        result = self.score(m, d)
        self.assertEqual(result['best_event_score'], 2.)
        selected = result['events'][result['selected_event']]
        self.assertEqual(selected['q_shape_index'], 3)
        self.assertEqual(selected['world_speed_at_q'], 3.)
        self.assertEqual(selected['shape_speed_at_q'], 2.)

    def test_record_differences_telescope_and_ties_choose_earliest(self):
        m, d = inputs([0, 1, 1, .2, 3, 2], [0, 2, 2, 2, 4, 3])
        result = self.score(m, d)
        np.testing.assert_array_equal(result['causal_best'], [0, 1, 1, 1, 3, 3])
        self.assertEqual([e['q_shape_index'] for e in result['events']], [1, 4])
        self.assertEqual(np.diff(result['causal_best']).sum(), result['episode_return'])
        self.assertEqual(result['mechanism_status'], 'unresolved')
        self.assertEqual(result['segments'], [])

    def test_initial_state_is_unpaid_baseline_and_single_sample_has_no_credit(self):
        m, d = inputs([100, 1, 2], [100, 3, 3])
        self.assertEqual(self.score(m, d)['best_event_score'], 2.)
        m, d = inputs([100], [100])
        result = self.score(m, d)
        self.assertEqual(result['best_event_score'], 0.)
        self.assertEqual(result['events'], [])
        np.testing.assert_array_equal(result['causal_best'], [0.])

    def test_failure_preserves_raw_prefix_and_pending_return_is_none(self):
        m, d = inputs([0, 2, 1], [0, 3, 4])
        good = self.score(m, d)
        failed = self.score(m, d, complete=False, failed=True)
        pending = self.score(m, d, complete=False)
        self.assertEqual(failed['episode_return'], -10.)
        self.assertEqual(failed['best_event_score'], 2.)
        np.testing.assert_array_equal(good['causal_best'], failed['causal_best'])
        self.assertIsNone(pending['episode_return'])

    def test_direction_and_gain_are_not_reward_gates(self):
        m, d = inputs([0, 2, 3], [10, 2, 3])
        m['world_forward_mps'] *= -1
        d['shape_velocity_world_mps'] *= -1
        a = self.score(m, d)
        b = score_events(m, d, [0, 1, 0], complete=True, failed=False)
        self.assertEqual(a['best_event_score'], 3.)
        self.assertEqual(a['best_event_score'], b['best_event_score'])
        self.assertLess(a['events'][-1]['shape_gain_mps'], 0.)
        self.assertLess(a['events'][-1]['world_forward_mps'], 0.)
        self.assertLess(a['events'][-1]['shape_forward_mps'], 0.)

    def test_hover_translation_and_rigid_rotation_have_no_substantial_credit(self):
        translated = self.controls['translated_curved_shape']
        hovering = dict(translated, cable_positions=np.broadcast_to(
            translated['cable_positions'][0], translated['cable_positions'].shape).copy(),
            cable_velocities=np.zeros_like(translated['cable_velocities']))
        self.assertEqual(self.analytic(hovering)['best_event_score'], 0.)
        self.assertEqual(self.analytic(translated)['best_event_score'], 0.)
        self.assertLess(self.analytic(self.controls['rigid_curved_rotation'])['best_event_score'], 1e-12)

    def test_stationary_world_tip_with_moving_attachment_gets_zero_credit(self):
        a = deepcopy(self.controls['small_travelling_wave'])
        tip = a['cable_positions'][:, -1:].copy()
        tip_velocity = a['cable_velocities'][:, -1:].copy()
        a['cable_positions'] -= tip
        a['cable_velocities'] -= tip_velocity
        m = extract_measurements(a, self.rest, [1, 0, 0])
        self.assertGreater(m['relative_speed_mps'][:, -1].max(), 0.)
        self.assertEqual(self.analytic(a)['best_event_score'], 0.)

    def test_small_wave_stays_small_and_plane_rotation_preserves_credit(self):
        a = self.controls['small_travelling_wave']
        original = self.analytic(a)
        rotation = np.array([[1., 0, 0], [0, 0, -1], [0, 1, 0]])
        changed = dict(a, cable_positions=a['cable_positions']@rotation.T,
                       cable_velocities=a['cable_velocities']@rotation.T)
        rotated = self.analytic(changed)
        self.assertGreater(original['best_event_score'], 0.)
        self.assertLess(original['best_event_score'], .01)
        self.assertAlmostEqual(original['best_event_score'], rotated['best_event_score'], places=13)

    def test_scorer_components_and_failure_mask_with_retained_finite_prefix(self):
        _, _, cfg, _ = load('configs/learning/whip_stage1_ppo_unified_fold_ar1_exp_v2.json')
        rest = torch.tensor(self.rest, dtype=torch.float64)
        time = torch.arange(451, dtype=torch.float64)*.01
        scorer = ExcitationScorer(rest, time, cfg, initial_position=[0, 0, 2])
        arrays = self.controls['small_travelling_wave']
        p = torch.tensor(arrays['cable_positions'][None]).repeat(2, 1, 1, 1)
        v = torch.tensor(arrays['cable_velocities'][None]).repeat(2, 1, 1, 1)
        valid = torch.ones((2, 451), dtype=torch.bool); valid[1, 101:] = False
        result = scorer(p, v, valid, failed=torch.tensor([False, True]))
        self.assertEqual(result.valid.tolist(), [True, False])
        self.assertNotIn('shape_release_score', result.components)
        self.assertIn('excitation_score', result.components)
        self.assertTrue(torch.isnan(result.score[1]))
        self.assertGreater(float(result.causal_best[1, 100]), 0.)
        self.assertTrue(torch.equal(result.causal_best[1, 101:], result.causal_best[1, 100].expand(350)))
        self.assertEqual(scorer.event_records[0]['schema'], MODE)
        self.assertEqual(scorer.event_records[1]['episode_return'], -10.)
        e = scorer.event_records[0]['events'][scorer.event_records[0]['selected_event']]
        self.assertAlmostEqual(float(result.components['world_speed_at_q'][0]), e['world_speed_at_q'])
        self.assertAlmostEqual(float(result.components['shape_speed_at_q'][0]), e['shape_speed_at_q'])

    def test_invalid_shapes_values_flags_and_missing_fields_rejected(self):
        m, d = inputs([0, 1, 2], [0, 1, 2])
        for key, value in [('world_speed_mps', np.ones((3, 2))),
                           ('world_speed_mps', -np.ones((3, 3))),
                           ('relative_speed_mps', np.full((3, 3), np.nan)),
                           ('time_s', [0, 0, .1])]:
            changed = dict(m, **{key: value})
            with self.subTest(key=key), self.assertRaises(ValueError): self.score(changed, d)
        with self.assertRaises(ValueError): self.score({}, d)
        with self.assertRaises(ValueError): self.score(m, dict(d, shape_velocity_world_mps=np.zeros((3, 3))))
        with self.assertRaises(ValueError): score_events(m, d, [0, 0, 0], complete=True, failed=False)
        with self.assertRaises(ValueError): score_events(m, d, [1, 0, 0], complete=1, failed=False)


if __name__ == '__main__':
    unittest.main()
