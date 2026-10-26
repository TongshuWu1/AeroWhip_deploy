"""Offline reward diagnostics; kinematic fixtures are not physical whip evidence."""
import json
from pathlib import Path
import unittest

import numpy as np
import torch

from helpers import wave, project_temporary
from evaluation.reward_review import candidate_return, evaluate, run_manifest, sha256
from training.configuration import FROZEN_PRESETS, load


class RewardReviewTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)
        _, cls.snapshot, _, _ = load(FROZEN_PRESETS[1])
        cls.rest = torch.tensor(cls.snapshot['model']['cable']['marker_interval_lengths_m'],
                                dtype=torch.float64)
        cls.time = torch.arange(451, dtype=torch.float64)*.01

    def history(self, moving_wave=False):
        p, v = wave(self.rest, self.time)
        if not moving_wave:
            p[:] = p[:, :1].clone()
            v.zero_()
        return dict(time_s=self.time.numpy(), cable_positions=p[0].numpy(),
                    cable_velocities=v[0].numpy(), drone_positions=np.tile([0., 0., 2.], (451, 1)),
                    drone_rotations=np.tile(np.eye(3), (451, 1, 1)), state_valid=np.ones(451, bool))

    def late_rigid_crossing(self, *, earlier_wave=False):
        a = self.history(earlier_wave)
        # A deliberately synthetic straight cable translating through the target
        # at 4 m/s, long after the earlier wave. Same start and contained drone.
        start = 250
        material = np.r_[0., self.rest.cumsum(0).numpy()]
        cable = np.stack((material, np.zeros_like(material), np.full_like(material, 2.)), -1)
        a['cable_positions'][start:] = cable
        a['cable_velocities'][start:] = 0
        shift = np.clip((a['time_s']-3.5)*4, 0, .4)
        velocity = np.where((a['time_s'] >= 3.5) & (a['time_s'] < 3.6), 4., 0.)
        a['cable_positions'][start:, :, 0] += shift[start:, None]
        a['cable_velocities'][start:, :, 0] = velocity[start:, None]
        a['drone_positions'][:, 0] = shift
        return a

    def test_envelope_orders_success_completion_and_failure(self):
        failures, completed, success = [], [], []
        for q in np.linspace(0, 1, 11):
            for c in np.linspace(0, 1, 11):
                failures += candidate_return(q, c, failed=True, success=True)
                completed += candidate_return(q, c, failed=False, success=False)
                success += candidate_return(q, c, failed=False, success=True)
        self.assertLess(max(failures), min(completed))
        self.assertLess(max(completed), min(success))
        for q, c in ((float('nan'), 0), (0, -1), (1.1, 0)):
            with self.assertRaises(ValueError):
                candidate_return(q, c, failed=False, success=False)

    def test_hover_zero_and_early_failure_cannot_improve_it(self):
        a = self.history()
        hover, _ = evaluate(a, self.snapshot)
        self.assertEqual(hover['final']['candidate_return_interval'], [0., 0.])
        a = {k: v[:79].copy() for k, v in a.items()}
        a['cable_positions'][-1, -1, 2] = .04
        failed, _ = evaluate(a, self.snapshot)
        self.assertEqual(failed['status'], 'failed')
        self.assertEqual(failed['final']['candidate_return_interval'], [-1., -1.])

    def test_incomplete_prefix_has_no_return_and_frozen_suffix_is_rejected(self):
        a = {k: v[:100].copy() for k, v in self.history().items()}
        result, _ = evaluate(a, self.snapshot)
        self.assertEqual(result['status'], 'incomplete')
        self.assertIsNone(result['final']['candidate_return_interval'])
        a['state_valid'][-1] = False
        with self.assertRaises(ValueError):
            evaluate(a, self.snapshot)

    def test_earlier_wave_cannot_qualify_later_rigid_crossing(self):
        result, _ = evaluate(self.late_rigid_crossing(earlier_wave=True), self.snapshot)
        self.assertGreater(result['best_whip_proxy'], .01)
        self.assertEqual(len(result['crossings']), 1)
        cross = result['crossings'][0]
        self.assertLess(cross['whip_proxy'], 1e-10)
        self.assertLess(cross['time_s'], cross['available_at_s'])
        self.assertEqual(result['final']['strict_crossings'], 1)
        self.assertIsNone(result['final']['reviewed_success'])
        lo, hi = result['final']['candidate_return_interval']
        self.assertAlmostEqual(hi-lo, 5.)

    def test_explicit_crossing_review_and_later_failure(self):
        a = self.late_rigid_crossing()
        result, _ = evaluate(a, self.snapshot)
        index = result['crossings'][0]['interval_index']
        label = dict(interval_index=index, is_whip=False, reviewer='synthetic test',
                     reason='Known rigid translation fixture')
        rejected, _ = evaluate(a, self.snapshot, whip_labels=[label])
        self.assertFalse(rejected['final']['reviewed_success'])
        # Exercise the positive branch as a supplied label, not a physics claim.
        label['is_whip'] = True
        accepted, _ = evaluate(a, self.snapshot, whip_labels=[label])
        self.assertGreaterEqual(accepted['final']['candidate_return_interval'][0], 4.75)
        a['cable_positions'][-1, -1, 2] = .04
        failed, _ = evaluate(a, self.snapshot, whip_labels=[label])
        self.assertEqual(failed['final']['candidate_return_interval'], [-1., -1.])

    def test_containment_is_required_and_cost_is_bounded(self):
        a = self.late_rigid_crossing()
        a['drone_positions'][-100:, 0] = 100
        result, _ = evaluate(a, self.snapshot)
        self.assertFalse(result['contained'])
        self.assertFalse(result['final']['reviewed_success'])
        self.assertGreater(result['bounded_workspace_cost'], 0)
        self.assertLessEqual(result['bounded_workspace_cost'], 1)
        self.assertGreater(result['final']['candidate_return_interval'][0], -1)

    def test_future_samples_do_not_change_crossing_mechanics(self):
        a = self.late_rigid_crossing(earlier_wave=True)
        original, _ = evaluate(a, self.snapshot)
        cut = original['crossings'][0]['interval_index']+2
        prefix, _ = evaluate({k: v[:cut] for k, v in a.items()}, self.snapshot)
        self.assertEqual(original['crossings'][0], prefix['crossings'][0])

    def test_manifest_hashes_and_no_overwrite(self):
        with project_temporary() as tmp:
            root = Path(tmp)
            (root/'snapshot.json').write_text(json.dumps(self.snapshot))
            np.savez_compressed(root/'hover.npz', **self.history())
            ref = dict(path='snapshot.json', sha256=sha256(root/'snapshot.json'))
            manifest = dict(schema='offline_whip_review_inputs_v1', reference_snapshot=ref,
                cases=[dict(name='hover', path='hover.npz', sha256=sha256(root/'hover.npz'),
                            source_snapshot=ref, evidence_kind='synthetic_kinematic_control')])
            path = root/'inputs.json'
            path.write_text(json.dumps(manifest))
            run_manifest(path, root/'result')
            self.assertTrue((root/'result/report.json').exists())
            with self.assertRaises(FileExistsError):
                run_manifest(path, root/'result')
            manifest['cases'][0]['sha256'] = 'wrong'
            path.write_text(json.dumps(manifest))
            with self.assertRaisesRegex(ValueError, 'hash mismatch'):
                run_manifest(path, root/'wrong-hash')

    def test_wrong_clock_and_start_are_rejected(self):
        a = self.history()
        a['time_s'] = a['time_s']*2
        with self.assertRaises(ValueError):
            evaluate(a, self.snapshot)
        a = self.history()
        a['drone_positions'][0, 2] += .1
        with self.assertRaises(ValueError):
            evaluate(a, self.snapshot)
