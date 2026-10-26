"""Soft workspace cost replaces travel without changing the fold guide."""
from copy import deepcopy
from pathlib import Path
import json
import unittest

import numpy as np
import torch

from helpers import project_temporary
from test_fold_release import analytic_fold
from tasks.whip.sac_environment import Stage1Environment
from training.configuration import load, collection_limit
from training.worker import record_simple_collection
from training.artifacts import validate_fold_handoff


PRESET = 'configs/learning/whip_stage1_ppo_fold_formation_soft_workspace_ar1_v2.json'


class FoldSoftWorkspaceTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)
        cls.settings, cls.snapshot, cls.cfg, cls.scene = load(PRESET)
        cls.rest = np.array(cls.snapshot['model']['cable']['marker_interval_lengths_m'])

    def env(self):
        env = Stage1Environment(dict(self.settings, device='cpu', parallel_environments=1),
                                self.snapshot, self.cfg, self.scene, capture=False)
        arrays = analytic_fold(self.rest)
        for key in ('cable_positions', 'cable_velocities'):
            env.history[key][0].copy_(torch.tensor(arrays[key]))
        env.history['drone_positions'][:] = env.origin
        env.numerical_valid[:] = True
        env.failed[:] = False
        env.active[:] = False
        env.decision = env.decisions
        env.lengths[:] = env.decisions
        return env

    def test_contained_motion_is_free_and_path_still_measured(self):
        env = self.env()
        env.history['drone_positions'][0, :, 0] += .4*torch.sin(torch.arange(451)*.04)
        ret, valid, score, _ = env.finish()
        self.assertTrue(bool(valid[0]))
        self.assertGreater(float(score.components['drone_path_length_m'][0]), 1.)
        self.assertEqual(float(score.components['movement_cost'][0]), 0.)
        self.assertEqual(float(score.components['workspace_cost'][0]), 0.)
        self.assertAlmostEqual(float(ret[0]), float(score.score[0]), places=6)
        torch.testing.assert_close(env.transition_rewards.sum(1), ret)

    def test_radial_and_vertical_excess_quadratic_and_failure_clawback(self):
        for position, expected in (((.55, 0., 2.6), 0.), ((.75, 0., 2.), .04),
                                   ((0., 0., 2.8), .04), ((0., 0., 1.3), .04),
                                   ((.95, 0., 3.), .32)):
            with self.subTest(position=position):
                env = self.env()
                env.history['drone_positions'][0, 1:] = torch.tensor(position)
                ret, valid, score, _ = env.finish()
                self.assertAlmostEqual(float(score.components['workspace_cost'][0]), expected, places=6)
                self.assertAlmostEqual(float(ret[0]), float(score.score[0])-expected, places=6)
                torch.testing.assert_close(env.transition_rewards.sum(1), ret)
                env.failed[:] = True
                ret, valid, score, _ = env.finish()
                self.assertFalse(bool(valid[0]))
                self.assertEqual(float(ret[0]), -10.)
                self.assertEqual(float(score.components['workspace_cost'][0]), 0.)
                self.assertAlmostEqual(float(score.components['workspace_accrued_cost'][0]), expected, places=6)
                torch.testing.assert_close(env.transition_rewards.sum(1), ret)

    def test_collection_records_raw_cost_and_net_separately(self):
        env = self.env()
        env.history['drone_positions'][0, 1:, 2] = 2.8
        ret, valid, score, _ = env.finish()
        with project_temporary() as directory:
            directory = Path(directory)
            record_simple_collection(directory, env, ret, valid, score, 1)
            row = json.loads((directory/'collection-00001-events.json').read_text())['lanes'][0]
            self.assertAlmostEqual(row['formation_only_episode_return']-row['workspace_cost'], row['episode_return'], places=6)
            self.assertEqual(row['movement_cost'], 0.)
            self.assertGreater(row['drone_path_length_m'], 0.)

    def test_contract_scope_and_old_release_cannot_restore_path_cost(self):
        self.assertEqual(collection_limit(self.settings), 30)
        self.assertEqual(self.settings['training_episodes'], 15360)
        for penalty in (dict(weight=.1, distance_scale_m=.1, mode='squared_excess_v1'),
                        dict(weight=.01, distance_scale_m=.1, mode='bounded_density_v1')):
            bad = deepcopy(self.settings)
            bad['workspace_penalty'] = penalty
            with self.assertRaises(ValueError):
                load(bad)
        bad = deepcopy(self.settings)
        bad['movement_penalty'] = dict(mode='path_length_v1', weight_per_cable_length=.25)
        with self.assertRaises(ValueError):
            load(bad)
        _, release, _, _ = load('configs/learning/whip_stage1_ppo_fold_release_ar1_v1.json')
        with self.assertRaisesRegex(ValueError, 'preserve the reviewed movement/workspace'):
            validate_fold_handoff(dict(snapshot=self.snapshot), release, 'a'*64)


if __name__ == '__main__':
    unittest.main()
