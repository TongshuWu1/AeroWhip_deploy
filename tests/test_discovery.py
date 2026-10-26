"""Approved Stage1 reward accounting; synthetic motion is not training evidence."""
from copy import deepcopy
from dataclasses import replace
import os
import unittest
from unittest.mock import patch

import numpy as np
import torch

from helpers import wave, project_temporary
from training.configuration import DEFAULT, CURRENT_PRESETS, load
from training.artifacts import identity, validate_actor_source
from training.worker import record_simple_collection
from tasks.whip.sac_environment import Stage1Environment
from tasks.whip.stage1 import Stage1Scorer
from tasks.whip.workspace import WorkspacePenalty


PRESET = 'configs/learning/whip_stage1_ppo_acceleration_discovery_v1.json'


class DiscoveryTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)
        cls.settings, cls.snapshot, cls.cfg, cls.scene = load(PRESET)
        cls.rest = torch.tensor(cls.snapshot['model']['cable']['marker_interval_lengths_m'], dtype=torch.float64)
        cls.time = torch.arange(451, dtype=torch.float64)*.01
        cls.positions, cls.velocities = wave(cls.rest, cls.time)
        cls.observed = torch.ones(1, 451, dtype=torch.bool)

    def scorer(self, cfg=None, device='cpu'):
        return Stage1Scorer(self.rest.to(device), self.time.to(device), cfg or self.cfg,
                            initial_position=[0, 0, 2])

    def environment(self):
        env = Stage1Environment(dict(self.settings, device='cpu', parallel_environments=1),
                                self.snapshot, self.cfg, self.scene, capture=False)
        env.history['cable_positions'].copy_(self.positions)
        env.history['cable_velocities'].copy_(self.velocities)
        env.history['drone_positions'][:] = env.origin+torch.tensor([.8, 0., 0.])
        env.history['drone_velocities'].zero_()
        env.numerical_valid[:] = True
        env.failed[:] = False
        env.active[:] = False
        env.decision = env.decisions
        env.lengths[:] = env.decisions
        return env

    def test_separate_preset_keeps_physics_control_and_ppo(self):
        # Compare this historical discovery preset with its original baseline;
        # the launcher default now selects the newer excitation experiment.
        old, snapshot, cfg, _ = load('configs/learning/whip_stage1_ppo_acceleration_v1.json')
        self.assertEqual(DEFAULT, CURRENT_PRESETS[1])
        self.assertEqual(cfg.matching.reward_mode, 'directed_compact_whip_v1')
        for key in old.keys() | self.settings.keys():
            if key not in ('task', 'workspace_penalty'):
                self.assertEqual(old[key], self.settings[key], key)
        for key in ('model', 'scene_text'):
            self.assertEqual(snapshot[key], self.snapshot[key])
        for key in ('episode', 'workspace', 'return', 'request', 'strike_surface', 'mechanics'):
            self.assertEqual(snapshot['task'][key], self.snapshot['task'][key])
        self.assertEqual(self.cfg.mechanics.event_end_s, 4.35)

    def test_exact_half_proxy_and_target_independence(self):
        raw = self.scorer(replace(self.cfg, matching=replace(self.cfg.matching,
            reward_mode='directed_whip_v1')))(self.positions, self.velocities, self.observed)
        result = self.scorer()(self.positions, self.velocities, self.observed)
        self.assertGreater(float(result.score[0]), 0)
        torch.testing.assert_close(result.score, .5*raw.score, rtol=0, atol=0)
        torch.testing.assert_close(result.causal_best, .5*raw.causal_best, rtol=0, atol=0)
        torch.testing.assert_close(result.components['whip_proxy'], raw.score)
        torch.testing.assert_close(result.components['mechanics_credit'], result.score)
        changed = replace(self.cfg, strike_surface=replace(self.cfg.strike_surface,
                          distance_m=2., width_m=.1, height_m=.1))
        other = self.scorer(changed)(self.positions, self.velocities, self.observed)
        torch.testing.assert_close(result.score, other.score, rtol=0, atol=0)
        self.assertEqual(float(result.components['approach_reward'][0]+result.components['strike_bonus'][0]), 0)

    def test_no_event_has_zero_proxy_credit(self):
        p = torch.zeros_like(self.positions)
        p[..., 2] = 2-torch.cat((self.rest.new_zeros(1), self.rest.cumsum(0)))
        result = self.scorer()(p, torch.zeros_like(p), self.observed)
        self.assertFalse(bool(result.has_event[0]))
        for value in (result.score, result.components['whip_proxy'], result.components['mechanics_credit']):
            self.assertEqual(float(value[0]), 0.)

    def test_capped_density_full_horizon_and_nan_suffix(self):
        penalty = WorkspacePenalty(**self.settings['workspace_penalty'])
        origin = torch.tensor([[0., 0., 2.]], dtype=torch.float64)
        p = origin[:, None].expand(1, 451, 3).clone()
        self.assertEqual(float(penalty.charges(p, origin, self.cfg.workspace, self.observed).sum()), 0)
        p[:, 1:, 0] = .6  # 5 cm beyond radius: density 0.25.
        self.assertAlmostEqual(float(penalty.charges(p, origin, self.cfg.workspace, self.observed).sum()), .0625)
        p[:, 1:, 0] = 100
        self.assertAlmostEqual(float(penalty.charges(p, origin, self.cfg.workspace, self.observed).sum()), .25)
        observed = self.observed.clone(); observed[:, 201:] = False
        p[:, 201:] = float('nan')
        charges = penalty.charges(p, origin, self.cfg.workspace, observed)
        self.assertAlmostEqual(float(charges.sum()), .25*200/450)
        self.assertEqual(int(charges[:, 200:].count_nonzero()), 0)
        self.assertGreater(float(WorkspacePenalty(.25, .1).charges(p, origin, self.cfg.workspace, observed).sum()), 1.)

    def test_failed_prefix_claws_back_credit_and_cost(self):
        env = self.environment()
        returns, valid, score, _ = env.finish()
        self.assertTrue(bool(valid[0]))
        self.assertAlmostEqual(float(returns[0]), float(score.score[0])-.25, places=6)
        torch.testing.assert_close(env.transition_rewards.sum(1), returns)
        self.assertGreater(float(score.causal_best[0, 300]), 0)
        before = env.transition_rewards.clone()
        env.numerical_valid[:, 301:] = False
        env.failed[:] = True
        env.lengths[:] = 150
        returns, valid, score, _ = env.finish()
        self.assertFalse(bool(valid[0]))
        self.assertEqual(float(returns[0]), -1.)
        self.assertEqual(float(score.components['workspace_cost'][0]), 0.)
        self.assertAlmostEqual(float(score.components['workspace_accrued_cost'][0]), .25*300/450)
        torch.testing.assert_close(env.transition_rewards[:, :149], before[:, :149], rtol=0, atol=0)
        torch.testing.assert_close(env.transition_rewards.sum(1), returns)
        self.assertEqual(int(env.transition_rewards[:, 150:].count_nonzero()), 0)

    def test_collection_retains_raw_proxy_effective_cost_and_physical_excess(self):
        env = self.environment()
        env.numerical_valid[:, 301:] = False
        env.failed[:] = True
        env.lengths[:] = 150
        returns, valid, score, _ = env.finish()
        with project_temporary() as directory:
            record_simple_collection(directory, env, returns, valid, score, 1)
            with np.load(directory+'/collection-00001.npz') as saved:
                self.assertEqual(saved['hit_definition'].item(), 'diagnostic_only_full_horizon_crossings')
                self.assertIn('component_whip_proxy', saved.files)
                self.assertEqual(saved['component_workspace_cost'][0], 0.)
                self.assertAlmostEqual(saved['component_workspace_accrued_cost'][0], .25*300/450)
                self.assertAlmostEqual(saved['workspace_workspace_mean_square_excess_m2'][0], .25**2)
                # Serialization must preserve the float32 sum exactly; the
                # environment checks the pre-cast return identity at 1e-12.
                np.testing.assert_array_equal(saved['stored_reward_sum'], env.transition_rewards.sum(1).numpy())
                np.testing.assert_array_equal(saved['returns'], returns.numpy())

    def test_configuration_rejects_unapproved_combinations(self):
        for key, value in [('policy_rate_hz', 25), ('discount', .99), ('training_stage', 2),
                           ('workspace_penalty', dict(weight=.1, distance_scale_m=.1))]:
            with self.subTest(key=key), self.assertRaises(ValueError):
                load(dict(self.settings, **{key: value}))
        with self.assertRaises(ValueError):
            load(dict(load(DEFAULT)[0], workspace_penalty=self.settings['workspace_penalty']))

    def test_transfer_requires_future_handoff_design(self):
        snapshot = deepcopy(self.snapshot)
        snapshot['execution_hashes'] = {}
        snapshot['identity'] = identity(snapshot)
        saved = dict(schema='ppo_stage1_checkpoint_v1', snapshot=snapshot,
                     identity=snapshot['identity'], agent={'actor': {'synthetic': torch.ones(1)}})
        with patch('training.artifacts.execution_hashes', return_value={}):
            validate_actor_source(saved, self.snapshot)
            with self.assertRaisesRegex(ValueError, 'motion review'):
                validate_actor_source(saved, load(CURRENT_PRESETS[2])[1])

    def test_ui_labels_explicit_preset_without_starting(self):
        os.environ.setdefault('QT_QPA_PLATFORM', 'offscreen')
        from PySide6 import QtWidgets
        from training.ui.window import TrainingWindow
        app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
        window = TrainingWindow(settings_path=PRESET)
        try:
            self.assertIn('Stage1 discovery', window.description_details)
            self.assertIn('Hits are diagnostics', window.description_details)
        finally:
            window.close(); app.processEvents()

    @unittest.skipUnless(torch.cuda.is_available(), 'CUDA unavailable')
    def test_cuda_scoring_matches_cpu(self):
        cpu = self.scorer()(self.positions, self.velocities, self.observed)
        gpu = self.scorer(device='cuda')(self.positions.cuda(), self.velocities.cuda(), self.observed.cuda())
        torch.testing.assert_close(cpu.score, gpu.score.cpu(), rtol=1e-8, atol=1e-10)
        torch.testing.assert_close(cpu.causal_best, gpu.causal_best.cpu(), rtol=1e-8, atol=1e-10)


if __name__ == '__main__':
    unittest.main()
