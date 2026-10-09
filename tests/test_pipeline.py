"""Current fixed-target stage contracts using local presets and synthetic state."""
from dataclasses import replace
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import torch

from training.artifacts import create_run
from training.configuration import FROZEN_PRESETS, iteration_budget, load, require_launchable
from training.curriculum import TargetCurriculum, target_workspace_gate
from tasks.whip.sac_environment import Stage1Environment
from simulation.tensor_model import controlled_values


class PipelineTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)
        cls.presets = {stage: load(path) for stage, path in FROZEN_PRESETS.items()}

    def environment(self, stage=1, scene=None):
        settings, snapshot, cfg, original_scene = self.presets[stage]
        return Stage1Environment(dict(settings, device='cpu', parallel_environments=1),
                                 snapshot, cfg, scene or original_scene, capture=False)

    def test_four_current_roles_budgets_and_shared_physics(self):
        modes = ['directed_whip_v1', 'directed_delivery_v1',
                 'directed_precision_v1', 'directed_precision_v1']
        reference = self.presets[1][1]
        for stage, (settings, snapshot, cfg, _) in self.presets.items():
            with self.subTest(stage=stage):
                require_launchable(settings)
                self.assertEqual(settings['algorithm'], 'ppo')
                self.assertEqual(settings['training_stage'], stage)
                self.assertEqual(settings['parallel_environments'], 128)
                self.assertEqual(settings['training_episodes'], 10240)
                self.assertEqual(iteration_budget(settings), 80)
                self.assertEqual(settings['seed'], 0 if stage == 1 else 1)
                self.assertEqual(cfg.matching.reward_mode, modes[stage-1])
                self.assertEqual(cfg.matching.alignment_mode, 'cone_plateau_v1')
                self.assertEqual(cfg.matching.alignment_cone_deg, 25)
                self.assertEqual(cfg.termination, 'full_horizon_v1')
                self.assertEqual(snapshot['model'], reference['model'])
                self.assertEqual(snapshot['scene_text'], reference['scene_text'])
                self.assertEqual(settings['velocity_limit_mps'], [2, 2, .8])
                self.assertNotIn('target_sampler', settings)
                self.assertNotIn('stage3_actor_training', settings)
        self.assertNotIn('target_curriculum', self.presets[2][0])
        self.assertEqual(self.presets[3][0]['stage3_role'], 'precision_v1')
        self.assertEqual(self.presets[4][0]['stage4_role'], 'learned_recovery_v1')

    def test_all_stage_goal_columns_and_fixed_nominal_target(self):
        for stage in self.presets:
            with self.subTest(stage=stage):
                env = self.environment(stage)
                columns = [98, 99, 100] if stage == 4 else [96, 97, 98]
                self.assertEqual(env.policy_encoder_options['goal_columns'], columns)
                self.assertEqual(env.frame_size, 101 if stage == 4 else 99)
                self.assertTrue(set(env.actor_columns).isdisjoint(columns))
                torch.testing.assert_close(env.scorer.center,
                    torch.tensor([1.1, 0., 2.], dtype=torch.float64), rtol=0, atol=0)
                expected = ((env.scorer.center-env.origin)/env.model.rest.sum()).float()
                torch.testing.assert_close(env.frames[:, 0, columns], expected[None], rtol=0, atol=0)
                self.assertEqual(env.model.dt, .01)
                self.assertEqual(env.stride, 10)

    def test_retired_training_tags_reject_before_run_creation(self):
        original = self.presets[3][0]
        for key, value in [('target_sampler', 'lateral_vertical_grid_v1'),
                           ('stage3_actor_training', 'combined_goal_head_v1')]:
            with self.subTest(key=key), tempfile.TemporaryDirectory() as directory:
                root = Path(directory)/'new-runs'
                with self.assertRaisesRegex(ValueError, 'retired'):
                    create_run(dict(original, **{key: value}), root=root)
                self.assertFalse(root.exists())

    def test_target_size_progression_keeps_center_and_final_precision(self):
        settings, _, cfg, _ = self.presets[3]
        curriculum = TargetCurriculum()
        centers = []
        for level, dimensions in enumerate(zip([1., .6, .3, .1], [.6, .4, .2, .1])):
            self.assertEqual(curriculum.level, level)
            current = curriculum.configuration(cfg, settings)
            self.assertEqual((current.strike_surface.width_m,
                              current.strike_surface.height_m), dimensions)
            centers.append(current.strike_surface.distance_m)
            self.assertEqual(current.request, cfg.request)
            self.assertEqual(current.request[3], 3.)
            self.assertEqual(current.strike_surface.maximum_entry_angle_deg,
                             cfg.strike_surface.maximum_entry_angle_deg)
            for _ in range(3):
                curriculum.observe(settings, .8, 1000., valid_fraction=1.)
            if level < 3:
                curriculum.advance(settings, valid=True, qualified=True, workspace_cost=1000.)
            else:
                self.assertFalse(curriculum.ready(settings, valid=True,
                                                 qualified=True, workspace_cost=1000.))
        self.assertEqual(centers, [cfg.strike_surface.distance_m]*4)
        self.assertEqual(cfg.matching.center_sigma_m, .05)

    def test_curriculum_requires_all_valid_cohorts_and_evaluation_gate(self):
        settings = self.presets[3][0]
        curriculum = TargetCurriculum()
        for _ in range(2):
            curriculum.observe(settings, .8, 0.)
        self.assertFalse(curriculum.ready(settings, valid=True, qualified=True, workspace_cost=0.))
        curriculum.observe(settings, .8, 0., valid_fraction=127/128)
        self.assertEqual(curriculum.streak, 0)
        for _ in range(3):
            curriculum.observe(settings, .8, 0.)
        for valid, qualified in [(False, True), (True, False)]:
            with self.subTest(valid=valid, qualified=qualified), self.assertRaises(ValueError):
                curriculum.advance(settings, valid=valid, qualified=qualified, workspace_cost=0.)
        self.assertEqual(curriculum.level, 0)
        self.assertTrue(target_workspace_gate(settings, 1000.)['workspace_gate_pass'])
        self.assertFalse(target_workspace_gate(settings, float('nan'))['workspace_gate_pass'])
        # Workspace cost stays a soft penalty and a diagnostic in current Stage3.
        self.assertIsNone(target_workspace_gate(settings, 1000.)['workspace_gate_limit'])

    def test_goal_uses_initial_origin_and_stays_constant_after_motion(self):
        for stage in self.presets:
            with self.subTest(stage=stage):
                scene = self.presets[stage][3]
                moved_start = replace(scene, initial_state=replace(scene.initial_state,
                    position_world_m=np.array([.2, -.3, 2.1])))
                env = self.environment(stage, moved_start)
                expected = ((env.scorer.center-env.origin)/env.model.rest.sum()).float()
                torch.testing.assert_close(env.goal_relative, expected, rtol=0, atol=0)
                self.assertAlmostEqual(float(env.goal_relative[2]), -.1/.9525, places=7)
                p, v, q, cv = env._observe(env.state)
                env.index = 1
                env._record(p+.02, v, q, cv, env.hold[None], env.hold[None],
                            env.active, ~env.active)
                torch.testing.assert_close(env.frames[:, 1, -3:], expected[None], rtol=0, atol=0)
                self.assertFalse(torch.equal(env.frames[:, 1, :3], env.frames[:, 0, :3]))

    def test_stage4_recovery_phase_does_not_overwrite_goal(self):
        env = self.environment(4)
        goal = env.frames[:, 0, -3:].clone()
        env.index = round(env.configuration.recovery.whip_duration_s/.01)
        p, v, q, cv = env._observe(env.state)
        env._record(p, v, q, cv, env.hold[None], env.hold[None], env.active, ~env.active)
        env._advance_recovery_clock()
        self.assertEqual(env.policy_encoder_options['phase_columns'], [96, 97])
        self.assertEqual(float(env.frames[0, env.index, 96]), 1.)
        self.assertEqual(float(env.frames[0, env.index, 97]), 0.)
        torch.testing.assert_close(env.frames[:, env.index, -3:], goal, rtol=0, atol=0)

    def test_physical_failure_and_nonfinite_endpoint_keep_invalid_suffix(self):
        for kind in ['unsafe_rotation', 'nonfinite_endpoint']:
            with self.subTest(kind=kind):
                env = self.environment()
                state = env.state
                if kind == 'unsafe_rotation':
                    rotation = state.rotation.clone()
                    rotation[0] = torch.tensor([[1., 0., 0.], [0., 0., -1.], [0., 1., 0.]],
                                                dtype=torch.float64)
                    endpoint = replace(state, rotation=rotation)
                else:
                    position = state.position_com.clone()
                    position.fill_(float('nan'))
                    endpoint = replace(state, position_com=position)
                with patch.object(env, '_advance', return_value=endpoint):
                    _, _, done = env.step(torch.zeros(1, 3))
                self.assertTrue(done)
                self.assertTrue(bool(env.failed[0]))
                self.assertFalse(bool(env.numerical_valid[0, 2:].any()))
                returns, valid, _, _ = env.finish()
                self.assertFalse(bool(valid[0]))
                self.assertEqual(float(returns[0]), env.configuration.returns.failure_return)
                if kind == 'nonfinite_endpoint':
                    for before, after in zip(controlled_values(state), controlled_values(env.state)):
                        torch.testing.assert_close(before, after, rtol=0, atol=0)


if __name__ == '__main__':
    unittest.main()
