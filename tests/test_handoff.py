"""Synthetic actor transfer and checkpoint guards for the current four stages."""
from copy import deepcopy
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import torch

from helpers import exact_state
from training.artifacts import (create_run, identity, read_json,
                                save_checkpoint, validate_actor_source)
from training.configuration import FROZEN_PRESETS, load
from training.ppo import PPO
from tasks.whip.sac_environment import Stage1Environment


class HandoffTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)
        cls.presets = {stage: load(path) for stage, path in FROZEN_PRESETS.items()}
        cls.interfaces = {}
        for stage, (settings, snapshot, cfg, scene) in cls.presets.items():
            env = Stage1Environment(dict(settings, device='cpu', parallel_environments=1),
                                     snapshot, cfg, scene, capture=False)
            cls.interfaces[stage] = (env.actor_columns, env.frame_size,
                                     env.critic_state_size, env.policy_encoder_options)
        cls.hashes = {'simulation/model.py': 'synthetic-physical-source',
                      'training/ppo.py': 'synthetic-learner-source'}

    def agent(self, stage=1):
        columns, frames, privileged, options = self.interfaces[stage]
        return PPO(dict(self.presets[stage][0], device='cpu', batch_size=8),
                   columns, frames, 'cpu', privileged_size=privileged, encoder_options=options)

    def checkpoint(self, stage=2, mutate=None):
        snapshot = deepcopy(self.presets[stage][1])
        snapshot['execution_hashes'] = deepcopy(self.hashes)
        saved = dict(schema='ppo_stage1_checkpoint_v1', snapshot=snapshot,
                     agent=self.agent(stage).state_dict(), round=1, episodes=128, steps=5760)
        if mutate:
            mutate(saved)
        saved['snapshot']['identity'] = identity(saved['snapshot'])
        saved['identity'] = saved['snapshot']['identity']
        return saved

    def target_snapshot(self, stage=3):
        return self.presets[stage][1]

    def test_all_stage_actor_handoffs_preserve_weights_and_fresh_learner_state(self):
        previous = self.agent()
        with torch.no_grad():
            previous.actor.encoder.goal_projection.weight.fill_(.02)
        for stage in (2, 3, 4):
            with self.subTest(stage=stage):
                # Give the synthetic source nonempty Adam state and a changed critic,
                # so accidentally restoring either is observable at each handoff.
                for optimizer in (previous.actor_optimizer, previous.value_optimizer):
                    for group in optimizer.param_groups:
                        for parameter in group['params']:
                            parameter.grad = torch.full_like(parameter, .01)
                    optimizer.step()
                self.assertTrue(previous.actor_optimizer.state)
                self.assertTrue(previous.value_optimizer.state)
                previous.collections = previous.updates = previous.value_updates = 1
                previous.environment_steps = 5760
                previous.set_completed_episodes(128)
                current = self.agent(stage)
                value, rng = deepcopy(current.value.state_dict()), torch.get_rng_state().clone()
                current.initialize_actor(previous.state_dict())
                self.assertTrue(exact_state(value, current.value.state_dict()))
                torch.testing.assert_close(torch.get_rng_state(), rng, rtol=0, atol=0)
                for name, value in previous.actor.state_dict().items():
                    if name != 'encoder.goal_columns':
                        torch.testing.assert_close(value, current.actor.state_dict()[name], rtol=0, atol=0)
                self.assertEqual((current.collections, current.updates, current.value_updates,
                                  current.environment_steps, current.completed_episodes,
                                  current.entropy_episode_offset, current.entropy_step_offset), (0,)*7)
                self.assertFalse(current.actor_optimizer.state)
                self.assertFalse(current.value_optimizer.state)
                clone = self.agent(stage)
                clone.load_state_dict(deepcopy(current.state_dict()))
                self.assertTrue(exact_state(current.state_dict(), clone.state_dict()))
                previous = current

    def test_stage4_preserves_learned_goal_and_trains_only_existing_recovery(self):
        source, target = self.agent(3), self.agent(4)
        with torch.no_grad():
            source.actor.encoder.goal_projection.weight.fill_(.02)
        target.initialize_actor(source.state_dict())
        self.assertEqual(target.actor.encoder.goal_columns.tolist(), [98, 99, 100])
        self.assertFalse(target.actor.encoder.goal_projection.weight.requires_grad)
        trainable = {name for name, value in target.actor.named_parameters() if value.requires_grad}
        self.assertEqual(trainable, {'recovery_log_std', 'recovery_head.0.weight',
            'recovery_head.0.bias', 'recovery_head.2.weight', 'recovery_head.2.bias'})
        self.assertEqual({id(p) for group in target.actor_optimizer.param_groups for p in group['params']},
                         {id(p) for p in target.actor.parameters() if p.requires_grad})
        history = torch.randn(2, 451, 99)*.05
        phase = history.new_zeros((2, 451, 2))
        full = torch.cat((history[:, :, :96], phase, history[:, :, 96:]), -1)
        full = torch.cat((full, full.new_zeros((2, 400, 101))), 1)
        index = torch.tensor([0, 450])
        for before, after in zip(source.actor.distribution(history, index),
                                 target.actor.distribution(full, index)):
            torch.testing.assert_close(before, after, rtol=0, atol=0)
        frozen = deepcopy(target.actor.state_dict())
        full[:, 600, 96:98] = torch.tensor([1., .1])
        full[:, 600, :3] = torch.tensor([.2, -.1, .3])
        target.actor.distribution(full, torch.full((2,), 600))[0].square().sum().backward()
        target.actor_optimizer.step()
        self.assertTrue(any(not torch.equal(frozen[name], target.actor.state_dict()[name])
                            for name in trainable if name.startswith('recovery_head')))
        for name, value in frozen.items():
            if name not in trainable:
                torch.testing.assert_close(value, target.actor.state_dict()[name], rtol=0, atol=0)

    def test_current_goal_actor_causality_and_full_training_mask(self):
        for stage in (1, 2, 3, 4):
            with self.subTest(stage=stage):
                agent = self.agent(stage)
                if stage < 4:
                    self.assertTrue(all(p.requires_grad for p in agent.actor.parameters()))
                with torch.no_grad():
                    agent.actor.encoder.goal_projection.weight.fill_(.02)
                frames = torch.randn(2, 851 if stage == 4 else 451,
                                     agent.interface['frame_size'])*.05
                if stage == 4:
                    frames[:, :, 96:98] = 0
                index = torch.tensor([10, 200])
                expected = agent.actor.distribution(frames, index)
                future = frames.clone()
                for lane, sample in enumerate(index):
                    future[lane, sample+1:] = float('nan')
                for original, changed in zip(expected, agent.actor.distribution(future, index)):
                    torch.testing.assert_close(original, changed, rtol=0, atol=0)
                prepared = agent.actor.encoder.prepare(frames, index)
                for original, cached in zip(expected, agent.actor.distribution(prepared, None)):
                    torch.testing.assert_close(original, cached, rtol=0, atol=0)

    def test_source_identity_schema_and_finite_actor_guards(self):
        mutations = [lambda saved: saved.update(schema='wrong-schema'),
                     lambda saved: next(value for value in saved['agent']['actor'].values()
                                        if value.is_floating_point()).fill_(float('nan'))]
        with patch('training.artifacts.execution_hashes', return_value=self.hashes):
            for mutate in mutations:
                with self.subTest(mutate=mutate), self.assertRaises(ValueError):
                    validate_actor_source(self.checkpoint(mutate=mutate), self.target_snapshot())
            saved = self.checkpoint()
            saved['identity'] = 'wrong-identity'
            with self.assertRaisesRegex(ValueError, 'identity-consistent'):
                validate_actor_source(saved, self.target_snapshot())
            bad = deepcopy(self.agent(3).state_dict())
            bad['actor']['encoder.goal_columns'][0] = 0
            with self.assertRaisesRegex(ValueError, 'goal columns'):
                self.agent(3).load_state_dict(bad)

    def test_physical_scene_control_and_source_hash_guards(self):
        mutations = [lambda saved: saved['snapshot']['model'].update(synthetic_mismatch=True),
                     lambda saved: saved['snapshot'].update(scene_text='incompatible'),
                     lambda saved: saved['snapshot']['settings'].update(policy_rate_hz=20),
                     lambda saved: saved['snapshot']['execution_hashes'].update(
                         {'simulation/model.py': 'changed-physical-source'})]
        with patch('training.artifacts.execution_hashes', return_value=self.hashes):
            validate_actor_source(self.checkpoint(), self.target_snapshot())
            for mutate in mutations:
                with self.subTest(mutate=mutate), self.assertRaises(ValueError):
                    validate_actor_source(self.checkpoint(mutate=mutate), self.target_snapshot())

    def test_fresh_run_stage_guards_and_recorded_actor_only_provenance(self):
        with patch('training.artifacts.execution_hashes', return_value=self.hashes):
            for source_stage, target_stage in [(1, 3), (2, 4)]:
                with self.subTest(source=source_stage, target=target_stage), tempfile.TemporaryDirectory() as folder:
                    source = Path(folder)/'source.pt'
                    save_checkpoint(source, self.checkpoint(source_stage))
                    root = Path(folder)/'new-runs'
                    with self.assertRaises(ValueError):
                        create_run(self.presets[target_stage][0], root=root, initialize_actor=source)
                    self.assertFalse(root.exists())
            with tempfile.TemporaryDirectory() as folder:
                source = Path(folder)/'source.pt'
                saved = self.checkpoint()
                save_checkpoint(source, saved)
                run = create_run(self.presets[3][0], root=Path(folder)/'new-runs', initialize_actor=source)
                self.assertEqual(read_json(run/'request.json'),
                                 {'resume': None, 'initialize_actor': 'initial-actor.pt'})
                self.assertEqual((run/'initial-actor.pt').read_bytes(), source.read_bytes())
                provenance = read_json(run/'configuration.json')['policy_initialization']
                self.assertEqual(provenance['source_identity'], saved['identity'])
                self.assertEqual(provenance['transfer'], 'actor_only_v1')
                self.assertEqual(provenance['value_and_optimizers'], 'fresh')
                with self.assertRaisesRegex(ValueError, 'resume OR actor-only'):
                    create_run(self.presets[3][0], source, initialize_actor=source,
                               root=Path(folder)/'invalid')


if __name__ == '__main__':
    unittest.main()
