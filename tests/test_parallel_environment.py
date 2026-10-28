"""Lanes split over worker processes equal the same lanes in one process, bit for bit."""
import numpy as np
import torch

from aerowhip.envs.batch_environment import BatchEnvironment
from aerowhip.envs.parallel_environment import ParallelEnvironment
from helpers import ENVIRONMENT, SCENE, SNAPSHOT, TASK


def run(env, actions):
    done = False
    while not done:
        _, _, done = env.step(actions[:, env.decision])
    return env.finish()


def test_worker_processes_reproduce_one_process_exactly():
    lanes, options = 5, dict(randomised=True, sensing_seed=3, randomisation_seed=4)
    single = BatchEnvironment(ENVIRONMENT, TASK, SNAPSHOT['model'], SCENE, device='cpu', lanes=lanes, **options)
    parallel = ParallelEnvironment(ENVIRONMENT, TASK, SNAPSHOT['model'], SCENE, device='cpu', lanes=lanes, workers=3, **options)
    try:
        assert parallel.slices == [(0, 2), (2, 2), (4, 1)]
        generator = torch.Generator().manual_seed(2)
        for cohort in range(2):                        # the second cohort draws new models and starts
            actions = (torch.rand(lanes, single.decisions, 3, generator=generator)*2-1)*.6
            if cohort:
                single.reset(); parallel.reset()
            first, second = run(single, actions), run(parallel, actions)
            for a, b in zip(first[:2], second[:2]):
                assert torch.equal(a.cpu(), b.cpu())
            for key, value in first[2].items():         # NaN where undefined (no hit) in both
                assert torch.equal(value.cpu().nan_to_num(-1e300), second[2][key].cpu().nan_to_num(-1e300)), key
            assert torch.equal(single.frames, parallel.frames.cpu())
            assert torch.equal(single.critic_states, parallel.critic_states.cpu())
            assert torch.equal(single.transition_rewards, parallel.transition_rewards.cpu())
            assert torch.equal(single.lengths, parallel.lengths.cpu())
            for key, value in single.history.items():
                assert torch.equal(value.nan_to_num(-1e300), parallel.history[key].nan_to_num(-1e300)), key
            import json
            assert json.dumps(single.episode_records()) == json.dumps(parallel.episode_records())
            replay = parallel.replay_arrays(3)
            assert np.array_equal(replay['drone_positions'], single.replay_arrays(3)['drone_positions'])
    finally:
        parallel.close()


def test_cohort_ends_together_when_slices_end_at_different_times():
    # Lanes 0-1 strike-free and hovering; lanes 2-3 pushed into an envelope violation early, so one
    # worker's lanes all end long before the cohort does.
    lanes, options = 4, dict(randomised=False, reference_states=False)
    single = BatchEnvironment(ENVIRONMENT, TASK, SNAPSHOT['model'], SCENE, device='cpu', lanes=lanes, **options)
    parallel = ParallelEnvironment(ENVIRONMENT, TASK, SNAPSHOT['model'], SCENE, device='cpu', lanes=lanes, workers=2, **options)
    try:
        actions = torch.zeros(lanes, single.decisions, 3)
        actions[2:, :, 2] = -1.                         # full downward command: the cable reaches the floor
        actions[2:, :, 0] = 1.
        first, second = run(single, actions), run(parallel, actions)
        # the scenario: the second worker's lanes ended early, the cohort ran on
        assert set(single.failure_reasons[2:]) == {'envelope_violation'} and single.failure_reasons[:2] == ['', '']
        assert max(single.terminal_times[2:]) < min(single.terminal_times[:2])
        assert torch.equal(first[0], second[0].cpu()) and torch.equal(first[1], second[1].cpu())
        assert single.index == parallel.index and single.decision == parallel.decision
        assert torch.equal(single.frames, parallel.frames.cpu())
        for key, value in single.history.items():
            assert torch.equal(value.nan_to_num(-1e300), parallel.history[key].nan_to_num(-1e300)), key
        assert torch.equal(single.lengths, parallel.lengths.cpu())
    finally:
        parallel.close()
