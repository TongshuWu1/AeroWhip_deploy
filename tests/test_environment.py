"""Environment schedule, reward ledger, validity and numerical-failure handling.

The plant step is replaced by fixed synthetic states so outcomes are known exactly;
the first test runs the real plant for one decision.
"""
from dataclasses import replace

import pytest
import torch

from aerowhip.envs.environment import make_environment
from helpers import EXPERIMENT, SCENE, SNAPSHOT, TASK


@pytest.fixture
def env():
    return make_environment(EXPERIMENT, SNAPSHOT, SCENE, lanes=1, device='cpu', capture=False)


def test_schedule_and_real_plant_step(env):
    assert (env.samples, env.decisions, env.stride) == (401, 200, 2)
    _, index, done = env.step(torch.zeros(1, 3))
    assert int(index[0]) == 2 and not done
    assert bool(env.numerical_valid[0, :3].all())


def replace_plant(env, state, elapsed):
    def fixed(current, command, controller, **kw):
        kw['observer'](state, elapsed)
        return state
    env.model.step_unchecked = fixed


def test_physical_violation_is_recorded_but_full_horizon_runs(env):
    # Cable pushed below the floor: invalid episode, but finite, so it continues to 4 s.
    below = replace(env.state, position_com=env.state.position_com+env.model.tensor([.8, 0, -2.3]))
    replace_plant(env, below, .01)
    for i in range(env.decisions):
        _, _, done = env.step(torch.zeros(1, 3))
        assert done == (i == env.decisions-1)
    returns, valid, components = env.finish()
    assert not bool(valid[0]) and not bool(env.failed[0])
    assert components['executed_duration_s'][0] == 4. and components['failure_cost'][0] == 0.
    torch.testing.assert_close(env.transition_rewards.sum(1), returns)
    assert env.episode_records()[0]['task'] == TASK.name
    assert len(env.replay_arrays()['time_s']) == 401
    with pytest.raises(RuntimeError):
        env.step(torch.zeros(1, 3))


def test_numerical_failure_stops_lane_and_charges_remaining_time(env):
    corrupt = replace(env.state, position_com=torch.full_like(env.state.position_com, float('nan')))
    replace_plant(env, corrupt, 1/600)
    _, _, done = env.step(torch.zeros(1, 3))
    assert done
    returns, valid, components = env.finish()
    assert not bool(valid[0]) and bool(env.failed[0])
    assert float(components['failure_cost'][0]) == pytest.approx(4-1/600)
    assert torch.isfinite(returns).all()


def test_rejects_unnormalized_actions(env):
    with pytest.raises(ValueError):
        env.step(torch.full((1, 3), 1.5))
