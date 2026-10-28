"""Environment schedule, reward ledger, validity and numerical-failure handling.

Two tests run the real native plant (hover, determinism, lane independence); the validity and
failure tests replace the plant by fixed synthetic observations so outcomes are known exactly.
"""
import numpy as np
import pytest
import torch

from aerowhip.envs.batch_environment import make_batch_environment
from aerowhip.envs.configuration import OBSERVATION_PERIOD_S
from helpers import ENVIRONMENT, EXPERIMENT, SCENE, SNAPSHOT, TASK


def make(lanes=1, snapshot=SNAPSHOT):
    return make_batch_environment(EXPERIMENT, snapshot, SCENE, lanes=lanes, device='cpu', workers=0)   # in-process internals


def without_residual(snapshot):
    """The snapshot with the cable residual removed: the plain physics."""
    model = dict(snapshot['model'], cable={k: v for k, v in snapshot['model']['cable'].items() if k != 'residual'})
    return dict(snapshot, model=model)


def run(env, actions=None):
    done = False
    while not done:
        _, _, done = env.step(torch.zeros(env.batch, 3) if actions is None else actions[:, env.decision])
    return env.finish()


@pytest.fixture
def env():
    return make()


class FakePlant:
    """Stands in for the BatchRuntime: every lane sits at a fixed observation row."""
    def __init__(self, env, row, fail_at=None):
        self.real, self.row, self.fail_at, self.steps = env.runtime, row, fail_at, 0
        self.errors = ['solver diverged']*env.batch

    @property
    def failed(self):
        return [self.fail_at is not None and self.steps >= self.fail_at]*len(self.errors)

    def step(self, command):
        self.steps += 1

    def observations(self):
        return np.tile(self.row, (len(self.errors), 1))

    def motors(self):
        return np.full((len(self.errors), 4), .1)


def fixed_plant(env, shift=(0., 0., 0.)):
    row = env.runtime.observations()[0].copy()
    n = env.nodes
    row[:3] += shift
    row[18:18+3*n] += np.tile(shift, n)
    env.runtime = FakePlant(env, row)
    return env.runtime


def test_schedule_and_real_plant_step(env):
    assert (env.samples, env.decisions, env.stride) == (501, 250, 2)
    _, index, done = env.step(torch.zeros(1, 3))
    assert int(index[0]) == 2 and not done
    assert bool(env.numerical_valid[0, :3].all())


def drift(env):
    returns, valid, _ = run(env)
    positions = env.history['drone_positions']
    offset = (positions-positions[:, :1]).norm(dim=-1)
    return offset, (positions.shape[1]-1)*OBSERVATION_PERIOD_S, bool(valid.all())


def test_zero_action_holds_hover():
    # Plain physics, the cable hanging straight: the firmware commands integer PWM, one count is
    # ~4e-5 of the hover thrust, up to 3e-4 m/s^2 of residual acceleration, i.e. at most a t^2/2 over
    # the episode (no credit for air drag).
    offset, duration, valid = drift(make(2, without_residual(SNAPSHOT)))
    assert float(offset.max()) < .5*3e-4*duration**2 and valid


def test_zero_action_holds_hover_with_the_cable_residual():
    # With the fitted cable residual the cable leans at rest and pulls on the clamp, 37 mm below the
    # centre of mass: the hover trims cancel its net force, and what remains is a slow wobble while the
    # attitude loop settles against that moment (about 5 mm, back by the end of the episode), not a drift.
    if not SNAPSHOT['model']['cable'].get('residual'):
        pytest.skip('the configured model has no cable residual')
    offset, duration, valid = drift(make(2))
    assert float(offset.max()) < .01 and float(offset[:, -1].max()) < .005 and valid


def test_runs_are_reproducible_and_lanes_independent():
    generator = torch.Generator().manual_seed(1)
    first, second, alone = make(3), make(3), make(1)
    actions = (torch.rand(3, first.decisions, 3, generator=generator)*2-1)*.5
    returns = run(first, actions)[0]
    assert torch.equal(returns, run(second, actions)[0])
    assert torch.equal(first.frames, second.frames) and torch.equal(first.critic_states, second.critic_states)
    # Lane 0 alone sees the same plant and the same sensing draws (seed, lane) as lane 0 of three.
    assert torch.equal(run(alone, actions[:1])[0][0], returns[0])
    assert torch.equal(alone.frames[0], first.frames[0])


def test_actor_sees_the_sensed_cable_with_confidences():
    env = make(2)
    n, m = env.nodes, 10
    assert env.frame_size == 18+6*n+m+3+8+3+1 and len(env.actor_columns) == 18+6*n+m+3+4   # +3: task target columns
    frame = env.frames[:, 0]
    assert bool((frame[:, env.confidence_offset:env.confidence_offset+m] == 1.).all())
    assert torch.equal(env.history['sensed_confidence'][:, 0], torch.ones(2, m, dtype=torch.float64))
    sensed = env.history['sensed_cable_positions'][:, 0]
    true = env.history['cable_positions'][:, 0, 1:]
    assert float((sensed-true).norm(dim=-1).max()) < 2e-3        # the filter's labelled markers, noise only
    env.step(torch.zeros(2, 3))
    assert env.replay_arrays(0)['sensing_latency_frames'].shape == ()
    assert 'sensing' in env.provenance()


def test_physical_violation_terminates_with_the_remaining_time_charged(env):
    # Cable pushed below the floor: the lane ends at its first sample with the termination cost.
    fixed_plant(env, shift=(0., 0., -2.3))
    _, _, done = env.step(torch.zeros(1, 3))
    assert done
    returns, valid, components = env.finish()
    assert not bool(valid[0]) and not bool(env.failed[0]) and env.failure_reasons[0] == 'envelope_violation'
    assert components['executed_duration_s'][0] == pytest.approx(.01)
    assert components['failure_cost'][0] == pytest.approx(TASK.failure_cost_per_remaining_s*(ENVIRONMENT.duration_s-.01))
    torch.testing.assert_close(env.transition_rewards.sum(1), returns)
    assert env.episode_records()[0]['task'] == TASK.name and env.episode_records()[0]['terminated'] == 'envelope_violation'
    with pytest.raises(RuntimeError):
        env.step(torch.zeros(1, 3))


def test_physical_violation_runs_the_full_horizon_when_termination_is_off():
    from dataclasses import replace
    from aerowhip.envs.batch_environment import BatchEnvironment
    env = BatchEnvironment(replace(ENVIRONMENT, terminate_on_violation=False), TASK, SNAPSHOT['model'], SCENE, device='cpu', lanes=1)
    fixed_plant(env, shift=(0., 0., -2.3))
    for i in range(env.decisions):
        _, _, done = env.step(torch.zeros(1, 3))
        assert done == (i == env.decisions-1)
    returns, valid, components = env.finish()
    assert not bool(valid[0]) and not bool(env.failed[0])
    assert components['executed_duration_s'][0] == ENVIRONMENT.duration_s and components['failure_cost'][0] == 0.
    torch.testing.assert_close(env.transition_rewards.sum(1), returns)
    assert len(env.replay_arrays()['time_s']) == env.samples


def test_numerical_failure_stops_lane_and_charges_remaining_time(env):
    plant = fixed_plant(env)
    plant.fail_at = 1
    _, _, done = env.step(torch.zeros(1, 3))
    assert done
    returns, valid, components = env.finish()
    assert not bool(valid[0]) and bool(env.failed[0])
    assert float(components['failure_cost'][0]) == pytest.approx(ENVIRONMENT.duration_s-.01)
    assert torch.isfinite(returns).all()
    assert env.failure_records[0]['error'] == 'solver diverged'


def test_rejects_unnormalized_actions(env):
    with pytest.raises(ValueError):
        env.step(torch.full((1, 3), 1.5))
