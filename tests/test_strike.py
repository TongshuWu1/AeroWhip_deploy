"""Strike task accounting (synthetic cable motions, no physics) and the curriculum controller."""
import json
import math

import numpy as np
import pytest

from aerowhip.envs.configuration import EnvironmentConfig
from aerowhip.envs.workspace import CylinderWorkspace
from aerowhip.learning.worker import Curriculum
from aerowhip.paths import ROOT
from aerowhip.tasks import task_config
from aerowhip.tasks.strike.strike import StrikeEpisode, StrikeTask, segment_distance

EXPERIMENT = json.loads((ROOT/'configs/experiments/strike_x_v1.json').read_text())
NODES = 11
L = .985
ORIGIN = np.array([0., 0., 2.])


@pytest.fixture
def config():
    return task_config(EXPERIMENT['task'])


@pytest.fixture
def environment():
    return EnvironmentConfig.from_dict(EXPERIMENT['environment'])


def hanging(drone):
    """Straight cable hanging below a drone at `drone` (attachment 55 mm below the tracking origin)."""
    root = drone+np.array([0., 0., -.055])
    p = np.array([root-np.array([0., 0., L*k/10]) for k in range(NODES)])
    return p, np.zeros_like(p)


def horizontal(root_x, config):
    """Straight cable along +X at the target's height, its root at x = root_x."""
    root = ORIGIN+np.array([root_x, 0., -config.target_below_origin_m])
    p = np.array([root+np.array([L*k/10, 0., 0.]) for k in range(NODES)])
    return p, np.zeros_like(p)


def episode(config, environment, stage=None, workspace=None):
    stage = stage or config.curriculum.final
    workspace = workspace or CylinderWorkspace(stage.workspace_radius_m, 1.5, 2.6)
    e = StrikeEpisode(config, environment, workspace, stage, length=L, workspace_origin=ORIGIN)
    p, v = hanging(ORIGIN)
    e.initialize(p, v)
    return e


def test_target_sits_along_x_below_the_origin(config, environment):
    e = episode(config, environment)
    assert np.allclose(e.target, ORIGIN+[config.curriculum.final.target_distance_m, 0., -config.target_below_origin_m])


def test_segment_distance_catches_a_fast_pass_between_samples():
    target = np.array([1., 0., 0.])
    assert segment_distance(np.array([0., .05, 0.]), np.array([2., .05, 0.]), target) == pytest.approx(.05)
    assert segment_distance(np.array([0., 0., 0.]), np.array([.5, 0., 0.]), target) == pytest.approx(.5)


def test_progress_shaping_telescopes_and_strike_pays_once(config, environment):
    e = episode(config, environment)
    p, v = horizontal(-1.2, config)
    e.previous_tip, e.previous_distance = p[-1].copy(), float(np.linalg.norm(e.target-p[-1]))
    e.trace[0][0] = e.previous_distance
    rewards = []
    # move the whole cable rigidly toward the target at 5 m/s for 0.3 s: the tip closes 1.5 m
    step = np.array([.05, 0., 0.])
    for k in range(1, 31):
        p = p+step
        v = np.full_like(v, 5.)*np.array([1., 0., 0.])
        rewards.append(e.observe(k*.01, p, v, drone_position=p[0]+[0., 0., .055], action=np.zeros(3)))
    assert e.struck and sum(r > 5 for r in rewards) == 1   # one strike bonus
    potential = config.progress_weight*(e.trace[0][0]-e.last['tip_target_distance_m'])/L
    hit = [row for row in e.log if row['kind'] == 'strike'][0]
    bonus = config.strike_bonus+config.speed_bonus_per_mps*(hit['tip_speed_mps']-e.v_star)
    assert hit['tip_speed_mps'] == pytest.approx(5.)
    # accrued = potential progress up to the hit + bounded approach credit + the bonus; nothing after the hit
    assert e.accrued == pytest.approx(sum(rewards)+e.cost+e.action_cost)
    assert e.accrued > bonus and e.accrued-bonus <= potential+config.approach_weight*.3+1e-9


def test_slow_pass_is_not_a_strike(config, environment):
    e = episode(config, environment)
    p, v = horizontal(-1.2, config)
    for k in range(1, 61):
        p = p+[.025, 0., 0.]
        v = np.full_like(v, 2.5)*np.array([1., 0., 0.])
        e.observe(k*.01, p, v, drone_position=p[0]+[0., 0., .055])
    assert not e.struck and e.closest < config.hit_radius_m and e.terminal is None


def test_recovery_bonus_and_episode_end(config, environment):
    e = episode(config, environment)
    p, v = horizontal(-1.2, config)
    for k in range(1, 31):
        p = p+[.05, 0., 0.]
        v = np.full_like(v, 5.)*np.array([1., 0., 0.])
        e.observe(k*.01, p, v, drone_position=p[0]+[0., 0., .055])
    assert e.struck
    before = e.accrued
    # return to the origin and stop: the episode ends recovery_s after the hit, with the bonus
    p, v = hanging(ORIGIN)
    k = 30
    while e.terminal is None:
        k += 1
        e.observe(k*.01, p, v, drone_position=ORIGIN)
    assert e.terminal == 'recovered' and e.recovered
    assert e.time == pytest.approx(e.hit_time+config.recovery_s)
    assert e.accrued == pytest.approx(before+config.recovery_bonus)
    assert e.components()['strike'] == 1. and e.components()['recovered'] == 1.


def test_termination_charges_remaining_time_and_return_telescopes(config, environment):
    e = episode(config, environment)
    p, v = hanging(ORIGIN)
    total = 0.
    for k in range(1, 11):
        total += e.observe(k*.01, p, v, drone_position=ORIGIN)
    total += e.fail(.1, 'envelope_violation')
    assert e.terminal == 'envelope_violation'
    assert e.failure_cost == pytest.approx(config.failure_cost_per_remaining_s*(environment.duration_s-.1))
    assert total == pytest.approx(e.total)


def test_workspace_cost_uses_the_stage_radius(config, environment):
    stage = config.curriculum.stages[0]
    wide = episode(config, environment, stage)
    narrow = episode(config, environment, config.curriculum.final)
    p, v = hanging(ORIGIN)
    drone = ORIGIN+[.7, 0., 0.]          # inside the 0.8 m stage radius, outside the final 0.55 m
    for k in range(1, 4):
        wide.observe(k*.01, p, v, drone_position=drone)
        narrow.observe(k*.01, p, v, drone_position=drone)
    assert wide.cost == 0. and narrow.cost == pytest.approx(3*config.workspace_cost_weight*.15**2/500)


def test_whip_event_needs_speed_amplification_and_extension(config, environment):
    e = episode(config, environment)
    p, v = hanging(ORIGIN)
    # the tip alone moves fast while the attachment stays still
    for k in range(1, 6):
        v = np.zeros_like(v); v[-1] = [0., 5., 0.]
        e.observe(k*.01, p, v, drone_position=ORIGIN)
    assert e.whip_event() and e.peak['speed'] == pytest.approx(5.) and e.peak['extension'] > .99
    e2 = episode(config, environment)
    for k in range(1, 6):
        v = np.full_like(v, 5.)*np.array([0., 1., 0.])     # rigid translation: no amplification
        e2.observe(k*.01, p, v, drone_position=ORIGIN)
    assert not e2.whip_event()


def test_task_batch_targets_and_features(config, environment):
    task = StrikeTask(config, environment, np.r_[.085, [.1]*9])
    task.set_stage(config.curriculum.stages[1], CylinderWorkspace(.7, 1.5, 2.6))
    p, v = hanging(ORIGIN)
    task.reset(dict(cable_positions=np.stack([p, p]), cable_velocities=np.stack([v, v]), drone_positions=np.stack([ORIGIN, ORIGIN]),
                    workspace_origin=np.stack([ORIGIN, ORIGIN])))
    assert task.targets().shape == (2, 3) and task.targets()[0, 0] == pytest.approx(1.)
    assert task.critic_features().shape == (2, task.critic_feature_size)
    assert set(task.components()) >= {'strike', 'whip_event', 'recovered', 'peak_tip_speed_mps'}


def test_curriculum_advances_on_the_window_rate(config):
    c = Curriculum(config.curriculum)
    window = config.curriculum.window_episodes
    assert c.index == 0 and not c.record([1.]*(window-1))
    assert not c.record([0.]*window)                      # window full but below the rate
    assert c.record([1.]*window) and c.index == 1 and c.outcomes == []
    c.record([1.]*window)
    assert c.index == 2 and not c.record([1.]*window)    # final stage stays
    restored = Curriculum(config.curriculum)
    restored.load_state_dict(c.state_dict())
    assert restored.index == 2


def test_hit_needs_speed_at_the_stage_threshold(config, environment):
    easy = episode(config, environment, config.curriculum.stages[0])
    p, v = horizontal(-1.2, config)
    for k in range(1, 51):
        p = p+[.025, 0., 0.]
        v = np.full_like(v, 2.5)*np.array([1., 0., 0.])
        easy.observe(k*.01, p, v, drone_position=p[0]+[0., 0., .055])
    assert easy.struck and math.isfinite(easy.hit_time)
