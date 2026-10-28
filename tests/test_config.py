"""Experiment-file validation: required sections, types, ranges and unimplemented choices."""
from copy import deepcopy

import pytest

from aerowhip.experiment import DEFAULT, Experiment, identity, load
from helpers import SNAPSHOT

RAW = SNAPSHOT['experiment']


def edited(path, value):
    raw = deepcopy(RAW)
    *parents, key = path
    target = raw
    for name in parents:
        target = target[name]
    if value is KeyError:
        del target[key]
    else:
        target[key] = value
    return raw


def test_default_experiment_loads_and_round_trips():
    experiment, snapshot, _ = load(DEFAULT)                # the strike task
    assert Experiment.from_dict(snapshot['experiment']) == experiment
    assert experiment.environment.observations == 500 and experiment.environment.terminate_on_violation
    assert experiment.run.collections == 400 and len(experiment.task.curriculum.stages) == 3
    assert Experiment.from_dict(RAW) == experiment


@pytest.mark.parametrize('path, value', [
    (('environment', 'duration_s'), 4.005),          # off the 10 ms grid
    (('environment', 'duration_s'), -1),
    (('environment', 'policy_rate_hz'), 30),         # does not divide 100 Hz
    (('environment', 'acceleration_limit_mps2'), [8, 8]),
    (('environment', 'maximum_tilt_deg'), 90),
    (('environment', 'workspace'), dict(radius_m=.55)),
    (('task', 'name'), 'unknown_v1'),
    (('task', 'target_direction_world'), [0, 1, 0]), # only +X is implemented
    (('task', 'hit_radius_m'), 0),
    (('task', 'curriculum'), KeyError),
    (('task', 'extra'), 1),
    (('run', 'device'), 'tpu'),
    (('run', 'seed'), 1.5),
    (('ppo', 'discount'), 0),
    (('ppo', 'post_step_kl_trials'), 13),
    (('ppo', 'entropy'), dict(initial_weight=.001, final_weight=.02, decay_exponent=8.)),
    (('scene',), KeyError),
])
def test_invalid_values_are_rejected(path, value):
    with pytest.raises(ValueError):
        load(edited(path, value))


def test_curriculum_rate_must_be_a_fraction():
    with pytest.raises(ValueError):
        load(edited(('task', 'curriculum', 'advance_strike_rate'), 1.5))


def test_identity_ignores_only_the_episode_budget():
    longer = deepcopy(SNAPSHOT); longer['experiment']['run']['training_episodes'] *= 2
    changed = deepcopy(SNAPSHOT); changed['experiment']['task']['strike_bonus'] = 11.
    assert identity(longer) == identity(SNAPSHOT) != identity(changed)
