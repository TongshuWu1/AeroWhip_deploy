"""The OptiTrack model: dropout statistics, latency, the cloud's contents and the filter on it."""
import numpy as np
import pytest

from aerowhip.deployment.cable_filter import MARKERS, cable_candidates
from aerowhip.envs.sensing import OptiTrackSensor, SensingConfig
from helpers import EXPERIMENT

ARC = np.cumsum([.095]+[.1]*9)
ATTACHMENT = np.array([0., 0., -.0423])


def config(**changes):
    raw = dict(latency_frames=[0, 2], position_noise_m=1e-4, dropout_hazard=[[0., .01], [1., .1]],
               reappear_probability=.3, drone_markers_body_m=[[.09, .01, 0.], [-.08, -.02, 0.]],
               clutter_world_m=[[3., 3., .1]])
    raw.update(changes)
    return SensingConfig.from_dict(raw)


def hanging(p, R):
    """Ten markers hanging straight below the attachment."""
    root = p+R@ATTACHMENT
    return root+np.outer(ARC, [0., 0., -1.])


def sensor(cfg, seed=0):
    return OptiTrackSensor(cfg, ARC, ATTACHMENT, np.random.default_rng(seed))


def test_config_validation():
    assert EXPERIMENT.environment.sensing.latency_frames == (0, 2)
    with pytest.raises(ValueError):
        config(dropout_hazard=[[.5, .01]])            # must start at speed 0
    with pytest.raises(ValueError):
        config(dropout_hazard=[[0., .01], [0., .02]])
    with pytest.raises(ValueError):
        config(latency_frames=[2, 1])
    with pytest.raises(ValueError):
        config(reappear_probability=0.)


def test_cloud_contents_and_candidates():
    s = sensor(config(latency_frames=[0, 0]))
    p, R = np.array([0., 0., 2.]), np.eye(3)
    markers = hanging(p, R)
    cloud = s.cloud(0, p, R, markers, np.zeros(MARKERS))
    assert cloud.shape == (MARKERS+2+1, 3)              # markers, two body markers, one clutter point
    candidates = cable_candidates(cloud, p)
    assert len(candidates) == MARKERS+2+1               # body markers at 8-9 cm and the clutter stay
    out = s.observe(0, p, R, np.zeros(3), np.zeros(3), markers, np.zeros((MARKERS, 3)))
    assert (out['confidence'] == 1.).all() and out['labelled'].all()
    assert np.abs(out['cable_positions']-markers).max() < 1e-3


def test_dropout_statistics_follow_the_hazard_table():
    cfg = config(latency_frames=[0, 0], dropout_hazard=[[0., .02], [1., .2]], reappear_probability=.4)
    p, R = np.array([0., 0., 2.]), np.eye(3)
    markers = hanging(p, R)
    for speed, hazard in ((0., .02), (1.5, .2)):
        s = sensor(cfg, seed=3)
        hidden = []
        for i in range(20000):
            s.cloud(i, p, R, markers, np.full(MARKERS, speed))
            hidden.append(s.hidden.copy())
        hidden = np.array(hidden)
        expected = hazard/(hazard+.4)
        assert abs(hidden.mean()-expected) < .01
        assert not hidden[0].any()                      # the first frame is complete


def test_latency_delays_what_the_policy_sees():
    s = sensor(config(latency_frames=[2, 2], dropout_hazard=[[0., 0.]]))
    R = np.eye(3)
    outs = []
    for i in range(6):
        p = np.array([.1*i, 0., 2.])
        outs.append(s.observe(i, p, R, np.zeros(3), np.zeros(3), hanging(p, R), np.zeros((MARKERS, 3))))
    assert [o['index'] for o in outs] == [0, 0, 0, 1, 2, 3]
    assert outs[5]['position'][0] == pytest.approx(.3)
    assert np.abs(outs[5]['cable_positions'][:, 0]-.3).max() < 2e-3


def test_hidden_markers_coast_with_reduced_confidence():
    cfg = config(latency_frames=[0, 0], dropout_hazard=[[0., 0.]])
    s = sensor(cfg)
    p, R = np.array([0., 0., 2.]), np.eye(3)
    markers = hanging(p, R)
    s.observe(0, p, R, np.zeros(3), np.zeros(3), markers, np.zeros((MARKERS, 3)))
    s.hidden[9] = True
    s.hazards[:] = 0.; s.config = config(latency_frames=[0, 0], dropout_hazard=[[0., 0.]], reappear_probability=1e-9)
    for i in range(1, 11):
        out = s.observe(i, p, R, np.zeros(3), np.zeros(3), markers, np.zeros((MARKERS, 3)))
    assert out['confidence'][9] < 1. and out['confidence'][:9].min() == 1.
    assert np.isfinite(out['cable_positions']).all() and not out['labelled'][9]


def test_lane_draws_are_a_function_of_seed_and_lane():
    cfg = config()
    a, b, c = sensor(cfg, seed=7), sensor(cfg, seed=7), sensor(cfg, seed=8)
    p, R = np.array([0., 0., 2.]), np.eye(3)
    clouds = [[s.cloud(i, p, R, hanging(p, R), np.full(MARKERS, 2.)) for i in range(50)] for s in (a, b, c)]
    assert all(np.array_equal(x, y) for x, y in zip(clouds[0], clouds[1]))
    assert not all(np.array_equal(x, y) for x, y in zip(clouds[0], clouds[2]))
