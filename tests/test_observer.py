"""The flight estimator (deployment.observer) and the policy frame (envs.observation): exact drone
velocities on jittered stamps, and the same rows in simulation and in flight."""
import math

import numpy as np

from aerowhip.deployment.cable_filter import MARKERS
from aerowhip.deployment.observer import Observer
from aerowhip.envs.observation import actor_columns, frame_row, frame_size
from aerowhip.envs.sensing import OptiTrackSensor, SensingConfig

ARC = np.cumsum([.095]+[.1]*9)
ATTACHMENT = np.array([0., 0., -.0423])


def yaw(angle):
    c, s = math.cos(angle), math.sin(angle)
    return np.array([[c, -s, 0.], [s, c, 0.], [0., 0., 1.]])


def hanging(p, R):
    root = p+R@ATTACHMENT
    return root+np.outer(ARC, [0., 0., -1.])


def test_drone_velocity_and_rate_from_jittered_poses():
    rng = np.random.default_rng(0)
    observer = Observer(ARC, ATTACHMENT)
    velocity, rate = np.array([.8, -.3, .2]), .5
    for frame in range(30):
        t = frame*.01
        p, R = np.array([0., 0., 2.])+velocity*t, yaw(rate*t)
        estimate = observer.observe(t+rng.uniform(-.004, .004), p, R, hanging(p, R))
    np.testing.assert_allclose(estimate['velocity'], velocity, atol=1e-9)
    np.testing.assert_allclose(estimate['angular_velocity'], [0., 0., rate], atol=1e-9)
    np.testing.assert_allclose(estimate['root_velocity'], velocity+R@np.cross([0., 0., rate], ATTACHMENT), atol=1e-9)


def test_simulation_and_flight_build_the_same_rows():
    # The simulated sensor without noise, dropout, jitter or latency hands the Observer exactly the
    # frames a flight node would receive; the rows must be identical.
    config = SensingConfig.from_dict(dict(latency_frames=[0, 0], position_noise_m=0., drone_position_noise_m=0., drone_rotation_noise_rad=0.,
                                          receipt_jitter_s=0., dropout_hazard=[[0., 0.]], reappear_probability=.5,
                                          drone_markers_body_m=[[.09, .01, 0.]], clutter_world_m=[]))
    sensor = OptiTrackSensor(config, ARC, ATTACHMENT, np.random.default_rng(1))
    flight = Observer(ARC, ATTACHMENT)
    origin, previous = np.array([0., 0., 2.]), np.array([.1, -.2, .3])
    for frame in range(20):
        p, R = np.array([.02*frame, 0., 2.]), yaw(.01*frame)
        markers = hanging(p, R)
        simulated = sensor.observe(frame, p, R, markers, np.zeros((MARKERS, 3)))
        cloud = np.vstack([markers, p+R@np.array([.09, .01, 0.])])
        real = flight.observe(frame*.01, p, R, cloud)
        rows = [frame_row(e, origin=origin, length=.995, previous_action=previous, progress=frame/400) for e in (simulated, real)]
        assert np.array_equal(rows[0], rows[1])
    assert len(rows[0]) == frame_size(MARKERS+1)
    columns = actor_columns(MARKERS+1)
    assert len(columns) == len(rows[0])-8 and np.array_equal(rows[0][columns][-4:-1], previous)
