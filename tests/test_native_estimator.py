"""The native flight estimator equals the Python reference it follows: the cable filter on a recorded
whip flight (labels and states bit-identical), the observer on simulated sensor streams (cable states
bit-identical; drone velocity and body rate to 1e-12, the velocity fit being solved differently)."""
import numpy as np
import pytest
import torch

from aerowhip.deployment.cable_filter import CableFilter, ReferenceCableFilter, cable_candidates, reference_cable_candidates
from aerowhip.deployment.observer import Observer, ReferenceObserver
from aerowhip.envs.batch_environment import BatchEnvironment
from helpers import ENVIRONMENT, SCENE, SNAPSHOT, TASK


def recorded_flight():
    from aerowhip.fitting.calibration_set import folders
    from aerowhip.takes import cable_labels
    try:
        return cable_labels.load_flight(folders('cf_2')['whip/002'])
    except Exception as exc:            # the calibration data lives in the companion repository
        pytest.skip(f'recorded whip flight unavailable: {exc}')


def test_cable_filter_matches_the_reference_on_a_recorded_whip():
    flight = recorded_flight()
    native, reference = CableFilter(flight.arc), ReferenceCableFilter(flight.arc)
    for i in range(0, len(flight.stamps)):
        a = reference.observe(float(flight.stamps[i]), flight.root[i], flight.points[i])
        b = native.observe(float(flight.stamps[i]), flight.root[i], flight.points[i])
        assert a['initialized'] == b['initialized'] and np.array_equal(a['index'], b['index']), i
        for key in ('positions', 'velocities', 'confidence'):
            assert np.array_equal(a[key], b[key], equal_nan=True), (i, key)
        assert a['quality'] == b['quality'], i
        if i % 13 == 0:
            assert np.array_equal(reference_cable_candidates(flight.points[i], flight.root[i]),
                                  cable_candidates(flight.points[i], flight.root[i]))


def test_filter_state_round_trip_and_override():
    flight = recorded_flight()
    native, reference = CableFilter(flight.arc), ReferenceCableFilter(flight.arc)
    for i in range(400):
        native.observe(float(flight.stamps[i]), flight.root[i], flight.points[i])
        reference.observe(float(flight.stamps[i]), flight.root[i], flight.points[i])
    saved = native.get_state()
    labels = np.array(native.index); labels[3] = -1
    native.override(flight.root[400], flight.points[400], labels)
    reference.override(flight.root[400], flight.points[400], labels)
    assert np.array_equal(native.state()['positions'], reference.state()['positions'], equal_nan=True)
    native.set_state(saved)
    assert np.array_equal(native.get_state()['positions'], saved['positions'], equal_nan=True)


def test_observer_matches_the_reference_on_simulated_sensor_streams():
    env = BatchEnvironment(ENVIRONMENT, TASK, SNAPSHOT['model'], SCENE, device='cpu', lanes=3, randomised=True, sensing_seed=7)
    streams = []
    for sensor in env.sensors:
        calls, original = [], sensor.observer.observe
        sensor.observer.observe = lambda *a, calls=calls, original=original: (calls.append([np.copy(x) for x in a]), original(*a))[1]
        streams.append(calls)
    env.reset()
    generator = np.random.default_rng(3)
    done = False
    while not done:
        _, _, done = env.step(torch.as_tensor(np.clip(generator.normal(0, .5, (3, 3)), -1, 1), dtype=torch.float32))
    arc = np.cumsum(SNAPSHOT['model']['cable']['marker_interval_lengths_m'])
    attachment = SNAPSHOT['model']['attachment']['position_tracking_m']
    for calls in streams:
        native, reference = Observer(arc, attachment), ReferenceObserver(arc, attachment)
        for stamp, position, rotation, cloud in calls:
            a, b = reference.observe(float(stamp), position, rotation, cloud), native.observe(float(stamp), position, rotation, cloud)
            assert a['initialized'] == b['initialized'] and np.array_equal(a['labelled'], b['labelled'])
            for key in ('root', 'cable_positions', 'cable_velocities', 'confidence'):
                assert np.array_equal(a[key], b[key]), key
            for key in ('velocity', 'angular_velocity', 'root_velocity'):
                assert np.allclose(a[key], b[key], rtol=0, atol=1e-12), key
