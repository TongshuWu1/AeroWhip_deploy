"""Native twist-free plant: exact checks that need no fitted parameters.

Free fall and momentum use zero air drag; the hanging hold uses the default model.
"""
from copy import deepcopy
import math

import numpy as np

from aerowhip.fitting.flight_drone import firmware_config
from aerowhip.physics.model import Model, native_module
from aerowhip.physics.model_configuration import DEFAULT_MODEL, load_model_configuration

G = 9.80665


def setup(drag=True):
    config = load_model_configuration(DEFAULT_MODEL)
    if not drag:
        config = deepcopy(config)
        config['cable']['external_drag_s_inv'] = 0.
        config['cable']['normal_drag_coefficient'] = 0.
    model = Model(config, load_weights=False)
    i = model.native.info()
    engine = native_module().Engine(dict(
        mass=i['mass'], stiffness=i['ei'], drag=i['drag'], bending_damping=i['cb'], air_density=i['air_density'],
        normal_drag_coefficient=i['cd'], diameter=i['diameter'], maximum_step=i['physics_dt'],
        projections=i['iterations'], inertia=i['inertia'], arm=i['arm'], gravity=i['gravity'],
        root_tangent_body=i['root_tangent_body'], rest=i['rest'], node_mass=i['node_mass']))
    return config, model, engine


def cable(model, state):
    n = model.n
    o = model.native.observation(state)
    return o[18:18+3*n].reshape(n, 3), o[18+3*n:18+6*n].reshape(n, 3)


def edge_error(model, state):
    q = cable(model, state)[0]
    return float(np.abs(np.linalg.norm(np.diff(q, axis=0), axis=1)-model.rest).max())


def momentum(model, state):
    return model.mass*state[12:15]+(model.node_mass[:, None]*cable(model, state)[1]).sum(0)


def test_free_fall_without_thrust_or_drag():
    _, model, engine = setup(drag=False)
    engine.reset(model.native.initial_state(np.array([0., 0., 2.]), np.eye(3), np.zeros(3), np.zeros(3), None, None))
    engine.advance(.1, np.zeros(4))
    state = engine.state()
    assert np.allclose(state[12:15], [0., 0., -G*.1], rtol=0, atol=1e-12)
    assert edge_error(model, state) < 1e-12


def test_weight_thrust_holds_hanging_cable_at_rest():
    _, model, engine = setup()
    start = model.native.initial_state(np.array([0., 0., 2.]), np.eye(3), np.zeros(3), np.zeros(3), None, None)
    engine.reset(start)
    weight = (model.mass+model.node_mass.sum())*G
    engine.advance(1., np.array([weight, 0., 0., 0.]))
    state = engine.state()
    assert np.abs(state[:3]-start[:3]).max() < 1e-12 and np.linalg.norm(state[12:18]) < 1e-12
    assert np.abs(cable(model, state)[1]).max() < 1e-12


def test_bending_cable_conserves_momentum_and_lengths_in_free_fall():
    # A straight cable at 40 deg to the body-down clamp tangent bends internally while falling.
    _, model, engine = setup(drag=False)
    tracking = np.array([0., 0., 2.])
    root = tracking+model.attachment_tracking
    angle = math.radians(40)
    direction = np.array([math.sin(angle), 0., -math.cos(angle)])
    positions = root+np.r_[0., np.cumsum(model.rest)][:, None]*direction
    state = model.native.initial_state(tracking, np.eye(3), np.zeros(3), np.zeros(3), positions, np.zeros_like(positions))
    engine.reset(state)
    total = model.mass+model.node_mass.sum()
    before = momentum(model, state)
    engine.advance(.2, np.zeros(4))
    after = engine.state()
    assert np.allclose(momentum(model, after), before+[0., 0., -G*.2*total], rtol=0, atol=1e-12)
    assert edge_error(model, after) < 1e-12
    assert abs(after[16]) > 1e-5      # the clamp couple acted: internal forces were present


def test_batch_lanes_reproduce_a_single_runtime():
    config, model, _ = setup()
    native = native_module()
    firmware = firmware_config()
    state = model.native.initial_state(np.array([0., 0., 1.5]), np.eye(3), np.zeros(3), np.zeros(3), None, None)
    hold = np.zeros(13)
    period, delay = config['controller']['controller_period_s'], config['controller']['command_delay_s']
    actuator = native.Actuator(config['actuator'])
    voltages = [7.0, 7.2, 7.4]
    batch = native.BatchRuntime(model.native, state, hold, period, delay, actuator,
                                [native.FlightController(firmware, config, v) for v in voltages])
    single = native.Runtime(model.native, state, hold, period, delay, actuator)
    controller = native.FlightController(firmware, config, voltages[1])
    for k in range(50):
        command = np.zeros((3, 13))
        command[:, 6] = 2*math.sin(.3*k)
        command[:, 8] = 1.
        batch.step(command)
        single.step([(single.time, command[1])], controller)
    assert not any(batch.failed)
    states = batch.states()
    assert np.array_equal(states[1], single.state())          # bit-identical lane
    assert not np.array_equal(states[0], states[2])           # voltage changes the thrust
