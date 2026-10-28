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
    # Internal forces conserve momentum up to the implicit solver's stopping rule: each step ends when
    # the velocities change by less than 2e-11 (relative), so 0.2 s of 2 ms steps can leave up to
    # steps x 2e-11 x |momentum|. With the measured cable stiffness (EI 5.7e-4 N m^2) the leftover is
    # ~1e-10 kg m/s, growing with stiffness; with a nearly limp cable it was ~1e-18.
    steps = round(.2/model.native.info()['physics_dt'])
    target = before+[0., 0., -G*.2*total]
    assert np.allclose(momentum(model, after), target, rtol=0, atol=steps*2e-11*np.linalg.norm(target))
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


def test_body_drag_is_linear_in_body_frame_velocity():
    # Drone-only acceleration change from the body drag: -R diag(k) R^T v, with the cable slack-free
    # contribution removed by comparing the same state with and without drag.
    config, model, engine = setup(drag=False)
    k = np.array([.4, .3, .2])
    i = model.native.info()
    raw = dict(mass=i['mass'], stiffness=i['ei'], drag=i['drag'], bending_damping=i['cb'], air_density=i['air_density'],
               normal_drag_coefficient=i['cd'], diameter=i['diameter'], maximum_step=i['physics_dt'],
               projections=i['iterations'], inertia=i['inertia'], arm=i['arm'], gravity=i['gravity'],
               root_tangent_body=i['root_tangent_body'], rest=i['rest'], node_mass=i['node_mass'])
    dragged = native_module().Engine(dict(raw, body_drag=k))
    angle = .3
    R = np.array([[math.cos(angle), -math.sin(angle), 0.], [math.sin(angle), math.cos(angle), 0.], [0., 0., 1.]])
    v = np.array([1.2, -.5, .3])
    state = model.native.initial_state(np.array([0., 0., 2.]), R, v, np.zeros(3), None, None)
    engine.reset(state); dragged.reset(state)
    w = np.array([(model.mass+model.node_mass.sum())*G, 0., 0., 0.])
    d = dragged.acceleration(w)-engine.acceleration(w)
    # The clamp transmits part of the drag to the cable, so the check is on the whole system: its
    # momentum changes by exactly the drag force. The clamped node 0 moves with the body point at
    # the arm (zero body rate here): a0 = a_com + R (alpha x arm).
    force = -model.mass*R@(k*(R.T@v))
    root = d[:3]+R@np.cross(d[3:6], i['arm'])
    total = model.mass*d[:3]+model.node_mass[0]*root+(model.node_mass[1:, None]*d[6:].reshape(-1, 3)).sum(0)
    np.testing.assert_allclose(total, force, rtol=0, atol=1e-12)
    np.testing.assert_allclose(Model(dict(config, drone=dict(config['drone'], body_drag_s_inv=list(k))), load_weights=False).native.info()['body_drag'], k)
