"""The compact cable residual: absent or all zero is bit-identical to the plain physics (coupled
engine); a steady horizontal load tilts a cable hanging from a held clamp as a chain under that load
would (prescribed-clamp path); the model reports what it was given."""
import math

import numpy as np

from aerowhip.fitting.flight_cable import CableReplay
from aerowhip.physics.model import Model, pack_state
from aerowhip.physics.model_configuration import DEFAULT_MODEL, load_model_configuration
from aerowhip.physics.native_runtime import native_module
from aerowhip.takes.calibration_flights import DT, Window

G = 9.80665


def swung_state(model, swing):
    """A straight cable hanging from the drone at (0, 0, 2), swung by `swing` rad about the clamp."""
    arc = np.r_[0., np.cumsum(model.rest)]
    direction = np.array([math.sin(swing), 0., -math.cos(swing)])
    root = np.array([0., 0., 2.])+np.asarray(model.attachment_tracking)
    q = root+arc[:, None]*direction                     # node 0 is the root
    return model.initial_state((0., 0., 2.), cable_positions=q, cable_velocities=np.zeros_like(q)), q


def coupled_rollout(config, seconds):
    """Packed state after `seconds` of the coupled engine under a constant weight-equal thrust."""
    model = Model(config)
    info = model.native.info()
    state, _ = swung_state(model, .3)
    engine = native_module().Engine(dict(
        mass=model.mass, inertia=np.asarray(model.inertia), arm=np.asarray(model.arm), gravity=np.asarray(model.gravity),
        rest=np.asarray(model.rest), node_mass=np.asarray(model.node_mass), root_tangent_body=np.asarray(model.root_tangent_body),
        body_drag=np.asarray(model.body_drag), bending_damping=model.cb, air_density=model.air_density,
        bending_damping_saturation=model.cb_saturation, normal_drag_coefficient=model.cd, diameter=model.diameter,
        stiffness=model.ei, drag=model.drag, maximum_step=model.physics_dt, projections=model.iterations,
        **{k: info[k] for k in ('residual_static', 'residual_body_frame', 'residual_normal_log', 'residual_tangential')}))
    engine.reset(pack_state(state))
    weight = -(model.mass+float(np.sum(model.node_mass)))*model.gravity[2]
    for _ in range(round(seconds/model.physics_dt)):
        engine.advance(model.physics_dt, np.array([weight, 0., 0., 0.]))
    return engine.state()


def held_clamp_rollout(config, seconds):
    """Marker positions after `seconds` with the clamp held still at the hanging pose."""
    model = Model(config)
    _, q = swung_state(model, 0.)
    n = round(seconds/DT)
    position, rotation = np.array([0., 0., 2.]), np.eye(3)
    markers = q[np.asarray(model.marker_indices)[1:]]
    window = Window('test', 'hold', 0., np.arange(n+1)*DT, np.repeat(position[None], n+1, 0), np.repeat(rotation[None], n+1, 0),
                    np.repeat(markers[None], n+1, 0), np.ones((n+1, len(markers)), bool), np.zeros((n+1, 3)), np.zeros((n+1, 3)),
                    np.repeat(position[None], 11, 0), np.repeat(rotation[None], 11, 0), np.repeat(markers[None], 11, 0), np.arange(11)*DT)
    return CableReplay(model.native).run(window)[-1], model, q[0]


def test_absent_and_zero_residual_are_identical():
    config = load_model_configuration(DEFAULT_MODEL)
    config = dict(config, cable={k: v for k, v in config['cable'].items() if k != 'residual'})
    zero = dict(config, cable=dict(config['cable'], residual=dict(frame='body', static_n_per_m=[[0.]*3]*3,
                                                                  normal_drag_log=[0., 0.], tangential_drag=[0., 0.])))
    assert np.array_equal(coupled_rollout(config, 1.), coupled_rollout(zero, 1.))


def test_steady_load_tilts_the_hanging_cable():
    config = load_model_configuration(DEFAULT_MODEL)
    load = .006                                          # N/m along world x: about the lean seen in flight
    tilted = dict(config, cable=dict(config['cable'], residual=dict(frame='world', static_n_per_m=[[load, 0., 0.], [0.]*3, [0.]*3])))
    markers, model, root = held_clamp_rollout(tilted, 30.)        # the swing into equilibrium decays over ~20 s
    rest, mass = np.asarray(model.rest), np.asarray(model.node_mass)
    # a chain of these nodes: each edge hangs at atan(horizontal load below it / weight below it),
    # the load on a node being f times its tributary length
    tributary = np.r_[rest[0]/2, (rest[:-1]+rest[1:])/2, rest[-1]/2]
    expected = np.array([math.atan(load*tributary[i+1:].sum()/(mass[i+1:].sum()*G)) for i in range(len(rest))])
    edges = np.diff(np.vstack([root, markers]), axis=0)
    angles = np.arctan2(edges[:, 0], -edges[:, 2])
    # the clamp's bending moment holds the top edge straighter and the free end has no load below it:
    # compare the interior edges
    np.testing.assert_allclose(angles[2:8], expected[2:8], atol=math.radians(.3))
    assert np.abs(markers[:, 1]-root[1]).max() < 1e-9


def test_info_reports_the_residual():
    config = load_model_configuration(DEFAULT_MODEL)
    r = dict(frame='world', static_n_per_m=[[1., 2., 3.], [4., 5., 6.], [7., 8., 9.]], normal_drag_log=[.1, .2], tangential_drag=[.3, .4])
    info = Model(dict(config, cable=dict(config['cable'], residual=r))).native.info()
    assert info['residual_body_frame'] is False
    np.testing.assert_array_equal(info['residual_static'], r['static_n_per_m'])
    np.testing.assert_array_equal(info['residual_normal_log'], r['normal_drag_log'])
    np.testing.assert_array_equal(info['residual_tangential'], r['tangential_drag'])
