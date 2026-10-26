"""Native coupled drone-cable plant: exact checks that need no fitted parameters.

Free fall and momentum use zero external cable drag; hover uses the default model.
"""
from copy import deepcopy
import math

import numpy as np
import torch

from controllers.full_state_controller.tensor_controller import GeometricController
from helpers import SNAPSHOT
from simulation.tensor_model import TensorModel

G = 9.80665


def model(drag=None):
    config = deepcopy(SNAPSHOT['model'])
    if drag is not None:
        config['cable']['external_drag_s_inv'] = drag
    return TensorModel(config)


def edge_error(m, state):
    q = m.cable_state(state)[0]
    return float((torch.linalg.norm(torch.diff(q, dim=0), dim=-1)-m.rest).abs().max())


def momentum(m, state):
    _, velocities = m.cable_state(state)
    return m.mass*state.velocity_com+(m.node_mass[:, None]*velocities).sum(0)


def test_free_fall_without_thrust_or_drag():
    m = model(drag=0.)
    state = m.advance_actuation(m.initial_state([0., 0., 2.]), 0., [0., 0., 0.], .1)
    torch.testing.assert_close(state.velocity_com, m.tensor([0., 0., -G*.1]), rtol=0, atol=1e-12)
    assert edge_error(m, state) < 1e-12


def test_zero_acceleration_command_holds_hanging_cable_at_rest():
    m = model()
    controller = GeometricController(m)
    state = m.controller_state(m.initial_state([0., 0., 2.]))
    start = m.tracking_pose(state)[0].clone()
    for _ in range(100):  # 1 s at the 100 Hz output step
        state = m.step_unchecked(state, m.tensor([0., 0., 0., 0.]), controller.acceleration)
    position, velocity = m.tracking_pose(state)
    assert float((position-start).abs().max()) < 1e-12 and float(velocity.norm()) < 1e-12
    assert float(m.cable_state(state)[1].norm(dim=-1).max()) < 1e-12


def test_bending_cable_conserves_momentum_and_lengths_in_free_fall():
    # A straight cable at 40° to the body-down clamp tangent bends internally while falling.
    m = model(drag=0.)
    root = m.cable_state(m.initial_state([0., 0., 2.]))[0][0].numpy()
    angle = math.radians(40)
    direction = np.array([math.sin(angle), 0., -math.cos(angle)])
    arc = np.r_[0., np.cumsum(m.rest.numpy())]
    positions = root+arc[:, None]*direction
    state = m.initial_state([0., 0., 2.], cable_positions=positions, cable_velocities=np.zeros_like(positions))
    total_mass = float(m.mass+m.node_mass.sum())
    before = momentum(m, state)
    state = m.advance_actuation(state, 0., [0., 0., 0.], .2)
    expected = before+m.tensor([0., 0., -G*.2*total_mass])
    torch.testing.assert_close(momentum(m, state), expected, rtol=0, atol=1e-12)
    assert edge_error(m, state) < 1e-12
    assert abs(float(state.omega_body[1])) > 1e-5  # the clamp couple acted: internal forces were present
