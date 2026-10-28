"""File/array binding to the C++ model; no Python physical equations."""
from dataclasses import dataclass, fields, replace

import numpy as np
from aerowhip.physics.model_configuration import load_model_configuration


def native_module():
    try:
        from aerowhip.physics import _native
    except ImportError as exc:
        raise RuntimeError('C++ engine is not built. Run build_engine.py with this interpreter.') from exc
    if _native.api_version != 12:
        raise RuntimeError('C++ engine version mismatch. Rebuild with build_engine.py.')
    return _native


def as_array(value):
    if hasattr(value, 'detach'):
        value = value.detach().cpu().numpy()
    return np.asarray(value, dtype=np.float64)


@dataclass(frozen=True)
class PlantState:
    position_com: object
    rotation: object
    velocity_com: object
    omega_body: object
    free_positions: object
    free_velocities: object


def unpack_state(value):
    value = np.asarray(value)
    if value.ndim != 1 or (len(value)-18)%6 or len(value)<24:
        raise ValueError("Expected packed twist-free state v3; historical states require explicit reconstruction")
    count = (len(value)-18)//6
    return PlantState(value[:3].copy(), value[3:12].reshape(3, 3).copy(),
                      value[12:15].copy(), value[15:18].copy(),
                      value[18:18+3*count].reshape(count, 3).copy(),
                      value[18+3*count:18+6*count].reshape(count, 3).copy())


def pack_state(state):
    return np.concatenate([as_array(getattr(state, f.name)).ravel() for f in fields(PlantState)])


def residual_net_force(config, rotation=np.eye(3)):
    """World net force (N) of the steady part of `cable.residual` on the whole cable, for a drone at
    this attitude (body to world): each node's force per unit length times its tributary length, as
    the engine applies it. Zero without a residual."""
    residual = config['cable'].get('residual') or {}
    static = np.asarray(residual.get('static_n_per_m', np.zeros((3, 3))), float)
    rest = np.asarray(config['cable']['marker_interval_lengths_m'], float)
    arc = np.r_[0., np.cumsum(rest)]
    tributary = np.r_[rest[0]/2, (rest[:-1]+rest[1:])/2, rest[-1]/2]
    s = arc/arc[-1]
    force = (static@np.vstack([np.ones_like(s), s, s*s])@tributary)
    return np.asarray(rotation)@force if residual.get('frame', 'body') == 'body' else force


def hanging_cable(config, root, rotation=np.eye(3)):
    """Node positions (n+1, 3, the root first) of the cable at rest below a still clamp at `root`
    under gravity and the steady part of `cable.residual`: each edge along the total load on the
    nodes below it (a chain; bending is negligible at rest, EI/(m g L^2) ~ 1e-3)."""
    cable, simulation = config['cable'], config['simulation']
    rest = np.asarray(cable['marker_interval_lengths_m'], float)
    total = float(np.sum(rest))
    nodes = np.r_[0., np.cumsum(rest)]
    # node masses as the engine lumps them: half of each adjacent edge, plus each marker's mass
    edge_mass = (cable['total_mass_kg']-float(np.sum(cable['moving_marker_masses_kg'])))*rest/total
    mass = np.r_[edge_mass[0]/2, (edge_mass[:-1]+edge_mass[1:])/2, edge_mass[-1]/2]
    mass[1:] += np.asarray(cable['moving_marker_masses_kg'], float)
    residual = cable.get('residual') or {}
    static = np.asarray(residual.get('static_n_per_m', np.zeros((3, 3))), float)
    s = nodes/total
    per_length = (static@np.vstack([np.ones_like(s), s, s*s])).T
    if residual.get('frame', 'body') == 'body':
        per_length = per_length@np.asarray(rotation).T
    tributary = np.r_[rest[0]/2, (rest[:-1]+rest[1:])/2, rest[-1]/2]
    load = mass[:, None]*np.asarray(simulation['gravity_world_m_s2'], float)+per_length*tributary[:, None]
    below = np.cumsum(load[::-1], axis=0)[::-1]           # total load on node i and every node below it
    q = [np.asarray(root, float)]
    for i, length in enumerate(rest):
        direction = below[i+1]/np.linalg.norm(below[i+1])
        q.append(q[-1]+direction*length)
    return np.array(q)


def settled_cable(model, root, rotation=np.eye(3), seconds=2.):
    """`hanging_cable`, then `seconds` of the model's own cable below a held clamp (bending at the
    point clamp included), so the result is the model's rest shape: node positions (n+1, 3)."""
    q = hanging_cable(model.config, root, rotation)
    native = model.native
    free = q[1:]
    state, previous, step = np.r_[free.ravel(), np.zeros(free.size)], None, 0.
    previous = state.copy()
    boundary = np.r_[np.asarray(root, float), np.zeros(6), np.asarray(rotation, float).ravel(), np.zeros(3)]
    h = float(model.physics_dt)
    rows = np.tile(np.r_[h, np.tile(boundary, 4)], (1, 1))
    for _ in range(round(seconds/h)):
        state, previous, step = native.prescribed_advance(state, rows, previous, step)
    return np.vstack([root, np.asarray(state[:free.size]).reshape(-1, 3)])


class Model:
    """Owned model metadata and native mesh/state validation."""
    def __init__(self, config):
        self.config, self.configuration_sources = load_model_configuration(config, with_sources=True)
        self.native = native_module().Model(self.config)
        for name, value in self.native.info().items():
            setattr(self, name, value)
        self.marker_indices = np.asarray(self.marker_indices, dtype=np.int64)

    def initial_state(self, tracking_position=(0., 0., 2.), *, rotation=None,
                      tracking_velocity=(0., 0., 0.), omega_body=(0., 0., 0.),
                      cable_positions=None, cable_velocities=None):
        state = unpack_state(self.native.initial_state(as_array(tracking_position),
            np.eye(3) if rotation is None else as_array(rotation),
            as_array(tracking_velocity), as_array(omega_body),
            None if cable_positions is None else as_array(cable_positions),
            None if cable_velocities is None else as_array(cable_velocities)))
        return state

    def tracking_pose(self, state):
        values = self.native.observation(pack_state(state))
        return values[:3].copy(), values[3:6].copy()

    def cable_state(self, state):
        values = self.native.observation(pack_state(state))
        return values[18:18+3*self.n].reshape(-1, 3).copy(), values[18+3*self.n:-3].reshape(-1, 3).copy()
