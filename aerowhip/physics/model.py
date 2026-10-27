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


def aerodynamic_weights(residual):
    if residual is None:
        return None
    return dict(cable_log_bound=residual.cable_log_bound, drone_max_rate=residual.drone_max_rate_s_inv,
                velocity_scale=residual.velocity_scale_m_s,
                cable_enabled=residual.specification['cable_enabled'],drone_enabled=residual.specification['drone_enabled'],
                **{name:[(as_array(branch[i].weight),as_array(branch[i].bias)) for i in (0,2,4)]
                   for name,branch in (('cable',residual.cable),('drone',residual.drone))})


class Model:
    """Owned model metadata and native mesh/state validation.

    Torch is used only when deserializing an existing .pt residual checkpoint;
    no Torch physics model, controller or integration runtime is constructed.
    """
    def __init__(self, config, *, residual=None, load_weights=True):
        self.config, self.configuration_sources = load_model_configuration(config, with_sources=True)
        reference = self.config.get('residual', {})
        self.residual = residual
        if self.residual is None and load_weights and reference.get('status') == 'fitted':
            from aerowhip.fitting.residual_network import load_residual
            self.residual = load_residual(reference, 'cpu')
        self.native = native_module().Model(self.config, aerodynamic_weights(self.residual))
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
