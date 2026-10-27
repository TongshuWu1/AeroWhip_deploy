"""Read-only world/robot display contract; no integration or control behavior.

Positions are metres in frame_id, with world +Z up; rotations are XYZW body to
world. Each robot and optional cable carries its own timestamp and provenance.
Display frames are not physical checkpoints or policy observations.
"""
from dataclasses import dataclass, fields, is_dataclass, replace
from enum import Enum

import numpy as np

from aerowhip.physics.state import DroneState, ForceTelemetry, SimulationState


class Provenance(str, Enum):
    SIMULATED = 'Simulated'
    MEASURED = 'Measured'
    ESTIMATED = 'Estimated'


def immutable(value):
    """Own array bytes so a producer or consumer cannot mutate a published frame."""
    if isinstance(value, np.ndarray):
        return np.frombuffer(value.tobytes(), dtype=value.dtype).reshape(value.shape)
    if is_dataclass(value):
        return replace(value, **{f.name: immutable(getattr(value, f.name)) for f in fields(value)})
    return value


def _array(value, shape, label, *, dtype=np.float64):
    array = np.asarray(value, dtype=dtype)
    if array.shape != shape or not np.isfinite(array).all():
        raise ValueError(f'{label} requires finite values with shape {shape}')
    return immutable(array)


@dataclass(frozen=True)
class WorldDisplay:
    frame_id: str = 'world'
    ground_z_m: float = 0.
    ground_size_m: float = 8.
    grid_spacing_m: float = .5
    world_axis_length_m: float = .25
    background_rgb: tuple = (.73, .77, .80)

    def __post_init__(self):
        if not self.frame_id or not np.isfinite(self.ground_z_m):
            raise ValueError('World requires a named frame and finite floor height')
        if any(not np.isfinite(v) or v <= 0 for v in
               (self.ground_size_m, self.grid_spacing_m, self.world_axis_length_m)):
            raise ValueError('World display dimensions must be finite and positive')
        rgb = tuple(self.background_rgb)
        if len(rgb) != 3 or any(not np.isfinite(v) or not 0 <= v <= 1 for v in rgb):
            raise ValueError('World background requires three values in [0,1]')
        object.__setattr__(self, 'background_rgb', rgb)

    @classmethod
    def from_configuration(cls, viewer):
        return cls(**{f.name: getattr(viewer, f.name) for f in fields(cls) if f.name != 'frame_id'})


@dataclass(frozen=True)
class CableDisplay:
    time_s: float
    provenance: Provenance
    positions_world_m: np.ndarray
    marker_indices: np.ndarray

    def __post_init__(self):
        positions = np.asarray(self.positions_world_m, dtype=np.float64)
        if positions.ndim != 2 or positions.shape[1] != 3 or len(positions) < 2:
            raise ValueError('Cable display requires at least two XYZ positions')
        object.__setattr__(self, 'positions_world_m', _array(positions, positions.shape, 'Cable positions'))
        markers = np.asarray(self.marker_indices)
        if (markers.ndim != 1 or not np.issubdtype(markers.dtype, np.integer)
                or np.any(markers < 0) or np.any(markers >= len(positions))
                or np.any(np.diff(markers) <= 0)):
            raise ValueError('Cable marker indices must be increasing integers within the mesh')
        object.__setattr__(self, 'marker_indices', immutable(markers.astype(np.int64)))
        object.__setattr__(self, 'provenance', Provenance(self.provenance))
        if not np.isfinite(self.time_s):
            raise ValueError('Cable timestamp must be finite')


@dataclass(frozen=True)
class CableTrackingDisplay:
    """Estimated material identities; null/missing nodes have no drawn geometry."""
    time_s: float
    root_world_m: np.ndarray
    positions_world_m: np.ndarray
    observed: np.ndarray
    confidence: np.ndarray
    raw_points_world_m: np.ndarray
    status: str

    def __post_init__(self):
        positions = np.asarray(self.positions_world_m)
        raw = np.asarray(self.raw_points_world_m)
        if positions.ndim != 2 or positions.shape[1] != 3 or raw.ndim != 2 or raw.shape[1] != 3:
            raise ValueError('Cable tracking requires XYZ arrays')
        for name, shape in (('root_world_m', (3,)), ('positions_world_m', positions.shape),
                            ('observed', (len(positions),)), ('confidence', (len(positions),)),
                            ('raw_points_world_m', raw.shape)):
            object.__setattr__(self, name, _array(getattr(self, name), shape, name,
                dtype=bool if name == 'observed' else np.float64))
        if not np.isfinite(self.time_s) or np.any((self.confidence < 0) | (self.confidence > 1)):
            raise ValueError('Invalid tracking timestamp/confidence')


@dataclass(frozen=True)
class RobotDisplay:
    name: str
    time_s: float
    provenance: Provenance
    drone: DroneState
    attachment_position_world_m: np.ndarray
    cable: CableDisplay | None = None
    forces: ForceTelemetry | None = None
    trajectory_epoch: int = 0
    cable_tracking: CableTrackingDisplay | None = None
    # Real flights: the acceleration the host sent this cycle (trims and feedback included) and the
    # planned reference acceleration, world frame, gravity excluded. None when not available.
    commanded_acceleration_world_mps2: np.ndarray | None = None
    desired_acceleration_world_mps2: np.ndarray | None = None

    def __post_init__(self):
        for name in ('commanded_acceleration_world_mps2', 'desired_acceleration_world_mps2'):
            if getattr(self, name) is not None:
                object.__setattr__(self, name, _array(getattr(self, name), (3,), name))
        if type(self.trajectory_epoch) is not int or self.trajectory_epoch < 0:
            raise ValueError('Trajectory epoch must be a nonnegative integer')
        if not isinstance(self.name, str) or not self.name or not np.isfinite(self.time_s):
            raise ValueError('Robot display requires a name and finite timestamp')
        drone = self.drone
        values = {}
        for name, shape in (('position_world_m', (3,)), ('quaternion_body_to_world_xyzw', (4,)),
                            ('linear_velocity_world_mps', (3,)), ('angular_velocity_body_radps', (3,)),
                            ('linear_acceleration_world_mps2', (3,))):
            value = getattr(drone, name)
            if value is None and name not in ('position_world_m', 'quaternion_body_to_world_xyzw'):
                values[name] = None
                continue  # Unavailable telemetry is explicitly absent, never zero-filled.
            values[name] = _array(value, shape, name)
        norm = np.linalg.norm(values['quaternion_body_to_world_xyzw'])
        if not np.isclose(norm, 1., atol=1e-8, rtol=0):
            raise ValueError('Display quaternion must be normalized XYZW')
        object.__setattr__(self, 'drone', replace(drone, **values))
        object.__setattr__(self, 'attachment_position_world_m',
                           _array(self.attachment_position_world_m, (3,), 'Attachment position'))
        object.__setattr__(self, 'provenance', Provenance(self.provenance))
        if self.forces is not None:
            object.__setattr__(self, 'forces', replace(self.forces, **{
                f.name: _array(getattr(self.forces, f.name), (3,), f.name) for f in fields(self.forces)}))

    # Geometry aliases keep the existing CAD and saved-motion actors reusable.
    @property
    def cable_node_positions_world_m(self):
        return None if self.cable is None else self.cable.positions_world_m

    @property
    def cable_marker_indices(self):
        return np.empty(0, dtype=np.int64) if self.cable is None else self.cable.marker_indices


@dataclass(frozen=True)
class DisplayFrame:
    time_s: float
    robots: tuple[RobotDisplay, ...]
    frame_id: str = 'world'
    source_label: str = 'Simulation flight'
    recorded: bool = False

    def __post_init__(self):
        robots = tuple(self.robots)
        if not np.isfinite(self.time_s) or not self.frame_id or not self.source_label:
            raise ValueError('Display frame requires finite time and named frame/source')
        if len({r.name for r in robots}) != len(robots):
            raise ValueError('Robot display names must be unique')
        if any(r.time_s > self.time_s + 1e-9 or
               (r.cable is not None and r.cable.time_s > self.time_s + 1e-9) or
               (r.cable_tracking is not None and r.cable_tracking.time_s > self.time_s+1e-9) for r in robots):
            raise ValueError('Robot/cable timestamps cannot exceed the display timeline')
        object.__setattr__(self, 'robots', robots)


def simulation_frame(state: SimulationState, *, name='drone', recorded=False):
    cable = CableDisplay(state.time_s, Provenance.SIMULATED,
                         state.cable_node_positions_world_m, state.cable_marker_indices)
    robot = RobotDisplay(name, state.time_s, Provenance.SIMULATED, state.drone,
                         state.attachment_position_world_m, cable, state.forces)
    return DisplayFrame(state.time_s, (robot,), recorded=recorded,
                        source_label='Recorded simulation' if recorded else 'Simulation flight')
