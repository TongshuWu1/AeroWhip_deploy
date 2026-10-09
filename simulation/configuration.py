from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import tomllib

import numpy as np
from numpy.typing import NDArray
from project_paths import project_path
from simulation.model_configuration import DEFAULT_MODEL


FloatArray = NDArray[np.float64]


@dataclass(frozen=True)
class InitialStateConfiguration:
    position_world_m: FloatArray
    quaternion_body_to_world_xyzw: FloatArray
    linear_velocity_world_mps: FloatArray
    angular_velocity_body_radps: FloatArray
    # Full simulation mesh, including attachment node 0; not just markers.
    # Omitted positions hang vertically; omitted velocities equal root velocity.
    cable_node_positions_world_m: FloatArray | None = None
    cable_node_velocities_world_mps: FloatArray | None = None
    # Relative to the root-transported frame; omitted means explicitly untwisted.
    cable_material_angles_rad: FloatArray | None = None


@dataclass(frozen=True)
class PlatformConfiguration:
    attachment_offset_body_m: FloatArray
    cable_segment_lengths_m: FloatArray


@dataclass(frozen=True)
class ViewerConfiguration:
    window_size_px: tuple[int, int]
    start_maximized: bool
    background_rgb: tuple[float, float, float]
    ground_z_m: float
    ground_size_m: float
    grid_spacing_m: float
    world_axis_length_m: float
    body_axis_length_m: float
    plate_gap_m: float
    cable_attachment_drop_m: float
    rotor_radius_m: float
    visual_rotor_rpm: float
    cable_radius_m: float
    cable_marker_radius_m: float
    # Optional world-space visual marker, not a physics or task parameter.
    target_position_world_m: FloatArray | None = None


@dataclass(frozen=True)
class SimulationConfiguration:
    initial_state: InitialStateConfiguration
    platform: PlatformConfiguration
    viewer: ViewerConfiguration
    timestep_s: float
    physics_backend: str = 'native'


def _vector(table: dict[str, object], key: str, length: int) -> FloatArray:
    try:
        value = np.asarray(table[key], dtype=np.float64)
    except KeyError as exc:
        raise ValueError(f"Missing configuration value: {key}") from exc

    if value.shape != (length,):
        raise ValueError(f"{key} must contain exactly {length} values")
    if not np.all(np.isfinite(value)):
        raise ValueError(f"{key} must contain only finite values")
    return value


def _positive_vector(table: dict[str, object], key: str) -> FloatArray:
    try:
        value = np.asarray(table[key], dtype=np.float64)
    except KeyError as exc:
        raise ValueError(f"Missing configuration value: {key}") from exc

    if value.ndim != 1 or value.size == 0:
        raise ValueError(f"{key} must be a non-empty one-dimensional list")
    if not np.all(np.isfinite(value)) or np.any(value <= 0.0):
        raise ValueError(f"{key} must contain positive finite values")
    return value


def _optional_node_array(table: dict[str, object], key: str) -> FloatArray | None:
    if key not in table:
        return None
    value = np.asarray(table[key], dtype=np.float64)
    if value.ndim != 2 or value.shape[1] != 3 or len(value) < 2:
        raise ValueError(f"{key} must contain XYZ rows including attachment node 0")
    if not np.isfinite(value).all():
        raise ValueError(f"{key} must contain only finite values")
    # Model-dependent node count and physical constraints are checked at startup.
    return value


def _positive_float(table: dict[str, object], key: str) -> float:
    try:
        value = float(table[key])
    except KeyError as exc:
        raise ValueError(f"Missing configuration value: {key}") from exc
    if not np.isfinite(value) or value <= 0.0:
        raise ValueError(f"{key} must be a positive finite value")
    return value


def load_configuration(path: str | Path) -> SimulationConfiguration:
    config_path = project_path(path)
    with config_path.open("rb") as config_file:
        raw = tomllib.load(config_file)

    try:
        initial = raw["initial_state"]
        platform = raw["platform"]
        viewer = raw["viewer"]
        simulation = raw["simulation"]
    except KeyError as exc:
        raise ValueError(f"Missing configuration section: {exc.args[0]}") from exc

    if 'cable_spin_rates_radps' in initial:
        raise ValueError('Material angles evolve by overdamped balance; independent spin rates are not state variables')

    backend = simulation.get('backend', 'native')
    if backend != 'native':
        raise ValueError('simulation.backend must be native; the old Python simulator was removed')

    window_size = _vector(viewer, "window_size_px", 2)
    if np.any(window_size <= 0.0) or np.any(window_size != np.floor(window_size)):
        raise ValueError("window_size_px must contain two positive integers")

    background = _vector(viewer, "background_rgb", 3)
    if np.any(background < 0.0) or np.any(background > 1.0):
        raise ValueError("background_rgb values must lie in [0, 1]")

    start_maximized = viewer["start_maximized"]
    if not isinstance(start_maximized, bool):
        raise ValueError("start_maximized must be true or false")

    quaternion = _vector(initial, "quaternion_body_to_world_xyzw", 4)
    quaternion_norm = float(np.linalg.norm(quaternion))
    if quaternion_norm <= 1e-12:
        raise ValueError("quaternion_body_to_world_xyzw must have non-zero norm")

    return SimulationConfiguration(
        timestep_s=_positive_float(simulation, "timestep_s"),
        physics_backend=backend,
        initial_state=InitialStateConfiguration(
            position_world_m=_vector(initial, "position_world_m", 3),
            quaternion_body_to_world_xyzw=quaternion / quaternion_norm,
            linear_velocity_world_mps=_vector(
                initial, "linear_velocity_world_mps", 3
            ),
            angular_velocity_body_radps=_vector(
                initial, "angular_velocity_body_radps", 3
            ),
            cable_node_positions_world_m=_optional_node_array(
                initial, "cable_node_positions_world_m"
            ),
            cable_node_velocities_world_mps=_optional_node_array(
                initial, "cable_node_velocities_world_mps"
            ),
            cable_material_angles_rad=(np.asarray(initial['cable_material_angles_rad'], dtype=np.float64)
                                       if 'cable_material_angles_rad' in initial else None),
        ),
        platform=PlatformConfiguration(
            attachment_offset_body_m=_vector(
                platform, "attachment_offset_body_m", 3
            ),
            cable_segment_lengths_m=_positive_vector(
                platform, "cable_segment_lengths_m"
            ),
        ),
        viewer=ViewerConfiguration(
            window_size_px=(int(window_size[0]), int(window_size[1])),
            start_maximized=start_maximized,
            background_rgb=tuple(float(value) for value in background),
            ground_z_m=float(viewer["ground_z_m"]),
            ground_size_m=_positive_float(viewer, "ground_size_m"),
            grid_spacing_m=_positive_float(viewer, "grid_spacing_m"),
            world_axis_length_m=_positive_float(viewer, "world_axis_length_m"),
            body_axis_length_m=_positive_float(viewer, "body_axis_length_m"),
            plate_gap_m=_positive_float(viewer, "plate_gap_m"),
            cable_attachment_drop_m=_positive_float(viewer, "cable_attachment_drop_m"),
            rotor_radius_m=_positive_float(viewer, "rotor_radius_m"),
            visual_rotor_rpm=_positive_float(viewer, "visual_rotor_rpm"),
            cable_radius_m=_positive_float(viewer, "cable_radius_m"),
            cable_marker_radius_m=_positive_float(
                viewer, "cable_marker_radius_m"
            ),
            target_position_world_m=(
                _vector(viewer, "target_position_world_m", 3)
                if "target_position_world_m" in viewer else None
            ),
        ),
    )
