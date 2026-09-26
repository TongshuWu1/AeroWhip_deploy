from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

FloatArray = NDArray[np.float64]


def quaternion_xyzw_to_rotation_matrix(quaternion_xyzw: FloatArray) -> FloatArray:
    quaternion = np.asarray(quaternion_xyzw, dtype=np.float64)
    if quaternion.shape != (4,):
        raise ValueError("quaternion_xyzw must have shape (4,)")

    norm = float(np.linalg.norm(quaternion))
    if norm <= 1e-12:
        raise ValueError("quaternion_xyzw must have non-zero norm")

    x, y, z, w = quaternion / norm
    return np.array(
        [
            [1.0 - 2.0 * (y * y + z * z), 2.0 * (x * y - z * w), 2.0 * (x * z + y * w)],
            [2.0 * (x * y + z * w), 1.0 - 2.0 * (x * x + z * z), 2.0 * (y * z - x * w)],
            [2.0 * (x * z - y * w), 2.0 * (y * z + x * w), 1.0 - 2.0 * (x * x + y * y)],
        ],
        dtype=np.float64,
    )


@dataclass(frozen=True)
class DroneState:
    position_world_m: FloatArray
    quaternion_body_to_world_xyzw: FloatArray
    linear_velocity_world_mps: FloatArray
    angular_velocity_body_radps: FloatArray
    # Net world-frame acceleration (dv/dt), not accelerometer specific force.
    # None for geometry-only initialization without a dynamics evaluation.
    linear_acceleration_world_mps2: FloatArray | None = None

    @property
    def rotation_body_to_world(self) -> FloatArray:
        return quaternion_xyzw_to_rotation_matrix(
            self.quaternion_body_to_world_xyzw
        )


@dataclass(frozen=True)
class SimulationState:
    time_s: float
    drone: DroneState
    attachment_position_world_m: FloatArray
    cable_node_positions_world_m: FloatArray
    cable_node_velocities_world_mps: FloatArray
    cable_marker_indices: NDArray[np.int64]


def rotation_matrix_to_quaternion_xyzw(rotation: FloatArray) -> FloatArray:
    """Convert a proper rotation using a stable largest-component branch."""
    r = rotation
    candidates = np.array([1+r[0,0]-r[1,1]-r[2,2], 1-r[0,0]+r[1,1]-r[2,2],
                           1-r[0,0]-r[1,1]+r[2,2], 1+np.trace(r)])
    index = int(np.argmax(candidates))
    q = np.zeros(4)
    q[index] = .5 * np.sqrt(max(0., candidates[index]))
    denominator = 4 * q[index]
    if index == 3:
        q[:3] = [r[2,1]-r[1,2], r[0,2]-r[2,0], r[1,0]-r[0,1]]
        q[:3] /= denominator
    else:
        j, k = (index+1) % 3, (index+2) % 3
        q[j] = (r[j,index]+r[index,j]) / denominator
        q[k] = (r[k,index]+r[index,k]) / denominator
        q[3] = (r[k,j]-r[j,k]) / denominator
    return q / np.linalg.norm(q)
