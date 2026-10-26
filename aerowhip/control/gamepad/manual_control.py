"""Fixed-step gamepad-to-reference integration, independent of input hardware."""

from dataclasses import dataclass
import math

import numpy as np

from aerowhip.physics.controller_interface import ControlInput
from aerowhip.physics.state import quaternion_xyzw_to_rotation_matrix


@dataclass(frozen=True)
class ManualLimits:
    # Gentle simulator defaults, not fitted physical/controller parameters.
    horizontal_speed_mps: float = 0.30
    vertical_speed_mps: float = 0.20
    horizontal_acceleration_mps2: float = 0.50
    vertical_acceleration_mps2: float = 0.40
    yaw_speed_radps: float = math.pi / 6
    yaw_acceleration_radps2: float = math.pi / 3


def _approach(current, target, maximum_change):
    change = target - current
    return current + change * min(1.0, maximum_change / max(float(np.linalg.norm(change)), 1e-15))


class ManualReference:
    def __init__(self, position, limits: ManualLimits = ManualLimits()) -> None:
        self.limits = limits
        self.position = np.array(position, dtype=np.float64, copy=True)
        self.velocity = np.zeros(3)
        self.yaw = 0.0
        self.yaw_rate = 0.0

    def advance(self, controls: ControlInput, dt: float) -> np.ndarray:
        """Return P/V/A/yaw at the interval start, then advance reference time.

        Releasing/disconnecting brakes the reference before holding its final
        position. The controller has no yaw-rate feedforward: yaw motion changes its angle
        target only. Physical response can lag/overshoot this reference.
        """
        axes = np.array((controls.x, controls.y, controls.z, controls.yaw), dtype=np.float64)
        if not np.isfinite(axes).all() or not math.isfinite(dt) or dt <= 0:
            raise ValueError("Manual input and positive timestep must be finite")
        axes = np.clip(axes, -1, 1)
        axes[:2] /= max(1.0, np.linalg.norm(axes[:2]))
        limits = self.limits
        next_velocity = np.empty(3)
        next_velocity[:2] = _approach(self.velocity[:2], axes[:2] * limits.horizontal_speed_mps,
                                     limits.horizontal_acceleration_mps2 * dt)
        next_velocity[2] = _approach(self.velocity[2], axes[2] * limits.vertical_speed_mps,
                                    limits.vertical_acceleration_mps2 * dt)
        next_yaw_rate = float(_approach(self.yaw_rate, axes[3] * limits.yaw_speed_radps,
                                       limits.yaw_acceleration_radps2 * dt))
        command = np.zeros(11)
        command[:3] = self.position
        command[3:6] = self.velocity
        command[6:9] = (next_velocity - self.velocity) / dt
        command[9] = self.yaw
        self.position += 0.5 * (self.velocity + next_velocity) * dt
        self.yaw = (self.yaw + 0.5 * (self.yaw_rate + next_yaw_rate) * dt + math.pi) % (2 * math.pi) - math.pi
        self.velocity = next_velocity
        self.yaw_rate = next_yaw_rate
        return command


class GamepadCommandSource:
    """Own manual input/reference state outside the simulation's stepping loop."""

    def __init__(self, initial_state, timestep_s):
        self.initial_state = initial_state
        self.timestep_s = timestep_s
        self.reset()

    def reset(self):
        self.controls = ControlInput()
        self.reference = ManualReference(self.initial_state.position_world_m)
        rotation = quaternion_xyzw_to_rotation_matrix(
            self.initial_state.quaternion_body_to_world_xyzw)
        self.reference.yaw = float(np.arctan2(rotation[1, 0], rotation[0, 0]))

    def __call__(self, step_index):
        return self.reference.advance(self.controls, self.timestep_s)
