"""Fixed-step numerical rollouts; no file formats, rendering or training logic."""

import numpy as np

from .runtime import Simulation


def _state_values(state):
    drone = state.drone
    return {
        'drone_position_world_m': drone.position_world_m,
        'drone_quaternion_body_to_world_xyzw': drone.quaternion_body_to_world_xyzw,
        'drone_velocity_world_mps': drone.linear_velocity_world_mps,
        'drone_angular_velocity_body_radps': drone.angular_velocity_body_radps,
        'drone_acceleration_world_mps2': drone.linear_acceleration_world_mps2,
        'cable_node_positions_world_m': state.cable_node_positions_world_m,
        'cable_node_velocities_world_mps': state.cable_node_velocities_world_mps,
    }


def rollout(simulation: Simulation, steps: int) -> dict[str, np.ndarray]:
    """Reset, then collect N intervals and N+1 states without wall-clock pacing.

    Uses the simulation's command source and reset hook. No source means the
    initial position/yaw hold. Command row k acts on [time_s[k], time_s[k+1]);
    requested and delayed commands are logged separately, each exactly once.
    Acceleration is the runtime's backward difference; row 0 is a zero placeholder.
    An exception aborts the rollout rather than returning partially filled arrays.
    """
    if isinstance(steps, (bool, np.bool_)) or not isinstance(steps, (int, np.integer)) or steps < 1:
        raise ValueError('steps must be a positive integer')
    steps = int(steps)
    simulation.reset()
    initial = _state_values(simulation.state)
    arrays = {name: np.empty((steps+1, *value.shape), dtype=np.float64)
              for name, value in initial.items()}
    for name, value in initial.items():
        arrays[name][0] = value
    arrays['time_s'] = np.arange(steps+1, dtype=np.float64)*simulation.timestep_s
    arrays['command_time_s'] = arrays['time_s'][:-1].copy()
    arrays['requested_commands'] = np.empty((steps, 11), dtype=np.float64)
    arrays['applied_commands'] = np.empty((steps, 11), dtype=np.float64)
    arrays['command_columns'] = np.array(
        ['x', 'y', 'z', 'vx', 'vy', 'vz', 'ax', 'ay', 'az', 'yaw', 'yaw_rate'])
    arrays['cable_marker_indices'] = simulation.state.cable_marker_indices.copy()
    for index in range(steps):
        state = simulation.step()
        for name, value in _state_values(state).items():
            arrays[name][index+1] = value
        arrays['requested_commands'][index] = simulation.dynamics.requested_command
        arrays['applied_commands'][index] = simulation.dynamics.applied_command
    return arrays
