"""One open-loop spline action per episode; deliberately no reward or success flag."""

from dataclasses import dataclass

import numpy as np
import torch

from trajectories.bspline_torch import BatchedBSplineCommand
from trajectories.full_state import FullStateTrajectory
from simulation.batched_rollout import BatchedRollout, BatchedRolloutResult

from .configuration import TrainingConfiguration


@dataclass(frozen=True)
class EpisodeResult:
    actions: torch.Tensor
    target_positions: torch.Tensor
    control_points: torch.Tensor
    commands: torch.Tensor
    action_valid: torch.Tensor
    context_valid: torch.Tensor
    command_valid: torch.Tensor
    speed_bounds: torch.Tensor
    acceleration_bounds: torch.Tensor
    rollout: BatchedRolloutResult
    command_times_s: torch.Tensor
    command_events: torch.Tensor

    @property
    def valid(self):
        """Input/command/numerical validity, NOT physical safety or whip success."""
        return self.context_valid & self.command_valid & self.rollout.valid

    def trajectory(self, index):
        """Explicit CPU handoff to the existing CSV exporter/controller viewer.

        Includes the configured post-maneuver hold. Load with the matching scene
        initial state; a command CSV alone does not encode the initial cable pose.
        """
        if not 0 <= index < len(self.commands):
            raise IndexError('Episode index out of range')
        if not bool(self.valid[index]):
            raise ValueError('Cannot export an invalid episode')
        command = self.command_events[index].detach().cpu().numpy()
        rows = np.zeros((len(command), 14))
        rows[:, 0] = self.command_times_s.detach().cpu().numpy()
        rows[:, 1:11] = command[:, :10]
        return FullStateTrajectory(rows)


class WhipEnvironment:
    """Connect bounded spline actions to a reusable forward-only batch simulator.

    Actions are in [-1, 1], ordered as interior XYZ controls then endpoint XYZ.
    Bounds are WORLD-axis offsets from the initial tracking origin, shared by
    the learned controls and endpoint, not a workspace guarantee for the drone.
    Initial command acceleration is zero, initial velocity is the scene velocity,
    yaw stays at the initial yaw, and final command velocity/acceleration are zero.
    The target is context only until the whip definition/reward are agreed.
    """

    def __init__(self, configuration: TrainingConfiguration, simulator: BatchedRollout):
        if abs(simulator.dt-.01) > 1e-12:
            raise ValueError('Whip scoring currently requires 100 Hz feedback; command rate is independent')
        self.configuration, self.simulator = configuration, simulator
        self.spline = BatchedBSplineCommand(configuration.control_count, configuration.duration_s,
            device=simulator.hold.device, dtype=simulator.hold.dtype)
        self.lower = simulator.model.tensor(configuration.offset_lower_m)
        self.upper = simulator.model.tensor(configuration.offset_upper_m)
        self.post_hold_steps = round(configuration.post_hold_s/simulator.dt)
        self.packet_spline = (self.spline if abs(configuration.command_rate_hz-1/simulator.dt) < 1e-10 else
            BatchedBSplineCommand(configuration.control_count, configuration.duration_s,
                timestep_s=1/configuration.command_rate_hz, device=simulator.hold.device, dtype=simulator.hold.dtype))
        self.duration_s = configuration.duration_s+configuration.post_hold_s
        packet_count = int(np.ceil(self.duration_s*configuration.command_rate_hz-1e-10))
        times = np.arange(packet_count)/configuration.command_rate_hz
        self.command_time_values = np.r_[times, self.duration_s]
        self.command_times = simulator.model.tensor(self.command_time_values)
        output = np.arange(round(self.duration_s/simulator.dt)+1)*simulator.dt
        self.packet_indices = torch.as_tensor(np.searchsorted(self.command_time_values,
            output+1e-12, side='right')-1, device=simulator.hold.device)
        self.start_position, self.start_velocity, self.cable_position, self.cable_velocity = simulator.observe(simulator.initial)
        self.yaw = simulator.hold[:, 9].clone()

    def _check(self, value, shape, name):
        if (not isinstance(value, torch.Tensor) or tuple(value.shape) != shape
                or value.device != self.lower.device or value.dtype != self.lower.dtype):
            raise ValueError(f'{name} must have shape {shape}, model device and float64 dtype')

    def observation(self, target_positions):
        """Structured initial context; no policy architecture or normalization imposed."""
        self._check(target_positions, (self.simulator.batch_size, 3), 'Targets')
        return dict(drone_position=self.start_position.clone(), drone_velocity=self.start_velocity.clone(),
            drone_rotation=self.simulator.initial[1].clone(), drone_angular_velocity=self.simulator.initial[3].clone(),
            cable_positions=self.cable_position.clone(), cable_velocities=self.cable_velocity.clone(),
            target_position=target_positions.clone())

    def decode(self, actions):
        """Differentiable decoding; invalid actions are flagged, never silently clipped."""
        self._check(actions, (self.simulator.batch_size, self.configuration.action_size), 'Actions')
        action_valid = (torch.isfinite(actions) & (actions.abs() <= 1)).all(-1)
        offsets = self.lower+(actions.reshape(len(actions), -1, 3)+1)*.5*(self.upper-self.lower)
        locations = self.start_position[:, None]+offsets
        points = self.spline.control_points(self.start_position, locations[:, -1], locations[:, :-1], self.start_velocity)
        commands = self.spline.commands(points, self.yaw)
        speed, acceleration = self.spline.derivative_bounds(points)
        # Same conservative continuous-time derivative bounds as the CPU editor.
        command_valid = (action_valid & torch.isfinite(commands).flatten(1).all(-1)
            & (speed <= self.configuration.max_speed_mps+1e-10)
            & (acceleration <= self.configuration.max_acceleration_mps2+1e-10))
        return points, commands, action_valid, command_valid, speed, acceleration

    def execute(self, points, commands, admissible):
        """Execute native packets; retain 100-Hz command snapshots for scoring/UI."""
        packets = commands if self.packet_spline is self.spline else self.packet_spline.commands(points, self.yaw)
        extra = len(self.command_time_values)-packets.shape[1]
        if extra:
            packets = torch.cat((packets, packets[:, -1:].expand(-1, extra, -1)), dim=1)
        sampled = packets[:, self.packet_indices]
        if self.packet_spline is self.spline:
            # Preserve the original fixed-grid path exactly for legacy protocols.
            result = self.simulator.run(packets[:, :-1], admissible=admissible)
        else:
            result = self.simulator.run(packets, admissible=admissible,
                command_times_s=self.command_time_values, duration_s=self.duration_s)
        return sampled, packets, result

    @torch.no_grad()
    def run(self, actions, target_positions):
        self._check(target_positions, (self.simulator.batch_size, 3), 'Targets')
        points, commands, action_valid, command_valid, speed, acceleration = self.decode(actions)
        context_valid = torch.isfinite(target_positions).all(-1)
        commands, packets, result = self.execute(points, commands, command_valid & context_valid)
        return EpisodeResult(actions.detach().clone(), target_positions.detach().clone(), points,
            commands, action_valid, context_valid, command_valid, speed, acceleration, result,
            self.command_times.clone(), packets)
