"""Fixed-step runtime for the calibrated coupled model and optional residuals."""

from collections import deque
from dataclasses import fields
import math
import warnings

import numpy as np
import torch

from simulation.state import (DroneState, SimulationState,
                              quaternion_xyzw_to_rotation_matrix,
                              rotation_matrix_to_quaternion_xyzw)
from .coupled import CoupledDynamics, CoupledState
from .coupled_control import CoupledController


class CoupledRuntime:
    @torch.no_grad()
    def __init__(self, configuration, *, model_config, device=None, graph=True, compile_physics=True):
        self.configuration = configuration
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        if self.device == "cpu":
            torch.set_num_threads(1)
        self.model = CoupledDynamics(model_config, device=self.device)
        self.controller = CoupledController(self.model)
        self.dt = self.model.dt
        if not math.isclose(configuration.timestep_s, self.dt, abs_tol=1e-12):
            raise ValueError("Scene and physical model timesteps must match")
        if not np.allclose(configuration.platform.attachment_offset_body_m,
                           self.model.config['attachment']['position_tracking_m'], atol=1e-12, rtol=0):
            raise ValueError("Scene and physical model attachment positions must match")
        if not np.array_equal(configuration.platform.cable_segment_lengths_m,
                              self.model.config['cable']['marker_interval_lengths_m']):
            raise ValueError("Scene and physical model cable lengths must match")
        delay = float(self.model.config['controller']['command_delay_s'])
        self.delay_steps = round(delay / self.dt)
        if delay < 0 or not math.isclose(self.delay_steps*self.dt, delay, abs_tol=1e-12):
            raise ValueError("Command delay must be a nonnegative integer number of steps")
        initial = configuration.initial_state
        R = self.model.tensor(quaternion_xyzw_to_rotation_matrix(initial.quaternion_body_to_world_xyzw))
        self._initial_physics = self.model.initial_state(
            initial.position_world_m, rotation=R,
            tracking_velocity=initial.linear_velocity_world_mps,
            omega_body=initial.angular_velocity_body_radps,
            cable_positions=initial.cable_node_positions_world_m,
            cable_velocities=initial.cable_node_velocities_world_mps)
        # Restart an experiment, not a checkpoint: the delay queue starts with
        # a position/yaw hold even when the supplied physical state is moving.
        self._initial_command = np.zeros(11)
        self._initial_command[:3] = initial.position_world_m
        self._initial_command[9] = float(np.arctan2(R[1,0].cpu(), R[0,0].cpu()))
        self.reset()
        self.graph = None
        self.command = self.model.tensor(self._last_command).clone()
        self.compiled = False
        if compile_physics and self.device.startswith('cuda'):
            original = self.model.substep
            try:
                self.model.substep = torch.compile(original, backend='inductor', fullgraph=True)
                thrust, torque = self.controller(self.physics, self.command)
                self.model.substep(self.physics, thrust, torque, self.dt/self.model.substeps)
                torch.cuda.synchronize()
                self.compiled = True
            except Exception as exc:
                self.model.substep = original
                warnings.warn(f'Physics compilation failed; using CUDA graphs without compilation: {exc}', RuntimeWarning)
        if graph and self.device.startswith('cuda'):
            torch.backends.cuda.preferred_linalg_library('cusolver')
            stream = torch.cuda.Stream()
            stream.wait_stream(torch.cuda.current_stream())
            with torch.cuda.stream(stream):
                for _ in range(3):
                    self._transition()
            torch.cuda.current_stream().wait_stream(stream)
            self.reset()
            self.graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(self.graph):
                self._transition()
            self.reset()

    @torch.no_grad()
    def reset(self):
        """Restore the accepted state in place, preserving CUDA graph buffers."""
        m = self.model
        if hasattr(self, 'physics'):
            for field in fields(self._initial_physics):
                getattr(self.physics, field.name).copy_(getattr(self._initial_physics, field.name))
            self.acceleration.zero_()
        else:
            self.physics = CoupledState(*(getattr(self._initial_physics, f.name).clone()
                                          for f in fields(self._initial_physics)))
            self.acceleration = m.tensor([0., 0., 0.])
        hold = self._initial_command.copy()
        self._last_command = hold
        self._applied_command = hold.copy()
        self._command_history = deque(hold.copy() for _ in range(self.delay_steps))
        if hasattr(self, 'command'):
            self.command.copy_(m.tensor(hold))

    def _transition(self):
        _, previous_v = self.model.tracking_pose(self.physics)
        output = self.model.step_unchecked(self.physics, self.command, self.controller)
        _, velocity = self.model.tracking_pose(output)
        self.acceleration.copy_((velocity - previous_v) / self.dt)
        for field in fields(output):
            getattr(self.physics, field.name).copy_(getattr(output, field.name))

    @property
    def requested_command(self):
        """Latest requested P/V/A/yaw/yaw-rate command; return an owned CPU copy."""
        return self._last_command.copy()

    @property
    def applied_command(self):
        """Command used for the last completed interval, after the delay queue."""
        return self._applied_command.copy()

    @torch.no_grad()
    def step(self, command=None):
        if command is not None:
            command = np.asarray(command, dtype=np.float64)
            if command.shape != (11,) or not np.isfinite(command).all():
                raise ValueError("Expected 11 finite P/V/A/yaw/yaw-rate values")
            if command[10] != 0:
                raise ValueError("Yaw-rate feedforward is not supported")
            self._last_command = command.copy()
        self._command_history.append(self._last_command)
        delayed = self._command_history.popleft()
        self.command.copy_(torch.from_numpy(delayed))
        if self.graph is not None:
            self.graph.replay()
        else:
            self._transition()
        self._applied_command = delayed.copy()

    @torch.no_grad()
    def snapshot(self, time_s):
        state = self.physics
        p, v = self.model.tracking_pose(state)
        q, qv = self.model.cable_state(state)
        # One packed device-to-host transfer per displayed snapshot.
        data = torch.cat((p, v, state.rotation.flatten(), state.omega_body,
                          q.flatten(), qv.flatten(), self.acceleration)).cpu().numpy()
        if not np.isfinite(data).all():
            raise FloatingPointError("Coupled dynamics produced a nonfinite state")
        rotation = data[6:15].reshape(3,3)
        if not np.allclose(rotation.T @ rotation, np.eye(3), atol=1e-6):
            raise FloatingPointError("Coupled dynamics produced an invalid rotation")
        n = q.numel()
        points = data[18:18+n].reshape(-1,3).copy()
        return SimulationState(time_s, DroneState(data[:3].copy(),
            rotation_matrix_to_quaternion_xyzw(rotation), data[3:6].copy(), data[15:18].copy(),
            data[-3:].copy()), points[0].copy(), points, data[18+n:18+2*n].reshape(-1,3).copy(),
            self.model.marker_indices.copy())
