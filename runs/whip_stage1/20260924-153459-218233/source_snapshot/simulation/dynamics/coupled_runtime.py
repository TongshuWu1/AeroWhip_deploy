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
from simulation.timing import TIME_EPS, integration_steps


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
        self.delay_s = float(self.model.config['controller']['command_delay_s'])
        if not math.isfinite(self.delay_s) or self.delay_s < 0:
            raise ValueError('Command delay must be finite and nonnegative')
        self.delay_steps = (round(self.delay_s/self.dt) if
            math.isclose(round(self.delay_s/self.dt)*self.dt, self.delay_s, abs_tol=TIME_EPS, rel_tol=0) else None)
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
        self._segment_graphs = {}
        self.command = self.model.tensor(self._last_command).clone()
        self.compiled = False
        if compile_physics and self.device.startswith('cuda'):
            original = self.model.controlled_substep
            try:
                self.model.controlled_substep = torch.compile(original, backend='inductor', fullgraph=True)
                self.model.controlled_substep(self.physics, self.command, self.controller, self.model.integration_dt)
                torch.cuda.synchronize()
                self.compiled = True
            except Exception as exc:
                self.model.controlled_substep = original
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
            self._segment_graphs[(self.model.substeps, self.model.integration_dt)] = self.graph

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
        self._requested_left = hold.copy()
        self._applied_command = hold.copy()
        self._pending = deque()
        self._output_index = 0
        self._held_applied = hold.copy()
        self.last_requested_events = ()
        self.last_applied_events = ()
        if hasattr(self, 'command'):
            self.command.copy_(m.tensor(hold))

    def _transition(self, duration_s=None):
        _, previous_v = self.model.tracking_pose(self.physics)
        output = self.model.step_unchecked(self.physics, self.command, self.controller, duration_s)
        _, velocity = self.model.tracking_pose(output)
        self.acceleration.copy_((velocity - previous_v) / self.dt)
        for field in fields(output):
            getattr(self.physics, field.name).copy_(getattr(output, field.name))

    @property
    def requested_command(self):
        """Requested command at the last interval's left endpoint; owned copy."""
        return self._requested_left.copy()

    @property
    def applied_command(self):
        """Delayed command at the last interval's left endpoint, not its average."""
        return self._applied_command.copy()

    def _integrate(self, duration_s):
        key = integration_steps(duration_s, self.model.physics_dt)
        graph = self._segment_graphs.get(key)
        # Periodic rates reuse a few chunk lengths. Irregular event timestamps
        # must not create an unbounded collection of large CUDA graphs.
        if graph is None and self.graph is not None and len(self._segment_graphs) < 8:
            saved = tuple(getattr(self.physics, f.name).clone() for f in fields(self.physics))
            acceleration = self.acceleration.clone()
            stream = torch.cuda.Stream()
            stream.wait_stream(torch.cuda.current_stream())
            with torch.cuda.stream(stream):
                for _ in range(3):
                    self._transition(duration_s)
            torch.cuda.current_stream().wait_stream(stream)
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph):
                self._transition(duration_s)
            for field, value in zip(fields(self.physics), saved):
                getattr(self.physics, field.name).copy_(value)
            self.acceleration.copy_(acceleration)
            self._segment_graphs[key] = graph
        if graph is None:
            self._transition(duration_s)
        else:
            graph.replay()

    @torch.no_grad()
    def step(self, command=None, *, command_events=()):
        """Advance one feedback interval; events carry absolute simulation time."""
        start, end = self._output_index*self.dt, (self._output_index+1)*self.dt
        if command is not None and len(command_events):
            raise ValueError('Supply an immediate command or timestamped events, not both')
        events = [(start, command)] if command is not None else list(command_events)
        checked, previous = [], -math.inf
        for time_s, value in events:
            time_s, value = float(time_s), np.asarray(value, dtype=np.float64)
            if (not math.isfinite(time_s) or time_s < start-TIME_EPS or time_s >= end-TIME_EPS
                    or time_s <= previous):
                raise ValueError('Command events must increase within the current feedback interval')
            if value.shape != (11,) or not np.isfinite(value).all():
                raise ValueError("Expected 11 finite P/V/A/yaw/yaw-rate values")
            if value[10] != 0:
                raise ValueError("Yaw-rate feedforward is not supported")
            checked.append((time_s, value.copy()))
            previous = time_s
        self.last_requested_events = tuple(checked)
        self._requested_left = self._last_command.copy()
        for time_s, value in checked:
            if time_s <= start+TIME_EPS:
                self._requested_left = value.copy()
            self._last_command = value.copy()
            self._pending.append((time_s+self.delay_s, value))
        applied = []
        while self._pending and self._pending[0][0] <= start+TIME_EPS:
            event = self._pending.popleft()
            self._held_applied = event[1]
            applied.append(event)
        self._applied_command = self._held_applied.copy()
        previous_v = self.model.tracking_pose(self.physics)[1].clone()
        cursor = start
        while self._pending and self._pending[0][0] < end-TIME_EPS:
            event = self._pending.popleft()
            if event[0] > cursor+TIME_EPS:
                self.command.copy_(torch.from_numpy(self._held_applied))
                self._integrate(event[0]-cursor)
            cursor, self._held_applied = max(cursor, event[0]), event[1]
            applied.append(event)
        self.command.copy_(torch.from_numpy(self._held_applied))
        self._integrate(end-cursor)
        self.acceleration.copy_((self.model.tracking_pose(self.physics)[1]-previous_v)/self.dt)
        self.last_applied_events = tuple(applied)
        self._output_index += 1

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
