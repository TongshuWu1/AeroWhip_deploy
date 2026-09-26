"""Forward-only, GPU-resident episodes using the existing coupled integrator."""

from dataclasses import dataclass, fields
import math

import torch

from .dynamics.coupled import CoupledState
from .dynamics.coupled_control import CoupledController
from .timing import CommandSchedule, TIME_EPS, integration_steps


@dataclass(frozen=True)
class BatchedRolloutResult:
    """All motion is world-frame except body angular velocity; rotations are body-to-world.

    Drone P/V refer to the tracking origin, not the COM. Motion tensors are
    batch-major. Finite/rotation diagnostics describe the proposed integration
    output; state_valid is the cumulative mask to use for scoring motion.
    """
    time_s: torch.Tensor
    requested_commands: torch.Tensor
    applied_commands: torch.Tensor
    drone_positions: torch.Tensor
    drone_velocities: torch.Tensor
    drone_rotations: torch.Tensor
    drone_angular_velocities: torch.Tensor
    cable_positions: torch.Tensor
    cable_velocities: torch.Tensor
    state_valid: torch.Tensor
    command_valid: torch.Tensor
    step_finite: torch.Tensor
    step_rotation_valid: torch.Tensor
    final_state: CoupledState
    cable_rest_lengths: torch.Tensor
    cable_marker_indices: torch.Tensor

    @property
    def valid(self):
        """Numerical/input validity only; not safety, a whip, or task success."""
        return self.state_valid.all(dim=1) & self.command_valid.all(dim=1)

    @property
    def drone_interval_accelerations(self):
        """Tracking-origin velocity differences, one per integration interval."""
        return torch.diff(self.drone_velocities, dim=1)/(self.time_s[1]-self.time_s[0])

    @property
    def cable_length_errors(self):
        """Signed edge-length errors in metres; no task-specific tolerance imposed."""
        return torch.linalg.vector_norm(torch.diff(self.cable_positions, dim=2), dim=-1)-self.cable_rest_lengths


class BatchedRollout:
    """Resettable fixed-size batch with independent initial states and delay queues.

    Commands have shape (batch, intervals, 11), acting on left-endpoint intervals.
    Histories contain intervals+1 samples. Each call resets to the initial states.
    Invalid command sequences reject the entire lane. On numerical failure a
    lane freezes at its last valid state, with all subsequent state_valid=False;
    these masked samples must never be scored as real motion. Other lanes continue.
    Solver exceptions still abort the call; masks handle returned invalid states.
    This class does not implement reward, collision detection or a flight envelope.
    CUDA graphs are forward-only and this object is not safe for concurrent calls.
    """

    @torch.no_grad()
    def __init__(self, model, initial_states, *, graph=True, pack_rejected=True):
        if not model.batched or not initial_states:
            raise ValueError('Provide a batched coupled model and at least one initial state')
        self.model, self.controller = model, CoupledController(model)
        self.batch_size = len(initial_states)
        self.pack_rejected = pack_rejected
        self._packed = None
        self.dt = model.dt
        delay = float(model.config['controller']['command_delay_s'])
        if not math.isfinite(delay) or delay < 0:
            raise ValueError('Command delay must be finite and nonnegative')
        self.delay_s = delay
        self.delay_steps = (round(delay/self.dt) if
            math.isclose(round(delay/self.dt)*self.dt, delay, abs_tol=TIME_EPS, rel_tol=0) else None)
        checked = []
        for initial in initial_states:
            state = CoupledState(*(model.tensor(getattr(initial, f.name)).detach().clone()
                                   for f in fields(CoupledState)))
            p, v = model.tracking_pose(state)
            q, qv = model.cable_state(state)
            checked.append(model.initial_state(p, rotation=state.rotation, tracking_velocity=v,
                omega_body=state.omega_body, cable_positions=q, cable_velocities=qv))
        self.initial = tuple(torch.stack([getattr(s, f.name) for s in checked])
                             for f in fields(CoupledState))
        self.hold = model.tensor([0.]*11).expand(self.batch_size, -1).clone()
        self.hold[:, :3] = self.observe(self.initial)[0]
        self.hold[:, 9] = torch.atan2(self.initial[1][:, 1, 0], self.initial[1][:, 0, 0])
        self.marker_indices = torch.as_tensor(model.marker_indices, device=self.hold.device, dtype=torch.long)
        self.graph = None
        self._segment_graphs = {}
        self._schedule_cache = None
        if graph and self.hold.is_cuda:
            torch.backends.cuda.preferred_linalg_library('cusolver')
            self._inputs = tuple(x.clone() for x in self.initial)+(self.hold.clone(),
                torch.ones(self.batch_size, device=self.hold.device, dtype=torch.bool))
            stream = torch.cuda.Stream(device=self.hold.device)
            stream.wait_stream(torch.cuda.current_stream())
            with torch.cuda.stream(stream):
                for _ in range(3):
                    self._advance(*self._inputs)
            torch.cuda.current_stream().wait_stream(stream)
            self.graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(self.graph, stream=stream):
                self._outputs = self._advance(*self._inputs)

    def observe(self, state):
        def one(*values):
            s = CoupledState(*values)
            return (*self.model.tracking_pose(s), *self.model.cable_state(s))
        return torch.vmap(one)(*state)

    def _advance(self, *args, duration_s=None):
        count = len(fields(CoupledState))
        previous, command, alive = args[:count], args[count], args[count+1]

        def one(*values):
            result = self.model.step_unchecked(CoupledState(*values[:-1]), values[-1], self.controller, duration_s)
            return tuple(getattr(result, f.name) for f in fields(result))

        proposed = torch.vmap(one)(*previous, command)
        finite = torch.stack([torch.isfinite(x).flatten(1).all(-1) for x in proposed]).all(0)
        rotation = proposed[1]
        orthogonal = (rotation.transpose(-1, -2) @ rotation-self.model.eye3).abs().amax(dim=(-1, -2)) <= 1e-6
        determinant = (torch.linalg.cross(rotation[:, :, 0], rotation[:, :, 1], dim=-1)*rotation[:, :, 2]).sum(-1)
        rotation_valid = orthogonal & ((determinant-1.).abs() <= 1e-6)
        alive = alive & finite & rotation_valid
        state = tuple(torch.where(alive.reshape(-1, *([1]*(x.ndim-1))), x, old)
                      for x, old in zip(proposed, previous))
        return (*state, finite, rotation_valid, alive, *self.observe(state))

    @torch.no_grad()
    def run(self, commands, *, admissible=None, command_times_s=None, duration_s=None):
        """No CPU transfers or host-side value checks inside the time loop."""
        if (not isinstance(commands, torch.Tensor) or commands.ndim != 3
                or commands.shape[0] != self.batch_size or commands.shape[1] < 1
                or commands.shape[2] != 11 or commands.device != self.hold.device
                or commands.dtype != self.hold.dtype):
            raise ValueError('Commands must be (batch, intervals, 11), on the model device with float64 dtype')
        batch, steps, _ = commands.shape
        if admissible is None:
            admissible = torch.ones(batch, dtype=torch.bool, device=commands.device)
        if (not isinstance(admissible, torch.Tensor) or admissible.shape != (batch,) or admissible.dtype != torch.bool
                or admissible.device != commands.device):
            raise ValueError('Admissibility must be a boolean tensor of shape (batch,) on the model device')
        requested = commands.detach().clone()
        command_valid = torch.isfinite(requested).all(-1) & (requested[:, :, 10] == 0)
        alive = admissible & command_valid.all(-1)
        if command_times_s is not None or self.delay_steps is None:
            if command_times_s is None:
                import numpy as np
                command_times_s = np.arange(steps)*self.dt
                duration_s = steps*self.dt
            if duration_s is None:
                raise ValueError('Timestamped commands require an explicit rollout duration')
            return self._run_scheduled(requested, command_valid, alive, command_times_s, duration_s)
        if duration_s is not None:
            raise ValueError('Explicit duration requires command_times_s')
        if self.pack_rejected and self.graph is not None and batch > 1:
            # One host count before integration, never inside the time loop.
            # Keep nearly full batches intact to avoid packing/capture overhead.
            if int(alive.sum()) <= 7*batch//8:
                return self._run_packed(requested, command_valid, alive)
        return self._run_full(requested, command_valid, alive)

    def _chunk(self, state, command, alive, duration):
        key = integration_steps(duration, self.model.physics_dt)
        inputs = (*state, command, alive)
        if self.graph is None:
            return self._advance(*inputs, duration_s=duration)
        if key == (self.model.substeps, self.model.integration_dt):
            buffers, graph, output = self._inputs, self.graph, self._outputs
        elif key in self._segment_graphs:
            buffers, graph, output = self._segment_graphs[key]
        elif len(self._segment_graphs) < 8:
            buffers = tuple(value.clone() for value in inputs)
            stream = torch.cuda.Stream(device=command.device)
            stream.wait_stream(torch.cuda.current_stream())
            with torch.cuda.stream(stream):
                for _ in range(3):
                    self._advance(*buffers, duration_s=duration)
            torch.cuda.current_stream().wait_stream(stream)
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph, stream=stream):
                output = self._advance(*buffers, duration_s=duration)
            self._segment_graphs[key] = buffers, graph, output
        else:
            return self._advance(*inputs, duration_s=duration)
        for target, value in zip(buffers, inputs):
            target.copy_(value)
        graph.replay()
        return output

    def _run_scheduled(self, native, native_valid, alive, times, duration, *, all_rejected=False):
        """Exact event splits; histories remain on the feedback clock.

        Planning is CPU-only and cached before the integration loop. Commands,
        masks, state and reusable CUDA graphs remain on the model device.
        """
        key = (tuple(times), float(duration))
        if self._schedule_cache is None or self._schedule_cache[0] != key:
            schedule = CommandSchedule.build(times, duration, self.dt, self.delay_s)
            if len(times) != native.shape[1]:
                raise ValueError('One timestamp is required per command row')
            self._schedule_cache = key, schedule
        schedule = self._schedule_cache[1]
        if len(times) != native.shape[1]:
            raise ValueError('One timestamp is required per command row')
        steps, batch = len(schedule.parts), len(native)
        indices = torch.as_tensor(schedule.requested_indices, device=native.device)
        requested, command_valid = native[:, indices], native_valid[:, indices]
        if self.pack_rejected and self.graph is not None and batch > 1 and not all_rejected:
            if int(alive.sum()) <= 7*batch//8:
                return self._run_packed(requested, command_valid, alive,
                    native=(native, native_valid, times, duration))
        safe = torch.where(alive[:, None, None], native, self.hold[:, None])
        state = tuple(x.clone() for x in self.initial)
        p, v, q, qv = self.observe(state)
        values = (p, v, state[1], state[3], q, qv)
        histories = [x.new_empty((batch, steps+1, *x.shape[1:])) for x in values]
        for history, value in zip(histories, values):
            history[:, 0] = value
        valid = torch.empty((batch, steps+1), dtype=torch.bool, device=native.device)
        valid[:, 0] = alive
        finite, rotations = (torch.empty_like(command_valid) for _ in range(2))
        applied = torch.empty_like(requested)
        count = len(fields(CoupledState))
        frozen_chunks = {}
        for k, parts in enumerate(schedule.parts):
            left = int(schedule.applied_indices[k])
            applied[:, k] = torch.where(alive[:, None], safe[:, left] if left >= 0 else self.hold, self.hold)
            all_finite, all_rotations = torch.ones_like(alive), torch.ones_like(alive)
            for interval, row in parts:
                command = torch.where(alive[:, None], safe[:, row] if row >= 0 else self.hold, self.hold)
                chunk_key = integration_steps(interval, self.model.physics_dt)
                if all_rejected and chunk_key in frozen_chunks:
                    output = frozen_chunks[chunk_key]
                else:
                    output = self._chunk(state, command, alive, interval)
                    if all_rejected:
                        # Graph outputs are reusable buffers; own these cached values.
                        output = tuple(value.clone() for value in output)
                        frozen_chunks[chunk_key] = output
                state = output[:count]
                all_finite = all_finite & output[count]
                all_rotations = all_rotations & output[count+1]
                alive = output[count+2]
            finite[:, k], rotations[:, k], valid[:, k+1] = all_finite, all_rotations, alive
            p, v, q, qv = output[count+3:]
            for history, value in zip(histories, (p, v, state[1], state[3], q, qv)):
                history[:, k+1] = value
        return BatchedRolloutResult(torch.as_tensor(schedule.time_s, device=native.device, dtype=native.dtype),
            requested, applied, *histories, valid, command_valid, finite, rotations,
            CoupledState(*(x.clone() for x in state)), self.model.rest.clone(), self.marker_indices.clone())

    def _run_full(self, requested, command_valid, alive):
        batch, steps, _ = requested.shape
        commands = requested
        safe = torch.where(alive[:, None, None], requested, self.hold[:, None])
        delayed = torch.cat((self.hold[:, None].expand(-1, self.delay_steps, -1), safe), dim=1)[:, :steps]
        state = tuple(x.clone() for x in self.initial)
        p, v, q, qv = self.observe(state)
        initial_values = (p, v, state[1], state[3], q, qv)
        histories = [x.new_empty((batch, steps+1, *x.shape[1:])) for x in initial_values]
        for history, value in zip(histories, initial_values):
            history[:, 0] = value
        valid = torch.empty((batch, steps+1), dtype=torch.bool, device=commands.device)
        valid[:, 0] = alive
        finite, rotations = (torch.empty_like(command_valid) for _ in range(2))
        applied = torch.empty_like(requested)
        for k in range(steps):
            command = torch.where(alive[:, None], delayed[:, k], self.hold)
            applied[:, k] = command
            inputs = (*state, command, alive)
            if self.graph is None:
                output = self._advance(*inputs)
            else:
                for target, value in zip(self._inputs, inputs):
                    target.copy_(value)
                self.graph.replay()
                output = self._outputs
            count = len(fields(CoupledState))
            state = output[:count]
            finite[:, k], rotations[:, k], alive = output[count:count+3]
            valid[:, k+1] = alive
            p, v, q, qv = output[count+3:]
            for history, value in zip(histories, (p, v, state[1], state[3], q, qv)):
                history[:, k+1] = value
        return BatchedRolloutResult(torch.arange(steps+1, device=commands.device, dtype=commands.dtype)*self.dt,
            requested, applied, *histories, valid, command_valid, finite, rotations,
            CoupledState(*(x.clone() for x in state)), self.model.rest.clone(), self.marker_indices.clone())

    def _run_packed(self, requested, command_valid, alive, *, native=None):
        """Skip known-rejected episodes without resampling or changing outputs."""
        indices = alive.nonzero().flatten()
        count, steps = len(indices), requested.shape[1]
        # Rejected lanes stay at their initial state. Evaluate the proposed
        # hold step once to preserve their finite/rotation diagnostics exactly.
        frozen = (self._run_full(requested[:, :1], command_valid[:, :1], torch.zeros_like(alive))
                  if native is None else self._run_scheduled(native[0], native[1], torch.zeros_like(alive),
                                                           native[2], native[3], all_rejected=True))
        result = {}
        interval_fields = {'applied_commands', 'step_finite', 'step_rotation_valid'}
        shared_fields = {'time_s', 'cable_rest_lengths', 'cable_marker_indices'}
        for field in fields(frozen):
            name, value = field.name, getattr(frozen, field.name)
            if name == 'final_state':
                result[name] = CoupledState(*(getattr(value, f.name).clone() for f in fields(value)))
            elif name == 'time_s':
                result[name] = torch.arange(steps+1, device=requested.device, dtype=requested.dtype)*self.dt
            elif name == 'requested_commands':
                result[name] = requested
            elif name == 'command_valid':
                result[name] = command_valid
            elif name in shared_fields:
                result[name] = value.clone()
            else:
                length = steps if name in interval_fields else steps+1
                result[name] = (value[:, :1].expand(-1, length, *value.shape[2:]).clone()
                                if native is None else value.clone())
        if count:
            if self._packed is None or self._packed.batch_size != count:
                # Bound the cache to one width; an accepted-count change rebuilds it.
                self._packed = None
                initial = CoupledState(*(x[0] for x in self.initial))
                self._packed = BatchedRollout(self.model, [initial]*count, pack_rejected=False)
            # Lane identities may change even when the accepted count does not.
            for target, source in zip(self._packed.initial, self.initial):
                target.copy_(source[indices])
            self._packed.hold.copy_(self.hold[indices])
            output = (self._packed.run(requested[indices].contiguous()) if native is None else
                      self._packed.run(native[0][indices].contiguous(), command_times_s=native[2], duration_s=native[3]))
            for name, value in result.items():
                if name == 'final_state':
                    for field in fields(value):
                        getattr(value, field.name).index_copy_(0, indices, getattr(output.final_state, field.name))
                elif name not in shared_fields:
                    value.index_copy_(0, indices, getattr(output, name))
        return BatchedRolloutResult(**result)
