"""Forward-only, GPU-resident episodes using the existing coupled integrator."""

from dataclasses import dataclass, fields
import math

import torch

from .dynamics.coupled import CoupledState
from .dynamics.coupled_control import CoupledController


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
    def __init__(self, model, initial_states, *, graph=True):
        if not model.batched or not initial_states:
            raise ValueError('Provide a batched coupled model and at least one initial state')
        self.model, self.controller = model, CoupledController(model)
        self.batch_size = len(initial_states)
        self.dt = model.dt
        delay = float(model.config['controller']['command_delay_s'])
        if not math.isfinite(delay) or delay < 0:
            raise ValueError('Command delay must be finite and nonnegative')
        self.delay_steps = round(delay/self.dt)
        if not math.isclose(self.delay_steps*self.dt, delay, abs_tol=1e-12, rel_tol=0):
            raise ValueError('Command delay must be an integer number of steps')
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

    def _advance(self, *args):
        previous, command, alive = args[:6], args[6], args[7]

        def one(*values):
            result = self.model.step_unchecked(CoupledState(*values[:-1]), values[-1], self.controller)
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
    def run(self, commands, *, admissible=None):
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
            state = output[:6]
            finite[:, k], rotations[:, k], alive = output[6:9]
            valid[:, k+1] = alive
            p, v, q, qv = output[9:]
            for history, value in zip(histories, (p, v, state[1], state[3], q, qv)):
                history[:, k+1] = value
        return BatchedRolloutResult(torch.arange(steps+1, device=commands.device, dtype=commands.dtype)*self.dt,
            requested, applied, *histories, valid, command_valid, finite, rotations,
            CoupledState(*(x.clone() for x in state)), self.model.rest.clone(), self.marker_indices.clone())
