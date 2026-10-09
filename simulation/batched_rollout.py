"""Forward-only, GPU-resident episodes using the existing coupled integrator."""

from dataclasses import dataclass, fields

import torch

from .tensor_model import ControlledState as PlantState
from controllers.full_state_controller.tensor_controller import GeometricController
from .timing import CommandSchedule, TIME_EPS


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
    final_state: PlantState
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
    """Independent canonical controller/motor episodes, with timestamped packets.

    Commands: (batch, packets, 11 or 13). Motor thrust and held wrench survive
    event/output boundaries. Output validity freezes failed lanes and must be
    used by consumers. CUDA graphs cache execution only, not another model.
    """
    @torch.no_grad()
    def __init__(self, model, initial_states, *, graph=True):
        if not model.batched or not initial_states:
            raise ValueError('Provide a batched model and initial states')
        self.model, self.controller = model, GeometricController(model)
        self.batch_size, self.dt = len(initial_states), model.dt
        self.delay_s = float(model.config['controller']['command_delay_s'])
        checked = []
        for s in initial_states:
            p,v = model.tracking_pose(s)
            q,qv = model.cable_state(s)
            physical = model.initial_state(p,rotation=s.rotation,tracking_velocity=v,
                omega_body=s.omega_body,cable_positions=q,cable_velocities=qv,
                material_angles=s.material_angles)
            checked.append(model.controller_state(physical))
        self.initial = tuple(torch.stack([getattr(s,f.name) for s in checked])
                             for f in fields(PlantState))
        self.hold = model.tensor([0.]*13).expand(self.batch_size,-1).clone()
        self.hold[:,:3] = self.observe(self.initial)[0]
        self.hold[:,9] = torch.atan2(self.initial[1][:,1,0],self.initial[1][:,0,0])
        self.marker_indices = torch.tensor(model.marker_indices,device=model.device,dtype=torch.long)
        self.graph = bool(graph and self.hold.is_cuda)
        self._graphs = {}

    def observe(self,state):
        def one(*values):
            s = PlantState(*values)
            return (*self.model.tracking_pose(s),*self.model.cable_state(s))
        return torch.vmap(one)(*state)

    def _advance(self,state,command,duration,start):
        def one(*values):
            result = self.model.step_unchecked(PlantState(*values[:-1]),values[-1],
                self.controller,duration,start_s=start)
            return tuple(getattr(result,f.name) for f in fields(result))
        return torch.vmap(one)(*state,command)

    def _chunk(self,state,command,duration,start):
        # Phase is the only absolute-time dependence. Round solely timestamp
        # roundoff, not physical intervals or event locations.
        phase = start % self.model.controller_period_s
        if abs(phase-self.model.controller_period_s) <= TIME_EPS or phase <= TIME_EPS:
            phase = 0.
        if not self.graph:
            return self._advance(state,command,duration,phase)
        key = (round(duration,14),round(phase,14))
        if key not in self._graphs:
            if len(self._graphs) >= 16:
                return self._advance(state,command,duration,phase)
            buffers = tuple(x.clone() for x in (*state,command))
            def execute():
                output = self._advance(buffers[:-1],buffers[-1],duration,phase)
                for target,value in zip(buffers[:-1],output):
                    target.copy_(value)
            stream = torch.cuda.Stream(device=command.device)
            stream.wait_stream(torch.cuda.current_stream())
            with torch.cuda.stream(stream):
                for _ in range(3):
                    execute()
            torch.cuda.current_stream().wait_stream(stream)
            captured = torch.cuda.CUDAGraph()
            with torch.cuda.graph(captured,stream=stream):
                execute()
            self._graphs[key] = buffers,captured
        buffers,captured = self._graphs[key]
        for target,value in zip(buffers,(*state,command)):
            target.copy_(value)
        captured.replay()
        # Own outputs: the next graph replay must not mutate last-valid state.
        return tuple(x.clone() for x in buffers[:-1])

    @torch.no_grad()
    def run(self,commands,*,admissible=None,command_times_s=None,duration_s=None):
        import numpy as np
        if (not isinstance(commands,torch.Tensor) or commands.ndim != 3
                or commands.shape[0] != self.batch_size or commands.shape[1] < 1
                or commands.shape[2] not in (11,13) or commands.device != self.hold.device
                or commands.dtype != self.hold.dtype):
            raise ValueError('Commands must be float64 (batch, packets, 11 or 13) on the model device')
        native = commands.detach().clone()
        native_valid = torch.isfinite(native).all(-1)
        if native.shape[-1] == 11:
            native_valid = native_valid & (native[:,:,10] == 0)
            native = torch.cat((native[:,:,:10],torch.zeros_like(native[:,:,:3])),dim=-1)
        if admissible is None:
            admissible = torch.ones(self.batch_size,device=native.device,dtype=torch.bool)
        if (admissible.shape != (self.batch_size,) or admissible.dtype != torch.bool
                or admissible.device != native.device):
            raise ValueError('Admissibility must be a batch-sized boolean tensor on the model device')
        alive = admissible & native_valid.all(-1)
        if command_times_s is None:
            if duration_s is not None:
                raise ValueError('Explicit duration requires timestamps')
            command_times_s = np.arange(native.shape[1])*self.dt
            duration_s = native.shape[1]*self.dt
        if duration_s is None:
            raise ValueError('Timestamped commands require duration')
        schedule = CommandSchedule.build(command_times_s,duration_s,self.dt,self.delay_s)
        if len(command_times_s) != native.shape[1]:
            raise ValueError('Command timestamp count mismatch')
        indices = torch.as_tensor(schedule.requested_indices,device=native.device)
        requested,command_valid = native[:,indices],native_valid[:,indices]
        safe = torch.where(alive[:,None,None],native,self.hold[:,None])
        state = tuple(x.clone() for x in self.initial)
        p,v,q,qv = self.observe(state)
        history = [[x.clone()] for x in (p,v,state[1],state[3],q,qv)]
        valid,finite_history,rotation_history,applied = [alive.clone()],[],[],[]
        for k,parts in enumerate(schedule.parts):
            row = int(schedule.applied_indices[k])
            applied.append(torch.where(alive[:,None],safe[:,row] if row >= 0 else self.hold,self.hold))
            cursor = float(schedule.time_s[k])
            finite,rotation_ok = torch.ones_like(alive),torch.ones_like(alive)
            for duration,row in parts:
                command = torch.where(alive[:,None],safe[:,row] if row >= 0 else self.hold,self.hold)
                proposed = self._chunk(state,command,duration,cursor)
                good = torch.stack([torch.isfinite(x).flatten(1).all(-1) for x in proposed]).all(0)
                R = proposed[1]
                orthogonal = (R.transpose(-1,-2)@R-self.model.eye3).abs().amax(dim=(-1,-2)) <= 1e-6
                det = (torch.linalg.cross(R[:,:,0],R[:,:,1],dim=-1)*R[:,:,2]).sum(-1)
                rot = orthogonal & ((det-1).abs() <= 1e-6)
                alive = alive & good & rot
                finite,rotation_ok = finite & good,rotation_ok & rot
                state = tuple(torch.where(alive.reshape(-1,*([1]*(x.ndim-1))),x,old)
                              for x,old in zip(proposed,state))
                cursor += duration
            p,v,q,qv = self.observe(state)
            for values,value in zip(history,(p,v,state[1],state[3],q,qv)):
                values.append(value.clone())
            valid.append(alive.clone())
            finite_history.append(finite)
            rotation_history.append(rotation_ok)
        return BatchedRolloutResult(
            torch.as_tensor(schedule.time_s,device=native.device,dtype=native.dtype),
            requested,torch.stack(applied,1),*(torch.stack(x,1) for x in history),
            torch.stack(valid,1),command_valid,torch.stack(finite_history,1),torch.stack(rotation_history,1),
            PlantState(*(x.clone() for x in state)),self.model.rest.clone(),self.marker_indices.clone())
