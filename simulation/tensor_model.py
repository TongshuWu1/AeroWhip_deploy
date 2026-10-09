"""Torch tensor bindings to C++ physics for fitting and batched research.

The only Python computations here pack inputs and invoke controller callbacks.
Forces, constraints, SO(3), implicit iterations and projection live in tensor_physics.cpp.
"""
import math
from contextlib import contextmanager
from dataclasses import dataclass
from types import FunctionType
import torch

from simulation.model import Model, PlantState
from simulation.timing import integration_steps
from simulation.actuator import Actuator

try:
    from simulation import _tensor_native
except ImportError as exc:
    raise RuntimeError('Cannot load C++ tensor physics: '+str(exc)+
                       '. Missing builds require build_engine.py; security-policy blocks require administrator review.') from exc
if _tensor_native.api_version != 9:
    raise RuntimeError('Tensor engine version mismatch. Rebuild with build_engine.py.')

ops = torch.ops.aerowhip


def state_values(state):
    return [state.position_com, state.rotation, state.velocity_com, state.omega_body,
            state.free_positions, state.free_velocities, state.material_angles]


@dataclass(frozen=True)
class IntegratedState(PlantState):
    previous_state: object
    previous_step_s: object


def integration_values(state):
    physical = state_values(state)
    if isinstance(state, IntegratedState):
        return [*physical, state.previous_state, state.previous_step_s]
    # Explicit startup only for a newly supplied physical state, never hidden
    # model-owned history. Leading batch dimensions are preserved.
    packed = torch.cat((physical[0], physical[1].flatten(-2), physical[2], physical[3],
                        physical[4].flatten(-2), physical[5].flatten(-2), physical[6]), dim=-1)
    return [*physical, packed.clone(), physical[0].new_zeros((*physical[0].shape[:-1], 1))]


@dataclass(frozen=True)
class ControlledState(IntegratedState):
    motor_thrust_n: object
    held_wrench: object


def controlled_values(state):
    return [*integration_values(state), state.motor_thrust_n, state.held_wrench]


def _batched_implicit_iteration(state, parameters, wrench, h, iterations,
                                weights, specification, work):
    """Compile one C++ Picard iteration, not an unrolled twenty-iteration graph."""
    def one(*values):
        return ops.implicit_iteration(list(values[:9]),parameters,values[9][0],
            values[9][1:],h,iterations,weights,specification,list(values[10:]),False)
    return torch.vmap(one)(*state,wrench,*work)


class TensorModel:
    def __init__(self, config, *, device='cpu', batched=False, residual=None):
        self.metadata = Model(config, load_weights=False)
        self.config, self.device, self.batched = self.metadata.config, device, batched
        for name in ('mass', 'inertia', 'com_tracking', 'attachment_tracking', 'arm', 'gravity',
                     'rest', 'node_mass', 'ei', 'drag', 'root_tangent_body', 'cb', 'gj', 'ct'):
            setattr(self, name, self.tensor(getattr(self.metadata, name)))
        for name in ('dt', 'physics_dt', 'iterations', 'n', 'marker_indices'):
            setattr(self, name, getattr(self.metadata, name))
        self.size = 6+3*(self.n-1)
        self.substeps, self.integration_dt = integration_steps(self.dt, self.physics_dt)
        self.eye3 = torch.eye(3, dtype=torch.float64, device=device)
        self._geometry = ops.setup(self._parameters()[:9])
        self.residual = residual
        if residual is None and self.config['residual']['status'] == 'fitted':
            from fitting.residual_network import load_residual
            self.residual = load_residual(self.config['residual'], device)
        self.actuator = Actuator(self.config['actuator'])
        self.actuator_parameters = self.actuator.tensor_parameters(device=device)
        self.controller_period_s = self.config['controller']['controller_period_s']
        self.controller_substeps, self.controller_integration_dt = integration_steps(
            self.controller_period_s,self.physics_dt)
        self._motor_times = {(self.controller_substeps,self.controller_integration_dt): self.tensor(
            [i*self.controller_integration_dt for i in range(self.controller_substeps)])}
        self._batch_implicit_iteration = _batched_implicit_iteration
        self._capture_warmup = False
        ratio = self.dt/self.controller_period_s
        if not math.isclose(ratio, round(ratio), rel_tol=0, abs_tol=1e-10):
            raise ValueError('Tensor output timestep must be a whole number of controller periods')

    def compile_batched_physics(self):
        """Compile one implicit iteration; leave convergence and clocks outside."""
        if not self.batched or not self.rest.is_cuda:
            raise ValueError('Compiled batched physics requires a CUDA batched model')
        # Avoid the default MAGMA LU dispatch above 16 lanes during CUDA capture.
        torch.backends.cuda.preferred_linalg_library('cusolver')
        if self._batch_implicit_iteration is _batched_implicit_iteration:
            # PyTorch 2.11 keys its specialization cache by code object. Give
            # this model its own entry so physical/residual fits and different
            # meshes do not consume each other's per-function guard budget.
            # This copies code identity, not equations; no guard is disabled.
            kernel = FunctionType(_batched_implicit_iteration.__code__.replace(),
                                  _batched_implicit_iteration.__globals__)
            self._batch_implicit_iteration = torch.compile(kernel,
                backend='inductor',fullgraph=True,options={'triton.cudagraphs':False})

    @contextmanager
    def cuda_graph_warmup(self):
        """Exercise the captured iteration schedule before CUDA capture.

        Even a converged first iteration can change work-tensor strides. Later
        iterations must compile outside capture too. C++ convergence masks keep
        the first accepted endpoint; this does not loosen any solver criterion.
        """
        previous = self._capture_warmup
        self._capture_warmup = True
        try:
            yield
        finally:
            self._capture_warmup = previous

    def _batch_motor_substep(self, state, initial, target, actuator, parameters,
                              h, start, iterations, weights, specification):
        # Motor response and all physical/solver arithmetic remain C++.
        wrench = torch.vmap(lambda a,b:ops.motor_wrench(
            ops.motor_response_elapsed(a,b,actuator,start+h),actuator))(initial,target)
        work = torch.vmap(lambda *s:ops.implicit_begin(list(s)))(*state)
        for _ in range(20):
            work = self._batch_implicit_iteration(state,parameters,wrench,h,
                iterations,weights,specification,work)
            # During outer CUDA graph capture no host read is legal. Freeze
            # each lane at its first converged endpoint through the full cap.
            capturing = self.rest.is_cuda and torch.cuda.is_current_stream_capturing()
            if not capturing and not self._capture_warmup and bool(work[1].all()):
                break
        return torch.vmap(lambda *v:ops.implicit_finish(list(v[:9]),list(v[9:]),h))(*state,*work)

    def controller_state(self, state):
        """Explicit provisional motor initialization; never infer motors from mocap."""
        if isinstance(state, ControlledState):
            return state
        return ControlledState(*integration_values(state),
            self.tensor(self.config['actuator']['initial_thrust_n']).clone(), self.tensor([0.]*4))

    def tensor(self, value):
        if value is None:
            raise ValueError('Physical parameters must be supplied')
        return torch.as_tensor(value, dtype=torch.float64, device=self.device)

    def _parameters(self):
        return [self.mass, self.inertia, self.com_tracking, self.arm, self.gravity,
                self.rest, self.node_mass, self.ei, self.drag, *getattr(self, '_geometry', []),
                self.root_tangent_body, self.cb, self.gj, self.ct]

    def _residual_parameters(self):
        r = self.residual
        if r is None:
            return [], []
        return r.tensor_parameters()

    def initial_state(self, *args, **kwargs):
        state = self.metadata.initial_state(*args, **kwargs)
        return PlantState(*(self.tensor(v).clone() for v in state_values(state)))

    def tracking_pose(self, state):
        return tuple(ops.observe(state_values(state), self._parameters())[:2])

    def cable_state(self, state):
        return tuple(ops.observe(state_values(state), self._parameters())[2:])

    def _acceleration(self, state, thrust_n, torque_body_nm):
        weights, specification = self._residual_parameters()
        return ops.acceleration(state_values(state), self._parameters(), self.tensor(thrust_n),
            self.tensor(torque_body_nm), weights, specification,
            self.rest.is_cuda and torch.compiler.is_compiling())

    def aerodynamic_accelerations(self, state, residual=None):
        r = self.residual if residual is None else residual
        weights, spec = ([],[]) if r is None else r.tensor_parameters()
        return ops.aerodynamic_accelerations(state_values(state), self._parameters(), weights, spec)

    def substep(self, state, thrust_n, torque_body_nm, h):
        weights, specification = self._residual_parameters()
        return IntegratedState(*ops.implicit_step(integration_values(state), self._parameters(),
            self.tensor(thrust_n), self.tensor(torque_body_nm), h, self.iterations,
            weights, specification, False))

    def step(self, state, command, controller):
        command = self.tensor(command)
        if (command.shape not in ((11,), (13,)) or not bool(torch.isfinite(command).all())
                or (command.shape == (11,) and float(command[10]) != 0)):
            raise ValueError('Expected finite P/V/A/yaw with three body rates or legacy zero yaw rate')
        state = self.controller_state(state)
        return self.step_unchecked(state, command, controller)

    def step_unchecked(self, state, command, controller, duration_s=None, *, start_s=0., batch=False):
        """Sample controller on its absolute clock; retain motors and held wrench.

        Fitting supplies already-delayed, held packets. Arbitrary event schedules
        pass start_s explicitly; no new feedback tick is invented at an event.
        """
        end = start_s+(self.dt if duration_s is None else duration_s)
        cursor = start_s
        period = self.controller_period_s
        while cursor < end-1e-12:
            index = round(cursor/period)
            wrench = state.held_wrench
            if abs(cursor-index*period) <= 1e-12:
                if batch:
                    def control_one(*values):
                        return controller(ControlledState(*values[:-1]),values[-1])
                    thrust,torque = torch.vmap(control_one)(*controlled_values(state),command)
                else:
                    thrust,torque = controller(state,command)
                wrench = torch.cat((thrust.unsqueeze(-1),torque),dim=-1)
                next_tick = (index+1)*period
            else:
                next_tick = (math.floor(cursor/period)+1)*period
            until = min(end,next_tick)
            plant,motors = self.advance_motors_unchecked(state,state.motor_thrust_n,wrench,
                                                        self.actuator_parameters,until-cursor,batch=batch)
            state = ControlledState(*integration_values(plant),motors,wrench)
            cursor = until
        return state

    def advance_actuation(self, state, thrust_n, torque_body_nm, duration_s):
        thrust, torque = self.tensor(thrust_n), self.tensor(torque_body_nm)
        if thrust.shape != () or torque.shape != (3,) or not bool(torch.isfinite(thrust)) or not bool(torch.isfinite(torque).all()) or float(thrust) < 0:
            raise ValueError('Provide finite nonnegative scalar thrust and body XYZ torque')
        return self.advance_actuation_unchecked(state, thrust, torque, duration_s)

    def advance_actuation_unchecked(self, state, thrust_n, torque_body_nm, duration_s):
        count, h = integration_steps(duration_s, self.physics_dt)
        for _ in range(count):
            state = self.substep(state, thrust_n, torque_body_nm, h)
        return state

    def advance_motors(self, state, motor_thrust_n, requested_wrench, actuator, duration_s):
        """Held requested wrench through four motors; returns plant and motor state.

        Plant primitive for diagnostics. Application/fitting rollouts use the
        sampled geometric controller above; there is no ideal-actuator mode.
        """
        motors, requested = self.tensor(motor_thrust_n), self.tensor(requested_wrench)
        if (motors.shape != (4,) or requested.shape != (4,)
                or not bool(torch.isfinite(requested).all()) or float(requested[0]) < 0):
            raise ValueError('Supply four motor thrusts and a finite nonnegative-collective requested wrench')
        actuator.native.validate(motors.detach().cpu().numpy())
        return self.advance_motors_unchecked(state, motors, requested,
                                             actuator.tensor_parameters(device=self.device), duration_s)

    def advance_motors_unchecked(self, state, motor_thrust_n, requested_wrench,
                                actuator_parameters, duration_s, *, batch=False):
        count, h = ((self.controller_substeps,self.controller_integration_dt)
                    if duration_s == self.controller_period_s else integration_steps(duration_s,self.physics_dt))
        allocate = lambda request: ops.motor_allocate(request,actuator_parameters)[1]
        target = torch.vmap(allocate)(requested_wrench) if batch else allocate(requested_wrench)
        weights, specification = self._residual_parameters()
        # Warmup also prepares off-tick intervals before CUDA graph capture.
        # Retain these constants for the model/graph lifetime: recreating them
        # would copy pageable host memory during capture; evicting them could
        # invalidate a previously captured graph's input storage.
        key = (count,h)
        if key not in self._motor_times:
            self._motor_times[key] = self.tensor([i*h for i in range(count)])
        times = self._motor_times[key]
        for i in range(count):
            args = (integration_values(state),motor_thrust_n,target,actuator_parameters,
                    self._parameters(),h,times[i],self.iterations,weights,specification)
            values = self._batch_motor_substep(*args) if batch else ops.motor_substep(*args,False)
            state = IntegratedState(*values)
        response = lambda initial,target: ops.motor_response(initial,target,actuator_parameters,duration_s)
        motors = torch.vmap(response)(motor_thrust_n,target) if batch else response(motor_thrust_n,target)
        return state, motors
