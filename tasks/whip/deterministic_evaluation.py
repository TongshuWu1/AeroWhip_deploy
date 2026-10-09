"""One fixed-state rollout; retain the checkpoint's CUDA actor batch arithmetic.

Reuse scalar C++ DER/motor integration and the SAME tensor controller, recording,
observation, safety and scorer. This is not a stochastic robustness evaluator.
"""
import numpy as np
import torch

from simulation.model import native_module, aerodynamic_weights, unpack_state
from simulation.tensor_model import ControlledState, controlled_values, state_values
from .sac_environment import Stage1Environment


class NativeNumericalFailure(RuntimeError):
    """A rejected native plant endpoint, not a CUDA/controller/programming error."""


class NativeStepper:
    def __init__(self, model):
        self.model = m = model
        host = lambda v: v.detach().cpu().numpy()
        self.engine = native_module().Engine(dict(mass=float(m.mass), inertia=host(m.inertia),
            arm=host(m.arm), gravity=host(m.gravity), rest=host(m.rest), node_mass=host(m.node_mass),
            root_tangent_body=host(m.root_tangent_body), stiffness=float(m.ei), drag=float(m.drag),
            bending_damping=float(m.cb), torsion_stiffness=float(m.gj), torsion_damping=float(m.ct),
            maximum_step=m.physics_dt, projections=m.iterations,
            residual=aerodynamic_weights(m.residual)))

    @staticmethod
    def native_call(callback,*args):
        try:
            return callback(*args)
        except (RuntimeError,ValueError) as exc:
            raise NativeNumericalFailure(str(exc)) from exc

    @torch.no_grad()
    def __call__(self, state, command, controller):
        if state.position_com.shape != (1,3):
            raise ValueError('Native deterministic evaluation requires one physical lane')
        m = self.model
        s = ControlledState(*(v[0] for v in controlled_values(state)))
        packed = torch.cat([v.flatten() for v in state_values(s)])
        count = len(packed)
        # Restore ALL BDF2 history, not just the current centerline/vehicle pose.
        self.native_call(self.engine.restore,
            torch.cat((packed,s.previous_state,s.previous_step_s)).cpu().numpy())
        motors = s.motor_thrust_n.cpu().numpy()
        for _ in range(round(m.dt/m.controller_period_s)):
            thrust, torque = controller(s,command[0])
            wrench = torch.cat((thrust[None],torque))
            motors = np.asarray(self.native_call(self.engine.advance_motors,m.controller_period_s,
                wrench.cpu().numpy(),m.actuator.native,motors))
            checkpoint = self.engine.checkpoint()
            physical = unpack_state(checkpoint[:count])
            s = ControlledState(*(m.tensor(v) for v in state_values(physical)),
                m.tensor(checkpoint[count:2*count]),m.tensor(checkpoint[-1:]),
                m.tensor(motors),wrench)
        return ControlledState(*(v[None] for v in controlled_values(s)))


class DeterministicEvaluationEnvironment(Stage1Environment):
    def __init__(self, settings, snapshot, configuration, scene, *, progress=None):
        self.policy_batch = settings.get('evaluation_environments',settings['parallel_environments'])
        super().__init__(dict(settings,parallel_environments=1),snapshot,configuration,scene,
                         progress=progress,capture=False)
        self.native_stepper = NativeStepper(self.model)
        self.numerical_failure = None

    @torch.no_grad()
    def reset(self):
        self.numerical_failure = None
        return super().reset()

    @torch.no_grad()
    def _advance(self, applied):
        try:
            return self.native_stepper(self.state,applied,self.controller)
        except NativeNumericalFailure as exc:
            # Native convergence/geometry failures throw instead of returning
            # the tensor engine's NaN endpoint. Retain the same validity gate:
            # never accept or score a fabricated continuation after failure.
            self.numerical_failure = str(exc)
            values = [v.clone() for v in controlled_values(self.state)]
            values[0].fill_(float('nan'))
            return ControlledState(*values)

    @torch.no_grad()
    def policy_action(self, callback):
        # Materialize the historical batch layout: a batch-1/CPU convolution
        # is NOT assumed numerically identical to the saved CUDA policy.
        frames = self.frames.repeat(self.policy_batch,1,1)
        index = self.indices().repeat(self.policy_batch)
        actions = callback(frames,index)
        if actions.shape != (self.policy_batch,self.action_size):
            raise ValueError('Expected checkpoint-width deterministic actions')
        return actions[:1]

    def provenance(self):
        return dict(evaluation_backend='native_fixed_state_v1',
            evaluation_environments=1,evaluation_policy_lanes=self.policy_batch,
            unique_initial_states=1,numerical_failure=self.numerical_failure)
