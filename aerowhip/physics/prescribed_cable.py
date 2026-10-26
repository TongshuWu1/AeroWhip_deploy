"""Recorded-boundary sampling and validation for C++ cable-only tensor physics."""
from dataclasses import dataclass
import math
import numpy as np
import torch
from aerowhip.physics.tensor_model import ops
from aerowhip.physics.timing import integration_steps


@dataclass(frozen=True)
class PrescribedCableState:
    free_positions: torch.Tensor
    free_velocities: torch.Tensor
    material_angles: torch.Tensor
    previous_state: torch.Tensor | None = None
    previous_step_s: torch.Tensor | None = None


def integration_values(state):
    values = [state.free_positions, state.free_velocities, state.material_angles]
    if state.previous_state is None and state.previous_step_s is None:
        return [*values, torch.cat([v.flatten() for v in values]), values[0].new_zeros(1)]
    if state.previous_state is None or state.previous_step_s is None:
        raise ValueError("Both BDF2 history fields are required together")
    return [*values, state.previous_state, state.previous_step_s]


class PrescribedCable:
    def __init__(self, model):
        self.model = model

    def acceleration(self, state, boundary):
        self._validate(state, [(0., (boundary,))])
        values = tuple(ops.prescribed_acceleration([state.free_positions, state.free_velocities, state.material_angles],
            list(boundary), self.model._parameters(), *self.model._residual_parameters(),
            self.model.rest.is_cuda and torch.compiler.is_compiling()))
        if not all(bool(torch.isfinite(value).all()) for value in values):
            raise FloatingPointError('Nonfinite prescribed cable acceleration or support wrench')
        return values

    def _samples(self, start_s, duration_s, boundary):
        if not math.isfinite(start_s):
            raise ValueError('Boundary start time must be finite')
        count, h = integration_steps(duration_s, self.model.physics_dt)
        samples = []
        for i in range(count):
            start = start_s+i*h
            end = start_s+duration_s if i == count-1 else start_s+(i+1)*h
            # Preserve the boundary sample format; implicit integration uses
            # the exact endpoint. Acceleration remains telemetry input.
            samples.append((end-start, (boundary(start), boundary((start+end)/2),
                                       boundary(math.nextafter(end, start)), boundary(end))))
        return samples

    def _validate(self, state, samples):
        device = self.model.rest.device
        history = integration_values(state)[3:]
        for value, shape in zip(history, ((7*(self.model.n-1),), (1,))):
            if (not isinstance(value, torch.Tensor) or value.shape != shape
                    or value.dtype != torch.float64 or value.device != device
                    or not bool(torch.isfinite(value).all())):
                raise ValueError('Invalid prescribed BDF2 history shape/device/values')
        if not 0 <= float(history[1].detach()[0]) <= self.model.physics_dt*(1+1e-9):
            raise ValueError('Invalid prescribed BDF2 previous timestep')
        for value in (state.free_positions, state.free_velocities):
            if (not isinstance(value, torch.Tensor) or value.shape != (self.model.n-1, 3)
                    or value.dtype != torch.float64 or value.device != device or not bool(torch.isfinite(value).all())):
                raise ValueError('Expected finite float64 free-node state on the model device')
        angle = state.material_angles
        if (not isinstance(angle, torch.Tensor) or angle.shape != (self.model.n-1,)
                or angle.dtype != torch.float64 or angle.device != device or not bool(torch.isfinite(angle).all())):
            raise ValueError('Expected finite float64 material-angle state on the model device')
        values = []
        for _, stages in samples:
            for boundary in stages:
                if not isinstance(boundary, (tuple, list)) or len(boundary) != 5:
                    raise ValueError('Material clamp requires root P/V/A, body-to-world rotation, body angular rate')
                for index, value in enumerate(boundary):
                    if (not isinstance(value, torch.Tensor) or value.shape != ((3,3) if index == 3 else (3,))
                            or value.dtype != torch.float64 or value.device != device):
                        raise ValueError('Boundary P/V/A/rotation/angular rate must be float64 tensors on the model device')
                    values.append(value.flatten())
        rotations = torch.stack([stage[3] for _, stages in samples for stage in stages])
        if (not bool(torch.allclose(rotations.transpose(-1,-2)@rotations, torch.eye(3,dtype=torch.float64,device=device).expand_as(rotations),atol=1e-8,rtol=0))
                or bool((torch.linalg.det(rotations)-1).abs().gt(1e-8).any())):
            raise ValueError('Material-clamp boundary requires proper rotations')
        if not bool(torch.isfinite(torch.cat(values)).all()):
            raise ValueError('Boundary P/V/A/tangent must be finite')
        initial = samples[0][1][0]
        q = torch.cat((initial[0][None], state.free_positions))
        if not bool(ops.geometry_valid(q, initial[3]@self.model.root_tangent_body)):
            raise ValueError('Singular DER geometry: collapsed edge or antiparallel root/interior directions')
        errors = ops.prescribed_constraints([state.free_positions, state.free_velocities, state.material_angles],
            list(samples[0][1][0]), self.model._parameters())
        if any(bool((error.abs() > 1e-6).any()) for error in errors):
            raise ValueError('Initial cable lengths/rates violate the 1e-6 m/mps constraints')

    def _advance(self, state, samples):
        for h, stages in samples:
            state = PrescribedCableState(*ops.prescribed_step(
                integration_values(state), [v for stage in stages for v in stage],
                self.model._parameters(), h, self.model.iterations, *self.model._residual_parameters(),
                self.model.rest.is_cuda and torch.compiler.is_compiling()))
        return state

    def step(self, state, start_s, duration_s, boundary):
        samples = self._samples(start_s, duration_s, boundary)
        self._validate(state, samples)
        output = self._advance(state, samples)
        if not all(bool(torch.isfinite(value).all()) for value in (output.free_positions, output.free_velocities, output.material_angles)):
            raise FloatingPointError('Invalid prescribed cable integration: nonfinite state or singular DER geometry')
        self._validate(output, [(0., (samples[-1][1][-1],))])
        return output

    def step_unchecked(self, state, start_s, duration_s, boundary):
        return self._advance(state, self._samples(start_s, duration_s, boundary))


class NativePrescribedCable(PrescribedCable):
    """Non-gradient CPU execution; the tensor class retains fitting derivatives."""
    _physical_fields = ('mass', 'inertia', 'com_tracking', 'attachment_tracking',
        'arm', 'gravity', 'rest', 'node_mass', 'ei', 'drag', 'root_tangent_body', 'cb', 'gj', 'ct')

    def __init__(self, model):
        super().__init__(model)
        from aerowhip.physics.model import Model
        self._metadata_native = model.metadata.native
        self.native = Model(model.config, residual=model.residual).native
        self._residual_reference = None if model.residual is None else (
            [p.detach().cpu().clone() for p in model.residual.parameters()], model.residual.tensor_parameters()[1])
        # Read the immutable native parameters, not metadata arrays that a
        # CPU tensor may alias through torch.as_tensor.
        native_info = self.native.info()
        self._physical_reference = {name: torch.as_tensor(native_info[name],dtype=torch.float64).clone()
                                    for name in self._physical_fields}
        self._native_n = native_info['n']
        self._native_iterations = native_info['iterations']
        self._check_model()

    def _check_model(self):
        m = self.model
        if (m.residual is None) != (self._residual_reference is None):
            raise ValueError('Native prescribed residual changed; reconstruct the adapter')
        if m.residual is not None:
            weights, spec = self._residual_reference
            if (spec != m.residual.tensor_parameters()[1] or
                    any(p.requires_grad or not torch.equal(p.detach().cpu(), ref)
                        for p,ref in zip(m.residual.parameters(), weights))):
                raise ValueError('Native prescribed residual must be frozen and unchanged; use PrescribedCable for fitting')
        for name in self._physical_fields:
            value = getattr(m, name)
            if (not isinstance(value, torch.Tensor) or value.device.type != 'cpu'
                    or value.dtype != torch.float64 or value.requires_grad):
                raise ValueError('Native prescribed parameters require non-gradient CPU float64; use PrescribedCable for gradients')
            original = self._physical_reference[name]
            if value.shape != original.shape or not torch.equal(value, original):
                raise ValueError('Native prescribed parameters changed after native construction; construct a new model from the modified configuration')
        if m.n != self._native_n or m.iterations != self._native_iterations or m.metadata.native is not self._metadata_native:
            raise ValueError('Native prescribed mesh/projection settings differ from the native model')

    @staticmethod
    def _pack(values):
        for value in values:
            if (not isinstance(value, torch.Tensor) or value.device.type != 'cpu'
                    or value.dtype != torch.float64 or value.requires_grad):
                raise ValueError('Native prescribed inputs require non-gradient CPU float64; use PrescribedCable for gradients')
        return np.concatenate([v.numpy().ravel() for v in values])

    def _state_values(self, state):
        values = (state.free_positions,state.free_velocities,state.material_angles)
        count = self.model.n-1
        if any(not isinstance(value,torch.Tensor) or value.shape != shape
               for value,shape in zip(values,((count,3),(count,3),(count,)))):
            raise ValueError('Unexpected prescribed free-node/material-angle state shape')
        return values

    @staticmethod
    def _boundary_values(boundary):
        if not isinstance(boundary,(tuple,list)) or len(boundary) != 5:
            raise ValueError('Material clamp requires root P/V/A, rotation and body angular rate')
        if any(not isinstance(value,torch.Tensor) or value.shape != ((3,3) if i == 3 else (3,))
               for i,value in enumerate(boundary)):
            raise ValueError('Unexpected material-clamp boundary shape')
        return boundary

    def acceleration(self, state, boundary):
        self._check_model()
        raw=self.native.prescribed_acceleration(self._pack(self._state_values(state)),self._pack(self._boundary_values(boundary)))
        m=self.model.n-1
        return self.model.tensor(raw[:3*m].reshape(m,3)),self.model.tensor(raw[3*m:3*m+3]),self.model.tensor(raw[-3:])

    def step(self, state, start_s, duration_s, boundary):
        self._check_model()
        values=self._state_values(state)
        packed=self._pack(values)
        if hasattr(boundary,'native_samples'):
            samples=boundary.native_samples(start_s,duration_s,self.model.physics_dt)
        else:
            sampled=self._samples(start_s,duration_s,boundary)
            samples=np.stack([np.r_[h,*[self._pack(self._boundary_values(b)) for b in stages]] for h,stages in sampled])
        history=integration_values(state)[3:]
        if history[0].shape != (7*(self.model.n-1),) or history[1].shape != (1,):
            raise ValueError('Invalid prescribed BDF2 history shape')
        previous=self._pack(history[:1]);previous_step=float(self._pack(history[1:])[0])
        out,previous,previous_step=self.native.prescribed_advance(packed,samples,previous,previous_step);m=self.model.n-1
        return PrescribedCableState(self.model.tensor(out[:3*m].reshape(m,3)),
            self.model.tensor(out[3*m:6*m].reshape(m,3)),self.model.tensor(out[6*m:]),
            self.model.tensor(previous),self.model.tensor([previous_step]))
