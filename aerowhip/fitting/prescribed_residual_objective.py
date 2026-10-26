"""Cable drag identification with measured support, never vehicle tracking loss."""
import numpy as np
import torch
from torch.utils.checkpoint import checkpoint

from aerowhip.fitting.coupled_fit import window_weights
from aerowhip.fitting.recorded_replay import AttachmentMotion
from aerowhip.physics.tensor_model import TensorModel, PlantState
from aerowhip.physics.prescribed_cable import PrescribedCable, PrescribedCableState, integration_values


def measured_boundary(model, window):
    """Offline interpolation of native pose knots, not a predicted drone path.

    Three-point root-velocity differences use measured neighboring samples;
    the initial derivative is the causal rigid-body initialization. Angular
    rate belongs to the attitude interpolant, not to a measured gyro.
    """
    def array(x):
        return x.detach().cpu().numpy() if isinstance(x, torch.Tensor) else np.asarray(x)
    p0, _ = model.metadata.tracking_pose(window.initial)
    positions = np.concatenate((p0[None], window.tracking_positions))
    rotations = np.concatenate((array(window.initial.rotation)[None], window.tracking_rotations))
    roots = positions + np.einsum('nij,j->ni', rotations, array(model.attachment_tracking))
    times = np.arange(len(roots))*model.dt
    if len(times) < 3:
        raise ValueError('Prescribed residual fitting requires at least two output intervals')
    velocities = np.gradient(roots, times, axis=0, edge_order=2)
    velocities[0] = array(window.initial.velocity_com) + rotations[0]@np.cross(
        array(window.initial.omega_body), array(model.arm))
    return AttachmentMotion(times, roots, velocities, rotations, array(model.root_tangent_body), device=model.device)


class PrescribedResidualObjective:
    """Sequential windows with global weights; recompute one output interval.

    Physical parameters and measured boundary are constant. Only cable drag
    weights receive gradients. No state detach/reset inside a segment.
    The fitting API uses loss(backward=True), i.e. ordinary Tensor.backward();
    reentrant checkpointing does not support torch.autograd.grad on window_loss.
    """
    def __init__(self, config, windows, network, magnitude_penalty, progress=None):
        self.network, self.windows = network, windows
        self.model = TensorModel(config, device=str(next(network.parameters()).device), residual=network)
        if float(self.model.drag) <= 0:
            raise ValueError('Multiplicative cable drag identification requires a positive fitted baseline drag')
        self.solver = PrescribedCable(self.model)
        self.weights = window_weights(windows)
        self.boundaries = [measured_boundary(self.model, w) for w in windows]
        self.magnitude_penalty, self.progress = magnitude_penalty, progress
        self.batch_size = 1

    def window_loss(self, window, boundary):
        m = self.model
        state = PrescribedCableState(*(m.tensor(getattr(window.initial, key)).detach().clone()
            for key in ('free_positions','free_velocities','material_angles')))
        if torch.is_grad_enabled():
            # Reentrant checkpoints require a differentiable incoming tensor.
            # This fixed initial position's adjoint is unused, not optimized.
            # Recurrent states below remain connected for the ENTIRE segment.
            state.free_positions.requires_grad_(True)
        markers = m.tensor(window.cable_markers)
        indices = torch.as_tensor(m.marker_indices[1:]-1, device=m.device)
        loss = m.tensor(0.)
        for k in range(len(markers)):
            # Bind time now: backward checkpoint recomputation must not observe
            # a loop variable changed to the last interval.
            start = k*m.dt
            duration = float(boundary.time_s[k+1]-start)
            def block(*values, start=start, duration=duration, k=k):
                result = self.solver.step_unchecked(PrescribedCableState(*values), start, duration, boundary)
                error = ((result.free_positions[indices]-markers[k])/.03).square().mean()
                b = boundary(float(boundary.time_s[k+1]))
                full = PlantState(b[0]-b[3]@m.arm, b[3],
                    b[1]-b[3]@torch.linalg.cross(b[4],m.arm), b[4],
                    result.free_positions, result.free_velocities, result.material_angles)
                penalty = self.network.magnitude(m, full, None)
                error = error + self.magnitude_penalty*penalty
                return (*integration_values(result), error)
            values = integration_values(state)
            out = checkpoint(block, *values, use_reentrant=True) if torch.is_grad_enabled() else block(*values)
            state = PrescribedCableState(*out[:5])
            loss = loss + out[5]
        return loss/len(markers)

    def loss(self, backward=False):
        total = 0.
        for index, (window,boundary,weight) in enumerate(zip(self.windows,self.boundaries,self.weights)):
            with torch.set_grad_enabled(backward):
                value = weight*self.window_loss(window,boundary)
                if not bool(torch.isfinite(value)):
                    raise FloatingPointError('Nonfinite measured-boundary cable residual loss')
                if backward:
                    value.backward()
                total += float(value.detach())
            if self.progress:
                self.progress(dict(phase='loss_pass',stage='cable',backward=backward,
                    completed_segments=index+1,total_segments=len(self.windows),execution_batch_size=1))
        return total
