"""Cable-material objective on calibration flights (measured attachment drives the cable).

For each window the measured drone pose (true OptiTrack stamps, ~100 Hz with bunching)
drives the native prescribed-cable engine from an initial cable state built from the
window's history; the loss is the RMS distance between simulated and measured markers over
the markers that were seen (hidden markers are skipped, never filled). Parameters are
multipliers on EI, Cb and the external drag, as in `cable_fit`.
"""
from copy import deepcopy
import math

import numpy as np
import torch

from aerowhip.fitting.prepare_window import initialize
from aerowhip.fitting.recorded_replay import AttachmentMotion
from aerowhip.physics.prescribed_cable import NativePrescribedCable, PrescribedCableState
from aerowhip.physics.tensor_model import TensorModel

NAMES = ('EI_n_m2', 'Cb_n_m2_s', 'external_drag_s_inv')


class MeasuredAttachment(AttachmentMotion):
    """AttachmentMotion on the measured, unevenly spaced OptiTrack stamps."""

    def __init__(self, time_s, positions, velocities, rotations, tangent_body, *, device='cpu'):
        t = np.asarray(time_s, float)
        if t.ndim != 1 or len(t) < 2 or t[0] != 0 or np.any(np.diff(t) <= 0):
            raise ValueError('Attachment knots must start at 0 and increase')
        # Build on a uniform dummy grid to pass the parent's checks, then install the true stamps.
        super().__init__(np.arange(len(t))*.01, positions, velocities, rotations, tangent_body, device=device)
        self.time_s = t.copy()
        rv_rates = self._body_rate*(.01/np.diff(t))[:, None]   # body rate per true interval
        self._body_rate = rv_rates


def scaled_config(base, scales):
    config = deepcopy(base)
    for name, scale in zip(NAMES, scales):
        config['cable'][name] = float(config['cable'][name]*scale)
    return config


def window_initial(model, window):
    """Initial plant state from the window's history (velocities by causal quadratic fit)."""
    hist = window.history_markers
    if not np.isfinite(hist).all():
        raise ValueError('Window history must have all markers')
    state, info = initialize(model, window.history_time, window.history_position, window.history_rotation, hist)
    return state, info


def attachment_motion(model, window, initial):
    """Measured attachment path on the window's stamps; the first knot's velocity is the
    rigid-body root velocity of `initial`, so the cable's initial state is consistent."""
    attachment = model.attachment_tracking.detach().cpu().numpy()
    t = window.time
    roots = window.position+np.einsum('nij,j->ni', window.rotation, attachment)
    velocities = np.gradient(roots, t, axis=0, edge_order=2)
    root_velocity = initial.velocity_com+initial.rotation@torch.linalg.cross(initial.omega_body, model.arm)
    velocities[0] = root_velocity.detach().cpu().numpy()
    return MeasuredAttachment(t, roots, velocities, window.rotation, model.root_tangent_body.detach().cpu().numpy(),
                              device=model.device)


def window_loss(base, window, scales, *, return_markers=False):
    """RMS marker error (m) over seen markers of one window for the given multipliers."""
    model = TensorModel(scaled_config(base, scales))
    cable = NativePrescribedCable(model)
    initial, _ = window_initial(model, window)
    motion = attachment_motion(model, window, initial)
    state = PrescribedCableState(initial.free_positions, initial.free_velocities, initial.material_angles)
    idx = model.marker_indices[1:]-1    # free-node index of each marker
    errors, predicted = [], []
    for i in range(len(window.time)-1):
        start, end = float(window.time[i]), float(window.time[i+1])
        state = cable.step(state, start, end-start, motion)
        q = state.free_positions.detach().cpu().numpy()[idx]
        predicted.append(q)
        seen = window.seen[i+1]
        if seen.any():
            errors.append(np.linalg.norm(q[seen]-window.markers[i+1][seen], axis=1))
    e = np.concatenate(errors) if errors else np.array([np.nan])
    loss = float(np.sqrt(np.mean(e**2)))
    if not math.isfinite(loss):
        raise FloatingPointError('Nonfinite cable loss')
    return (loss, np.array(predicted)) if return_markers else loss
