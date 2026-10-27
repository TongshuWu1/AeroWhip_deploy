"""Cable replay on calibration flights with the native engine (no tensor engine).

The measured drone pose drives the cable clamp (prescribed boundary); the cable starts from
the measured markers at the window start and runs open loop. The loss is the RMS distance
between simulated and measured markers over the markers that were seen; hidden markers are
skipped, never filled. Windows run in parallel threads (the native step releases the GIL).

The boundary between two OptiTrack frames is a cubic Hermite in position (knot velocities
from central differences of the measured attachment path) and a constant-rate geodesic in
rotation, the same interpolant as the live cable driver.
"""
from concurrent.futures import ThreadPoolExecutor
from copy import deepcopy
import math
import os

import numpy as np

from aerowhip.physics.timing import integration_steps


def endpoint_velocity(time, values):
    """Quadratic least-squares derivative at the newest sample (causal history only)."""
    time = np.asarray(time, float)
    u = (time-time[-1])/(time[-1]-time[0])
    design = np.column_stack((np.ones_like(u), u, u*u))
    shape = values.shape[1:]
    flat = (values-values[-1]).reshape(len(time), -1)
    coefficients = np.linalg.lstsq(design, flat, rcond=None)[0]
    return (coefficients[1]/(time[-1]-time[0])).reshape(shape)


def _quaternion(R):
    w = math.sqrt(max(0., 1+R[0, 0]+R[1, 1]+R[2, 2]))/2
    if w > 1e-6:
        return np.array([(R[2, 1]-R[1, 2])/(4*w), (R[0, 2]-R[2, 0])/(4*w), (R[1, 0]-R[0, 1])/(4*w), w])
    i = int(np.argmax(np.diag(R)))
    j, k = (i+1) % 3, (i+2) % 3
    q = np.zeros(4)
    q[i] = math.sqrt(max(0., 1+R[i, i]-R[j, j]-R[k, k]))/2
    q[3] = (R[k, j]-R[j, k])/(4*q[i])
    q[j] = (R[j, i]+R[i, j])/(4*q[i])
    q[k] = (R[k, i]+R[i, k])/(4*q[i])
    return q


class CableReplay:
    """One native model (any parameter set) replaying windows of a calibration flight."""

    def __init__(self, native_model, *, state_width=6):
        self.native = native_model
        info = native_model.info()
        self.rest = np.asarray(info['rest'])
        self.attachment = np.asarray(info['attachment_tracking'])
        self.physics_dt = float(info['physics_dt'])
        self.markers = np.asarray(info['marker_indices'])[1:]-1   # free-node index of each marker
        self.free = len(self.rest)
        self.width = state_width                                 # 6 (twist-free) or 7 (v1 angles)
        if not np.array_equal(np.diff(np.asarray(info['marker_indices'])), np.ones(len(self.markers), int)):
            raise ValueError('Replay assumes one cable segment per marker interval')

    def roots(self, positions, rotations):
        return positions+np.einsum('nij,j->ni', rotations, self.attachment)

    def initial(self, window):
        """Measured markers at the window start, projected onto the configured lengths;
        velocities from the causal history, projected onto the length constraints."""
        t = window.history_time
        roots = self.roots(window.history_position, window.history_rotation)
        markers = window.history_markers
        if not np.isfinite(markers).all():
            raise ValueError('Window history must have every marker')
        root, root_v = roots[-1], endpoint_velocity(t, roots)
        velocities = endpoint_velocity(t, markers)
        q, v = np.empty((self.free, 3)), np.empty((self.free, 3))
        previous, previous_v = root, root_v
        for i in range(self.free):
            edge = markers[-1, i]-previous
            tangent = edge/np.linalg.norm(edge)
            q[i] = previous+tangent*self.rest[i]
            relative = velocities[i]-previous_v
            v[i] = previous_v+relative-tangent*(tangent@relative)
            previous, previous_v = q[i], v[i]
        state = np.r_[q.ravel(), v.ravel()]
        if self.width == 7:
            state = np.r_[state, np.zeros(self.free)]
        return state, root_v

    def samples(self, window, root_velocity):
        """Per-frame-interval boundary samples (rows of h plus four 21-value samples)."""
        t = window.time
        roots = self.roots(window.position, window.rotation)
        velocities = np.gradient(roots, t, axis=0, edge_order=2)
        velocities[0] = root_velocity
        quaternions = np.array([_quaternion(R) for R in window.rotation])
        intervals = []
        for k in range(len(t)-1):
            duration = t[k+1]-t[k]
            count, _ = integration_steps(duration, self.physics_dt)
            h = duration/count
            starts, ends = np.arange(count)*h, (np.arange(count)+1)*h
            ends[-1] = duration
            u = (np.column_stack((starts, (starts+ends)/2, np.nextafter(ends, starts), ends)).ravel()/duration)[:, None]
            p0, p1, v0, v1 = roots[k], roots[k+1], velocities[k], velocities[k+1]
            p = (2*u**3-3*u**2+1)*p0+(u**3-2*u**2+u)*duration*v0+(-2*u**3+3*u**2)*p1+(u**3-u**2)*duration*v1
            v = (6*u*u-6*u)*(p0-p1)/duration+(3*u*u-4*u+1)*v0+(3*u*u-2*u)*v1
            a = (12*u-6)*(p0-p1)/duration**2+(6*u-4)*v0/duration+(6*u-2)*v1/duration
            q0, q1 = quaternions[k], quaternions[k+1]
            if q0@q1 < 0:
                q1 = -q1
            x0, w0, x1, w1 = q0[:3], q0[3], q1[:3], q1[3]
            relative = w0*x1-w1*x0-np.cross(x0, x1)
            length = np.linalg.norm(relative)
            angle = 2*math.atan2(length, np.clip(w0*w1+x0@x1, -1., 1.))
            axis = relative/length if length > 1e-14 else np.zeros(3)
            K = np.array([[0., -axis[2], axis[1]], [axis[2], 0., -axis[0]], [-axis[1], axis[0], 0.]])
            phi = (u[:, 0]*angle)[:, None, None]
            rotations = window.rotation[k]@(np.eye(3)+np.sin(phi)*K+2*np.sin(phi/2)**2*(K@K))
            omega = np.broadcast_to(axis*angle/duration, (len(u), 3))
            values = np.column_stack((p, v, a, rotations.reshape(-1, 9), omega))
            intervals.append(np.column_stack((ends-starts, values.reshape(count, 84))))
        return intervals

    def run(self, window):
        """Simulated markers (N, 10, 3) at the window frames 1..N."""
        state, root_v = self.initial(window)
        previous, previous_step = state.copy(), 0.
        predicted = np.empty((len(window.time)-1, len(self.markers), 3))
        for k, rows in enumerate(self.samples(window, root_v)):
            state, previous, previous_step = self.native.prescribed_advance(state, rows, previous, previous_step)
            predicted[k] = state[:3*self.free].reshape(self.free, 3)[self.markers]
        return predicted

    def errors(self, window):
        """Per-frame, per-marker distance (m), NaN where the marker was not seen."""
        predicted = self.run(window)
        error = np.linalg.norm(predicted-window.markers[1:], axis=2)
        error[~window.seen[1:]] = np.nan
        return error


def evaluate(native_model, windows, *, state_width=6, threads=None):
    """Errors for every window (list of (N, 10) arrays); failed windows give None."""
    replay = CableReplay(native_model, state_width=state_width)

    def one(window):
        try:
            return replay.errors(window)
        except Exception:
            return None
    with ThreadPoolExecutor(threads or os.cpu_count()) as pool:
        return list(pool.map(one, windows))


def rms(errors):
    """Pooled RMS (m) over every seen marker of every successful window; NaN if any failed."""
    if any(e is None for e in errors):
        return math.nan
    values = np.concatenate([e[np.isfinite(e)] for e in errors])
    return float(np.sqrt(np.mean(values**2)))


def with_cable(config, **values):
    """Copy of a model configuration with cable fields replaced."""
    config = deepcopy(config)
    config['cable'].update({k: float(v) for k, v in values.items()})
    return config
