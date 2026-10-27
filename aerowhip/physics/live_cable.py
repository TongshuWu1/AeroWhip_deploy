"""Causal, measured-pose boundary driver for the existing native cable model.

No drone integration, controller or ROS dependency. State advances only after
both endpoints have arrived. Cable output is a simulated initial-value solution,
not a measurement or a corrected/calibrated cable-state estimate.
"""
from dataclasses import dataclass

import numpy as np

from aerowhip.physics.state import quaternion_xyzw_to_rotation_matrix
from aerowhip.physics.timing import integration_steps


@dataclass(frozen=True)
class LiveCableState:
    time_s: float
    positions_world_m: np.ndarray
    velocities_world_mps: np.ndarray


class LiveCable:
    refinement_factors = (1, 2, 4, 8)
    _numerical_rejections = (
        'Prescribed implicit endpoint iteration failed',
        'Prescribed implicit position projection failed',
        'Prescribed implicit velocity constraint failed',
    )

    def __init__(self, model, *, maximum_gap_s=.25):
        if not np.isfinite(maximum_gap_s) or maximum_gap_s <= 0:
            raise ValueError('Cable tracking gap limit must be positive and finite')
        self.model, self.maximum_gap_s = model, maximum_gap_s
        self.reset()

    def reset(self):
        self.latest = None
        self.error = None
        self.steps = 0
        self.refinement_retries = 0
        self.maximum_refinement = 1
        self._packed = self._previous = None
        self._previous_step = 0.
        self._rotation = self._quaternion = self._root = None
        self._velocity = np.zeros(3)

    def fail(self, message):
        self.error = str(message)

    def observe(self, time_s, position, quaternion):
        """Preserve cable/BDF2 history across samples; duplicate stamps do nothing."""
        if self.error:
            return self.latest
        p, q = np.asarray(position, dtype=np.float64), np.asarray(quaternion, dtype=np.float64)
        if (not np.isfinite(time_s) or p.shape != (3,) or q.shape != (4,)
                or not np.isfinite(p).all() or not np.isfinite(q).all()
                or not np.isclose(np.linalg.norm(q), 1., atol=1e-8, rtol=0)):
            raise ValueError('Live cable requires a finite measured position and unit XYZW rotation')
        R = quaternion_xyzw_to_rotation_matrix(q)
        root = p+R@self.model.attachment_tracking
        if self.latest is None:
            # Explicit prior: hanging, stationary cable at the first
            # measured tracking pose. Never use the hardware YAML position here.
            initial = self.model.initial_state(p, rotation=R)
            self._packed = np.concatenate((initial.free_positions.ravel(),
                initial.free_velocities.ravel()))
            self._previous = self._packed.copy()
        else:
            duration = float(time_s-self.latest.time_s)
            if duration == 0:
                return self.latest
            if duration < 0:
                self.fail('Cable capture time regressed; restart the real viewer')
                return self.latest
            if duration > self.maximum_gap_s:
                self.fail(f'Cable frozen after {duration:.3f} s tracking gap; restart the real viewer')
                return self.latest
            # Endpoint velocity uses only the current and preceding attachment
            # positions. The previous endpoint velocity is retained for C1 root
            # motion. Derived acceleration/rate belong to this interpolant.
            velocity = (root-self._root)/duration
            for refinement in self.refinement_factors:
                samples = self._samples(duration, root, velocity, q, refinement=refinement)
                try:
                    packed, previous, previous_step = self.model.native.prescribed_advance(
                        self._packed, samples, self._previous, self._previous_step)
                except Exception as exc:
                    # Native inputs/history are copied and only successful
                    # output is committed. Refine a numerical rejection;
                    # never weaken tolerances or reset history to bypass it.
                    if (not any(message in str(exc) for message in self._numerical_rejections)
                            or refinement == self.refinement_factors[-1]):
                        self.fail('Cable simulation stopped: '+str(exc))
                        return self.latest
                    self.refinement_retries += 1
                else:
                    self.maximum_refinement = max(self.maximum_refinement, refinement)
                    break
            self._packed, self._previous, self._previous_step = packed, previous, previous_step
            self._velocity = velocity
            self.steps += len(samples)
        self._root, self._rotation, self._quaternion = root.copy(), R, q.copy()
        n = self.model.n-1
        self.latest = LiveCableState(float(time_s),
            np.vstack((root, self._packed[:3*n].reshape(n, 3))),
            np.vstack((self._velocity, self._packed[3*n:6*n].reshape(n, 3))))
        return self.latest

    def _samples(self, duration, root, velocity, quaternion, *, refinement=1):
        count, _ = integration_steps(duration, self.model.physics_dt)
        # Refine the actual interval subdivision. Dividing the maximum step
        # alone repeats an identical single step for short capture intervals.
        count *= refinement
        h = duration/count
        starts = np.arange(count)*h
        ends = (np.arange(count)+1)*h
        ends[-1] = duration
        u = np.column_stack((starts, (starts+ends)/2, np.nextafter(ends, starts), ends)).ravel()/duration
        u = u[:, None]
        p0, p1, v0, v1 = self._root, root, self._velocity, velocity
        p = (2*u**3-3*u**2+1)*p0+(u**3-2*u**2+u)*duration*v0+(-2*u**3+3*u**2)*p1+(u**3-u**2)*duration*v1
        v = (6*u*u-6*u)*(p0-p1)/duration+(3*u*u-4*u+1)*v0+(3*u*u-2*u)*v1
        a = (12*u-6)*(p0-p1)/duration**2+(6*u-4)*v0/duration+(6*u-2)*v1/duration
        # Shortest SO(3) geodesic; rates are piecewise attitude-interpolant
        # derivatives, not gyro telemetry. Quaternion sign changes are harmless.
        q0 = self._quaternion
        q1 = quaternion if q0@quaternion >= 0 else -quaternion
        x0, w0, x1, w1 = q0[:3], q0[3], q1[:3], q1[3]
        relative_xyz = w0*x1-w1*x0-np.cross(x0, x1)
        relative_w = np.clip(w0*w1+x0@x1, -1., 1.)
        length = np.linalg.norm(relative_xyz)
        angle = 2*np.arctan2(length, relative_w)
        axis = relative_xyz/length if length > 1e-14 else np.zeros(3)
        x, y, z = axis
        K = np.array([[0., -z, y], [z, 0., -x], [-y, x, 0.]])
        phi = (u[:, 0]*angle)[:, None, None]
        rotations = self._rotation@(np.eye(3)+np.sin(phi)*K+2*np.sin(phi/2)**2*(K@K))
        omega = np.broadcast_to(axis*angle/duration, (len(u), 3))
        values = np.column_stack((p, v, a, rotations.reshape(-1, 9), omega))
        return np.column_stack((ends-starts, values.reshape(count, 84)))
