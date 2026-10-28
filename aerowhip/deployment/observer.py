"""What the policy knows about the drone and the cable, from OptiTrack alone: the estimator that
runs in flight and, unchanged, inside the training environment (envs.sensing feeds it simulated
frames). Causal; no ROS, no simulator.

Each frame brings a receipt stamp, the drone rigid-body pose and the unlabelled marker cloud.
The frame clock (deployment.cable_filter.FrameClock) turns the jittered stamps into camera frame
times. The drone's velocity and body rate are the endpoint derivatives of a quadratic fitted to
the last `history` frame poses (position, and the rotation's log relative to the newest pose).
The cable is the live CableFilter on the cloud, drone markers and duplicates removed.

`Observer` is the native implementation (native/estimator.cpp); `ReferenceObserver` is the Python
reference it follows (identical cable states; drone velocity and body rate agree to ~1e-13, the
quadratic fit being solved in frame units instead of by NumPy's least squares).
"""
import math

import numpy as np

from .cable_filter import MARKERS, ReferenceCableFilter, ReferenceFrameClock, reference_cable_candidates


def endpoint_derivative(time, values):
    """Derivative at the newest sample of a least-squares quadratic (a line for two samples)."""
    time = np.asarray(time, float)
    if len(time) < 2:
        return np.zeros(np.shape(values)[1:])
    u = time-time[-1]
    design = np.column_stack([u**k for k in range(min(3, len(time)))])
    flat = (values-values[-1]).reshape(len(time), -1)
    return np.linalg.lstsq(design, flat, rcond=None)[0][1].reshape(np.shape(values)[1:])


def rotation_log(R):
    angle = math.acos(min(1., max(-1., (np.trace(R)-1)/2)))
    v = np.array([R[2, 1]-R[1, 2], R[0, 2]-R[2, 0], R[1, 0]-R[0, 1]])/2
    return v if angle < 1e-9 else v*angle/math.sin(angle)


class ReferenceObserver:
    def __init__(self, arc, attachment_body, *, dt=.01, history=5):
        """arc: marker distances along the cable (m); attachment_body: tracking origin to the cable
        attachment, body frame (m); history: frames in the velocity fit (5 = 40 ms span)."""
        self.attachment = np.asarray(attachment_body, float)
        self.dt, self.history = dt, history
        self.filter = ReferenceCableFilter(arc, dt=dt)
        self.clock = ReferenceFrameClock(dt)
        self.reset()

    def reset(self):
        self.filter.reset()
        self.clock.reset()
        self.times, self.positions, self.rotations = [], [], []

    def observe(self, stamp, position, rotation, cloud):
        """One OptiTrack frame -> the estimate at that frame (a dict)."""
        p, R = np.asarray(position, float), np.asarray(rotation, float)
        step = self.clock.advance(stamp)
        frame_time = (self.times[-1]+step) if self.times else 0.
        self.times.append(frame_time); self.positions.append(p); self.rotations.append(R)
        del self.times[:-self.history], self.positions[:-self.history], self.rotations[:-self.history]
        t = np.array(self.times)
        velocity = endpoint_derivative(t, np.array(self.positions))
        omega = endpoint_derivative(t, np.array([rotation_log(R.T@Ri) for Ri in self.rotations]))
        root = p+R@self.attachment
        root_velocity = velocity+R@np.cross(omega, self.attachment)
        # The filter gets frame times: its own clock then sees exact frames.
        state = self.filter.observe(frame_time, root, reference_cable_candidates(cloud, p))
        positions = state['positions']
        if not state['initialized'] or not np.isfinite(positions).all():
            # No labelled cable yet: markers at the attachment, zero confidence (the policy is
            # not started in this state).
            positions, velocities, confidence = np.tile(root, (MARKERS, 1)), np.zeros((MARKERS, 3)), np.zeros(MARKERS)
        else:
            velocities, confidence = state['velocities'], state['confidence']
        return dict(time_s=frame_time, stamp=float(stamp), position=p, rotation=R, velocity=velocity,
                    angular_velocity=omega, root=root, root_velocity=root_velocity, cable_positions=positions,
                    cable_velocities=velocities, confidence=confidence, labelled=state['index'] >= 0,
                    initialized=bool(state['initialized']))


from aerowhip.physics._native import Observer as _NativeObserver  # noqa: E402


class Observer(_NativeObserver):
    """The flight estimator (native; the Python reference is ReferenceObserver)."""

    def __init__(self, arc, attachment_body, *, dt=.01, history=5):
        super().__init__([float(a) for a in np.asarray(arc, float)], np.asarray(attachment_body, float), dt, history)

    def observe(self, stamp, position, rotation, cloud):
        return super().observe(float(stamp), np.asarray(position, float), np.asarray(rotation, float),
                               np.asarray(cloud, float).reshape(-1, 3))
