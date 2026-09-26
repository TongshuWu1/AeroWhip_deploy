"""Clamped quintic position commands, independent of physics, UI and learning."""

import numpy as np


def _vector(value, name):
    value = np.asarray(value, dtype=np.float64)
    if value.shape != (3,) or not np.isfinite(value).all():
        raise ValueError(f'{name} must contain three finite values')
    return value


def _derivative(points, knots, degree):
    spans = knots[degree+1:degree+len(points)] - knots[1:len(points)]
    return degree*np.diff(points, axis=0)/spans[:, None], knots[1:-1], degree-1


def _evaluate(points, knots, degree, times):
    """Vectorized Cox-de Boor basis, with the closed right endpoint included."""
    x = times[:, None]
    basis = ((x >= knots[:-1]) & (x < knots[1:])).astype(np.float64)
    for order in range(1, degree+1):
        count = len(knots)-order-1
        left_span = knots[order:order+count]-knots[:count]
        right_span = knots[order+1:order+1+count]-knots[1:count+1]
        left = np.divide(x-knots[:count], left_span,
                         out=np.zeros((len(times), count)), where=left_span > 0)
        right = np.divide(knots[order+1:order+1+count]-x, right_span,
                          out=np.zeros((len(times), count)), where=right_span > 0)
        basis = left*basis[:, :count]+right*basis[:, 1:count+1]
    basis[times == knots[-1]] = 0.
    basis[times == knots[-1], -1] = 1.
    return basis @ points


class BSplineCommand:
    """Free world-space control points plus endpoint; start P/V/A constrained.

    Simple interior knots give C4 continuity. Three controls at each end impose
    the boundary conditions; end velocity and acceleration are both zero.
    Shape points are controls, not interpolated waypoints. Limits below bound
    the commanded curve, not the coupled system's physical feasibility.
    """

    def __init__(self, start_position, end_position, shape_points, duration_s,
                 start_velocity=(0., 0., 0.), start_acceleration=(0., 0., 0.), yaw=0.):
        start = _vector(start_position, 'Start position')
        end = _vector(end_position, 'End position')
        velocity = _vector(start_velocity, 'Start velocity')
        acceleration = _vector(start_acceleration, 'Start acceleration')
        shape = np.asarray(shape_points, dtype=np.float64)
        if shape.ndim != 2 or shape.shape[1] != 3 or not np.isfinite(shape).all():
            raise ValueError('Shape points must be finite XYZ rows')
        self.duration_s = float(duration_s)
        self.yaw = float(yaw)
        if not np.isfinite(self.duration_s) or self.duration_s <= 0:
            raise ValueError('Duration must be positive and finite')
        if not np.isfinite(self.yaw):
            raise ValueError('Yaw must be finite')
        degree = 5
        points = np.empty((len(shape)+6, 3))
        spans = len(points)-degree
        knots = np.r_[np.zeros(degree+1), np.arange(1, spans)/spans,
                      np.ones(degree+1)]
        points[0] = start
        d0 = velocity*self.duration_s
        points[1] = start+d0*(knots[degree+1]-knots[1])/degree
        d1 = d0+acceleration*self.duration_s**2*(knots[degree+1]-knots[2])/(degree-1)
        points[2] = points[1]+d1*(knots[degree+2]-knots[2])/degree
        points[3:-3] = shape
        points[-3:] = end
        self._curves = [(points, knots, degree)]
        for _ in range(2):
            self._curves.append(_derivative(*self._curves[-1]))

    def evaluate(self, time_s):
        times = np.asarray(time_s, dtype=np.float64)
        if (times.ndim != 1 or not np.isfinite(times).all() or
                np.any(times < 0.) or np.any(times > self.duration_s)):
            raise ValueError('Times must be a finite vector within the maneuver duration')
        phase = times/self.duration_s
        return tuple(_evaluate(*curve, phase)/self.duration_s**order
                     for order, curve in enumerate(self._curves))

    def derivative_bounds(self):
        """Conservative continuous-time Euclidean speed/acceleration bounds."""
        return tuple(float(np.linalg.norm(curve[0], axis=1).max()/self.duration_s**order)
                     for order, curve in enumerate(self._curves[1:], start=1))

    def samples(self, timestep_s=1/30, max_speed_mps=None, max_acceleration_mps2=None):
        timestep_s = float(timestep_s)
        if not np.isfinite(timestep_s) or timestep_s <= 0:
            raise ValueError('Timestep must be positive and finite')
        intervals = self.duration_s/timestep_s
        if not np.isfinite(intervals) or not np.isclose(intervals, round(intervals), rtol=0, atol=1e-8):
            raise ValueError('Duration must be an exact multiple of the command timestep')
        if intervals < 1:
            raise ValueError('Duration must contain at least one command interval')
        for name, bound, limit in zip(('Speed', 'Acceleration'), self.derivative_bounds(),
                                     (max_speed_mps, max_acceleration_mps2)):
            if limit is not None:
                if not np.isfinite(limit) or limit <= 0:
                    raise ValueError(f'{name} limit must be positive and finite')
                if bound > limit+1e-10:
                    raise ValueError(f'{name} bound {bound:.3g} exceeds limit {limit:g}. '
                                     'Increase duration or adjust the controls. '
                                     'This continuous-time bound is conservative.')
        time = np.arange(round(intervals)+1)*timestep_s
        position, velocity, acceleration = self.evaluate(np.minimum(time, self.duration_s))
        # FullStateTrajectory column order: time, P, V, A, yaw, zero body rates.
        rows = np.zeros((len(time), 14))
        rows[:, 0] = time
        rows[:, 1:4], rows[:, 4:7], rows[:, 7:10] = position, velocity, acceleration
        rows[:, 10] = self.yaw
        return rows
