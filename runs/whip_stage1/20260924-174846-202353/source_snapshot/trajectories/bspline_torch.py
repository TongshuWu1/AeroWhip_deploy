"""Batched quintic commands with cached bases; no physics or UI dependencies."""

import operator

import numpy as np
import torch
from torch import nn

from .bspline import _derivative, _evaluate


class BatchedBSplineCommand(nn.Module):
    """One control count, duration and sample grid per reusable batch evaluator.

    Inputs are world-space tensors on the evaluator's device/dtype. Start P/V/A
    and end position are constrained; end V/A are zero. Yaw is constant. Unlike
    the interactive generator, tensor values are not checked on the CPU: callers
    must reject nonfinite inputs/outputs and enforce command/physical limits.
    Duration is fixed here, not a differentiable policy output; group different
    durations into separate evaluators to preserve the actual controller rate.
    """

    def __init__(self, control_count, duration_s, timestep_s=.01, *,
                 device="cpu", dtype=torch.float64):
        super().__init__()
        self.control_count = operator.index(control_count)
        self.duration_s, self.timestep_s = float(duration_s), float(timestep_s)
        if self.control_count < 6:
            raise ValueError('At least six total control points are required')
        if dtype not in (torch.float32, torch.float64):
            raise ValueError('Use float32 or float64')
        if (not np.isfinite([self.duration_s, self.timestep_s]).all()
                or self.duration_s <= 0 or self.timestep_s <= 0):
            raise ValueError('Duration and timestep must be positive and finite')
        intervals = self.duration_s/self.timestep_s
        if (not np.isfinite(intervals) or intervals < 1
                or not np.isclose(intervals, round(intervals), rtol=0, atol=1e-8)):
            raise ValueError('Duration must be an exact positive number of command intervals')
        self.sample_count = round(intervals)+1
        spans = self.control_count-5
        knots = np.r_[np.zeros(6), np.arange(1, spans)/spans, np.ones(6)]
        self._start_spans = (float(knots[6]-knots[1]), float(knots[6]-knots[2]),
                             float(knots[7]-knots[2]))
        time = np.arange(self.sample_count)*self.timestep_s
        phase = np.minimum(time/self.duration_s, 1.)
        self.register_buffer('basis', torch.as_tensor(
            _evaluate(np.eye(self.control_count), knots, 5, phase), device=device, dtype=dtype))
        curve = (np.eye(self.control_count), knots, 5)
        for order in (1, 2):
            points, curve_knots, degree = curve
            widths = curve_knots[degree+1:degree+len(points)]-curve_knots[1:len(points)]
            self.register_buffer(f'derivative_scale_{order}', torch.as_tensor(
                degree/(widths*self.duration_s), device=device, dtype=dtype)[:, None])
            curve = _derivative(*curve)
            self.register_buffer(f'derivative_basis_{order}', torch.as_tensor(
                _evaluate(np.eye(len(curve[0])), curve[1], curve[2], phase), device=device, dtype=dtype))
        self.register_buffer('time_s', torch.as_tensor(time, device=device, dtype=dtype))

    def _check(self, value, shape, name):
        if (not isinstance(value, torch.Tensor) or tuple(value.shape) != shape
                or value.device != self.basis.device or value.dtype != self.basis.dtype):
            raise ValueError(f'{name} must have shape {shape} on {self.basis.device} with {self.basis.dtype}')

    def control_points(self, start_position, end_position, shape_points,
                       start_velocity=None, start_acceleration=None):
        if not isinstance(start_position, torch.Tensor) or start_position.ndim != 2:
            raise ValueError('Start position must have shape (batch, 3)')
        batch = start_position.shape[0]
        self._check(start_position, (batch, 3), 'Start position')
        self._check(end_position, (batch, 3), 'End position')
        self._check(shape_points, (batch, self.control_count-6, 3), 'Shape points')
        velocity = torch.zeros_like(start_position) if start_velocity is None else start_velocity
        acceleration = torch.zeros_like(start_position) if start_acceleration is None else start_acceleration
        self._check(velocity, (batch, 3), 'Start velocity')
        self._check(acceleration, (batch, 3), 'Start acceleration')
        span0, span1, span2 = self._start_spans
        d0 = velocity*self.duration_s
        first = start_position+d0*span0/5
        d1 = d0+acceleration*self.duration_s**2*span1/4
        second = first+d1*span2/5
        return torch.cat((torch.stack((start_position, first, second), dim=1),
                          shape_points, end_position[:, None, :].expand(-1, 3, -1)), dim=1)

    def _check_points(self, points):
        if not isinstance(points, torch.Tensor) or points.ndim != 3:
            raise ValueError('Control points must have shape (batch, controls, 3)')
        self._check(points, (points.shape[0], self.control_count, 3), 'Control points')

    def evaluate(self, points):
        """Return P, V, A tensors, each (batch, samples, 3), with autograd intact."""
        self._check_points(points)
        centered = points-points[:, :1]
        velocity, acceleration = self._derivative_controls(points)
        return (self.basis @ centered+points[:, :1],
                self.derivative_basis_1 @ velocity, self.derivative_basis_2 @ acceleration)

    def _derivative_controls(self, points):
        # Difference adjacent controls before scaling: preserves exact stopped
        # endpoints and avoids cancellation from a dense differentiated basis.
        velocity = torch.diff(points, dim=1)*self.derivative_scale_1
        acceleration = torch.diff(velocity, dim=1)*self.derivative_scale_2
        return velocity, acceleration

    def derivative_bounds(self, points):
        """Per-trajectory conservative continuous-time speed/acceleration bounds."""
        self._check_points(points)
        return tuple(torch.linalg.vector_norm(controls, dim=-1).amax(dim=-1)
                     for controls in self._derivative_controls(points))

    def commands(self, points, yaw=None):
        """Return (batch, samples, 11): P, V, A, yaw, zero yaw rate."""
        position, velocity, acceleration = self.evaluate(points)
        if yaw is None:
            yaw = points.new_zeros(points.shape[0])
        self._check(yaw, (points.shape[0],), 'Yaw')
        yaw = yaw[:, None, None].expand(-1, self.sample_count, 1)
        return torch.cat((position, velocity, acceleration, yaw, torch.zeros_like(yaw)), dim=-1)

    def forward(self, start_position, end_position, shape_points,
                start_velocity=None, start_acceleration=None, yaw=None):
        points = self.control_points(start_position, end_position, shape_points,
                                     start_velocity, start_acceleration)
        return self.commands(points, yaw)
