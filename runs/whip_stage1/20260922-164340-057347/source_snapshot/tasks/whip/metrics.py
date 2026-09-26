"""Descriptive cable-motion measurements, not a whip detector or reward."""

from dataclasses import dataclass

import torch


@dataclass(frozen=True)
class WhipMeasurements:
    """Batch-major histories; invalid/undefined values are NaN, never zero-filled.

    B = episodes, T = state samples, N = cable nodes (attachment first).
    Curvature has shape (B,T,N-2); regional histories (B,T,3) are ordered
    proximal, middle, distal. Scalar histories have shape (B,T), except centroid
    rate (B,T-1), located at interval_time_s. Coordinates are rest arclength.

    Speed is in m/s. Relative speed subtracts attachment velocity, not rigid-body
    rotation: a rotating fixed shape can still have large relative speeds.
    All time histories describe the supplied observation window. There is no
    automatic event selection, peak matching, success flag, or energy estimate.
    """

    time_s: torch.Tensor
    interval_time_s: torch.Tensor
    node_arclength_m: torch.Tensor
    region_boundaries_m: torch.Tensor
    region_bend_support_m: torch.Tensor
    sample_valid: torch.Tensor
    turning_angle_rad: torch.Tensor
    curvature_per_m: torch.Tensor
    region_rms_curvature_per_m: torch.Tensor
    total_turn_rad: torch.Tensor
    squared_curvature_integral_per_m: torch.Tensor
    bend_centroid_m: torch.Tensor
    bend_spread_m: torch.Tensor
    bend_centroid_rate_mps: torch.Tensor
    region_rms_speed_mps: torch.Tensor
    region_relative_rms_speed_mps: torch.Tensor
    tip_speed_mps: torch.Tensor
    tip_relative_speed_mps: torch.Tensor


class WhipMetrics:
    """Cache a material mesh/time grid; measure batches without host synchronization.

    Supply the rollout's rest lengths and time grid, and FOUR explicit region
    boundaries [0, proximal_end, middle_end, L] in metres. No region sizes,
    smoothing scale, whip threshold, or event window are silently selected.
    Region averages use material-length weights, not counts of simulation nodes.
    Construction validates values (and may synchronize CUDA); calls only check
    tensor metadata on the host. Calls are forward-only, device-resident and
    vectorized over episodes, samples, nodes and regions.

    Usage:
        metrics = WhipMetrics(rollout.cable_rest_lengths, rollout.time_s,
                              region_boundaries_m=boundaries)
        measured = metrics(rollout.cable_positions, rollout.cable_velocities,
                           rollout.state_valid)

    Rebuild when mesh, timestamps, dtype, device or region boundaries change.
    Reuse for subsequent batches with the same mesh/time grid. Caller must pass
    matching positions/velocities; this interface does not realign recordings.
    """

    @torch.no_grad()
    def __init__(self, rest_lengths, time_s, *, region_boundaries_m):
        if (not isinstance(rest_lengths, torch.Tensor) or rest_lengths.ndim != 1
                or rest_lengths.numel() < 2
                or rest_lengths.dtype not in (torch.float32, torch.float64)):
            raise ValueError('Rest lengths must be a float32/float64 vector with at least two edges')
        if (not isinstance(time_s, torch.Tensor) or time_s.ndim != 1 or time_s.numel() < 2
                or time_s.device != rest_lengths.device or time_s.dtype != rest_lengths.dtype):
            raise ValueError('Times must be a vector of at least two samples on the mesh dtype/device')
        if not bool((torch.isfinite(rest_lengths) & (rest_lengths > 0)).all()):
            raise ValueError('Rest lengths must be finite and positive')
        if not bool(torch.isfinite(time_s).all() & torch.isfinite(torch.diff(time_s)).all()
                    & (torch.diff(time_s) > 0).all()):
            raise ValueError('Times must be finite and strictly increasing')
        self.rest = rest_lengths.detach().clone()
        self.time = time_s.detach().clone()
        self.s = torch.cat((self.rest.new_zeros(1), self.rest.cumsum(0)))
        boundaries = torch.as_tensor(region_boundaries_m, device=self.rest.device,
                                     dtype=self.rest.dtype).detach().clone()
        if (boundaries.shape != (4,) or not bool(torch.isfinite(boundaries).all())
                or not bool((torch.diff(boundaries) > 0).all())
                or not bool(boundaries[0] == 0)
                or not bool(torch.isclose(boundaries[-1], self.s[-1], rtol=0,
                                         atol=8*torch.finfo(self.rest.dtype).eps*float(self.s[-1])))):
            raise ValueError('Provide four increasing region boundaries spanning [0, cable rest length]')
        # Snap only endpoint roundoff; internal region boundaries are unchanged.
        boundaries[-1] = self.s[-1]
        self.boundaries = boundaries
        self.region_lengths = torch.diff(boundaries)
        self.dual = (self.rest[:-1]+self.rest[1:])*.5
        midpoints = (self.s[:-1]+self.s[1:])*.5
        left = torch.maximum(boundaries[:-1, None], midpoints[:-1])
        right = torch.minimum(boundaries[1:, None], midpoints[1:])
        self.bend_overlap = (right-left).clamp_min(0)
        self.bend_support = self.bend_overlap.sum(-1)

        # Exact material-coordinate integrals of squared piecewise-linear
        # velocity over each regional/edge overlap. u ranges from 0 to 1 on an
        # edge and v(u)=(1-u)*v_left+u*v_right.
        u0 = ((boundaries[:-1, None]-self.s[:-1])/self.rest).clamp(0, 1)
        u1 = ((boundaries[1:, None]-self.s[:-1])/self.rest).clamp(0, 1)
        d1, d2, d3 = u1-u0, u1.square()-u0.square(), u1.pow(3)-u0.pow(3)
        scale = self.rest/self.region_lengths[:, None]
        self.speed_left = (d1-d2+d3/3)*scale
        self.speed_cross = (d2-2*d3/3)*scale
        self.speed_right = d3/3*scale
        self.dt = torch.diff(self.time)
        self.interval_time = self.time[:-1]+self.dt*.5

    def _check(self, positions, velocities, state_valid):
        shape = (positions.shape[0], self.time.numel(), self.s.numel(), 3) if isinstance(
            positions, torch.Tensor) and positions.ndim == 4 else None
        for name, value in (('Positions', positions), ('Velocities', velocities)):
            if (not isinstance(value, torch.Tensor) or tuple(value.shape) != shape
                    or value.shape[0] < 1 or value.dtype != self.rest.dtype
                    or value.device != self.rest.device):
                raise ValueError(f'{name} must be (batch, times, nodes, 3) on the mesh dtype/device')
        if (not isinstance(state_valid, torch.Tensor) or state_valid.shape != positions.shape[:2]
                or state_valid.dtype != torch.bool or state_valid.device != self.rest.device):
            raise ValueError('State validity must be boolean (batch, times) on the mesh device')

    def _region_speed(self, velocity):
        left, right = velocity[..., :-1, :], velocity[..., 1:, :]
        square = (left.square().sum(-1) @ self.speed_left.T
                  + (left*right).sum(-1) @ self.speed_cross.T
                  + right.square().sum(-1) @ self.speed_right.T)
        # The integral is nonnegative; cancellation can yield tiny negatives.
        return square.clamp_min(0).sqrt()

    @torch.no_grad()
    def __call__(self, positions, velocities, state_valid):
        self._check(positions, velocities, state_valid)
        edges = torch.diff(positions, dim=2)
        lengths = torch.linalg.vector_norm(edges, dim=-1)
        valid = (state_valid & torch.isfinite(positions).flatten(2).all(-1)
                 & torch.isfinite(velocities).flatten(2).all(-1)
                 & torch.isfinite(lengths).all(-1) & (lengths > 0).all(-1))
        tangent = edges/torch.where(lengths > 0, lengths, torch.ones_like(lengths))[..., None]
        cross = torch.linalg.vector_norm(torch.linalg.cross(tangent[..., :-1, :],
                                                           tangent[..., 1:, :], dim=-1), dim=-1)
        dot = (tangent[..., :-1, :]*tangent[..., 1:, :]).sum(-1)
        angle = torch.atan2(cross, dot)
        curvature = angle/self.dual
        weight = curvature.square()*self.dual
        integral = weight.sum(-1)
        denominator = torch.where(integral > 0, integral, torch.ones_like(integral))
        centroid = (weight*self.s[1:-1]).sum(-1)/denominator
        spread = ((weight*(self.s[1:-1]-centroid[..., None]).square()).sum(-1)/denominator).sqrt()
        support = torch.where(self.bend_support > 0, self.bend_support, torch.ones_like(self.bend_support))
        regional_curvature = (curvature.square() @ self.bend_overlap.T/support).sqrt()
        relative_velocity = velocities-velocities[..., :1, :]
        regional_speed = self._region_speed(velocities)
        relative_speed = self._region_speed(relative_velocity)
        tip_speed = torch.linalg.vector_norm(velocities[..., -1, :], dim=-1)
        tip_relative = torch.linalg.vector_norm(relative_velocity[..., -1, :], dim=-1)

        # Overflow/nonfinite derived values must not masquerade as measurements.
        valid = (valid & torch.isfinite(curvature).all(-1) & torch.isfinite(integral)
                 & torch.isfinite(centroid) & torch.isfinite(spread)
                 & torch.isfinite(regional_curvature).all(-1)
                 & torch.isfinite(regional_speed).all(-1) & torch.isfinite(relative_speed).all(-1)
                 & torch.isfinite(tip_speed) & torch.isfinite(tip_relative))

        def masked(value, mask=valid):
            while mask.ndim < value.ndim:
                mask = mask.unsqueeze(-1)
            return torch.where(mask, value, float('nan'))

        bend_valid = valid & (integral > 0)
        centroid = masked(centroid, bend_valid)
        spread = masked(spread, bend_valid)
        rate = torch.diff(centroid, dim=1)/self.dt
        rate = masked(rate, bend_valid[:, 1:] & bend_valid[:, :-1] & torch.isfinite(rate))
        return WhipMeasurements(
            self.time.clone(), self.interval_time.clone(), self.s.clone(), self.boundaries.clone(),
            self.bend_support.clone(), valid, masked(angle), masked(curvature),
            masked(regional_curvature, valid[..., None] & (self.bend_support > 0)),
            masked(angle.sum(-1)), masked(integral), centroid, spread, rate,
            masked(regional_speed), masked(relative_speed), masked(tip_speed), masked(tip_relative))
