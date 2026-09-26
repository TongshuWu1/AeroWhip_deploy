"""Measure a caller-selected bend/release event without selecting or scoring it."""

from dataclasses import dataclass

import torch

from .metrics import WhipMetrics
from .skill import SKILL_PARAMETERS, SkillConfiguration, SkillRequest


@dataclass(frozen=True)
class SkillMeasurements:
    """Raw same-event measurements, not a whip label or reward.

    release_parameters and parameter_errors are (B,5), in SKILL_PARAMETERS order.
    Errors are measured minus requested; heading error wraps to [-pi, pi].
    Direction error is the unsigned 3-D angle between requested and measured
    tip velocity, avoiding heading ambiguity at the poles. At zero tip speed,
    direction/elevation/heading are undefined; at vertical speed only heading
    is undefined. Other release measurements remain available.

    All other outputs are (B,), except indices (B,3), XYZ vectors (B,3), and
    bend centroid/spread endpoints (B,2). Invalid/undefined values are NaN.
    event_valid checks index order and every sample from bend start to release,
    not safety or the remainder of the episode. bend_defined checks that the
    curvature centroid/spread exist throughout the bend window; neither mask
    establishes bend transport or whipping. Positive centroid shift alone can
    also result from stationary bends changing relative strength.
    """

    event_indices: torch.Tensor
    request_valid: torch.Tensor
    event_valid: torch.Tensor
    bend_defined: torch.Tensor
    direction_defined: torch.Tensor
    heading_defined: torch.Tensor
    release_parameters: torch.Tensor
    parameter_errors: torch.Tensor
    direction_error_rad: torch.Tensor
    tip_position_world_m: torch.Tensor
    tip_velocity_world_mps: torch.Tensor
    bend_centroid_endpoints_m: torch.Tensor
    bend_spread_endpoints_m: torch.Tensor
    bend_centroid_shift_m: torch.Tensor
    minimum_turn_rad: torch.Tensor


class SkillMetrics:
    """Reuse a WhipMetrics mesh/time grid and explicit request configuration.

    The caller supplies int64 (B,3) indices [bend_start, bend_end, release], with
    0 <= start < end <= release < T. Use -1 for missing events. This API neither
    finds an event nor assumes a supplied event is a physical whip. It does not
    combine independent maxima, look for a favorable release, or use a target.
    Release means a selected motion phase, not physical cable detachment.
    Calls are forward-only, batched, and keep tensors on their existing device.
    """

    def __init__(self, metrics: WhipMetrics, configuration: SkillConfiguration):
        self.metrics = metrics
        self.configuration = configuration
        bounds = metrics.rest.new_tensor([getattr(configuration, key) for key in SKILL_PARAMETERS])
        self.lower, self.upper = bounds[:, 0], bounds[:, 1]
        self.samples = torch.arange(metrics.time.numel(), device=metrics.rest.device)[None]

    @torch.no_grad()
    def __call__(self, request: SkillRequest, positions, velocities, state_valid, event_indices):
        measured = self.metrics(positions, velocities, state_valid)
        batch, times = positions.shape[:2]
        if (not isinstance(request, SkillRequest) or request.values.shape != (batch, 5)
                or request.values.dtype != positions.dtype or request.values.device != positions.device):
            raise ValueError('Requests must match the history batch, dtype and device')
        if (not isinstance(event_indices, torch.Tensor) or event_indices.shape != (batch, 3)
                or event_indices.dtype != torch.int64 or event_indices.device != positions.device):
            raise ValueError('Event indices must be int64 (batch, 3) on the history device')
        start, end, release = event_indices.unbind(-1)
        index_valid = (start >= 0) & (start < end) & (end <= release) & (release < times)
        event_window = (self.samples >= start[:, None]) & (self.samples <= release[:, None])
        bend_window = (self.samples >= start[:, None]) & (self.samples <= end[:, None])
        event_valid = index_valid & (~event_window | measured.sample_valid).all(-1)
        bend_defined = event_valid & (~bend_window | (
            torch.isfinite(measured.bend_centroid_m) & torch.isfinite(measured.bend_spread_m))).all(-1)

        # Clamping is only for memory-safe indexing; invalid events remain NaN.
        safe_indices = event_indices.clamp(0, times-1)
        rows = torch.arange(batch, device=positions.device)
        tip_position = positions[rows, safe_indices[:, 2], -1]
        tip_velocity = velocities[rows, safe_indices[:, 2], -1]
        speed = torch.linalg.vector_norm(tip_velocity, dim=-1)
        horizontal_speed = torch.linalg.vector_norm(tip_velocity[:, :2], dim=-1)
        direction_defined = event_valid & (speed > 0)
        heading_defined = direction_defined & (horizontal_speed > 0)

        def masked(value, mask):
            while mask.ndim < value.ndim:
                mask = mask.unsqueeze(-1)
            return torch.where(mask, value, float('nan'))

        heading = masked(torch.atan2(tip_velocity[:, 1], tip_velocity[:, 0]), heading_defined)
        elevation = masked(torch.atan2(tip_velocity[:, 2], horizontal_speed), direction_defined)
        elapsed = self.metrics.time[safe_indices[:, 2]]-self.metrics.time[0]
        parameters = masked(torch.stack((heading, elevation, tip_position[:, 2], speed, elapsed), -1),
                            event_valid)
        request_valid = (torch.isfinite(request.values).all(-1)
                         & ((request.values >= self.lower) & (request.values <= self.upper)).all(-1)
                         & (request.values[:, 4] <= self.metrics.time[-1]-self.metrics.time[0]))
        errors = parameters-request.values
        errors[:, 0] = torch.atan2(errors[:, 0].sin(), errors[:, 0].cos())
        errors = masked(errors, request_valid & event_valid)

        unit_velocity = tip_velocity/torch.where(speed > 0, speed, torch.ones_like(speed))[:, None]
        requested_direction = request.direction_world
        cross = torch.linalg.vector_norm(torch.linalg.cross(unit_velocity, requested_direction, dim=-1), dim=-1)
        dot = (unit_velocity*requested_direction).sum(-1)
        direction_error = masked(torch.atan2(cross, dot), request_valid & direction_defined)
        centroids = masked(measured.bend_centroid_m.gather(1, safe_indices[:, :2]), bend_defined)
        spreads = masked(measured.bend_spread_m.gather(1, safe_indices[:, :2]), bend_defined)
        minimum_turn = masked(torch.where(bend_window, measured.total_turn_rad, float('inf')).amin(-1),
                              event_valid)
        return SkillMeasurements(
            event_indices.clone(), request_valid, event_valid, bend_defined, direction_defined,
            heading_defined, parameters, errors, direction_error,
            masked(tip_position, event_valid), masked(tip_velocity, event_valid), centroids, spreads,
            centroids[:, 1]-centroids[:, 0], minimum_turn)
