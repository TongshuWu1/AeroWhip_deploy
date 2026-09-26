"""Task returns and sampled simulation-envelope checks, separate from physics."""

from dataclasses import dataclass
import math

import torch

from .metrics import WhipMetrics
from .peak_scoring import PeakWhipScorer
from .scoring import WhipScore


@dataclass(frozen=True)
class WhipReturnConfiguration:
    acceleration_cost_weight: float
    failure_return: float
    drone_lower_world_m: tuple[float, float, float]
    drone_upper_world_m: tuple[float, float, float]
    maximum_tilt_deg: float
    minimum_cable_height_m: float

    def __post_init__(self):
        for name in ('acceleration_cost_weight', 'failure_return', 'maximum_tilt_deg',
                     'minimum_cable_height_m'):
            if not math.isfinite(getattr(self, name)):
                raise ValueError(f'{name} must be finite')
        if (self.acceleration_cost_weight < 0 or not 0 < self.maximum_tilt_deg < 90
                or self.failure_return >= -self.acceleration_cost_weight):
            raise ValueError('Require a nonnegative cost, tilt in (0,90), and failure below valid-return range')
        for name in ('drone_lower_world_m', 'drone_upper_world_m'):
            values = tuple(getattr(self, name))
            if len(values) != 3 or not all(math.isfinite(x) for x in values):
                raise ValueError(f'{name} must contain three finite values')
            object.__setattr__(self, name, values)
        if not all(a < b for a, b in zip(self.drone_lower_world_m, self.drone_upper_world_m)):
            raise ValueError('Envelope lower bounds must be below upper bounds')


@dataclass(frozen=True)
class WhipReturn:
    returns: torch.Tensor
    valid: torch.Tensor
    numerical_valid: torch.Tensor
    envelope_valid: torch.Tensor
    score_valid: torch.Tensor
    score: WhipScore
    acceleration_cost: torch.Tensor
    closest_distance_m: torch.Tensor
    selected_tip_speed_mps: torch.Tensor


class WhipReward:
    def __init__(self, environment, scoring_configuration, return_configuration):
        self.environment = environment
        self.configuration = return_configuration
        model = environment.simulator.model
        trajectory = environment.configuration
        self.intervals = round(trajectory.duration_s/model.dt)
        time = torch.arange(self.intervals+environment.post_hold_steps+1,
                            device=model.rest.device, dtype=model.rest.dtype)*model.dt
        delay = environment.simulator.delay_steps*model.dt
        if trajectory.post_hold_s+1e-12 < delay+scoring_configuration.tip_window_s:
            raise ValueError('Post-hold must cover command delay and the full tip-scoring window')
        self.metrics = WhipMetrics(model.rest, time, region_boundaries_m=
                                  model.tensor(scoring_configuration.region_fractions)*model.rest.sum())
        self.scorer = PeakWhipScorer(time, scoring_configuration)
        self.window = model.tensor([delay, delay+trajectory.duration_s])[None].expand(
            environment.simulator.batch_size, -1)
        self.lower = model.tensor(return_configuration.drone_lower_world_m)
        self.upper = model.tensor(return_configuration.drone_upper_world_m)
        self.tilt_cosine = math.cos(math.radians(return_configuration.maximum_tilt_deg))
        self.height = return_configuration.minimum_cable_height_m
        self.acceleration_scale = trajectory.max_acceleration_mps2

    @torch.no_grad()
    def __call__(self, episode):
        rollout = episode.rollout
        measured = self.metrics(rollout.cable_positions, rollout.cable_velocities, rollout.state_valid)
        score = self.scorer(measured.region_rms_curvature_per_m*self.metrics.s[-1],
            rollout.cable_positions[:, :, -1], rollout.cable_velocities[:, :, -1],
            measured.sample_valid, self.window, episode.target_positions)
        numerical = episode.valid
        envelope = (((rollout.drone_positions >= self.lower) & (rollout.drone_positions <= self.upper))
                    .flatten(1).all(-1)
                    & (rollout.drone_rotations[:, :, 2, 2] >= self.tilt_cosine).all(-1)
                    & (rollout.cable_positions[:, :, :, 2] >= self.height).flatten(1).all(-1))
        # Left-endpoint command samples cover exactly the maneuver, no hold.
        cost = self.configuration.acceleration_cost_weight*(
            episode.commands[:, :self.intervals, 6:9]/self.acceleration_scale).square().sum(-1).mean(-1)
        score_valid = score.valid & torch.isfinite(cost)
        valid = numerical & envelope & score_valid
        returns = torch.where(valid, score.score-cost, self.configuration.failure_return)
        distance = torch.linalg.vector_norm(rollout.cable_positions[:, :, -1]
                                           -episode.target_positions[:, None], dim=-1)
        closest = distance.amin(-1).masked_fill(~numerical, float('nan'))
        index = score.event_indices[:, 3].clamp_min(0)
        speed = measured.tip_speed_mps.gather(1, index[:, None])[:, 0]
        speed = speed.masked_fill(~score.has_event, float('nan'))
        return WhipReturn(returns, valid, numerical, envelope, score_valid, score, cost, closest, speed)
