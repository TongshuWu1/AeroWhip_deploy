"""Reach-first task scoring; no changes to physics or the legacy whip score."""

from dataclasses import dataclass
import math

import torch

from .reward import WhipReturn


@dataclass(frozen=True)
class ReachConfiguration:
    window_width_m: float
    window_height_m: float
    whip_bonus_weight: float

    def __post_init__(self):
        for name in ('window_width_m', 'window_height_m'):
            if not math.isfinite(getattr(self, name)) or getattr(self, name) <= 0:
                raise ValueError(f'{name} must be positive and finite')
        if not math.isfinite(self.whip_bonus_weight) or not 0 <= self.whip_bonus_weight <= 1:
            raise ValueError('Whip bonus weight must be finite and between zero and one')


def window_distance(positions, initial_positions, targets, configuration):
    """Distance to a finite vertical rectangle, not a sphere or half-space.

    The horizontal normal points from initial drone XY toward target XY. The
    rectangle is centered on the target and extends sideways and vertically.
    All inputs are batched, with positions shaped (batch, samples, 3).
    """
    offset = targets[:, :2]-initial_positions[:, :2]
    normal = offset/torch.linalg.vector_norm(offset, dim=-1, keepdim=True)
    side = torch.stack((-normal[:, 1], normal[:, 0]), dim=-1)
    delta = positions-targets[:, None]
    forward = (delta[..., :2]*normal[:, None]).sum(-1)
    lateral = ((delta[..., :2]*side[:, None]).sum(-1).abs()
               -configuration.window_width_m/2).clamp_min(0)
    vertical = (delta[..., 2].abs()-configuration.window_height_m/2).clamp_min(0)
    return torch.stack((forward, lateral, vertical), dim=-1).norm(dim=-1)


@dataclass(frozen=True)
class ReachScore:
    valid: torch.Tensor
    score: torch.Tensor
    progress: torch.Tensor
    bonus: torch.Tensor
    quality: torch.Tensor
    bend: torch.Tensor
    speed_score: torch.Tensor
    tip_speed_mps: torch.Tensor
    closest_window_distance_m: torch.Tensor
    selected_window_distance_m: torch.Tensor
    selected_center_distance_m: torch.Tensor
    selected_extension_m: torch.Tensor
    selected_index: torch.Tensor
    selected_time_s: torch.Tensor


@torch.no_grad()
def score_reach(time, tip_positions, tip_velocities, sample_valid, quality_history,
                initial_positions, targets, event_windows, tip_window_s,
                workspace_radius_m, speed_reference_mps, configuration):
    """Progress plus a bounded same-sample quality bonus, with explicit timing.

    quality_history is the existing unweighted B*V curve, already restricted to
    eligible peak associations. Reach does not require a positive B*V event.
    Values below initial reach get zero progress, not a bonus for fast motion.
    """
    batch, count, _ = tip_positions.shape
    if (tip_positions.shape != (batch, len(time), 3) or tip_velocities.shape != tip_positions.shape
            or sample_valid.shape != (batch, count) or quality_history.shape != (batch, count)
            or initial_positions.shape != (batch, 3) or targets.shape != (batch, 3)
            or event_windows.shape != (batch, 2)):
        raise ValueError('Reach histories, initial positions, targets and windows must agree')
    distance = window_distance(tip_positions, initial_positions, targets, configuration)
    speed = tip_velocities.norm(dim=-1)
    initial_distance = distance[:, 0]
    target_offset = targets[:, :2]-initial_positions[:, :2]
    target_range = target_offset.norm(dim=-1)
    valid = (sample_valid.all(-1) & torch.isfinite(tip_positions).flatten(1).all(-1)
             & torch.isfinite(tip_velocities).flatten(1).all(-1)
             & torch.isfinite(speed).all(-1)
             & torch.isfinite(quality_history).all(-1)
             & ((quality_history >= 0) & (quality_history <= 1)).all(-1)
             & torch.isfinite(distance).all(-1) & (initial_distance > 0)
             & torch.isfinite(initial_positions).all(-1) & torch.isfinite(targets).all(-1)
             & (target_range > workspace_radius_m)
             & torch.isfinite(event_windows).all(-1)
             & (event_windows[:, 0] <= event_windows[:, 1]))
    denominator = torch.where(valid, initial_distance, torch.ones_like(initial_distance))
    progress = (1-distance/denominator[:, None]).clamp(0, 1)
    bonus = configuration.whip_bonus_weight*progress*quality_history
    tolerance = 16*torch.finfo(time.dtype).eps*time.abs().amax().clamp_min(1.)
    permitted = ((time[None] >= event_windows[:, :1]-tolerance)
                 & (time[None] <= event_windows[:, 1:]+tip_window_s+tolerance))
    valid &= permitted.any(-1)
    history = (progress+bonus).masked_fill(~permitted | ~valid[:, None], -float('inf'))
    value, index = history.max(-1)
    speed_score = speed/(speed+speed_reference_mps)
    bend = torch.where(speed_score > 0,
        quality_history/speed_score.clamp_min(torch.finfo(speed_score.dtype).tiny), 0.)
    normal = target_offset/target_range[:, None]
    extension = ((tip_positions[..., :2]-initial_positions[:, None, :2])*normal[:, None]).sum(-1)
    extension = extension-workspace_radius_m

    def selected(values):
        return values.gather(1, index[:, None])[:, 0].masked_fill(~valid, float('nan'))

    return ReachScore(valid, value.masked_fill(~valid, float('nan')), selected(progress),
        selected(bonus), selected(quality_history), selected(bend), selected(speed_score),
        selected(speed), distance.masked_fill(~permitted, float('inf')).amin(-1).masked_fill(~valid, float('nan')),
        selected(distance), selected((tip_positions-targets[:, None]).norm(dim=-1)),
        selected(extension), index.masked_fill(~valid, -1),
        time[index].masked_fill(~valid, float('nan')))


@dataclass(frozen=True)
class ReachReturn(WhipReturn):
    score: ReachScore


class ReachReward:
    """Wrap the unchanged workspace reward, retaining its exact validity/cost.

    The legacy history is Q*(w_whip+w_target*A). Dividing by that known,
    strictly positive factor recovers the same-sample Q history without a
    second peak search. Legacy B/V/event summaries are not relabeled as reach.
    """
    def __init__(self, base_reward, configuration):
        self.base = base_reward
        self.configuration = configuration

    @torch.no_grad()
    def __call__(self, episode):
        base = self.base(episode)
        rollout = episode.rollout
        cfg = self.base.scorer.configuration
        positions, velocities = rollout.cable_positions[:, :, -1], rollout.cable_velocities[:, :, -1]
        center_distance = (positions-episode.target_positions[:, None]).norm(dim=-1)
        accuracy = torch.exp(-.5*(center_distance/cfg.target_width_m).square())
        quality = base.score.score_history/(cfg.whip_weight+cfg.target_weight*accuracy)
        reach = score_reach(rollout.time_s, positions, velocities, rollout.state_valid, quality,
            self.base.environment.start_position, episode.target_positions, self.base.window,
            cfg.tip_window_s, self.base.environment.workspace.radius_m,
            cfg.speed_reference_mps, self.configuration)
        valid = base.valid & reach.valid
        returns = torch.where(valid, reach.score-base.acceleration_cost, self.base.configuration.failure_return)
        return ReachReturn(**{**base.__dict__, 'returns': returns, 'valid': valid, 'score': reach,
            'score_valid': base.score_valid & reach.valid, 'selected_tip_speed_mps': reach.tip_speed_mps})


def statistics(result):
    """Reach components are measured at the selected reach event, not a global peak."""
    reach, valid = result.score, result.valid

    def average(value):
        value = float(value.masked_fill(~valid, float('nan')).nanmean())
        return value if math.isfinite(value) else None

    return dict(score_mean=average(reach.score), reach_progress_mean=average(reach.progress),
        reach_bonus_mean=average(reach.bonus), quality_mean=average(reach.quality),
        bend_mean=average(reach.bend), speed_score_mean=average(reach.speed_score),
        selected_tip_speed_mean_mps=average(reach.tip_speed_mps),
        closest_window_distance_mean_m=average(reach.closest_window_distance_m),
        selected_window_distance_mean_m=average(reach.selected_window_distance_m),
        selected_distance_mean_m=average(reach.selected_center_distance_m),
        selected_extension_mean_m=average(reach.selected_extension_m),
        reach_time_mean_s=average(reach.selected_time_s))


def record_arrays(result):
    reach = result.score
    return dict(score=reach.score, bend=reach.bend, speed=reach.speed_score,
        reach_progress=reach.progress, reach_bonus=reach.bonus, quality=reach.quality,
        closest_window_distance_m=reach.closest_window_distance_m,
        selected_window_distance_m=reach.selected_window_distance_m,
        selected_distance_m=reach.selected_center_distance_m,
        selected_extension_m=reach.selected_extension_m,
        selected_tip_speed_mps=reach.tip_speed_mps,
        reach_indices=reach.selected_index, reach_times_s=reach.selected_time_s)
