"""Task episode timing and safety/cost settings; no controller or optimizer."""

from dataclasses import dataclass
import math



@dataclass(frozen=True)
class EpisodeTiming:
    duration_s: float
    post_hold_s: float

    def __post_init__(self):
        for name in ('duration_s', 'post_hold_s'):
            value = getattr(self, name)
            if isinstance(value, bool) or not math.isfinite(value) or value <= 0:
                raise ValueError(f'{name} must be positive and finite')


@dataclass(frozen=True)
class LearnedRecoveryConfiguration:
    """Explicit nominal Stage4 reward/timing, not a scripted controller."""
    whip_duration_s: float
    recovery_duration_s: float
    weight: float
    position_scale_m: float
    velocity_scale_mps: float
    final_hold_s: float
    position_tolerance_m: float
    speed_tolerance_mps: float
    timing_mode: str = 'fixed_window_v1'
    quality_mode: str = 'gaussian_v1'
    return_position_scale_m: float = 1.0
    return_velocity_scale_mps: float = 0.5
    cable_velocity_scale_mps: float = 1.0
    cable_horizontal_scale_m: float = 0.25
    cable_speed_tolerance_mps: float = 0.2
    cable_horizontal_tolerance_m: float = 0.1

    def __post_init__(self):
        for name, value in vars(self).items():
            if name in ('timing_mode','quality_mode'):
                continue
            if isinstance(value, bool) or not math.isfinite(value) or value <= 0:
                raise ValueError('Recovery requires positive finite '+name)
        if self.weight > 1 or self.final_hold_s > self.recovery_duration_s:
            raise ValueError('Recovery weight <=1 and hold must fit recovery window')
        for name in ('whip_duration_s', 'recovery_duration_s', 'final_hold_s'):
            if abs(getattr(self,name)/.01-round(getattr(self,name)/.01)) > 1e-8:
                raise ValueError('Recovery timing requires the 100Hz grid')
        if self.timing_mode not in ('fixed_window_v1','after_first_geometric_hit_v1'):
            raise ValueError('Unknown recovery timing')
        if self.quality_mode not in ('gaussian_v1','reciprocal_return_gaussian_hold_v1','return_and_cable_settle_v1'):
            raise ValueError('Unknown bounded recovery quality')
        if self.quality_mode=='return_and_cable_settle_v1' and self.timing_mode!='after_first_geometric_hit_v1':
            raise ValueError('Cable settling requires event-relative recovery')


@dataclass(frozen=True)
class WhipReturnConfiguration:
    acceleration_cost_weight: float
    failure_return: float
    maximum_tilt_deg: float
    minimum_cable_height_m: float
    acceleration_cost_scale_mps2: float

    def __post_init__(self):
        for name in ('acceleration_cost_weight', 'failure_return', 'maximum_tilt_deg',
                     'minimum_cable_height_m'):
            if not math.isfinite(getattr(self, name)):
                raise ValueError(f'{name} must be finite')
        if (self.acceleration_cost_weight < 0 or not 0 < self.maximum_tilt_deg < 90
                or self.failure_return >= -self.acceleration_cost_weight):
            raise ValueError('Require a nonnegative cost, tilt in (0,90), and failure_return < -acceleration_cost_weight')
        value = self.acceleration_cost_scale_mps2
        if isinstance(value, bool) or not math.isfinite(value) or value <= 0:
            raise ValueError('acceleration_cost_scale_mps2 must be positive and finite')
