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
            if isinstance(value, bool) or not math.isfinite(value) or (value < 0 if name == 'post_hold_s' else value <= 0):
                raise ValueError(f'{name} must be finite; duration positive and post-hold nonnegative')




@dataclass(frozen=True)
class WhipReturnConfiguration:
    acceleration_cost_weight: float
    failure_return: float | None
    maximum_tilt_deg: float
    minimum_cable_height_m: float
    acceleration_cost_scale_mps2: float
    failure_cost_per_remaining_s: float | None = None

    def __post_init__(self):
        for name in ('acceleration_cost_weight', 'maximum_tilt_deg', 'minimum_cable_height_m'):
            if not math.isfinite(getattr(self,name)):raise ValueError(f'{name} must be finite')
        if self.acceleration_cost_weight<0 or not 0<self.maximum_tilt_deg<90:
            raise ValueError('Require a nonnegative cost and tilt in (0,90)')
        if self.failure_cost_per_remaining_s is None:
            if self.failure_return is None or not math.isfinite(self.failure_return) or self.failure_return>=-self.acceleration_cost_weight:
                raise ValueError('Fixed failure return must be finite and below -acceleration_cost_weight')
        elif (self.failure_return is not None or isinstance(self.failure_cost_per_remaining_s,bool)
                or not math.isfinite(self.failure_cost_per_remaining_s) or self.failure_cost_per_remaining_s<=0):
            raise ValueError('Remaining-time failure cost requires positive rate and no fixed failure return')
        value = self.acceleration_cost_scale_mps2
        if isinstance(value, bool) or not math.isfinite(value) or value <= 0:
            raise ValueError('acceleration_cost_scale_mps2 must be positive and finite')
