"""Task episode timing and physical-envelope settings; no controller or optimizer."""

from dataclasses import dataclass
import math


@dataclass(frozen=True)
class EpisodeTiming:
    duration_s: float

    def __post_init__(self):
        if isinstance(self.duration_s, bool) or not math.isfinite(self.duration_s) or self.duration_s <= 0:
            raise ValueError('duration_s must be finite and positive')


@dataclass(frozen=True)
class WhipReturnConfiguration:
    maximum_tilt_deg: float
    minimum_cable_height_m: float
    failure_cost_per_remaining_s: float

    def __post_init__(self):
        if not math.isfinite(self.minimum_cable_height_m) or not 0 < self.maximum_tilt_deg < 90:
            raise ValueError('Require a finite cable height and tilt in (0,90)')
        value = self.failure_cost_per_remaining_s
        if isinstance(value, bool) or not math.isfinite(value) or value <= 0:
            raise ValueError('Remaining-time failure cost must be positive and finite')
