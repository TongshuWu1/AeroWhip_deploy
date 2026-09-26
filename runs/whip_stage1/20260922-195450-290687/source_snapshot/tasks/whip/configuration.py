"""Explicit maneuver settings; no reward or task-success definition."""

from dataclasses import dataclass
import math
import operator


@dataclass(frozen=True)
class TrainingConfiguration:
    duration_s: float
    post_hold_s: float
    offset_lower_m: tuple[float, float, float]
    offset_upper_m: tuple[float, float, float]
    max_speed_mps: float
    max_acceleration_mps2: float
    control_count: int = 32

    def __post_init__(self):
        if operator.index(self.control_count) < 6:
            raise ValueError('At least six total control points are required')
        for name in ('duration_s', 'max_speed_mps', 'max_acceleration_mps2'):
            value = getattr(self, name)
            if not math.isfinite(value) or value <= 0:
                raise ValueError(f'{name} must be positive and finite')
        if not math.isfinite(self.post_hold_s) or self.post_hold_s < 0:
            raise ValueError('post_hold_s must be finite and nonnegative')
        for name in ('offset_lower_m', 'offset_upper_m'):
            values = tuple(getattr(self, name))
            if len(values) != 3 or not all(math.isfinite(x) for x in values):
                raise ValueError(f'{name} must contain three finite world-XYZ offsets')
            object.__setattr__(self, name, values)
        if not all(a < b for a, b in zip(self.offset_lower_m, self.offset_upper_m)):
            raise ValueError('Each lower offset must be strictly below its upper offset')
        for value in (self.duration_s, self.post_hold_s):
            if not math.isclose(value/.01, round(value/.01), rel_tol=0, abs_tol=1e-8):
                raise ValueError('Durations must be exact multiples of the 100 Hz timestep')

    @property
    def action_size(self):
        return 3*(self.control_count-6)+3
