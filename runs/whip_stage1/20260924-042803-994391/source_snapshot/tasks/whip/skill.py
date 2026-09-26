"""Stage-1 skill requests: desired outcomes, not guaranteed feasible commands."""

from dataclasses import dataclass, fields
import math
from pathlib import Path
import tomllib

import torch


SKILL_PARAMETERS = (
    'heading_rad', 'elevation_rad', 'release_height_world_m',
    'tip_speed_mps', 'release_time_s',
)


@dataclass(frozen=True)
class SkillConfiguration:
    """Inclusive request bounds in SKILL_PARAMETERS order, not success limits.

    Infinite height/speed/time bounds explicitly leave capability limits open.
    Requests themselves must always be finite. Equal bounds fix a parameter.
    These bounds do not specify a sampling distribution or training curriculum.
    """

    heading_rad: tuple[float, float]
    elevation_rad: tuple[float, float]
    release_height_world_m: tuple[float, float]
    tip_speed_mps: tuple[float, float]
    release_time_s: tuple[float, float]

    def __post_init__(self):
        domains = ((-math.pi, math.pi), (-math.pi/2, math.pi/2),
                   (-math.inf, math.inf), (0, math.inf), (0, math.inf))
        for name, (domain_lower, domain_upper) in zip(SKILL_PARAMETERS, domains):
            pair = tuple(getattr(self, name))
            if (len(pair) != 2 or any(isinstance(x, bool) or not isinstance(x, (int, float))
                                      or math.isnan(x) for x in pair)
                    or pair[0] > pair[1] or pair[0] == math.inf or pair[1] == -math.inf
                    or pair[0] < domain_lower or pair[1] > domain_upper):
                raise ValueError(f'{name} requires ordered lower/upper bounds within its domain')
            object.__setattr__(self, name, pair)

    @classmethod
    def load(cls, path):
        with Path(path).open('rb') as stream:
            raw = tomllib.load(stream)
        if (set(raw) != {'request_bounds'}
                or set(raw['request_bounds']) != {field.name for field in fields(cls)}):
            raise ValueError('Provide only request_bounds with all five skill parameters')
        return cls(**raw['request_bounds'])


@dataclass(frozen=True)
class SkillRequest:
    """A (batch, 5) floating tensor in SKILL_PARAMETERS order.

    Heading: world +X toward +Y, radians in [-pi, pi]. Elevation: above world XY,
    radians in [-pi/2, pi/2]. Height: world Z in metres. Strength: world-frame
    tip-speed magnitude in m/s. Time: seconds since the first rollout sample.
    The requested direction is tip velocity, NOT drone yaw or cable orientation.

    Construction checks metadata only. SkillMetrics returns row-wise validity
    for nonfinite/out-of-bounds values; requests are never clipped or resampled.
    Initial drone/cable state belongs to the rollout, not to this request vector.
    No policy/state encoder or spline generator is implemented by this class.
    """

    values: torch.Tensor

    def __post_init__(self):
        if (not isinstance(self.values, torch.Tensor) or self.values.ndim != 2
                or self.values.shape[0] < 1 or self.values.shape[1] != len(SKILL_PARAMETERS)
                or self.values.dtype not in (torch.float32, torch.float64)):
            raise ValueError('Skill requests must be a float32/float64 tensor of shape (batch, 5)')

    @property
    def direction_world(self):
        heading, elevation = self.values[:, 0], self.values[:, 1]
        return torch.stack((elevation.cos()*heading.cos(), elevation.cos()*heading.sin(),
                            elevation.sin()), dim=-1)
