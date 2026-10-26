"""Explicit drone travel cost, separate from cable-event measurements and physics."""
from dataclasses import dataclass
import math
import torch


@dataclass(frozen=True)
class MovementPenalty:
    mode: str
    weight_per_cable_length: float

    def __post_init__(self):
        if (self.mode != 'path_length_v1' or isinstance(self.weight_per_cable_length,bool)
                or not isinstance(self.weight_per_cable_length,(int,float))
                or not math.isfinite(self.weight_per_cable_length) or self.weight_per_cable_length <= 0):
            raise ValueError('Movement penalty requires path_length_v1 and a positive finite weight')

    def measure(self, positions, observed, cable_length):
        """100-Hz polyline path of world tracking position, not CoM or energy.

        Only adjacent samples in the finite executed prefix are charged. Full
        horizon includes settling and post-release drift. No time averaging,
        endpoint displacement substitution, discount, clipping or free allowance.
        """
        if (positions.ndim != 3 or positions.shape[-1] != 3
                or observed.shape != positions.shape[:2] or observed.dtype != torch.bool
                or not math.isfinite(float(cable_length)) or float(cable_length) <= 0):
            raise ValueError('Travel measurement requires positions, observed mask and positive cable length')
        prefix=observed.long().cumprod(1).bool()
        if not bool(torch.isfinite(positions[prefix]).all()):
            raise ValueError('Observed tracking positions must be finite')
        safe=torch.where(prefix[...,None],positions,torch.zeros_like(positions))
        distance=safe.diff(dim=1).norm(dim=-1)
        distance=torch.where(prefix[:,:-1] & prefix[:,1:],distance,0.)
        return distance, self.weight_per_cable_length*distance/cable_length
