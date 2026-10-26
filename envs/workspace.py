"""Cylindrical drone operating region for task costs and metrics; no wall forces or state correction."""

from dataclasses import dataclass
import math

import torch


@dataclass(frozen=True)
class CylinderWorkspace:
    radius_m: float
    height_lower_world_m: float
    height_upper_world_m: float

    def __post_init__(self):
        if (not all(math.isfinite(x) for x in (
                self.radius_m, self.height_lower_world_m, self.height_upper_world_m))
                or self.radius_m <= 0 or self.height_lower_world_m >= self.height_upper_world_m):
            raise ValueError('Workspace requires a positive radius and increasing finite world heights')

    def excess(self, positions, initial_positions):
        """Radial and vertical distance outside the workspace [m]; zero inside."""
        radial = ((positions[..., :2]-initial_positions[:, None, :2]).norm(dim=-1)-self.radius_m).clamp_min(0)
        vertical = ((self.height_lower_world_m-positions[..., 2]).clamp_min(0)
                    +(positions[..., 2]-self.height_upper_world_m).clamp_min(0))
        return radial, vertical

    def contains(self, positions, initial_positions):
        """(batch,samples,3) world tracking positions; fixed initial XY center.

        Boundaries are inclusive. No clipping, growing radius or moving center.
        Nonfinite positions never pass. The cable is not confined to this volume.
        """
        relative_xy = positions[..., :2]-initial_positions[:, None, :2]
        return (torch.isfinite(positions).all(-1)
                & (relative_xy.square().sum(-1) <= self.radius_m**2)
                & (positions[..., 2] >= self.height_lower_world_m)
                & (positions[..., 2] <= self.height_upper_world_m))

    def description(self, initial_position):
        return dict(radius_m=self.radius_m,
                    center_xy_world_m=[float(x) for x in initial_position[:2]],
                    height_lower_world_m=self.height_lower_world_m,
                    height_upper_world_m=self.height_upper_world_m)
