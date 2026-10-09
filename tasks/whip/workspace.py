"""Optional cylindrical task constraints; no wall forces or state correction."""

from dataclasses import dataclass
import math

import torch

@dataclass(frozen=True)
class WorkspacePenalty:
    weight: float
    distance_scale_m: float

    def __post_init__(self):
        if (isinstance(self.weight, bool) or not math.isfinite(self.weight) or self.weight < 0
                or isinstance(self.distance_scale_m, bool) or not math.isfinite(self.distance_scale_m)
                or self.distance_scale_m <= 0):
            raise ValueError('Workspace penalty needs nonnegative weight and positive distance scale')

    def distances(self, positions, initial_positions, workspace):
        radial = ((positions[..., :2]-initial_positions[:, None, :2]).norm(dim=-1)-workspace.radius_m).clamp_min(0)
        vertical = ((workspace.height_lower_world_m-positions[..., 2]).clamp_min(0)
                    +(positions[..., 2]-workspace.height_upper_world_m).clamp_min(0))
        return radial, vertical

    def charges(self, positions, initial_positions, workspace, observed):
        """Right-endpoint 100-Hz integration divided by the fixed full horizon.

        Includes settling, excludes initial sample and unexecuted suffixes.
        No position projection, command filtering or termination decision.
        """
        radial, vertical = self.distances(positions, initial_positions, workspace)
        density = self.weight*((radial/self.distance_scale_m).square()+(vertical/self.distance_scale_m).square())
        return torch.where(observed[:, 1:], density[:, 1:]/(positions.shape[1]-1), 0.)

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

    def validate_placement(self, initial_position, target_position):
        if not self.height_lower_world_m <= initial_position[2] <= self.height_upper_world_m:
            raise ValueError('Initial tracking position is outside the cylindrical workspace')
        if math.hypot(target_position[0]-initial_position[0], target_position[1]-initial_position[1]) <= self.radius_m:
            raise ValueError('The target must be horizontally outside the drone workspace')

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
