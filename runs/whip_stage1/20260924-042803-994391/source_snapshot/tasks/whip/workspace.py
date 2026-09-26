"""Optional cylindrical task constraints; no wall forces or state correction."""

from dataclasses import dataclass, replace
import math

import torch

from .environment import WhipEnvironment
from .reward import WhipReward


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


class WorkspaceWhipEnvironment(WhipEnvironment):
    def __init__(self, configuration, simulator, workspace):
        super().__init__(configuration, simulator)
        self.workspace = workspace

    def decode(self, actions):
        points, commands, action_valid, valid, speed, acceleration = super().decode(actions)
        # A B-spline is inside its control-point convex hull; a cylinder is convex.
        # This conservative continuous-command check can reject an inside curve
        # with outside controls. It does not silently remap the policy's action.
        inside = self.workspace.contains(points, self.start_position).all(-1)
        return points, commands, action_valid, valid & inside, speed, acceleration


class WorkspaceWhipReward(WhipReward):
    @torch.no_grad()
    def __call__(self, episode):
        result = super().__call__(episode)
        inside = self.environment.workspace.contains(
            episode.rollout.drone_positions, self.environment.start_position).all(-1)
        valid = result.valid & inside
        return replace(result, valid=valid, envelope_valid=result.envelope_valid & inside,
                       returns=torch.where(valid, result.returns, self.configuration.failure_return))
