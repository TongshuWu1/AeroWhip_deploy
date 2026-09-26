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
    control_mappings = ('box', 'cylinder_v1')

    def __init__(self, configuration, simulator, workspace):
        super().__init__(configuration, simulator)
        self.workspace = workspace
        if configuration.control_mapping == 'cylinder_v1':
            # Keep configuration bounds meaningful rather than silently ignoring
            # them. XY is centered at each initial tracking position; Z is world.
            if (configuration.offset_lower_m[:2] != (-workspace.radius_m,)*2
                    or configuration.offset_upper_m[:2] != (workspace.radius_m,)*2):
                raise ValueError('Cylinder mapping requires XY offsets equal to +/- workspace radius')
            if not bool(((self.start_position[:, 2]+self.lower[2]-workspace.height_lower_world_m).abs() <= 1e-12).all()
                        & ((self.start_position[:, 2]+self.upper[2]-workspace.height_upper_world_m).abs() <= 1e-12).all()):
                raise ValueError('Cylinder mapping requires Z offsets to match workspace heights')

    def _control_locations(self, actions):
        if self.configuration.control_mapping == 'box':
            return super()._control_locations(actions)
        a = actions.reshape(len(actions), -1, 3)
        # Smooth square-to-disk map: x^2+y^2-x^2*y^2 <= 1 for |x|,|y|<=1.
        # Inward machine-roundoff margin keeps corner/edge actions inside the
        # unchanged strict cylinder test; it is not a physical safety margin.
        radius = self.workspace.radius_m*(1-8*torch.finfo(a.dtype).eps)
        xy = radius*a[..., :2]*torch.sqrt(1-a[..., [1, 0]].square()/2)
        z = (self.workspace.height_lower_world_m
             +(a[..., 2:3]+1)*.5*(self.workspace.height_upper_world_m-self.workspace.height_lower_world_m))
        return torch.cat((self.start_position[:, None, :2]+xy, z), dim=-1)

    def decode(self, actions):
        points, commands, action_valid, valid, speed, acceleration = super().decode(actions)
        # A B-spline is inside its control-point convex hull; a cylinder is convex.
        # Also retain this check for cylinder_v1: initial P/V/A controls are not
        # learned locations. Box mapping retains its conservative rejection.
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
