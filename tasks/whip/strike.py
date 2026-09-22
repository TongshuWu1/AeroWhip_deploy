"""Virtual strike-surface measurements. No reward bonus or collision response."""

import math

import torch


def surface_events(positions, velocities, time_s, center, normal, surface, minimum_speed):
    """Interpolate forward crossings; retain other state samples for shaping.

    Input histories (B,T,N,3) must already be validated/sanitized by the scorer.
    Each interval contributes its forward plane crossing, if any, otherwise
    its start sample. The final state is also retained. Interpolation of the
    full cable position/velocity permits same-event non-rigid measurements.
    It is not a physics substep or a between-step safety guarantee.
    """
    offset = ((positions[:, :, -1]-center)*normal).sum(-1)
    crossing = (offset[:, :-1] < 0) & (offset[:, 1:] >= 0)
    change = offset[:, 1:]-offset[:, :-1]
    denominator = torch.where(crossing, change, torch.ones_like(change))
    fraction = torch.where(crossing, -offset[:, :-1]/denominator, 0).clamp(0, 1)
    fraction = torch.cat((fraction, torch.zeros_like(fraction[:, :1])), 1)
    crossing = torch.cat((crossing, torch.zeros_like(crossing[:, :1])), 1)
    p_next = torch.cat((positions[:, 1:], positions[:, -1:]), 1)
    v_next = torch.cat((velocities[:, 1:], velocities[:, -1:]), 1)
    p = positions+fraction[..., None, None]*(p_next-positions)
    v = velocities+fraction[..., None, None]*(v_next-velocities)
    velocity, delta = v[:, :, -1], p[:, :, -1]-center
    side = torch.stack((-normal[1], normal[0], normal.new_zeros(())))
    lateral, vertical = (delta*side).sum(-1), delta[..., 2]
    normal_error = (delta*normal).sum(-1)
    distance = torch.stack((normal_error, (lateral.abs()-surface.width_m/2).clamp_min(0),
                            (vertical.abs()-surface.height_m/2).clamp_min(0)), -1).norm(dim=-1)
    forward, speed = (velocity*normal).sum(-1), velocity.norm(dim=-1)
    angle = torch.atan2(torch.linalg.vector_norm(torch.linalg.cross(
        velocity, normal.expand_as(velocity), dim=-1), dim=-1), forward)
    angle = torch.where(speed > 0, angle, torch.full_like(angle, math.pi))
    # Inclusive boundaries with dtype roundoff only, not a physical tolerance.
    eps = 16*torch.finfo(positions.dtype).eps
    position_eps = eps*torch.maximum(p[:, :, -1].abs().amax(-1), center.abs().amax()).clamp_min(1)
    qualified = (crossing & (lateral.abs() <= surface.width_m/2+position_eps)
                 & (vertical.abs() <= surface.height_m/2+position_eps)
                 & (angle <= math.radians(surface.maximum_entry_angle_deg)+eps)
                 & (forward >= minimum_speed-eps*max(1, minimum_speed)))
    next_time = torch.cat((time_s[1:], time_s[-1:]))
    elapsed = time_s-time_s[0]+fraction*(next_time-time_s)
    return dict(positions=p, velocities=v, fraction=fraction, crossing=crossing,
                qualified=qualified, distance=distance, normal_error=normal_error,
                lateral=lateral, vertical=vertical, forward=forward, speed=speed,
                angle=angle, elapsed=elapsed)
