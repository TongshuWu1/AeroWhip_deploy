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
    geometric_hit = (crossing & (lateral.abs() <= surface.width_m/2+position_eps)
                     & (vertical.abs() <= surface.height_m/2+position_eps)
                     & (forward > 0))
    qualified = (crossing & (lateral.abs() <= surface.width_m/2+position_eps)
                 & (vertical.abs() <= surface.height_m/2+position_eps)
                 & (angle <= math.radians(surface.maximum_entry_angle_deg)+eps)
                 & (forward >= minimum_speed-eps*max(1, minimum_speed)))
    next_time = torch.cat((time_s[1:], time_s[-1:]))
    elapsed = time_s-time_s[0]+fraction*(next_time-time_s)
    return dict(positions=p, velocities=v, fraction=fraction, crossing=crossing,
                qualified=qualified, geometric_hit=geometric_hit, distance=distance, normal_error=normal_error,
                lateral=lateral, vertical=vertical, forward=forward, speed=speed,
                angle=angle, elapsed=elapsed)


def crossing_precision_diagnostics(surface, valid):
    """All-lane first/best forward plane crossings, including rectangle misses.

    NaN means no crossing or invalid episode, never zero error. Best is selected
    within one episode; it is not an independent trial. Final tolerance is the
    approved +/-5cm PER AXIS, not a 5cm radial disk or workspace-excess gate.
    """
    crossing = surface['crossing'] & (surface['forward'] > 0)
    error = (surface['lateral'].square()+surface['vertical'].square()).sqrt()
    best = error.masked_fill(~crossing, float('inf')).argmin(1)
    first = surface['elapsed'].masked_fill(~crossing, float('inf')).argmin(1)
    exists = crossing.any(1) & valid
    take = lambda field,index: surface[field].gather(1,index[:,None])[:,0].masked_fill(~exists,float('nan'))
    position_eps = 16*torch.finfo(surface['positions'].dtype).eps*surface['positions'][:,:,-1].abs().amax(-1).clamp_min(1)
    hit = crossing & (surface['lateral'].abs() <= .05+position_eps) & (surface['vertical'].abs() <= .05+position_eps)
    result = dict(forward_plane_crossing_success=exists.to(error.dtype).masked_fill(~valid,float('nan')),
        forward_plane_crossing_count=crossing.sum(1).to(error.dtype).masked_fill(~valid,float('nan')),
        final_precision_hit_success=hit.any(1).to(error.dtype).masked_fill(~valid,float('nan')),
        final_precision_hit_count=hit.sum(1).to(error.dtype).masked_fill(~valid,float('nan')))
    for label,index in (('first',first),('best',best)):
        for field,name in (('lateral','lateral_error_m'),('vertical','vertical_error_m'),
                           ('elapsed','time_s'),('forward','forward_speed_mps'),('angle','angle_rad')):
            result[f'{label}_crossing_{name}']=take(field,index)
        result[f'{label}_crossing_center_error_m']=error.gather(1,index[:,None])[:,0].masked_fill(~exists,float('nan'))
    return result
