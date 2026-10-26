"""Attachment-anchored rigid/shape velocity decomposition for offline review.

This is a geometric projection, not causal energy attribution, a whip classifier,
or an enabled reward. Rest-length quadrature weights are not physical masses.
"""
import numpy as np

from aerowhip.evaluation.whip_measurements import _finite_array, _norm


def decompose(positions, velocities, rest_lengths):
    """Fit one angular velocity about the moving attachment at each sample.

    r_i = p_i-p_0; v_rel_i = v_i-v_0.
    omega minimizes sum_i w_i |v_rel_i - omega cross r_i|^2.
    Shape velocity is the residual. Under an arbitrary superposed time-dependent
    rigid motion, it rotates with the chain and its norm is unchanged. The
    least-squares angular velocity may be nonunique for collinear nodes; fitted
    nodal velocities remain unique. No threshold or temporal smoothing is used.
    """
    p = _finite_array(positions, 'positions')
    v = _finite_array(velocities, 'velocities')
    rest = _finite_array(rest_lengths, 'rest_lengths')
    if p.ndim != 3 or p.shape[-1] != 3 or p.shape[1] < 3 or not len(p) or v.shape != p.shape:
        raise ValueError('positions/velocities need matching (time,nodes>=3,3) shapes')
    if rest.shape != (p.shape[1]-1,) or np.any(rest <= 0):
        raise ValueError('Need one positive rest length per edge')
    if np.any(_norm(np.diff(p, axis=1)) <= 0):
        raise ValueError('Measured cable edges must have positive lengths')
    weights = np.r_[rest[0], rest[:-1]+rest[1:], rest[-1]]/(2*rest.sum())
    if not np.isfinite(weights).all() or np.any(weights <= 0):
        raise ValueError('Unusable rest-length scale')
    root = p[:, :1]
    offsets, relative = p-root, v-v[:, :1]
    if not np.isfinite(offsets).all() or not np.isfinite(relative).all():
        raise ValueError('Unusable relative coordinate scale')
    omega = np.empty((len(p), 3))
    ranks = np.empty(len(p), dtype=int)
    rigid = np.empty_like(relative)
    sqrt_weights = np.repeat(np.sqrt(weights), 3)
    for frame, (r, vr) in enumerate(zip(offsets, relative)):
        x, y, z = r.T
        design = np.zeros((len(r), 3, 3))
        design[:, 0, 1], design[:, 0, 2] = z, -y
        design[:, 1, 0], design[:, 1, 2] = -z, x
        design[:, 2, 0], design[:, 2, 1] = y, -x
        omega[frame], _, ranks[frame], _ = np.linalg.lstsq(
            design.reshape(-1, 3)*sqrt_weights[:, None],
            vr.reshape(-1)*sqrt_weights, rcond=None)
        rigid[frame] = np.cross(omega[frame], r)
    residual = relative-rigid
    result = dict(angular_velocity_world_radps=omega, fit_rank=ranks,
                  relative_velocity_world_mps=relative,
                  fitted_rigid_velocity_world_mps=rigid,
                  shape_velocity_world_mps=residual,
                  shape_speed_mps=_norm(residual),
                  rest_length_weights=weights)
    if not all(np.isfinite(a).all() for a in result.values()):
        raise ValueError('Decomposition exceeds finite numeric range')
    return result
