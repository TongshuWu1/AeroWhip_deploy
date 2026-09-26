"""DDER geometry/projection equations extracted from simulator/cable/dder.py.

Shared DDER geometry and direct constraint projection equations.
No custom CUDA kernels or old-project imports.
"""
from __future__ import annotations
import torch

PinnedEndpointMask = tuple[bool | int, bool | int]
FREE_ENDPOINTS = (False, False)
TWO_PINNED_ENDPOINTS = (True, True)
START_PINNED_FREE_END = (True, False)
CLAMPED_START_FREE_END = (2, False)

def _validate_pinned_endpoints(
    pinned_endpoints: PinnedEndpointMask,
    *,
    supported_only: bool = False,
) -> PinnedEndpointMask:
    if not isinstance(pinned_endpoints, tuple) or len(pinned_endpoints) != 2:
        raise ValueError("pinned_endpoints must contain start/end pin counts.")
    if any(type(value) not in (bool, int) for value in pinned_endpoints):
        raise ValueError("Pinned-node counts must be booleans or integers.")
    normalized = (int(pinned_endpoints[0]), int(pinned_endpoints[1]))
    if any(value < 0 for value in normalized):
        raise ValueError("Pinned-node counts cannot be negative.")
    supported = ((0, 0), (1, 1), (1, 0), (2, 0))
    if supported_only and normalized not in supported:
        raise ValueError(
            "DDER dynamics supports two pinned endpoints, a pivot start with "
            "a free end, or a two-node clamped start with a free end."
        )
    return normalized


def _combine_boundary_and_free_values(
    free_values: torch.Tensor,
    boundary_values: torch.Tensor,
    pinned_endpoints: PinnedEndpointMask,
) -> torch.Tensor:
    start_count, end_count = (int(value) for value in pinned_endpoints)
    parts = []
    if start_count:
        parts.append(boundary_values[:, :start_count])
    parts.append(free_values)
    if end_count:
        parts.append(boundary_values[:, start_count : start_count + end_count])
    return torch.cat(parts, dim=1)


def _replace_pinned_values(
    values: torch.Tensor,
    boundary_values: torch.Tensor,
    pinned_endpoints: PinnedEndpointMask,
) -> torch.Tensor:
    free_start = int(pinned_endpoints[0])
    free_stop = values.shape[1] - int(pinned_endpoints[1])
    return _combine_boundary_and_free_values(
        values[:, free_start:free_stop],
        boundary_values,
        pinned_endpoints,
    )


def _normalize(vector: torch.Tensor, *, name: str) -> torch.Tensor:
    epsilon = 64.0 * torch.finfo(vector.dtype).eps
    norm = torch.linalg.vector_norm(vector, dim=-1, keepdim=True)
    if bool(torch.any(norm <= epsilon).detach().cpu()):
        raise ValueError(f"{name} contains a zero-length vector.")
    return vector / norm


def _normalize_unchecked(vector: torch.Tensor) -> torch.Tensor:
    epsilon = 64.0 * torch.finfo(vector.dtype).eps
    norm = torch.linalg.vector_norm(vector, dim=-1, keepdim=True)
    return vector / torch.clamp(norm, min=epsilon)


def _skew_matrix(vector: torch.Tensor) -> torch.Tensor:
    """Return matrices whose product with ``x`` is ``vector cross x``."""

    x, y, z = vector.unbind(dim=-1)
    zero = torch.zeros_like(x)
    return torch.stack(
        (zero, -z, y, z, zero, -x, -y, x, zero),
        dim=-1,
    ).reshape(vector.shape[:-1] + (3, 3))


def analytic_isotropic_bending_force(positions, stiffness, dual_lengths):
    """Closed-form gradient of the same zero-rest-curvature bending energy.

For unit tangents a,b, |2(a cross b)/(1+a.b)|Â²=4(1-a.b)/(1+a.b).
The fitting domain excludes folded/singular edges, as does the reference DER.
"""
    edges = positions[:, 1:] - positions[:, :-1]
    lengths = torch.linalg.vector_norm(edges, dim=-1, keepdim=True).clamp_min(1e-12)
    tangent = edges / lengths
    a, b = tangent[:, :-1], tangent[:, 1:]
    cosine = (a * b).sum(-1, keepdim=True)
    factor = -4 * stiffness[:, None, None] / dual_lengths[None, :, None]
    factor = factor / (1 + cosine).clamp_min(1e-12).square()
    previous = factor * (b - cosine * a) / lengths[:, :-1]
    following = factor * (a - cosine * b) / lengths[:, 1:]
    zero = torch.zeros_like(previous[:, :1])
    edge_gradient = torch.cat((previous, zero), 1) + torch.cat((zero, following), 1)
    return torch.cat((edge_gradient, zero), 1) - torch.cat((zero, edge_gradient), 1)


def _curvature_rate_jacobian_impl(
    positions: torch.Tensor,
    rest_lengths: torch.Tensor,
    endpoint_orientations: torch.Tensor | None,
    endpoint_orientation_rates: torch.Tensor | None,
    *,
    frame_regularization: float = 0.0,
    vectorized: bool = False,
    functional_assembly: bool = False,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Return ``J, r, w`` for ``0.5 * Cb * ||J v + r||_w^2``."""

    batch, node_count, _ = positions.shape
    edges = positions[:, 1:] - positions[:, :-1]
    epsilon = 64.0 * torch.finfo(positions.dtype).eps
    lengths = torch.clamp(torch.linalg.vector_norm(edges, dim=-1), min=epsilon)
    tangents = edges / lengths[..., None]
    identity = torch.eye(3, dtype=positions.dtype, device=positions.device)
    tangent_rate_blocks = (
        identity[None, None]
        - tangents[..., :, None] * tangents[..., None, :]
    ) / lengths[..., None, None]

    def curvature_derivatives(
        previous: torch.Tensor,
        following: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        cross = torch.linalg.cross(previous, following, dim=-1)
        denominator = torch.clamp(
            1.0 + torch.sum(previous * following, dim=-1),
            min=128.0 * torch.finfo(positions.dtype).eps,
        )
        derivative_previous = 2.0 * (
            -_skew_matrix(following) / denominator[..., None, None]
            - cross[..., :, None]
            * following[..., None, :]
            / denominator.square()[..., None, None]
        )
        derivative_following = 2.0 * (
            _skew_matrix(previous) / denominator[..., None, None]
            - cross[..., :, None]
            * previous[..., None, :]
            / denominator.square()[..., None, None]
        )
        return derivative_previous, derivative_following

    def objective_curvature_derivatives(
        previous: torch.Tensor,
        following: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Derivative of curvature in the minimally rotating local frame.

        The ordinary world-frame derivative of the curvature binormal is not
        objective: a rigidly rotating curved rod would dissipate energy.  The
        two adjacent tangents determine the local angular velocity through
        ``t_dot = omega cross t``.  Removing that rigid spin gives the
        corotational Kelvin--Voigt bending strain rate.
        """

        derivative_previous, derivative_following = curvature_derivatives(
            previous, following
        )
        curvature = 2.0 * torch.linalg.cross(previous, following, dim=-1) / torch.clamp(
            1.0 + torch.sum(previous * following, dim=-1),
            min=128.0 * torch.finfo(positions.dtype).eps,
        )[..., None]
        identity_batch = identity[None].expand(previous.shape[0], -1, -1)
        tangent_difference = previous - following
        normal = torch.linalg.cross(previous, following, dim=-1)
        if frame_regularization>0:
            # Versioned smooth transition at an unobservable straight-rod bend
            # plane. Algebraically combine the weight and normal projection so
            # no normalized zero vector or hard rank switch is differentiated.
            # This is a regularized damping model, not identical legacy physics.
            normal_squared=normal.square().sum(dim=-1)
            regularized=normal_squared+frame_regularization
            coefficient=-2.0/torch.clamp(1.0+(previous*following).sum(dim=-1),
                min=128.0*torch.finfo(positions.dtype).eps)
            bent_previous=(coefficient[...,None,None]*normal[..., :,None]
                *torch.linalg.cross(normal,previous,dim=-1)[...,None,:]/regularized[...,None,None])
            bent_following=(coefficient[...,None,None]*normal[..., :,None]
                *torch.linalg.cross(following,normal,dim=-1)[...,None,:]/regularized[...,None,None])
            straight_inverse=.5*(identity_batch-previous[..., :,None]*previous[...,None,:])
            corotation=_skew_matrix(curvature)
            weight=(frame_regularization/regularized)[...,None,None]
            return (bent_previous+weight*(derivative_previous+corotation@straight_inverse@_skew_matrix(previous)),
                bent_following+weight*(derivative_following+corotation@straight_inverse@_skew_matrix(following)))
        # On unit-tangent variations, removing the pair's rigid spin leaves
        # only the rate of 2*tan(theta/2), along its binormal. Evaluate this
        # directly: the equivalent angular pseudoinverse divides by 1-cos(theta)
        # and then cancels large terms. Clamping that eigenvalue by float32
        # epsilon previously changed damping by nearly 100% at small bends.
        unit_normal = normal / torch.sqrt(torch.clamp(
            normal.square().sum(dim=-1, keepdim=True), min=1.0e-20))
        coefficient = -2.0 / torch.clamp(
            1.0 + (previous * following).sum(dim=-1),
            min=128.0 * torch.finfo(positions.dtype).eps)
        bent_previous = coefficient[..., None, None] * unit_normal[..., :, None] * (
            torch.linalg.cross(unit_normal, previous, dim=-1)[..., None, :])
        bent_following = coefficient[..., None, None] * unit_normal[..., :, None] * (
            torch.linalg.cross(following, unit_normal, dim=-1)[..., None, :])
        straight_inverse = 0.5 * (
            identity_batch
            - previous[..., :, None] * previous[..., None, :]
        )
        spin_previous = straight_inverse @ _skew_matrix(previous)
        spin_following = straight_inverse @ _skew_matrix(following)
        corotation = _skew_matrix(curvature)
        # ||a-b||Â² = 2(1-cos(theta)), without cancellation near zero.
        bent = (tangent_difference.square().sum(dim=-1) > 2.0e-7)[..., None, None]
        return (
            torch.where(bent, bent_previous, derivative_previous + corotation @ spin_previous),
            torch.where(bent, bent_following, derivative_following + corotation @ spin_following),
        )

    if vectorized and endpoint_orientations is None:
        # Evaluate independent interior vertices in the batch dimension instead
        # of launching the same geometry algebra once per cable vertex.
        count=node_count-2
        dp,df=objective_curvature_derivatives(tangents[:,:-1].reshape(-1,3),tangents[:,1:].reshape(-1,3))
        previous=(dp@tangent_rate_blocks[:,:-1].reshape(-1,3,3)).reshape(batch,count,3,3)
        following=(df@tangent_rate_blocks[:,1:].reshape(-1,3,3)).reshape(batch,count,3,3)
        if functional_assembly:
            # Same blocks, assembled out of place for a leading vmap dimension.
            basis=torch.eye(node_count,dtype=positions.dtype,device=positions.device)
            jacobian=(-previous[:,:,:,None,:]*basis[:count][None,:,None,:,None]
                      +(previous-following)[:,:,:,None,:]*basis[1:count+1][None,:,None,:,None]
                      +following[:,:,:,None,:]*basis[2:count+2][None,:,None,:,None])
        else:
            jacobian=positions.new_zeros(batch,count,3,node_count,3)
            index=torch.arange(count,device=positions.device)
            jacobian[:,index,:,index,:]=-previous.transpose(0,1)
            jacobian[:,index,:,index+1,:]=(previous-following).transpose(0,1)
            jacobian[:,index,:,index+2,:]=following.transpose(0,1)
        weights=(2.0/(rest_lengths[:-1]+rest_lengths[1:]))[None].expand(batch,-1).repeat_interleave(3,dim=1)
        return jacobian.reshape(batch,3*count,3*node_count),positions.new_zeros(batch,3*count),weights

    zero_block = torch.zeros(
        (batch, 3, 3), dtype=positions.dtype, device=positions.device
    )
    rows: list[torch.Tensor] = []
    affine_rows: list[torch.Tensor] = []
    weights: list[torch.Tensor] = []
    for interior in range(node_count - 2):
        derivative_previous, derivative_following = objective_curvature_derivatives(
            tangents[:, interior], tangents[:, interior + 1]
        )
        previous_block = derivative_previous @ tangent_rate_blocks[:, interior]
        following_block = derivative_following @ tangent_rate_blocks[:, interior + 1]
        blocks = [zero_block] * node_count
        blocks[interior] = -previous_block
        blocks[interior + 1] = previous_block - following_block
        blocks[interior + 2] = following_block
        rows.append(torch.cat(blocks, dim=-1))
        affine_rows.append(torch.zeros_like(positions[:, 0]))
        weights.append(
            torch.ones_like(lengths[:, interior])
            * 2.0
            / (rest_lengths[interior] + rest_lengths[interior + 1])
        )

    if endpoint_orientations is not None:
        if endpoint_orientation_rates is None:
            raise ValueError(
                "Endpoint orientation rates are required for terminal damping."
            )
        holder_raw = endpoint_orientations[..., :, 0]
        holder_raw_rate = endpoint_orientation_rates[..., :, 0]
        holder_length = torch.clamp(
            torch.linalg.vector_norm(holder_raw, dim=-1, keepdim=True), min=epsilon
        )
        holder = holder_raw / holder_length
        holder_rate = (
            holder_raw_rate
            - holder * torch.sum(holder * holder_raw_rate, dim=-1, keepdim=True)
        ) / holder_length

        derivative_holder, derivative_edge = objective_curvature_derivatives(
            holder[:, 0], tangents[:, 0]
        )
        edge_block = derivative_edge @ tangent_rate_blocks[:, 0]
        blocks = [zero_block] * node_count
        blocks[0], blocks[1] = -edge_block, edge_block
        rows.append(torch.cat(blocks, dim=-1))
        affine_rows.append((derivative_holder @ holder_rate[:, 0, :, None])[..., 0])
        weights.append(
            torch.ones_like(lengths[:, 0]) / rest_lengths[0]
        )

        derivative_edge, derivative_holder = objective_curvature_derivatives(
            tangents[:, -1], holder[:, 1]
        )
        edge_block = derivative_edge @ tangent_rate_blocks[:, -1]
        blocks = [zero_block] * node_count
        blocks[-2], blocks[-1] = -edge_block, edge_block
        rows.append(torch.cat(blocks, dim=-1))
        affine_rows.append((derivative_holder @ holder_rate[:, 1, :, None])[..., 0])
        weights.append(
            torch.ones_like(lengths[:, -1]) / rest_lengths[-1]
        )

    jacobian = torch.cat(rows, dim=1)
    affine = torch.stack(affine_rows, dim=1).reshape(batch, -1)
    weight = torch.stack(weights, dim=1).repeat_interleave(3, dim=1)
    return jacobian, affine, weight


def solve_symmetric_tridiagonal_dense(
    diagonal: torch.Tensor,
    off_diagonal: torch.Tensor,
    right_hand_side: torch.Tensor,
) -> torch.Tensor:
    """Solve the same batched system with a dense CUDA-parallel factorization.

    The rod systems are only one equation per edge (ten equations for the
    current OptiTrack cable).  During offline differentiation, the additional
    storage is small and ``torch.linalg.solve`` exposes substantially more
    parallel work to CUDA than the sequential Thomas recurrence.  Runtime
    filtering keeps the linear-memory tridiagonal solver above.
    """

    if diagonal.ndim != 2 or right_hand_side.shape != diagonal.shape:
        raise ValueError("Diagonal and right-hand side must have shape BxN.")
    if off_diagonal.shape != (diagonal.shape[0], diagonal.shape[1] - 1):
        raise ValueError("Off diagonal must have shape Bx(N-1).")
    system = torch.diag_embed(diagonal)
    system = system + torch.diag_embed(off_diagonal, offset=1)
    system = system + torch.diag_embed(off_diagonal, offset=-1)
    if system.is_cuda and torch.cuda.is_current_stream_capturing():
        # Same dense solve, with asynchronous status during a captured graph.
        # Runtime callers reject nonfinite/invalid trajectories after replay.
        return torch.linalg.solve_ex(system, right_hand_side.unsqueeze(-1), check_errors=False).result.squeeze(-1)
    return torch.linalg.solve(system, right_hand_side.unsqueeze(-1)).squeeze(-1)


def momentum_project_lengths(
    positions: torch.Tensor,
    rest_lengths: torch.Tensor,
    masses: torch.Tensor,
    boundary_positions: torch.Tensor | None,
    *,
    iterations: int,
    dense_solve: bool = False,
    pinned_endpoints: PinnedEndpointMask = TWO_PINNED_ENDPOINTS,
) -> torch.Tensor:
    """Coupled inverse-mass edge projection with optional pinned endpoints."""

    if positions.ndim != 3 or positions.shape[2] != 3:
        raise ValueError("positions must have shape BxNx3.")
    batch, node_count, _ = positions.shape
    if rest_lengths.shape != (node_count - 1,):
        raise ValueError("rest_lengths must contain one value per edge.")
    if masses.shape != (node_count,):
        raise ValueError("masses must contain one value per vertex.")
    pinned_endpoints = _validate_pinned_endpoints(pinned_endpoints)
    effective_pins = (
        (False, False) if boundary_positions is None else pinned_endpoints
    )
    boundary_count = int(effective_pins[0]) + int(effective_pins[1])
    if boundary_positions is not None and boundary_positions.shape != (
        batch,
        boundary_count,
        3,
    ):
        raise ValueError(
            f"boundary_positions must have shape Bx{boundary_count}x3 for "
            "the selected pinned endpoints."
        )
    if iterations < 1:
        raise ValueError("Projection iterations must be positive.")

    inverse_mass = torch.reciprocal(masses).clone()
    start_count, end_count = (int(value) for value in effective_pins)
    if start_count:
        inverse_mass = torch.cat(
            (torch.zeros_like(inverse_mass[:start_count]), inverse_mass[start_count:])
        )
    if end_count:
        inverse_mass = torch.cat(
            (inverse_mass[:-end_count], torch.zeros_like(inverse_mass[-end_count:]))
        )

    def clamp_boundary(value: torch.Tensor) -> torch.Tensor:
        if boundary_positions is None:
            return value
        return _replace_pinned_values(value, boundary_positions, effective_pins)

    if dense_solve is not True:
        raise ValueError("This port supports the direct dense solve only")
    solve = solve_symmetric_tridiagonal_dense
    value = clamp_boundary(positions)
    epsilon = 64.0 * torch.finfo(value.dtype).eps
    for _ in range(iterations):
        delta = value[:, 1:] - value[:, :-1]
        distance = torch.linalg.vector_norm(delta, dim=-1)
        direction = delta / torch.clamp(distance[..., None], min=epsilon)
        diagonal = (inverse_mass[:-1] + inverse_mass[1:])[None].expand(batch, -1)
        adjacent = torch.sum(direction[:, :-1] * direction[:, 1:], dim=-1)
        off_diagonal = -inverse_mass[1:-1][None] * adjacent
        solver_regularization = 8.0 * torch.finfo(value.dtype).eps
        regularization = solver_regularization * torch.amax(
            diagonal, dim=1, keepdim=True
        )
        multiplier = solve(
            diagonal + regularization,
            off_diagonal,
            -(distance - rest_lengths[None]),
        )
        edge_correction = multiplier[..., None] * direction
        left = -inverse_mass[:-1][None, :, None] * edge_correction
        right = inverse_mass[1:][None, :, None] * edge_correction
        node_correction = torch.nn.functional.pad(left, (0, 0, 0, 1))
        node_correction = node_correction + torch.nn.functional.pad(
            right, (0, 0, 1, 0)
        )
        value = clamp_boundary(value + node_correction)
    return value


def momentum_project_velocities(
    positions: torch.Tensor,
    velocities: torch.Tensor,
    masses: torch.Tensor,
    boundary_velocities: torch.Tensor,
    *,
    validate: bool = True,
    dense_solve: bool = False,
    pinned_endpoints: PinnedEndpointMask = TWO_PINNED_ENDPOINTS,
) -> torch.Tensor:
    """Mass-weighted projection onto the inextensible velocity constraints."""

    if positions.ndim != 3 or positions.shape[2] != 3:
        raise ValueError("positions must have shape BxNx3.")
    if velocities.shape != positions.shape:
        raise ValueError("velocities must match positions.")
    batch, node_count, _ = positions.shape
    if masses.shape != (node_count,):
        raise ValueError("masses must contain one value per vertex.")
    pinned_endpoints = _validate_pinned_endpoints(pinned_endpoints)
    boundary_count = int(pinned_endpoints[0]) + int(pinned_endpoints[1])
    if boundary_velocities.shape != (batch, boundary_count, 3):
        raise ValueError(
            f"boundary_velocities must have shape Bx{boundary_count}x3 for "
            "the selected pinned endpoints."
        )

    inverse_mass = torch.reciprocal(masses).clone()
    start_count, end_count = (int(value) for value in pinned_endpoints)
    if start_count:
        inverse_mass = torch.cat(
            (torch.zeros_like(inverse_mass[:start_count]), inverse_mass[start_count:])
        )
    if end_count:
        inverse_mass = torch.cat(
            (inverse_mass[:-end_count], torch.zeros_like(inverse_mass[-end_count:]))
        )
    value = _replace_pinned_values(
        velocities,
        boundary_velocities,
        pinned_endpoints,
    )
    edges = positions[:, 1:] - positions[:, :-1]
    direction = (
        _normalize(edges, name="rod edges")
        if validate
        else _normalize_unchecked(edges)
    )
    diagonal = (inverse_mass[:-1] + inverse_mass[1:])[None].expand(batch, -1)
    adjacent = torch.sum(direction[:, :-1] * direction[:, 1:], dim=-1)
    off_diagonal = -inverse_mass[1:-1][None] * adjacent
    constraint_velocity = torch.sum(
        direction * (value[:, 1:] - value[:, :-1]),
        dim=-1,
    )
    solver_regularization = 8.0 * torch.finfo(value.dtype).eps
    regularization = solver_regularization * torch.amax(
        diagonal, dim=1, keepdim=True
    )
    if dense_solve is not True:
        raise ValueError("This port supports the direct dense solve only")
    solve = solve_symmetric_tridiagonal_dense
    multiplier = solve(
        diagonal + regularization,
        off_diagonal,
        -constraint_velocity,
    )
    edge_correction = multiplier[..., None] * direction
    left = -inverse_mass[:-1][None, :, None] * edge_correction
    right = inverse_mass[1:][None, :, None] * edge_correction
    node_correction = torch.nn.functional.pad(left, (0, 0, 0, 1))
    node_correction = node_correction + torch.nn.functional.pad(
        right,
        (0, 0, 1, 0),
    )
    return _replace_pinned_values(
        value + node_correction,
        boundary_velocities,
        pinned_endpoints,
    )
