"""Full-rank spline exploration prior; no physics, reward or learner imports."""

import math

import torch

from .bspline_torch import BatchedBSplineCommand


@torch.no_grad()
def smooth_action_matrix(control_count, duration_s, *, smoothness_time_s, amplitude_scale):
    """Return a float64 CPU map for interleaved interior-XYZ/endpoint latents.

    Q is mean squared command acceleration for scalar free control offsets,
    integrated by trapezoidal quadrature on the native 100-Hz grid. The prior
    covariance is proportional to (I + tau**4 Q)**-1. Its symmetric square root
    favors coordinated changes while retaining every mode (no eigen truncation).
    Average squared row norm equals amplitude_scale**2. This normalizes latent
    control variation, not physical excursion after tanh, or flight feasibility.
    """
    if any(not math.isfinite(x) or x <= 0 for x in (smoothness_time_s, amplitude_scale)):
        raise ValueError('Exploration smoothness time and amplitude scale must be positive and finite')
    spline = BatchedBSplineCommand(control_count, duration_s, device='cpu', dtype=torch.float64)
    count = spline.control_count-5
    free = torch.eye(count, dtype=torch.float64)
    # Initial P/V/A perturbations are zero; the last free value is repeated to
    # preserve the original stopped endpoint, rather than smoothing a command.
    embedding = torch.cat((torch.zeros((3, count), dtype=torch.float64), free[:-1], free[-1:].expand(3, -1)))
    velocity = torch.diff(embedding, dim=0)*spline.derivative_scale_1
    acceleration = torch.diff(velocity, dim=0)*spline.derivative_scale_2
    sampled = spline.derivative_basis_2 @ acceleration
    weights = torch.ones(spline.sample_count, dtype=torch.float64)
    weights[[0, -1]] = .5
    gram = sampled.T @ (sampled*weights[:, None])*(spline.timestep_s/spline.duration_s)
    precision = free+smoothness_time_s**4*gram
    values, vectors = torch.linalg.eigh((precision+precision.T)*.5)
    if not bool(torch.isfinite(values).all() & (values > 0).all()):
        raise ValueError('Exploration precision must be finite and positive definite')
    factor = (vectors*values.rsqrt()[None, :]) @ vectors.T
    factor *= amplitude_scale/factor.square().sum(-1).mean().sqrt()
    return torch.kron(factor.contiguous(), torch.eye(3, dtype=torch.float64))
