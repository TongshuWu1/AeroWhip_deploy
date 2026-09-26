"""Diagonal KL projections, following Otto et al. (CoRL 2022), Appendix A.

https://proceedings.mlr.press/v205/otto23a/otto23a.pdf
Independent PyTorch implementation: batched bisection and implicit first-order
derivatives, no CPU optimizer. Limits use actual KL terms (including 1/2).
Reference distributions are constants. Double backward is not supported.
"""

import math

import torch
from torch.autograd.function import once_differentiable


def gaussian_log_prob(latent, mean, log_std):
    """Joint latent density, summed over action dimensions, in float64.

    A fixed invertible affine map followed by tanh has the same cancelling
    Jacobian in importance ratios. Keep the sampled latent; do not invert
    numerically saturated actions or clip/resample them.
    """
    latent, mean, log_std = (x.double() for x in (latent, mean, log_std))
    return (-.5*((latent-mean)*torch.exp(-log_std)).square()
            - log_std - .5*math.log(2*math.pi)).sum(-1)


def diagonal_kl(mean, log_std, old_mean, old_log_std):
    """Separate mean/covariance contributions to KL(new || old)."""
    mean, log_std, old_mean, old_log_std = (
        x.double() for x in (mean, log_std, old_mean, old_log_std))
    mean_part = .5*((mean-old_mean)*torch.exp(-old_log_std)).square().sum(-1)
    log_ratio = 2*(log_std-old_log_std)
    covariance_part = .5*(torch.expm1(log_ratio)-log_ratio).sum(-1)
    return mean_part, covariance_part


class _CovarianceProjection(torch.autograd.Function):
    @staticmethod
    def forward(ctx, log_std, old_log_std, limit):
        # t = projected_variance / old_variance, r=new/old variance.
        # t=(1+eta)/(1/r+eta). Solve in log(eta) for extreme variance ratios.
        log_r = 2*(log_std-old_log_std)
        active = .5*(torch.expm1(log_r)-log_r).sum(-1, keepdim=True) > limit
        low, high = torch.full_like(log_r[..., :1], -80.), torch.full_like(log_r[..., :1], 80.)
        for _ in range(64):
            log_eta = (low+high)*.5
            log_t = torch.logaddexp(torch.zeros_like(log_eta), log_eta)-torch.logaddexp(-log_r, log_eta)
            outside = .5*(torch.expm1(log_t)-log_t).sum(-1, keepdim=True) > limit
            low = torch.where(outside, log_eta, low)
            high = torch.where(outside, high, log_eta)
        log_t = torch.logaddexp(torch.zeros_like(high), high)-torch.logaddexp(-log_r, high)
        b = torch.sigmoid(-high-log_r)
        h = torch.sigmoid(high)-torch.sigmoid(high+log_r)
        g = .5*torch.expm1(log_t)
        ctx.save_for_backward(active, b, h, g)
        return torch.where(active, old_log_std+.5*log_t, log_std)

    @staticmethod
    @once_differentiable
    def backward(ctx, grad):
        active, b, h, g = ctx.saved_tensors
        denominator = (g*h).sum(-1, keepdim=True)
        denominator = torch.where(active, denominator, torch.ones_like(denominator))
        correction = (grad*h).sum(-1, keepdim=True)/denominator
        derivative = b*(grad-g*correction)
        return torch.where(active, derivative, grad), None, None


def project(mean, log_std, old_mean, old_log_std, *, mean_limit, covariance_limit):
    """Project each row; old parameters are detached, solver arithmetic float64.

    The mean projection is the radial solution in the old covariance metric.
    The covariance projection minimizes KL(projected || proposed) subject to
    its covariance KL(projected || old) bound. Fixed iterations avoid host sync.
    """
    if not all(math.isfinite(x) and x > 0 for x in (mean_limit, covariance_limit)):
        raise ValueError("KL limits must be positive and finite")
    mean, log_std = mean.double(), log_std.double()
    old_mean, old_log_std = old_mean.detach().double(), old_log_std.detach().double()
    if mean.shape != log_std.shape or mean.shape != old_mean.shape or mean.shape != old_log_std.shape:
        raise ValueError("Distribution parameter shapes must match")
    distance, _ = diagonal_kl(mean, log_std, old_mean, old_log_std)
    scale = torch.sqrt(mean_limit/distance.clamp_min(mean_limit)).unsqueeze(-1)
    projected_mean = old_mean+scale*(mean-old_mean)
    projected_log_std = _CovarianceProjection.apply(log_std, old_log_std, covariance_limit)
    return projected_mean, projected_log_std
