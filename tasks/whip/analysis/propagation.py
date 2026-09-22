"""Study-only constant-advection diagnostic; NOT a ridge tracker or whip classifier.

Fit kappa_t + c*kappa_s = 0 to a material-coordinate curvature field. Positive
rho and speed mean distal advection. Shape changes, reflected waves and changing
speed can lower coherence; a translating ramp can have perfect coherence without
a localized bend. No physical success threshold or reward is defined here.
"""

import math

import torch


@torch.no_grad()
def curvature_advection(curvature, time, material_s, sample_valid, windows):
    """Return batched signed coherence and its constant-advection fit.

    Inputs: float64 curvature (B,T,K) in 1/m; time (T) in s; strictly increasing
    material_s (K) in m; boolean sample_valid (B,T); windows (B,2) in s.
    All tensors share a device. Spatial/temporal derivatives are centered on
    adjacent space-time cells. Only cells wholly inside each window contribute,
    with material-length/time quadrature weights, without resampling or smoothing.

    rho=-<k_t,k_s>/sqrt(<k_t,k_t><k_s,k_s>), c=-<k_t,k_s>/<k_s,k_s>.
    normalized_residual is the unconstrained least-squares residual 1-rho**2;
    negative rho is proximal motion, NOT successful distal propagation.

    Any invalid/nonfinite sample in a lane invalidates its fit, including samples
    outside the selected window. Gaps are never deleted or bridged. Ratio fields
    are NaN when undefined. RMS fields remain available for otherwise valid
    static/zero fields, so their degeneracy can be inspected. Coverage is the
    valid included-cell duration divided by requested window duration; it can be
    below one for unaligned/out-of-recording window boundaries.

    Numerical activity guard (not a physical threshold): delta_k = 128*eps times
    max(max(abs(kappa)), 1/min(ds)). Require RMS curvature above delta_k and RMS
    derivatives above weighted RMS bounds 2*delta_k/dt and 2*delta_k/ds. The mesh
    scale covers roundoff-level curvature of rigid/straight geometric histories.
    This is a pragmatic precision guard, not a bound on measurement noise or
    ill-conditioned cable geometry. No GPU synchronization occurs in arithmetic;
    metadata/time/mesh validation may synchronize.
    """
    if not isinstance(curvature, torch.Tensor) or curvature.ndim != 3:
        raise ValueError('Curvature must have shape (batch,times,material_samples)')
    batch, count, nodes = curvature.shape
    if curvature.dtype != torch.float64 or batch < 1 or count < 2 or nodes < 2:
        raise ValueError('Use float64 curvature with nonempty batch and at least two times/material samples')
    for name, value, shape, dtype in (
        ('Time', time, (count,), torch.float64),
        ('Material coordinate', material_s, (nodes,), torch.float64),
        ('Sample validity', sample_valid, (batch, count), torch.bool),
        ('Windows', windows, (batch, 2), torch.float64),
    ):
        if (not isinstance(value, torch.Tensor) or tuple(value.shape) != shape
                or value.dtype != dtype or value.device != curvature.device):
            raise ValueError(f'{name} must have shape {shape} and share the required dtype/device')
    dt, ds = time.diff(), material_s.diff()
    if not bool(torch.isfinite(time).all() & torch.isfinite(dt).all() & (dt > 0).all()):
        raise ValueError('Time must be finite and strictly increasing')
    if not bool(torch.isfinite(material_s).all() & torch.isfinite(ds).all() & (ds > 0).all()):
        raise ValueError('Material coordinates must be finite and strictly increasing')

    finite_samples = torch.isfinite(curvature).all(-1)
    lane_valid = finite_samples.all(-1) & sample_valid.all(-1)
    requested = windows[:, 1]-windows[:, 0]
    window_valid = torch.isfinite(windows).all(-1) & torch.isfinite(requested) & (requested > 0)
    # Include mathematically aligned endpoints despite float64 subtraction.
    tolerance = 16*torch.finfo(time.dtype).eps*torch.maximum(
        time.abs().amax().clamp_min(1.), windows.abs().amax(-1))
    inside = ((time[:-1][None] >= windows[:, :1]-tolerance[:, None])
              & (time[1:][None] <= windows[:, 1:]+tolerance[:, None]) & window_valid[:, None])
    cell_valid = (finite_samples[:, :-1] & finite_samples[:, 1:]
                  & sample_valid[:, :-1] & sample_valid[:, 1:])
    coverage = (dt[None]*(inside & cell_valid)).sum(-1)/torch.where(
        window_valid, requested, torch.ones_like(requested))
    coverage = coverage.clamp(0, 1).masked_fill(~window_valid, float('nan'))

    clean = torch.where(lane_valid[:, None, None], curvature, 0.)
    a, b, c, d = clean[:, :-1, :-1], clean[:, :-1, 1:], clean[:, 1:, :-1], clean[:, 1:, 1:]
    temporal = ((c+d)-(a+b))/(2*dt[None, :, None])
    spatial = ((b+d)-(a+c))/(2*ds[None, None, :])
    centered = .25*(a+b+c+d)
    weights = inside[:, :, None]*dt[None, :, None]*ds[None, None, :]
    measure = weights.sum((-2, -1))
    denominator = torch.where(measure > 0, measure, torch.ones_like(measure))

    def mean(value):
        return (weights*value).sum((-2, -1))/denominator

    temporal_square = mean(temporal.square())
    spatial_square = mean(spatial.square())
    cross = mean(temporal*spatial)
    curvature_rms = mean(centered.square()).clamp_min(0).sqrt()
    temporal_rms = temporal_square.clamp_min(0).sqrt()
    spatial_rms = spatial_square.clamp_min(0).sqrt()
    roundoff = 128*torch.finfo(curvature.dtype).eps*torch.maximum(
        clean.abs().amax((-2, -1)), 1/ds.min())
    temporal_floor = mean((2*roundoff[:, None, None]/dt[None, :, None]).square()).sqrt()
    spatial_floor = mean((2*roundoff[:, None, None]/ds[None, None, :]).square()).sqrt()
    measurable = (lane_valid & window_valid & (measure > 0)
                  & torch.isfinite(curvature_rms) & torch.isfinite(temporal_rms)
                  & torch.isfinite(spatial_rms) & torch.isfinite(cross))
    defined = (measurable & (curvature_rms > roundoff)
               & (temporal_rms > temporal_floor) & (spatial_rms > spatial_floor))
    rho = -cross/torch.where(defined, temporal_rms*spatial_rms, torch.ones_like(cross))
    speed = -cross/torch.where(defined, spatial_square, torch.ones_like(cross))
    defined &= torch.isfinite(rho) & torch.isfinite(speed)
    rho = rho.clamp(-1, 1)
    residual = (1-rho.square()).clamp_min(0)
    return dict(
        defined=defined,
        rho=rho.masked_fill(~defined, float('nan')),
        speed_mps=speed.masked_fill(~defined, float('nan')),
        normalized_residual=residual.masked_fill(~defined, float('nan')),
        curvature_rms_per_m=curvature_rms.masked_fill(~measurable, float('nan')),
        temporal_rms_per_m_s=temporal_rms.masked_fill(~measurable, float('nan')),
        spatial_rms_per_m2=spatial_rms.masked_fill(~measurable, float('nan')),
        coverage=coverage,
    )


def verify_advection():
    """Small CPU analytic checks; no physics rollout, training or filesystem writes.

    The stationary amplitude pulse has zero endpoint curvature, so its integrated
    temporal/spatial cross term vanishes. A translating ramp is deliberately
    retained as a positive-coherence non-pulse counterexample, not relabeled.
    Rigid Euclidean invariance through WhipMetrics belongs in the study runner.
    """
    time = torch.linspace(0., 1., 201, dtype=torch.float64)
    s = torch.linspace(0., 1., 101, dtype=torch.float64)
    tt, ss = time[:, None], s[None]
    forward = 2+torch.cos(2*math.pi*(ss-.5*tt))
    backward = 2+torch.cos(2*math.pi*(ss+.5*tt))
    static = (2+torch.cos(2*math.pi*ss)).expand_as(forward)
    growing = (1+tt)*torch.sin(math.pi*ss).square()
    zero = torch.zeros_like(forward)
    nonfinite = forward.clone(); nonfinite[100, 50] = float('nan')
    ramp = 2+ss-.5*tt
    rigid_noise = static+8*torch.finfo(torch.float64).eps*torch.sin(2*math.pi*tt)*torch.sin(4*math.pi*ss)
    curves = torch.stack((forward, backward, static, growing, zero, nonfinite, ramp, rigid_noise, forward))
    valid = torch.ones(curves.shape[:2], dtype=torch.bool)
    valid[-1, 100] = False
    windows = torch.tensor([[0., 1.]], dtype=torch.float64).expand(len(curves), -1)
    result = curvature_advection(curves, time, s, valid, windows)
    checks = dict(
        forward_direction=bool(result['defined'][0] & (result['rho'][0] > .999999)),
        backward_direction=bool(result['defined'][1] & (result['rho'][1] < -.999999)),
        forward_speed=bool(torch.isclose(result['speed_mps'][0], torch.tensor(.5, dtype=torch.float64), rtol=.001, atol=0)),
        backward_speed=bool(torch.isclose(result['speed_mps'][1], torch.tensor(-.5, dtype=torch.float64), rtol=.001, atol=0)),
        static_undefined=not bool(result['defined'][2]),
        stationary_growth_not_transport=bool(result['defined'][3] & (result['rho'][3].abs() < 1e-12)),
        zero_undefined=not bool(result['defined'][4]),
        nonfinite_undefined=not bool(result['defined'][5]),
        translating_ramp_counterexample=bool(result['defined'][6] & (result['rho'][6] > .999999)),
        roundoff_noise_undefined=not bool(result['defined'][7]),
        invalid_gap_not_bridged=bool(~result['defined'][8] & (result['coverage'][8] < 1)),
        undefined_ratios_are_nan=bool(torch.isnan(result['rho'][~result['defined']]).all()),
    )
    failed = [name for name, passed in checks.items() if not passed]
    if failed:
        raise AssertionError(f'Advection checks failed: {failed}')
    return dict(checks=checks, forward_speed_mps=float(result['speed_mps'][0]),
                forward_rho=float(result['rho'][0]), stationary_growth_rho=float(result['rho'][3]),
                ramp_rho=float(result['rho'][6]))
