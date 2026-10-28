"""Fisher information of a trajectory about the fitted parameters, from finite-difference
sensitivities of the simulated measurements (cable markers and drone tilt) to each parameter.

The measurement vector of a simulated trajectory is the marker positions every 0.1 s (10 markers,
3 coordinates) and the drone's thrust axis; its noise is taken as sigma_m (markers, m) and sigma_t
(axis, rad). The Fisher information F = sum J^T J / sigma^2 over the samples, J the Jacobian of the
measurements with respect to the parameters in **relative** units (each parameter perturbed by a
fraction of its scale), so F is dimensionless and comparable across parameters. Scores:
  marginal(i) = 1 / (F^-1)_ii: the information about parameter i after the others are accounted for
      (what a least-squares fit would learn about it alone);
  score = min_i log10 marginal(i) over the parameters that matter: the trajectory is as good as the
      parameter it teaches least about (E-optimal flavour; IDTD's Schur score is this marginal).
  coupling: the correlations of the parameter estimates, C = F^-1 normalised. Coupled parameters
      (two that move the measurements alike) already lower each other's marginal; the pairs are
      reported so it is visible which ones a trajectory fails to separate.
Parameters: the cable fit's search vector (log EI, log Cb, Cd, log drag, residual terms) and the
drone's (motor response, drag xy, drag z); the inertia scale is left out (not identifiable).
"""
import numpy as np

from aerowhip.planning.simulate import simulate_motion

SIGMA_MARKER_M = .004
SIGMA_AXIS_RAD = .005
RELATIVE_STEP = .05      # fraction of each parameter's scale for the finite difference


def parameter_set(ensemble):
    """[(name, kind, index, scale)]: the parameters the information is computed for, with the step
    scale (the fit's standard deviation, or a fraction of the value for parameters without one)."""
    out = []
    cable_sd = np.asarray(ensemble.cable_fit['uncertainty']['standard_deviation'], float)
    for i, name in enumerate(ensemble.cable_fit['parameters']):
        out.append((name, 'cable', i, max(float(cable_sd[i]), 1e-3)))
    if ensemble.drone_fit is not None:
        drone_sd = np.asarray(ensemble.drone_fit['uncertainty']['standard_deviation'], float)
        for i, name in enumerate(ensemble.drone_fit['parameters']):
            if name.startswith('inertia'):
                continue
            out.append((name, 'drone', i, max(float(drone_sd[i]), 1e-3)))
    return out


def measurement(result):
    """The measurement vector of a simulation: markers (M, 10, 3) and the thrust axis (M, 3), scaled by
    their noise so the Fisher information is dimensionless."""
    m = result['markers'].reshape(len(result['markers']), -1)/SIGMA_MARKER_M
    axis = result['rotation'][:, :, 2]/SIGMA_AXIS_RAD
    return np.concatenate([m, axis], 1).ravel()


def perturbed_configs(ensemble, parameters, step=RELATIVE_STEP):
    """[(name, config)] with each parameter moved by `step` x its scale from the adopted values."""
    x0 = np.asarray(ensemble.cable_fit['best']['x'], float)
    d0 = ensemble.drone_best()
    out = []
    for name, kind, i, scale in parameters:
        x, d = x0.copy(), (None if d0 is None else d0.copy())
        if kind == 'cable':
            x[i] += step*scale
        else:
            d[i] += step*scale
        out.append((name, ensemble.member_config(x, d)))
    return out


def fisher(base_result, perturbed_results, parameters, step=RELATIVE_STEP):
    """F (n, n) from the adopted simulation and one simulation per perturbed parameter (same length)."""
    y0 = measurement(base_result)
    columns = []
    for (name, kind, i, scale), r in zip(parameters, perturbed_results):
        n = min(len(base_result['markers']), len(r['markers']))
        y = measurement(dict(markers=r['markers'][:n], rotation=r['rotation'][:n]))
        y0n = measurement(dict(markers=base_result['markers'][:n], rotation=base_result['rotation'][:n]))
        columns.append((y-y0n)/step)                 # per unit of relative change (one scale)
    m = min(len(c) for c in columns)
    J = np.stack([c[:m] for c in columns], 1)
    return J.T@J


def scores(F, parameters, coupled=.8):
    """Per-parameter marginal information (log10), the overall score (the weakest parameter) and the
    parameter pairs whose estimates correlate beyond `coupled` (strongest first)."""
    n = len(parameters)
    F = np.asarray(F, float)+1e-9*np.eye(n)
    try:
        inverse = np.linalg.inv(F)
        marginal = 1./np.maximum(np.diag(inverse), 1e-300)
        d = np.sqrt(np.maximum(np.diag(inverse), 1e-300))
        correlation = inverse/np.outer(d, d)
    except np.linalg.LinAlgError:
        marginal = np.full(n, 1e-300)
        correlation = np.eye(n)
    log_marginal = np.log10(np.maximum(marginal, 1e-300))
    pairs = sorted(((parameters[i][0], parameters[j][0], float(correlation[i, j]))
                    for i in range(n) for j in range(i+1, n) if abs(correlation[i, j]) > coupled),
                   key=lambda r: -abs(r[2]))
    return dict(marginal_log10={p[0]: float(v) for p, v in zip(parameters, log_marginal)},
                score=float(log_marginal.min()), weakest=parameters[int(np.argmin(log_marginal))][0],
                log_det=float(np.linalg.slogdet(F)[1]/np.log(10)), coupled=pairs,
                correlation=np.round(correlation, 3).tolist())


def trajectory_information(ensemble, firmware, motion, parameters=None, runner=None):
    """Fisher information and scores of a motion for the adopted model: 1 + n simulations
    (sequential unless a callable `runner(configs) -> results` is given)."""
    parameters = parameters or parameter_set(ensemble)
    configs = [('adopted', ensemble.member_config(ensemble.cable_fit['best']['x'], ensemble.drone_best()))]+perturbed_configs(ensemble, parameters)
    if runner is None:
        results = [simulate_motion(c, firmware, motion) for _, c in configs]
    else:
        results = runner(configs)
    base, rest = results[0], results[1:]
    if base is None or base['failed'] or any(r is None or r['failed'] for r in rest):
        return None
    F = fisher(base, rest, parameters)
    out = scores(F, parameters)
    out['F'] = F
    return out
