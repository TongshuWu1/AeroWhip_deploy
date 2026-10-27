"""Least-squares identification on short segments of the calibration flights.

Each flight is cut into consecutive segments (default 1 s) and every segment is replayed from the
measured state at its start (multiple shooting): errors cannot accumulate over long open-loop
horizons, and the horizon is close to what a policy sees between corrections. The residuals are
the per-frame simulated-minus-measured positions; scipy's trust-region least squares fits the
parameters (bounded, finite-difference Jacobian).

Failures are counted, never hidden. Segments that fail at the start point are dropped once, with
their reasons recorded; a segment that fails later contributes a fixed large residual
(FAILED_RESIDUAL_M on every entry), so the trial point is rejected by the trust region, and every
such failure is counted in the result. Only numerical failures of the replay are caught
(NUMERICAL); anything else is a bug and stops the fit.

Uncertainty: the covariance of the fitted parameters is the sandwich estimate clustered by flight,
(JᵀJ)⁻¹ (Σ_f J_fᵀ r_f r_fᵀ J_f) (JᵀJ)⁻¹ · F/(F−1), because residuals within a flight are strongly
correlated (neighbouring frames and segments); the classical s²(JᵀJ)⁻¹, which assumes independent
residuals, is reported beside it and is much too small. Both are local (linearised at the optimum).
"""
from concurrent.futures import ThreadPoolExecutor
import math
import os

import numpy as np
from scipy.optimize import least_squares

FAILED_RESIDUAL_M = .5
NUMERICAL = (RuntimeError, ValueError, FloatingPointError, np.linalg.LinAlgError, ZeroDivisionError)


def run_all(function, items, *, threads=None):
    """[(result or None, error message or None)] for every item, in parallel threads."""
    def one(item):
        try:
            return function(item), None
        except NUMERICAL as exc:
            return None, f'{type(exc).__name__}: {exc}'
    with ThreadPoolExecutor(threads or os.cpu_count()) as pool:
        return list(pool.map(one, items))


def bias_profile(errors):
    """Remove one constant acceleration bias per axis shared by all segments of a flight:
    e − b t²/2 with b the least-squares value. errors: list of (N, 3) arrays from segment starts
    (t = 0.01, 0.02, ...). Returns (residual arrays, b)."""
    numerator, denominator = np.zeros(3), 0.
    for e in errors:
        basis = .5*((np.arange(len(e))+1)*.01)**2
        numerator += basis@e
        denominator += basis@basis
    b = numerator/denominator if denominator else np.zeros(3)
    return [e-np.outer(.5*((np.arange(len(e))+1)*.01)**2, b) for e in errors], b


def uncertainty(J, r, groups, names):
    """Local uncertainty at the optimum. groups: index arrays of the residuals of each flight."""
    J, r = np.asarray(J, float), np.asarray(r, float)
    m, n = J.shape
    information = J.T@J
    singular = np.linalg.svd(J, compute_uv=False)
    inverse = np.linalg.pinv(information)
    classical = inverse*(r@r)/max(1, m-n)
    meat = np.zeros((n, n))
    for g in groups:
        s = J[g].T@r[g]
        meat += np.outer(s, s)
    F = len(groups)
    clustered = inverse@meat@inverse*(F/(F-1) if F > 1 else 1.)
    sd = np.sqrt(np.clip(np.diag(clustered), 0, None))
    with np.errstate(invalid='ignore', divide='ignore'):
        correlation = clustered/np.outer(sd, sd)
    return dict(parameters=list(names), standard_deviation=sd.tolist(),
                standard_deviation_independent=np.sqrt(np.clip(np.diag(classical), 0, None)).tolist(),
                correlation=np.nan_to_num(correlation).round(4).tolist(), flights=F,
                condition_number=float(singular[0]/singular[-1]) if singular[-1] > 0 else math.inf,
                note='clustered by flight (sandwich), linearised at the optimum; 95 % interval ±1.96 sd')


def strong_correlations(u, threshold=.9):
    names, c = u['parameters'], np.asarray(u['correlation'])
    return [(names[i], names[j], float(c[i, j])) for i in range(len(names)) for j in range(i+1, len(names))
            if abs(c[i, j]) >= threshold]


class Fit:
    """scipy least squares on a residual function, with progress reporting and failure counts.
    residuals(x) returns (r, failures); value(r) the loss shown in the progress (an RMS in m)."""

    def __init__(self, residuals, value, *, progress=None, extra=lambda x: list(x)):
        self.residuals, self.value, self.progress, self.extra = residuals, value, progress, extra
        self.best = math.inf
        self.evaluations = self.failed_evaluations = self.failed_segments = 0

    def __call__(self, x):
        r, failures = self.residuals(x)
        self.evaluations += 1
        if failures:
            self.failed_evaluations += 1
            self.failed_segments += failures
        value = self.value(r)
        if self.progress is not None:
            self.progress.evaluation(self.extra(x), value if not failures else math.nan)
            if not failures and value < self.best:
                self.progress.iteration(self.evaluations, value, self.extra(x))
        if not failures:
            self.best = min(self.best, value)
        return r

    def solve(self, x0, *, lower, upper, diff_step, max_evaluations):
        return least_squares(self, np.asarray(x0, float), bounds=(lower, upper), method='trf', jac='2-point',
                             diff_step=diff_step, x_scale='jac', max_nfev=max_evaluations, xtol=1e-6, ftol=1e-6)

    def counts(self):
        return dict(evaluations=self.evaluations, evaluations_with_failed_segments=self.failed_evaluations,
                    failed_segment_replays=self.failed_segments)
