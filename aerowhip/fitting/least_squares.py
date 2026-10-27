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
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, wait
from concurrent.futures.process import BrokenProcessPool
import math
import multiprocessing
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


# Worker processes. The items (segments) are set as a module global before the pool forks, so every
# worker has them without copying; each task sends only the parameters and a range of items.
_ITEMS, _PREPARE = None, None


def _ready():
    return True


def _chunk(task):
    parameters, lo, hi = task
    function = _PREPARE(parameters)
    out = []
    for item in _ITEMS[lo:hi]:
        try:
            out.append((function(item), None))
        except NUMERICAL as exc:
            out.append((None, f'{type(exc).__name__}: {exc}'))
    return out


class ProcessRunner:
    """Evaluate `prepare(parameters)(item)` for every item of a fixed list in worker processes.
    Python threads serialise on the interpreter lock (24 threads gave 4–7× here); processes do not.
    `prepare` builds what the items share (e.g. one replay model) once per chunk. Same results and
    failure handling as `run_all`, in item order."""

    RETRIES = 2

    def __init__(self, items, prepare, *, workers=None, chunks_per_worker=3):
        self.items, self.prepare = list(items), prepare
        self.workers = workers or os.cpu_count()
        self.restarts = 0
        self._start()
        count = max(1, min(len(self.items), self.workers*chunks_per_worker))
        edges = np.linspace(0, len(self.items), count+1).round().astype(int)
        self.ranges = [(int(a), int(b)) for a, b in zip(edges[:-1], edges[1:]) if b > a]

    def _start(self):
        # an executor, not multiprocessing.Pool: a worker that dies raises BrokenProcessPool instead of
        # leaving the fit waiting forever
        global _ITEMS, _PREPARE
        _ITEMS, _PREPARE = self.items, self.prepare
        self.pool = ProcessPoolExecutor(self.workers, mp_context=multiprocessing.get_context('fork'))
        wait([self.pool.submit(_ready) for _ in range(self.workers)])          # fork every worker now, with the items

    def __call__(self, parameters):
        """A worker process that dies (this machine shows sporadic memory faults in any program) does
        not end the fit: the evaluation is repeated on new workers, at most RETRIES times. The
        computation is deterministic, so a repeated evaluation gives the same numbers; each restart is
        counted (`restarts`) and printed."""
        for attempt in range(self.RETRIES+1):
            try:
                parts = self.pool.map(_chunk, [(parameters, a, b) for a, b in self.ranges])
                return [r for part in parts for r in part]
            except BrokenProcessPool:
                if attempt == self.RETRIES:
                    raise
                self.restarts += 1
                print(f'a worker process died; repeating the evaluation on new workers (restart {self.restarts})', flush=True)
                self.pool.shutdown(wait=False, cancel_futures=True)
                self._start()

    def close(self):
        self.pool.shutdown(cancel_futures=True)


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

    @staticmethod
    def budget(parameters, max_evaluations):
        """Most residual evaluations `solve` can make: each step one trial plus a 2-point
        Jacobian (one evaluation per parameter)."""
        return max_evaluations*(parameters+1)+parameters

    def solve(self, x0, *, lower, upper, diff_step, max_evaluations):
        return least_squares(self, np.asarray(x0, float), bounds=(lower, upper), method='trf', jac='2-point',
                             diff_step=diff_step, x_scale='jac', max_nfev=max_evaluations, xtol=1e-6, ftol=1e-6)

    def counts(self):
        return dict(evaluations=self.evaluations, evaluations_with_failed_segments=self.failed_evaluations,
                    failed_segment_replays=self.failed_segments)
