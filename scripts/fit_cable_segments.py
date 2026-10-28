"""Stage 3, least squares on short segments: bending stiffness EI, bending damping Cb, normal drag
coefficient Cd and linear drag, on the cable flights; with --terms, jointly with the compact cable
residual (`cable.residual`, docs/simulation_model.md):

    static  a steady force per unit length along the cable, per axis a quadratic in the arc fraction,
            searched in the Legendre basis P0, P1, P2 on [0, 1] (units of 0.01 N/m) and stored as
            monomial coefficients; in the drone's body frame or the world frame (--frame)
    drag    the normal-drag multiplier as its log at the clamp and at the tip, and a tangential
            quadratic drag coefficient at the clamp and at the tip (each linear in between)
    saturation  the bending damping's saturation rate (`cable.Cb_saturation_rate_rad_s`, log10 rad/s): each
            joint's Cb divided by sqrt(1 + (bend rate / rate)^2), so fast bending is damped less (a
            physical parameter, absent = linear damping)
    clamp   the direction in which the clamp holds the cable (`attachment.tangent_body`), as tilts
            about the body x and y axes from straight down, in degrees (a physical parameter: it
            bends the hanging shape without any net force on drone and cable)

The measured drone pose drives the clamp; every flight is cut into consecutive segments (--horizon,
1 s) and each segment's cable starts from the measured markers. Residuals: simulated minus measured
marker position, per component, for every marker seen (hidden markers contribute nothing). The
reported loss is the RMS marker distance. Parameters are searched as (log10 EI, log10 Cb, Cd,
log10 linear drag). Failures and confidence intervals: `aerowhip.fitting.least_squares`.

    python scripts/fit_cable_segments.py --output runs/fits/cable/<time>.json
"""
import argparse
import json
import math
from pathlib import Path
import sys
import time

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from aerowhip.fitting import flight_cable
from aerowhip.fitting.calibration_set import is_held_out, split
from aerowhip.fitting.least_squares import Fit, FAILED_RESIDUAL_M, ProcessRunner, strong_correlations, uncertainty
from aerowhip.fitting.progress import Progress
from aerowhip.physics.model import Model
from aerowhip.physics.model_configuration import DEFAULT_MODEL, fit_configuration, load_model_configuration

KEYS = ('EI_n_m2', 'Cb_n_m2_s', 'normal_drag_coefficient', 'external_drag_s_inv')
PHYSICS = ('log10 EI', 'log10 Cb', 'Cd', 'log10 linear drag')
STATIC = tuple(f'static {axis} P{k} (0.01 N/m)' for axis in 'xyz' for k in range(3))
DRAG = ('normal drag log, clamp', 'normal drag log, tip', 'tangential drag, clamp', 'tangential drag, tip')
CLAMP = ('clamp tilt about x (deg)', 'clamp tilt about y (deg)')
SATURATION = ('log10 Cb saturation rate (rad/s)',)
STATIC_UNIT = .01                     # N/m per search unit
# Legendre P0, P1, P2 on [0, 1] in monomials (rows: P_k; columns: 1, s, s^2)
LEGENDRE = np.array([[1., 0., 0.], [-1., 2., 0.], [1., -6., 6.]])


class Parameters:
    """The search vector: the four physical parameters, then the residual terms asked for, without the
    ones held fixed (`fixed`: names held at their start value; vertical='constant': the vertical static
    force is P0 only, its P1 and P2 held at zero)."""

    def __init__(self, terms, frame, fixed=(), vertical='quadratic'):
        self.terms, self.frame = terms, frame
        self.full = (list(PHYSICS)+(list(STATIC) if 'static' in terms else [])+(list(DRAG) if 'drag' in terms else [])
                     +(list(CLAMP) if 'clamp' in terms else [])+(list(SATURATION) if 'saturation' in terms else []))
        self.zeroed = {'static z P1 (0.01 N/m)', 'static z P2 (0.01 N/m)'} if vertical == 'constant' and 'static' in terms else set()
        unknown = set(fixed)-set(self.full)
        if unknown:
            raise SystemExit(f'cannot fix {sorted(unknown)}: not among {self.full}')
        self.held = set(fixed) | self.zeroed
        self.names = [n for n in self.full if n not in self.held]
        self.fixed_values = {}

    def bounds(self):
        lower, upper = [-9., -7., 0., -4.], [-3., -1., 6., .5]
        if 'static' in self.terms:
            lower += [-5.]*9; upper += [5.]*9
        if 'drag' in self.terms:
            lower += [math.log(.3)]*2+[0.]*2; upper += [math.log(3.)]*2+[3.]*2
        if 'clamp' in self.terms:
            lower += [-15.]*2; upper += [15.]*2
        if 'saturation' in self.terms:
            lower += [-1.]; upper += [3.]
        keep = [n in self.names for n in self.full]
        return np.array(lower)[keep], np.array(upper)[keep]

    def expand(self, x):
        """{name: value} of every parameter, the held ones included."""
        return dict(self.fixed_values, **dict(zip(self.names, map(float, x))))

    def start(self, config, previous=None, override=None):
        """The configuration's values (residual terms from its `cable.residual` if present), overridden
        by any parameter of the same name in a previous result (warm start), then by `override`.
        Records the values of the held parameters; returns the free ones."""
        cable = config['cable']
        x = dict(zip(PHYSICS, [math.log10(cable['EI_n_m2']), math.log10(cable['Cb_n_m2_s']),
                               cable['normal_drag_coefficient'], math.log10(cable['external_drag_s_inv'])]))
        r = cable.get('residual') or {}
        if 'static' in self.terms:
            monomial = np.asarray(r.get('static_n_per_m', np.zeros((3, 3))), float)
            legendre = monomial@np.linalg.inv(LEGENDRE)/STATIC_UNIT
            x.update(zip(STATIC, legendre.ravel()))
        if 'drag' in self.terms:
            b, c = r.get('normal_drag_log', [0., 0.]), r.get('tangential_drag', [0., 0.])
            x.update(zip(DRAG, [b[0], b[0]+b[1], c[0], c[0]+c[1]]))
        if 'clamp' in self.terms:
            t = np.asarray(config['attachment']['tangent_body'], float)
            t = t/np.linalg.norm(t)
            x.update(zip(CLAMP, [math.degrees(math.asin(t[1])), math.degrees(math.atan2(-t[0], -t[2]))]))
        if 'saturation' in self.terms:
            rate = cable.get('Cb_saturation_rate_rad_s')
            x[SATURATION[0]] = math.log10(rate) if rate else 2.
        if previous is not None:
            x.update({n: v for n, v in zip(previous['parameters'], previous['best']['x']) if n in x})
        x.update({n: v for n, v in (override or {}).items() if n in x})
        x.update({n: 0. for n in self.zeroed})
        self.fixed_values = {n: float(x[n]) for n in self.held}
        return np.array([x[n] for n in self.names], float)

    def physics(self, x):
        v = self.expand(x)
        return dict(EI_n_m2=10**v[PHYSICS[0]], Cb_n_m2_s=10**v[PHYSICS[1]], normal_drag_coefficient=float(v[PHYSICS[2]]),
                    external_drag_s_inv=10**v[PHYSICS[3]])

    def tangent(self, x):
        """`attachment.tangent_body` for x (None without the clamp term): straight down tilted about
        body x, then about body y."""
        if 'clamp' not in self.terms:
            return None
        v = self.expand(x)
        ax, ay = (math.radians(v[n]) for n in CLAMP)
        return [-math.sin(ay)*math.cos(ax), math.sin(ax), -math.cos(ay)*math.cos(ax)]

    def residual(self, x):
        """The `cable.residual` entry for x (None without residual terms)."""
        if not set(self.terms) & {'static', 'drag'}:
            return None
        values = self.expand(x)
        r = dict(frame=self.frame)
        if 'static' in self.terms:
            legendre = np.array([values[n] for n in STATIC]).reshape(3, 3)*STATIC_UNIT
            r['static_n_per_m'] = (legendre@LEGENDRE).tolist()
        if 'drag' in self.terms:
            m0, m1, t0, t1 = (values[n] for n in DRAG)
            r['normal_drag_log'], r['tangential_drag'] = [m0, m1-m0], [t0, t1-t0]
        return r

    def config(self, base, x):
        c = flight_cable.with_cable(base, **self.physics(x))
        residual = self.residual(x)
        if residual is not None:
            c['cable']['residual'] = residual
        tangent = self.tangent(x)
        if tangent is not None:
            c['attachment'] = dict(c['attachment'], tangent_body=tangent)
        if 'saturation' in self.terms:
            c['cable']['Cb_saturation_rate_rad_s'] = 10**self.expand(x)[SATURATION[0]]
        return c

    def saturation(self, x):
        return 10**self.expand(x)[SATURATION[0]] if 'saturation' in self.terms else None


def _replay(config):
    replay = flight_cable.CableReplay(Model(fit_configuration(config)).native)
    return lambda item: replay.run(item[1])


class Residuals:
    def __init__(self, config, groups, parameters):
        self.config, self.groups, self.names = config, groups, list(groups)
        self.parameters = parameters
        self.items = [(n, w) for n in self.names for w in self.groups[n]]
        self.runner = ProcessRunner(self.items, _replay)

    def errors(self, x):
        results = self.runner(self.parameters.config(self.config, x))
        out, messages = {n: [] for n in self.names}, {n: [] for n in self.names}
        for (name, w), (predicted, message) in zip(self.items, results):
            out[name].append(None if predicted is None else predicted-w.markers[1:])
            if message:
                messages[name].append(message)
        return out, messages

    def __call__(self, x):
        """(r, failures, per-flight RMS marker distance)."""
        errors, _ = self.errors(x)
        pieces, failures, per = [], 0, {}
        for name in self.names:
            squares, seen = 0., 0
            for d, w in zip(errors[name], self.groups[name]):
                mask = w.seen[1:]
                if d is None:
                    failures += 1
                    piece = np.where(mask[..., None], FAILED_RESIDUAL_M, 0.)
                else:
                    piece = np.where(mask[..., None], d, 0.)
                    squares += float(np.sum(piece**2))
                    seen += int(mask.sum())
                pieces.append(piece.ravel())
            per[name] = math.sqrt(squares/seen) if seen else math.nan
        return np.concatenate(pieces), failures, per

    def seen(self):
        return sum(int(w.seen[1:].sum()) for n in self.names for w in self.groups[n])

    def flight_groups(self):
        groups, start = [], 0
        for name in self.names:
            size = sum((len(w.time)-1)*10*3 for w in self.groups[name])
            groups.append(np.arange(start, start+size))
            start += size
        return groups


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--horizon', type=float, default=1., help='segment length, s')
    parser.add_argument('--evaluations', type=int, default=25,
                        help='least-squares steps (residual evaluations, Jacobian evaluations not counted)')
    parser.add_argument('--terms', default='', help="residual terms fitted with the physics: '', 'static' or 'static,drag'")
    parser.add_argument('--frame', choices=('body', 'world'), default='body', help='frame of the static residual force')
    parser.add_argument('--start', type=Path, help='warm start from a previous result of this stage')
    parser.add_argument('--windows', choices=('segments', 'strokes'), default='segments',
                        help='consecutive segments of every flight, or windows from each stroke onset (use --horizon 0.6)')
    parser.add_argument('--fix', nargs='+', default=[], help='parameter names held at their start value, e.g. "log10 EI"')
    parser.add_argument('--set', nargs='+', default=[], metavar='NAME=VALUE', help='start values (search units), applied after --start')
    parser.add_argument('--vertical', choices=('quadratic', 'constant'), default='quadratic',
                        help='shape of the vertical static force along the cable')
    args = parser.parse_args()
    t0 = time.time()
    config = load_model_configuration(DEFAULT_MODEL)
    terms = tuple(t for t in args.terms.split(',') if t)
    if set(terms)-{'static', 'drag', 'clamp', 'saturation'}:
        raise SystemExit(f'unknown residual terms {terms}')
    parameters = Parameters(terms, args.frame, args.fix, args.vertical)
    NAMES = parameters.names
    now = {k: float(config['cable'][k]) for k in KEYS}
    previous = json.loads(args.start.read_text()) if args.start else None
    override = {n: float(v) for n, v in (s.rsplit('=', 1) for s in args.set)}
    x_now = parameters.start(config, previous, override)
    lower, upper = parameters.bounds()
    progress = Progress(args.output, stage='cable', parameters=list(NAMES), x0=x_now, unit='m')
    # loading weighs about three residual evaluations; the final check is two passes
    progress.plan(['loading flights', 'start point', 'least squares', 'final check'],
                  [3, 2, Fit.budget(len(NAMES), args.evaluations), 3])
    progress.phase(0)
    if args.windows == 'strokes':
        # windows from each stroke onset of the decay, strokes and whip flights (stroke_diagnostics),
        # --horizon long; the role follows the flight
        import stroke_diagnostics
        train, held = {}, {}
        for name, items in stroke_diagnostics.stroke_windows('cf_2', args.horizon).items():
            if items:
                (held if is_held_out(items[0][1].folder) else train)[name] = [w for w, _ in items]
    else:
        train, held = split('cf_2', duration_s=args.horizon, stride_s=args.horizon)
    progress.phase_done(math.nan)
    progress.phase(1)
    residuals = Residuals(config, train, parameters)
    errors, messages = residuals.errors(x_now)
    dropped = {n: m for n, m in messages.items() if m}
    train = {n: [w for w, e in zip(train[n], errors[n]) if e is not None] for n in train}
    residuals.runner.close()
    residuals = Residuals(config, {n: v for n, v in train.items() if v}, parameters)
    seen = residuals.seen()

    def value(r):
        return float(np.sqrt(np.sum(r**2)/seen))
    print(f'{sum(map(len, residuals.groups.values()))} segments of {args.horizon} s from {len(residuals.names)} flights, '
          f'{sum(map(len, held.values()))} held out; {sum(map(len, dropped.values()))} dropped (failed at the start point)', flush=True)
    r0, _, per0 = residuals(x_now)
    start = dict(x=x_now.tolist(), values=dict(now, **parameters.physics(x_now)), residual=parameters.residual(x_now),
                 rms_m=value(r0), per_flight_rms_m=per0)
    print(f'start: {start["rms_m"]*1e3:.2f} mm', flush=True)
    progress.phase_done(start['rms_m'], x_now)
    progress.phase(2)
    fit = Fit(lambda x: residuals(x)[:2], value, progress=progress)
    out = fit.solve(np.clip(x_now, lower, upper), lower=lower, upper=upper, diff_step=np.full(len(NAMES), 1e-2),
                    max_evaluations=args.evaluations)
    x = out.x
    progress.phase_done(value(out.fun), x)
    u = uncertainty(out.jac, out.fun, residuals.flight_groups(), NAMES)
    progress.phase(3)
    passes = [residuals(x) for _ in range(2)]
    reproduced = bool(np.array_equal(passes[0][0], passes[1][0]))
    r, failures, per = passes[0]
    held_score = None
    if any(held.values()):
        h = Residuals(config, {n: v for n, v in held.items() if v}, parameters)
        rh0, _, ph0 = h(x_now)
        rh, fh, ph = h(x)
        held_score = dict(rms_m=float(np.sqrt(np.sum(rh**2)/h.seen())), failed=fh, per_flight_rms_m=ph,
                          start_rms_m=float(np.sqrt(np.sum(rh0**2)/h.seen())), start_per_flight_rms_m=ph0)
        print(f'held out: {held_score["start_rms_m"]*1e3:.3f} -> {held_score["rms_m"]*1e3:.3f} mm', flush=True)
        h.runner.close()
    print(f'best: {value(r)*1e3:.3f} mm, x {np.round(x, 4).tolist()} ± {np.round(1.96*np.array(u["standard_deviation"]), 4).tolist()} '
          f'(95 %); {fit.evaluations} evaluations, {fit.failed_evaluations} with failed segments; reproduced twice: {reproduced}', flush=True)
    for a, b, c in strong_correlations(u):
        print(f'  correlated: {a} / {b} {c:+.2f}', flush=True)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(dict(
        method='least_squares', stage='cable', model='der_twist_free_kelvin_voigt_v2, measured drone pose drives the clamp',
        data='manifest flights, setup cf_2 (cable labels)', parameters=list(NAMES), terms=list(terms), frame=args.frame,
        held=parameters.fixed_values, vertical=args.vertical,
        warm_start=str(args.start) if args.start else None,
        segments=dict(kind=args.windows, horizon_s=args.horizon, fit={n: len(v) for n, v in residuals.groups.items()},
                      held_out={n: len(v) for n, v in held.items()}, dropped_at_start=dropped, seen_marker_frames=seen),
        loss='RMS marker distance over every seen marker of every segment',
        start=start,
        best=dict(x=x.tolist(), values=parameters.physics(x), residual=parameters.residual(x), tangent_body=parameters.tangent(x),
                  rms_m=value(r), per_flight_rms_m=per,
                  failed_segments=failures, reproduced_twice=reproduced, held_out=held_score),
        uncertainty=u, counts=fit.counts(), status=int(out.status),
        adopt=dict({k: float(v) for k, v in parameters.physics(x).items()},
                   **({'residual': parameters.residual(x)} if parameters.residual(x) is not None else {}),
                   **({'tangent_body': parameters.tangent(x)} if parameters.tangent(x) is not None else {}),
                   **({'Cb_saturation_rate_rad_s': parameters.saturation(x)} if parameters.saturation(x) is not None else {})),
        search=dict(optimiser='scipy.optimize.least_squares trf, 2-point Jacobian, bounds', lower=lower.tolist(),
                    upper=upper.tolist()),
        seconds=time.time()-t0), indent=1, default=float))
    progress.done(x, value(r))


if __name__ == '__main__':
    main()
