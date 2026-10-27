"""Stage 2, least squares on short segments: roll/pitch inertia, motor response, body drag, and
the command delay profiled on a grid, on the drone-only flights (measured motor arm kept).

Every flight is cut into consecutive segments (--horizon, 1 s) replayed open loop from the measured
state with the sent commands (commands still in the delay line at a segment start included). The
residual is the simulated minus measured position after removing one constant acceleration bias per
flight and axis (attitude-estimate offsets and trims the host PID absorbs; three numbers per flight,
so they cannot absorb dynamics as a per-window bias can). The delay acts only at the 2 ms controller
ticks, so the residual is a staircase in it: the other parameters are fitted at each delay of the
grid (warm-started) and the profile of the loss over the delay is reported. Confidence intervals:
`aerowhip.fitting.least_squares` (clustered by flight).

    python scripts/fit_drone_segments.py --output runs/fits/drone/<time>.json
"""
import argparse
import json
import math
from pathlib import Path
import sys
import time
import tomllib

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from aerowhip.fitting import flight_drone
from aerowhip.fitting.calibration_set import flight_windows, folders, is_held_out
from aerowhip.fitting.least_squares import FAILED_RESIDUAL_M, Fit, ProcessRunner, bias_profile, strong_correlations, uncertainty
from aerowhip.fitting.progress import Progress
from aerowhip.physics.model_configuration import DEFAULT_MODEL, load_model_configuration
from aerowhip.takes.calibration_flights import load
sys.path.insert(0, str(Path(__file__).resolve().parent))
from fit_firmware_drone import FIRMWARE, configured as vehicle

NAMES = ('inertia_scale_xy', 'motor_response_s', 'body_drag_xy_s_inv', 'body_drag_z_s_inv')
SCALE = np.array([1., .01, 1., 1.])          # the search runs on x/SCALE (all of order one)


def segments(setup, horizon):
    """({name: [(window, flight)]} fitted, {name: [...]} held out)."""
    train, held = {}, {}
    for name, folder in folders(setup).items():
        flight = load(folder)
        flight.markers = None
        items = [(w, flight) for w in flight_windows(folder, cable=False, duration_s=horizon, stride_s=horizon)]
        (held if is_held_out(folder) else train)[name] = items
    return train, held


def _replay(parameters):
    config, firmware = parameters
    replay = flight_drone.DroneReplay(config, firmware)
    return lambda item: replay.errors(*item[1], pipeline=True)


class Residuals:
    def __init__(self, base, firmware, groups):
        self.base, self.firmware, self.groups = base, firmware, groups
        self.names = list(groups)
        self.items = [(name, i) for name in self.names for i in self.groups[name]]
        self.runner = ProcessRunner(self.items, _replay)

    def config(self, x, delay):
        c = vehicle(base=self.base, x=[x[0], 1., x[1], delay], cable=False)
        c['drone']['body_drag_s_inv'] = [float(x[2]), float(x[2]), float(x[3])]
        return c

    def errors(self, x, delay):
        """{name: [(N, 3) error or None]}, {name: [message]} at physical parameters x."""
        results = self.runner((self.config(x, delay), self.firmware))
        errors, messages = {n: [] for n in self.names}, {n: [] for n in self.names}
        for (name, _), (e, message) in zip(self.items, results):
            errors[name].append(e)
            if message:
                messages[name].append(message)
        return errors, messages

    def __call__(self, x, delay):
        """(r, failures, per-flight biases, per-flight RMS) at physical parameters x."""
        errors, _ = self.errors(x, delay)
        pieces, failures, biases, per = [], 0, {}, {}
        for name in self.names:
            good = [e for e in errors[name] if e is not None]
            profiled, b = bias_profile(good)
            biases[name] = b.tolist()
            per[name] = rms(profiled)
            it = iter(profiled)
            for e, (w, _) in zip(errors[name], self.groups[name]):
                if e is None:
                    failures += 1
                    pieces.append(np.full(3*(len(w.time)-1), FAILED_RESIDUAL_M))
                else:
                    pieces.append(next(it).ravel())
        return np.concatenate(pieces), failures, biases, per

    def flight_groups(self):
        groups, start = [], 0
        for name in self.names:
            size = sum(3*(len(w.time)-1) for w, _ in self.groups[name])
            groups.append(np.arange(start, start+size))
            start += size
        return groups


def rms(arrays):
    """3-D RMS position error per frame (m) over a list of (N, 3) arrays."""
    if not arrays:
        return math.nan
    e = np.concatenate(arrays)
    return float(np.sqrt(np.mean(np.sum(e**2, axis=1))))


def value(r):
    return float(np.sqrt(np.sum(r**2)/(len(r)/3)))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--horizon', type=float, default=1., help='segment length, s')
    parser.add_argument('--delays-ms', type=float, nargs='+', default=[0, 10, 20, 30, 40, 50, 60])
    parser.add_argument('--evaluations', type=int, default=20,
                        help='least-squares steps per delay (residual evaluations, Jacobian evaluations not counted)')
    args = parser.parse_args()
    t0 = time.time()
    base = load_model_configuration(DEFAULT_MODEL)
    firmware = tomllib.loads(FIRMWARE.read_text())
    if firmware.get('law') != 'firmware':
        raise SystemExit('The firmware law is required')
    J = base['drone']['inertia_body_about_com_kg_m2']
    drag = base['drone']['body_drag_s_inv']
    x_now = np.array([1., float(base['actuator']['response_time_s'][0]), float(drag[0]), float(drag[2])])
    delay_now = float(base['controller']['command_delay_s'])
    lower = np.array([max(.3, J[2][2]/(J[0][0]+J[1][1])*1.001), .001, 0., 0.])     # triangle inequality Ixx + Iyy >= Izz
    upper = np.array([3., .08, 3., 3.])
    progress = Progress(args.output, stage='drone', parameters=[*NAMES, 'command_delay_s'],
                        x0=[*x_now, delay_now], unit='m')
    # loading weighs about ten residual evaluations; the final check is two passes
    progress.plan(['loading flights', 'start point', *(f'delay {d:.0f} ms' for d in args.delays_ms), 'final check'],
                  [10, 2, *[Fit.budget(4, args.evaluations)]*len(args.delays_ms), 3])
    progress.phase(0)
    train, held = segments('cf_2_nocable', args.horizon)
    progress.phase_done(math.nan)
    progress.phase(1)
    residuals = Residuals(base, firmware, train)
    # Segments that fail at the configuration now are dropped once, with their reasons.
    errors, messages = residuals.errors(x_now, delay_now)
    dropped = {n: m for n, m in messages.items() if m}
    for name in train:
        train[name] = [item for item, e in zip(train[name], errors[name]) if e is not None]
    residuals.runner.close()
    residuals = Residuals(base, firmware, {n: v for n, v in train.items() if v})
    count = sum(map(len, residuals.groups.values()))
    print(f'{count} segments of {args.horizon} s from {len(residuals.names)} flights, {sum(map(len, held.values()))} held out; '
          f'{sum(map(len, dropped.values()))} dropped (failed at the start point)', flush=True)
    r0, _, biases0, per0 = residuals(x_now, delay_now)
    start = dict(x=x_now.tolist(), command_delay_s=delay_now, rms_m=value(r0), per_flight_rms_m=per0, bias_m_s2=biases0)
    print(f'start: {start["rms_m"]*1e3:.2f} mm (bias removed per flight)', flush=True)
    progress.phase_done(start['rms_m'], [*x_now, delay_now])
    profile, x = [], x_now.copy()
    fits = {}
    for k, d in enumerate(args.delays_ms):
        progress.phase(2+k)
        delay = d/1e3
        fit = Fit(lambda u: residuals(u*SCALE, delay)[:2], value, progress=progress,
                  extra=lambda u: [*(u*SCALE), delay])
        out = fit.solve(np.clip(x, lower, upper)/SCALE, lower=lower/SCALE, upper=upper/SCALE,
                        diff_step=np.full(4, 1e-2), max_evaluations=args.evaluations)
        x = out.x*SCALE
        loss = value(out.fun)
        fits[d] = (out, fit)
        profile.append(dict(command_delay_s=delay, rms_m=loss, x=x.tolist(), status=int(out.status), **fit.counts()))
        progress.phase_done(loss, [*x, delay])
        print(f'delay {d:4.0f} ms: {loss*1e3:.3f} mm  x {np.round(x, 4).tolist()}  ({fit.evaluations} evaluations, '
              f'{fit.failed_evaluations} with failed segments)', flush=True)
    best = min(range(len(profile)), key=lambda i: profile[i]['rms_m'])
    delay = profile[best]['command_delay_s']
    out, fit = fits[args.delays_ms[best]]
    x = out.x*SCALE
    u = uncertainty(out.jac/SCALE, out.fun, residuals.flight_groups(), NAMES)   # Jacobian per physical unit
    progress.phase(2+len(args.delays_ms))
    passes = [residuals(x, delay) for _ in range(2)]
    reproduced = bool(np.array_equal(passes[0][0], passes[1][0]))
    r, failures, biases, per = passes[0]
    held_score = None
    if any(held.values()):
        rh, fh, bh, ph = Residuals(base, firmware, {n: v for n, v in held.items() if v})(x, delay)
        held_score = dict(rms_m=value(rh), failed=fh, per_flight_rms_m=ph, bias_m_s2=bh)
    print(f'best: delay {delay*1e3:.0f} ms, {value(r)*1e3:.3f} mm, x {np.round(x, 5).tolist()} ± '
          f'{np.round(1.96*np.array(u["standard_deviation"]), 5).tolist()} (95 %); reproduced twice: {reproduced}', flush=True)
    for a, b, c in strong_correlations(u):
        print(f'  correlated: {a} / {b} {c:+.2f}', flush=True)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(dict(
        method='least_squares', stage='drone', model='firmware law, measured motor arm, drone-only (one 2 cm 1 mg link)',
        data='manifest flights, setup cf_2_nocable', parameters=list(NAMES), firmware=firmware,
        segments=dict(horizon_s=args.horizon, fit={n: len(v) for n, v in residuals.groups.items()},
                      held_out={n: len(v) for n, v in held.items()}, dropped_at_start=dropped),
        loss='RMS 3-D position error per frame over all segments, one constant acceleration bias per flight and axis removed',
        start=start,
        best=dict(x=x.tolist(), command_delay_s=delay, rms_m=value(r), per_flight_rms_m=per, bias_m_s2=biases,
                  failed_segments=failures, reproduced_twice=reproduced, held_out=held_score),
        uncertainty=u, delay_profile=profile,
        adopt=dict(inertia_xy_kg_m2=float(J[0][0]*x[0]), arm_scale=1., motor_response_s=float(x[1]),
                   command_delay_s=float(delay), body_drag_s_inv=[float(x[2]), float(x[2]), float(x[3])]),
        search=dict(optimiser='scipy.optimize.least_squares trf, 2-point Jacobian, bounds',
                    lower=lower.tolist(), upper=upper.tolist(), evaluations_per_delay=args.evaluations),
        seconds=time.time()-t0), indent=1, default=float))
    progress.done([*x, delay], value(r))


if __name__ == '__main__':
    main()
