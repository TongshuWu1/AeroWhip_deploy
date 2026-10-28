"""Stage 4, coupled refinement: least squares on 1 s segments of the cable flights with the full
model (drone driven by the sent commands, carrying the cable), from the measured drone state and
markers. Residuals: drone position, drone tilt and cable shape, each normalised to 1 at the
configuration now (`aerowhip.fitting.coupled`).

Which parameters are fitted is a choice (--fit): the coupled data is the only data that sees the
force the cable puts on the drone, so the default set is the geometry of that coupling, the drone's
centre-of-mass height (unmeasured) and the attachment height (measured 55 mm below the tracking
origin ± 3 mm, bounded to that). The cable's own parameters (EI, Cb, Cd, linear drag) keep their
measured-clamp values unless named; the drone's own parameters stay as the drone-only fit found
them. Held-out flights are scored at the start and the end. Confidence intervals: clustered by
flight (`aerowhip.fitting.least_squares`).

    python scripts/fit_coupled_segments.py --output runs/fits/coupled_fit/<time>.json [--fit com_z attachment_z]
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
from aerowhip.fitting import coupled
from aerowhip.fitting.least_squares import Fit, strong_correlations, uncertainty
from aerowhip.fitting.progress import Progress
from aerowhip.physics.model_configuration import DEFAULT_MODEL, load_model_configuration
sys.path.insert(0, str(Path(__file__).resolve().parent))
from fit_firmware_drone import FIRMWARE

BOUNDS = dict(com_z=(-.08, .02), attachment_z=(-.058, -.052), log10_EI=(-9., -3.), log10_Cb=(-7., -1.),
              Cd=(0., 6.), log10_drag=(-4., .5))
STEP = dict(com_z=.002, attachment_z=.001, log10_EI=.02, log10_Cb=.02, Cd=.02, log10_drag=.02)   # finite-difference steps


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--fit', nargs='+', default=['com_z', 'attachment_z'], choices=coupled.KEYS)
    parser.add_argument('--horizon', type=float, default=1.)
    parser.add_argument('--evaluations', type=int, default=25, help='least-squares steps')
    args = parser.parse_args()
    t0 = time.time()
    config = load_model_configuration(DEFAULT_MODEL)
    firmware = tomllib.loads(FIRMWARE.read_text())
    keys = list(args.fit)
    x_all = coupled.current(config)
    x_now = np.array([x_all[k] for k in keys])
    lower, upper = np.array([BOUNDS[k][0] for k in keys]), np.array([BOUNDS[k][1] for k in keys])
    progress = Progress(args.output, stage='coupled_fit', parameters=keys, x0=x_now, unit='relative')
    progress.plan(['loading flights', 'start point', 'least squares', 'final check'],
                  [3, 2, Fit.budget(len(keys), args.evaluations), 3])
    progress.phase(0)
    train, held = coupled.segments(args.horizon)
    progress.phase_done(math.nan)
    progress.phase(1)
    residuals = coupled.Residuals(config, firmware, train)
    errors, messages = residuals.errors(**x_all)
    dropped = {n: m for n, m in messages.items() if m}
    train = {n: [w for w, e in zip(train[n], errors[n]) if e is not None] for n in train}
    residuals.runner.close()
    residuals = coupled.Residuals(config, firmware, {n: v for n, v in train.items() if v})
    residuals.set_scales(**x_all)

    def values(x):
        return dict(x_all, **dict(zip(keys, map(float, x))))

    def evaluate(x):
        r, failures, _ = residuals(**values(x))
        return r, failures
    r0, _, report0 = residuals(**x_all)
    print(f'{sum(map(len, residuals.groups.values()))} segments of {args.horizon} s from {len(residuals.names)} flights, '
          f'{sum(map(len, held.values()))} held out; {sum(map(len, dropped.values()))} dropped (failed at the start point)', flush=True)
    print(f'fitting {keys}; start: position {report0["position_m"]*1e3:.2f} mm, tilt {report0["tilt_deg"]:.3f} deg, '
          f'shape {report0["shape_m"]*1e3:.2f} mm; loss {residuals.value(r0):.4f}', flush=True)
    progress.phase_done(residuals.value(r0), x_now)
    progress.phase(2)
    fit = Fit(evaluate, residuals.value, progress=progress)
    out = fit.solve(np.clip(x_now, lower, upper), lower=lower, upper=upper,
                    diff_step=np.array([STEP[k]/max(abs(x_all[k]), 1e-9) for k in keys]), max_evaluations=args.evaluations)
    x = out.x
    progress.phase_done(residuals.value(out.fun), x)
    u = uncertainty(out.jac, out.fun, residuals.flight_groups(), keys)
    progress.phase(3)
    passes = [residuals(**values(x)) for _ in range(2)]
    reproduced = bool(np.array_equal(passes[0][0], passes[1][0]))
    r, failures, report = passes[0]
    held_score = None
    if any(held.values()):
        h = coupled.Residuals(config, firmware, {n: v for n, v in held.items() if v})
        h.scale = residuals.scale
        rh0, _, reph0 = h(**x_all)
        rh, fh, reph = h(**values(x))
        held_score = dict(start=dict(loss=h.value(rh0), position_m=reph0['position_m'], tilt_deg=reph0['tilt_deg'], shape_m=reph0['shape_m']),
                          best=dict(loss=h.value(rh), position_m=reph['position_m'], tilt_deg=reph['tilt_deg'], shape_m=reph['shape_m'],
                                    per_flight_shape_m=reph['per_flight_shape_m'], per_flight_position_m=reph['per_flight_position_m']),
                          failed=fh)
        print(f'held out: loss {h.value(rh0):.4f} -> {h.value(rh):.4f} (position {reph0["position_m"]*1e3:.1f} -> {reph["position_m"]*1e3:.1f} mm, '
              f'tilt {reph0["tilt_deg"]:.2f} -> {reph["tilt_deg"]:.2f} deg, shape {reph0["shape_m"]*1e3:.1f} -> {reph["shape_m"]*1e3:.1f} mm)', flush=True)
        h.runner.close()
    print(f'best: loss {residuals.value(r):.4f} (position {report["position_m"]*1e3:.2f} mm, tilt {report["tilt_deg"]:.3f} deg, '
          f'shape {report["shape_m"]*1e3:.2f} mm), {dict(zip(keys, np.round(x, 5).tolist()))} ± '
          f'{np.round(1.96*np.array(u["standard_deviation"]), 5).tolist()} (95 %); {fit.evaluations} evaluations, '
          f'{fit.failed_evaluations} with failed segments; reproduced twice: {reproduced}', flush=True)
    for a, b, c in strong_correlations(u):
        print(f'  correlated: {a} / {b} {c:+.2f}', flush=True)
    best_values = values(x)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(dict(
        method='least_squares', stage='coupled_fit', model='full model from the measured drone state and markers, sent commands',
        data='manifest cable flights, role fit', parameters=keys, fixed={k: v for k, v in x_all.items() if k not in keys},
        segments=dict(horizon_s=args.horizon, fit={n: len(v) for n, v in residuals.groups.items()},
                      held_out={n: len(v) for n, v in held.items()}, dropped_at_start=dropped),
        loss='drone position (per-flight bias removed), drone tilt (per-flight offset removed) and cable shape, each '
             'normalised by its RMS at the start; sqrt of the mean of the three squared ratios',
        scales=dict(position_m=residuals.scale[0], tilt_deg=math.degrees(residuals.scale[1]), shape_scale_m=residuals.scale[2]),
        start=dict(values=x_all, loss=residuals.value(r0), position_m=report0['position_m'], tilt_deg=report0['tilt_deg'], shape_m=report0['shape_m'],
                   per_flight_position_m=report0['per_flight_position_m'], per_flight_tilt_deg=report0['per_flight_tilt_deg'],
                   per_flight_shape_m=report0['per_flight_shape_m']),
        best=dict(x=x.tolist(), values=best_values, loss=residuals.value(r), position_m=report['position_m'], tilt_deg=report['tilt_deg'],
                  shape_m=report['shape_m'], per_flight_position_m=report['per_flight_position_m'],
                  per_flight_tilt_deg=report['per_flight_tilt_deg'], per_flight_shape_m=report['per_flight_shape_m'],
                  failed_segments=failures, reproduced_twice=reproduced, held_out=held_score),
        uncertainty=u, counts=fit.counts(), status=int(out.status),
        adopt={k: float(best_values[k]) for k in keys},
        search=dict(optimiser='scipy.optimize.least_squares trf, 2-point Jacobian, bounds', lower=lower.tolist(), upper=upper.tolist()),
        seconds=time.time()-t0), indent=1, default=float))
    progress.done(x, residuals.value(r))


if __name__ == '__main__':
    main()
