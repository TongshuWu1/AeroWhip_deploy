"""Fit the twist-free cable (EI, Cb, normal drag Cd, linear drag) on the cable calibration flights.

The measured drone pose drives the clamp; the cable runs open loop for 2 s from the measured
markers. Nelder-Mead on (log10 EI, log10 Cb, Cd, log10 linear drag) minimizes the pooled RMS
marker error over the fitting flights; the held-out flights are scored only at the start and
end. Every reported loss is computed twice (the second pass must reproduce the first).

    python scripts/fit_cable_flights.py --output runs/calibration_2026-10-27/cable_fit_v2.json
"""
import argparse
import json
import math
from pathlib import Path
import sys
import time

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from aerowhip.fitting import flight_cable
from aerowhip.fitting.progress import Progress
from aerowhip.fitting.calibration_set import split
from aerowhip.physics.model import Model
from aerowhip.physics.model_configuration import DEFAULT_MODEL, load_model_configuration

NAMES = ('EI_n_m2', 'Cb_n_m2_s', 'normal_drag_coefficient', 'external_drag_s_inv')


def to_values(x):
    values = dict(EI_n_m2=10**x[0], Cb_n_m2_s=10**x[1], normal_drag_coefficient=max(0., x[2]),
                  external_drag_s_inv=10**x[3])
    if len(x) == 5:              # saturating bending damping: log10 of the saturation rate (rad/s)
        values['Cb_saturation_rate_rad_s'] = 10**x[4]
    return values


def score(config, groups, values):
    native = Model(flight_cable.with_cable(config, **values)).native
    per = {}
    for motion, ws in groups.items():
        per[motion] = flight_cable.rms(flight_cable.evaluate(native, ws))
    pooled = flight_cable.rms(flight_cable.evaluate(native, [w for ws in groups.values() for w in ws]))
    return pooled, per


def nelder_mead(f, x0, step, *, iterations=80, tolerance=2e-5, progress=None):
    """Minimise f from x0. progress (aerowhip.fitting.progress.Progress or None) receives every
    evaluation and every iteration's best point."""
    if progress is not None:
        objective = f

        def f(x):
            value = objective(x)
            progress.evaluation(x, value)
            return value
    n = len(x0)
    simplex = [np.array(x0, float)]+[np.array(x0, float)+np.eye(n)[i]*step[i] for i in range(n)]
    values = [f(x) for x in simplex]
    history = []
    for it in range(iterations):
        order = np.argsort(values)
        simplex, values = [simplex[i] for i in order], [values[i] for i in order]
        history.append((it, float(values[0]), simplex[0].tolist()))
        if progress is not None:
            progress.iteration(it, values[0], simplex[0])
        print(f'  iteration {it:2d}  best {values[0]*1e3:.3f} mm  x {np.round(simplex[0], 4).tolist()}', flush=True)
        if values[-1]-values[0] < tolerance:
            break
        centroid = np.mean(simplex[:-1], axis=0)
        reflected = centroid+(centroid-simplex[-1]); fr = f(reflected)
        if fr < values[0]:
            expanded = centroid+2*(centroid-simplex[-1]); fe = f(expanded)
            simplex[-1], values[-1] = (expanded, fe) if fe < fr else (reflected, fr)
        elif fr < values[-2]:
            simplex[-1], values[-1] = reflected, fr
        else:
            contracted = centroid+.5*(simplex[-1]-centroid); fc = f(contracted)
            if fc < values[-1]:
                simplex[-1], values[-1] = contracted, fc
            else:
                for i in range(1, n+1):
                    simplex[i] = simplex[0]+.5*(simplex[i]-simplex[0]); values[i] = f(simplex[i])
    best = int(np.argmin(values))
    return simplex[best], values[best], history


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--iterations', type=int, default=60)
    parser.add_argument('--start', type=float, nargs=4, metavar=('LOG10_EI', 'LOG10_CB', 'CD', 'LOG10_DRAG'),
                        help='search start (default: the configured values)')
    parser.add_argument('--step', type=float, nargs='+', default=[.5, .5, .6, .5])
    parser.add_argument('--saturation', type=float, metavar='LOG10_RATE',
                        help='also fit a saturating bending damping, starting at this log10 saturation rate (rad/s)')
    args = parser.parse_args()
    config = load_model_configuration(DEFAULT_MODEL)
    train, held = split('cf_2')
    windows = [w for ws in train.values() for w in ws]
    print(f'{len(windows)} fitting windows, {sum(map(len, held.values()))} held out', flush=True)
    start = {k: float(config['cable'][k]) for k in NAMES}
    x0 = [math.log10(start['EI_n_m2']), math.log10(start['Cb_n_m2_s']), start['normal_drag_coefficient'],
          math.log10(start['external_drag_s_inv'])]
    if args.start:
        x0 = list(args.start)
    step = list(args.step)
    if args.saturation is not None:
        x0.append(args.saturation)
        step = step[:4]+[step[4] if len(step) > 4 else .5]
    evaluations = []

    def objective(x):
        if x[2] < 0 or x[2] > 6:
            return 1.
        native = Model(flight_cable.with_cable(config, **to_values(x))).native
        value = flight_cable.rms(flight_cable.evaluate(native, windows))
        value = 1. if not math.isfinite(value) else value
        evaluations.append((list(map(float, x)), value))
        return value
    t0 = time.time()
    initial_train, initial_per = score(config, train, start)
    initial_held, initial_held_per = score(config, held, start)
    print(f'start: fit {initial_train*1e3:.2f} mm, held out {initial_held*1e3:.2f} mm', flush=True)
    names = ['log10 EI', 'log10 Cb', 'Cd', 'log10 linear drag']+(['log10 Cb saturation rate'] if len(x0) == 5 else [])
    progress = Progress(args.output, stage='cable', parameters=names, x0=x0, unit='m')
    x, value, history = nelder_mead(objective, x0, step, iterations=args.iterations, progress=progress)
    best = to_values(x)
    passes = [score(config, train, best)+score(config, held, best) for _ in range(2)]
    reproduced = passes[0] == passes[1]
    fit_rms, fit_per, held_rms, held_per = passes[0]
    print(f'best {best}\nfit {fit_rms*1e3:.2f} mm {dict((k, round(v*1e3, 2)) for k, v in fit_per.items())}\n'
          f'held out {held_rms*1e3:.2f} mm {dict((k, round(v*1e3, 2)) for k, v in held_per.items())}\nreproduced: {reproduced}', flush=True)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(dict(
        model='der_twist_free_kelvin_voigt_v2'+(' with saturating bending damping' if len(x0) == 5 else ''), data='calibration flights 2026-10-27, setup cf_2 (cable)',
        windows=dict(duration_s=2., stride_s=1., fit={k: len(v) for k, v in train.items()}, held_out={k: len(v) for k, v in held.items()}),
        start=dict(values=start, fit_rms_m=initial_train, fit_per_flight_rms_m=initial_per, held_out_rms_m=initial_held,
                   held_out_per_flight_rms_m=initial_held_per),
        best=dict(values=best, fit_rms_m=fit_rms, fit_per_flight_rms_m=fit_per, held_out_rms_m=held_rms,
                  held_out_per_flight_rms_m=held_per, reproduced_twice=reproduced),
        search=dict(method='Nelder-Mead on (log10 EI, log10 Cb, Cd, log10 linear drag'
                           +(', log10 Cb saturation rate)' if len(x0) == 5 else ')'), start=list(map(float, x0)),
                    step=step, history=history,
                    evaluations=evaluations), seconds=time.time()-t0), indent=1))
    progress.done(x, value)


if __name__ == '__main__':
    main()
