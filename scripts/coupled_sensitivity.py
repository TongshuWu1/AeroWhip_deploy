"""What the coupled data can identify: sensitivity of the coupled loss to each candidate parameter
at the configuration now, and the correlations between them.

Forward differences of the coupled residual vector (`aerowhip.fitting.coupled`) on the cable fit
flights give a Jacobian column per parameter. From it: the loss change for a one-step change of
each parameter, split into drone position, tilt and cable shape; the correlation matrix of the
columns (parameters that move the residuals the same way cannot be told apart); and the
linearised 95 % intervals a joint fit of all of them would have (clustered by flight).

    python scripts/coupled_sensitivity.py --output runs/fits/coupled_sensitivity/<time>.json
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
from aerowhip.fitting.least_squares import strong_correlations, uncertainty
from aerowhip.fitting.progress import Progress
from aerowhip.physics.model_configuration import DEFAULT_MODEL, load_model_configuration
sys.path.insert(0, str(Path(__file__).resolve().parent))
from fit_firmware_drone import FIRMWARE

STEPS = dict(com_z=.005, attachment_z=.003, log10_EI=.1, log10_Cb=.1, Cd=.1, log10_drag=.1)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--horizon', type=float, default=1.)
    args = parser.parse_args()
    t0 = time.time()
    config = load_model_configuration(DEFAULT_MODEL)
    firmware = tomllib.loads(FIRMWARE.read_text())
    x0 = coupled.current(config)
    keys = list(coupled.KEYS)
    progress = Progress(args.output, stage='coupled_sensitivity', parameters=keys, x0=[x0[k] for k in keys], unit='relative')
    progress.plan(['loading flights', 'reference point', *(f'step {k}' for k in keys)], [3, 2, *[1]*len(keys)])
    progress.phase(0)
    train, held = coupled.segments(args.horizon)
    progress.phase_done(math.nan)
    progress.phase(1)
    residuals = coupled.Residuals(config, firmware, train)
    errors, messages = residuals.errors(**x0)
    dropped = {n: m for n, m in messages.items() if m}
    train = {n: [w for w, e in zip(train[n], errors[n]) if e is not None] for n in train}
    residuals.runner.close()
    residuals = coupled.Residuals(config, firmware, {n: v for n, v in train.items() if v})
    residuals.set_scales(**x0)
    r0, failures, report0 = residuals(**x0)
    print(f'{sum(map(len, residuals.groups.values()))} segments from {len(residuals.names)} fit flights '
          f'({sum(map(len, dropped.values()))} dropped at the start point); reference: position {report0["position_m"]*1e3:.2f} mm, '
          f'tilt {report0["tilt_deg"]:.3f} deg, shape {report0["shape_m"]*1e3:.2f} mm', flush=True)
    progress.phase_done(1., [x0[k] for k in keys])
    n = len(r0)//15
    kinds = np.concatenate([np.repeat([0, 1, 2], [3, 2, 10])]*n)          # which component each residual is
    columns, rows = [], []
    for i, key in enumerate(keys):
        progress.phase(2+i)
        x = dict(x0); x[key] += STEPS[key]
        r, f, report = residuals(**x)
        progress.evaluation([x[k] for k in keys], residuals.value(r))
        column = (r-r0)/STEPS[key]
        columns.append(column)
        change = residuals.value(r)-1.
        parts = {name: float(np.sqrt(np.mean(r[kinds == c]**2))-np.sqrt(np.mean(r0[kinds == c]**2)))
                 for c, name in enumerate(('position', 'tilt', 'shape'))}
        rows.append(dict(parameter=key, step=STEPS[key], loss_change=change, component_change=parts,
                         position_m=report['position_m'], tilt_deg=report['tilt_deg'], shape_m=report['shape_m'], failed=f))
        print(f'{key:14s} +{STEPS[key]:<6g}: loss {1+change:.4f} ({change:+.4f}); position {parts["position"]:+.4f}, '
              f'tilt {parts["tilt"]:+.4f}, shape {parts["shape"]:+.4f} (normalised units)', flush=True)
        progress.phase_done(1.+change, [x[k] for k in keys])
    J = np.column_stack(columns)
    u = uncertainty(J, r0, residuals.flight_groups(), keys)
    sd = np.array(u['standard_deviation'])
    print('if all six were fitted jointly, 95 % half-widths at this point:',
          {k: f'{1.96*s:.3g}' for k, s in zip(keys, sd)}, flush=True)
    for a, b, c in strong_correlations(u, .8):
        print(f'  correlated: {a} / {b} {c:+.2f}', flush=True)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(dict(
        stage='coupled_sensitivity', data='manifest cable flights, role fit', horizon_s=args.horizon,
        reference=dict(values=x0, position_m=report0['position_m'], tilt_deg=report0['tilt_deg'], shape_m=report0['shape_m'],
                       per_flight_position_m=report0['per_flight_position_m'], per_flight_tilt_deg=report0['per_flight_tilt_deg'],
                       per_flight_shape_m=report0['per_flight_shape_m'], failed=failures),
        scales=dict(position_m=residuals.scale[0], tilt_deg=math.degrees(residuals.scale[1]), shape_m=residuals.scale[2]),
        segments={n: len(v) for n, v in residuals.groups.items()}, held_out={n: len(v) for n, v in held.items()},
        dropped_at_start=dropped, sensitivity=rows, joint_uncertainty=u, seconds=time.time()-t0), indent=1, default=float))
    progress.done([x0[k] for k in keys], 1.)


if __name__ == '__main__':
    main()
