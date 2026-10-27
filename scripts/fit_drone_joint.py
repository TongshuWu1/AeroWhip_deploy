"""Stage 2 refit together with the drone air drag, on the drone-only flights.

Six parameters, relative to the adopted model: roll/pitch inertia scale, motor arm scale,
motor response time, command delay, and the body drag rates (k_xy, k_z). Loss: pooled raw 2 s
open-loop RMS position error on the fitting flights (with the drag in the model, the raw error is
what the simulation must reproduce; the debiased error is reported beside it). Held-out and
cable flights are scored at the start and the end. Every reported number is computed twice.

    python scripts/fit_drone_joint.py --output runs/calibration_2026-10-27/drone_joint_fit.json
"""
import argparse
from copy import deepcopy
import json
import math
from pathlib import Path
import sys
import time
import tomllib

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from aerowhip.fitting import flight_drone
from aerowhip.physics.model_configuration import DEFAULT_MODEL, load_model_configuration
sys.path.insert(0, str(Path(__file__).resolve().parent))
from fit_cable_flights import nelder_mead
from fit_drone_flights import items_for
from fit_firmware_drone import FIRMWARE, cable_items, configured as vehicle

NAMES = ('inertia_scale_xy', 'arm_scale', 'motor_response_s', 'command_delay_s', 'body_drag_xy_s_inv', 'body_drag_z_s_inv')


def configured(base, x, *, cable):
    c = vehicle(base, x[:4], cable=cable)
    c['drone']['body_drag_s_inv'] = [float(x[4]), float(x[4]), float(x[5])]
    return c


def score(base, firmware, groups, x, *, cable=False):
    config = configured(base, x, cable=cable)
    per = {m: flight_drone.evaluate(config, firmware, items) for m, items in groups.items()}
    every = [e for v in per.values() for e in v]
    return dict(raw_rms_m=flight_drone.rms(every), debiased_rms_m=flight_drone.rms(every, debiased=True),
                per_flight_raw_rms_m={m: flight_drone.rms(v) for m, v in per.items()},
                per_flight_debiased_rms_m={m: flight_drone.rms(v, debiased=True) for m, v in per.items()})


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--iterations', type=int, default=60)
    args = parser.parse_args()
    base = load_model_configuration(DEFAULT_MODEL)
    firmware = tomllib.loads(FIRMWARE.read_text())
    if firmware.get('law') != 'firmware':
        raise SystemExit('The firmware law is required')
    drag = base['drone']['body_drag_s_inv']
    x0 = [1., 1., float(base['actuator']['response_time_s'][0]), float(base['controller']['command_delay_s']), float(drag[0]), float(drag[2])]
    step = [.25, .15, .008, .01, .15, .2]
    train, held = items_for('cf_2_nocable')
    cable_train, cable_held = cable_items()
    items = [i for v in train.values() for i in v]
    print(f'{len(items)} fitting windows, {sum(map(len, held.values()))} held out; cable cross-check '
          f'{sum(map(len, cable_train.values()))} + {sum(map(len, cable_held.values()))} windows; start {x0}', flush=True)
    evaluations = []

    def clean(x):
        return [x[0], x[1], x[2], max(0., x[3]), x[4], x[5]]    # a negative delay is zero delay

    def objective(x):
        if x[0] <= .1 or x[1] <= .1 or x[2] < 0 or x[3] > .2 or x[4] < 0 or x[5] < 0:
            return 1.
        x = clean(x)
        try:
            value = flight_drone.rms(flight_drone.evaluate(configured(base, x, cable=False), firmware, items))
        except ValueError:          # not a physical vehicle (e.g. inertia violating the triangle inequality)
            return 1.
        value = 1. if not math.isfinite(value) else value
        evaluations.append((list(map(float, x)), value))
        return value

    def report(label, x):
        rows = dict(fit=score(base, firmware, train, x), held_out=score(base, firmware, held, x),
                    cable_fit=score(base, firmware, cable_train, x, cable=True),
                    cable_held_out=score(base, firmware, cable_held, x, cable=True))
        print(f'{label} x {np.round(x, 4).tolist()}: ' + '; '.join(
            f'{k} {r["raw_rms_m"]*1e3:.0f} raw / {r["debiased_rms_m"]*1e3:.1f} debiased' for k, r in rows.items()), flush=True)
        return rows
    t0 = time.time()
    start = report('start (adopted)', x0)
    x, value, history = nelder_mead(objective, x0, step, iterations=args.iterations)
    x = clean(list(x))
    passes = [report('best', x) for _ in range(2)]
    reproduced = passes[0] == passes[1]
    print('reproduced twice:', reproduced, flush=True)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(dict(
        data='calibration flights 2026-10-27, setups cf_2_nocable (fit) and cf_2 (cross-check)', law='firmware',
        firmware=firmware, parameters=list(NAMES), relative_to='adopted model (inertia and arm scales multiply the configured values)',
        windows=dict(duration_s=2., stride_s=1., fit={k: len(v) for k, v in train.items()}, held_out={k: len(v) for k, v in held.items()},
                     cable_fit={k: len(v) for k, v in cable_train.items()}, cable_held_out={k: len(v) for k, v in cable_held.items()}),
        start=dict(x=x0, **start), best=dict(x=list(map(float, x)), **passes[0], reproduced_twice=reproduced),
        search=dict(method='Nelder-Mead, raw RMS position error, drone-only flights', step=step, history=history, evaluations=evaluations),
        seconds=time.time()-t0), indent=1))


if __name__ == '__main__':
    main()
