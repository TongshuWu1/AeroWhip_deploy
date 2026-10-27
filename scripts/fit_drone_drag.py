"""Drone air drag (linear rates per body axis, `drone.body_drag_s_inv`) on the drone-only flights.

Everything else is the adopted model (firmware law, stage-2 vehicle parameters). The rates are
(k_xy, k_xy, k_z): horizontal body axes share one rate. Loss: pooled raw 2 s open-loop RMS
position error on the fitting flights (drag is a slowly varying force; the debiased loss removes
most of what it explains). The held-out flights and the cable flights (never fitted here) are
scored at the start and the end, raw and debiased. Every reported number is computed twice.

    python scripts/fit_drone_drag.py --output runs/calibration_2026-10-27/drone_drag_fit.json
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
from fit_drone_flights import DRONE_MASS_KG, items_for
from fit_firmware_drone import FIRMWARE, cable_items

NAMES = ('body_drag_xy_s_inv', 'body_drag_z_s_inv')


def configured(base, x, *, cable):
    c = deepcopy(base) if cable else flight_drone.drone_only(base)
    c['drone']['mass_kg'] = DRONE_MASS_KG
    c['drone']['body_drag_s_inv'] = [float(x[0]), float(x[0]), float(x[1])]
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
    parser.add_argument('--iterations', type=int, default=30)
    parser.add_argument('--start', type=float, nargs=2, default=[.4, .3])
    parser.add_argument('--step', type=float, nargs=2, default=[.2, .2])
    args = parser.parse_args()
    base = load_model_configuration(DEFAULT_MODEL)
    firmware = tomllib.loads(FIRMWARE.read_text())
    if firmware.get('law') != 'firmware':
        raise SystemExit('The firmware law is required')
    train, held = items_for('cf_2_nocable')
    cable_train, cable_held = cable_items()
    items = [i for v in train.values() for i in v]
    print(f'{len(items)} fitting windows, {sum(map(len, held.values()))} held out; cable cross-check '
          f'{sum(map(len, cable_train.values()))} + {sum(map(len, cable_held.values()))} windows', flush=True)
    evaluations = []

    def objective(x):
        if min(x) < 0 or max(x) > 5:
            return 1.
        value = flight_drone.rms(flight_drone.evaluate(configured(base, x, cable=False), firmware, items))
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
    none = report('no drag', [0., 0.])
    x, value, history = nelder_mead(objective, list(args.start), list(args.step), iterations=args.iterations)
    x = [max(0., v) for v in x]
    passes = [report('best', x) for _ in range(2)]
    reproduced = passes[0] == passes[1]
    print('reproduced twice:', reproduced, flush=True)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(dict(
        data='calibration flights 2026-10-27, setups cf_2_nocable (fit) and cf_2 (cross-check)', law='firmware',
        firmware=firmware, drone_mass_kg=DRONE_MASS_KG, parameters=list(NAMES),
        windows=dict(duration_s=2., stride_s=1., fit={k: len(v) for k, v in train.items()}, held_out={k: len(v) for k, v in held.items()},
                     cable_fit={k: len(v) for k, v in cable_train.items()}, cable_held_out={k: len(v) for k, v in cable_held.items()}),
        no_drag=dict(x=[0., 0.], **none), best=dict(x=list(map(float, x)), **passes[0], reproduced_twice=reproduced),
        search=dict(method='Nelder-Mead, raw RMS position error, drone-only flights', start=list(args.start),
                    history=history, evaluations=evaluations),
        seconds=time.time()-t0), indent=1))


if __name__ == '__main__':
    main()
