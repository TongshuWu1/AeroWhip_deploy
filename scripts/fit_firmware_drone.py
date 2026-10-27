"""Stage 2 of the identification: physical parameters of the firmware-law drone on the drone-only flights.

The flight law is the ported Mellinger controller (configs/controllers/crazyflie_firmware.toml); its
gains are firmware constants. What is identified here is the vehicle around it: the roll/pitch
inertia scale (about the CAD box estimate), the motor arm scale (torque per motor-force difference,
also absorbing the motor-map slope), the motor response time and the command delay. Loss: pooled
debiased 2 s open-loop RMS position error on the fitting flights; the held-out flights are scored
only at the start and the end, and the cable flights (never fitted here) are scored coupled with the
fitted cable, raw and debiased, as the cross-check. Every reported number is computed twice.

    python scripts/fit_firmware_drone.py --output runs/calibration_2026-10-27/firmware_drone_fit.json
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
from aerowhip.fitting.calibration_set import folders, flight_windows, HELD_OUT
from aerowhip.paths import ROOT
from aerowhip.physics.model_configuration import DEFAULT_MODEL, load_model_configuration
from aerowhip.takes.calibration_flights import load
sys.path.insert(0, str(Path(__file__).resolve().parent))
from fit_cable_flights import nelder_mead
from fit_drone_flights import DRONE_MASS_KG, items_for

FIRMWARE = ROOT/'configs/controllers/crazyflie_firmware.toml'
NAMES = ('inertia_scale_xy', 'arm_scale', 'motor_response_s', 'command_delay_s')


def configured(base, x, *, cable):
    inertia, arm, response, delay = x
    c = deepcopy(base) if cable else flight_drone.drone_only(base)
    c['drone']['mass_kg'] = DRONE_MASS_KG
    J = c['drone']['inertia_body_about_com_kg_m2']
    J[0][0] *= float(inertia); J[1][1] *= float(inertia)
    c['controller'] = dict(c['controller'], command_delay_s=float(delay))
    c['actuator'] = deepcopy(c['actuator'])
    c['actuator']['motor_positions_com_m'] = [[p[0]*float(arm), p[1]*float(arm), p[2]] for p in c['actuator']['motor_positions_com_m']]
    c['actuator']['response_time_s'] = [float(response)]*4
    return c


def score(base, firmware, groups, x, *, cable=False):
    config = configured(base, x, cable=cable)
    per = {m: flight_drone.evaluate(config, firmware, items) for m, items in groups.items()}
    every = [e for v in per.values() for e in v]
    return dict(raw_rms_m=flight_drone.rms(every), debiased_rms_m=flight_drone.rms(every, debiased=True),
                per_flight_debiased_rms_m={m: flight_drone.rms(v, debiased=True) for m, v in per.items()},
                per_flight_raw_rms_m={m: flight_drone.rms(v) for m, v in per.items()})


def cable_items():
    train, held = {}, {}
    for motion, folder in folders('cf_2').items():
        flight = load(folder)
        (held if motion in HELD_OUT else train)[motion] = [(w, flight) for w in flight_windows(folder)]
    return train, held


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--iterations', type=int, default=40)
    parser.add_argument('--start', type=float, nargs=4, metavar=tuple(n.upper() for n in NAMES), default=[1., 1., .015, .03])
    parser.add_argument('--step', type=float, nargs=4, default=[.5, .3, .01, .015])
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
    x0 = list(args.start)
    evaluations = []

    def objective(x):
        if x[0] <= .1 or x[1] <= .1 or x[2] < 0 or x[3] > .2:
            return 1.
        x = [x[0], x[1], x[2], max(0., x[3])]      # a negative delay is zero delay, not a penalty
        value = flight_drone.rms(flight_drone.evaluate(configured(base, x, cable=False), firmware, items), debiased=True)
        value = 1. if not math.isfinite(value) else value
        evaluations.append((list(map(float, x)), value))
        return value

    def report(label, x):
        rows = dict(fit=score(base, firmware, train, x), held_out=score(base, firmware, held, x),
                    cable_fit=score(base, firmware, cable_train, x, cable=True), cable_held_out=score(base, firmware, cable_held, x, cable=True))
        print(f'{label} x {np.round(x, 4).tolist()}: drone-only fit {rows["fit"]["debiased_rms_m"]*1e3:.1f} mm debiased '
              f'({rows["fit"]["raw_rms_m"]*1e3:.0f} raw), held out {rows["held_out"]["debiased_rms_m"]*1e3:.1f} ({rows["held_out"]["raw_rms_m"]*1e3:.0f}); '
              f'cable {rows["cable_fit"]["debiased_rms_m"]*1e3:.1f} ({rows["cable_fit"]["raw_rms_m"]*1e3:.0f}), '
              f'cable held out {rows["cable_held_out"]["debiased_rms_m"]*1e3:.1f} ({rows["cable_held_out"]["raw_rms_m"]*1e3:.0f})', flush=True)
        return rows
    t0 = time.time()
    start = report('start', x0)
    x, value, history = nelder_mead(objective, x0, args.step, iterations=args.iterations)
    x = [x[0], x[1], x[2], max(0., x[3])]
    passes = [report('best', x) for _ in range(2)]
    reproduced = passes[0] == passes[1]
    print('reproduced twice:', reproduced, flush=True)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(dict(
        data='calibration flights 2026-10-27, setups cf_2_nocable (fit) and cf_2 (cross-check)', law='firmware',
        firmware=firmware, drone_mass_kg=DRONE_MASS_KG, parameters=list(NAMES),
        windows=dict(duration_s=2., stride_s=1., fit={k: len(v) for k, v in train.items()}, held_out={k: len(v) for k, v in held.items()},
                     cable_fit={k: len(v) for k, v in cable_train.items()}, cable_held_out={k: len(v) for k, v in cable_held.items()}),
        start=dict(x=x0, **start), best=dict(x=list(map(float, x)), **passes[0], reproduced_twice=reproduced),
        search=dict(method='Nelder-Mead, debiased RMS position error, drone-only flights', history=history, evaluations=evaluations),
        seconds=time.time()-t0), indent=1))


if __name__ == '__main__':
    main()
