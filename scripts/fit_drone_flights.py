"""Fit the drone's flight pipeline dynamics on the drone-only calibration flights.

The sent acceleration commands (sideways trims removed) drive the native FlightController
and plant open loop for 2 s from the measured state. Nelder-Mead on the attitude loop
(frequency, damping ratio), the motor response time and the command delay minimizes the
debiased RMS position error (per-window constant acceleration bias removed); the thrust map
is fixed from its direct fit. Held-out flights are scored only at the start and the end;
every reported loss is computed twice.

    python scripts/fit_drone_flights.py --output runs/calibration_2026-10-27/drone_fit.json
"""
import argparse
from copy import deepcopy
import json
import math
from pathlib import Path
import sys
import time

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from aerowhip.fitting import flight_drone
from aerowhip.paths import ROOT
from aerowhip.fitting.calibration_set import folders, flight_windows, is_held_out, manifest
from aerowhip.physics.model_configuration import DEFAULT_MODEL, load_model_configuration
from aerowhip.takes.calibration_flights import load
sys.path.insert(0, str(Path(__file__).resolve().parent))
from fit_cable_flights import nelder_mead

DRONE_MASS_KG = manifest().measurement('drone_mass_kg')     # from the calibration manifest (weighed, battery and clamp)


def configured(base, x):
    frequency, damping, response, delay = x
    c = flight_drone.drone_only(base)
    c['drone']['mass_kg'] = DRONE_MASS_KG
    c['controller'] = dict(c['controller'])
    c['controller']['attitude_frequency_rad_s'] = [float(frequency), float(frequency), c['controller']['attitude_frequency_rad_s'][2]]
    c['controller']['attitude_damping_ratio'] = [float(damping), float(damping), c['controller']['attitude_damping_ratio'][2]]
    c['controller']['command_delay_s'] = float(delay)
    c['actuator'] = deepcopy(c['actuator'])
    c['actuator']['response_time_s'] = [float(response)]*4
    return c


def items_for(setup):
    train, held = {}, {}
    for motion, folder in folders(setup).items():
        flight = load(folder)
        flight.markers = None
        (held if is_held_out(folder) else train)[motion] = [(w, flight) for w in flight_windows(folder, cable=False)]
    return train, held


def score(base, firmware, groups, x):
    config = configured(base, x)
    per = {m: flight_drone.evaluate(config, firmware, items) for m, items in groups.items()}
    every = [e for v in per.values() for e in v]
    return dict(raw_rms_m=flight_drone.rms(every), debiased_rms_m=flight_drone.rms(every, debiased=True),
                per_flight_debiased_rms_m={m: flight_drone.rms(v, debiased=True) for m, v in per.items()},
                per_flight_raw_rms_m={m: flight_drone.rms(v) for m, v in per.items()})


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--iterations', type=int, default=50)
    args = parser.parse_args()
    base = load_model_configuration(DEFAULT_MODEL)
    firmware = flight_drone.firmware_config(ROOT/'configs/controllers/crazyflie_flight.toml')   # identified law
    train, held = items_for('cf_2_nocable')
    items = [i for v in train.values() for i in v]
    print(f'{len(items)} fitting windows, {sum(map(len, held.values()))} held out', flush=True)
    references = {'component fits (attitude 20.5 rad/s, 1.38; motors 35 ms; delay 3 ms)': [20.5, 1.38, .035, .003],
                  'previous simulator (20 rad/s, 1.0; motors 15 ms; delay 30 ms)': [20., 1., .015, .03]}
    evaluations = []

    def objective(x):
        if x[0] <= 2 or x[1] <= .05 or x[2] < 0 or x[3] < 0 or x[3] > .2:
            return 1.
        value = flight_drone.rms(flight_drone.evaluate(configured(base, x), firmware, items), debiased=True)
        value = 1. if not math.isfinite(value) else value
        evaluations.append((list(map(float, x)), value))
        return value
    t0 = time.time()
    reference_scores = {name: dict(x=x, fit=score(base, firmware, train, x), held_out=score(base, firmware, held, x))
                        for name, x in references.items()}
    for name, r in reference_scores.items():
        print(f'{name}: fit {r["fit"]["debiased_rms_m"]*1e3:.1f} mm debiased ({r["fit"]["raw_rms_m"]*1e3:.0f} raw), '
              f'held out {r["held_out"]["debiased_rms_m"]*1e3:.1f} mm', flush=True)
    x, value, history = nelder_mead(objective, [20.5, 1.38, .035, .003], [4., .3, .015, .02], iterations=args.iterations)
    passes = [(score(base, firmware, train, x), score(base, firmware, held, x)) for _ in range(2)]
    fit, out = passes[0]
    print(f'best x {np.round(x, 4).tolist()}: fit {fit["debiased_rms_m"]*1e3:.1f} mm debiased ({fit["raw_rms_m"]*1e3:.0f} raw), '
          f'held out {out["debiased_rms_m"]*1e3:.1f} mm ({out["raw_rms_m"]*1e3:.0f} raw); reproduced {passes[0] == passes[1]}', flush=True)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(dict(
        data='calibration flights 2026-10-27, setup cf_2_nocable', firmware=firmware, drone_mass_kg=DRONE_MASS_KG,
        parameters=['attitude_frequency_rad_s (x, y)', 'attitude_damping_ratio (x, y)', 'motor response_time_s', 'command_delay_s'],
        windows=dict(duration_s=2., stride_s=1., fit={k: len(v) for k, v in train.items()}, held_out={k: len(v) for k, v in held.items()}),
        references=reference_scores, best=dict(x=list(map(float, x)), fit=fit, held_out=out, reproduced_twice=passes[0] == passes[1]),
        search=dict(method='Nelder-Mead, debiased RMS position error', history=history, evaluations=evaluations),
        seconds=time.time()-t0), indent=1))


if __name__ == '__main__':
    main()
