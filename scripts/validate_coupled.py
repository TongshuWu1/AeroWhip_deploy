"""Validate the coupled model (drone pipeline + cable) on the cable calibration flights.

The sent acceleration commands (sideways trims removed) drive the full native model open loop
for 2 s from the measured drone state and measured markers; nothing is fitted here. Scores:
drone tracking position (raw and debiased RMS) and cable shape (markers relative to the
drone's tracking origin, seen markers only). Every number is computed twice.

    python scripts/validate_coupled.py --drone-fit runs/.../drone_fit.json --cable-fit runs/.../cable_fit_v2.json --output runs/.../coupled_validation.json
    python scripts/validate_coupled.py --adopted --output runs/.../coupled_validation_adopted.json   # the configuration files as they are
"""
import argparse
from concurrent.futures import ThreadPoolExecutor
import json
import math
import os
from pathlib import Path
import sys

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from aerowhip.fitting import flight_drone
from aerowhip.fitting.calibration_set import folders, flight_windows, HELD_OUT
from aerowhip.fitting.flight_cable import with_cable
from aerowhip.physics.model_configuration import DEFAULT_MODEL, load_model_configuration
from aerowhip.takes.calibration_flights import load
from fit_drone_flights import DRONE_MASS_KG, configured


def coupled_config(base, drone_x, cable_values):
    config = configured(base, drone_x)
    full = with_cable(base, **cable_values)
    config['cable'] = full['cable']          # the real cable back on
    return config


def score(config, firmware, items):
    replay = flight_drone.DroneReplay(config, firmware)

    def one(item):
        try:
            return flight_drone.coupled_errors(replay, *item)
        except Exception as exc:
            return repr(exc)
    with ThreadPoolExecutor(os.cpu_count()) as pool:
        results = list(pool.map(one, items))
    failed = [r for r in results if isinstance(r, str)]
    good = [r for r in results if not isinstance(r, str)]
    drone = [d for d, _ in good]
    shape = np.concatenate([s[np.isfinite(s)] for _, s in good]) if good else np.array([np.nan])
    tip = np.concatenate([s[:, -1][np.isfinite(s[:, -1])] for _, s in good]) if good else np.array([np.nan])
    return dict(windows=len(items), failed=len(failed), failures=failed[:3],
                drone_raw_rms_m=flight_drone.rms(drone), drone_debiased_rms_m=flight_drone.rms(drone, debiased=True),
                cable_shape_rms_m=float(np.sqrt(np.mean(shape**2))), tip_shape_rms_m=float(np.sqrt(np.mean(tip**2))))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--drone-fit', type=Path)
    parser.add_argument('--cable-fit', type=Path)
    parser.add_argument('--adopted', action='store_true', help='score the configuration files as they are (default flight law)')
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    base = load_model_configuration(DEFAULT_MODEL)
    base['drone']['mass_kg'] = DRONE_MASS_KG
    firmware = flight_drone.firmware_config()
    if args.adopted:
        drone_x, cable_values = None, {k: base['cable'][k] for k in ('EI_n_m2', 'Cb_n_m2_s', 'normal_drag_coefficient', 'external_drag_s_inv')}
        models = {f'adopted configuration ({firmware.get("law", "identified")} law)': base}
    else:
        firmware = flight_drone.firmware_config(flight_drone.ROOT/'configs/controllers/crazyflie_flight.toml')   # identified law
        drone_x = json.loads(args.drone_fit.read_text())['best']['x']
        cable_values = json.loads(args.cable_fit.read_text())['best']['values']
        models = {'fitted (drone pipeline + cable v2)': coupled_config(base, drone_x, cable_values),
                  'unfitted (previous simulator gains, configured cable)': coupled_config(
                      base, [20., 1., .015, .03], {k: base['cable'][k] for k in cable_values})}
    flights = {}
    for motion, folder in folders('cf_2').items():
        flight = load(folder)
        flights[motion] = [(w, flight) for w in flight_windows(folder)]
    results = {}
    for name, config in models.items():
        rows = {}
        for motion, items in flights.items():
            first, second = score(config, firmware, items), score(config, firmware, items)
            first['reproduced_twice'] = first == second
            first['held_out'] = motion in HELD_OUT
            rows[motion] = first
            print(f'{name[:28]:28s} {motion:12s} {"HELD" if motion in HELD_OUT else "    "} drone {first["drone_debiased_rms_m"]*1e3:6.1f} mm debiased '
                  f'({first["drone_raw_rms_m"]*1e3:5.0f} raw)  cable shape {first["cable_shape_rms_m"]*1e3:5.1f} mm  tip {first["tip_shape_rms_m"]*1e3:5.1f} mm  '
                  f'failed {first["failed"]}  reproduced {first["reproduced_twice"]}', flush=True)
        results[name] = rows
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(dict(drone_x=drone_x, cable=cable_values, firmware=firmware, results=results), indent=1))


if __name__ == '__main__':
    main()
