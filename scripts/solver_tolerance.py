"""Effect of the implicit solver's stopping rule (simulation.solver_tolerance) on the simulation.

Two paths, each against a tight reference tolerance (the first of --tolerances):

    coupled   the full model (firmware controller, motors, drone, cable) replays every 2 s window of
              the chosen flights open loop with the sent commands x1 and x1.5, as step_convergence.py:
              drone and marker position differences, peak tip (c10) speed and its timing
    cable     the prescribed-clamp cable replay of the cable fit on every 1 s segment of the cf_2
              cable flights: marker differences, and the RMS marker error against the recording

Each tolerance replaces simulation.solver_tolerance; the fit setting (fit_solver_tolerance) is
removed so the replay helpers do not override it. Seconds per tolerance are wall time and depend
on the machine's load.

    python scripts/solver_tolerance.py --output runs/calibration_2026-10-27/solver_tolerance.json
"""
import argparse
from concurrent.futures import ThreadPoolExecutor
from copy import deepcopy
import json
import math
from pathlib import Path
import sys
import time

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from aerowhip.fitting import flight_cable, flight_drone
from aerowhip.fitting.calibration_set import flight_windows, folders, split
from aerowhip.physics.model import Model
from aerowhip.physics.model_configuration import DEFAULT_MODEL, load_model_configuration
from aerowhip.takes.calibration_flights import load
from step_convergence import compare, summary


def with_tolerance(config, tolerance):
    c = deepcopy(config)
    c['simulation'] = {k: v for k, v in c['simulation'].items() if k != 'fit_solver_tolerance'}
    c['simulation']['solver_tolerance'] = tolerance
    return c


def coupled(config, firmware, items, scale, threads):
    replay = flight_drone.DroneReplay(config, firmware)

    def one(item):
        try:
            return replay.run(*item, markers=True, scale=scale)
        except Exception as error:
            return str(error)
    with ThreadPoolExecutor(threads) as pool:
        return list(pool.map(one, items))


def cable(config, windows, threads):
    replay = flight_cable.CableReplay(Model(config).native)

    def one(window):
        try:
            return replay.run(window)
        except Exception as error:
            return str(error)
    with ThreadPoolExecutor(threads) as pool:
        return list(pool.map(one, windows))


def cable_compare(result, reference, windows):
    failed = sum(isinstance(a, str) for a in result)
    ok = [(a, b, w) for a, b, w in zip(result, reference, windows) if not isinstance(a, str) and not isinstance(b, str)]
    differences = np.concatenate([np.linalg.norm(a-b, axis=2).ravel() for a, b, _ in ok])
    errors = np.concatenate([np.linalg.norm(a-w.markers[1:], axis=2)[w.seen[1:]] for a, _, w in ok])
    return dict(segments=len(result), failed=failed, marker_max_m=float(differences.max()),
                marker_median_m=float(np.median(differences)), rms_error_m=float(np.sqrt(np.mean(errors**2))))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--flights', nargs='+', default=['whip/002', 'strokes/002'])
    parser.add_argument('--tolerances', type=float, nargs='+', default=[1e-13, 2e-11, 1e-7, 1e-5, 1e-4],
                        help='the first is the reference')
    parser.add_argument('--scales', type=float, nargs='+', default=[1., 1.5])
    parser.add_argument('--threads', type=int, default=6)
    args = parser.parse_args()
    base = load_model_configuration(DEFAULT_MODEL)
    firmware = flight_drone.firmware_config()
    found = folders('cf_2')
    reference_tolerance = args.tolerances[0]
    out = dict(model=str(DEFAULT_MODEL), tolerances=args.tolerances, reference_tolerance=reference_tolerance,
               coupled={}, cable={})
    t0 = time.time()
    for motion in args.flights:
        flight = load(found[motion])
        items = [(w, flight) for w in flight_windows(found[motion])]
        for scale in args.scales:
            key = f'{motion} x{scale:g}'
            runs, seconds = {}, {}
            for tolerance in args.tolerances:
                start = time.time()
                runs[tolerance] = coupled(with_tolerance(base, tolerance), firmware, items, scale, args.threads)
                seconds[tolerance] = time.time()-start
            reference = runs[reference_tolerance]
            out['coupled'][key] = dict(windows=len(items), seconds={str(t): s for t, s in seconds.items()},
                                       tolerances={str(t): summary(compare(runs[t], reference)) for t in args.tolerances[1:]})
            for t in args.tolerances[1:]:
                m = out['coupled'][key]['tolerances'][str(t)]
                print(f"{key:18s} {t:.0e}: drone max {m['drone_max_m']*1e3:.4f} mm, marker max {m['marker_max_m']*1e3:.4f} mm, "
                      f"peak tip speed {m['peak_tip_speed_relative_max']*100:.4f} %, shift <= {m['peak_time_shift_frames_max']} frames, "
                      f"failed {m['failed']}; {seconds[t]:.1f} s vs {seconds[reference_tolerance]:.1f} s", flush=True)
    train, held = split('cf_2', duration_s=1., stride_s=1.)
    windows = [w for group in (train, held) for v in group.values() for w in v]
    runs, seconds = {}, {}
    for tolerance in args.tolerances:
        start = time.time()
        runs[tolerance] = cable(with_tolerance(base, tolerance), windows, args.threads)
        seconds[tolerance] = time.time()-start
    reference = runs[reference_tolerance]
    out['cable'] = dict(segments=len(windows), seconds={str(t): s for t, s in seconds.items()},
                        tolerances={str(t): cable_compare(runs[t], reference, windows) for t in args.tolerances})
    for t in args.tolerances:
        m = out['cable']['tolerances'][str(t)]
        print(f"cable segments     {t:.0e}: marker max {m['marker_max_m']*1e3:.4f} mm (median {m['marker_median_m']*1e3:.6f}), "
              f"RMS error {m['rms_error_m']*1e3:.3f} mm, failed {m['failed']} of {m['segments']}; {seconds[t]:.1f} s", flush=True)
    out['seconds'] = time.time()-t0
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(out, indent=1))
    print(f'written to {args.output} ({math.ceil(out["seconds"])} s)')


if __name__ == '__main__':
    main()
