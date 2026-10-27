"""Physics step convergence of the coupled model on the cable flights.

The adopted model replays every 2 s window of the chosen flights open loop (sent commands, as in
validate_coupled.py) at several maximum physics steps; the smallest is the reference. Reported per
step against the reference: drone position and marker position differences, and the peak tip
(c10) speed of each window with its timing. `--scale` multiplies the policy part of the commands
to drive the cable harder than the recordings did (stress test). Every number is computed twice.

    python scripts/step_convergence.py --output runs/calibration_2026-10-27/step_convergence.json
"""
import argparse
from concurrent.futures import ThreadPoolExecutor
from copy import deepcopy
import json
from pathlib import Path
import sys
import time
import tomllib

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from aerowhip.fitting import flight_drone
from aerowhip.fitting.calibration_set import is_held_out, flight_windows, folders
from aerowhip.physics.model_configuration import DEFAULT_MODEL, load_model_configuration
from aerowhip.takes.calibration_flights import load


def replays(config, firmware, items, step, scale):
    c = deepcopy(config)
    c['simulation'] = dict(c['simulation'], physics_timestep_s=step)
    replay = flight_drone.DroneReplay(c, firmware)

    def one(item):
        try:
            return replay.run(*item, markers=True, scale=scale)
        except Exception as error:
            return str(error)
    with ThreadPoolExecutor() as pool:
        return list(pool.map(one, items))


def compare(result, reference):
    rows = []
    for a, b in zip(result, reference):
        if isinstance(a, str) or isinstance(b, str):
            rows.append(dict(failed=a if isinstance(a, str) else 'reference failed'))
            continue
        speed_a, speed_b = np.linalg.norm(a[2][:, -1], axis=1), np.linalg.norm(b[2][:, -1], axis=1)
        rows.append(dict(drone_max_m=float(np.abs(a[0]-b[0]).max()),
                         marker_max_m=float(np.linalg.norm(a[1]-b[1], axis=2).max()),
                         marker_median_m=float(np.median(np.linalg.norm(a[1]-b[1], axis=2))),
                         peak_tip_speed_mps=float(speed_a.max()), reference_peak_tip_speed_mps=float(speed_b.max()),
                         peak_tip_speed_relative=float(speed_a.max()/speed_b.max()-1),
                         peak_time_shift_frames=int(speed_a.argmax()-speed_b.argmax()),
                         tip_speed_max_difference_mps=float(np.abs(speed_a-speed_b).max())))
    return rows


def summary(rows):
    ok = [r for r in rows if 'failed' not in r]
    if not ok:
        return dict(windows=len(rows), failed=len(rows))
    fastest = sorted(ok, key=lambda r: -r['reference_peak_tip_speed_mps'])[:5]
    return dict(windows=len(rows), failed=len(rows)-len(ok),
                drone_max_m=max(r['drone_max_m'] for r in ok), marker_max_m=max(r['marker_max_m'] for r in ok),
                marker_median_m=float(np.median([r['marker_median_m'] for r in ok])),
                peak_tip_speed_relative_max=max(abs(r['peak_tip_speed_relative']) for r in ok),
                peak_time_shift_frames_max=max(abs(r['peak_time_shift_frames']) for r in ok),
                tip_speed_max_difference_mps=max(r['tip_speed_max_difference_mps'] for r in ok),
                fastest_reference_peaks_mps=[r['reference_peak_tip_speed_mps'] for r in fastest],
                fastest_peak_relative=[r['peak_tip_speed_relative'] for r in fastest])


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--flights', nargs='+', default=['whip', 'strokes'])
    parser.add_argument('--steps', type=float, nargs='+', default=[.002, .001, .0005], help='the last is the reference')
    parser.add_argument('--scales', type=float, nargs='+', default=[1., 1.5])
    args = parser.parse_args()
    config = load_model_configuration(DEFAULT_MODEL)
    firmware = flight_drone.firmware_config()
    found = folders('cf_2')
    out = dict(model=str(DEFAULT_MODEL), steps_s=args.steps, reference_step_s=args.steps[-1], results={})
    t0 = time.time()
    for motion in args.flights:
        folder = found[motion]
        flight = load(folder)
        items = [(w, flight) for w in flight_windows(folder)]
        for scale in args.scales:
            key = f'{motion} x{scale:g}'
            costs, runs = {}, {}
            for step in args.steps:
                start = time.time()
                first, second = replays(config, firmware, items, step, scale), replays(config, firmware, items, step, scale)
                if any(not isinstance(a, str) and not (np.array_equal(a[0], b[0]) and np.array_equal(a[1], b[1]))
                       for a, b in zip(first, second)):
                    raise RuntimeError(f'{key} at {step} s did not reproduce')
                runs[step], costs[step] = first, (time.time()-start)/2
            reference = runs[args.steps[-1]]
            out['results'][key] = dict(held_out=is_held_out(folder), windows=len(items), seconds=costs,
                                       steps={str(s): dict(summary=summary(rows := compare(runs[s], reference)), windows=rows)
                                              for s in args.steps[:-1]})
            for s in args.steps[:-1]:
                m = out['results'][key]['steps'][str(s)]['summary']
                if 'drone_max_m' not in m:
                    print(f'{key:14s} step {s*1e3:.1f} ms: all failed', flush=True); continue
                print(f"{key:14s} step {s*1e3:.1f} ms vs {args.steps[-1]*1e3:.1f}: drone max {m['drone_max_m']*1e3:.2f} mm, "
                      f"marker max {m['marker_max_m']*1e3:.2f} mm (median {m['marker_median_m']*1e3:.2f}), peak tip speed "
                      f"max {m['peak_tip_speed_relative_max']*100:.2f} %, shift <= {m['peak_time_shift_frames_max']} frames, "
                      f"fastest peaks {np.round(m['fastest_reference_peaks_mps'], 2).tolist()} m/s; failed {m['failed']}; "
                      f"cost {costs[s]:.1f} s vs {costs[args.steps[-1]]:.1f} s", flush=True)
    out['seconds'] = time.time()-t0
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(out, indent=1))


if __name__ == '__main__':
    main()
