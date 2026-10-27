"""Where the coupled cable error comes from: the cable model alone or the drone model.

For every 2 s window of the cable flights, two replays of the adopted model:
  cable alone   the measured drone pose drives the clamp (aerowhip.fitting.flight_cable); the
                error is the cable model's own.
  coupled       the sent commands drive the drone and the cable (validate_coupled.py); the error
                adds the drone's.
Both are scored as marker positions relative to the drone's tracking origin (cable shape), seen
markers only. Reported per flight: pooled RMS, per marker, by time into the window, and by the
measured marker speed. Every number is computed twice.

    python scripts/cable_error_split.py --output runs/calibration_2026-10-27/cable_error_split.json
"""
import argparse
from concurrent.futures import ThreadPoolExecutor
import json
from pathlib import Path
import sys
import time

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from aerowhip.fitting import flight_cable, flight_drone
from aerowhip.fitting.calibration_set import HELD_OUT, flight_windows, folders
from aerowhip.physics.model import Model
from aerowhip.physics.model_configuration import DEFAULT_MODEL, load_model_configuration
from aerowhip.takes.calibration_flights import load

SPEED_BINS = (0., 1., 2., 3., 4., 6.)
TIME_BINS = (0., .5, 1., 1.5, 2.)


def shape_errors(config, firmware, items):
    """(cable alone, coupled) per window: (N, 10) distances, NaN where unseen or failed."""
    native = Model(config, load_weights=False).native
    cable, replay = flight_cable.CableReplay(native), flight_drone.DroneReplay(config, firmware)

    def one(item):
        window, flight = item
        try:
            alone = cable.errors(window)
        except Exception:
            alone = None
        try:
            _, coupled = flight_drone.coupled_errors(replay, window, flight)
        except Exception:
            coupled = None
        return alone, coupled
    with ThreadPoolExecutor() as pool:
        return list(pool.map(one, items))


def pooled(values):
    values = np.concatenate([v[np.isfinite(v)] for v in values]) if values else np.array([])
    return float(np.sqrt(np.mean(values**2))) if len(values) else None


def breakdown(errors, windows):
    """Pooled RMS overall, per marker, by time and by measured marker speed."""
    ok = [(e, w) for e, w in zip(errors, windows) if e is not None]
    out = dict(windows=len(errors), failed=len(errors)-len(ok), rms_m=pooled([e for e, _ in ok]),
               per_marker_rms_m=[pooled([e[:, k] for e, _ in ok]) for k in range(10)])
    t = (np.arange(len(ok[0][0]))+1)*.01 if ok else np.array([])
    out['by_time_rms_m'] = {f'{a:g}-{b:g} s': pooled([e[(t > a) & (t <= b)] for e, _ in ok]) for a, b in zip(TIME_BINS[:-1], TIME_BINS[1:])}
    rows = {}
    for a, b in zip(SPEED_BINS, SPEED_BINS[1:]+(np.inf,)):
        parts = []
        for e, w in ok:
            speed = np.linalg.norm(np.gradient(w.markers, .01, axis=0), axis=2)[1:]
            parts.append(np.where((speed >= a) & (speed < b), e, np.nan))
        rows[f'{a:g}-{b:g} m/s'] = dict(rms_m=pooled(parts), samples=int(sum(np.isfinite(p).sum() for p in parts)))
    out['by_marker_speed'] = rows
    return out


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    config = load_model_configuration(DEFAULT_MODEL)
    firmware = flight_drone.firmware_config()
    results = {}
    t0 = time.time()
    for motion, folder in folders('cf_2').items():
        flight = load(folder)
        windows = flight_windows(folder)
        items = [(w, flight) for w in windows]
        passes = [shape_errors(config, firmware, items) for _ in range(2)]
        same = all((a is None and b is None) or (a is not None and b is not None and np.array_equal(a, b, equal_nan=True))
                   for p, q in zip(*passes) for a, b in zip(p, q))
        if not same:
            raise RuntimeError(f'{motion} did not reproduce')
        alone, coupled = [p[0] for p in passes[0]], [p[1] for p in passes[0]]
        worst = sorted(((float(np.sqrt(np.nanmean(c**2))), float(np.sqrt(np.nanmean(a**2))), w.phase, float(w.start_stamp))
                        for a, c, w in zip(alone, coupled, windows) if a is not None and c is not None), reverse=True)[:5]
        results[motion] = dict(held_out=motion in HELD_OUT, cable_alone=breakdown(alone, windows), coupled=breakdown(coupled, windows),
                               worst_windows=[dict(coupled_rms_m=c, cable_alone_rms_m=a, phase=p, start_stamp=s) for c, a, p, s in worst])
        r = results[motion]
        print(f"{motion:12s}{' HELD' if motion in HELD_OUT else '     '} cable alone {r['cable_alone']['rms_m']*1e3:5.1f} mm  coupled {r['coupled']['rms_m']*1e3:5.1f} mm"
              f"  | tip alone {r['cable_alone']['per_marker_rms_m'][-1]*1e3:5.1f} coupled {r['coupled']['per_marker_rms_m'][-1]*1e3:5.1f}"
              f"  | alone by time {[round(v*1e3, 1) for v in r['cable_alone']['by_time_rms_m'].values()]}", flush=True)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(dict(model=str(DEFAULT_MODEL), reproduced_twice=True, flights=results, seconds=time.time()-t0), indent=1))


if __name__ == '__main__':
    main()
