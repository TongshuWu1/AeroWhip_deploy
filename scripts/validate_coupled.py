"""Stage 5, coupled validation: the configured model (drone driven by the sent commands, carrying the
cable) on 1 s segments of every cable flight, from the measured drone state and markers. Nothing is
fitted. Reported per flight and pooled, separately for the fit and the held-out flights: drone
position (one constant acceleration bias per flight removed), drone tilt (one constant offset per
flight removed), cable shape (markers relative to the drone) and tip (last marker). Uses
`aerowhip.fitting.coupled`, the same residuals as the coupled sensitivity and refinement.

    python scripts/validate_coupled.py --output runs/fits/coupled/<time>.json
"""
import argparse
import json
import math
from pathlib import Path
import sys
import time
import tomllib

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from aerowhip.fitting import coupled
from aerowhip.fitting.flight_drone import FIRMWARE
from aerowhip.fitting.progress import Progress
from aerowhip.physics.model_configuration import DEFAULT_MODEL, load_model_configuration


def score(config, firmware, groups, values):
    """Pooled and per-flight errors of one group of flights (None without flights)."""
    groups = {n: v for n, v in groups.items() if v}
    if not groups:
        return None
    residuals = coupled.Residuals(config, firmware, groups)
    passes = [residuals(**values) for _ in range(2)]
    (r, failures, report), (r2, _, _) = passes
    residuals.runner.close()
    return dict(position_m=report['position_m'], tilt_deg=report['tilt_deg'], shape_m=report['shape_m'], tip_m=report['tip_m'],
                failed_segments=failures, reproduced_twice=bool((r == r2).all()), segments={n: len(v) for n, v in groups.items()},
                per_flight={n: dict(position_m=report['per_flight_position_m'][n], tilt_deg=report['per_flight_tilt_deg'][n],
                                    shape_m=report['per_flight_shape_m'][n], tip_m=report['per_flight_tip_m'][n]) for n in groups})


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--horizon', type=float, default=1., help='segment length, s')
    args = parser.parse_args()
    t0 = time.time()
    config = load_model_configuration(DEFAULT_MODEL)
    firmware = tomllib.loads(FIRMWARE.read_text())
    values = coupled.current(config)
    progress = Progress(args.output, stage='coupled', parameters=[], x0=[], unit='m')
    progress.plan(['loading flights', 'fit flights', 'held-out flights'], [3, 2, 2])
    progress.phase(0)
    train, held = coupled.segments(args.horizon)
    progress.phase_done(math.nan)
    results = {}
    for index, (role, groups) in enumerate((('fit', train), ('held_out', held)), start=1):
        progress.phase(index)
        results[role] = score(config, firmware, groups, values)
        if results[role] is None:
            continue
        for name, v in results[role]['per_flight'].items():
            progress.write(kind='flight', name=name, role=role, **v)
            print(f'{role:8s} {name:20s} position {v["position_m"]*1e3:6.1f} mm  tilt {v["tilt_deg"]:5.2f} deg  '
                  f'shape {v["shape_m"]*1e3:6.1f} mm  tip {v["tip_m"]*1e3:6.1f} mm', flush=True)
        s = results[role]
        print(f'{role:8s} pooled               position {s["position_m"]*1e3:6.1f} mm  tilt {s["tilt_deg"]:5.2f} deg  '
              f'shape {s["shape_m"]*1e3:6.1f} mm  tip {s["tip_m"]*1e3:6.1f} mm  failed {s["failed_segments"]}  '
              f'reproduced {s["reproduced_twice"]}', flush=True)
        progress.phase_done(s['shape_m'])
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(dict(
        stage='coupled', method='segments', model='configured model, full coupled replay from the sent commands',
        horizon_s=args.horizon, values=values, firmware=firmware, results=results, seconds=time.time()-t0),
        indent=1, default=float))
    progress.done([], math.nan)


if __name__ == '__main__':
    main()
