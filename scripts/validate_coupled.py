"""Stage 5, coupled validation: the configured model (drone driven by the sent commands, carrying the
cable) replayed from the measured drone state and markers, nothing fitted. Separately for the fit
and the held-out flights.

1. Segments (--horizon, 1 s) of every cable flight, as `aerowhip.fitting.coupled` scores them: drone
   position (one constant acceleration bias per flight removed), drone tilt (one constant offset per
   flight removed), cable shape (markers relative to the drone) and tip (last marker).
2. The same segments replayed again with each flight's bias from 1 cancelled in the flight controller
   (`acceleration_bias`): the bias drags the clamp, so without this the cable numbers carry the
   drone's bias as well as the cable's own error.
3. The stroke test, bias-cancelled as in 2: 0.6 s windows from the onset of every stroke of the
   decay, strokes and whip flights (`scripts/stroke_diagnostics.py`). Per stroke the tip's peak
   speed against the measured one (ratio, frame shift), the tip error over the stroke, the drone's
   position error and its tilt error (per-flight offset removed). This is the yardstick of the
   fitting design (internal/archive/fitting_and_adaptation_design.md).

The per-flight bias is the flight's own, estimated from its own segments, as a trim measured in hover
would be in flight.

    python scripts/validate_coupled.py --output runs/fits/coupled/<time>.json
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
sys.path.insert(0, str(Path(__file__).resolve().parent))
from aerowhip.fitting import coupled, flight_drone
from aerowhip.fitting.calibration_set import is_held_out, default_setup
from aerowhip.fitting.least_squares import ProcessRunner
from aerowhip.fitting.progress import Progress
from aerowhip.physics.model_configuration import DEFAULT_MODEL, load_model_configuration
from aerowhip.takes.calibration_flights import DT
import stroke_diagnostics


def score(config, firmware, groups, values, biases=None):
    """Pooled and per-flight errors of one group of flights (None without flights), and the per-flight
    bias {name: b} the scoring removed."""
    groups = {n: v for n, v in groups.items() if v}
    if not groups:
        return None, {}
    residuals = coupled.Residuals(config, firmware, groups)
    residuals.biases = biases or {}
    passes = [residuals(**values) for _ in range(2)]
    (r, failures, report), (r2, _, _) = passes
    residuals.runner.close()
    result = dict(position_m=report['position_m'], tilt_deg=report['tilt_deg'], shape_m=report['shape_m'], tip_m=report['tip_m'],
                  failed_segments=failures, reproduced_twice=bool((r == r2).all()), segments={n: len(v) for n, v in groups.items()},
                  per_flight={n: dict(position_m=report['per_flight_position_m'][n], tilt_deg=report['per_flight_tilt_deg'][n],
                                      shape_m=report['per_flight_shape_m'][n], tip_m=report['per_flight_tip_m'][n],
                                      bias_m_s2=report['bias_m_s2'][n]) for n in groups})
    return result, report['bias_m_s2']


def _stroke_replay(parameters):
    config, firmware, biases = parameters
    replay = flight_drone.DroneReplay(config, firmware)

    def one(item):
        w, flight = item
        p, R, markers, rates = replay.run(w, flight, pipeline=True, markers=True, attitude=True, bias=biases.get(str(flight.folder)))
        tip = markers[:, -1]-w.markers[1:, -1]
        return dict(position=p-w.position[1:], tilt=flight_drone.tilt_error(R, w.rotation[1:]),
                    tip=np.where(w.seen[1:, -1:], tip, np.nan), tip_speed=np.linalg.norm(rates[:, -1], axis=1))
    return one


def stroke_test(config, firmware, biases, horizon):
    """Per-stroke metrics on the decay, strokes and whip flights, summarised per role."""
    groups = {n: v for n, v in stroke_diagnostics.stroke_windows(default_setup(), horizon).items() if v}
    items = [item for v in groups.values() for item in v]
    runner = ProcessRunner(items, _stroke_replay)
    results = [r for r, _ in runner((config, firmware, biases))]
    runner.close()
    strokes, k = [], 0
    for name, v in groups.items():
        rs = results[k:k+len(v)]
        k += len(v)
        good = [r for r in rs if r is not None]
        offset = np.concatenate([r['tilt'] for r in good]).mean(0) if good else np.zeros(2)
        for (w, flight), r in zip(v, rs):
            if r is None:
                strokes.append(dict(flight=name, phase=w.phase, failed=True))
                continue
            measured = stroke_diagnostics.measured_tip_speed(w)
            row = dict(flight=name, phase=w.phase, role='held_out' if is_held_out(flight.folder) else 'fit', failed=False,
                       tip_path_m=float(np.sqrt(np.nanmean(np.sum(r['tip']**2, 1)))) if np.isfinite(r['tip']).any() else math.nan,
                       position_m=float(np.sqrt(np.mean(np.sum(r['position']**2, 1)))),
                       tilt_deg=math.degrees(float(np.sqrt(np.mean(np.sum((r['tilt']-offset)**2, 1))))))
            if np.isfinite(measured).sum() > len(measured)//2:
                im, isim = int(np.nanargmax(measured)), int(np.argmax(r['tip_speed']))
                row.update(measured_peak_m_s=float(measured[im]), simulated_peak_m_s=float(r['tip_speed'][isim]),
                           peak_ratio=float(r['tip_speed'][isim]/measured[im]), peak_shift_frames=isim-im)
            strokes.append(row)
    summary = {}
    for role in ('fit', 'held_out'):
        rows = [s for s in strokes if not s['failed'] and s['role'] == role]
        if not rows:
            continue
        ratio = [s['peak_ratio'] for s in rows if 'peak_ratio' in s]
        summary[role] = dict(strokes=len(rows), peak_ratio_median=float(np.median(ratio)) if ratio else math.nan,
                             peak_ratio_iqr=[float(np.percentile(ratio, 25)), float(np.percentile(ratio, 75))] if ratio else None,
                             peak_shift_median_frames=float(np.median([s['peak_shift_frames'] for s in rows if 'peak_shift_frames' in s])) if ratio else math.nan,
                             tip_path_m=float(np.sqrt(np.nanmean([s['tip_path_m']**2 for s in rows]))),
                             position_m=float(np.sqrt(np.mean([s['position_m']**2 for s in rows]))),
                             tilt_deg=float(np.sqrt(np.mean([s['tilt_deg']**2 for s in rows]))))
    return dict(horizon_s=horizon, summary=summary, strokes=strokes, failed=sum(s['failed'] for s in strokes))


def print_segments(label, s):
    print(f'{label:34s} position {s["position_m"]*1e3:6.1f} mm  tilt {s["tilt_deg"]:5.2f} deg  shape {s["shape_m"]*1e3:6.1f} mm  '
          f'tip {s["tip_m"]*1e3:6.1f} mm  failed {s["failed_segments"]}  reproduced {s["reproduced_twice"]}', flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--horizon', type=float, default=1., help='segment length, s')
    parser.add_argument('--stroke-horizon', type=float, default=.6, help='stroke window length, s')
    args = parser.parse_args()
    t0 = time.time()
    config = load_model_configuration(DEFAULT_MODEL)
    firmware = tomllib.loads(flight_drone.FIRMWARE.read_text())
    values = coupled.current(config)
    progress = Progress(args.output, stage='coupled', parameters=[], x0=[], unit='m')
    progress.plan(['loading flights', 'segments', 'segments, bias cancelled', 'stroke test'], [3, 4, 4, 2])
    progress.phase(0)
    train, held = coupled.segments(args.horizon)
    folder_of = {n: str(v[0][1].folder) for g in (train, held) for n, v in g.items() if v}
    progress.phase_done(math.nan)
    results, corrected, biases = {}, {}, {}
    progress.phase(1)
    for role, groups in (('fit', train), ('held_out', held)):
        results[role], b = score(config, firmware, groups, values)
        biases.update({folder_of[n]: (-np.asarray(v)).tolist() for n, v in b.items()})
        if results[role] is not None:
            for name, v in results[role]['per_flight'].items():
                progress.write(kind='flight', name=name, role=role, **{k: x for k, x in v.items() if k != 'bias_m_s2'})
            print_segments(f'{role} segments', results[role])
    progress.phase_done(results['fit']['shape_m'])
    progress.phase(2)
    for role, groups in (('fit', train), ('held_out', held)):
        corrected[role], _ = score(config, firmware, groups, values, biases)
        if corrected[role] is not None:
            print_segments(f'{role} segments, bias cancelled', corrected[role])
    progress.phase_done(corrected['fit']['shape_m'])
    progress.phase(3)
    strokes = stroke_test(config, firmware, biases, args.stroke_horizon)
    for role, s in strokes['summary'].items():
        print(f'{role:8s} stroke test ({s["strokes"]} strokes): tip peak speed ratio median {s["peak_ratio_median"]:.3f} '
              f'(IQR {s["peak_ratio_iqr"][0]:.3f}-{s["peak_ratio_iqr"][1]:.3f}), peak shift {s["peak_shift_median_frames"]:+.0f} frames; '
              f'tip path {s["tip_path_m"]*1e3:.1f} mm, drone position {s["position_m"]*1e3:.1f} mm, tilt {s["tilt_deg"]:.2f} deg', flush=True)
    progress.phase_done(strokes['summary'].get('held_out', {}).get('tip_path_m', math.nan))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(dict(
        stage='coupled', method='segments', model='configured model, full coupled replay from the sent commands',
        horizon_s=args.horizon, values=values, cable_residual=config['cable'].get('residual'), firmware=firmware,
        results=results, bias_cancelled=corrected, biases_applied_m_s2=biases, stroke_test=strokes,
        seconds=time.time()-t0), indent=1, default=float))
    progress.done([], math.nan)


if __name__ == '__main__':
    main()
