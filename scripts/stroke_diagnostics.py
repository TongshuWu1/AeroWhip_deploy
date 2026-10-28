"""Coupling diagnostics on the stroke windows of the calibration flights.

A stroke window starts where the scripted reference acceleration of a decay, strokes or whip phase
rises above a threshold and lasts --horizon seconds (0.1 s of history before it). Every such window
of every usable flight, in both setups, is replayed open loop from the measured state with the sent
commands: the drone-only model on the flights without cable, the coupled model on the flights with
it. Reported, with and without the cable, the pooled drone position and tilt error as a function of
time into the stroke (raw, and with one constant acceleration bias per flight removed); with the
cable also the shape and tip error over time and the tip's peak speed against the measured one.
Then the sensitivity of the coupled stroke loss (`aerowhip.fitting.coupled` residuals) to the drone's
roll/pitch inertia, motor arm, motor response and centre-of-mass height: the cable's moment on the
drone during a stroke is an excitation the drone-only flights never gave.

    python scripts/stroke_diagnostics.py --output runs/diagnostics/strokes_<time>.json
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
from aerowhip.fitting import coupled, flight_drone
from aerowhip.fitting.calibration_set import folders, is_held_out
from aerowhip.fitting.least_squares import ProcessRunner, bias_profile, strong_correlations, uncertainty
from aerowhip.physics.model_configuration import DEFAULT_MODEL, load_model_configuration
from aerowhip.physics.state import quaternion_xyzw_to_rotation_matrix
from aerowhip.takes.calibration_flights import DT, MARKERS, Window, load

STROKE_PHASES = ('decay ', 'strokes ', 'whip ')
THRESHOLD = 1.5        # m/s^2 of reference acceleration: above the slow returns (1 m/s^2), below every stroke peak
HISTORY = 10           # frames before the window


def stroke_onsets(folder, flight, phases=STROKE_PHASES):
    """Stamps at which a stroke's reference acceleration starts (one per stroke)."""
    rows = [json.loads(l) for l in (Path(folder)/'flight.jsonl').read_text().splitlines()]
    a = np.array([np.linalg.norm(r['desired_acceleration']) for r in rows])
    stamps = flight.log_stamp
    onsets, quiet = [], 0
    for k, r in enumerate(rows):
        fast = r['phase'].startswith(phases) and a[k] > THRESHOLD
        if fast and quiet >= 10 and np.isfinite(stamps[k]):      # 0.2 s below the threshold before it
            onsets.append(float(stamps[k]))
        quiet = 0 if fast else quiet+1
    return onsets


def window_at(flight, first, steps):
    """The Window starting at pose frame `first` (as `calibration_flights.windows` builds them), or
    None where frames are missing or the markers are not usable."""
    if first < HISTORY or first+steps >= flight.frames:
        return None
    sel = slice(first-HISTORY, first+steps+1)
    if not (np.diff(flight.frame[sel]) == 1).all():
        return None
    m = flight.markers[sel] if flight.markers is not None else np.full((steps+HISTORY+1, MARKERS, 3), np.nan)
    seen = np.isfinite(m).all(2)
    if flight.markers is not None and (not seen[:HISTORY+1].all() or seen[HISTORY:].sum(1).mean() < 8):
        return None
    R = np.stack([quaternion_xyzw_to_rotation_matrix(q) for q in flight.quaternion[sel]])
    t = (flight.frame[sel]-flight.frame[first])*DT
    stamp = flight.stamp
    return Window(flight.folder.name, flight.phase_at(stamp[first]), float(stamp[first]), np.arange(steps+1)*DT,
                  flight.position[sel][HISTORY:].copy(), R[HISTORY:].copy(), m[HISTORY:].copy(), seen[HISTORY:].copy(),
                  flight.command_at(stamp[sel][HISTORY:]), flight.command_at(stamp[sel][HISTORY:], strip_trim=False),
                  flight.position[sel][:HISTORY+1].copy(), R[:HISTORY+1].copy(), m[:HISTORY+1].copy(), t[:HISTORY+1].copy())


def stroke_windows(setup, horizon, lead=0., phases=STROKE_PHASES):
    """{flight name: [(window, flight)]} of every usable flight of the setup; `lead` seconds of the
    window precede the stroke onset."""
    steps = round(horizon/DT)
    out = {}
    for name, folder in folders(setup).items():
        flight = load(folder)
        items = []
        for onset in stroke_onsets(folder, flight, phases):
            first = int(np.searchsorted(flight.stamp, onset-1e-6))-round(lead/DT)
            w = window_at(flight, first, steps)
            if w is not None:
                items.append((w, flight))
        out[name] = items
    return out


def _prepare(parameters):
    config, firmware, cable = parameters
    replay = flight_drone.DroneReplay(config, firmware)

    def one(item):
        w, flight = item
        if cable:
            p, R, markers, rates = replay.run(w, flight, pipeline=True, markers=True, attitude=True)
            shape = np.linalg.norm((markers-p[:, None])-(w.markers[1:]-w.position[1:, None]), axis=2)
            shape[~w.seen[1:]] = np.nan
            return dict(position=p-w.position[1:], tilt=flight_drone.tilt_error(R, w.rotation[1:]), shape=shape,
                        tip_speed=np.linalg.norm(rates[:, -1], axis=1))
        p, R = replay.run(w, flight, pipeline=True, attitude=True)
        return dict(position=p-w.position[1:], tilt=flight_drone.tilt_error(R, w.rotation[1:]))
    return one


def measured_tip_speed(w):
    """Speed of the last marker from central differences, NaN where it or a neighbour was hidden."""
    tip = w.markers[:, -1]
    v = np.full(len(tip), np.nan)
    v[1:-1] = np.linalg.norm((tip[2:]-tip[:-2])/(2*DT), axis=1)
    return v[1:]


def summarise(groups, results, times, cable):
    """Pooled error against time into the stroke, raw and debiased per flight."""
    index = {t: int(round(t/DT))-1 for t in times}
    raw, debiased, tilt, tilt_offset, shape, tip = ({t: [] for t in times} for _ in range(6))
    peaks = []
    failed = 0
    k = 0
    for name, items in groups.items():
        errors = []
        for (w, _), r in zip(items, results[k:k+len(items)]):
            if r is None:
                failed += 1
                continue
            errors.append(r)
            for t, i in index.items():
                raw[t].append(np.sum(r['position'][i]**2))
                tilt[t].append(np.sum(r['tilt'][i]**2))
                if cable:
                    s = r['shape'][i]
                    if np.isfinite(s).any():
                        shape[t].append(float(np.nanmean(s**2)))
                    if np.isfinite(s[-1]):
                        tip[t].append(float(s[-1]**2))
            if cable:
                measured = measured_tip_speed(w)
                if np.isfinite(measured).sum() > len(measured)//2:
                    im, isim = int(np.nanargmax(measured)), int(np.argmax(r['tip_speed']))
                    peaks.append(dict(flight=name, phase=w.phase, measured_m_s=float(measured[im]), simulated_m_s=float(r['tip_speed'][isim]),
                                      ratio=float(r['tip_speed'][isim]/measured[im]), shift_frames=isim-im))
        k += len(items)
        profiled, _ = bias_profile([e['position'] for e in errors])
        offset = np.concatenate([e['tilt'] for e in errors]).mean(0) if errors else np.zeros(2)
        for e, p in zip(errors, profiled):
            for t, i in index.items():
                debiased[t].append(np.sum(p[i]**2))
                tilt_offset[t].append(np.sum((e['tilt'][i]-offset)**2))

    def rms(d):
        return {str(t): (float(np.sqrt(np.mean(v)))) if v else math.nan for t, v in d.items()}
    out = dict(windows=sum(map(len, groups.values())), failed=failed, position_raw_m=rms(raw), position_debiased_m=rms(debiased),
               tilt_deg={t: math.degrees(v) for t, v in rms(tilt).items()},
               tilt_offset_removed_deg={t: math.degrees(v) for t, v in rms(tilt_offset).items()})
    if cable:
        out.update(shape_m=rms(shape), tip_m=rms(tip), tip_peaks=peaks,
                   tip_peak_ratio_median=float(np.median([p['ratio'] for p in peaks])) if peaks else math.nan,
                   tip_peak_shift_median_frames=float(np.median([p['shift_frames'] for p in peaks])) if peaks else math.nan)
    return out


def print_summary(label, s, cable):
    times = list(s['position_raw_m'])
    print(f'{label}: {s["windows"]} stroke windows, {s["failed"]} failed')
    print('   t (s)              ' + ''.join(f'{t:>8s}' for t in times))
    for key, scale, unit in (('position_raw_m', 1e3, 'mm'), ('position_debiased_m', 1e3, 'mm'), ('tilt_deg', 1, 'deg'),
                             ('tilt_offset_removed_deg', 1, 'deg')):
        print(f'   {key:20s}' + ''.join(f'{s[key][t]*scale:8.1f}' for t in times) + f'  {unit}')
    if cable:
        for key in ('shape_m', 'tip_m'):
            print(f'   {key:20s}' + ''.join(f'{s[key][t]*1e3:8.1f}' for t in times) + '  mm')
        print(f'   tip peak speed: simulated/measured median {s["tip_peak_ratio_median"]:.3f}, '
              f'peak shift median {s["tip_peak_shift_median_frames"]:+.0f} frames ({len(s["tip_peaks"])} strokes)')


SENSITIVITY_KEYS = ('inertia_scale_xy', 'arm_scale', 'motor_response_s', 'com_z')
STEPS = dict(inertia_scale_xy=.3, arm_scale=.1, motor_response_s=.01, com_z=.005)


_apply = coupled.apply


def apply_vehicle(config, **values):
    """coupled.apply extended with the stage-2 vehicle parameters."""
    c = _apply(config, **{k: v for k, v in values.items() if k in coupled.KEYS})
    if 'inertia_scale_xy' in values:
        J = c['drone']['inertia_body_about_com_kg_m2']
        J[0][0] *= values['inertia_scale_xy']; J[1][1] *= values['inertia_scale_xy']
    if 'arm_scale' in values:
        c['actuator'] = deepcopy(c['actuator'])
        c['actuator']['motor_positions_com_m'] = [[q[0]*values['arm_scale'], q[1]*values['arm_scale'], q[2]]
                                                  for q in c['actuator']['motor_positions_com_m']]
    if 'motor_response_s' in values:
        c['actuator'] = deepcopy(c['actuator'])
        c['actuator']['response_time_s'] = [values['motor_response_s']]*4
    return c


def sensitivity(config, firmware, groups):
    coupled.apply = apply_vehicle                      # the forked workers see this module state
    x0 = dict(inertia_scale_xy=1., arm_scale=1., motor_response_s=float(config['actuator']['response_time_s'][0]),
              com_z=float(config['drone']['center_of_mass_tracking_m'][2]))
    residuals = coupled.Residuals(config, firmware, groups)
    errors, messages = residuals.errors(**x0)
    groups = {n: [w for w, e in zip(groups[n], errors[n]) if e is not None] for n in groups}
    residuals.runner.close()
    residuals = coupled.Residuals(config, firmware, {n: v for n, v in groups.items() if v})
    residuals.set_scales(**x0)
    r0, failures, report0 = residuals(**x0)
    print(f'sensitivity on {sum(map(len, residuals.groups.values()))} coupled stroke windows: reference position '
          f'{report0["position_m"]*1e3:.1f} mm, tilt {report0["tilt_deg"]:.2f} deg, shape {report0["shape_m"]*1e3:.1f} mm, tip {report0["tip_m"]*1e3:.1f} mm', flush=True)
    n = len(r0)//15
    kinds = np.concatenate([np.repeat([0, 1, 2], [3, 2, 10])]*n)
    rows, columns = [], []
    for key in SENSITIVITY_KEYS:
        x = dict(x0); x[key] += STEPS[key]
        r, f, report = residuals(**x)
        columns.append((r-r0)/STEPS[key])
        change = residuals.value(r)-1.
        parts = {name: float(np.sqrt(np.mean(r[kinds == c]**2))-np.sqrt(np.mean(r0[kinds == c]**2)))
                 for c, name in enumerate(('position', 'tilt', 'shape'))}
        rows.append(dict(parameter=key, step=STEPS[key], loss_change=change, component_change=parts,
                         position_m=report['position_m'], tilt_deg=report['tilt_deg'], shape_m=report['shape_m'], tip_m=report['tip_m'], failed=f))
        print(f'   {key:18s} +{STEPS[key]:<6g}: loss {1+change:.4f} ({change:+.4f}); position {parts["position"]:+.4f}, '
              f'tilt {parts["tilt"]:+.4f}, shape {parts["shape"]:+.4f}; tip {report["tip_m"]*1e3:.1f} mm', flush=True)
    u = uncertainty(np.column_stack(columns), r0, residuals.flight_groups(), list(SENSITIVITY_KEYS))
    print('   joint 95 % half-widths if fitted on these strokes:', {k: f'{1.96*s:.3g}' for k, s in zip(SENSITIVITY_KEYS, u['standard_deviation'])}, flush=True)
    for a, b, c in strong_correlations(u, .8):
        print(f'   correlated: {a} / {b} {c:+.2f}', flush=True)
    residuals.runner.close()
    return dict(reference=dict(values=x0, position_m=report0['position_m'], tilt_deg=report0['tilt_deg'], shape_m=report0['shape_m'],
                               tip_m=report0['tip_m'], failed=failures), windows={n: len(v) for n, v in residuals.groups.items()},
                sensitivity=rows, joint_uncertainty=u)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--horizon', type=float, default=.6)
    parser.add_argument('--lead', type=float, default=0., help='seconds of window before the stroke onset (hover: the initialisation floor)')
    parser.add_argument('--times', type=float, nargs='+', default=[.1, .2, .3, .4, .6])
    parser.add_argument('--phases', nargs='+', default=list(STROKE_PHASES), help="stroke phase prefixes ('decay ' alone: 8 s holds before each stroke)")
    parser.add_argument('--no-sensitivity', action='store_true')
    args = parser.parse_args()
    t0 = time.time()
    config = load_model_configuration(DEFAULT_MODEL)
    firmware = tomllib.loads(flight_drone.FIRMWARE.read_text())
    out = dict(horizon_s=args.horizon, lead_s=args.lead, phases=args.phases, threshold_m_s2=THRESHOLD, results={})
    for setup, cable in (('cf_2_nocable', False), ('cf_2', True)):
        groups = stroke_windows(setup, args.horizon, args.lead, tuple(args.phases))
        groups = {n: v for n, v in groups.items() if v}
        items = [item for v in groups.values() for item in v]
        print(f'{setup}: {len(items)} stroke windows from {len(groups)} flights ' +
              '(' + ', '.join(f'{n} {len(v)}' for n, v in groups.items()) + ')', flush=True)
        model = config if cable else flight_drone.drone_only(config)
        runner = ProcessRunner(items, _prepare)
        results = [r for r, _ in runner((model, firmware, cable))]
        runner.close()
        split = {}
        for role in ('fit', 'held_out'):
            sub = {n: v for n, v in groups.items() if (is_held_out(v[0][1].folder)) == (role == 'held_out')}
            if not sub:
                continue
            picked = []
            k = 0
            for n, v in groups.items():
                if n in sub:
                    picked.extend(results[k:k+len(v)])
                k += len(v)
            split[role] = summarise(sub, picked, args.times, cable)
            print_summary(f'{setup} {role}', split[role], cable)
        out['results'][setup] = split
        if cable and not args.no_sensitivity:
            out['sensitivity'] = sensitivity(config, firmware, {n: v for n, v in groups.items() if not is_held_out(v[0][1].folder)})
    out['seconds'] = time.time()-t0
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(out, indent=1, default=float))
    print(f'written {args.output} ({out["seconds"]:.0f} s)')


if __name__ == '__main__':
    main()
