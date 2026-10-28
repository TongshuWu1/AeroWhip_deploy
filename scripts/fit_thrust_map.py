"""Stage 1: the PWM-to-thrust map from the drone-only flights' motor logs.

Every drone-only flight with `motors.jsonl` (firmware motor PWM m1..m4 and battery at ~33 Hz)
gives, at each motor sample, the thrust the drone must have produced: m (a - g + a_drag) along
the body z axis, with a the drone's measured acceleration (least-squares quadratic over the
OptiTrack poses within +-`--span` s, evaluated `--lag` s after the motor sample: the motors
respond with a ~22 ms time constant and the log is received late) and a_drag the adopted body
drag. The map
    T = kappa (1 + beta (V - V0)) / 4 * sum_i (pwm_i / massThrust)^p
is fitted (Nelder-Mead on kappa, p, beta; V0 and massThrust fixed) to the mean required thrust
of each mean-PWM bin, in log ratio: fitting the individual samples would let the PWM scatter at
a given thrust (attitude loop, motor lag) pull the exponent down (regression dilution). Samples
on the ground, saturated (any motor at 65535) or below `--minimum-thrust` are left out. The
report gives the map/required ratio per bin and per thrust band. Every number is computed twice.

    python scripts/fit_thrust_map.py --output runs/calibration_2026-10-27/thrust_map_fit.json
"""
import argparse
import json
from pathlib import Path
import sys
import time

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from aerowhip.fitting import flight_drone
from aerowhip.fitting.progress import Progress
from aerowhip.fitting.calibration_set import manifest
from aerowhip.physics.model_configuration import DEFAULT_MODEL, load_model_configuration
from aerowhip.physics.state import quaternion_xyzw_to_rotation_matrix
from aerowhip.takes.calibration_flights import DT, camera_frames

G = 9.80665


def nelder_mead(f, x0, step, *, iterations=80, tolerance=2e-5, progress=None):
    """Minimise f from x0. progress (aerowhip.fitting.progress.Progress or None) receives every
    evaluation and every iteration's best point."""
    if progress is not None:
        objective = f

        def f(x):
            value = objective(x)
            progress.evaluation(x, value)
            return value
    n = len(x0)
    simplex = [np.array(x0, float)]+[np.array(x0, float)+np.eye(n)[i]*step[i] for i in range(n)]
    values = [f(x) for x in simplex]
    history = []
    for it in range(iterations):
        order = np.argsort(values)
        simplex, values = [simplex[i] for i in order], [values[i] for i in order]
        history.append((it, float(values[0]), simplex[0].tolist()))
        if progress is not None:
            progress.iteration(it, values[0], simplex[0])
        print(f'  iteration {it:2d}  best {values[0]:.6g}  x {np.round(simplex[0], 4).tolist()}', flush=True)
        if values[-1]-values[0] < tolerance:
            break
        centroid = np.mean(simplex[:-1], axis=0)
        reflected = centroid+(centroid-simplex[-1]); fr = f(reflected)
        if fr < values[0]:
            expanded = centroid+2*(centroid-simplex[-1]); fe = f(expanded)
            simplex[-1], values[-1] = (expanded, fe) if fe < fr else (reflected, fr)
        elif fr < values[-2]:
            simplex[-1], values[-1] = reflected, fr
        else:
            contracted = centroid+.5*(simplex[-1]-centroid); fc = f(contracted)
            if fc < values[-1]:
                simplex[-1], values[-1] = contracted, fc
            else:
                for i in range(1, n+1):
                    simplex[i] = simplex[0]+.5*(simplex[i]-simplex[0]); values[i] = f(simplex[i])
    best = int(np.argmin(values))
    return simplex[best], values[best], history


def samples(folder, model, span_s, lag_s=0.):
    """Per motor sample: required thrust (N), the four PWM values, battery voltage."""
    poses = [json.loads(l) for l in (folder/'poses.jsonl').read_text().splitlines()]
    stamp = np.array([p['stamp'] for p in poses]); order = np.argsort(stamp, kind='stable')
    stamp = stamp[order]; keep = np.r_[True, np.diff(stamp) > 1e-6]
    stamp = stamp[keep]; received = np.array([poses[i]['received'] for i in order])[keep]
    position = np.array([poses[i]['position'] for i in order])[keep]
    quaternion = np.array([poses[i]['quaternion_xyzw'] for i in order])[keep]
    # camera frame times (the receipt stamps jitter by a few ms; see calibration_flights.camera_frames)
    period = DT/2 if np.median(np.diff(stamp)) < .75*DT else DT
    camera = stamp[0]+camera_frames(stamp, period)*period
    motors = [json.loads(l) for l in (folder/'motors.jsonl').read_text().splitlines()]
    rows = [json.loads(l) for l in (folder/'flight.jsonl').read_text().splitlines()]
    # flying: between the first and last log row whose phase is neither takeoff/settle nor landing
    flying = [r['pose_stamp'] for r in rows if r.get('pose_stamp') is not None
              and not r['phase'].startswith(('takeoff', 'settle', 'land'))]
    if not flying:
        return np.empty((0, 6))
    t_flying = (min(flying), max(flying))
    mass = model['drone']['mass_kg']; drag = np.asarray(model['drone']['body_drag_s_inv'])
    out = []
    for m in motors:
        pwm = np.asarray(m['motors'], float)
        if m.get('vbat') is None or pwm.max() >= 65535 or pwm.min() <= 0:
            continue
        t = float(np.interp(m['received'], received, stamp))+lag_s   # motor sample on the OptiTrack clock, plus the motor lag
        if not t_flying[0] <= t <= t_flying[1]:
            continue
        c = float(np.interp(t, stamp, camera))
        k = (camera >= c-span_s) & (camera <= c+span_s)
        if k.sum() < 7:
            continue
        u = camera[k]-c
        design = np.column_stack((np.ones_like(u), u, u*u))
        coefficients = np.linalg.lstsq(design, position[k], rcond=None)[0]
        velocity, acceleration = coefficients[1], 2*coefficients[2]
        j = int(np.argmin(np.abs(stamp-t)))
        R = quaternion_xyzw_to_rotation_matrix(quaternion[j])
        drag_acceleration = R@(drag*(R.T@velocity))                     # the force the drag removes
        thrust = mass*((acceleration-np.array([0., 0., -G])+drag_acceleration)@R[:, 2])
        out.append((thrust, *pwm, float(m['vbat'])))
    return np.array(out)


def predicted(x, pwm, vbat, mass_thrust, v0):
    kappa, p, beta = x
    return kappa*(1.+beta*(vbat-v0))/4.*np.sum((pwm/mass_thrust)**p, axis=1)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--span', type=float, default=.075, help='half-width of the acceleration fit, s')
    parser.add_argument('--iterations', type=int, default=200)
    parser.add_argument('--minimum-thrust', type=float, default=1.2, help='fit samples at or above this required thrust, N')
    parser.add_argument('--lag', type=float, default=.04, help='acceleration evaluated this long after the motor sample, s')
    parser.add_argument('--bin', type=float, default=2500., help='mean-PWM bin width')
    parser.add_argument('--minimum-bin', type=int, default=50, help='samples a bin needs to count')
    args = parser.parse_args()
    model = load_model_configuration(DEFAULT_MODEL)
    firmware = flight_drone.firmware_config()
    mass_thrust, v0 = float(firmware['ctrlMel']['massThrust']), float(firmware['reference_voltage_v'])
    m = manifest()
    data, per_flight = [], {}
    from aerowhip.fitting.calibration_set import usable
    for f in usable(m, m.flights('cf_2_nocable', roles=('fit', 'held_out', 'hover'))):   # never a failed take
        if 'motors.jsonl' not in f.get('logs', []):
            continue
        s = samples(m.folder(f), model, args.span, args.lag)
        per_flight[f['folder']] = int(len(s))
        if len(s):
            data.append(s)
    data = np.concatenate(data)
    data = data[data[:, 0] >= args.minimum_thrust]
    thrust, pwm, vbat = data[:, 0], data[:, 1:5], data[:, 5]
    print(f'{len(data)} motor samples from {len(per_flight)} flights; thrust {thrust.min():.2f}..{thrust.max():.2f} N, '
          f'battery {vbat.min():.2f}..{vbat.max():.2f} V', flush=True)
    start = [float(firmware['thrust_scale']), float(firmware['thrust_exponent']), float(firmware['voltage_slope_v_inv'])]

    mean_pwm = pwm.mean(1)
    edges = np.arange(np.floor(mean_pwm.min()/args.bin)*args.bin, mean_pwm.max()+args.bin, args.bin)
    bins = [np.flatnonzero((mean_pwm >= lo) & (mean_pwm < hi)) for lo, hi in zip(edges[:-1], edges[1:])]
    bins = [k for k in bins if len(k) >= args.minimum_bin]
    print(f'{len(bins)} PWM bins of {args.bin:g}: '+' '.join(f'{int(mean_pwm[k].min()//args.bin*args.bin)}:{len(k)}' for k in bins), flush=True)

    def bin_ratios(x):
        return np.array([predicted(x, pwm[k], vbat[k], mass_thrust, v0).mean()/thrust[k].mean() for k in bins])

    def objective(x):
        if x[0] <= 0 or not .5 <= x[1] <= 4 or not -1 <= x[2] <= 1:
            return 1e9
        return float(np.sqrt(np.mean(np.log(bin_ratios(x))**2)))

    def report(x):
        residual = predicted(x, pwm, vbat, mass_thrust, v0)-thrust
        bands = {}
        for lo, hi in ((0., 1.3), (1.3, 1.6), (1.6, 2.), (2., 2.5), (2.5, 3.), (3., 4.), (4., 9.)):
            k = (thrust >= lo) & (thrust < hi)
            if k.sum() >= 20:
                bands[f'{lo:g}-{hi:g} N'] = dict(samples=int(k.sum()), rms_n=float(np.sqrt(np.mean(residual[k]**2))),
                                                mean_ratio=float(np.mean(predicted(x, pwm[k], vbat[k], mass_thrust, v0))/np.mean(thrust[k])))
        return dict(x=list(map(float, x)), rms_n=float(np.sqrt(np.mean(residual**2))),
                    bin_rms=float(np.sqrt(np.mean(np.log(bin_ratios(x))**2))), bin_ratios=bin_ratios(x).round(4).tolist(), bands=bands)
    t0 = time.time()
    initial = report(start)
    progress = Progress(args.output, stage='thrust map', parameters=['kappa', 'exponent p', 'voltage slope beta'], x0=start, unit='log ratio')
    x, value, history = nelder_mead(objective, start, [.05, .1, .05], iterations=args.iterations, tolerance=1e-9, progress=progress)
    # The real motors swing around their mean (attitude loop on gyro noise); with a convex map that
    # delivers more than four motors at the mean, which the noise-free simulated firmware does not do.
    # The collective scale the simulator uses carries that factor.
    spread = float(np.mean(np.mean((pwm/mass_thrust)**x[1], axis=1)/(pwm.mean(1)/mass_thrust)**x[1]))
    passes = [report(x) for _ in range(2)]
    reproduced = passes[0] == passes[1]
    for label, r in (('current', initial), ('fitted', passes[0])):
        print(f"{label}: kappa {r['x'][0]:.4f} p {r['x'][1]:.3f} beta {r['x'][2]:.4f}  bins {r['bin_rms']*100:.2f} % {r['bin_ratios']}  sample rms {r['rms_n']*1e3:.0f} mN  "
              + '  '.join(f"{b}: {v['mean_ratio']:.3f}" for b, v in r['bands'].items()), flush=True)
    print('reproduced twice:', reproduced, '; motor spread factor %.4f -> collective thrust scale %.4f' % (spread, x[0]*spread))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(dict(
        data='drone-only flights with motor logs (manifest), 2026-10-27', flights=per_flight, samples=int(len(data)),
        minimum_thrust_n=args.minimum_thrust, lag_s=args.lag, bin=args.bin, loss='rms over mean-PWM bins of log(mean map / mean required thrust)',
        span_s=args.span, mass_kg=model['drone']['mass_kg'], body_drag_s_inv=model['drone']['body_drag_s_inv'],
        massThrust=mass_thrust, reference_voltage_v=v0, parameters=['thrust_scale', 'thrust_exponent', 'voltage_slope_v_inv'],
        current=initial, best=dict(passes[0], reproduced_twice=reproduced), motor_spread_factor=spread,
        adopt=dict(thrust_scale=float(x[0]*spread), thrust_exponent=float(x[1]), voltage_slope_v_inv=float(x[2])),
        history=history, seconds=time.time()-t0), indent=1))
    progress.done(x, value)


if __name__ == '__main__':
    main()
