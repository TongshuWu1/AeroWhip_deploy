"""Pre-screen of an attitude-loop tuning plan in the simulator, before it is flown.

Each set of a plan in the flight stack's config/tuning.yaml is flown in the simulator the way
`acceleration_circle_cf2.py --tune` flies it (hover, then sharp out-and-back strokes along +x and
+y), by the drone without cable (stage-2 model), the flown firmware's attitude law
(`configs/controllers/crazyflie_firmware.toml`) with the set's ctrlMel gains, and the flight
script's host PID at 50 Hz on the true state. Per set: tilt lag, gain and overshoot of the thrust
axis against the commanded one, the lowest single-motor force against the hover share, and the
position error; a set whose tilt or position runs away is marked UNSTABLE.

What it cannot judge: sensor noise and vibration (the simulator has none, so motor jitter and
the damping/noise trade-off are flight-only), the onboard estimate (locSrv.extQuatStdDev is not
modelled and is ignored here), motor imbalance (the hover trim and the attitude integral's effect
on it), and powerDist.idleThrust.

    python scripts/tuning_prescreen.py stiffness
"""
import argparse
from copy import deepcopy
import math
from pathlib import Path
import sys
import tomllib

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from aerowhip.fitting import flight_drone
from aerowhip.physics.model import Model, native_module
from aerowhip.physics.model_configuration import DEFAULT_MODEL, fit_configuration, load_model_configuration

FLIGHT_STACK = ROOT.parent/'Crazyflie'/'crazyswarm2_jazzy_whipping'
COMMAND_WIDTH = 13


def flight_script(stack):
    sys.path.insert(0, str(stack/'scripts'))
    import acceleration_circle_cf2 as flight
    return flight


def simulate(firmware, config, voltage, motion, k, controller_class):
    """Fly set k of the tuning motion; returns per-10 ms arrays (time, commanded axis, actual axis,
    position error, min motor force) or None if the engine failed."""
    native = native_module()
    model = Model(config)
    weight = -(model.mass+float(np.sum(model.node_mass)))*float(model.gravity[2])
    controller = native.FlightController(firmware, config, voltage)
    start = np.array([0., 0., 1.5])
    state = model.native.initial_state(start, np.eye(3), np.zeros(3), np.zeros(3), None, None)

    def command(a):
        c = np.zeros(COMMAND_WIDTH); c[6:9] = a; return c
    low, high = -5., 12.
    for _ in range(60):                                     # vertical feed-forward for hover
        mid = .5*(low+high)
        if controller(state, command([0., 0., mid]))[0] < weight:
            low = mid
        else:
            high = mid
    trim = np.array([0., 0., .5*(low+high)])
    actuator = deepcopy(config['actuator'])
    thrust = controller(state, command(trim))[0]
    actuator['initial_thrust_n'] = [float(np.clip(thrust/4, actuator['minimum_thrust_n'][i], actuator['maximum_thrust_n'][i])) for i in range(4)]
    runtime = native.Runtime(model.native, state, command(trim), config['controller']['controller_period_s'],
                             config['controller']['command_delay_s'], native.Actuator(actuator))
    starts = [t for t in np.arange(0, motion.duration, .01) if motion.set_index(t) == k]
    t0, t1 = starts[0]-1., starts[-1]+.5                   # one settling second before the set
    host = controller_class()
    out = dict(t=[], commanded=[], actual=[], error=[], motor=[], phase=[])
    sent = command(trim)
    hover_share = weight/4
    try:
        for i, t in enumerate(np.arange(t0, t1, .01)):
            s = runtime.state()
            obs = model.native.observation(s)
            p, v, R = obs[:3], obs[3:6], obs[6:15].reshape(3, 3)
            tm = min(max(t, 0.), motion.duration)
            dp, dv, da = motion.state(tm) if t >= 0 else (np.zeros(3), np.zeros(3), np.zeros(3))
            events = []
            if i % 2 == 0:                                  # host PID at 50 Hz
                a, _ = host.command(p, v, start+dp, dv, da, .02, trim)
                sent = command(a)
                events = [(runtime.time, sent)]
            runtime.step(events, controller)
            axis = sent[6:9]+np.array([0., 0., 9.81])
            out['t'].append(t); out['commanded'].append(axis/np.linalg.norm(axis)); out['actual'].append(R[:, 2])
            out['error'].append(np.linalg.norm(start+dp-p)); out['motor'].append(min(controller.motors(s, sent))/hover_share)
            out['phase'].append(motion.phase(tm) if t >= 0 else 'settle')
            if np.degrees(np.arccos(np.clip(R[2, 2], -1, 1))) > 60 or np.linalg.norm(start+dp-p) > .5:
                return None
    except Exception:
        return None
    return {k: np.array(v) for k, v in out.items()}


def response(run):
    """lag (s), gain, overshoot (median peak ratio) over the step part, 10 ms resolution."""
    step = np.array([ph.endswith(' step') for ph in run['phase']])
    c, a = run['commanded'][:, :2], run['actual'][:, :2]
    best = None
    for lag in range(0, 31):
        idx = np.flatnonzero(step)
        idx = idx[idx+lag < len(a)]
        cc, aa = c[idx]-c[idx].mean(0), a[idx+lag]-a[idx+lag].mean(0)
        score = float(np.sum(cc*aa))/math.sqrt(float(np.sum(cc*cc))*float(np.sum(aa*aa)))
        if best is None or score > best[0]:
            best = (score, lag, float(np.sum(cc*aa))/float(np.sum(cc*cc)), idx)
    _, lag, gain, idx = best
    size_c = np.linalg.norm(c-c[idx].mean(0), axis=1)
    size_a = np.linalg.norm(a-a[idx].mean(0), axis=1)
    active = step & (size_c > .3*size_c[step].max())
    edges = np.flatnonzero(np.diff(np.r_[0, active.astype(int), 0]))
    ratios = [size_a[s:e+lag+1].max()/size_c[s:e].max() for s, e in zip(edges[::2], edges[1::2]) if e-s > 2]
    return lag*.01, gain, float(np.median(ratios)) if ratios else math.nan


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('plan')
    p.add_argument('--stack', type=Path, default=FLIGHT_STACK, help='crazyswarm2 repository with config/tuning.yaml')
    p.add_argument('--voltage', type=float, default=7.4, help='battery voltage for the thrust map (default 7.4 V)')
    args = p.parse_args()
    flight = flight_script(args.stack)
    sets, motion = flight.tuning_plan(args.plan)
    base = tomllib.loads(flight_drone.FIRMWARE.read_text())
    config = fit_configuration(flight_drone.drone_only(load_model_configuration(DEFAULT_MODEL)))
    print(f'Plan {args.plan}: simulator, drone without cable, firmware attitude law, host PID, {args.voltage} V')
    print(f'{"set":18s} {"lag ms":>7s} {"gain":>6s} {"over":>6s} {"min motor/hover":>16s} {"pos mm":>7s}  ignored here')
    for k, item in enumerate(sets):
        firmware = deepcopy(base)
        ignored = []
        for key, value in item['params'].items():
            group, name = key.split('.')
            if group == 'ctrlMel':
                firmware['ctrlMel'][name] = value
            else:
                ignored.append(key)
        run = simulate(firmware, config, args.voltage, motion, k, flight.AccelerationController)
        label = f'{k} {item["name"]}'
        if run is None:
            print(f'{label:18s} UNSTABLE (tilt above 60 deg, position error above 0.5 m, or engine failure)')
            continue
        lag, gain, over = response(run)
        print(f'{label:18s} {lag*1e3:7.0f} {gain:6.2f} {over:6.2f} {run["motor"].min():16.2f} {1e3*math.sqrt(np.mean(run["error"]**2)):7.0f}'
              f'  {", ".join(ignored)}', flush=True)


if __name__ == '__main__':
    main()
