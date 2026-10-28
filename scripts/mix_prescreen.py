"""Cable flip-over screening of a planned flight, before it is flown.

The flight stack's flight (a mixed calibration flight, `--mix NAME`, or any argument list the flight
script accepts) is flown in the simulator with the cable: the configured model (the cable as fitted
on cf_2, cf_6's geometry and mass), the flown firmware's attitude law at the gains in the flight
stack's crazyflies.yaml (kR_xy, kd_omega_rp, ...), and the flight script's host PID at 50 Hz
following the same reference, from the hover point. Per part (phase): how high any point of the
simulated cable rose above the clamp (`rise`, m; > 0 means the cable went above the attachment
level), the largest angle of the cable's first link from hanging straight down, and the drone's
largest tilt. A part is flagged RISK when the cable rose above the clamp.

NOT A SAFETY SCREEN (2026-10-28): on the part that flipped the real cable over the drone within 1 s
of the first stroke (mix_a, sharp stop at 6 m/s^2, `--override 1:level=6 --override 1:hold=5`) this
simulation gives a first-link angle of 33 deg and no rise above the clamp (the real cable: 162-175 deg
from hanging; 59 deg real vs 21 simulated at 3 m/s^2). The configured cable (cf_2's fit) underpredicts
the response to sharp strokes by that much. Kept as a sim-against-real comparison tool for the mixed
flights; flip-over limits come from measured cable angles (`cloud.jsonl`), not from this.

    python scripts/mix_prescreen.py --mix mix_a
    python scripts/mix_prescreen.py --all
"""
import argparse
from copy import deepcopy
import math
from pathlib import Path
import sys
import tomllib

import numpy as np
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from aerowhip.fitting import flight_drone
from aerowhip.physics.model import Model, native_module, settled_cable
from aerowhip.physics.model_configuration import DEFAULT_MODEL, fit_configuration, load_model_configuration

FLIGHT_STACK = ROOT.parent/'Crazyflie'/'crazyswarm2_jazzy_whipping'
COMMAND_WIDTH = 13
MIXES = ('mix_a', 'mix_b', 'mix_c', 'mix_d', 'mix_e', 'mix_val')


def flight_script(stack):
    sys.path.insert(0, str(stack/'scripts'))
    import acceleration_circle_cf2 as flight
    return flight


def firmware_with_flown_gains(stack):
    """The firmware law's config with the ctrlMel gains the flight stack's YAML sets."""
    firmware = tomllib.loads(flight_drone.FIRMWARE.read_text())
    flown = yaml.safe_load((stack/'config'/'crazyflies.yaml').read_text())['all']['firmware_params'].get('ctrlMel', {})
    for key in ('kR_xy', 'kw_xy', 'kd_omega_rp', 'ki_m_xy', 'i_range_m_xy', 'kR_z', 'kw_z'):
        if key in flown:
            firmware['ctrlMel'][key] = float(flown[key])
    return firmware


def screen(flight, args, firmware, config, voltage=7.6):
    """Per-phase maxima of cable rise above the clamp (m), first-link angle (deg) and drone tilt (deg)."""
    native = native_module()
    model = Model(config)
    gravity = 9.81
    weight = (model.mass+float(np.sum(model.node_mass)))*gravity
    controller = native.FlightController(firmware, config, voltage)
    hover = np.array([0., 0., args.height+args.lift])
    root = hover+np.asarray(model.attachment_tracking)
    cable = settled_cable(model, root)
    state = model.native.initial_state(hover, np.eye(3), np.zeros(3), np.zeros(3), cable, np.zeros_like(cable))

    def command(a):
        c = np.zeros(COMMAND_WIDTH); c[6:9] = a; return c
    low, high = -5., 12.
    for _ in range(60):
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
    host = flight.AccelerationController()
    start = flight.motion_start(args)
    end = start+args.motion.duration
    origin = np.zeros(3)
    n = len(model.rest)+1
    rows = {}
    sent = command(trim)
    for i, t in enumerate(np.arange(start-1., end+3., .01)):
        s = runtime.state()
        obs = model.native.observation(s)
        p, v, R = obs[:3], obs[3:6], obs[6:15].reshape(3, 3)
        q = obs[18:18+3*n].reshape(n, 3)
        phase, ref, dref, aref = flight.reference(max(t, start-1.), origin, args)
        if i % 2 == 0:
            a, _ = host.command(p, v, ref, dref, aref, .02, trim)
            sent = command(a)
            events = [(runtime.time, sent)]
        else:
            events = []
        try:
            runtime.step(events, controller)
        except Exception as exc:
            rows.setdefault(phase, dict(rise=math.inf, link=math.nan, tilt=math.nan, failed=str(exc)))
            break
        rise = float((q[1:, 2]-q[0, 2]).max())
        link = q[1]-q[0]
        angle = math.degrees(math.acos(np.clip(-link[2]/np.linalg.norm(link), -1, 1)))
        tilt = math.degrees(math.acos(np.clip(R[2, 2], -1, 1)))
        r = rows.setdefault(phase, dict(rise=-math.inf, link=0., tilt=0.))
        r['rise'], r['link'], r['tilt'] = max(r['rise'], rise), max(r['link'], angle), max(r['tilt'], tilt)
        if tilt > 80:
            r['failed'] = 'drone tilt above 80 deg'
            break
    return rows


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--mix', action='append', help='mix name(s) from the flight stack; --all for every mix')
    p.add_argument('--all', action='store_true')
    p.add_argument('--setup', default='cf_6', help='physical setup (with cable by default)')
    p.add_argument('--override', action='append', default=[], metavar='PART:KEY=VALUE',
                   help='change one part of the mix for this screen only, e.g. 1:level=6 or 1:hold=5 (0-based part index)')
    p.add_argument('--stack', type=Path, default=FLIGHT_STACK)
    args = p.parse_args()
    flight = flight_script(args.stack)
    firmware = firmware_with_flown_gains(args.stack)
    config = fit_configuration(load_model_configuration(DEFAULT_MODEL))
    names = list(MIXES) if args.all else (args.mix or [])
    if not names:
        p.error('give --mix NAME or --all')
    print(f"Gains: kR_xy {firmware['ctrlMel']['kR_xy']:g}, kw_xy {firmware['ctrlMel']['kw_xy']:g}, "
          f"kd_omega_rp {firmware['ctrlMel']['kd_omega_rp']:g}; setup {args.setup}; model cable as configured")
    for name in names:
        fargs = flight.arguments(['--mix', name, '--setup', args.setup])
        if args.override:
            import flight_trajectories as ft
            mixes = yaml.safe_load((args.stack/'config'/'calibration_mixes.yaml').read_text())['mixes']
            specs = [dict(s) for s in mixes[name]['parts']]
            cable = not args.setup.endswith('_nocable')
            for spec in specs:
                for key in [k for k in spec if k.startswith('cable_')]:
                    value = spec.pop(key)
                    if cable:
                        spec[key[6:]] = value
            for item in args.override:
                index, assignment = item.split(':', 1)
                key, value = assignment.split('=', 1)
                specs[int(index)][key] = float(value)
            fargs.motion = ft.mix(name, specs, float(mixes[name].get('stroke_length', 1.)))
            fargs.lift = max(0., -min(fargs.motion.state(t)[0][2] for t in np.arange(0, fargs.motion.duration+.005, .005)))
        rows = screen(flight, fargs, firmware, config)
        print(f'\n{name}' + (f'  (overrides: {", ".join(args.override)})' if args.override else ''))
        print(f'  {"part":24s} {"cable rise above clamp":>22s} {"first link from down":>21s} {"drone tilt":>11s}')
        for phase, r in rows.items():
            if phase in ('takeoff', 'settle', 'landing', 'settling') or phase.endswith('-move'):
                continue
            flag = 'RISK' if r['rise'] > 0 or r.get('failed') else ''
            print(f"  {phase:24s} {r['rise']*100:+18.1f} cm {r['link']:17.0f} deg {r['tilt']:8.0f} deg  {flag} {r.get('failed', '')}", flush=True)


if __name__ == '__main__':
    main()
