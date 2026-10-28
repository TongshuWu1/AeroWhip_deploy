"""A synthetic flight with known parameters, written like a real one: the simulator flies a planned
trajectory (closed loop, firmware law at the flown gains, the host PID) with deliberately changed
"true" parameters, and the flight folder is written line by line at real-time pace in the flight
script's format (parameters.json, flight.jsonl at 50 Hz, poses.jsonl and cloud.jsonl at 100 Hz with
the cable markers and four drone markers plus noise, result.json at the end). The online estimator
following it live (`online_fit.py --watch`) must move towards the true values; nothing about the
flight stack or the radio is involved. Display and test only.

    python scripts/synthetic_flight.py --trajectory runs/planning/excitation_002.json \
        --truth "log10 EI=+0.25, Cd=*1.3, motor_response_s=*0.6, command_delay_s=+0.006"
"""
import argparse
import json
from pathlib import Path
import sys
import time

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT/'scripts'))
from aerowhip.fitting.flight_cable import _quaternion  # noqa: E402
from aerowhip.online.estimator import ModelSpace  # noqa: E402
from aerowhip.physics.model import Model  # noqa: E402
from aerowhip.planning.ensemble import load_fit  # noqa: E402
from aerowhip.planning.simulate import FLIGHT_STACK, flight_script, simulate_motion  # noqa: E402
from online_fit import base_model, parse_start  # noqa: E402
from plan_flight import ARCHIVE, flown_gains  # noqa: E402

DRONE_MARKERS = np.array([[.04, .03, .01], [-.04, .03, .01], [-.03, -.04, .01], [.04, -.02, .015]])   # body frame, m


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--trajectory', type=Path, required=True)
    p.add_argument('--truth', default='log10 EI=+0.25, Cd=*1.3, motor_response_s=*0.6, command_delay_s=+0.006')
    p.add_argument('--root', type=Path, default=ROOT/'runs'/'online'/'synthetic'/'cf_6', help='the "setup" folder to write into')
    p.add_argument('--noise-mm', type=float, default=1.)
    p.add_argument('--speed', type=float, default=1., help='write pace (1 = real time; 0 = at once)')
    p.add_argument('--seed', type=int, default=0)
    args = p.parse_args()
    rng = np.random.default_rng(args.seed)
    base = base_model()
    space = ModelSpace(base, load_fit(ARCHIVE/'cable'/'20261028-012011.json'), load_fit(ARCHIVE/'drone'/'20261027-201939.json'))
    truth = parse_start(args.truth, space)
    config = space.config(truth)
    firmware = flown_gains(FLIGHT_STACK)
    _, trajectories = flight_script()
    data = json.loads(args.trajectory.read_text())
    motion = trajectories.TrajectoryFile(data, str(data.get('name', args.trajectory.stem)))
    print('simulating the flight with the true parameters ...', flush=True)
    r = simulate_motion(config, firmware, motion, markers_every=1, nodes=True)
    if r['failed']:
        raise SystemExit(f"simulation failed: {r['failed']}")
    index = np.asarray(Model(config).marker_indices)[1:]
    take = args.root/'trajectory'
    take.mkdir(parents=True, exist_ok=True)
    folder = take/f'{len(list(take.glob("[0-9][0-9][0-9]")))+1:03d}'
    folder.mkdir()
    gains = {f'ctrlMel.{k}': float(v) for k, v in firmware['ctrlMel'].items() if k in ('kR_xy', 'kw_xy', 'kd_omega_rp', 'ki_m_xy', 'kR_z', 'kw_z')}
    t0 = time.time()
    (folder/'parameters.json').write_text(json.dumps(dict(
        settings=dict(setup=args.root.name, take='trajectory', trajectory=str(args.trajectory), synthetic=True),
        yaw_rad=0., physical={}, attitude=gains, origin_m=[0., 0., 0.],
        truth={n: float(v) for n, v in zip(space.names, truth)}, note='synthetic flight (scripts/synthetic_flight.py)'), indent=1))
    (folder/'trajectory.json').write_text(args.trajectory.read_text())        # as the flight script keeps it
    files = {n: (folder/n).open('w', buffering=1) for n in ('flight.jsonl', 'poses.jsonl', 'cloud.jsonl')}
    n = len(r['nodes'])
    trim = np.asarray(r['trim'], float)
    print(f'writing {folder} ({n} frames) ...', flush=True)
    for k in range(n):
        stamp = t0+r['t'][k]
        if args.speed > 0:
            time.sleep(max(0., t0+r['t'][k]/args.speed-time.time()))
        R = r['rotation'][k]
        pos = np.asarray(r['position'][k], float)
        files['poses.jsonl'].write(json.dumps(dict(stamp=stamp, position=pos.tolist(), quaternion_xyzw=_quaternion(R).tolist()))+'\n')
        cable = r['nodes'][k][index]+rng.normal(0., args.noise_mm/1000, (len(index), 3))
        drone = pos+DRONE_MARKERS@R.T+rng.normal(0., args.noise_mm/1000, (4, 3))
        points = [list(map(float, q))+[0, 0, 0] for q in np.vstack([drone, cable])]
        files['cloud.jsonl'].write(json.dumps(dict(stamp=stamp, points=points))+'\n')
        if k % 2 == 0:                                       # the host log at 50 Hz
            files['flight.jsonl'].write(json.dumps(dict(t=float(r['t'][k]), phase=r['phase'][k], pose_stamp=stamp,
                                                        sent_acceleration=np.asarray(r['command'][k], float).tolist(),
                                                        trim=[0., 0., float(trim[2])], battery_v=7.6, position=pos.tolist()))+'\n')
    for f in files.values():
        f.close()
    (folder/'result.json').write_text(json.dumps(dict(completed=True, failure=None, synthetic=True)))
    print('done:', folder)


if __name__ == '__main__':
    main()
