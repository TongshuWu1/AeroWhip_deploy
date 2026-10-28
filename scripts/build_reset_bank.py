"""Reference states for training: native initial states sampled along the recorded scripted whip
flights, for reference-state starts (a fraction of the training lanes start mid-whip instead of
from hover; docs/strike_task.md).

Every 0.1 s inside the whip strokes of the cable flights whose manifest take is `whip`, a window
with every marker visible gives the measured drone pose, the velocities from the causal history
and the cable projected onto the configured lengths (the same construction as the cable replays);
the state is moved so the flight's first hover position is at the origin and rotated about the
vertical so that hover yaw is zero (the training scene's frame).

    python scripts/build_reset_bank.py --output runs/reset_bank/whip.npz
"""
import argparse
import json
import math
from pathlib import Path
import sys
import time

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from aerowhip.fitting import flight_drone
from aerowhip.fitting.calibration_set import flight_windows, folders
from aerowhip.fitting.flight_cable import endpoint_velocity
from aerowhip.physics.model_configuration import DEFAULT_MODEL, load_model_configuration
from aerowhip.takes.calibration_flights import load


def rotation_z(angle):
    c, s = math.cos(angle), math.sin(angle)
    return np.array([[c, -s, 0.], [s, c, 0.], [0., 0., 1.]])


def hover_origin(flight):
    """Tracking position and yaw averaged over the flight's first hover phase."""
    phases = np.array([flight.phase_at(s) for s in flight.stamp])
    hover = np.flatnonzero(np.char.startswith(phases.astype(str), 'hover'))
    if not len(hover):
        raise ValueError(f'{flight.folder}: no hover phase')
    first = hover[(np.diff(np.r_[hover[0]-1, hover]) == 1).cumprod().astype(bool)]
    position = flight.position[first].mean(0)
    yaw = np.mean([math.atan2(2*(q[3]*q[2]+q[0]*q[1]), 1-2*(q[1]**2+q[2]**2)) for q in flight.quaternion[first]])
    return position, float(yaw)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--stride', type=float, default=.1, help='seconds between sampled states')
    parser.add_argument('--phases', nargs='+', default=['whip'], help='log phase prefixes to sample inside')
    args = parser.parse_args()
    t0 = time.time()
    config = load_model_configuration(DEFAULT_MODEL)
    replay = flight_drone.DroneReplay(config, flight_drone.firmware_config())
    native_model = replay.model.native
    states, sources, times, tip_speeds = [], [], [], []
    for name, folder in folders().items():
        if not name.startswith('whip'):
            continue
        flight = load(folder)
        origin, yaw = hover_origin(flight)
        Rz = rotation_z(-yaw)
        for w in flight_windows(folder, duration_s=args.stride, stride_s=args.stride):
            if not any(w.phase.startswith(p) for p in args.phases):
                continue
            hp, hR, ht = w.history_position, w.history_rotation, w.history_time
            v, omega = endpoint_velocity(ht, hp), flight_drone.body_rate(ht, hR)
            q, qv = replay.initial_cable(w, v, omega)
            p, R = hp[-1], hR[-1]
            # into the training frame: hover origin at zero, hover yaw zero
            p, R, v = Rz@(p-origin), Rz@R, Rz@v
            q, qv = (q-origin)@Rz.T, qv@Rz.T
            try:
                state = native_model.initial_state(p, R, v, omega, q, qv)
            except Exception as exc:        # a cable shape the engine rejects is not a usable start
                print(f'  skipped {name} t={w.start_stamp:.2f}: {exc}', flush=True)
                continue
            states.append(np.asarray(state, float))
            sources.append(f'{name}@{w.start_stamp:.3f}')
            times.append(float(w.start_stamp))
            tip_speeds.append(float(np.linalg.norm(endpoint_velocity(ht, w.history_markers[:, -1]))))
        print(f'{name}: {sum(s.startswith(name) for s in sources)} states', flush=True)
    if not states:
        raise SystemExit('No reference states')
    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.savez(args.output, states=np.array(states), source=np.array(sources), stamp=np.array(times),
             tip_speed_mps=np.array(tip_speeds), model=np.array(json.dumps(dict(
                 cable=config['cable']['marker_interval_lengths_m'], attachment=config['attachment']['position_tracking_m']))),
             built=np.array(time.strftime('%Y-%m-%d %H:%M:%S')))
    print(f'{len(states)} states from {len(set(s.split("@")[0] for s in sources))} flights, tip speed up to '
          f'{max(tip_speeds):.1f} m/s, written to {args.output} ({time.time()-t0:.0f} s)')


if __name__ == '__main__':
    main()
