"""Run the flight estimator (deployment.observer) over recorded cable flights exactly as the flight
node will: raw receipt stamps, rigid-body pose and the unfiltered marker cloud, frame by frame.

Reported per flight: the drone velocity and body rate against non-causal central differences of
the recorded poses on the 10 ms camera clock (the estimator's lag shows as error), the share of
frames with all ten markers labelled, the cost per frame, and the policy frames themselves for
the first `--save-frames` flights (envs.observation layout; origin at the first pose).

    python scripts/replay_observer.py --output runs/calibration_2026-10-27/observer_replay.json
"""
import argparse
import json
from pathlib import Path
import sys
import time

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from aerowhip.deployment.observer import Observer, rotation_log
from aerowhip.envs.observation import frame_row
from aerowhip.fitting.calibration_set import HELD_OUT, folders
from aerowhip.physics.model_configuration import DEFAULT_MODEL, load_model_configuration
from aerowhip.physics.state import quaternion_xyzw_to_rotation_matrix

DT = .01


def recorded_frames(folder):
    """(stamp, position, rotation, cloud) per cloud frame with a pose of the same stamp."""
    poses = {round(p['stamp'], 6): p for p in map(json.loads, (folder/'poses.jsonl').read_text().splitlines())}
    for line in (folder/'cloud.jsonl').read_text().splitlines():
        cloud = json.loads(line)
        pose = poses.get(round(cloud['stamp'], 6))
        if pose is None:
            continue
        yield (float(cloud['stamp']), np.array(pose['position'], float),
               quaternion_xyzw_to_rotation_matrix(np.array(pose['quaternion_xyzw'], float)),
               np.array([q[:3] for q in cloud['points']], float).reshape(-1, 3))


def replay(folder, arc, attachment, length):
    observer = Observer(arc, attachment, dt=DT)
    estimates, positions, rotations, frames = [], [], [], []
    origin = None
    t0 = time.perf_counter()
    for stamp, p, R, cloud in recorded_frames(folder):
        e = observer.observe(stamp, p, R, cloud)
        origin = p if origin is None else origin
        frames.append(frame_row(e, origin=origin, length=length, previous_action=np.zeros(3), progress=0.))
        estimates.append((e['velocity'], e['angular_velocity'], e['labelled'].all(), e['initialized']))
        positions.append(p); rotations.append(R)
    cost = (time.perf_counter()-t0)/len(estimates)
    P, R = np.array(positions), np.array(rotations)
    v_ref = np.full(P.shape, np.nan); v_ref[1:-1] = (P[2:]-P[:-2])/(2*DT)
    w_ref = np.full(P.shape, np.nan)
    for f in range(1, len(R)-1):
        w_ref[f] = (rotation_log(R[f].T@R[f+1])-rotation_log(R[f].T@R[f-1]))/(2*DT)
    v_est = np.array([e[0] for e in estimates]); w_est = np.array([e[1] for e in estimates])
    ok = np.isfinite(v_ref).all(1)
    speed = np.linalg.norm(v_ref, axis=1)
    lags = {}
    for lag in range(4):        # estimate later than the reference by `lag` frames
        a, b = v_ref[:len(v_ref)-lag], v_est[lag:]
        m = np.isfinite(a).all(1)
        lags[lag] = float(np.sqrt(np.mean(np.sum((a[m]-b[m])**2, 1))))
    return dict(frames=len(estimates), seconds_per_frame=cost,
                velocity_rms_mps=float(np.sqrt(np.mean(np.sum((v_est[ok]-v_ref[ok])**2, 1)))),
                velocity_rms_by_lag_mps=lags,
                velocity_rms_fast_mps=float(np.sqrt(np.mean(np.sum((v_est[ok & (speed > 1)]-v_ref[ok & (speed > 1)])**2, 1))))
                if (ok & (speed > 1)).any() else None,
                body_rate_rms_radps=float(np.sqrt(np.mean(np.sum((w_est[ok]-w_ref[ok])**2, 1)))),
                all_markers_labelled=float(np.mean([e[2] for e in estimates])),
                initialized=float(np.mean([e[3] for e in estimates]))), np.array(frames)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--save-frames', type=int, default=1)
    args = parser.parse_args()
    model = load_model_configuration(DEFAULT_MODEL)
    arc = np.cumsum(model['cable']['marker_interval_lengths_m'])
    attachment = np.asarray(model['attachment']['position_tracking_m'], float)
    length = float(np.sum(model['cable']['marker_interval_lengths_m']))
    results = {}
    for k, (motion, folder) in enumerate(folders('cf_2').items()):
        (first, frames), (second, _) = replay(folder, arc, attachment, length), replay(folder, arc, attachment, length)
        same = {key: v for key, v in first.items() if key != 'seconds_per_frame'} == {key: v for key, v in second.items() if key != 'seconds_per_frame'}
        if not same:
            raise RuntimeError(f'{motion} did not reproduce')
        results[motion] = r = dict(first, held_out=motion in HELD_OUT)
        if k < args.save_frames:
            np.savez_compressed(args.output.with_name(f'observer_frames_{motion}.npz'), frames=frames)
        print(f"{motion:12s} {r['frames']:5d} frames, {r['seconds_per_frame']*1e3:.2f} ms/frame; drone velocity rms {r['velocity_rms_mps']*1e3:.0f} mm/s "
              f"(above 1 m/s {r['velocity_rms_fast_mps']*1e3 if r['velocity_rms_fast_mps'] else float('nan'):.0f}), best lag "
              f"{min(r['velocity_rms_by_lag_mps'], key=r['velocity_rms_by_lag_mps'].get)} frames; body rate rms {r['body_rate_rms_radps']:.3f} rad/s; "
              f"all ten markers labelled {r['all_markers_labelled']*100:.1f} % of frames", flush=True)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(dict(model=str(DEFAULT_MODEL), reference='central differences of the recorded poses, 10 ms clock',
                                           reproduced_twice=True, flights=results), indent=1))


if __name__ == '__main__':
    main()
