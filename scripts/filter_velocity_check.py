"""Score the live cable filter's marker velocities on the recorded clouds.

The filter (`deployment.cable_filter.CableFilter`, causal, as it runs in flight and in the training
environment) is run over every cable flight's recorded cloud. Its velocity estimate is compared
with a non-causal reference: central differences of the whole-recording labels
(`cable_labels.npz`) on the uniform 100 Hz frame clock, where the marker and both neighbours are
labelled. Reported per flight and per marker: RMS velocity error, its lag (the shift of the
filter's tip speed against the reference that best matches), and at every snap (reference tip
speed peak above `--snap` m/s, fully labelled five frames either side) the ratio of the filter's peak to the reference peak and the
delay between them. Every number is computed twice.

    python scripts/filter_velocity_check.py --output runs/calibration_2026-10-27/filter_velocity_check.json
"""
import argparse
import json
from pathlib import Path
import sys

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from aerowhip.deployment.cable_filter import CableFilter, MARKERS
from aerowhip.fitting.calibration_set import is_held_out, folders
from aerowhip.takes.cable_labels import load_flight

DT = .01


def run_filter(flight):
    filt = CableFilter(flight.arc)
    velocity, seen = np.full((flight.frames, MARKERS, 3), np.nan), np.zeros((flight.frames, MARKERS), bool)
    for f in range(flight.frames):
        state = filt.observe(float(flight.stamps[f]), flight.root[f], flight.points[f])
        if state['initialized']:
            velocity[f] = state['velocities']
            seen[f] = state['index'] >= 0
    return velocity, seen


def reference(positions):
    v = np.full(positions.shape, np.nan)
    v[1:-1] = (positions[2:]-positions[:-2])/(2*DT)
    return v


def snaps(speed, threshold, separation=30, cover=5):
    """Frames of reference tip-speed peaks above threshold, at least `separation` apart, where the
    reference covers `cover` frames on both sides and the frame is the maximum of that span (a peak
    inside a labelling gap has no reference and is not scored)."""
    peaks = []
    order = np.argsort(np.nan_to_num(speed, nan=0.))[::-1]
    for f in order:
        if not np.isfinite(speed[f]) or speed[f] < threshold:
            break
        span = speed[max(0, f-cover):f+cover+1]
        if f < cover or len(span) < 2*cover+1 or not np.isfinite(span).all() or span.max() > speed[f]:
            continue
        if all(abs(f-p) >= separation for p in peaks):
            peaks.append(int(f))
    return sorted(peaks)


def flight_check(folder, threshold, clock):
    flight = load_flight(folder)
    labels = np.load(folder/'cable_labels.npz')
    if not np.allclose(labels['stamps'], flight.stamps):
        raise ValueError('Labels and cloud frames differ')
    if clock == 'uniform':      # the camera frame clock; the recorded stamps are host receipt times
        flight.stamps = np.arange(flight.frames)*DT
    estimate, seen = run_filter(flight)
    truth = reference(labels['positions'])
    ok = np.isfinite(truth).all(2) & np.isfinite(estimate).all(2)
    error = np.linalg.norm(estimate-truth, axis=2)
    per_marker = [float(np.sqrt(np.mean(error[ok[:, k], k]**2))) for k in range(MARKERS)]
    tip_true, tip_est = np.linalg.norm(truth[:, -1], axis=1), np.linalg.norm(estimate[:, -1], axis=1)
    both = np.isfinite(tip_true) & np.isfinite(tip_est)
    lags = {}
    for lag in range(0, 8):          # filter later than the reference by `lag` frames
        a, b = tip_true[:len(tip_true)-lag], tip_est[lag:]
        m = np.isfinite(a) & np.isfinite(b)
        lags[lag] = float(np.sqrt(np.mean((a[m]-b[m])**2)))
    best_lag = min(lags, key=lags.get)
    events = []
    for f in snaps(np.where(both, tip_true, np.nan), threshold):
        lo, hi = max(0, f-10), min(len(tip_est), f+11)
        window = tip_est[lo:hi]
        if not np.isfinite(window).any():
            continue
        g = lo+int(np.nanargmax(window))
        events.append(dict(frame=f, reference_peak_mps=float(tip_true[f]), filter_peak_mps=float(tip_est[g]),
                           ratio=float(tip_est[g]/tip_true[f]), delay_frames=int(g-f),
                           tip_labelled_by_filter=bool(seen[f, -1])))
    return dict(frames=int(flight.frames), compared_marker_frames=int(ok.sum()),
                per_marker_rms_mps=per_marker, rms_mps=float(np.sqrt(np.mean(error[ok]**2))),
                tip_speed_rms_by_lag_mps=lags, best_lag_frames=best_lag, snaps=events)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--snap', type=float, default=3., help='reference tip speed (m/s) above which a peak is a snap')
    parser.add_argument('--clock', choices=('uniform', 'stamps'), default='uniform',
                        help='filter time: the 10 ms camera frame clock, or the recorded (host receipt) stamps')
    args = parser.parse_args()
    results = {}
    for motion, folder in folders().items():
        first, second = flight_check(folder, args.snap, args.clock), flight_check(folder, args.snap, args.clock)
        if json.dumps(first) != json.dumps(second):
            raise RuntimeError(f'{motion} did not reproduce')
        results[motion] = r = dict(first, held_out=is_held_out(folder))
        ratios = [e['ratio'] for e in r['snaps']]
        delays = [e['delay_frames'] for e in r['snaps']]
        print(f"{motion:12s} velocity rms {r['rms_mps']:.3f} m/s (tip {r['per_marker_rms_mps'][-1]:.3f}), best lag {r['best_lag_frames']} frames; "
              f"{len(ratios)} snaps > {args.snap:g} m/s" + (f": peak ratio median {np.median(ratios):.2f} (min {min(ratios):.2f}), "
              f"delay median {np.median(delays):.0f} frames (max {max(delays)})" if ratios else ''), flush=True)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(dict(snap_threshold_mps=args.snap, clock=args.clock, reference='central differences of cable_labels.npz, uniform 10 ms clock',
                                           reproduced_twice=True, flights=results), indent=1))


if __name__ == '__main__':
    main()
