"""OptiTrack sensing statistics of the 2026-10-27 cable flights (identification of the
environment's sensing model): camera-level marker dropout by speed, label-level statistics of
the live filter, static noise, timing. Writes one JSON with every flight and the pooled values.

    python scripts/identify_sensing.py --output runs/calibration_2026-10-27/sensing_statistics.json
"""
import argparse
import json
from pathlib import Path
import sys

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from aerowhip.fitting.calibration_set import folders
from aerowhip.fitting.sensing_statistics import SPEED_BINS, flight_statistics


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--labels', default='cable_labels.npz', help='cable_labels.npz (sequence labels) or cable_labels_filter.npz')
    args = parser.parse_args()
    flights = {}
    for motion, folder in folders().items():
        flights[motion] = s = flight_statistics(folder, args.labels)
        print(f"{motion:12s} frames {s['camera']['frames']:5d}  in-reach <10: {s['camera']['frames_in_reach_lt10']*100:5.2f} %  "
              f"label gap frames: {s['labels_statistics']['frames_with_gap']*100:5.2f} %  "
              f"noise cable {np.mean(s['noise']['cable_std_m'])*1e3:.2f} mm", flush=True)
    # pooled camera-level hazard per marker: missing fraction P -> hazard h = P p / (1 - P) with the
    # pooled label-level reappearance p (the camera-level run length is not observable)
    frames = np.zeros(len(SPEED_BINS)); missing = np.zeros(len(SPEED_BINS))
    for s in flights.values():
        for i, row in enumerate(s['camera']['by_speed']):
            if row['missing_per_marker'] is not None:
                frames[i] += row['frames']; missing[i] += row['missing_per_marker']*row['frames']
    P = missing/np.maximum(frames, 1)
    hidden = sum(s['labels_statistics']['label_transitions']['hidden_frames'] for s in flights.values())
    reappear = sum(s['labels_statistics']['label_transitions']['hidden_to_visible_per_frame']
                   *s['labels_statistics']['label_transitions']['hidden_frames'] for s in flights.values())/max(hidden, 1)
    hazard = P*reappear/np.maximum(1-P, 1e-9)
    pooled = dict(speed_bins_mps=list(SPEED_BINS), frames=frames.astype(int).tolist(), missing_per_marker=P.round(6).tolist(),
                  label_reappear_per_frame=float(reappear), hazard_per_frame=hazard.round(6).tolist(),
                  cable_noise_std_m=float(np.mean([np.mean(s['noise']['cable_std_m']) for s in flights.values()])),
                  drone_noise_std_m=float(np.mean([np.mean(s['noise']['drone_std_m']) for s in flights.values()])))
    print('pooled hazard per marker per frame by speed bin', SPEED_BINS, ':', pooled['hazard_per_frame'])
    print('label reappearance per frame', round(reappear, 4), '; cable noise', round(pooled['cable_noise_std_m']*1e3, 3), 'mm')
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(dict(data='calibration flights 2026-10-27, setup cf_2', flights=flights, pooled=pooled), indent=1))


if __name__ == '__main__':
    main()
