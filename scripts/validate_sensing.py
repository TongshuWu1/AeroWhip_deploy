"""Score the environment's OptiTrack model against the recordings: on each cable flight the
sensor synthesizes clouds from the reconstructed cable (measured markers, reconstructed where
hidden) and the recorded drone pose, the live filter labels them, and the label statistics are
compared with the same filter run over the real clouds (on their recorded stamps). The
synthetic path uses a uniform 10 ms clock; the real one the recorded stamps. Run twice.

    python scripts/validate_sensing.py --output runs/calibration_2026-10-27/sensing_validation.json
"""
import argparse
import json
from pathlib import Path
import sys

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from aerowhip.deployment.cable_filter import CableFilter
from aerowhip.envs.sensing import OptiTrackSensor
from aerowhip.experiment import DEFAULT, load
from aerowhip.fitting.calibration_set import folders
from aerowhip.fitting.sensing_statistics import DT, label_statistics, marker_speeds

LENGTH_TOLERANCE_M = .015   # the live filter's length tolerance
from aerowhip.takes.cable_labels import load_flight


def synthetic(folder, config, arc, attachment, seed):
    flight = load_flight(folder)
    rec = np.load(folder/'cable_reconstruction.npz')
    positions = rec['positions']
    ok = np.isfinite(positions).all((1, 2))
    # A usable 'truth' frame is a physically possible cable: every link at most the length tolerance
    # longer than its interval (reconstructed markers past the last seen one can violate it).
    chain = np.concatenate([flight.root[:, None], np.nan_to_num(positions)], 1)
    excess = np.linalg.norm(np.diff(chain, axis=1), axis=2)-np.diff(np.r_[0., arc])
    ok &= (excess <= LENGTH_TOLERANCE_M).all(1)
    # the longest run of complete, possible reconstructed frames is the simulated 'truth'
    runs, start = [], None
    for f, good in enumerate(np.r_[ok, False]):
        if good and start is None:
            start = f
        elif not good and start is not None:
            runs.append((start, f)); start = None
    a, b = max(runs, key=lambda r: r[1]-r[0])
    markers = positions[a:b]
    velocity = np.gradient(markers, DT, axis=0)
    sensor = OptiTrackSensor(config, arc, attachment, np.random.default_rng(seed))
    sensor.reset()
    labelled, confidence = [], []
    for i in range(b-a):
        out = sensor.observe(i, flight.drone[a+i], flight.rotation[a+i], np.zeros(3), np.zeros(3), markers[i], velocity[i])
        labelled.append(out['labelled'].copy()); confidence.append(out['confidence'].copy())
    labelled = np.array(labelled)
    stats = label_statistics(labelled, np.linalg.norm(velocity, axis=2))
    stats.update(frames_range=[int(a), int(b)], mean_confidence=float(np.mean(confidence)), latency_frames=sensor.latency)
    return stats, (a, b)


def real(folder, span):
    """The current live filter run over the recorded cloud (on its recorded stamps)."""
    flight = load_flight(folder)
    filt = CableFilter(flight.arc)
    index, confidence = [], []
    for f in range(flight.frames):
        state = filt.observe(float(flight.stamps[f]), flight.root[f], flight.points[f])
        index.append(state['index'].copy()); confidence.append(state['confidence'].copy())
    labels = np.load(folder/'cable_labels.npz')
    a, b = span
    seen = np.array(index)[a:b] >= 0
    stats = label_statistics(seen, marker_speeds(labels['positions'])[a:b])
    stats['mean_confidence'] = float(np.mean(np.array(confidence)[a:b]))
    return stats


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--experiment', type=Path, default=DEFAULT)
    parser.add_argument('--seed', type=int, default=0)
    args = parser.parse_args()
    experiment, snapshot, _ = load(args.experiment)
    model = snapshot['model']
    arc = np.cumsum(model['cable']['marker_interval_lengths_m'])
    attachment = np.asarray(model['attachment']['position_tracking_m'], float)
    config = experiment.environment.sensing
    results = {}
    for motion, folder in folders('cf_2').items():
        if not (folder/'cable_reconstruction.npz').exists():
            print(motion, 'skipped: no reconstruction'); continue
        passes = [synthetic(folder, config, arc, attachment, args.seed) for _ in range(2)]
        if passes[0] != passes[1]:
            raise RuntimeError('Synthetic sensing did not reproduce')
        sim, span = passes[0]
        rec = real(folder, span)
        results[motion] = dict(synthetic=sim, real=rec)
        print(f"{motion:12s} frames {span[1]-span[0]:5d}  labelled fraction sim {np.mean(sim['labelled_fraction']):.4f} real {np.mean(rec['labelled_fraction']):.4f}"
              f"  gap frames sim {sim['frames_with_gap']*100:5.2f} % real {rec['frames_with_gap']*100:5.2f} %"
              f"  mean run sim {sim['unlabelled_runs']['mean_frames']:.2f} real {rec['unlabelled_runs']['mean_frames']:.2f}"
              f"  confidence sim {sim['mean_confidence']:.4f} real {rec['mean_confidence']:.4f}", flush=True)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(dict(experiment=str(args.experiment), seed=args.seed,
        sensing=dict(latency_frames=list(config.latency_frames), position_noise_m=config.position_noise_m,
                     dropout_hazard=[list(r) for r in config.dropout_hazard], reappear_probability=config.reappear_probability),
        flights=results), indent=1))


if __name__ == '__main__':
    main()
