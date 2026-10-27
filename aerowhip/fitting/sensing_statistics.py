"""OptiTrack sensing statistics from recorded cable flights (identification of envs.sensing).

Two levels are distinguished. Camera level: how many points lie within the cable's reach of the
attachment, against the ten markers; a shortfall is a marker the cameras did not report
(labelling cannot create or remove points). Label level: which markers the live filter
labelled in each frame, run lengths of unlabelled stretches, confidences; this is what the
policy experiences and what the synthetic clouds are scored on (`scripts/validate_sensing.py`).
Static position noise is the second difference of markers at rest (white part only; slow
camera drift is invisible to it). Receipt jitter is the spread of host receipt minus frame stamp.
"""
import json

import numpy as np

from aerowhip.takes.cable_labels import load_flight

DT = .01
SPEED_BINS = (0., .25, .5, 1., 2., 4., 8.)
REACH_MARGIN_M = .03
REST_SPEED_MPS = .05


def marker_speeds(positions):
    """(F, 10) speed from central differences of labelled positions, NaN where not available."""
    v = np.full(positions.shape, np.nan)
    v[1:-1] = (positions[2:]-positions[:-2])/(2*DT)
    return np.linalg.norm(v, axis=2)


def run_lengths(seen):
    """Lengths (frames) of every unlabelled stretch in a (F,) boolean sequence."""
    runs, r = [], 0
    for x in seen:
        if x:
            if r:
                runs.append(r)
            r = 0
        else:
            r += 1
    if r:
        runs.append(r)
    return runs


def label_statistics(seen, speeds=None):
    """Per-marker labelled fraction, unlabelled run lengths, label transitions by speed bin."""
    seen = np.asarray(seen, bool)
    out = dict(frames=int(len(seen)), labelled_fraction=seen.mean(0).round(4).tolist(),
               frames_with_gap=float((~seen.all(1)).mean()))
    runs = np.array([r for k in range(seen.shape[1]) for r in run_lengths(seen[:, k])])
    out['unlabelled_runs'] = dict(count=int(len(runs)), mean_frames=float(runs.mean()) if len(runs) else 0.,
                                  histogram_1_to_10_and_over=[int((runs == i).sum()) for i in range(1, 11)]+[int((runs > 10).sum())])
    if speeds is not None:
        bins = np.array(SPEED_BINS)
        b = np.searchsorted(bins, np.nan_to_num(speeds[:-1], nan=0.), 'right')-1
        hide, visible, reappear, hidden = np.zeros(len(bins)), np.zeros(len(bins)), 0, 0
        was, now = seen[:-1], seen[1:]
        for i in range(len(bins)):
            m = (b == i) & was
            visible[i], hide[i] = m.sum(), (m & ~now).sum()
        hidden, reappear = (~was).sum(), (~was & now).sum()
        out['label_transitions'] = dict(speed_bins_mps=list(SPEED_BINS),
            visible_to_hidden_per_frame=(hide/np.maximum(visible, 1)).round(5).tolist(), visible_frames=visible.astype(int).tolist(),
            hidden_to_visible_per_frame=float(reappear/max(hidden, 1)), hidden_frames=int(hidden))
    return out


def camera_statistics(flight, labels):
    """Points within reach of the attachment against the ten markers, by the fastest marker's speed."""
    positions = labels['positions']
    reach = flight.arc[-1]+REACH_MARGIN_M
    fastest = np.nanmax(marker_speeds(positions), axis=1, initial=0.)
    inreach = np.array([int((np.linalg.norm(pts-flight.root[f], axis=1) <= reach).sum()) for f, pts in enumerate(flight.points)])
    bins = np.array(SPEED_BINS)
    b = np.searchsorted(bins, fastest, 'right')-1
    missing = np.maximum(0, 10-inreach)
    rows = []
    for i in range(len(bins)):
        m = b == i
        rows.append(dict(speed_from_mps=float(bins[i]), frames=int(m.sum()),
                         missing_per_marker=float(missing[m].mean()/10) if m.any() else None,
                         frames_with_missing=float((missing[m] > 0).mean()) if m.any() else None))
    return dict(frames=int(len(inreach)), points_in_reach_mean=float(inreach.mean()),
                frames_in_reach_lt10=float((inreach < 10).mean()), frames_in_reach_gt10=float((inreach > 10).mean()),
                by_speed=rows)


def noise_statistics(flight, labels):
    """White position noise (per axis std, m) of markers and the drone at rest, by second differences."""
    pos, speed = labels['positions'], marker_speeds(labels['positions'])
    cable = []
    for k in range(pos.shape[1]):
        ok = np.isfinite(pos[:, k]).all(1)
        for f in range(2, len(pos)-2):
            if ok[f-2:f+3].all() and np.nanmax(speed[f-2:f+3, k]) < REST_SPEED_MPS:
                cable.append((pos[f+1, k]-2*pos[f, k]+pos[f-1, k])/np.sqrt(6))
    d = flight.drone
    vd = np.linalg.norm(np.gradient(d, DT, axis=0), axis=1)
    drone = [(d[f+1]-2*d[f]+d[f-1])/np.sqrt(6) for f in range(2, len(d)-2) if vd[f-2:f+3].max() < REST_SPEED_MPS]
    cable, drone = np.array(cable).reshape(-1, 3), np.array(drone).reshape(-1, 3)
    return dict(cable_rest_samples=int(len(cable)), cable_std_m=cable.std(0).tolist() if len(cable) else None,
                drone_rest_samples=int(len(drone)), drone_std_m=drone.std(0).tolist() if len(drone) else None)


def timing_statistics(folder):
    cloud = [json.loads(line) for line in (folder/'cloud.jsonl').read_text().splitlines()]
    stamps = np.array([c['stamp'] for c in cloud])
    jitter = np.array([c['received']-c['stamp'] for c in cloud])
    jitter -= np.median(jitter)
    return dict(stamp_step_ms_1_50_99=np.percentile(np.diff(stamps)*1e3, [1, 50, 99]).round(2).tolist(),
                receipt_jitter_ms_1_50_99=np.percentile(jitter*1e3, [1, 50, 99]).round(2).tolist(),
                note='absolute capture-to-host latency is not measurable from these files')


def flight_statistics(folder, labels_file='cable_labels.npz'):
    flight = load_flight(folder)
    labels = np.load(folder/labels_file)
    if not np.allclose(labels['stamps'], flight.stamps):
        raise ValueError('Labels and cloud frames differ')
    seen = labels['index'] >= 0
    return dict(folder=str(folder), labels=labels_file, camera=camera_statistics(flight, labels),
                labels_statistics=label_statistics(seen, marker_speeds(labels['positions'])),
                noise=noise_statistics(flight, labels), timing=timing_statistics(folder))
