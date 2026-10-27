"""Cable-marker labels from a recorded OptiTrack marker cloud.

A calibration flight folder holds `cloud.jsonl` (every OptiTrack marker cloud during the
recorded part) and `poses.jsonl` (the drone pose of the same frames, same stamps). This
module removes the drone's own markers, merges duplicate points, and associates the
remaining points with the cable's material markers c1..c10 using the geometric/temporal
CableTracker. Manual labels (from the review GUI) are hard constraints: the tracker is
re-seeded from them and re-run forward. Labels are written to `cable_labels.npz`.

Label sources: 0 none, 1 automatic, 2 manual. A manual label with point -1 means
"this marker is not visible in this frame". Reconstruction of markers OptiTrack did not
see is a separate step (`reconstruct`), stored apart from the measured labels.
"""
from dataclasses import dataclass, field
import json
from pathlib import Path

import numpy as np

from aerowhip.deployment.cable_tracking import CableTracker, TrackingSettings
from aerowhip.paths import ROOT
from aerowhip.physics.state import quaternion_xyzw_to_rotation_matrix

MODEL = ROOT/'configs/physics/drone_cable.json'
TRACKING = ROOT/'configs/ros/cable_tracking.json'
DRONE_RADIUS_M = .07        # points this close to the tracking origin are the drone's markers
DUPLICATE_RADIUS_M = .01    # points this close together are one marker
LOW_CONFIDENCE = .3
LENGTH_TOLERANCE_M = .02
JUMP_M = .08                # a label moving further than this between two frames is flagged
GAP_POINT_M = .04           # an unlabelled point this close to a hidden marker's expected position is flagged
START_TOLERANCE_M = .02     # start: marker k at arc[k] (+-this) from the attachment
NONE, AUTO, MANUAL = 0, 1, 2
LABELS_FILE = 'cable_labels.npz'


@dataclass
class Flight:
    folder: Path
    stamps: np.ndarray          # (F,) OptiTrack stamps, s
    points: list                # per frame: (N, 3) cable candidate points, world, m
    drone: np.ndarray           # (F, 3) tracking origin
    rotation: np.ndarray        # (F, 3, 3) drone body to world
    root: np.ndarray            # (F, 3) cable attachment
    phases: list                # per frame: flight phase of the nearest 50 Hz log line
    times: np.ndarray           # per frame: flight time t of the nearest 50 Hz log line, s
    arc: np.ndarray             # (10,) marker distances along the cable from the attachment

    @property
    def frames(self):
        return len(self.stamps)


@dataclass
class Labels:
    index: np.ndarray           # (F, 10) point index in Flight.points[f], -1 none
    source: np.ndarray          # (F, 10) NONE / AUTO / MANUAL
    confidence: np.ndarray      # (F, 10) tracker separation heuristic, 0..1
    manual: dict = field(default_factory=dict)   # {(frame, marker): point index or -1}
    reviewed: set = field(default_factory=set)   # frames whose flags were checked and accepted

    def positions_of(self, flight, frame):
        out = np.full((len(flight.arc), 3), np.nan)
        for k, i in enumerate(self.index[frame]):
            if i >= 0:
                out[k] = flight.points[frame][i]
        return out

    def positions(self, flight):
        out = np.full((flight.frames, len(flight.arc), 3), np.nan)
        for f in range(flight.frames):
            for k, i in enumerate(self.index[f]):
                if i >= 0:
                    out[f, k] = flight.points[f][i]
        return out


def _model():
    model = json.loads(MODEL.read_text())
    return (np.cumsum(model['cable']['marker_interval_lengths_m']),
            np.array(model['attachment']['position_tracking_m'], float))


def load_flight(folder):
    """Cable candidates per cloud frame, with the drone pose of the same frame."""
    folder = Path(folder)
    arc, attachment = _model()
    poses = {round(p['stamp'], 6): p for p in map(json.loads, (folder/'poses.jsonl').read_text().splitlines())}
    log = [json.loads(line) for line in (folder/'flight.jsonl').read_text().splitlines()]
    log_stamps = np.array([r['pose_stamp'] if r.get('pose_stamp') is not None else np.nan for r in log])
    stamps, points, drone, rotation, root, phases, times = [], [], [], [], [], [], []
    for line in (folder/'cloud.jsonl').read_text().splitlines():
        cloud = json.loads(line)
        pose = poses.get(round(cloud['stamp'], 6))
        if pose is None:
            continue
        p = np.array(pose['position'], float)
        R = quaternion_xyzw_to_rotation_matrix(np.array(pose['quaternion_xyzw'], float))
        xyz = np.array([q[:3] for q in cloud['points']], float).reshape(-1, 3)
        xyz = xyz[np.linalg.norm(xyz-p, axis=1) > DRONE_RADIUS_M]
        kept = []
        for q in xyz:  # merge duplicates: keep the first of points closer than DUPLICATE_RADIUS_M
            if not kept or np.min(np.linalg.norm(np.array(kept)-q, axis=1)) > DUPLICATE_RADIUS_M:
                kept.append(q)
        stamps.append(cloud['stamp'])
        points.append(np.array(kept).reshape(-1, 3))
        drone.append(p)
        rotation.append(R)
        root.append(p+R@attachment)
        nearest = int(np.nanargmin(np.abs(log_stamps-cloud['stamp']))) if np.isfinite(log_stamps).any() else 0
        phases.append(log[nearest]['phase'])
        times.append(log[nearest]['t'])
    return Flight(folder, np.array(stamps), points, np.array(drone), np.array(rotation), np.array(root), phases,
                  np.array(times), arc)


def _tracker(arc):
    values = json.loads(TRACKING.read_text())
    values['marker_distances_m'] = arc.tolist()
    return CableTracker(TrackingSettings(**values))


def _state(tracker):
    return (tracker.positions.copy(), tracker.velocities.copy(), tracker.observed_at.copy(),
            tracker.initialized, tracker.previous_time)


def _restore(tracker, state):
    tracker.positions, tracker.velocities, tracker.observed_at = (x.copy() for x in state[:3])
    tracker.initialized, tracker.previous_time = state[3], state[4]


START_MINIMUM_MARKERS = 6   # OptiTrack does not see every marker in every frame


def hanging_start(flight, frame):
    """Labels for a cable at rest: marker k is the one point at distance arc[k] from the
    attachment (within START_TOLERANCE_M) and below it. Markers OptiTrack does not see stay
    unlabelled (-1); at least START_MINIMUM_MARKERS must be found, none may be ambiguous,
    and consecutive found markers must be no further apart than the cable between them.
    Works for a cable that sways; None if not usable."""
    rel = flight.points[frame]-flight.root[frame]
    if not len(rel):
        return None
    distance = np.linalg.norm(rel, axis=1)
    index = np.full(len(flight.arc), -1)
    for k, arc in enumerate(flight.arc):
        candidates = np.flatnonzero((np.abs(distance-arc) < START_TOLERANCE_M) & (rel[:, 2] < 0))
        if len(candidates) > 1:
            return None
        if len(candidates):
            index[k] = candidates[0]
    found = np.flatnonzero(index >= 0)
    if len(found) < START_MINIMUM_MARKERS or len(set(index[found])) != len(found):
        return None
    arc = np.r_[0., flight.arc]
    chain = np.vstack([np.zeros(3), rel[index[found]]])
    gaps = np.diff(np.r_[0., arc[found+1]])
    if np.any(np.linalg.norm(np.diff(chain, axis=0), axis=1) > gaps+LENGTH_TOLERANCE_M):
        return None
    return index


class AutoLabeller:
    """Forward tracker run with manual constraints; keeps per-frame tracker state so a
    correction re-runs only from the corrected frame."""

    def __init__(self, flight, labels=None):
        self.flight = flight
        n, m = flight.frames, len(flight.arc)
        self.labels = labels or Labels(np.full((n, m), -1), np.zeros((n, m), np.uint8), np.zeros((n, m)))
        self.tracker = _tracker(flight.arc)
        self.states = [None]*n          # tracker state before each frame

    def run(self, start=0, converge=False):
        """Label frames from `start` on. With converge=True, stop once the labels and tracker
        state match the previous run again (corrections usually rejoin within a few frames)."""
        f, tr, lab = self.flight, self.tracker, self.labels
        if start == 0 or self.states[start] is None:
            start = 0
            tr.reset()
        else:
            _restore(tr, self.states[start])
        manual_by_frame = {}
        for (frame, marker), point in lab.manual.items():
            manual_by_frame.setdefault(frame, {})[marker] = point
        for i in range(start, f.frames):
            old = self.states[i]
            if converge and i > start+5 and old is not None and _same(old, _state(tr)):
                return self.labels  # later frames, including later manual labels, are unchanged
            self.states[i] = _state(tr)
            before = _state(tr)
            seed = None if tr.initialized else hanging_start(f, i)
            if seed is not None:
                tr.previous_time = float(f.stamps[i])
                index, confidence = seed.copy(), (seed >= 0).astype(float)
            else:
                result = tr.observe(float(f.stamps[i]), f.root[i], f.points[i])
                index = np.array(result['point_indices'], int)
                confidence = np.array(result['confidence'], float)
            source = np.where(index >= 0, AUTO, NONE).astype(np.uint8)
            manual = manual_by_frame.get(i, {})
            for marker, point in manual.items():
                if point >= 0:
                    index[(index == point) & (np.arange(len(index)) != marker)] = -1
                index[marker], source[marker], confidence[marker] = point, MANUAL, 1.
            source[index < 0] = np.where(np.isin(np.flatnonzero(index < 0), list(manual)), MANUAL, NONE)
            if manual or seed is not None:
                self._apply(before, i, index, manual)
            lab.index[i], lab.source[i], lab.confidence[i] = index, source, confidence
        return self.labels

    def _apply(self, before, frame, index, manual):
        """Re-seed the tracker from the corrected labels of this frame."""
        tr, f = self.tracker, self.flight
        positions, velocities, observed = (x.copy() for x in before[:3])
        for marker, point in enumerate(index):
            if point < 0:
                continue
            p = f.points[frame][point]
            if np.isfinite(observed[marker]) and f.stamps[frame] > observed[marker]:
                velocities[marker] = (p-positions[marker])/(f.stamps[frame]-observed[marker])
            positions[marker], observed[marker] = p, f.stamps[frame]
        for marker, point in manual.items():
            if point < 0:  # explicitly not visible: keep the prediction from before this frame
                positions[marker], velocities[marker], observed[marker] = before[0][marker], before[1][marker], before[2][marker]
        tr.positions, tr.velocities, tr.observed_at = positions, velocities, observed
        tr.initialized = True
        tr.previous_time = float(f.stamps[frame])

    def _ready(self, frame):
        """Tracker states are rebuilt on the first edit after loading saved labels."""
        if self.states[frame] is None:
            self.run()

    def set_manual(self, frame, marker, point):
        """Label `point` (index in frame's points, or -1 = not visible) as `marker`; re-run."""
        self._ready(frame)
        lab = self.labels
        if point >= 0:
            for other in [k for (fr, k), p in lab.manual.items() if fr == frame and p == point and k != marker]:
                del lab.manual[(frame, other)]
        lab.manual[(frame, marker)] = point
        return self.run(frame, converge=True)

    def clear_manual(self, frame, marker):
        self._ready(frame)
        self.labels.manual.pop((frame, marker), None)
        return self.run(frame, converge=True)


def _same(a, b):
    """Tracker states equal to rounding (velocity history decays by ~10x per frame)."""
    return (a[3] == b[3] and a[4] == b[4] and np.array_equal(a[2], b[2]) and
            np.allclose(a[0], b[0], rtol=0, atol=1e-9) and np.allclose(a[1], b[1], rtol=0, atol=1e-6))


def estimate(flight, labels, frame):
    """Display-only positions for the markers missing in `frame`: interpolated along the cable
    between the nearest labelled neighbours (the attachment counts as one); markers past the
    last labelled one continue its direction. Labelled markers are returned as measured.
    Never saved: the labels keep gaps where a marker was not seen."""
    arc = np.r_[0., flight.arc]
    known = np.vstack([flight.root[frame], labels.positions_of(flight, frame)])
    ok = np.isfinite(known).all(1)
    out = known.copy()
    idx = np.flatnonzero(ok)
    for k in np.flatnonzero(~ok):
        before, after = idx[idx < k], idx[idx > k]
        a = before[-1]
        if len(after):
            b = after[0]
            u = (arc[k]-arc[a])/(arc[b]-arc[a])
            out[k] = known[a]+u*(known[b]-known[a])
        else:
            prev = before[-2] if len(before) > 1 else None
            direction = (known[a]-known[prev]) if prev is not None else np.array([0., 0., -1.])
            direction = direction/max(np.linalg.norm(direction), 1e-9)
            out[k] = known[a]+(arc[k]-arc[a])*direction
    return out[1:]


def flags(flight, labels):
    """Per frame: list of reasons to review it. A marker OptiTrack did not see is normal
    and not flagged by itself; it is flagged only when an unassigned point lies where the
    marker is expected (display estimate within GAP_POINT_M), i.e. a possibly missed label."""
    out = []
    positions = labels.positions(flight)
    for f in range(flight.frames):
        reasons = []
        index = labels.index[f]
        if (index < 0).any() and (index >= 0).any():
            free = np.array([q for i, q in enumerate(flight.points[f]) if i not in set(index[index >= 0])]).reshape(-1, 3)
            if len(free):
                expected = estimate(flight, labels, f)
                near = [k+1 for k in np.flatnonzero(index < 0)
                        if np.linalg.norm(free-expected[k], axis=1).min() < GAP_POINT_M]
                if near:
                    reasons.append('c'+', c'.join(map(str, near))+' unlabelled point nearby')
        elif not (index >= 0).any():
            reasons.append('no markers labelled')
        auto = labels.source[f] == AUTO
        if (auto & (labels.confidence[f] < LOW_CONFIDENCE)).any():
            reasons.append('low confidence')
        chain = np.vstack([flight.root[f], positions[f]])
        arc = np.r_[0., flight.arc]
        ok = np.isfinite(chain).all(1)
        idx = np.flatnonzero(ok)
        for a, b in zip(idx[:-1], idx[1:]):
            if np.linalg.norm(chain[b]-chain[a]) > arc[b]-arc[a]+LENGTH_TOLERANCE_M:
                reasons.append(f'c{b} too far from c{a}' if a else f'c{b} too far from the attachment')
                break
        if f and len(reasons) < 3:
            moved = np.linalg.norm(positions[f]-positions[f-1], axis=1)
            jumped = np.flatnonzero(np.nan_to_num(moved) > JUMP_M)
            if len(jumped):
                reasons.append('c'+', c'.join(str(k+1) for k in jumped)+' jumped')
        out.append(reasons)
    return out


def segments(reasons, reviewed=(), gap=10):
    """Runs of flagged, unreviewed frames, merging runs less than `gap` frames apart:
    [(first, last, reasons), ...]."""
    out, start, why = [], None, set()
    for f, r in enumerate(list(reasons)+[[]]):
        r = r if f not in reviewed else []
        if r and start is None:
            start, why = f, set(r)
        elif r:
            why |= set(r)
        elif start is not None:
            if out and start-out[-1][1] <= gap:
                out[-1] = (out[-1][0], f-1, sorted(set(out[-1][2]) | why))
            else:
                out.append((start, f-1, sorted(why)))
            start = None
    return out


def save(flight, labels, path=None):
    path = Path(path or flight.folder/LABELS_FILE)
    manual = np.array([(f, k, p) for (f, k), p in sorted(labels.manual.items())], int).reshape(-1, 3)
    np.savez_compressed(path, stamps=flight.stamps, positions=labels.positions(flight), index=labels.index,
                        source=labels.source, confidence=labels.confidence, manual=manual, arc=flight.arc,
                        root=flight.root, drone=flight.drone, reviewed=np.array(sorted(labels.reviewed), int))
    reasons = flags(flight, labels)
    summary = dict(frames=flight.frames, complete_frames=int((labels.index >= 0).all(1).sum()),
                   flagged_frames=int(sum(bool(r) for r in reasons)),
                   unreviewed_flagged_frames=int(sum(bool(r) and f not in labels.reviewed for f, r in enumerate(reasons))),
                   manual_labels=len(labels.manual), reviewed_frames=len(labels.reviewed),
                   per_marker_labelled_percent=[round(100*float((labels.index[:, k] >= 0).mean()), 1)
                                                for k in range(len(flight.arc))])
    path.with_suffix('.json').write_text(json.dumps(summary, indent=2)+'\n')
    return summary


def load_labels(flight, path=None):
    path = Path(path or flight.folder/LABELS_FILE)
    if not path.exists():
        return None
    data = np.load(path)
    if len(data['stamps']) != flight.frames or not np.array_equal(data['stamps'], flight.stamps):
        raise ValueError(f'{path} does not match this flight\'s cloud frames')
    manual = {(int(f), int(k)): int(p) for f, k, p in data['manual']}
    reviewed = set(int(f) for f in data['reviewed']) if 'reviewed' in data else set()
    return Labels(data['index'].copy(), data['source'].copy(), data['confidence'].copy(), manual, reviewed)


# --- Reconstruction of markers OptiTrack did not see -----------------------------------

MEASURED, TIME_GAP, BETWEEN, END = 0, 1, 2, 3   # reconstruction source per marker and frame
MAX_TIME_GAP_FRAMES = 10                         # short gaps: interpolate in time across them


def _hermite(p0, v0, p1, v1, u, span):
    """Cubic Hermite position at fraction u of a gap of `span` seconds."""
    h00, h10, h01, h11 = 2*u**3-3*u**2+1, u**3-2*u**2+u, -2*u**3+3*u**2, u**3-u**2
    return h00*p0+h10*span*v0+h01*p1+h11*span*v1


def _on_circle(a, b, ra, rb, guess):
    """Point at distance ra from a and rb from b closest to `guess`; on the segment a-b
    (split ra:rb) when the two cannot be met because the points are too far apart."""
    ab = b-a
    d = np.linalg.norm(ab)
    if d < 1e-9:
        return a+(guess-a)/max(np.linalg.norm(guess-a), 1e-9)*ra
    if d >= ra+rb:
        return a+ab*ra/(ra+rb)
    axis = ab/d
    along = (d*d+ra*ra-rb*rb)/(2*d)
    radius = np.sqrt(max(ra*ra-along*along, 0.))
    centre = a+along*axis
    offset = (guess-centre)-np.dot(guess-centre, axis)*axis
    if np.linalg.norm(offset) < 1e-9:
        offset = np.cross(axis, [0., 0., 1.]) if abs(axis[2]) < .9 else np.cross(axis, [1., 0., 0.])
    return centre+radius*offset/np.linalg.norm(offset)


def reconstruct(flight, labels, usable=None, frames=None):
    """Positions of all markers in every frame: measured where labelled, otherwise
    reconstructed. Returns (positions (F, 10, 3) with NaN where nothing is possible,
    source (F, 10) MEASURED / TIME_GAP / BETWEEN / END, -1 where nothing is possible).

    1. TIME_GAP: a marker missing for at most MAX_TIME_GAP_FRAMES frames between two
       measured frames is interpolated in time (cubic Hermite with the velocities at both
       ends, from neighbouring measured frames).
    2. BETWEEN: a marker still missing with known markers (or the attachment) on both sides
       along the cable is placed at the cable distances from its two nearest known
       neighbours, at the solution closest to its motion prediction (else the straight
       interpolation between the neighbours).
    3. END: markers past the last known one continue its direction at their cable
       distances, nudged towards their motion prediction.
    Frames with usable[f] False are left as measured only. frames=(start, stop) limits the
    work to that range (positions outside it are left as measured)."""
    start, stop = frames if frames is not None else (0, flight.frames)
    start, stop = max(int(start), 0), min(int(stop), flight.frames)
    measured = np.full((flight.frames, len(flight.arc), 3), np.nan)
    for f in range(max(start-MAX_TIME_GAP_FRAMES-2, 0), min(stop+MAX_TIME_GAP_FRAMES+2, flight.frames)):
        measured[f] = labels.positions_of(flight, f)
    pos = measured.copy()
    source = np.where(np.isfinite(measured).all(2), MEASURED, -1)
    t = flight.stamps
    markers = len(flight.arc)
    # 1. short gaps in time
    for k in range(markers):
        seen = np.flatnonzero(source[:, k] == MEASURED)
        seen = seen[(seen >= start-MAX_TIME_GAP_FRAMES-1) & (seen < stop+MAX_TIME_GAP_FRAMES+1)]
        for i, j in zip(seen[:-1], seen[1:]):
            if 1 < j-i <= MAX_TIME_GAP_FRAMES+1:
                def velocity(f, side):
                    g = f+side
                    if 0 <= g < flight.frames and source[g, k] == MEASURED:
                        return (measured[max(f, g), k]-measured[min(f, g), k])/abs(t[g]-t[f])
                    return np.zeros(3)
                v0, v1 = velocity(i, -1), velocity(j, 1)
                span = t[j]-t[i]
                for f in range(max(i+1, start), min(j, stop)):
                    if usable is None or usable[f]:
                        pos[f, k] = _hermite(measured[i, k], v0, measured[j, k], v1, (t[f]-t[i])/span, span)
                        source[f, k] = TIME_GAP
    # 2./3. geometry along the cable, with a motion prediction from the previous frames
    arc = np.r_[0., flight.arc]
    for f in range(start, stop):
        if usable is not None and not usable[f]:
            continue
        chain = np.vstack([flight.root[f], pos[f]])
        for k in np.flatnonzero(source[f] < 0):
            guess = None
            if f >= 2 and np.isfinite(pos[f-1, k]).all() and np.isfinite(pos[f-2, k]).all():
                guess = 2*pos[f-1, k]-pos[f-2, k]
            known = np.flatnonzero(np.isfinite(chain).all(1))
            before, after = known[known < k+1], known[known > k+1]
            a = before[-1]
            if len(after):
                b = after[0]
                straight = chain[a]+(chain[b]-chain[a])*(arc[k+1]-arc[a])/(arc[b]-arc[a])
                pos[f, k] = _on_circle(chain[a], chain[b], arc[k+1]-arc[a], arc[b]-arc[k+1],
                                       guess if guess is not None else straight)
                source[f, k] = BETWEEN
            else:
                prev = before[-2] if len(before) > 1 else None
                direction = chain[a]-chain[prev] if prev is not None else np.array([0., 0., -1.])
                direction /= max(np.linalg.norm(direction), 1e-9)
                point = chain[a]+(arc[k+1]-arc[a])*direction
                if guess is not None:  # keep the cable distance, lean towards the motion
                    lean = point+.5*(guess-point)
                    point = chain[a]+(arc[k+1]-arc[a])*(lean-chain[a])/max(np.linalg.norm(lean-chain[a]), 1e-9)
                pos[f, k] = point
                source[f, k] = END
            chain[k+1] = pos[f, k]
    return pos, source


RECONSTRUCTION_FILE = 'cable_reconstruction.npz'


def save_reconstruction(flight, labels, usable=None, path=None):
    """Write cable_reconstruction.npz: every marker in every frame, measured where seen and
    reconstructed otherwise, with the method per position. Never mixes into cable_labels."""
    positions, source = reconstruct(flight, labels, usable)
    path = Path(path or flight.folder/RECONSTRUCTION_FILE)
    np.savez_compressed(path, stamps=flight.stamps, positions=positions, source=source, arc=flight.arc,
                        root=flight.root, drone=flight.drone,
                        usable=np.ones(flight.frames, bool) if usable is None else np.asarray(usable, bool))
    counts = {name: int((source == code).sum()) for name, code in
              (('measured', MEASURED), ('time_gap', TIME_GAP), ('between', BETWEEN), ('end', END))}
    counts['impossible'] = int((source < 0).sum())
    return counts
