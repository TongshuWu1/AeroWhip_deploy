"""Calibration flights (companion repository, `calibration_data/`) as fitting inputs.

A flight folder holds `flight.jsonl` (50 Hz host log: planned reference, sent acceleration
command, trims, battery), `poses.jsonl` (every OptiTrack drone pose, ~100 Hz, OptiTrack
stamps), `cable_labels.npz` (cable markers labelled from the marker cloud on the same
stamps, NaN where a marker was not seen) and `parameters.json`. Everything is on one clock
(the host's), OptiTrack stamps included, so no clock alignment is needed.

`windows(...)` cuts fitting windows on the OptiTrack frames: 100 Hz, gap-free in the pose
stream, with the acceleration command that was in force at each frame (zero-order hold,
the model's command delay already included by the flight script's own timing being on the
same clock), the measured drone pose, and the measured markers with a mask for hidden ones.
Markers are never filled in: the fit must skip masked ones.
"""
from dataclasses import dataclass
import json
from pathlib import Path

import numpy as np

from aerowhip.physics.state import quaternion_xyzw_to_rotation_matrix

DT = .01
MARKERS = 10


@dataclass
class CalibrationFlight:
    folder: Path
    setup: str
    phases: list              # per 50 Hz log sample
    log_time: np.ndarray      # (L,) flight time t, s
    log_stamp: np.ndarray     # (L,) OptiTrack stamp of the log sample's pose
    sent: np.ndarray          # (L, 3) acceleration command sent, world, m/s^2 (trims included)
    trim: np.ndarray          # (L, 3) trims inside `sent`
    stamp: np.ndarray         # (F,) OptiTrack stamps of the pose frames
    position: np.ndarray      # (F, 3) tracking origin
    quaternion: np.ndarray    # (F, 4) xyzw
    markers: np.ndarray | None   # (F, 10, 3) measured cable markers, NaN where hidden
    yaw: float
    physical: dict
    battery: np.ndarray | None = None   # (L,) battery voltage under load, V (NaN if not logged)

    @property
    def frames(self):
        return len(self.stamp)

    def phase_at(self, stamp):
        i = int(np.clip(np.searchsorted(self.log_stamp, stamp, 'right')-1, 0, len(self.phases)-1))
        return self.phases[i]

    def command_at(self, stamps, *, strip_trim=True):
        """Acceleration command in force at each stamp (zero-order hold on the 50 Hz log)."""
        i = np.clip(np.searchsorted(self.log_stamp, stamps, 'right')-1, 0, len(self.sent)-1)
        return (self.sent-self.trim)[i] if strip_trim else self.sent[i]


def on_frame_grid(stamp):
    """Mask keeping the frames nearest a DT grid when the capture ran faster than 100 Hz (Motive
    at 200 Hz gives 5 ms steps); every frame when the capture is at 100 Hz or slower."""
    keep = np.ones(len(stamp), bool)
    if len(stamp) < 3 or np.median(np.diff(stamp)) > .75*DT:
        return keep
    keep[:] = False
    i, target = 0, stamp[0]
    while i < len(stamp):
        keep[i] = True
        target += DT
        j = int(np.searchsorted(stamp, target))
        candidates = [k for k in (j-1, j) if i < k < len(stamp)]
        if not candidates:
            break
        i = min(candidates, key=lambda k: abs(stamp[k]-target))
        target = stamp[i]
    return keep


def load(folder):
    folder = Path(folder)
    rows = [json.loads(line) for line in (folder/'flight.jsonl').read_text().splitlines()]
    params = json.loads((folder/'parameters.json').read_text())
    poses = [json.loads(line) for line in (folder/'poses.jsonl').read_text().splitlines()]
    stamp = np.array([p['stamp'] for p in poses])
    order = np.argsort(stamp, kind='stable')
    stamp = stamp[order]
    keep = np.r_[True, np.diff(stamp) > 1e-6]     # drop repeated frames
    keep &= on_frame_grid(stamp)                  # a 200 Hz capture is thinned to the 100 Hz grid
    stamp = stamp[keep]
    position = np.array([poses[i]['position'] for i in order])[keep]
    quaternion = np.array([poses[i]['quaternion_xyzw'] for i in order])[keep]
    markers = None
    if (folder/'cable_labels.npz').exists():
        labels = np.load(folder/'cable_labels.npz')
        markers = np.full((len(stamp), MARKERS, 3), np.nan)
        j = np.searchsorted(stamp, labels['stamps'])
        ok = (j < len(stamp)) & np.isclose(stamp[np.minimum(j, len(stamp)-1)], labels['stamps'], atol=1e-6, rtol=0)
        markers[j[ok]] = labels['positions'][ok]
    log_stamp = np.array([r['pose_stamp'] if r.get('pose_stamp') is not None else np.nan for r in rows])
    return CalibrationFlight(folder, params['settings']['setup'], [r['phase'] for r in rows],
                             np.array([r['t'] for r in rows]), log_stamp,
                             np.array([r['sent_acceleration'] for r in rows]), np.array([r['trim'] for r in rows]),
                             stamp, position, quaternion, markers, float(params['yaw_rad']), params['physical'],
                             np.array([np.nan if r.get('battery_v') is None else r['battery_v'] for r in rows], float))


@dataclass
class Window:
    flight: str
    phase: str
    start_stamp: float
    time: np.ndarray          # (N+1,) seconds from the window start, 0.01 steps
    position: np.ndarray      # (N+1, 3)
    rotation: np.ndarray      # (N+1, 3, 3)
    markers: np.ndarray       # (N+1, 10, 3) NaN where hidden
    seen: np.ndarray          # (N+1, 10) bool
    command: np.ndarray       # (N+1, 3) acceleration command in force, trims stripped
    sent: np.ndarray          # (N+1, 3) the command actually sent, trims included
    history_position: np.ndarray   # (H+1, 3) samples before the window, for velocities
    history_rotation: np.ndarray
    history_markers: np.ndarray
    history_time: np.ndarray


def windows(flight, *, duration_s=2., history_s=.1, stride_s=1., phases=None, min_seen=8):
    """Gap-free 100 Hz windows over the flight's pose frames. `phases` limits to log phases
    (prefix match, e.g. 'decay 0'); windows whose history has a hidden marker, or whose
    frames average fewer than `min_seen` markers, are skipped. A frame is 'gap-free' when
    consecutive stamps are within 10 +- 3 ms (OptiTrack stamps are bunched; see loader)."""
    steps, hist = round(duration_s/DT), round(history_s/DT)
    stride = max(1, round(stride_s/DT))
    stamp = flight.stamp
    good = np.abs(np.diff(stamp)-DT) < 3e-3
    out = []
    first = hist
    while first+steps < flight.frames:
        sel = slice(first-hist, first+steps+1)
        phase = flight.phase_at(stamp[first])
        if phases is not None and not any(phase.startswith(p) for p in phases):
            first += stride; continue
        if not good[first-hist:first+steps].all():
            first += stride; continue
        m = flight.markers[sel] if flight.markers is not None else np.full((steps+hist+1, MARKERS, 3), np.nan)
        seen = np.isfinite(m).all(2)
        if flight.markers is not None and (not seen[:hist+1].all() or seen[hist:].sum(1).mean() < min_seen):
            first += stride; continue
        R = np.stack([quaternion_xyzw_to_rotation_matrix(q) for q in flight.quaternion[sel]])
        t = stamp[sel]-stamp[first]
        out.append(Window(flight.folder.name, phase, float(stamp[first]), np.arange(steps+1)*DT,
                          flight.position[sel][hist:].copy(), R[hist:].copy(), m[hist:].copy(), seen[hist:].copy(),
                          flight.command_at(stamp[sel][hist:]), flight.command_at(stamp[sel][hist:], strip_trim=False),
                          flight.position[sel][:hist+1].copy(), R[:hist+1].copy(), m[:hist+1].copy(), t[:hist+1].copy()))
        first += stride
    return out
