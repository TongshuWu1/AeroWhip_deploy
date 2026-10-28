"""A flight folder read as it is written: the flight script appends its logs line by line
(`flight.jsonl`, `poses.jsonl`, `cloud.jsonl`, line-buffered), so the online estimator needs no ROS
connection (and makes no ROS queries during a flight). The same reader replays a recorded flight
as if live: everything is read at once and revealed up to a moving stamp.

The cable markers are labelled as in flight: every cloud frame goes through the native Observer
(the live cable filter, aerowhip.deployment.observer) with the drone pose of the same stamp; a
marker counts only where the filter associated a point to it this frame with confidence. So the
estimator sees what a live estimate can see, not the offline sequence labels.

`flight()` gives the recent part (the last `keep_s`) as a CalibrationFlight, the structure the
replays and windows of the offline fits use.
"""
import json
from pathlib import Path

import numpy as np

from aerowhip.physics.state import quaternion_xyzw_to_rotation_matrix
from aerowhip.takes.calibration_flights import MARKERS, from_records

CONFIDENT = .5               # a marker counts where the live filter's confidence is at least this


class _Tail:
    """New complete lines of a file that keeps growing (a partial last line waits)."""
    def __init__(self, path):
        self.path, self.offset, self.partial = Path(path), 0, b''

    def lines(self):
        if not self.path.exists():
            return []
        with self.path.open('rb') as f:
            f.seek(self.offset)
            data = f.read()
        self.offset += len(data)
        data = self.partial+data
        *complete, self.partial = data.split(b'\n')
        out = []
        for line in complete:
            if line.strip():
                try:
                    out.append(json.loads(line))
                except ValueError:
                    pass                              # a torn line from an interrupted writer
        return out


class FlightStream:
    def __init__(self, folder, arc, attachment, *, keep_s=6., replay=False):
        """arc: marker distances along the cable from the attachment (m); attachment: tracking origin
        to the attachment, body frame (m). replay: read the whole folder now and reveal it with
        `advance(stamp)` (as if live); otherwise `advance()` reads what has been written since."""
        from aerowhip.deployment.observer import Observer
        self.folder, self.keep, self.replay = Path(folder), float(keep_s), replay
        self.observer = Observer(np.asarray(arc, float), np.asarray(attachment, float))
        self.tails = {name: _Tail(self.folder/name) for name in ('flight.jsonl', 'poses.jsonl', 'cloud.jsonl')}
        self.params = None
        self.rows, self.poses, self.labels = [], [], []          # recent records (labels: (stamp, (10, 3)))
        self.queue = {'flight.jsonl': [], 'poses.jsonl': [], 'cloud.jsonl': []}
        self.pose_by_stamp = {}
        self.latest = -np.inf                                    # newest revealed pose stamp
        self.has_cloud = (self.folder/'cloud.jsonl').exists()
        self.finished = False
        if replay:
            self._read()
            self.end = max((r['stamp'] for r in self.queue['poses.jsonl']), default=-np.inf)

    @property
    def has_cable_labels(self):
        """The flight records a marker cloud (a cable flight): the cable family is fitted."""
        return self.has_cloud

    @property
    def ready(self):
        return self.params is not None and len(self.poses) > 0

    def _read(self):
        if self.params is None and (self.folder/'parameters.json').exists():
            try:
                self.params = json.loads((self.folder/'parameters.json').read_text())
            except ValueError:
                return                                           # still being written
        for name, tail in self.tails.items():
            self.queue[name] += tail.lines()
        self.has_cloud = self.has_cloud or (self.folder/'cloud.jsonl').exists()
        self.finished = (self.folder/'result.json').exists()

    def advance(self, until=None):
        """Take in what was written (live) or what was recorded up to `until` (replay)."""
        if not self.replay:
            self._read()
            until = np.inf
        poses = self.queue['poses.jsonl']
        n = 0
        while n < len(poses) and poses[n]['stamp'] <= until:
            n += 1
        for p in poses[:n]:
            self.poses.append(p)
            self.pose_by_stamp[round(p['stamp'], 6)] = p
            self.latest = p['stamp']
        del poses[:n]
        rows = self.queue['flight.jsonl']
        n = 0
        while n < len(rows) and (rows[n].get('pose_stamp') is None or rows[n]['pose_stamp'] <= until):
            n += 1
        self.rows += rows[:n]
        del rows[:n]
        clouds = self.queue['cloud.jsonl']
        n = 0
        while n < len(clouds) and clouds[n]['stamp'] <= min(until, self.latest):
            self._observe(clouds[n])
            n += 1
        del clouds[:n]
        self._trim()

    def _observe(self, cloud):
        pose = self.pose_by_stamp.get(round(cloud['stamp'], 6))
        if pose is None:
            return
        p = np.asarray(pose['position'], float)
        R = quaternion_xyzw_to_rotation_matrix(np.asarray(pose['quaternion_xyzw'], float))
        points = np.array([q[:3] for q in cloud['points']], float).reshape(-1, 3)
        state = self.observer.observe(cloud['stamp'], p, R, points)
        markers = np.full((MARKERS, 3), np.nan)
        if state['initialized']:
            ok = np.asarray(state['labelled'], bool) & (np.asarray(state['confidence']) >= CONFIDENT)
            markers[ok] = np.asarray(state['cable_positions'])[ok]
        self.labels.append((cloud['stamp'], markers))

    def _trim(self):
        cut = self.latest-self.keep
        drop = 0
        while drop < len(self.poses) and self.poses[drop]['stamp'] < cut:
            self.pose_by_stamp.pop(round(self.poses[drop]['stamp'], 6), None)
            drop += 1
        del self.poses[:drop]
        drop = 0
        while drop < len(self.rows)-1 and (self.rows[drop+1].get('pose_stamp') or -np.inf) < cut:
            drop += 1                                            # keep the row in force at the cut
        del self.rows[:drop]
        drop = 0
        while drop < len(self.labels) and self.labels[drop][0] < cut:
            drop += 1
        del self.labels[:drop]

    def flight(self):
        """The recent part as a CalibrationFlight (markers None without a cloud), or None."""
        if not self.ready or len(self.rows) < 2 or len(self.poses) < 20:
            return None
        stamps = positions = None
        if self.has_cloud and self.labels:
            stamps = np.array([s for s, _ in self.labels])
            positions = np.stack([m for _, m in self.labels])
        return from_records(self.folder, list(self.rows), self.params, list(self.poses), stamps, positions)

    def phase(self):
        return self.rows[-1]['phase'] if self.rows else ''
