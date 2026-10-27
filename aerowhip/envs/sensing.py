"""OptiTrack as the policy sees it: the marker cloud the cameras would report for a simulated
state, delivered late, and the live CableFilter run on that cloud.

The policy flies on OptiTrack, not on the plant state: a drone pose from the rigid body and a
cloud of unlabelled points in which the cable's markers are sometimes missing, the drone's own
markers and a few static room markers are present, and everything arrives a few frames late.
Each lane has its own sensor: a per-marker visible/hidden Markov chain whose hazard of hiding
rises with the marker's speed, white position noise, the drone's body markers and the room's
static points added to the cloud, a frame queue for the latency, and the same `CableFilter`
that labels the recordings offline and will run in flight. What the actor receives are the
filter's marker positions, velocities and confidences from the delayed frame, with the delayed
drone pose. Statistics: `aerowhip.fitting.sensing_statistics` (identification) and
`scripts/validate_sensing.py` (the synthetic clouds through the filter against the real ones).
"""
from collections import deque
from dataclasses import dataclass

import numpy as np

from aerowhip.deployment.cable_filter import CableFilter, MARKERS, cable_candidates
from .configuration import OBSERVATION_PERIOD_S, number, require_keys


@dataclass(frozen=True)
class SensingConfig:
    """OptiTrack model. Identified values: configs/experiments (environment.sensing), from
    scripts/identify_sensing.py on the 2026-10-27 cable flights; the latency range is assumed."""
    latency_frames: tuple          # [low, high] 100 Hz frames, sampled per lane and episode (uniform)
    position_noise_m: float        # white, per axis, on every cloud point
    dropout_hazard: tuple          # ((speed_mps, hazard), ...): per-frame probability that a visible
                                   # marker moving at least that fast becomes hidden (piecewise constant)
    reappear_probability: float    # per-frame probability that a hidden marker is seen again
    drone_markers_body_m: tuple    # the drone's rigid-body markers (body frame) present in the cloud
    clutter_world_m: tuple         # static room points present in the cloud

    @classmethod
    def from_dict(cls, raw):
        require_keys(cls, raw, 'environment.sensing')
        return cls(**raw)

    def __post_init__(self):
        low, high = (int(v) for v in self.latency_frames)
        if not (0 <= low <= high) or list(self.latency_frames) != [low, high]:
            raise ValueError('latency_frames must be [low, high] whole frames, 0 <= low <= high')
        object.__setattr__(self, 'latency_frames', (low, high))
        object.__setattr__(self, 'position_noise_m', number('position_noise_m', self.position_noise_m, positive=False))
        table = tuple((number('dropout speed', s, positive=False), number('dropout hazard', h, positive=False))
                      for s, h in self.dropout_hazard)
        speeds = [s for s, _ in table]
        if not table or speeds[0] != 0. or speeds != sorted(speeds) or len(set(speeds)) != len(speeds) \
                or any(h > 1. for _, h in table):
            raise ValueError('dropout_hazard must start at speed 0, increase in speed, with hazards in [0, 1]')
        object.__setattr__(self, 'dropout_hazard', table)
        p = number('reappear_probability', self.reappear_probability)
        if p > 1.:
            raise ValueError('reappear_probability must lie in (0, 1]')
        object.__setattr__(self, 'reappear_probability', p)
        for name in ('drone_markers_body_m', 'clutter_world_m'):
            points = np.asarray(getattr(self, name), float)
            if points.size and (points.ndim != 2 or points.shape[1] != 3 or not np.isfinite(points).all()):
                raise ValueError(f'{name} must be a list of finite XYZ points')
            object.__setattr__(self, name, tuple(tuple(map(float, q)) for q in points.reshape(-1, 3)))


class OptiTrackSensor:
    """One lane's cameras, latency and cable filter. `observe` takes the true state of one 100 Hz
    frame and returns what the policy receives at that frame: the sensed frame `latency` frames
    earlier (the first frame until the queue has filled), with the filter's cable estimate."""

    def __init__(self, config, arc, attachment_body, rng):
        self.config, self.rng = config, rng
        self.arc, self.attachment = np.asarray(arc, float), np.asarray(attachment_body, float)
        if len(self.arc) != MARKERS:
            raise ValueError('The cable filter tracks ten markers')
        self.filter = CableFilter(self.arc)
        self.speeds = np.array([s for s, _ in config.dropout_hazard])
        self.hazards = np.array([h for _, h in config.dropout_hazard])
        self.drone_markers = np.asarray(config.drone_markers_body_m, float).reshape(-1, 3)
        self.clutter = np.asarray(config.clutter_world_m, float).reshape(-1, 3)
        self.reset()

    def reset(self):
        low, high = self.config.latency_frames
        self.latency = int(self.rng.integers(low, high+1))
        self.hidden = np.zeros(MARKERS, bool)       # the first frame of an episode is complete
        self.queue = deque()
        self.filter.reset()
        self.sensed = None
        self.frames = 0

    def cloud(self, index, p, R, markers, marker_speeds):
        """The cloud the cameras report for frame `index`: visible cable markers with noise, the
        drone's body markers and the static clutter, in random order."""
        if index > 0:
            hazard = self.hazards[np.searchsorted(self.speeds, marker_speeds, 'right')-1]
            u = self.rng.random(MARKERS)
            self.hidden = np.where(self.hidden, u >= self.config.reappear_probability, u < hazard)
        points = [markers[~self.hidden], p+self.drone_markers@R.T, self.clutter]
        cloud = np.concatenate(points, 0)
        cloud = cloud+self.rng.normal(0., self.config.position_noise_m, cloud.shape)
        return cloud[self.rng.permutation(len(cloud))]

    def observe(self, index, p, R, v, w, markers, marker_velocities):
        """True state of frame `index` in -> sensed frame out (a dict; see `_sense`)."""
        speeds = np.linalg.norm(marker_velocities, axis=1)
        cloud = self.cloud(index, p, R, markers, speeds)
        self.queue.append((index, p.copy(), R.copy(), v.copy(), w.copy(), cloud, (~self.hidden).copy()))
        if len(self.queue) > self.latency+1:
            raise RuntimeError('Sensor queue overflow')
        if len(self.queue) == self.latency+1 or self.sensed is None:
            self.sensed = self._sense(*self.queue.popleft())
        return self.sensed

    def _sense(self, index, p, R, v, w, cloud, visible):
        root = p+R@self.attachment
        candidates = cable_candidates(cloud, p)
        state = self.filter.observe(index*OBSERVATION_PERIOD_S, root, candidates)
        positions = state['positions']
        if not state['initialized'] or not np.isfinite(positions).all():
            # The filter could not label a hanging cable from this cloud: no estimate yet, the
            # markers are reported at the attachment with zero confidence (deployment would not
            # start the policy in this state; the first frame of an episode is complete).
            positions = np.tile(root, (MARKERS, 1))
            velocities = np.zeros((MARKERS, 3))
            confidence = np.zeros(MARKERS)
        else:
            velocities, confidence = state['velocities'], state['confidence']
        root_velocity = v+R@np.cross(w, self.attachment)
        self.frames += 1
        return dict(index=index, position=p, rotation=R, velocity=v, angular_velocity=w, root=root,
                    root_velocity=root_velocity, cable_positions=positions, cable_velocities=velocities,
                    confidence=confidence, visible=visible, labelled=self.filter.index >= 0,
                    cloud_size=len(candidates), latency=self.latency)
