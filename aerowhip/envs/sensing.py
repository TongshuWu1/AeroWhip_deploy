"""OptiTrack as the policy sees it: the marker cloud the cameras would report for a simulated
state, delivered late, and the live CableFilter run on that cloud.

The policy flies on OptiTrack, not on the plant state: a drone pose from the rigid body and a
cloud of unlabelled points in which the cable's markers are sometimes missing, the drone's own
markers and a few static room markers are present, and everything arrives a few frames late.
Each lane has its own sensor: a per-marker visible/hidden Markov chain whose hazard of hiding
rises with the marker's speed, white position noise, the drone's body markers and the room's
static points added to the cloud, noise on the drone pose, jittered receipt stamps, a frame
queue for the latency, and the same estimator that runs in flight (deployment.observer: frame
clock, drone velocity and body rate from the pose history, the live CableFilter). What the
actor receives is that estimate of the delayed frame (envs.observation builds the row). Statistics: `aerowhip.fitting.sensing_statistics` (identification) and
`scripts/validate_sensing.py` (the synthetic clouds through the filter against the real ones).
"""
from collections import deque
from dataclasses import dataclass

import numpy as np

from aerowhip.deployment.cable_filter import MARKERS
from aerowhip.deployment.observer import Observer
from .configuration import OBSERVATION_PERIOD_S, number, require_keys


@dataclass(frozen=True)
class SensingConfig:
    """OptiTrack model. Identified values: configs/experiments (environment.sensing), from
    scripts/identify_sensing.py on the 2026-10-27 cable flights; the latency range is assumed."""
    latency_frames: tuple          # [low, high] 100 Hz frames, sampled per lane and episode (uniform)
    position_noise_m: float        # white, per axis, on every cloud point
    drone_position_noise_m: float  # white, per axis, on the drone rigid-body position
    drone_rotation_noise_rad: float  # white, per body axis, on the drone rigid-body rotation
    receipt_jitter_s: float        # host receipt stamp = frame time + uniform(-j, j)
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
        for name in ('position_noise_m', 'drone_position_noise_m', 'drone_rotation_noise_rad', 'receipt_jitter_s'):
            object.__setattr__(self, name, number(name, getattr(self, name), positive=False))
        if self.receipt_jitter_s >= .5*OBSERVATION_PERIOD_S:
            raise ValueError('receipt_jitter_s must stay below half a frame')
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


def rotation(v):
    """Rotation matrix of the rotation vector v (Rodrigues)."""
    angle = float(np.linalg.norm(v))
    if angle < 1e-12:
        return np.eye(3)
    k = v/angle
    K = np.array([[0., -k[2], k[1]], [k[2], 0., -k[0]], [-k[1], k[0], 0.]])
    return np.eye(3)+np.sin(angle)*K+(1-np.cos(angle))*K@K


class OptiTrackSensor:
    """One lane's cameras, latency and estimator. `observe` takes the true state of one 100 Hz
    frame and returns what the policy receives at that frame: the Observer estimate
    (deployment.observer, the same as in flight) of the sensed frame `latency` frames earlier (the
    first frame until the queue has filled)."""

    def __init__(self, config, arc, attachment_body, rng):
        self.config, self.rng = config, rng
        self.arc, self.attachment = np.asarray(arc, float), np.asarray(attachment_body, float)
        if len(self.arc) != MARKERS:
            raise ValueError('The cable filter tracks ten markers')
        self.observer = Observer(self.arc, self.attachment, dt=OBSERVATION_PERIOD_S)
        self.speeds = np.array([s for s, _ in config.dropout_hazard])
        self.hazards = np.array([h for _, h in config.dropout_hazard])
        self.drone_markers = np.asarray(config.drone_markers_body_m, float).reshape(-1, 3)
        self.clutter = np.asarray(config.clutter_world_m, float).reshape(-1, 3)
        self.reset()

    @property
    def filter(self):
        return self.observer.filter

    def reset(self):
        low, high = self.config.latency_frames
        self.latency = int(self.rng.integers(low, high+1))
        self.hidden = np.zeros(MARKERS, bool)       # the first frame of an episode is complete
        self.queue = deque()
        self.observer.reset()
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

    def observe(self, index, p, R, markers, marker_velocities):
        """True state of frame `index` in -> the Observer estimate the policy receives (a dict)."""
        speeds = np.linalg.norm(marker_velocities, axis=1)
        cloud = self.cloud(index, p, R, markers, speeds)
        pose = p+self.rng.normal(0., self.config.drone_position_noise_m, 3)
        R = R@rotation(self.rng.normal(0., self.config.drone_rotation_noise_rad, 3))
        stamp = index*OBSERVATION_PERIOD_S+self.rng.uniform(-1., 1.)*self.config.receipt_jitter_s
        self.queue.append((index, stamp, pose, R, cloud, (~self.hidden).copy()))
        if len(self.queue) > self.latency+1:
            raise RuntimeError('Sensor queue overflow')
        if len(self.queue) == self.latency+1 or self.sensed is None:
            index, stamp, pose, R, cloud, visible = self.queue.popleft()
            estimate = self.observer.observe(stamp, pose, R, cloud)
            self.frames += 1
            self.sensed = dict(estimate, index=index, visible=visible, latency=self.latency)
        return self.sensed
