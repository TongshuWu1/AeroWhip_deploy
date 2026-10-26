"""Typed settings for the experiment file's "environment" section and shared parsing helpers."""
from dataclasses import dataclass, fields
import math

from .workspace import CylinderWorkspace

OBSERVATION_PERIOD_S = .01  # Environments observe the plant's 100 Hz output clock.


def number(name, value, *, positive=True):
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
        raise ValueError(f'{name} must be a finite number')
    if positive and value <= 0:
        raise ValueError(f'{name} must be positive')
    return float(value)


def integer(name, value, minimum=1):
    if type(value) is not int or value < minimum:
        raise ValueError(f'{name} must be an integer >= {minimum}')
    return value


def on_observation_grid(name, value):
    steps = value/OBSERVATION_PERIOD_S
    if abs(steps-round(steps)) > 1e-9:
        raise ValueError(f'{name} must be a whole number of 10 ms observation periods')


def require_keys(cls, raw, name):
    """Exact key set of a dataclass section: unknown or missing keys fail."""
    if not isinstance(raw, dict):
        raise ValueError(f'{name} must be an object')
    expected = {f.name for f in fields(cls)}
    missing, unknown = expected-raw.keys(), raw.keys()-expected
    if missing or unknown:
        raise ValueError(f'Invalid {name}: missing={sorted(missing)}, unknown={sorted(unknown)}')


@dataclass(frozen=True)
class EnvironmentConfig:
    """Episode horizon, policy control and the physical validity limits.

    Actions are normalized world-frame accelerations (excluding gravity) scaled by
    acceleration_limit_mps2, held for one policy period, with fixed initial yaw.
    Tilt above maximum_tilt_deg or any cable node below minimum_cable_height_m
    marks an episode physically invalid; it does not stop the episode.
    """
    duration_s: float
    policy_rate_hz: int
    acceleration_limit_mps2: tuple
    workspace: CylinderWorkspace
    maximum_tilt_deg: float
    minimum_cable_height_m: float

    @classmethod
    def from_dict(cls, raw):
        require_keys(cls, raw, 'environment')
        require_keys(CylinderWorkspace, raw['workspace'], 'environment.workspace')
        return cls(**dict(raw, workspace=CylinderWorkspace(**raw['workspace'])))

    def __post_init__(self):
        object.__setattr__(self, 'duration_s', number('duration_s', self.duration_s))
        on_observation_grid('duration_s', self.duration_s)
        integer('policy_rate_hz', self.policy_rate_hz)
        stride = 1/(OBSERVATION_PERIOD_S*self.policy_rate_hz)
        if abs(stride-round(stride)) > 1e-9:
            raise ValueError('policy_rate_hz must divide the 100 Hz observation rate')
        limits = self.acceleration_limit_mps2
        if not isinstance(limits, (list, tuple)) or len(limits) != 3:
            raise ValueError('acceleration_limit_mps2 needs three XYZ limits [m/s²]')
        object.__setattr__(self, 'acceleration_limit_mps2',
                           tuple(number('acceleration limit', v) for v in limits))
        object.__setattr__(self, 'maximum_tilt_deg', number('maximum_tilt_deg', self.maximum_tilt_deg))
        if self.maximum_tilt_deg >= 90:
            raise ValueError('maximum_tilt_deg must lie in (0, 90)')
        object.__setattr__(self, 'minimum_cable_height_m',
                           number('minimum_cable_height_m', self.minimum_cable_height_m, positive=False))

    @property
    def observations(self):
        """Number of 100 Hz steps in one episode (samples excluding the reset state)."""
        return round(self.duration_s/OBSERVATION_PERIOD_S)
