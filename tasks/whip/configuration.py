"""Typed Stage 1 task settings, read from the experiment file's "task" section.

Every task constant used by the reward and environment lives here; code reads
these fields instead of repeating numbers. Validation checks types and ranges.
Values the code does not implement (another reward version or output direction)
are rejected explicitly rather than silently ignored.
"""

from dataclasses import dataclass, fields
import math

from .workspace import CylinderWorkspace

REWARD_VERSION = 'continuous_directional_whip_four_second_v6'
OBSERVATION_PERIOD_S = .01  # The task observes the plant's 100 Hz output clock.


def _number(name, value, *, positive=True):
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
        raise ValueError(f'{name} must be a finite number')
    if positive and value <= 0:
        raise ValueError(f'{name} must be positive')
    return float(value)


def _on_observation_grid(name, value):
    steps = value/OBSERVATION_PERIOD_S
    if abs(steps-round(steps)) > 1e-9:
        raise ValueError(f'{name} must be a whole number of 10 ms observation periods')


def _strict(cls, raw, name):
    if not isinstance(raw, dict):
        raise ValueError(f'{name} must be an object')
    expected = {f.name for f in fields(cls)}
    missing, unknown = expected-raw.keys(), raw.keys()-expected
    if missing or unknown:
        raise ValueError(f'Invalid {name}: missing={sorted(missing)}, unknown={sorted(unknown)}')


@dataclass(frozen=True)
class FoldSupportConfig:
    """Causal fold-transport support F (see doc/stage1.md, "How fold support F is measured")."""
    history_window_s: float
    support_fade_s: float
    travel_reference_fraction: float
    turn_lower_deg: float
    turn_upper_deg: float
    coherence_reference: float

    def __post_init__(self):
        for f in fields(self):
            object.__setattr__(self, f.name, _number(f.name, getattr(self, f.name)))
        _on_observation_grid('history_window_s', self.history_window_s)
        if not self.turn_lower_deg < self.turn_upper_deg <= 180:
            raise ValueError('Require 0 < turn_lower_deg < turn_upper_deg <= 180')


@dataclass(frozen=True)
class Stage1TaskConfig:
    scene: str
    model: str
    reward: str
    duration_s: float
    output_direction_world: tuple
    workspace: CylinderWorkspace
    workspace_cost_weight: float
    maximum_tilt_deg: float
    minimum_cable_height_m: float
    failure_cost_per_remaining_s: float
    fold: FoldSupportConfig
    rate_weight: float
    reach_power: float
    alignment_power: float

    @classmethod
    def from_dict(cls, raw, *, scene, model):
        """Build from the experiment file's task section; unknown or missing keys fail."""
        raw = dict(raw, scene=scene, model=model)
        _strict(cls, raw, 'task section')
        _strict(CylinderWorkspace, raw['workspace'], 'task.workspace')
        _strict(FoldSupportConfig, raw['fold'], 'task.fold')
        return cls(**dict(raw, workspace=CylinderWorkspace(**raw['workspace']),
                          fold=FoldSupportConfig(**raw['fold'])))

    def __post_init__(self):
        if not all(isinstance(p, str) and p for p in (self.scene, self.model)):
            raise ValueError('Scene/model paths are required')
        if self.reward != REWARD_VERSION:
            raise ValueError(f'Unknown reward {self.reward!r}; implemented: {REWARD_VERSION}')
        direction = tuple(_number('output_direction_world', v, positive=False)
                          for v in self.output_direction_world)
        if direction != (1., 0., 0.):
            raise ValueError('Only world +X output is implemented (output_direction_world = [1, 0, 0])')
        object.__setattr__(self, 'output_direction_world', direction)
        for name in ('duration_s', 'workspace_cost_weight', 'failure_cost_per_remaining_s',
                     'rate_weight', 'reach_power', 'alignment_power', 'maximum_tilt_deg'):
            object.__setattr__(self, name, _number(name, getattr(self, name)))
        object.__setattr__(self, 'minimum_cable_height_m',
                           _number('minimum_cable_height_m', self.minimum_cable_height_m, positive=False))
        _on_observation_grid('duration_s', self.duration_s)
        if self.maximum_tilt_deg >= 90:
            raise ValueError('maximum_tilt_deg must lie in (0, 90)')
