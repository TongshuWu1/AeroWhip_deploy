"""Typed settings for the fold-transport task (the experiment file's "task" section).

Every reward constant lives here; the reward reads these fields instead of
repeating numbers. Choices the code does not implement (another output
direction) are rejected explicitly rather than silently ignored.
"""
from dataclasses import dataclass, fields

from aerowhip.envs.configuration import number, on_observation_grid, require_keys


@dataclass(frozen=True)
class FoldSupportConfig:
    """Causal fold-transport support F (see docs/fold_transport_task.md, "How fold support F is measured")."""
    history_window_s: float
    support_fade_s: float
    travel_reference_fraction: float
    turn_lower_deg: float
    turn_upper_deg: float
    coherence_reference: float

    def __post_init__(self):
        for f in fields(self):
            object.__setattr__(self, f.name, number(f.name, getattr(self, f.name)))
        on_observation_grid('history_window_s', self.history_window_s)
        if not self.turn_lower_deg < self.turn_upper_deg <= 180:
            raise ValueError('Require 0 < turn_lower_deg < turn_upper_deg <= 180')


@dataclass(frozen=True)
class FoldTransportConfig:
    """Reward weights for +X fold transport; workspace and horizon come from the environment.

    Rate = rate_weight × F × reach^reach_power × alignment^alignment_power × outward speed / L.
    The workspace cost is workspace_cost_weight × mean squared workspace excess over the
    horizon. A numerical failure costs failure_cost_per_remaining_s per unexecuted second.
    """
    name: str
    output_direction_world: tuple
    workspace_cost_weight: float
    failure_cost_per_remaining_s: float
    fold: FoldSupportConfig
    rate_weight: float
    reach_power: float
    alignment_power: float

    @classmethod
    def from_dict(cls, raw):
        require_keys(cls, raw, 'task')
        require_keys(FoldSupportConfig, raw['fold'], 'task.fold')
        return cls(**dict(raw, fold=FoldSupportConfig(**raw['fold'])))

    def __post_init__(self):
        direction = tuple(number('output_direction_world', v, positive=False)
                          for v in self.output_direction_world)
        if direction != (1., 0., 0.):
            raise ValueError('Only world +X output is implemented (output_direction_world = [1, 0, 0])')
        object.__setattr__(self, 'output_direction_world', direction)
        for name in ('workspace_cost_weight', 'failure_cost_per_remaining_s',
                     'rate_weight', 'reach_power', 'alignment_power'):
            object.__setattr__(self, name, number(name, getattr(self, name)))
