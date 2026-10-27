"""Typed settings for the strike task (the experiment file's "task" section) and its curriculum.

Every reward constant and every evaluation threshold lives here; the task reads these fields
instead of repeating numbers. The curriculum stages change only the target distance, the tip
speed threshold and the workspace radius; everything else is fixed for a run.
"""
from dataclasses import dataclass, fields

from aerowhip.envs.configuration import integer, number, on_observation_grid, require_keys


@dataclass(frozen=True)
class StrikeStage:
    """One curriculum stage. The target sits target_distance_m along the task direction from the
    workspace origin; a hit needs tip speed >= speed_threshold_mps; the drone's workspace
    radius is workspace_radius_m (the heights come from the environment)."""
    target_distance_m: float
    speed_threshold_mps: float
    workspace_radius_m: float

    def __post_init__(self):
        for f in fields(self):
            object.__setattr__(self, f.name, number(f.name, getattr(self, f.name)))


@dataclass(frozen=True)
class CurriculumConfig:
    """Stages advance when the strike rate over the last window_episodes training episodes
    reaches advance_strike_rate; the last stage is final and is the evaluation stage.
    reference_state_fraction of the training lanes start from a recorded whip state
    (reference_bank, an .npz written by scripts/build_reset_bank.py) instead of hover."""
    stages: tuple
    advance_strike_rate: float
    window_episodes: int
    reference_state_fraction: float
    reference_bank: str | None

    def __post_init__(self):
        if not self.stages:
            raise ValueError('The curriculum needs at least one stage')
        object.__setattr__(self, 'stages', tuple(self.stages))
        object.__setattr__(self, 'advance_strike_rate', number('advance_strike_rate', self.advance_strike_rate))
        if not 0 < self.advance_strike_rate <= 1:
            raise ValueError('advance_strike_rate must lie in (0, 1]')
        integer('window_episodes', self.window_episodes)
        object.__setattr__(self, 'reference_state_fraction',
                           number('reference_state_fraction', self.reference_state_fraction, positive=False))
        if not 0 <= self.reference_state_fraction < 1:
            raise ValueError('reference_state_fraction must lie in [0, 1)')
        if self.reference_state_fraction > 0 and not self.reference_bank:
            raise ValueError('reference_state_fraction > 0 needs a reference_bank file')

    @property
    def final(self):
        return self.stages[-1]


@dataclass(frozen=True)
class StrikeConfig:
    """Strike task: one strike then recovery (docs/strike_task.md).

    Target: target_distance_m (stage) along target_direction_world from the workspace origin,
    target_below_origin_m below it. Hit: the tip passes within hit_radius_m of the target with
    speed >= the stage threshold. Whip event (reported): peak tip speed >= threshold, at least
    amplification_ratio times the attachment's peak speed over the previous amplification_window_s,
    with the tip at >= extension_fraction of the cable length. Rewards: strike_bonus +
    speed_bonus_per_mps per m/s above the threshold at the hit; potential-based progress_weight
    per cable length of tip-to-target distance closed; approach_weight per second at the full
    threshold speed toward the target (capped); recovery_bonus when the drone is in the workspace
    and slower than recovery_speed_mps recovery_s after the hit. Costs: workspace_cost_weight ×
    mean squared workspace excess over the horizon, action_cost_weight × mean squared normalized
    action, failure_cost_per_remaining_s per unexecuted second on termination.
    """
    name: str
    target_direction_world: tuple
    target_below_origin_m: float
    hit_radius_m: float
    amplification_ratio: float
    amplification_window_s: float
    extension_fraction: float
    recovery_s: float
    recovery_speed_mps: float
    strike_bonus: float
    speed_bonus_per_mps: float
    recovery_bonus: float
    progress_weight: float
    approach_weight: float
    workspace_cost_weight: float
    action_cost_weight: float
    failure_cost_per_remaining_s: float
    curriculum: CurriculumConfig

    @classmethod
    def from_dict(cls, raw):
        require_keys(cls, raw, 'task')
        require_keys(CurriculumConfig, raw['curriculum'], 'task.curriculum')
        stages = raw['curriculum']['stages']
        if not isinstance(stages, list):
            raise ValueError('task.curriculum.stages must be a list')
        for i, stage in enumerate(stages):
            require_keys(StrikeStage, stage, f'task.curriculum.stages[{i}]')
        curriculum = CurriculumConfig(**dict(raw['curriculum'], stages=tuple(StrikeStage(**s) for s in stages)))
        return cls(**dict(raw, curriculum=curriculum))

    def __post_init__(self):
        direction = tuple(number('target_direction_world', v, positive=False) for v in self.target_direction_world)
        if direction != (1., 0., 0.):
            raise ValueError('Only a world +X target is implemented (target_direction_world = [1, 0, 0])')
        object.__setattr__(self, 'target_direction_world', direction)
        for name in ('hit_radius_m', 'amplification_ratio', 'amplification_window_s', 'extension_fraction',
                     'recovery_s', 'recovery_speed_mps', 'strike_bonus', 'speed_bonus_per_mps', 'progress_weight',
                     'approach_weight', 'workspace_cost_weight', 'failure_cost_per_remaining_s'):
            object.__setattr__(self, name, number(name, getattr(self, name)))
        for name in ('target_below_origin_m', 'recovery_bonus', 'action_cost_weight'):
            object.__setattr__(self, name, number(name, getattr(self, name), positive=False))
        if not 0 < self.extension_fraction <= 1:
            raise ValueError('extension_fraction must lie in (0, 1]')
        on_observation_grid('recovery_s', self.recovery_s)
        on_observation_grid('amplification_window_s', self.amplification_window_s)
