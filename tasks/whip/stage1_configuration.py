"""Task-only scene, episode, workspace and Stage-1 reward settings."""

from dataclasses import dataclass, fields
import math

from .configuration import EpisodeTiming, WhipReturnConfiguration, LearnedRecoveryConfiguration
from .workspace import CylinderWorkspace


def _positive(value, name):
    if isinstance(value, bool) or not math.isfinite(value) or value <= 0:
        raise ValueError(f'{name} must be finite and positive')


@dataclass(frozen=True)
class BendReleaseConfiguration:
    smoothing_sigma_fraction: float
    windows_s: tuple[float, ...]
    window_stride_s: float
    tip_lag_s: float
    event_start_s: float
    event_end_s: float
    turn_reference_rad: float
    travel_reference_fraction: float
    spread_scale_fraction: float
    nonrigid_reference_mps: float
    release_reference_mps: float
    forward_reference_mps: float
    rigid_pinv_rtol: float

    def __post_init__(self):
        for field in fields(self):
            if field.name not in ('windows_s', 'event_start_s'):
                _positive(getattr(self, field.name), field.name)
        windows = tuple(self.windows_s)
        if not windows or len(set(windows)) != len(windows):
            raise ValueError('Provide distinct positive event-window durations')
        for value in windows:
            _positive(value, 'window duration')
        object.__setattr__(self, 'windows_s', windows)
        if (not math.isfinite(self.event_start_s) or self.event_start_s < 0
                or self.event_end_s <= self.event_start_s or self.rigid_pinv_rtol >= 1):
            raise ValueError('Require ordered nonnegative event times and pinv rtol in (0,1)')
        for value in (*windows, self.window_stride_s, self.tip_lag_s,
                      self.event_start_s, self.event_end_s):
            if not math.isclose(value/.01, round(value/.01), rel_tol=0, abs_tol=1e-8):
                raise ValueError('Event times must be on the 100 Hz grid')


@dataclass(frozen=True)
class ReleaseMatchingConfiguration:
    height_width_m: float
    speed_width_mps: float
    direction_width_deg: float
    time_preference: float
    speed_mode: str = 'match'
    mechanics_credit_weight: float = 0.0
    # Bound supporting motion/approach credit independently of the strike bonus.
    # Default preserves historical task snapshots.
    shaping_weight: float = 1.0
    center_weight: float = 0.0
    center_sigma_m: float = 0.3
    # None preserves the exact historical M-weighted strike arithmetic and
    # maneuver-limited search. Current presets select a bounded hit-priority
    # bonus M/(M+scale); this scale is not a physical whip qualification.
    strike_mechanics_scale: float | None = None
    # Versioned experiment; omitted in retained tasks means legacy arithmetic.
    # directed_hit_v1 uses spatial record-progress, an unconditional qualified
    # hit bonus, and time_preference as a full-horizon elapsed-time cost.
    reward_mode: str = 'mechanics_event_v1'
    # Explicit geometric-hit experiment only; historical reward modes stay unchanged.
    hit_speed_weight: float = 0.0
    hit_straightness_weight: float = 0.0
    hit_speed_scale_mps: float = 3.0
    # Omitted fields retain byte-exact historical squared-cosine arithmetic.
    alignment_mode: str = 'squared_cosine_v1'
    alignment_cone_deg: float = 25.0
    # Explicit compact Stage1 experiment only. Omission preserves saved recipes.
    reach_weight: float | None = None
    preparation_path_weight: float | None = None
    # Explicit level-outreach Stage1 experiment; older recipes omit these.
    outreach_base_fraction: float | None = None
    outreach_height_width_fraction: float | None = None
    # Explicit demonstration-seeded, speed/outreach Stage1 experiment.
    outreach_mechanics_scale: float | None = None
    outreach_speed_scale_mps: float | None = None
    outreach_speed_power: float | None = None
    # Separate approved linear-height Stage1 experiment; omitted in old tasks.
    outreach_reach_weight: float | None = None
    outreach_height_weight: float | None = None
    outreach_joint_weight: float | None = None
    # Explicit crossing-priority Stage2 experiment; historical tasks omit these.
    crossing_whip_weight: float | None = None
    crossing_side_scale_m: float | None = None
    crossing_height_scale_m: float | None = None
    crossing_geometric_weight: float | None = None
    crossing_strict_weight: float | None = None

    def __post_init__(self):
        for name in ('crossing_whip_weight', 'crossing_side_scale_m',
                     'crossing_height_scale_m', 'crossing_geometric_weight',
                     'crossing_strict_weight'):
            value = getattr(self, name)
            if self.reward_mode == 'crossing_priority_delivery_v1':
                if value is None:
                    raise ValueError('Crossing-priority delivery requires explicit '+name)
                _positive(value, name)
            elif value is not None:
                raise ValueError(name+' requires crossing-priority delivery')
        if self.alignment_mode not in ('squared_cosine_v1', 'cone_plateau_v1'):
            raise ValueError('Unknown directed alignment mode')
        _positive(self.alignment_cone_deg, 'alignment_cone_deg')
        if self.alignment_cone_deg >= 90:
            raise ValueError('Alignment cone must be strictly below 90 degrees')
        if self.alignment_mode != 'squared_cosine_v1' and self.reward_mode not in (
                'directed_whip_v1', 'directed_compact_whip_v1', 'directed_level_outreach_whip_v1', 'directed_outreach_speed_whip_v1', 'directed_outreach_height_whip_v1', 'crossing_priority_delivery_v1', 'directed_outreach_speed_delivery_v1', 'directed_outreach_speed_precision_v1', 'directed_delivery_v1', 'directed_precision_v1'):
            raise ValueError('Cone alignment requires a directed whip reward')
        for name in ('height_width_m', 'speed_width_mps', 'direction_width_deg'):
            _positive(getattr(self, name), name)
        if not math.isfinite(self.time_preference) or not 0 <= self.time_preference < 1:
            raise ValueError('Time preference must be in [0,1)')
        if self.speed_mode not in ('match', 'minimum'):
            raise ValueError('Speed mode must be match or minimum')
        if (isinstance(self.mechanics_credit_weight, bool)
                or not math.isfinite(self.mechanics_credit_weight)
                or not 0 <= self.mechanics_credit_weight <= 1):
            raise ValueError('Mechanics credit weight must be finite and in [0,1]')
        if (isinstance(self.shaping_weight, bool) or not math.isfinite(self.shaping_weight)
                or not 0 <= self.shaping_weight <= 1):
            raise ValueError('Shaping weight must be finite and in [0,1]')
        if isinstance(self.center_weight, bool) or not math.isfinite(self.center_weight) or not 0 <= self.center_weight <= 1:
            raise ValueError('Center weight must be finite and in [0,1]')
        _positive(self.center_sigma_m, 'center_sigma_m')
        if self.strike_mechanics_scale is not None:
            _positive(self.strike_mechanics_scale, 'strike_mechanics_scale')
        _positive(self.hit_speed_scale_mps, 'hit_speed_scale_mps')
        for name in ('hit_speed_weight', 'hit_straightness_weight'):
            value = getattr(self, name)
            if isinstance(value, bool) or not math.isfinite(value) or not 0 <= value <= 1:
                raise ValueError(name+' must be finite and in [0,1]')
        if self.reward_mode != 'geometric_hit_quality_v1' and (self.hit_speed_weight or self.hit_straightness_weight):
            raise ValueError('Hit-quality weights require the geometric-hit reward mode')
        if self.reward_mode not in ('mechanics_event_v1', 'directed_hit_v1', 'geometric_hit_quality_v1', 'directed_whip_v1', 'directed_compact_whip_v1', 'directed_level_outreach_whip_v1', 'directed_outreach_speed_whip_v1', 'directed_outreach_height_whip_v1', 'crossing_priority_delivery_v1', 'directed_outreach_speed_delivery_v1', 'directed_outreach_speed_precision_v1', 'directed_delivery_v1', 'directed_precision_v1'):
            raise ValueError('Unknown reward mode')
        if self.reward_mode in ('directed_whip_v1', 'directed_compact_whip_v1', 'directed_level_outreach_whip_v1', 'directed_outreach_speed_whip_v1', 'directed_outreach_height_whip_v1', 'crossing_priority_delivery_v1', 'directed_outreach_speed_delivery_v1', 'directed_outreach_speed_precision_v1', 'directed_delivery_v1', 'directed_precision_v1') and (
                self.mechanics_credit_weight != 1 or self.shaping_weight != 1
                or self.time_preference != 0 or self.center_weight != 0
                or self.strike_mechanics_scale is not None):
            raise ValueError('Directed whip uses one unit mechanics event and no matching/time/center bonus')
        for name in ('reach_weight', 'preparation_path_weight'):
            value = getattr(self, name)
            modes = (('directed_compact_whip_v1',) if name == 'reach_weight' else
                     ('directed_compact_whip_v1', 'directed_level_outreach_whip_v1', 'directed_outreach_speed_whip_v1', 'directed_outreach_height_whip_v1', 'crossing_priority_delivery_v1', 'directed_outreach_speed_delivery_v1', 'directed_outreach_speed_precision_v1'))
            if self.reward_mode in modes:
                if (value is None or isinstance(value, bool) or not math.isfinite(value) or value < 0):
                    label = 'Compact Stage1' if self.reward_mode == 'directed_compact_whip_v1' else 'Level outreach Stage1'
                    raise ValueError(label+' requires explicit finite nonnegative '+name)
            elif value is not None:
                raise ValueError(name+' requires an applicable compact Stage1 reward mode')
        for name in ('outreach_base_fraction', 'outreach_height_width_fraction'):
            value = getattr(self, name)
            modes = (('directed_level_outreach_whip_v1',) if name == 'outreach_base_fraction' else
                     ('directed_level_outreach_whip_v1', 'directed_outreach_speed_whip_v1', 'directed_outreach_speed_delivery_v1', 'directed_outreach_speed_precision_v1'))
            if self.reward_mode in modes:
                if value is None or isinstance(value, bool) or not math.isfinite(value):
                    raise ValueError('Level outreach Stage1 requires explicit finite '+name)
                if (name == 'outreach_base_fraction' and not 0 <= value <= 1
                        or name == 'outreach_height_width_fraction' and value <= 0):
                    raise ValueError('Invalid level outreach Stage1 '+name)
            elif value is not None:
                raise ValueError(name+' requires the level outreach Stage1 reward mode')
        for name in ('outreach_mechanics_scale', 'outreach_speed_scale_mps', 'outreach_speed_power'):
            value = getattr(self, name)
            if self.reward_mode in ('directed_outreach_speed_whip_v1', 'directed_outreach_height_whip_v1', 'crossing_priority_delivery_v1', 'directed_outreach_speed_delivery_v1', 'directed_outreach_speed_precision_v1'):
                if value is None:
                    raise ValueError('Speed outreach Stage1 requires explicit '+name)
                _positive(value, name)
            elif value is not None:
                raise ValueError(name+' requires the speed outreach Stage1 reward mode')
        for name in ('outreach_reach_weight', 'outreach_height_weight', 'outreach_joint_weight'):
            value = getattr(self, name)
            if self.reward_mode in ('directed_outreach_height_whip_v1', 'crossing_priority_delivery_v1'):
                if value is None or isinstance(value, bool) or not math.isfinite(value) or value < 0:
                    raise ValueError('Linear outreach/height Stage1 requires explicit finite nonnegative '+name)
            elif value is not None:
                raise ValueError(name+' requires the linear outreach/height Stage1 reward mode')
        if self.reward_mode in ('directed_outreach_height_whip_v1', 'crossing_priority_delivery_v1'):
            total = self.outreach_reach_weight+self.outreach_height_weight+self.outreach_joint_weight
            if not math.isfinite(total) or total <= 0:
                raise ValueError('Linear outreach/height weights must have a finite positive sum')
        if self.reward_mode in ('directed_hit_v1', 'geometric_hit_quality_v1') and (
                self.mechanics_credit_weight != 0 or self.center_weight != 0
                or self.strike_mechanics_scale is not None):
            raise ValueError('Simple directed reward has no mechanics/centering multiplier')


@dataclass(frozen=True)
class StrikeSurfaceConfiguration:
    """Virtual vertical region, not a contact model or a whip classifier."""

    distance_m: float
    width_m: float
    height_m: float
    maximum_entry_angle_deg: float
    approach_width_m: float
    strike_bonus_weight: float = 0.0
    # None preserves the isotropic rectangle-distance matching in saved runs.
    overflow_width_m: float | None = None

    def __post_init__(self):
        for field in fields(self):
            if field.name not in ('strike_bonus_weight', 'overflow_width_m'):
                _positive(getattr(self, field.name), field.name)
        if self.overflow_width_m is not None:
            _positive(self.overflow_width_m, 'overflow_width_m')
        if (isinstance(self.strike_bonus_weight, bool) or not math.isfinite(self.strike_bonus_weight)
                or self.strike_bonus_weight < 0):
            raise ValueError('Strike bonus weight must be finite and nonnegative')
        if self.maximum_entry_angle_deg >= 90:
            raise ValueError('Entry angle must be strictly between zero and 90 degrees')

    def description(self, initial, request):
        heading, _, height, speed = request
        normal = [math.cos(heading), math.sin(heading), 0.]
        return dict(center_world_m=[float(initial[0])+self.distance_m*normal[0],
                    float(initial[1])+self.distance_m*normal[1], height], normal_world=normal,
                    width_m=self.width_m, height_m=self.height_m,
                    maximum_entry_angle_deg=self.maximum_entry_angle_deg,
                    minimum_forward_speed_mps=speed)


@dataclass(frozen=True)
class Stage1Configuration:
    scene: str
    model: str
    episode: EpisodeTiming
    workspace: CylinderWorkspace
    returns: WhipReturnConfiguration
    request: tuple[float, float, float, float]
    mechanics: BendReleaseConfiguration
    matching: ReleaseMatchingConfiguration
    strike_surface: StrikeSurfaceConfiguration | None = None
    termination: str = 'full_horizon_v1'
    recovery: LearnedRecoveryConfiguration | None = None

    def __post_init__(self):
        if self.termination not in ('full_horizon_v1', 'first_qualified_hit_v1', 'first_geometric_hit_v1'):
            raise ValueError('Unknown task termination rule')
        if self.matching.reward_mode in ('directed_whip_v1', 'directed_compact_whip_v1', 'directed_level_outreach_whip_v1', 'directed_outreach_speed_whip_v1', 'directed_outreach_height_whip_v1', 'crossing_priority_delivery_v1', 'directed_outreach_speed_delivery_v1', 'directed_outreach_speed_precision_v1', 'directed_delivery_v1', 'directed_precision_v1') and (
                self.termination != 'full_horizon_v1' or self.returns.acceleration_cost_weight != 0
                or (self.strike_surface is not None and self.strike_surface.strike_bonus_weight != 0)):
            raise ValueError('Directed whip requires full horizon, zero acceleration cost and zero strike bonus')
        if self.matching.reward_mode in ('directed_delivery_v1', 'directed_outreach_speed_delivery_v1', 'directed_outreach_speed_precision_v1', 'directed_precision_v1', 'crossing_priority_delivery_v1') and (
                self.strike_surface is None or self.strike_surface.overflow_width_m is not None):
            raise ValueError('Directed delivery requires a fixed rectangle and isotropic distance guidance')
        if self.termination in ('first_qualified_hit_v1', 'first_geometric_hit_v1') and self.strike_surface is None:
            raise ValueError('First-hit termination requires a directed wall')
        if self.matching.reward_mode == 'directed_hit_v1' and (
                self.termination != 'first_qualified_hit_v1' or self.strike_surface is None):
            raise ValueError('Simple directed reward requires a first-qualified-hit wall task')
        if (self.matching.reward_mode == 'geometric_hit_quality_v1') != (self.termination == 'first_geometric_hit_v1'):
            raise ValueError('Geometric-hit quality reward and first-geometric-hit termination must be selected together')
        if self.matching.strike_mechanics_scale is not None and (
                self.strike_surface is None or self.termination != 'first_qualified_hit_v1'):
            raise ValueError('Hit-priority reward requires a directed first-hit task')
        if any(not isinstance(value, str) or not value for value in (self.scene, self.model)):
            raise ValueError('Provide scene/model paths relative to the repository')
        request = tuple(self.request)
        if len(request) != 4 or not all(math.isfinite(value) for value in request):
            raise ValueError('Provide four finite release-request parameters')
        heading, elevation, _, speed = request
        if not (-math.pi <= heading <= math.pi and -math.pi/2 <= elevation <= math.pi/2 and speed > 0):
            raise ValueError('Require canonical heading/elevation and positive requested speed')
        object.__setattr__(self, 'request', request)
        if self.strike_surface is not None:
            if self.strike_surface.distance_m <= self.workspace.radius_m:
                raise ValueError('Strike surface must be outside the initial-position drone cylinder')
            if elevation != 0 or self.matching.speed_mode != 'minimum':
                raise ValueError('Directed Stage 1 requires a horizontal normal and minimum-speed mode')
        horizon = self.episode.duration_s+self.episode.post_hold_s
        if self.recovery is not None:
            r = self.recovery
            if (self.matching.reward_mode not in ('directed_precision_v1', 'directed_outreach_speed_precision_v1')
                    or self.termination != 'full_horizon_v1'
                    or not math.isclose(horizon,r.whip_duration_s+r.recovery_duration_s,abs_tol=1e-12)
                    or not math.isclose(self.mechanics.event_end_s+self.mechanics.tip_lag_s,r.whip_duration_s,abs_tol=1e-12)):
                raise ValueError('Learned recovery requires precision, full horizon and explicit separate whip window')
        if self.mechanics.event_end_s+self.mechanics.tip_lag_s > horizon+1e-12:
            raise ValueError('post_hold_s must cover the model command delay plus mechanics.tip_lag_s')
        if any(window > self.mechanics.event_end_s-self.mechanics.event_start_s+1e-12
               for window in self.mechanics.windows_s):
            raise ValueError('Every configured bend duration must fit the event search interval')
