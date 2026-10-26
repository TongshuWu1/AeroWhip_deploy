"""Task-only scene, episode, workspace and Stage-1 reward settings."""

from dataclasses import dataclass, fields
import math

from .configuration import EpisodeTiming, WhipReturnConfiguration
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
    # Explicit same-q Stage2 guidance width; absent in all historical tasks.
    delivery_sigma_m: float | None = None
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
        if self.reward_mode != 'continuous_directional_whip_four_second_v6':
            raise ValueError('Retired reward mode; only current Stage 1 is supported')


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
    recovery: None = None
    horizontal_fold: dict | None = None
    retained_fold_delivery: dict | None = None
    continuous_whip: dict | None = None

    def __post_init__(self):
        from .continuous_whip import SHORT_MODE, SHORT_CONTRACT
        if (self.matching.reward_mode != SHORT_MODE or self.continuous_whip != SHORT_CONTRACT
                or self.termination != 'continuous_whip_horizon_v1'
                or self.episode.duration_s != 4 or self.episode.post_hold_s != 0
                or self.returns.acceleration_cost_weight != 0 or self.returns.failure_return is not None
                or self.returns.failure_cost_per_remaining_s != 1.
                or self.matching.time_preference != 0 or self.matching.center_weight != 0
                or self.strike_surface.strike_bonus_weight != 0
                or self.recovery is not None or self.horizontal_fold is not None or self.retained_fold_delivery is not None):
            raise ValueError('Stage 1 differs from the approved four-second contract')
        if not all(isinstance(p,str) and p for p in (self.scene,self.model)):
            raise ValueError('Scene/model paths are required')
        if len(self.request)!=4 or not all(math.isfinite(v) for v in self.request):
            raise ValueError('Four finite display request parameters required')
