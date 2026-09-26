"""Explicit settings for the isolated, fixed-request Stage-1 objective."""

from dataclasses import dataclass, fields
import math
from pathlib import Path
import tomllib

from .configuration import TrainingConfiguration
from .reward import WhipReturnConfiguration
from .skill import SKILL_PARAMETERS
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

    def __post_init__(self):
        for name in ('height_width_m', 'speed_width_mps', 'direction_width_deg'):
            _positive(getattr(self, name), name)
        if not math.isfinite(self.time_preference) or not 0 <= self.time_preference < 1:
            raise ValueError('Time preference must be in [0,1)')
        if self.speed_mode not in ('match', 'minimum'):
            raise ValueError('Speed mode must be match or minimum')


@dataclass(frozen=True)
class StrikeSurfaceConfiguration:
    """Virtual vertical region, not a contact model or a whip classifier."""

    distance_m: float
    width_m: float
    height_m: float
    maximum_entry_angle_deg: float
    approach_width_m: float

    def __post_init__(self):
        for field in fields(self):
            _positive(getattr(self, field.name), field.name)
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
    trajectory: TrainingConfiguration
    workspace: CylinderWorkspace
    returns: WhipReturnConfiguration
    request: tuple[float, float, float, float]
    mechanics: BendReleaseConfiguration
    matching: ReleaseMatchingConfiguration
    strike_surface: StrikeSurfaceConfiguration | None = None

    def __post_init__(self):
        self.returns.acceleration_scale(self.trajectory)
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
        horizon = self.trajectory.duration_s+self.trajectory.post_hold_s
        if self.mechanics.event_end_s+self.mechanics.tip_lag_s > horizon+1e-12:
            raise ValueError('The full event window and release lag must fit the episode')
        if any(window > self.mechanics.event_end_s-self.mechanics.event_start_s+1e-12
               for window in self.mechanics.windows_s):
            raise ValueError('Every configured bend duration must fit the event search interval')

    @classmethod
    def load(cls, path):
        with Path(path).open('rb') as stream:
            raw = tomllib.load(stream)
        return cls.from_mapping(raw)

    @classmethod
    def from_mapping(cls, raw):
        training = raw.get('experiment', {}).get('mode') == 'training'
        sections = {'experiment', 'trajectory', 'workspace', 'return', 'request', 'mechanics', 'matching'}
        if training:
            sections |= {'exploration', 'cem'}
        if 'strike_surface' in raw:
            sections.add('strike_surface')
        if set(raw) != sections:
            raise ValueError('Unexpected Stage-1 objective configuration sections')
        exp = raw['experiment']
        keys = {'name', 'mode', 'scene', 'model'}
        if training:
            keys |= {'device', 'batch_size', 'updates', 'seeds', 'evaluation_interval'}
        if (set(exp) != keys or exp['name'] != 'whip_stage1'
                or exp['mode'] not in ('objective_evaluation', 'training')):
            raise ValueError('Unexpected Stage-1 experiment fields or mode')
        if set(raw['request']) != set(SKILL_PARAMETERS[:4]):
            raise ValueError('Stage 1 requests direction, height and speed; exact release time is free')
        return cls(exp['scene'], exp['model'], TrainingConfiguration(**raw['trajectory']),
                   CylinderWorkspace(**raw['workspace']), WhipReturnConfiguration(**raw['return']),
                   tuple(raw['request'][name] for name in SKILL_PARAMETERS[:4]),
                   BendReleaseConfiguration(**raw['mechanics']), ReleaseMatchingConfiguration(**raw['matching']),
                   StrikeSurfaceConfiguration(**raw['strike_surface']) if 'strike_surface' in raw else None)
