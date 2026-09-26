"""Explicit settings for the isolated, fixed-request Stage-1 objective."""

from copy import deepcopy
from dataclasses import dataclass, fields
import json
import math
from pathlib import Path
import tomllib

from .configuration import TrainingConfiguration
from .reward import WhipReturnConfiguration
from .skill import SKILL_PARAMETERS
from .workspace import CylinderWorkspace
from simulation.configuration import load_configuration


ROOT = Path(__file__).resolve().parents[2]


def resolve_stage1_settings(raw, *, scene, model):
    """Resolve omitted dependent fields; never rewrite explicit saved settings.

    The returned mapping is complete and is retained in the run protocol.
    Physical/reward choices (including the wall and post-hold) remain explicit.
    """
    resolved = deepcopy(raw)
    trajectory, mechanics = resolved['trajectory'], resolved['mechanics']
    workspace, returns = resolved['workspace'], resolved['return']
    initial = scene.initial_state.position_world_m
    delay = model['controller']['command_delay_s']
    if isinstance(delay, bool) or not math.isfinite(delay) or delay < 0:
        raise ValueError('Model command delay must be finite and nonnegative')

    def pair(section, lower, upper, values):
        present = (lower in section, upper in section)
        if present[0] != present[1]:
            raise ValueError(f'Provide both {lower} and {upper}, or omit both for automatic values')
        if not present[0]:
            section[lower], section[upper] = values

    pair(mechanics, 'event_start_s', 'event_end_s',
         (delay, delay+trajectory['duration_s']))
    if (not math.isclose(mechanics['event_start_s'], delay, rel_tol=0, abs_tol=1e-12)
            or not math.isclose(mechanics['event_end_s'], delay+trajectory['duration_s'], rel_tol=0, abs_tol=1e-12)):
        raise ValueError('Event bounds must match model delay and maneuver duration; omit both event bounds to derive them')
    radius = workspace['radius_m']
    lower_z, upper_z = workspace['height_lower_world_m'], workspace['height_upper_world_m']
    if trajectory.get('control_mapping', 'box') == 'cylinder_v1':
        lower = [-radius, -radius, float(lower_z-initial[2])]
        upper = [radius, radius, float(upper_z-initial[2])]
        pair(trajectory, 'offset_lower_m', 'offset_upper_m', (lower, upper))
        if any(not math.isclose(a, b, rel_tol=0, abs_tol=1e-12)
               for name, expected in (('offset_lower_m', lower), ('offset_upper_m', upper))
               for a, b in zip(trajectory[name], expected)):
            raise ValueError('Cylinder offsets must match workspace and initial position; omit both offsets to derive them')
    elif 'offset_lower_m' not in trajectory or 'offset_upper_m' not in trajectory:
        raise ValueError('Box mapping requires explicit trajectory offsets')
    pair(returns, 'drone_lower_world_m', 'drone_upper_world_m',
         ([float(initial[0]-radius), float(initial[1]-radius), lower_z],
          [float(initial[0]+radius), float(initial[1]+radius), upper_z]))
    return resolved


def load_stage1_settings(path, *, root=ROOT):
    """Resolve a user preset using its selected scene/model, without simulation."""
    with Path(path).open('rb') as stream:
        raw = tomllib.load(stream)
    root = Path(root)
    scene = load_configuration(root/raw['experiment']['scene'])
    model = json.loads((root/raw['experiment']['model']).read_text(encoding='utf-8'))
    resolved = resolve_stage1_settings(raw, scene=scene, model=model)
    return resolved, Stage1Configuration.from_mapping(resolved)


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
    strike_bonus_weight: float = 0.0

    def __post_init__(self):
        for field in fields(self):
            if field.name != 'strike_bonus_weight':
                _positive(getattr(self, field.name), field.name)
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
            raise ValueError('post_hold_s must cover the model command delay plus mechanics.tip_lag_s')
        if any(window > self.mechanics.event_end_s-self.mechanics.event_start_s+1e-12
               for window in self.mechanics.windows_s):
            raise ValueError('Every configured bend duration must fit the event search interval')

    @classmethod
    def load(cls, path):
        return load_stage1_settings(path)[1]

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
