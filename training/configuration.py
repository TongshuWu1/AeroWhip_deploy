"""Stage 1 experiment files: one file per experiment, validated by type and range.

An experiment file holds the run, control and PPO settings plus an inline
"task" section (typed by tasks.whip.configuration.Stage1TaskConfig). The
physical model and scene stay in their own files because simulation, fitting
and deployment share them; the experiment references them by path.

Runs are frozen as snapshots: settings (the file without "task"), the task
section, the resolved model and the scene text.
"""
from copy import deepcopy
import hashlib
import json
import math
from project_paths import ROOT, project_path
from simulation.configuration import load_configuration
from simulation.model_configuration import load_model_configuration
from tasks.whip.configuration import OBSERVATION_PERIOD_S, Stage1TaskConfig

DEFAULT = ROOT / 'configs/experiments/stage1_fold_x_v6.json'
ACTION_INTERFACE = 'acceleration_direct_v1'

SETTINGS = {'model', 'scene', 'device', 'seed', 'parallel_environments', 'evaluation_environments',
            'training_episodes', 'evaluation_interval_episodes', 'policy_rate_hz', 'action_interface',
            'acceleration_limit_mps2', 'learning_rate', 'batch_size', 'hidden_units', 'initial_log_std',
            'discount', 'reward_scale', 'ppo'}


def run_schema(settings):
    return 'ppo_stage1_acceleration_direct_v1'


def checkpoint_schema(settings):
    return 'ppo_stage1_checkpoint_v1'


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, allow_nan=False).encode()).hexdigest()


def episode_budget(settings):
    return settings['training_episodes']


def iteration_budget(settings):
    total, width = episode_budget(settings), settings['parallel_environments']
    return (total+width-1)//width


def evaluation_due(settings, iteration, before, after):
    if iteration == 1 or after == episode_budget(settings):
        return True
    interval = settings['evaluation_interval_episodes']
    return before//interval != after//interval


def collection_limit(settings):
    return iteration_budget(settings)


def _finite(name, value):
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
        raise ValueError(f'{name} must be a finite number')
    return value


def validate(settings):
    """Run, control and learner settings: required keys, types and ranges."""
    missing, unknown = SETTINGS-settings.keys(), settings.keys()-SETTINGS
    if missing or unknown:
        raise ValueError(f'Invalid experiment settings: missing={sorted(missing)}, unknown={sorted(unknown)}')
    for name in ('seed', 'parallel_environments', 'evaluation_environments', 'training_episodes',
                 'evaluation_interval_episodes', 'policy_rate_hz', 'batch_size', 'hidden_units'):
        if type(settings[name]) is not int or settings[name] < (0 if name == 'seed' else 1):
            raise ValueError('Invalid integer setting: '+name)
    if settings['device'] not in ('cpu', 'cuda'):
        raise ValueError('Device must be cpu or cuda')
    if settings['action_interface'] != ACTION_INTERFACE:
        raise ValueError(f'Only the {ACTION_INTERFACE} action interface is implemented')
    stride = 1/(OBSERVATION_PERIOD_S*settings['policy_rate_hz'])
    if abs(stride-round(stride)) > 1e-9:
        raise ValueError('policy_rate_hz must divide the 100 Hz observation rate')
    limits = settings['acceleration_limit_mps2']
    if not isinstance(limits, list) or len(limits) != 3 or any(_finite('acceleration limit', v) <= 0 for v in limits):
        raise ValueError('acceleration_limit_mps2 needs three positive XYZ limits [m/s²]')
    if _finite('learning_rate', settings['learning_rate']) <= 0:
        raise ValueError('Positive learning rate required')
    if not -5 <= _finite('initial_log_std', settings['initial_log_std']) <= 2:
        raise ValueError('initial_log_std must lie in [-5, 2]')
    if not 0 < _finite('discount', settings['discount']) <= 1:
        raise ValueError('discount must lie in (0, 1]')
    if _finite('reward_scale', settings['reward_scale']) <= 0:
        raise ValueError('reward_scale must be positive')
    from training.ppo import validate_ppo
    validate_ppo(settings['ppo'])


def task_configuration(raw, settings, model, scene):
    """Typed task settings, checked against the model they will run on."""
    validate(settings)
    cfg = Stage1TaskConfig.from_dict(raw, scene=settings['scene'], model=settings['model'])
    if model['simulation']['segments_per_marker_interval'] != [1]*10:
        raise ValueError('The fold reward requires the ten-link mesh (one segment per marker interval)')
    mass = model['drone']['mass_kg']
    trim = -(mass+model['cable']['total_mass_kg'])*model['simulation']['gravity_world_m_s2'][2]/mass
    if settings['acceleration_limit_mps2'][2] >= trim:
        raise ValueError('Downward acceleration limit must keep desired vertical force positive')
    delay = model['controller']['command_delay_s']
    if abs(delay/OBSERVATION_PERIOD_S-round(delay/OBSERVATION_PERIOD_S)) > 1e-8:
        raise ValueError('Command delay must lie on the 100 Hz grid')
    return cfg


def load(source=DEFAULT):
    """Experiment file path or dict -> (settings, snapshot, task config, scene)."""
    experiment = deepcopy(source) if isinstance(source, dict) else json.loads(
        project_path(source).read_text(encoding='utf-8-sig'))
    if 'task' not in experiment:
        raise ValueError('Experiment needs an inline "task" section')
    task = experiment.pop('task')
    settings = experiment
    validate(settings)
    model, sources = load_model_configuration(settings['model'], with_sources=True)
    # Fitted correction promotion is separate from starting policy training.
    if model['residual']['status'] != 'unfitted':
        raise ValueError('Learning currently requires the unfitted correction; model promotion requires separate approval')
    scene_text = project_path(settings['scene']).read_text(encoding='utf-8-sig')
    scene = load_configuration(project_path(settings['scene']))
    cfg = task_configuration(task, settings, model, scene)
    return settings, dict(schema=run_schema(settings), reward_delivery='continuous_fold_rate_v1',
        settings=deepcopy(settings), task=task,
        model=model, model_sources=sources, scene_text=scene_text), cfg, scene


def experiment(settings, task):
    """Recombine settings and task into the single experiment-file layout."""
    return dict(deepcopy(settings), task=deepcopy(task))
