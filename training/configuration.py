"""Current Stage 1 PPO configuration and immutable experiment snapshots.

Historical training/reward modes were retired on 2026-10-25. Restore Git commit
3c9a49e to inspect them; they are deliberately not accepted by this loader.
"""
from copy import deepcopy
import hashlib
import json
import math
from project_paths import ROOT, project_path
from simulation.configuration import load_configuration
from simulation.model_configuration import load_model_configuration
from tasks.whip.configuration import EpisodeTiming, WhipReturnConfiguration
from tasks.whip.stage1_configuration import Stage1Configuration
from tasks.whip.workspace import CylinderWorkspace, WorkspacePenalty
from tasks.whip.continuous_whip import SHORT_MODE, SHORT_CONTRACT

DEFAULT = ROOT / 'configs/learning/whip_stage1_ppo_continuous_directional_four_second_v6.json'

def require_launchable(settings):
    if settings.get('continuous_whip_contract') != SHORT_MODE:
        raise ValueError('Only the current four-second +X Stage 1 PPO task is supported')

def run_schema(settings):
    return 'ppo_stage1_acceleration_direct_v1'

def checkpoint_schema(settings):
    return 'ppo_stage1_checkpoint_v1'

def action_limit_key(settings):
    if settings.get('action_interface') != 'acceleration_direct_v1':
        raise ValueError('Current Stage 1 uses acceleration_direct_v1')
    return 'acceleration_limit_mps2'

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

def validate(settings):
    require_launchable(settings)
    required = {'continuous_whip_contract','task','model','scene',
        'device','seed','parallel_environments','training_episodes','evaluation_interval_episodes',
        'policy_rate_hz','action_interface','acceleration_limit_mps2','workspace_penalty',
        'learning_rate','batch_size','hidden_units','initial_log_std','discount','ppo'}
    missing = required - settings.keys()
    unknown = settings.keys() - required - {'evaluation_environments','reward_scale'}
    if missing or unknown:
        raise ValueError(f'Invalid learning settings: missing={sorted(missing)}, unknown={sorted(unknown)}')
    action_limit_key(settings)
    for name in ('seed','parallel_environments','training_episodes','evaluation_interval_episodes',
                 'policy_rate_hz','batch_size','hidden_units','evaluation_environments'):
        if name not in settings: continue
        if type(settings[name]) is not int or settings[name] < (0 if name=='seed' else 1):
            raise ValueError('Invalid integer setting: '+name)
    if settings['device'] not in ('cpu','cuda'):
        raise ValueError('Device must be cpu or cuda')
    if not math.isfinite(settings['learning_rate']) or settings['learning_rate']<=0:
        raise ValueError('Positive finite learning rate required')
    if not -5<=settings['initial_log_std']<=2:
        raise ValueError('Invalid initial log standard deviation')
    if (settings['discount'] != 1 or settings.get('reward_scale',1) != 1
            or settings['policy_rate_hz'] != 50 or settings['acceleration_limit_mps2'] != [8,8,4]
            or settings['workspace_penalty'] != dict(mode='squared_excess_v1',weight=.1,distance_scale_m=.1)
            or settings['ppo'].get('entropy') != dict(mode='exponential_budget_v1',initial_weight=.02,final_weight=.001,decay_exponent=8.)
            or settings['ppo'].get('action_noise') != dict(mode='ar1_v1',correlation_time_s=.1)):
        raise ValueError('Current Stage 1 requires its approved control, cost and exploration contract')
    WorkspacePenalty(**settings['workspace_penalty'])
    from training.ppo import validate_ppo
    validate_ppo(settings['ppo'])

def task_configuration(raw, settings, model, scene):
    validate(settings)
    if set(raw) != {'episode','workspace','return','continuous_whip'}:
        raise ValueError('Unexpected Stage 1 task sections')
    if (raw['episode'] != dict(duration_s=4)
            or raw['continuous_whip'] != SHORT_CONTRACT
            or raw['workspace'] != dict(radius_m=.55,height_lower_world_m=1.5,height_upper_world_m=2.6)
            or model['simulation']['segments_per_marker_interval'] != [1]*10):
        raise ValueError('Current Stage 1 requires its four-second ten-link contract')
    mass = model['drone']['mass_kg']
    trim = -(mass+model['cable']['total_mass_kg'])*model['simulation']['gravity_world_m_s2'][2]/mass
    if settings['acceleration_limit_mps2'][2]>=trim:
        raise ValueError('Downward acceleration limit must keep desired vertical force positive')
    delay=model['controller']['command_delay_s']
    if abs(delay/.01-round(delay/.01))>1e-8:
        raise ValueError('Command delay must lie on the 100 Hz grid')
    returns=WhipReturnConfiguration(**raw['return'])
    if returns.maximum_tilt_deg!=60 or returns.minimum_cable_height_m!=.05:
        raise ValueError('Physical diagnostic limits must remain unchanged')
    return Stage1Configuration(settings['scene'],settings['model'],EpisodeTiming(4),
        CylinderWorkspace(**raw['workspace']),returns,raw['continuous_whip'])

def load(path=DEFAULT):
    settings = deepcopy(path) if isinstance(path, dict) else json.loads(project_path(path).read_text(encoding='utf-8-sig'))
    validate(settings)
    task = json.loads(project_path(settings['task']).read_text(encoding='utf-8-sig'))
    model, sources = load_model_configuration(settings['model'], with_sources=True)
    # Fitted correction promotion is separate from starting policy training.
    if model['residual']['status'] != 'unfitted':
        raise ValueError('Learning currently requires the unfitted correction; model promotion requires separate approval')
    scene_text = project_path(settings['scene']).read_text(encoding='utf-8-sig')
    scene = load_configuration(project_path(settings['scene']))
    cfg = task_configuration(task, settings, model, scene)
    delivery = 'continuous_fold_rate_v1'
    return settings, dict(schema=run_schema(settings), reward_delivery=delivery,
        settings=deepcopy(settings), task=task,
        model=model, model_sources=sources, scene_text=scene_text), cfg, scene
