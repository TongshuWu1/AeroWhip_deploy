"""Explicit PPO/SAC/task settings and immutable, portable experiment snapshots."""

from copy import deepcopy
import hashlib
import json
import math

from project_paths import ROOT, project_path
from simulation.configuration import load_configuration
from simulation.model_configuration import load_model_configuration
from tasks.whip.configuration import EpisodeTiming, WhipReturnConfiguration, LearnedRecoveryConfiguration
from tasks.whip.stage1_configuration import (Stage1Configuration, BendReleaseConfiguration,
    ReleaseMatchingConfiguration, StrikeSurfaceConfiguration)
from tasks.whip.workspace import CylinderWorkspace, WorkspacePenalty

SAC_DEFAULT = ROOT / 'configs/learning/whip_stage1_curriculum.json'
STAGED_PRESETS = {i: ROOT / f'configs/learning/whip_stage{i}_ppo_staged_v1.json' for i in (1, 2, 3)}
# Historical API/presets remain available for exact legacy tests and replay.
# Standard CLI/stage selector uses the explicitly versioned, untrained cone recipes.
FROZEN_PRESETS = {i: ROOT / f'configs/learning/whip_stage{i}_ppo_cone25_v4.json' for i in (1, 2, 3, 4)}
DEFAULT = STAGED_PRESETS[1]
RUN_SCHEMA = 'sac_stage1_velocity_v2'
CHECKPOINT_SCHEMA = 'sac_stage1_checkpoint_v7'


def algorithm(settings):
    return settings.get('algorithm', 'sac')  # Frozen SAC snapshots omit this key.


def training_stage(settings):
    return settings.get('training_stage', 2 if 'target_curriculum' in settings else 1)


def require_launchable(settings):
    # Keep saved contracts readable for inference/replay, not runnable training.
    if 'target_sampler' in settings or 'stage3_actor_training' in settings:
        raise ValueError('Multi-target training and combined goal/head adaptation are retired. '
                         'Use the current fixed-target stage preset for new training; '
                         'historical checkpoints remain available for inference and replay.')
    # No fabricated uncertainty ranges or nominal run mislabeled as robustness.
    if training_stage(settings) == 3 and settings.get('stage3_role') != 'precision_v1':
        raise ValueError('Stage 3 is a blocked robustness template: collect/identify new data, approve uncertainty ranges, then implement and verify the randomization sampler before training')
    if training_stage(settings) == 4 and settings.get('stage4_role') != 'learned_recovery_v1':
        raise ValueError('Stage4 requires explicit learned_recovery_v1')


def run_schema(settings):
    return 'ppo_stage1_velocity_v1' if algorithm(settings) == 'ppo' else RUN_SCHEMA


def checkpoint_schema(settings):
    return 'ppo_stage1_checkpoint_v1' if algorithm(settings) == 'ppo' else CHECKPOINT_SCHEMA


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, allow_nan=False).encode()).hexdigest()


def episode_budget(settings):
    return settings['training_episodes']


def iteration_budget(settings):
    total, width = episode_budget(settings), settings['parallel_environments']
    return (total+width-1)//width


def evaluation_settings(settings):
    return dict(settings, parallel_environments=settings.get(
        'evaluation_environments', settings['parallel_environments']))


def collection_updates(settings, before, after):
    # Integer cumulative allocation also handles a final partial collection.
    updates, width = settings['gradient_updates_per_round'], settings['parallel_environments']
    return after*updates//width-before*updates//width


def evaluation_due(settings, iteration, before, after):
    if iteration == 1 or after == episode_budget(settings):
        return True
    interval = settings['evaluation_interval_episodes']
    return before//interval != after//interval


def validate(settings):
    from training.curriculum import validate_curriculum
    required = {'task', 'model', 'scene', 'device', 'seed', 'parallel_environments',
        'training_episodes', 'evaluation_interval_episodes', 'policy_rate_hz',
        'action_interface', 'velocity_limit_mps', 'workspace_penalty', 'learning_rate',
        'batch_size', 'hidden_units', 'initial_log_std', 'discount'}
    method = algorithm(settings)
    if method not in ('sac', 'ppo'):
        raise ValueError('Select sac or ppo explicitly')
    if method == 'sac':
        required |= {'replay_episodes', 'gradient_updates_per_round',
                     'minimum_replay_transitions', 'initial_temperature', 'target_entropy', 'target_update_tau'}
    else:
        required |= {'algorithm', 'ppo'}
    missing = required - settings.keys()
    optional = {'algorithm', 'evaluation_environments', 'reward_scale'}
    if method == 'sac': optional |= {'curriculum', 'exploration'}
    else: optional |= {'target_curriculum', 'training_stage', 'domain_randomization', 'stage3_role', 'stage4_role', 'stage4_recovery_policy', 'plateau_detection', 'target_observation', 'stage3_actor_training', 'target_sampler'}
    unknown = settings.keys() - required - optional
    if missing or unknown:
        raise ValueError(f'Invalid learning settings: missing={sorted(missing)}, unknown={sorted(unknown)}')
    if method == 'ppo':
        from training.ppo import validate_stage3_actor_training
        validate_stage3_actor_training(settings)
    from tasks.whip.target_sampling import validate_target_sampler
    validate_target_sampler(settings)
    if 'target_observation' in settings and (
            method != 'ppo' or settings.get('training_stage') not in (1, 2, 3, 4)
            or settings['target_observation'] != 'initial_relative_position_v1'):
        raise ValueError('Relative target observations require explicit PPO Stage1/2/3/4 initial_relative_position_v1')
    if 'training_stage' in settings:
        stage = settings['training_stage']
        if type(stage) is not int or stage not in (1, 2, 3, 4) or method != 'ppo':
            raise ValueError('Explicit training_stage requires PPO stage 1, 2, 3 or 4')
        precision = stage == 3 and settings.get('stage3_role') == 'precision_v1'
        recovery = stage == 4 and settings.get('stage4_role') == 'learned_recovery_v1'
        if ('stage4_role' in settings and not recovery) or (stage == 4 and not recovery):
            raise ValueError('Stage4 requires explicit learned_recovery_v1 role')
        if recovery and ('target_curriculum' in settings or 'domain_randomization' in settings):
            raise ValueError('Nominal recovery Stage4 uses a fixed target, no curriculum/randomization')
        if 'stage3_role' in settings and not precision:
            raise ValueError('stage3_role requires explicit Stage3 precision_v1')
        if 'target_curriculum' in settings and stage != 2 and not precision:
            raise ValueError('Only explicit Stage 2 uses the narrowing target curriculum')
        if precision:
            if 'domain_randomization' in settings or 'target_curriculum' not in settings:
                raise ValueError('Precision Stage3 requires a target curriculum, not domain randomization')
        elif stage == 3:
            if settings.get('domain_randomization') != {'status': 'pending_identification', 'ranges': None}:
                raise ValueError('Stage 3 randomization is pending identified and approved uncertainty ranges')
        elif 'domain_randomization' in settings:
            raise ValueError('Domain randomization belongs to Stage 3')
    elif 'domain_randomization' in settings or 'stage3_role' in settings or 'stage4_role' in settings:
        raise ValueError('Randomization requires an explicit Stage 3 label')
    validate_curriculum(settings)
    if 'stage4_recovery_policy' in settings:
        c = settings['stage4_recovery_policy']
        if settings.get('training_stage') != 4 or settings.get('stage4_role') != 'learned_recovery_v1':
            raise ValueError('Frozen whip recovery is Stage4 only')
        if set(c) != {'architecture','position_gain_s_inv','velocity_damping','cable_length_m','initial_log_std'} or c['architecture'] != 'frozen_whip_pd_residual_v1':
            raise ValueError('Unknown recovery policy architecture/settings')
        for name in ('position_gain_s_inv','velocity_damping','cable_length_m'):
            if isinstance(c[name],bool) or not math.isfinite(c[name]) or c[name]<=0:
                raise ValueError('Invalid recovery policy '+name)
        if isinstance(c['initial_log_std'],bool) or not math.isfinite(c['initial_log_std']) or not -5<=c['initial_log_std']<=2:
            raise ValueError('Invalid recovery exploration')
    from training.curriculum import validate_target_curriculum
    validate_target_curriculum(settings)
    WorkspacePenalty(**settings['workspace_penalty'])
    if settings.get('action_interface') != 'velocity_v1':
        raise ValueError('Use current velocity settings; old interfaces require their saved implementation')
    if any(k in settings for k in ('body_rate_limit_rad_s','rate_gain_s_inv','acceleration_limit_mps2')):
        raise ValueError('Remove retired CTBR/acceleration settings from velocity configuration')
    integers = ('seed', 'parallel_environments', 'training_episodes', 'evaluation_interval_episodes',
        'policy_rate_hz', 'batch_size', 'hidden_units') + tuple(k for k in
        ('replay_episodes', 'gradient_updates_per_round', 'minimum_replay_transitions') if method == 'sac') + tuple(k for k in
        ('evaluation_environments',) if k in settings)
    for name in integers:
        value = settings[name]
        if type(value) is not int or value < (0 if name == 'seed' else 1):
            raise ValueError('Invalid integer setting: ' + name)
    if 100 % settings['policy_rate_hz']:
        raise ValueError('Policy rate must divide the 100 Hz observation clock')
    for name in ('learning_rate',) + (('initial_temperature', 'target_update_tau') if method == 'sac' else ()):
        if not math.isfinite(settings[name]) or settings[name] <= 0:
            raise ValueError('Expected positive finite ' + name)
    if method == 'sac' and not 0 < settings['target_update_tau'] <= 1:
        raise ValueError('Target update tau must lie in (0,1]')
    if settings['discount'] != 1.0:
        raise ValueError('Redistributed finite-horizon rewards require discount=1 to preserve the task return')
    scale = settings.get('reward_scale', 1.0)  # Historical snapshots use original units.
    if isinstance(scale, bool) or not isinstance(scale, (int, float)) or not math.isfinite(scale) or scale <= 0:
        raise ValueError('Reward scale must be positive and finite')
    if not -5 <= settings['initial_log_std'] <= 2 or (method == 'sac' and not math.isfinite(settings['target_entropy'])):
        raise ValueError('Invalid entropy settings')
    exploration = settings.get('exploration')
    if exploration is not None:
        if set(exploration) != {'initial_target_entropy', 'hold_transitions', 'anneal_end_transitions'}:
            raise ValueError('Unexpected exploration schedule settings')
        initial = exploration['initial_target_entropy']
        # Differential entropy of normalized three-dimensional [-1,1] actions.
        if (isinstance(initial, bool) or not math.isfinite(initial)
                or not settings['target_entropy'] < initial < 3*math.log(2)):
            raise ValueError('Entropy schedule must decrease from a feasible initial target')
        hold, end = exploration['hold_transitions'], exploration['anneal_end_transitions']
        if type(hold) is not int or type(end) is not int or not 0 <= hold < end:
            raise ValueError('Require 0 <= hold transitions < anneal end transitions')
    if settings['device'] not in ('cpu', 'cuda'):
        raise ValueError('Device must be cpu or cuda')
    for name in ('velocity_limit_mps',):
        v = settings[name]
        if len(v) != 3 or any(not math.isfinite(x) or x <= 0 for x in v):
            raise ValueError('Expected three positive finite values: ' + name)
    if method == 'sac' and settings['replay_episodes'] < settings['parallel_environments']:
        raise ValueError('Replay must hold at least one parallel collection round')
    if method == 'ppo':
        from training.ppo import validate_ppo
        validate_ppo(settings['ppo'])
        from training.plateau import validate_plateau
        validate_plateau(settings)


def task_configuration(raw, settings, model, scene):
    recovery_mode = training_stage(settings) == 4 and settings.get('stage4_role') == 'learned_recovery_v1'
    sections = {'episode', 'workspace', 'return', 'mechanics', 'request', 'matching', 'strike_surface'}
    if set(raw) != sections | ({'recovery'} if recovery_mode else set()):
        raise ValueError('Unexpected task sections; use the current episode-based task configuration')
    recovery = LearnedRecoveryConfiguration(**raw['recovery']) if recovery_mode else None
    if 'stage4_recovery_policy' in settings:
        if recovery is None or recovery.quality_mode != 'return_and_cable_settle_v1':
            raise ValueError('Frozen whip recovery requires versioned settling task')
        if abs(settings['stage4_recovery_policy']['cable_length_m']-sum(model['cable']['marker_interval_lengths_m']))>1e-10:
            raise ValueError('Recovery position decoding must match physical cable length')
    if set(raw['episode'])-{'termination'} != {'duration_s', 'settling_s'}:
        raise ValueError('Episode settings require duration_s, settling_s and optional termination')
    timing = EpisodeTiming(raw['episode']['duration_s'], raw['episode']['settling_s'])
    for t in (timing.duration_s, timing.post_hold_s):
        if not math.isfinite(t) or t <= 0 or abs(t * settings['policy_rate_hz'] - round(t * settings['policy_rate_hz'])) > 1e-8:
            raise ValueError('Episode intervals must be positive whole policy periods')
    delay = model['controller']['command_delay_s']
    if abs(delay / .01 - round(delay / .01)) > 1e-8:
        raise ValueError('This learning adapter requires delay on the 100 Hz grid')
    workspace = CylinderWorkspace(**raw['workspace'])
    returns = WhipReturnConfiguration(**raw['return'])
    matching = ReleaseMatchingConfiguration(**raw['matching'])
    # Current first-hit rewards accept crossings throughout the full episode.
    # Retained snapshots without this explicit selection keep old search bounds.
    event_end = (timing.duration_s+timing.post_hold_s-raw['mechanics']['tip_lag_s']
                 if (matching.strike_mechanics_scale is not None
                     or matching.reward_mode in ('directed_hit_v1', 'geometric_hit_quality_v1', 'directed_whip_v1', 'directed_delivery_v1', 'directed_precision_v1')) else delay+timing.duration_s)
    if recovery is not None:
        event_end = recovery.whip_duration_s-raw['mechanics']['tip_lag_s']
    mechanics = BendReleaseConfiguration(**raw['mechanics'], event_start_s=delay,
                                        event_end_s=event_end)
    cfg = Stage1Configuration(settings['scene'], settings['model'], timing, workspace,
        returns, tuple(raw['request']), mechanics, matching,
        StrikeSurfaceConfiguration(**raw['strike_surface']), raw['episode'].get('termination', 'full_horizon_v1'), recovery)
    if matching.reward_mode == 'directed_whip_v1' and (training_stage(settings) != 1
            or 'target_curriculum' in settings or 'curriculum' in settings):
        raise ValueError('Directed-whip discovery is Stage1 only, without automatic target curriculum')
    if matching.reward_mode == 'directed_delivery_v1' and (training_stage(settings) != 2
            or 'target_curriculum' in settings or 'curriculum' in settings):
        raise ValueError('Directed delivery is fixed-target Stage2 only, without automatic curriculum')
    precision = training_stage(settings) == 3 and settings.get('stage3_role') == 'precision_v1'
    if (matching.reward_mode == 'directed_precision_v1') != (precision or recovery_mode):
        raise ValueError('Directed precision reward requires explicit precision Stage3 or learned recovery Stage4')
    if recovery_mode and (cfg.strike_surface.width_m != .1 or cfg.strike_surface.height_m != .1
            or matching.center_sigma_m != .05):
        raise ValueError('Recovery Stage4 preserves fixed +/-5cm precision target and sigma')
    if training_stage(settings) == 2 and 'target_curriculum' not in settings and matching.reward_mode != 'directed_delivery_v1':
        raise ValueError('Stage2 without target curriculum requires directed_delivery_v1')
    if ('training_stage' in settings and matching.reward_mode not in ('directed_whip_v1', 'directed_delivery_v1', 'directed_precision_v1')
            and cfg.termination not in ('first_qualified_hit_v1', 'first_geometric_hit_v1')):
        raise ValueError('Staged presets require explicit first-hit termination')
    from training.curriculum import validate_curriculum
    validate_curriculum(settings, cfg)
    from training.curriculum import validate_target_curriculum
    validate_target_curriculum(settings, cfg)
    return cfg


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
    return settings, dict(schema=run_schema(settings), reward_delivery='causal_best_event_differences_v1',
        settings=deepcopy(settings), task=task,
        model=model, model_sources=sources, scene_text=scene_text), cfg, scene
