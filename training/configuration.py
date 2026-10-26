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

# Retained velocity recipes; their checkpoints keep their original action units.
FROZEN_PRESETS = {i: ROOT / f'configs/learning/whip_stage{i}_ppo_cone25_v4.json' for i in (1, 2, 3, 4)}
FROZEN_PRESETS[1] = ROOT / 'configs/learning/whip_stage1_ppo_compact_v5.json'
DEFAULT = ROOT / 'configs/learning/whip_stage1_ppo_acceleration_v1.json'
CURRENT_PRESETS = dict(FROZEN_PRESETS)
CURRENT_PRESETS.update({i: ROOT / f'configs/learning/whip_stage{i}_ppo_acceleration_v1.json' for i in (1, 2, 3)})
CURRENT_PRESETS[2] = ROOT / 'configs/learning/whip_stage2_ppo_retained_fold_delivery_v1.json'
CURRENT_PRESETS[1] = ROOT / 'configs/learning/whip_stage1_ppo_continuous_directional_four_second_v6.json'
DEFAULT = CURRENT_PRESETS[1]
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
    if settings.get('action_interface') == 'acceleration_direct_v1':
        return 'ppo_stage1_acceleration_direct_v1'
    return 'ppo_stage1_velocity_v1' if algorithm(settings) == 'ppo' else RUN_SCHEMA


def action_limit_key(settings):
    interface = settings.get('action_interface')
    if interface == 'velocity_v1':
        return 'velocity_limit_mps'
    if interface == 'acceleration_direct_v1':
        return 'acceleration_limit_mps2'
    raise ValueError('Unknown policy action interface: '+str(interface))


def checkpoint_schema(settings):
    return 'ppo_stage1_checkpoint_v1' if algorithm(settings) == 'ppo' else CHECKPOINT_SCHEMA


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, allow_nan=False).encode()).hexdigest()


def episode_budget(settings):
    return settings['training_episodes']


def iteration_budget(settings):
    total, width = episode_budget(settings), settings['parallel_environments']
    return (total+width-1)//width


def collection_limit(settings):
    """Stop formation at its review boundary while retaining the full entropy budget."""
    phase=settings.get('fold_curriculum',{})
    return (phase['warmup_collections'] if phase.get('phase')=='formation_v1'
            else iteration_budget(settings))


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
        'action_interface', action_limit_key(settings), 'workspace_penalty', 'learning_rate',
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
    else: optional |= {'target_curriculum', 'training_stage', 'domain_randomization', 'stage3_role', 'stage4_role', 'stage4_recovery_policy', 'plateau_detection', 'movement_penalty', 'target_observation', 'stage3_actor_training', 'target_sampler', 'fold_curriculum', 'stage1_return_transform', 'stage2_return_transform', 'stage2_first_hit', 'horizontal_fold_contract', 'retained_fold_delivery_contract', 'continuous_whip_contract'}
    unknown = settings.keys() - required - optional
    if missing or unknown:
        raise ValueError(f'Invalid learning settings: missing={sorted(missing)}, unknown={sorted(unknown)}')
    if 'fold_curriculum' in settings:
        c=settings['fold_curriculum']
        if (not isinstance(c,dict) or c.get('phase') not in ('formation_v1','release_v1')
                or set(c)!=({'phase','warmup_collections','total_collections'} |
                            ({'motion_review'} if c.get('phase')=='release_v1' else set()))
                or type(c.get('warmup_collections')) is not int or c['warmup_collections']!=30
                or type(c.get('total_collections')) is not int or c['total_collections']!=120
                or settings['parallel_environments']!=128 or training_stage(settings)!=1
                or 'plateau_detection' in settings or 'curriculum' in settings or 'target_curriculum' in settings):
            raise ValueError('Fold curriculum requires the approved30/120 Stage1 review plan at128 lanes')
        if c['phase']=='formation_v1' and settings['training_episodes']!=15360:
            raise ValueError('Formation keeps the full15360-episode entropy budget and stops at review')
        if c['phase']=='release_v1' and (not isinstance(c['motion_review'],str) or not c['motion_review']):
            raise ValueError('Release requires an explicit formation motion review')
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
    if 'movement_penalty' in settings:
        from tasks.whip.movement import MovementPenalty
        MovementPenalty(**settings['movement_penalty'])

    if settings['action_interface'] == 'acceleration_direct_v1':
        if method != 'ppo' or training_stage(settings) == 4 or 'stage4_recovery_policy' in settings:
            raise ValueError('Direct acceleration supports PPO Stages 1–3; velocity recovery is incompatible')
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
    for name in (action_limit_key(settings),):
        v = settings[name]
        if (not isinstance(v, (list, tuple)) or len(v) != 3 or
                any(isinstance(x, bool) or not isinstance(x, (int, float)) or not math.isfinite(x) or x <= 0 for x in v)):
            raise ValueError('Expected three positive finite values: ' + name)
    if method == 'sac' and settings['replay_episodes'] < settings['parallel_environments']:
        raise ValueError('Replay must hold at least one parallel collection round')
    if method == 'ppo':
        from training.ppo import validate_ppo
        validate_ppo(settings['ppo'])
        from training.plateau import validate_plateau
        validate_plateau(settings)


def task_configuration(raw, settings, model, scene):
    if settings.get('action_interface') == 'acceleration_direct_v1':
        # Positive vertical desired force keeps attitude construction nonsingular
        # for every action in the configured box, including zero lateral input.
        mass = model['drone']['mass_kg']
        upward_trim_acceleration = -(mass+model['cable']['total_mass_kg'])*model['simulation']['gravity_world_m_s2'][2]/mass
        if settings['acceleration_limit_mps2'][2] >= upward_trim_acceleration:
            raise ValueError('Downward acceleration limit must keep desired vertical force positive')
    recovery_mode = training_stage(settings) == 4 and settings.get('stage4_role') == 'learned_recovery_v1'
    sections = {'episode', 'workspace', 'return', 'mechanics', 'request', 'matching', 'strike_surface'}
    horizontal_fold = raw['matching']['reward_mode'] == 'horizontal_fold_outreach_v2'
    retained_fold = raw['matching']['reward_mode'] == 'retained_fold_delivery_v1'
    continuous_whip = raw['matching']['reward_mode'] in ('continuous_horizontal_whip_v1', 'continuous_horizontal_whip_soft_workspace_v2','continuous_horizontal_whip_distance_v3','continuous_horizontal_whip_full_horizon_v4', 'continuous_directional_whip_full_horizon_v5', 'continuous_directional_whip_four_second_v6')
    if horizontal_fold: sections.add('horizontal_fold')
    if retained_fold: sections.add('retained_fold_delivery')
    if continuous_whip: sections.add('continuous_whip')
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
        if not math.isfinite(t) or (t <= 0 and not (t == timing.post_hold_s == 0 and raw['matching']['reward_mode'] in ('first_hit_flexible_delivery_v2','horizontal_fold_outreach_v2','retained_fold_delivery_v1','continuous_horizontal_whip_v1','continuous_horizontal_whip_soft_workspace_v2','continuous_horizontal_whip_distance_v3','continuous_horizontal_whip_full_horizon_v4', 'continuous_directional_whip_full_horizon_v5', 'continuous_directional_whip_four_second_v6'))) or abs(t * settings['policy_rate_hz'] - round(t * settings['policy_rate_hz'])) > 1e-8:
            raise ValueError('Episode intervals must be positive whole policy periods')
    delay = model['controller']['command_delay_s']
    if abs(delay / .01 - round(delay / .01)) > 1e-8:
        raise ValueError('This learning adapter requires delay on the 100 Hz grid')
    workspace = CylinderWorkspace(**raw['workspace'])
    returns = WhipReturnConfiguration(**raw['return'])
    matching = ReleaseMatchingConfiguration(**raw['matching'])
    if 'stage1_return_transform' in settings and (
            settings['stage1_return_transform'] != 'negative_net_rational_v1'
            or matching.reward_mode != 'flexible_excitation_v1'
            or settings['workspace_penalty'] != dict(mode='squared_excess_v1', weight=.1, distance_scale_m=.1)):
        raise ValueError('Stage1 return transform requires the approved uncapped excitation profile')
    if horizontal_fold != (settings.get('horizontal_fold_contract') == 'horizontal_fold_outreach_v2'):
        raise ValueError('Horizontal-fold preset and task must agree')
    if retained_fold != (settings.get('retained_fold_delivery_contract') == 'retained_fold_delivery_v1'):
        raise ValueError('Retained fold-delivery preset and task must agree')
    if (settings.get('continuous_whip_contract') != raw['matching']['reward_mode'] if continuous_whip
            else 'continuous_whip_contract' in settings):
        raise ValueError('Continuous-whip preset and task must agree')
    if continuous_whip:
        from tasks.whip.continuous_whip import CONTRACT, SOFT_CONTRACT, SOFT_MODE, PLAIN_CONTRACT, PLAIN_MODE, FULL_CONTRACT, FULL_MODE, DIRECTED_CONTRACT, DIRECTED_MODE, SHORT_CONTRACT, SHORT_MODE
        plain=raw['matching']['reward_mode'] in (PLAIN_MODE,FULL_MODE,DIRECTED_MODE,SHORT_MODE)
        soft=plain or raw['matching']['reward_mode']==SOFT_MODE
        expected_contract=SHORT_CONTRACT if raw['matching']['reward_mode']==SHORT_MODE else DIRECTED_CONTRACT if raw['matching']['reward_mode']==DIRECTED_MODE else FULL_CONTRACT if raw['matching']['reward_mode']==FULL_MODE else PLAIN_CONTRACT if plain else SOFT_CONTRACT if soft else CONTRACT
        if (raw['continuous_whip'] != expected_contract or training_stage(settings) != 1 or algorithm(settings) != 'ppo'
                or settings['action_interface'] != 'acceleration_direct_v1' or settings['policy_rate_hz'] != 50
                or settings['acceleration_limit_mps2'] != [8,8,4] or settings['discount'] != 1
                or settings.get('reward_scale',1) != 1
                or settings['workspace_penalty'] != dict(mode='squared_excess_v1',weight=.1 if soft else 0.,distance_scale_m=.1)
                or raw['workspace'] != dict(radius_m=.55,height_lower_world_m=1.5,height_upper_world_m=2.6)
                or returns.maximum_tilt_deg != 60 or returns.minimum_cable_height_m != .05
                or raw['strike_surface']['strike_bonus_weight'] != 0
                or settings['ppo']['entropy'] != dict(mode='exponential_budget_v1',initial_weight=.02,final_weight=.001,decay_exponent=8.)
                or settings['ppo'].get('action_noise') != dict(mode='ar1_v1',correlation_time_s=.1)
                or settings['ppo'].get('transfer_entropy_clock') != ('inherit' if soft and raw['matching']['reward_mode'] not in (DIRECTED_MODE,SHORT_MODE) else 'stage_local')
                or any(k in settings for k in ('movement_penalty','stage1_return_transform','stage2_return_transform',
                    'fold_curriculum','target_curriculum','target_observation','horizontal_fold_contract',
                    'retained_fold_delivery_contract','plateau_detection'))
                or model['simulation']['segments_per_marker_interval'] != [1]*10):
            raise ValueError('Continuous Stage1 requires its fixed horizon, control, safety and exploration contract')
    if retained_fold:
        from tasks.whip.retained_fold_delivery import CONTRACT
        if (raw['retained_fold_delivery'] != CONTRACT or training_stage(settings) != 2 or algorithm(settings) != 'ppo'
                or settings['action_interface'] != 'acceleration_direct_v1' or settings['policy_rate_hz'] != 50
                or settings['acceleration_limit_mps2'] != [8,8,4] or settings['discount'] != 1
                or settings.get('reward_scale',1) != 1
                or settings['workspace_penalty'] != dict(mode='squared_excess_v1',weight=0.,distance_scale_m=.1)
                or raw['workspace'] != dict(radius_m=.55,height_lower_world_m=1.5,height_upper_world_m=2.6)
                or returns.maximum_tilt_deg != 60 or returns.minimum_cable_height_m != .05
                or tuple(raw['request']) != (0,0,2,1)
                or raw['strike_surface'] != dict(distance_m=1.1,width_m=1,height_m=.6,
                    maximum_entry_angle_deg=30,approach_width_m=.35,strike_bonus_weight=0)
                or settings['ppo']['entropy'] != dict(mode='exponential_budget_v1',initial_weight=.02,final_weight=.001,decay_exponent=8.)
                or settings['ppo'].get('action_noise') != dict(mode='ar1_v1',correlation_time_s=.1)
                or settings['ppo'].get('transfer_entropy_clock') != 'stage_local'
                or any(k in settings for k in ('movement_penalty','stage1_return_transform','stage2_return_transform',
                    'fold_curriculum','target_curriculum','target_observation','horizontal_fold_contract','plateau_detection'))
                or model['simulation']['segments_per_marker_interval'] != [1]*10):
            raise ValueError('Retained fold delivery requires its approved fixed-target, control, workspace and exploration contract')
    if horizontal_fold:
        from tasks.whip.horizontal_fold import CONTRACT
        if (raw['horizontal_fold'] != CONTRACT or training_stage(settings) != 1 or algorithm(settings) != 'ppo'
                or settings['action_interface'] != 'acceleration_direct_v1' or settings['policy_rate_hz'] != 50
                or settings['acceleration_limit_mps2'] != [8,8,4] or settings['discount'] != 1
                or settings.get('reward_scale',1) != 1
                or settings['workspace_penalty'] != dict(mode='squared_excess_v1',weight=0.,distance_scale_m=.1)
                or raw['workspace'] != dict(radius_m=.55,height_lower_world_m=1.5,height_upper_world_m=2.6)
                or returns.maximum_tilt_deg != 60 or returns.minimum_cable_height_m != .05
                or raw['strike_surface']['strike_bonus_weight'] != 0
                or settings['ppo']['entropy'] != dict(mode='exponential_budget_v1',initial_weight=.02,final_weight=.001,decay_exponent=8.)
                or settings['ppo'].get('action_noise') != dict(mode='ar1_v1',correlation_time_s=.1)
                or any(k in settings for k in ('movement_penalty','stage1_return_transform','stage2_return_transform','fold_curriculum','target_curriculum','target_observation'))
                or model['simulation']['segments_per_marker_interval'] != [1]*10):
            raise ValueError('Horizontal-fold task requires its approved geometry, control, workspace and exploration contract')
    bounded_discovery = matching.reward_mode == 'directed_discovery_bounded_v1'
    bounded_excitation = (matching.reward_mode == 'flexible_excitation_v1'
        and settings['workspace_penalty'] == dict(mode='bounded_density_v1',
                                                  weight=5., distance_scale_m=math.sqrt(.5)))
    if bounded_discovery:
        if (algorithm(settings) != 'ppo' or training_stage(settings) != 1
                or settings['action_interface'] != 'acceleration_direct_v1'
                or settings['policy_rate_hz'] != 50 or settings['discount'] != 1
                or settings['workspace_penalty'] != dict(weight=.25, distance_scale_m=.1,
                                                         mode='bounded_density_v1')
                or returns.failure_return != -1 or returns.acceleration_cost_weight != 0
                or raw['strike_surface']['strike_bonus_weight'] != 0):
            raise ValueError('Bounded discovery requires approved Stage1 acceleration/reward settings')
    elif (settings['workspace_penalty'].get('mode', 'squared_excess_v1') != 'squared_excess_v1'
          and not bounded_excitation):
        raise ValueError('Bounded workspace cost requires an approved Stage1 discovery or excitation profile')
    first_hit_delivery = matching.reward_mode == 'first_hit_flexible_delivery_v2'
    flexible_delivery = matching.reward_mode in ('first_hit_flexible_delivery_v2','flexible_delivery_v1')
    if first_hit_delivery != ('stage2_first_hit' in settings):
        raise ValueError('First-hit settings require the first-hit reward')
    if first_hit_delivery and settings['stage2_first_hit'] != dict(time_cost_per_second=.1,timeout_penalty=2.,minimum_quality=1.,workspace_reference_samples=450,history_samples=451,time_reference_samples=450):
        raise ValueError('First-hit settings differ from approved coefficients/history contract')
    if ('stage2_return_transform' in settings) != flexible_delivery:
        raise ValueError('Stage2 transform is reserved for flexible delivery')
    if flexible_delivery and (settings.get('stage2_return_transform') != 'negative_net_rational_v1'
            or 'stage1_return_transform' in settings or training_stage(settings) != 2
            or tuple(raw['request']) != (0,0,2,3)
            or raw['strike_surface'] != dict(distance_m=1.1,width_m=1,height_m=.6,
                maximum_entry_angle_deg=30,approach_width_m=.35,strike_bonus_weight=0)
            or settings.get('target_observation') != 'initial_relative_position_v1'
            or settings['ppo'].get('transfer_entropy_clock') != 'inherit'
            or settings['ppo']['entropy'] != dict(mode='exponential_episodes_v1',initial_weight=.02,final_weight=.001,decay_episodes=2560)):
        raise ValueError('Flexible delivery requires the approved fixed-target and inherited exploration contract')
    shape_delivery = matching.reward_mode == 'directed_shape_delivery_v1'
    soft_full_horizon_guide = matching.reward_mode in ('directed_unified_fold_v1', 'flexible_excitation_v1', 'first_hit_flexible_delivery_v2', 'flexible_delivery_v1')
    if soft_full_horizon_guide and ((settings['workspace_penalty'] != dict(mode='squared_excess_v1',weight=.1,distance_scale_m=.1)
                                    and not bounded_excitation)
            or 'movement_penalty' in settings or 'fold_curriculum' in settings
            or settings['parallel_environments'] != 128 or settings['training_episodes'] != (7680 if first_hit_delivery else 15360)
            or raw['workspace'] != dict(radius_m=.55,height_lower_world_m=1.5,height_upper_world_m=2.6)):
        raise ValueError('The full-horizon guide requires the approved120-collection soft-workspace contract')
    shape_release = matching.reward_mode in ('directed_shape_release_v1','directed_unified_fold_v1', 'flexible_excitation_v1', 'directed_fold_formation_v1','directed_fold_release_aligned_v1')
    fold_phase={'directed_fold_formation_v1':'formation_v1','directed_fold_release_aligned_v1':'release_v1'}.get(matching.reward_mode)
    if fold_phase != settings.get('fold_curriculum',{}).get('phase'):
        raise ValueError('Fold reward and explicit internal curriculum phase must agree')
    soft_fold_formation = (fold_phase == 'formation_v1'
        and 'movement_penalty' not in settings
        and settings['workspace_penalty'] == dict(mode='squared_excess_v1', weight=.01, distance_scale_m=.1)
        and raw['workspace'] == dict(radius_m=.55, height_lower_world_m=1.5, height_upper_world_m=2.6))
    if fold_phase and not soft_fold_formation and settings.get('movement_penalty')!=dict(mode='path_length_v1',weight_per_cable_length=.25):
        raise ValueError('Fold curriculum requires the retained path contract or approved soft-boundary formation preset')
    if 'movement_penalty' in settings and not (shape_release or shape_delivery):
        raise ValueError('Path-length cost requires shape release or same-event shape delivery')
    if shape_release or shape_delivery or flexible_delivery:
        if (algorithm(settings) != 'ppo' or training_stage(settings) != (2 if shape_delivery or flexible_delivery else 1)
                or settings['action_interface'] != 'acceleration_direct_v1'
                or settings['policy_rate_hz'] != 50 or settings['discount'] != 1
                or settings.get('reward_scale', 1) != 1
                or (settings['workspace_penalty']['weight'] != 0 and not (soft_fold_formation or soft_full_horizon_guide))
                or returns.failure_return != -10 or returns.acceleration_cost_weight != 0
                or timing.duration_s != (5 if first_hit_delivery else 4) or timing.post_hold_s != (0 if first_hit_delivery else .5)
                or settings['acceleration_limit_mps2'] != [8., 8., 4.]
                or 'plateau_detection' in settings
                or settings['ppo'].get('action_noise') != dict(mode='ar1_v1', correlation_time_s=.1)):
            raise ValueError('Shape release/delivery requires the approved reward/exploration contract')
    if shape_delivery and (tuple(raw['request']) != (0,0,2,3)
            or raw['strike_surface']['distance_m'] != 1.1
            or raw['strike_surface']['width_m'] != 1
            or raw['strike_surface']['height_m'] != .6
            or matching.delivery_sigma_m != .7
            or settings.get('movement_penalty') != dict(mode='path_length_v1', weight_per_cable_length=.25)):
        raise ValueError('Shape delivery v1 requires the recorded fixed broad target and movement contract')
    if not (shape_release or shape_delivery or flexible_delivery or horizontal_fold or retained_fold or continuous_whip) and 'action_noise' in settings.get('ppo', {}):
        raise ValueError('Correlated exploration requires shape release or same-event shape delivery')
    # Current first-hit rewards accept crossings throughout the full episode.
    # Retained snapshots without this explicit selection keep old search bounds.
    event_end = (timing.duration_s+timing.post_hold_s-raw['mechanics']['tip_lag_s']
                 if (horizontal_fold or retained_fold or continuous_whip or matching.strike_mechanics_scale is not None
                     or matching.reward_mode in ('directed_hit_v1', 'geometric_hit_quality_v1', 'directed_whip_v1', 'directed_discovery_bounded_v1', 'directed_shape_release_v1', 'directed_unified_fold_v1', 'flexible_excitation_v1', 'directed_fold_formation_v1', 'directed_fold_release_aligned_v1', 'first_hit_flexible_delivery_v2', 'flexible_delivery_v1', 'directed_shape_delivery_v1', 'directed_compact_whip_v1', 'directed_level_outreach_whip_v1', 'directed_outreach_speed_whip_v1', 'directed_outreach_height_whip_v1', 'crossing_priority_delivery_v1', 'directed_outreach_speed_delivery_v1', 'directed_outreach_speed_precision_v1', 'directed_delivery_v1', 'directed_precision_v1')) else delay+timing.duration_s)
    if recovery is not None:
        event_end = recovery.whip_duration_s-raw['mechanics']['tip_lag_s']
    mechanics = BendReleaseConfiguration(**raw['mechanics'], event_start_s=delay,
                                        event_end_s=event_end)
    cfg = Stage1Configuration(settings['scene'], settings['model'], timing, workspace,
        returns, tuple(raw['request']), mechanics, matching,
        StrikeSurfaceConfiguration(**raw['strike_surface']), raw['episode'].get('termination', 'full_horizon_v1'), recovery,
        raw.get('horizontal_fold'), raw.get('retained_fold_delivery'), raw.get('continuous_whip'))
    if matching.reward_mode in ('directed_whip_v1', 'directed_discovery_bounded_v1', 'directed_shape_release_v1', 'directed_unified_fold_v1', 'flexible_excitation_v1', 'directed_fold_formation_v1', 'directed_fold_release_aligned_v1', 'directed_compact_whip_v1', 'directed_level_outreach_whip_v1', 'directed_outreach_speed_whip_v1', 'directed_outreach_height_whip_v1') and (training_stage(settings) != 1
            or 'target_curriculum' in settings or 'curriculum' in settings):
        raise ValueError('Directed-whip discovery is Stage1 only, without automatic target curriculum')
    if matching.reward_mode in ('retained_fold_delivery_v1', 'first_hit_flexible_delivery_v2', 'flexible_delivery_v1', 'directed_shape_delivery_v1', 'directed_delivery_v1', 'directed_outreach_speed_delivery_v1', 'crossing_priority_delivery_v1') and (training_stage(settings) != 2
            or 'target_curriculum' in settings or 'curriculum' in settings):
        raise ValueError('Directed delivery is fixed-target Stage2 only, without automatic curriculum')
    precision = training_stage(settings) == 3 and settings.get('stage3_role') == 'precision_v1'
    if (matching.reward_mode in ('directed_precision_v1', 'directed_outreach_speed_precision_v1')) != (precision or recovery_mode):
        raise ValueError('Directed precision reward requires explicit precision Stage3 or learned recovery Stage4')
    if recovery_mode and (cfg.strike_surface.width_m != .1 or cfg.strike_surface.height_m != .1
            or matching.center_sigma_m != .05):
        raise ValueError('Recovery Stage4 preserves fixed +/-5cm precision target and sigma')
    if training_stage(settings) == 2 and 'target_curriculum' not in settings and matching.reward_mode not in ('retained_fold_delivery_v1', 'first_hit_flexible_delivery_v2', 'flexible_delivery_v1', 'directed_shape_delivery_v1', 'directed_delivery_v1', 'directed_outreach_speed_delivery_v1', 'crossing_priority_delivery_v1'):
        raise ValueError('Stage2 without target curriculum requires a fixed-target directed delivery reward')
    if ('training_stage' in settings and matching.reward_mode not in ('continuous_horizontal_whip_v1', 'continuous_horizontal_whip_soft_workspace_v2', 'continuous_horizontal_whip_distance_v3','continuous_horizontal_whip_full_horizon_v4', 'continuous_directional_whip_full_horizon_v5', 'continuous_directional_whip_four_second_v6', 'retained_fold_delivery_v1', 'horizontal_fold_outreach_v2', 'directed_whip_v1', 'directed_discovery_bounded_v1', 'directed_shape_release_v1', 'directed_unified_fold_v1', 'flexible_excitation_v1', 'directed_fold_formation_v1', 'directed_fold_release_aligned_v1', 'first_hit_flexible_delivery_v2', 'flexible_delivery_v1', 'directed_shape_delivery_v1', 'directed_compact_whip_v1', 'directed_level_outreach_whip_v1', 'directed_outreach_speed_whip_v1', 'directed_outreach_height_whip_v1', 'crossing_priority_delivery_v1', 'directed_outreach_speed_delivery_v1', 'directed_outreach_speed_precision_v1', 'directed_delivery_v1', 'directed_precision_v1')
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
    delivery = ('continuous_fold_rate_v1' if settings.get('continuous_whip_contract') else
                'retained_coherent_progress_v1' if settings.get('retained_fold_delivery_contract') else
                'terminal_event_zero_potential_v2' if settings.get('horizontal_fold_contract') else
                'causal_transformed_net_differences_v1' if (settings.get('stage1_return_transform') or settings.get('stage2_return_transform'))
                else 'causal_best_event_differences_v1')
    return settings, dict(schema=run_schema(settings), reward_delivery=delivery,
        settings=deepcopy(settings), task=task,
        model=model, model_sources=sources, scene_text=scene_text), cfg, scene
