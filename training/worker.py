"""UI-owned PPO/SAC worker. No automatic training on import; cooperative stop/save."""
import argparse
import hashlib
from dataclasses import asdict
import os
from pathlib import Path
from time import perf_counter, sleep
import traceback
import numpy as np
import torch

from fitting.artifacts import atomic_save
from simulation.configuration import load_configuration
from tasks.whip.sac_environment import make_environment
from tasks.whip.deterministic_evaluation import DeterministicEvaluationEnvironment
from training.configuration import (task_configuration, validate, run_schema, checkpoint_schema, algorithm,
    episode_budget, evaluation_settings, collection_updates, evaluation_due, training_stage, require_launchable)
from training.artifacts import (EventLog, read_json, save_json, execution_hashes, save_checkpoint,
                                checkpoint_replay, replay_from_checkpoint)
from training.sac import SAC, EpisodeReplay, scalar_metrics
from training.ppo import PPO
from training.plateau import PlateauDetector, collection_metrics
from training.curriculum import Curriculum, TargetCurriculum, target_workspace_gate
from tasks.whip.learned_recovery import make_scorer
from tasks.whip.target_sampling import target_surface


def record_horizontal_fold_collection(directory,env,returns,valid,score,round_number):
    """Retain every causal-fold outcome, with task-specific qualification labels."""
    is_retained=env.configuration.matching.reward_mode=='retained_fold_delivery_v1'
    is_continuous=env.configuration.matching.reward_mode in ('continuous_horizontal_whip_v1', 'continuous_horizontal_whip_soft_workspace_v2','continuous_horizontal_whip_distance_v3','continuous_horizontal_whip_full_horizon_v4', 'continuous_directional_whip_full_horizon_v5', 'continuous_directional_whip_four_second_v6')
    qualification=(torch.zeros_like(returns) if is_continuous else
                   score.components['qualified_hit' if is_retained else 'qualified_outreach'])
    payload=dict(returns=returns.cpu().numpy(),valid=valid.cpu().numpy(),
        qualified=qualification.cpu().numpy(),lengths=env.lengths.cpu().numpy(),
        stored_reward_sum=env.transition_rewards.sum(1).cpu().numpy(),
        **{'component_'+k:v.cpu().numpy() for k,v in score.components.items()})
    def save_arrays(name,arrays):
        def writer(path):
            with path.open('wb') as f:np.savez_compressed(f,**arrays)
        atomic_save(directory/name,writer)
    # Preserve every attempted cable trajectory so event accounting can be
    # reconstructed independently, including failures and zero-output attempts.
    payload.update(time_s=env.time.cpu().numpy(),
        cable_positions=env.history['cable_positions'].cpu().numpy(),
        cable_velocities=env.history['cable_velocities'].cpu().numpy(),
        drone_positions=env.history['drone_positions'].cpu().numpy(),
        state_valid=env.numerical_valid.cpu().numpy(),
        terminal_time_s=env.terminal_times.cpu().numpy(),
        transition_rewards=env.transition_rewards.cpu().numpy())
    save_arrays(f'collection-{round_number:05d}.npz',payload)
    save_json(directory/f'collection-{round_number:05d}-events.json',
        dict(schema=env.configuration.matching.reward_mode,lanes=env.event_records(),substep_failures=env.substep_records))
    ordered=returns.argsort();selected={int(ordered[0]),int(ordered[len(ordered)//2]),int(ordered[-1])}
    for mask in (valid,~valid,qualification.bool(),valid&~qualification.bool()):
        lanes=mask.nonzero().flatten()
        if len(lanes):selected.add(int(lanes[0]))
    retained=[]
    for lane in sorted(selected):
        arrays=env.replay_arrays(lane)
        if arrays is not None:
            name=f'collection-{round_number:05d}-fold-lane-{lane:03d}.npz'
            save_arrays(name,arrays)
            retained.append(dict(lane=lane,replay=name,valid=bool(valid[lane]),
                qualified=None if is_continuous else bool(qualification[lane]),episode_return=float(returns[lane])))
    save_json(directory/f'collection-{round_number:05d}-motions.json',dict(samples=retained))
    return dict(phase='collection_evidence',round=round_number,collection_episodes=len(returns),
        qualified_count=None if is_continuous else int(qualification.sum()),invalid_count=int((~valid).sum()),
        accounting=f'collection-{round_number:05d}.npz',motions=retained,
        hit_definition=('continuous diagnostic only; no binary whip-success or target-hit gate'
                        if is_continuous else 'verified same-opening forward rectangle hit with outward H>=0.75 and world vx>=1m/s'
                        if env.configuration.matching.reward_mode=='retained_fold_delivery_v1'
                        else 'not a hit task; qualified horizontal fold outreach'),
        policy_measurement='stochastic pre-update collection')


def record_simple_collection(directory, env, returns, valid, score, round_number):
    """Retain all-lane accounting and up to three actual stochastic hit motions.

    Saved first/min-boundary/max-return hits are evidence, not independently
    chosen initial-state robustness trials. No changes to rollout or optimizer.
    """
    directory = Path(directory)
    if env.settings.get('horizontal_fold_contract') or env.settings.get('retained_fold_delivery_contract') or env.settings.get('continuous_whip_contract'):
        return record_horizontal_fold_collection(directory,env,returns,valid,score,round_number)
    workspace = env.workspace_metrics()
    qualified = valid & (score.components['strike_success'] == 1)
    whip_mode = env.configuration.matching.reward_mode in ('directed_whip_v1', 'directed_discovery_bounded_v1', 'directed_shape_release_v1', 'directed_unified_fold_v1', 'flexible_excitation_v1', 'directed_fold_formation_v1', 'directed_fold_release_aligned_v1', 'first_hit_flexible_delivery_v2', 'flexible_delivery_v1', 'directed_shape_delivery_v1', 'directed_compact_whip_v1', 'directed_level_outreach_whip_v1', 'directed_outreach_speed_whip_v1', 'directed_outreach_height_whip_v1', 'crossing_priority_delivery_v1', 'directed_outreach_speed_delivery_v1', 'directed_outreach_speed_precision_v1', 'directed_delivery_v1', 'directed_precision_v1')
    quality_mode = env.configuration.matching.reward_mode == 'geometric_hit_quality_v1'
    strict_qualified = (valid & (score.components['strict_strike_success'] == 1)) if quality_mode else qualified
    payload = dict(returns=returns.cpu().numpy(), valid=valid.cpu().numpy(),
        qualified=strict_qualified.cpu().numpy(), lengths=env.lengths.cpu().numpy(),
        stored_reward_sum=env.transition_rewards.sum(1).cpu().numpy(),
        target_width_m=np.array(env.configuration.strike_surface.width_m),
        target_height_m=np.array(env.configuration.strike_surface.height_m))
    payload.update({'component_'+k:v.cpu().numpy() for k,v in score.components.items()})
    payload.update({'workspace_'+k:v.cpu().numpy() for k,v in workspace.items()})
    if whip_mode:
        payload.update(geometric_hit=(valid & (score.components['geometric_hit_success'] == 1)).cpu().numpy(),
            event_score=score.score.cpu().numpy(),
            hit_definition=np.array('diagnostic_only_full_horizon_crossings'))
        if env.configuration.matching.reward_mode == 'crossing_priority_delivery_v1':
            payload['hit_definition'] = np.array('best_complete_forward_crossing_reward_full_horizon_v1')
    if quality_mode:
        payload.update(geometric_hit=qualified.cpu().numpy(),
            strict_qualified=strict_qualified.cpu().numpy(),
            hit_definition=np.array('first_geometric_forward_crossing_in_current_rectangle'))
    if env.configuration.matching.reward_mode in ('directed_shape_release_v1', 'directed_unified_fold_v1', 'flexible_excitation_v1', 'directed_fold_formation_v1', 'directed_fold_release_aligned_v1', 'first_hit_flexible_delivery_v2', 'flexible_delivery_v1', 'directed_shape_delivery_v1'):
        records = env.scorer.event_records
        event_key={'first_hit_flexible_delivery_v2':'delivery_only_episode_return','flexible_delivery_v1':'delivery_only_episode_return','flexible_excitation_v1':'excitation_only_episode_return',
                   'directed_unified_fold_v1':'event_only_episode_return',
                   'directed_fold_formation_v1':'formation_only_episode_return',
                   'directed_fold_release_aligned_v1':'fold_only_episode_return',
                   'directed_shape_delivery_v1':'event_only_episode_return'}.get(
                       env.configuration.matching.reward_mode,'shape_only_episode_return')
        if 'drone_path_length_m' in score.components:
            records = [dict(record,
                **{event_key: record['episode_return']},
                episode_return=float(returns[i]),
                drone_path_length_m=float(score.components['drone_path_length_m'][i]),
                movement_accrued_cost=float(score.components['movement_accrued_cost'][i]),
                movement_cost=float(score.components['movement_cost'][i]),
                workspace_accrued_cost=float(score.components.get('workspace_accrued_cost',score.components['workspace_cost'])[i]),
                workspace_cost=float(score.components['workspace_cost'][i]))
                for i,record in enumerate(records)]
        if env.settings.get('stage1_return_transform') or env.settings.get('stage2_return_transform'):
            records = [dict(record, return_transform=env.settings.get('stage1_return_transform',env.settings.get('stage2_return_transform')),
                **{key:float(score.components[key][i]) for key in (
                    ('raw_net_delivery_return' if env.configuration.matching.reward_mode in ('flexible_delivery_v1','first_hit_flexible_delivery_v2') else 'raw_net_excitation_return'), 'return_compression_adjustment',
                    ('failure_applied_total_cost' if env.first_hit_delivery else 'failure_applied_workspace_cost'), 'net_episode_return')})
                for i, record in enumerate(records)]
        if env.first_hit_delivery:
            records=[dict(record,terminal_outcome=('failure' if not bool(valid[i]) else
                'hit' if bool(env.terminal_hit[i]) else 'timeout'),
                **{key:float(score.components[key][i]) for key in
                   ('time_cost','executed_duration_s','timeout_penalty','terminal_quality_hit')})
                for i,record in enumerate(records)]
        save_json(directory/f'collection-{round_number:05d}-events.json',
                  dict(schema=env.configuration.matching.reward_mode, lanes=records))
    if env.configuration.matching.reward_mode == 'directed_shape_delivery_v1':
        payload['hit_definition'] = np.array('diagnostic_global_crossings_selected_release_hits_separate_v1')
    if env.configuration.matching.reward_mode in ('flexible_delivery_v1','first_hit_flexible_delivery_v2'):
        payload['hit_definition'] = np.array('confirmed_forward_crossing_quality_separate_v1')
        payload['quality_hit'] = (valid & (score.components['quality_hit_success']==1)).cpu().numpy()
    filename = f'collection-{round_number:05d}.npz'
    def write_arrays(path, arrays):
        with path.open('wb') as stream: np.savez_compressed(stream, **arrays)
    atomic_save(directory/filename, lambda path:write_arrays(path, payload))
    hit_lanes = qualified.nonzero().flatten()
    selected = set()
    if hit_lanes.numel():
        selected = {int(hit_lanes[0]), int(hit_lanes[returns[hit_lanes].argmax()]),
            int(hit_lanes[workspace['workspace_mean_square_excess_m2'][hit_lanes].argmin()])}
    if whip_mode:
        # Keep actual broad crossings represented, not only strict diagnostic
        # hits or reward-selected approaches. Source criteria remain unchanged.
        geometric_lanes = ((valid & (score.components['geometric_hit_success'] == 1)).nonzero().flatten()
            if env.configuration.matching.reward_mode in ('first_hit_flexible_delivery_v2', 'flexible_delivery_v1', 'directed_delivery_v1', 'directed_outreach_speed_delivery_v1', 'directed_outreach_speed_precision_v1', 'directed_precision_v1', 'crossing_priority_delivery_v1') else qualified.nonzero().flatten())
        if geometric_lanes.numel():
            selected |= {int(geometric_lanes[0]),int(geometric_lanes[returns[geometric_lanes].argmax()]),
                int(geometric_lanes[workspace['workspace_mean_square_excess_m2'][geometric_lanes].argmin()])}
        candidates = valid.nonzero().flatten()
        if candidates.numel():
            selected |= {int(candidates[returns[candidates].argmax()]),
                int(candidates[score.score[candidates].argmax()])}
            # Retain a contained representative among the top half of event
            # scores, not an uneventful hover selected purely by boundary cost.
            scores = score.score[candidates]
            upper = candidates[scores >= scores.median()]
            selected.add(int(upper[workspace['workspace_mean_square_excess_m2'][upper].argmin()]))
        if env.configuration.matching.reward_mode in ('directed_precision_v1', 'directed_outreach_speed_precision_v1'):
            errors=score.components['best_crossing_center_error_m']
            crossing_lanes=(valid & torch.isfinite(errors)).nonzero().flatten()
            if crossing_lanes.numel():
                selected.add(int(crossing_lanes[errors[crossing_lanes].argmin()]))
        if env.configuration.recovery is not None:
            candidates=valid.nonzero().flatten()
            if candidates.numel():
                selected.add(int(candidates[score.components['recovery_quality'][candidates].argmax()]))
    if env.configuration.matching.reward_mode in ('flexible_excitation_v1','first_hit_flexible_delivery_v2', 'flexible_delivery_v1'):
        # Keep typical/weak responses and a failure as well as reward maxima.
        candidates = valid.nonzero().flatten()
        if candidates.numel():
            ordered = candidates[score.score[candidates].argsort()]
            selected |= {int(ordered[0]), int(ordered[len(ordered)//2])}
        failed_with_prefix = ((~valid) & env.numerical_valid[:, :2].all(1)).nonzero().flatten()
        if failed_with_prefix.numel():
            selected.add(int(failed_with_prefix[0]))
    retained = []
    for lane in sorted(selected):
        arrays = env.replay_arrays(lane)
        if arrays is None: raise ValueError('Qualified stochastic hit lacks a valid saved prefix')
        kind = 'event' if whip_mode else 'hit'
        name = f'collection-{round_number:05d}-{kind}-lane-{lane:03d}.npz'
        atomic_save(directory/name, lambda path:write_arrays(path, arrays))
        retained.append(dict(lane=lane, replay=name, valid=bool(valid[lane]),
            raw_return=float(returns[lane]),
            components={k:float(v[lane]) for k,v in score.components.items()},
            workspace={k:float(v[lane]) for k,v in workspace.items()},
            target_surface=target_surface(env,lane)))
    return dict(phase='collection_evidence', round=round_number,
        accounting=filename, collection_episodes=len(returns),
        qualified_count=int(strict_qualified.sum()), invalid_count=int((~valid).sum()),
        **(dict(geometric_hit_count=int((valid & (score.components['geometric_hit_success']==1)).sum()),
                 strict_qualified_count=int(strict_qualified.sum())) if quality_mode or whip_mode else {}),
        hit_definition=('confirmed forward passage; q>=1 quality separate'
            if env.configuration.matching.reward_mode in ('flexible_delivery_v1','first_hit_flexible_delivery_v2') else
            'diagnostic only; flexible excitation is not target success'
            if env.configuration.matching.reward_mode == 'flexible_excitation_v1' else
            'diagnostic only; mechanics event reward over full horizon' if whip_mode else
            'geometric forward crossing; strict qualification recorded separately'
            if env.configuration.matching.reward_mode == 'geometric_hit_quality_v1' else 'strict qualified crossing'),
        motions=retained, policy_measurement='stochastic pre-update collection')


def run(directory):
    from project_paths import project_path
    directory = project_path(directory)
    # Never attach another worker to a run, even when a stale lock needs review.
    lock = directory/'worker.lock'
    with lock.open('x',encoding='utf-8') as stream: stream.write(str(os.getpid()))
    started = perf_counter()
    log = EventLog(directory)
    round_number = episodes = steps = evaluation_steps = 0
    best = -float('inf')
    best_replay = None
    best_rollout = best_event = None
    agent = replay = None
    plateau = None
    plateau_result = None
    plateau_stop = False
    last_saved_round = None
    status = 'failed'
    reward_scale = 1.0
    def event(**value):
        value['reward_scale'] = reward_scale
        for key in ('return_mean', 'return_max', 'evaluation_return', 'best_return', 'strict_return_mean'):
            if key in value:
                value['scaled_'+key] = reward_scale*value[key]
        log(dict(value,elapsed_s=perf_counter()-started))
    def stop(): return (directory/'stop.request').exists()
    try:
        snapshot = read_json(directory/'configuration.json')
        expected_delivery = ('continuous_fold_rate_v1' if snapshot['settings'].get('continuous_whip_contract') else
            'retained_coherent_progress_v1' if snapshot['settings'].get('retained_fold_delivery_contract') else
            'terminal_event_zero_potential_v2' if snapshot['settings'].get('horizontal_fold_contract') else
            'causal_transformed_net_differences_v1'
            if (snapshot['settings'].get('stage1_return_transform') or snapshot['settings'].get('stage2_return_transform')) == 'negative_net_rational_v1'
            else 'causal_best_event_differences_v1')
        if (snapshot.get('schema') != run_schema(snapshot['settings'])
                or snapshot.get('reward_delivery') != expected_delivery):
            raise ValueError('Prepare a current learning run; old transitions cannot be reused')
        if snapshot['execution_hashes'] != execution_hashes():
            raise ValueError('Source changed after run preparation; prepare a new run')
        settings = snapshot['settings']; validate(settings)
        require_launchable(settings)
        is_ppo = algorithm(settings) == 'ppo'
        if 'plateau_detection' in settings:plateau=PlateauDetector(settings)
        reward_scale = settings.get('reward_scale', 1.0)
        if settings['device'] == 'cuda' and not torch.cuda.is_available():
            raise RuntimeError('CUDA is unavailable. Select CPU explicitly; no silent fallback')
        torch.set_num_threads(min(8,os.cpu_count() or 1))
        torch.manual_seed(settings['seed'])
        if torch.cuda.is_available(): torch.cuda.manual_seed_all(settings['seed'])
        scene = load_configuration(directory/'scene.toml')
        cfg = task_configuration(snapshot['task'],settings,snapshot['model'],scene)
        curriculum = Curriculum() if 'curriculum' in settings else None
        target_curriculum = TargetCurriculum() if 'target_curriculum' in settings else None
        event(phase='initializing',pid=os.getpid(),algorithm=algorithm(settings),message='Preparing '+algorithm(settings).upper()+' and physics')
        environments = {}
        strict_scorers = {}
        def environment(width):
            if width not in environments:
                local_settings = dict(settings,parallel_environments=width)
                environments[width] = make_environment(local_settings,snapshot,cfg,scene,
                    progress=lambda x:event(**dict(x,parallel_environments=width)))
                strict_scorers[width] = environments[width].scorer
            return environments[width]

        env = environment(min(settings['parallel_environments'],episode_budget(settings)))
        evaluation_env = (make_environment(dict(settings,parallel_environments=1),snapshot,cfg,scene)
                          if settings.get('horizontal_fold_contract') or settings.get('retained_fold_delivery_contract') or settings.get('continuous_whip_contract') else
                          DeterministicEvaluationEnvironment(settings,snapshot,cfg,scene))
        evaluation_scorer = evaluation_env.scorer  # Immutable strict-final scorer, not the mutable env slot.
        curriculum_env = environment(evaluation_settings(settings)['parallel_environments']) if curriculum else None
        training_scorers = {}

        def set_reward(strict=False):
            selected = cfg
            if not strict:
                if target_curriculum is not None:
                    selected = target_curriculum.configuration(cfg,settings)
                elif curriculum is not None:
                    selected = curriculum.configuration(cfg,settings)
            env.configuration = selected
            if selected == cfg:
                env.scorer = evaluation_scorer if env is evaluation_env else strict_scorers[env.batch]
            else:
                key = (env.batch,target_curriculum.level if target_curriculum is not None else curriculum.stage)
                if key not in training_scorers:
                    training_scorers[key] = make_scorer(
                        env.model.rest,env.time,selected,initial_position=env.origin)
                env.scorer = training_scorers[key]
        agent = (PPO if is_ppo else SAC)(settings,env.actor_columns,env.frame_size,env.device,env.action_size,
                    privileged_size=env.critic_state_size,
                    **({'encoder_options':env.policy_encoder_options} if is_ppo else {}))
        if not is_ppo:
            replay = EpisodeReplay(settings['replay_episodes'],env.samples,env.frame_size,env.stride,env.device,env.action_size,
                                   privileged_size=env.critic_state_size)
        ppo_rollout = None
        request = read_json(directory/'request.json')
        if request.get('initialize_actor'):
            if request['resume'] or not is_ppo:
                raise ValueError('Actor initialization is distinct from exact PPO resume')
            name = request['initialize_actor']
            if name != 'initial-actor.pt':
                raise ValueError('Actor initializer must be the immutable run-local checkpoint copy')
            payload = (directory/name).read_bytes()
            provenance = snapshot.get('policy_initialization',{})
            if provenance.get('source_sha256') != hashlib.sha256(payload).hexdigest():
                raise ValueError('Actor initialization payload changed after run preparation')
            import io
            saved_actor = torch.load(io.BytesIO(payload),map_location='cpu',weights_only=True)
            from training.artifacts import validate_actor_source
            validate_actor_source(saved_actor,snapshot)
            agent.initialize_actor(saved_actor['agent'])
            event(phase='actor_initialized',source=provenance,steps=0,episodes=0,round=0,
                  entropy_step_offset=agent.entropy_step_offset,entropy_weight=agent.entropy_weight,
                  message='Actor-only stage transfer; value, optimizers, RNG and run counters are fresh')
        if request['resume']:
            saved = torch.load(project_path(request['resume']),map_location=env.device,weights_only=True)
            if saved.get('schema') != checkpoint_schema(settings):
                raise ValueError('Resume requires a matching learner checkpoint; SAC/PPO cannot be interchanged')
            if saved['identity'] != snapshot['identity']: raise ValueError('Resume snapshot does not match checkpoint')
            agent.load_state_dict(saved['agent'])
            if replay is not None: replay.load_state_dict(saved['replay'])
            round_number,episodes,steps,evaluation_steps = (saved[k] for k in ('round','episodes','steps','evaluation_steps'))
            if ('exploration' in settings or is_ppo) and agent.environment_steps != steps:
                raise ValueError('Checkpoint exploration clock differs from collected training steps')
            if is_ppo and agent.collections != round_number:
                raise ValueError('PPO checkpoint is not a completed collection boundary')
            if is_ppo and agent.entropy_clock == 'completed_episodes_v1' and agent.completed_episodes != episodes:
                raise ValueError('Checkpoint completed-episode entropy clock differs from training episodes')
            best,best_replay = saved['best_return'],saved.get('best_replay')
            best_rollout,best_event = saved.get('best_rollout'),saved.get('best_event')
            if curriculum is not None:
                curriculum = Curriculum(**saved['curriculum'])
            if target_curriculum is not None:
                target_curriculum = TargetCurriculum(**saved['target_curriculum'])
                target_curriculum.validate(settings)
            if plateau is not None:
                if 'plateau' not in saved:raise ValueError('Exact resume requires saved plateau state')
                plateau.load_state_dict(saved['plateau'])
                if plateau.last_round>round_number:raise ValueError('Plateau clock exceeds completed collection count')
            if best_rollout is not None:
                best_replay = 'inherited-best.npz'
                with (directory/best_replay).open('wb') as stream:
                    np.savez_compressed(stream,**replay_from_checkpoint(best_rollout))
                event(**dict(best_event,replay=best_replay,inherited=True))
            torch.set_rng_state(saved['rng_cpu'].cpu())
            if env.device == 'cuda': torch.cuda.set_rng_state_all([s.cpu() for s in saved['rng_cuda']])
            event(phase='resumed',round=round_number,episodes=episodes,steps=steps,
                  evaluation_migration=saved.get('evaluation_migration'),
                  learning_migration=saved.get('learning_migration'))

        remaining = max(0,episode_budget(settings)-episodes)
        total_rounds = round_number+(remaining+settings['parallel_environments']-1)//settings['parallel_environments']

        def checkpoint(name='latest.pt'):
            nonlocal last_saved_round
            saved = dict(schema=checkpoint_schema(settings),identity=snapshot['identity'],snapshot=snapshot,
                agent=agent.state_dict(),replay=replay.state_dict() if replay is not None else None,round=round_number,
                episodes=episodes,steps=steps,evaluation_steps=evaluation_steps,
                best_return=best,best_replay=best_replay,best_rollout=best_rollout,best_event=best_event,rng_cpu=torch.get_rng_state(),
                rng_cuda=torch.cuda.get_rng_state_all() if env.device=='cuda' else [])
            if curriculum is not None: saved['curriculum'] = asdict(curriculum)
            if target_curriculum is not None: saved['target_curriculum'] = asdict(target_curriculum)
            if plateau is not None:saved['plateau']=plateau.state_dict()
            save_checkpoint(directory/name,saved)
            last_saved_round = round_number

        def collect(deterministic=False, *, evaluation=False):
            nonlocal steps,evaluation_steps,ppo_rollout
            env.reset()
            previous_raw = previous_index = None
            if is_ppo and not deterministic and not evaluation:
                shape = (env.batch, env.decisions)
                ppo_rollout = dict(collection=agent.collections+1, stride=env.stride,
                    frames=env.frames, privileged=env.critic_states,
                    raw=torch.zeros((*shape,env.action_size),device=env.device),
                    mean=torch.zeros((*shape,env.action_size),device=env.device),
                    log_std=torch.zeros((*shape,env.action_size),device=env.device),
                    logp=torch.zeros(shape,device=env.device),values=torch.zeros(shape,device=env.device))
            last_notice = perf_counter()
            while True:
                if stop(): return False
                active = int(env.active.sum())
                if is_ppo and not deterministic and not evaluation:
                    action,*saved_policy = agent.collect_action(env.frames,env.indices(),env.critic_states[:,env.decision],
                        previous_raw=previous_raw, previous_index=previous_index)
                    previous_raw, previous_index = saved_policy[0], env.indices().clone()
                    for key, value in zip(('raw','logp','mean','log_std','values'),saved_policy):
                        ppo_rollout[key][:,env.decision].copy_(value)
                elif is_ppo and not deterministic and agent.noise_rho is not None:
                    action,previous_raw,*_ = agent.collect_action(env.frames,env.indices(),env.critic_states[:,env.decision],
                        previous_raw=previous_raw,previous_index=previous_index)
                    previous_index = env.indices().clone()
                else:
                    action = (env.policy_action(lambda h,i:agent.action(h,i,True)) if deterministic
                              else agent.action(env.frames,env.indices(),False))
                _,_,done = env.step(action)
                if deterministic or evaluation: evaluation_steps += active
                else:
                    steps += active
                    agent.set_environment_steps(steps)
                if perf_counter()-last_notice >= 2:
                    event(phase='evaluating' if deterministic or evaluation else 'collecting',round=round_number if deterministic or evaluation else round_number+1,
                        total_rounds=total_rounds,simulation_s=env.index*.01,
                        active=int(env.active.sum()),steps=steps,episodes=episodes)
                    last_notice = perf_counter()
                if done: return True

        def save_arrays(filename, arrays):
            def writer(path):
                with path.open('wb') as f: np.savez_compressed(f, **arrays)
            atomic_save(directory/filename, writer)

        def advance():
            curriculum.advance(settings)
            # Stored transition rewards belong to the previous objective. Do not
            # mix them, or reconstruct float64 scoring histories from float32 frames.
            replay.lengths.zero_(); replay.cursor = 0
            set_reward()
            event(phase='curriculum_advanced',round=round_number,stage=curriculum.stage+1,
                  replay_cleared=True,policy_and_optimizers_retained=True)
            checkpoint()

        def await_review():
            review = read_json(directory/'curriculum_review.json')
            if review['round'] != curriculum.review_round or review['identity'] != snapshot['identity']:
                raise ValueError('Motion-review evidence does not match the checkpoint')
            for row in review['samples']: event(**row)
            checkpoint()
            event(phase='curriculum_review',round=round_number,stage=curriculum.stage+1,
                  message='Review saved stochastic motions, then Approve motion; training is paused')
            while not stop():
                approval = directory/'curriculum_review_approved.json'
                if approval.exists():
                    value = read_json(approval)
                    if value == dict(identity=snapshot['identity'],round=curriculum.review_round):
                        curriculum.reviewed = True
                        advance()
                        return
                    raise ValueError('Motion approval does not match the pending review')
                if (directory/'save.request').exists():
                    checkpoint(); (directory/'save.request').unlink()
                sleep(.25)

        def evaluate_curriculum():
            # Independent stochastic evaluation; never added to replay/episode
            # budget, and never changes the training RNG sequence.
            devices = [torch.cuda.current_device()] if env.device == 'cuda' else []
            noise_seed = settings['seed']+1000000+round_number
            with torch.random.fork_rng(devices=devices):
                torch.manual_seed(noise_seed)
                if env.device == 'cuda': torch.cuda.manual_seed_all(noise_seed)
                if not collect(evaluation=True): return
            values,valid,score,cost = env.finish()
            success = valid & (score.components['strike_success'] == 1)
            fraction = float(success.double().mean())
            workspace_cost = float(score.components['workspace_cost'].mean())
            filename = f'curriculum-{round_number:05d}.npz'
            # Retain all lanes (including failed suffix masks), not a lucky example.
            save_arrays(filename,dict({k:v.cpu().numpy() for k,v in env.history.items()},
                time_s=env.time.cpu().numpy(),state_valid=env.numerical_valid.cpu().numpy(),
                returns=values.cpu().numpy(),valid=valid.cpu().numpy(),
                qualified_strike=success.cpu().numpy(),lengths=env.lengths.cpu().numpy()))
            ready = curriculum.observe(settings,fraction,workspace_cost,round_number)
            event(phase='curriculum_evaluation',round=round_number,stage=curriculum.stage+1,
                  strict_return_mean=float(values.mean()),valid_fraction=float(valid.double().mean()),
                  qualified_strike_fraction=fraction,mean_workspace_cost=workspace_cost,
                  consecutive_passes=curriculum.streak,episodes=episodes,evaluation_episodes=env.batch,
                  noise_seed=noise_seed,replay=filename,evaluation_steps=evaluation_steps)
            if ready and not curriculum.reviewed:
                rows = []
                for lane in range(env.batch):
                    arrays = env.replay_arrays(lane)
                    name = f'review-{round_number:05d}-lane-{lane:02d}.npz' if arrays is not None else None
                    if arrays is not None: save_arrays(name,arrays)
                    rows.append(dict(phase='curriculum_sample',round=round_number,lane=lane,
                        evaluation_return=float(values[lane]),valid=bool(valid[lane]),replay=name,
                        components={k:float(v[lane]) for k,v in score.components.items()},
                        workspace={k:float(v[lane]) for k,v in env.workspace_metrics().items()}))
                save_json(directory/'curriculum_review.json',dict(identity=snapshot['identity'],
                    round=round_number,samples=rows))
                await_review()
            elif ready:
                advance()

        checkpoint()  # Stop during the first cohort still leaves a resumable policy.
        if curriculum is not None and curriculum.review_round is not None:
            await_review()
        from training.configuration import collection_limit
        while episodes < episode_budget(settings) and round_number < collection_limit(settings) and not stop():
            # A smaller final graph avoids padded/fictitious episodes and preserves
            # the exact user budget without changing the environment's physics.
            env = environment(min(settings['parallel_environments'],episode_budget(settings)-episodes))
            set_reward()
            if not collect():
                event(phase='discarded',message='Interrupted cohort omitted from learning data',decisions=env.decision)
                break
            returns,valid,score,cost = env.finish()
            observed_plateau_metrics=(collection_metrics(training_stage(settings),returns,valid,score,env.workspace_metrics())
                if plateau is not None else None)
            if is_ppo and env.configuration.matching.reward_mode in ('continuous_horizontal_whip_v1', 'continuous_horizontal_whip_soft_workspace_v2', 'continuous_horizontal_whip_distance_v3','continuous_horizontal_whip_full_horizon_v4', 'continuous_directional_whip_full_horizon_v5', 'continuous_directional_whip_four_second_v6', 'retained_fold_delivery_v1', 'horizontal_fold_outreach_v2', 'directed_hit_v1', 'geometric_hit_quality_v1', 'directed_whip_v1', 'directed_discovery_bounded_v1', 'directed_shape_release_v1', 'directed_unified_fold_v1', 'flexible_excitation_v1', 'directed_fold_formation_v1', 'directed_fold_release_aligned_v1', 'first_hit_flexible_delivery_v2', 'flexible_delivery_v1', 'directed_shape_delivery_v1', 'directed_compact_whip_v1', 'directed_level_outreach_whip_v1', 'directed_outreach_speed_whip_v1', 'directed_outreach_height_whip_v1', 'crossing_priority_delivery_v1', 'directed_outreach_speed_delivery_v1', 'directed_outreach_speed_precision_v1', 'directed_delivery_v1', 'directed_precision_v1'):
                event(**record_simple_collection(directory,env,returns,valid,score,round_number+1))
            if replay is not None:
                replay.add(env.frames,env.actions,env.transition_rewards,env.lengths,env.critic_states)
            before_episodes = episodes
            episodes += env.batch
            losses = {}
            if is_ppo:
                agent.set_completed_episodes(episodes)
                ppo_rollout.update(rewards=env.transition_rewards, lengths=env.lengths)
                updates = settings['ppo']['epochs']*((int(env.lengths.sum())+settings['batch_size']-1)//settings['batch_size'])
                losses = agent.optimize(ppo_rollout, progress=lambda n,m:event(
                    phase='optimizing',round=round_number+1,gradient_update=n,gradient_updates=agent.updates,
                    value_gradient_updates=agent.value_updates,**m))
                ppo_rollout = None  # Never carry this on-policy data to a later collection.
            else:
                updates = collection_updates(settings,before_episodes,episodes)
            if replay is not None and replay.transitions >= settings['minimum_replay_transitions']:
                for update in range(updates):
                    if stop(): break
                    losses = agent.update(replay.sample(settings['batch_size']),report_metrics=False)
                    if (update+1)%20 == 0:
                        event(phase='optimizing',round=round_number+1,gradient_update=update+1,
                              gradient_updates=agent.updates,**scalar_metrics(losses))
            round_number += 1
            if plateau is not None:
                level=target_curriculum.level if target_curriculum is not None else None
                final=(target_curriculum is None or level==len(settings['target_curriculum']['widths_m'])-1)
                floor=settings['ppo']['entropy']['final_weight']
                ready=agent.entropy_weight<=floor+1e-12
                plateau_result=plateau.observe(round_number,observed_plateau_metrics,
                    target=level,final_target=final,entropy_ready=ready)
                event(phase='plateau_check',round=round_number,episodes=episodes,
                    metrics=observed_plateau_metrics,**plateau_result)
            if target_curriculum is not None:
                success_key = ('geometric_hit_success' if settings.get('stage3_role') == 'precision_v1' else 'strike_success')
                qualified = valid & (score.components[success_key] == 1)
                fraction = float(qualified.double().mean())  # ALL lanes; failure is not dropped.
                boundary_cost = float(score.components['workspace_cost'].double().mean())
                gate_cost = float(env.boundary_gate_cost().double().mean()) if 'training_stage' in settings else boundary_cost
                validity_fraction = float(valid.double().mean())
                target_curriculum.observe(settings,fraction,gate_cost,valid_fraction=validity_fraction)
                event(phase='target_curriculum_collection',round=round_number,
                    stage=target_curriculum.level+1,episodes=episodes,
                    target_width_m=env.configuration.strike_surface.width_m,
                    target_height_m=env.configuration.strike_surface.height_m,
                    qualified_fraction=fraction,valid_fraction=validity_fraction,mean_workspace_cost=boundary_cost,
                    qualification_definition=success_key,
                    **target_workspace_gate(settings,gate_cost),
                    consecutive_passes=target_curriculum.streak,
                    policy_measurement='stochastic collection before this round optimizer updates')
            components = {k:float(torch.nanmean(v.double())) for k,v in score.components.items()}
            event(phase='round',round=round_number,total_rounds=total_rounds,episodes=episodes,
                total_episodes=episode_budget(settings),collection_episodes=env.batch,scheduled_gradient_updates=updates,
                steps=steps,gradient_updates=agent.updates,return_mean=float(returns.mean()),
                return_max=float(returns.max()),valid_fraction=float(valid.float().mean()),
                reward_sum_max_error=float((env.transition_rewards.double().sum(1)-returns.double()).abs().max()),
                nonzero_reward_transitions=int(torch.count_nonzero(env.transition_rewards)),
                acceleration_cost=float(cost.mean()),components=components,
                workspace={k:float(v.double().mean()) for k,v in env.workspace_metrics().items()},
                **(losses if is_ppo else scalar_metrics(losses)))
            if is_ppo:
                event(phase='optimizer_clocks',round=round_number,gradient_updates=agent.updates,
                      value_gradient_updates=agent.value_updates)
            if is_ppo:
                event(phase='exploration',round=round_number,steps=steps,
                      completed_episodes=agent.completed_episodes,entropy_clock=agent.entropy_clock,
                      entropy_episode_budget=agent.entropy_episode_budget,entropy_episode_offset=agent.entropy_episode_offset,
                      entropy_weight=agent.entropy_weight,entropy=losses.get('entropy'),
                      entropy_step_offset=agent.entropy_step_offset,
                      entropy_measurement='last PPO update minibatch; bounded-action MC estimate')
            else:
                event(phase='exploration',round=round_number,steps=steps,
                      target_entropy=agent.target_entropy,temperature=float(agent.log_alpha.exp().detach()),
                      entropy=scalar_metrics(losses).get('entropy'))
            if curriculum is not None:
                event(phase='curriculum_stage',round=round_number,stage=curriculum.stage+1,
                      normal_width_m=env.configuration.strike_surface.approach_width_m,
                      angle_width_deg=env.configuration.matching.direction_width_deg)
            delivery_review = cfg.matching.reward_mode in ('flexible_delivery_v1','first_hit_flexible_delivery_v2') and round_number % 10 == 0
            if delivery_review and not stop():
                # Frozen development trials do not enter PPO or advance its
                # exploration clock. Restore training RNG after the probe.
                env = environment(128)
                set_reward(strict=True)
                devices = [torch.cuda.current_device()] if env.device == 'cuda' else []
                seed = (2026103000 if cfg.matching.reward_mode == 'first_hit_flexible_delivery_v2' else 2026102900)+round_number
                with torch.random.fork_rng(devices=devices):
                    torch.manual_seed(seed)
                    if env.device == 'cuda': torch.cuda.manual_seed_all(seed)
                    complete = collect(evaluation=True)
                if complete:
                    values, vmask, measured, _ = env.finish()
                    folder = directory/f'development-{round_number:05d}'
                    folder.mkdir(exist_ok=True)
                    evidence = record_simple_collection(folder,env,values,vmask,measured,round_number)
                    save_arrays(f'development-{round_number:05d}.npz',dict(
                        **{k:v.cpu().numpy() for k,v in env.history.items()},
                        time_s=env.time.cpu().numpy(),state_valid=env.numerical_valid.cpu().numpy(),
                        failed=env.failed.cpu().numpy(),valid=vmask.cpu().numpy(),returns=values.cpu().numpy()))
                    event(phase='delivery_development',round=round_number,seed=seed,
                        attempts=128,valid_count=int(vmask.sum()),
                        geometric_hits=int((vmask & (measured.components['geometric_hit_success']==1)).sum()),
                        quality_hits=int((vmask & (measured.components['quality_hit_success']==1)).sum()),
                        evidence=str(folder.relative_to(directory)),episodes=episodes,
                        evaluation_steps=evaluation_steps)
                    # Retain all development candidates for later hit-rate AND
                    # motion selection; deterministic best.pt is not that choice.
                    checkpoint(f'development-{round_number:05d}.pt')
            if not stop() and (delivery_review or evaluation_due(settings,round_number,before_episodes,episodes)):
                env = evaluation_env
                set_reward(strict=True)
                if collect(deterministic=True):
                    evaluated,valid,score,cost = env.finish()
                    # Identical initial states: deterministic evaluation is one unique
                    # case, repeated at the checkpoint's evaluation width.
                    result = float(evaluated[0])
                    arrays = env.replay_arrays(0)
                    filename = f'round-{round_number:05d}.npz'
                    if arrays is not None:
                        def writer(path):
                            with path.open('wb') as f: np.savez_compressed(f,**arrays)
                        atomic_save(directory/filename,writer)
                    improved = result > best
                    if improved: best,best_replay = result,filename if arrays is not None else None
                    evaluation_event = dict(phase='evaluation',round=round_number,evaluation_return=result,best_return=best,
                          valid=bool(valid[0]),replay=filename if arrays is not None else None,
                          components={k:float(v[0]) for k,v in score.components.items()},
                          evaluation_steps=evaluation_steps,**env.provenance(),
                          workspace={k:float(v[0]) for k,v in env.workspace_metrics().items()})
                    if improved:
                        best_event = evaluation_event
                        best_rollout = checkpoint_replay(arrays)
                    event(**evaluation_event)
                    if improved: checkpoint('best.pt')
                    if target_curriculum is not None:
                        # SAME physical rollout, rescored at the current training
                        # target; no extra fixed-state trials or GPU physics lanes.
                        set_reward()
                        current_returns,current_valid,current_score,_ = env.finish()
                        target_gate = dict(valid=bool(current_valid[0]),
                            qualified=bool(current_score.components[success_key][0] == 1),
                            workspace_cost=float(current_score.components['workspace_cost'][0]))
                        if 'training_stage' in settings:
                            target_gate['workspace_cost'] = float(env.boundary_gate_cost()[0])
                        logged_gate = dict(target_gate)
                        if 'training_stage' in settings:
                            logged_gate['workspace_gate_equivalent_cost'] = target_gate['workspace_cost']
                            logged_gate['workspace_cost'] = float(current_score.components['workspace_cost'][0])
                        event(phase='target_curriculum_evaluation',round=round_number,
                            stage=target_curriculum.level+1,episodes=episodes,
                            target_width_m=env.configuration.strike_surface.width_m,
                            target_height_m=env.configuration.strike_surface.height_m,
                            evaluation_return=float(current_returns[0]),**logged_gate,
                            components={k:float(v[0]) for k,v in current_score.components.items()},
                            replay=filename,physical_rollout='same as strict-final evaluation',
                            consecutive_passes=target_curriculum.streak,
                            **target_workspace_gate(settings,target_gate['workspace_cost']),**env.provenance())
                        if target_curriculum.ready(settings,**target_gate) and not stop():
                            target_curriculum.advance(settings,**target_gate)
                            if plateau is not None:plateau.reset()
                            next_target = target_curriculum.configuration(cfg,settings).strike_surface
                            event(phase='target_curriculum_advanced',round=round_number,
                                stage=target_curriculum.level+1,episodes=episodes,
                                target_width_m=next_target.width_m,target_height_m=next_target.height_m,
                                policy_and_optimizers_retained=True,
                                message='Next cohort uses narrower target; no old on-policy data retained')
                    if curriculum is not None and not stop():
                        env = curriculum_env
                        set_reward(strict=True)
                        evaluate_curriculum()
            checkpoint()
            if (directory/'save.request').exists():
                (directory/'save.request').unlink()
                event(phase='saved',round=round_number,checkpoint='latest.pt')
            if plateau_result is not None and plateau_result['flagged']:
                event(phase='plateau_warning',round=round_number,episodes=episodes,
                    message='Sustained plateau/regression heuristic, not proven convergence or success',detector=plateau_result)
                if plateau_result['stop'] and not stop() and episodes<episode_budget(settings):
                    # Only complete collection boundaries, after safe checkpoint save.
                    plateau_stop=True
                    break
        checkpoint()
        fold_review=(settings.get('fold_curriculum',{}).get('phase')=='formation_v1'
                     and round_number>=collection_limit(settings) and not stop())
        status = 'stopped' if stop() or fold_review else 'plateau_stopped' if plateau_stop else 'completed'
        summary = dict(phase=status,round=round_number,episodes=episodes,steps=steps,training_stage=training_stage(settings),
            evaluation_steps=evaluation_steps,gradient_updates=agent.updates,best_return=best,
            reward_scale=reward_scale,scaled_best_return=reward_scale*best,
            best_replay=best_replay,latest_checkpoint='latest.pt',elapsed_s=perf_counter()-started)
        if is_ppo: summary['value_gradient_updates'] = agent.value_updates
        if fold_review:
            summary.update(termination_reason='fold_formation_motion_review_required',
                full_stage1_complete=False,release_phase_started=False,
                message='Formation warm-up ended at30 collections; independent evaluation and motion review required. No automatic reward or stage change.')
        if plateau is not None:
            summary['plateau_detection']=plateau_result
            summary['termination_reason']='user_stop' if stop() else 'plateau_heuristic' if plateau_stop else 'hard_episode_budget'
        if curriculum is not None: summary['curriculum'] = asdict(curriculum)
        if target_curriculum is not None:
            summary['target_curriculum'] = asdict(target_curriculum)
            summary['final_target_level_reached'] = target_curriculum.level == len(settings['target_curriculum']['widths_m'])-1
        save_json(directory/'summary.json',summary); log(summary)
    except Exception as exc:
        failure = dict(phase='failed',error=str(exc),round=round_number,last_saved_round=last_saved_round,
                       elapsed_s=perf_counter()-started)
        save_json(directory/'summary.json',failure); log(failure)
        traceback.print_exc()
        raise
    finally:
        lock.unlink()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run',required=True,type=Path)
    run(parser.parse_args().run)
