"""UI-owned Stage 1 PPO worker. No automatic training on import; cooperative stop/save."""
import argparse
import os
from pathlib import Path
from time import perf_counter, sleep
import traceback
import numpy as np
import torch

from fitting.artifacts import atomic_save
from simulation.configuration import load_configuration
from tasks.whip.environment import make_environment
from training.configuration import (task_configuration, validate, run_schema, checkpoint_schema,
    episode_budget, evaluation_due, require_launchable)
from training.artifacts import (EventLog, read_json, save_json, execution_hashes, save_checkpoint,
                                checkpoint_replay, replay_from_checkpoint)
from training.ppo import PPO


def record_collection(directory,env,returns,valid,score,round_number):
    """Retain every causal-fold outcome, with task-specific qualification labels."""
    qualification=(torch.zeros_like(returns))
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
        dict(schema=env.configuration.reward_mode,lanes=env.event_records(),substep_failures=env.substep_records))
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
                qualified=None,episode_return=float(returns[lane])))
    save_json(directory/f'collection-{round_number:05d}-motions.json',dict(samples=retained))
    return dict(phase='collection_evidence',round=round_number,collection_episodes=len(returns),
        qualified_count=None,invalid_count=int((~valid).sum()),
        accounting=f'collection-{round_number:05d}.npz',motions=retained,
        hit_definition=('continuous diagnostic only; no binary whip-success or target-hit gate'),
        policy_measurement='stochastic pre-update collection')


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
    agent = None
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
        expected_delivery = 'continuous_fold_rate_v1'
        if (snapshot.get('schema') != run_schema(snapshot['settings'])
                or snapshot.get('reward_delivery') != expected_delivery):
            raise ValueError('Prepare a current learning run; old transitions cannot be reused')
        if snapshot['execution_hashes'] != execution_hashes():
            raise ValueError('Source changed after run preparation; prepare a new run')
        settings = snapshot['settings']; validate(settings)
        require_launchable(settings)
        reward_scale = settings.get('reward_scale', 1.0)
        if settings['device'] == 'cuda' and not torch.cuda.is_available():
            raise RuntimeError('CUDA is unavailable. Select CPU explicitly; no silent fallback')
        torch.set_num_threads(min(8,os.cpu_count() or 1))
        torch.manual_seed(settings['seed'])
        if torch.cuda.is_available(): torch.cuda.manual_seed_all(settings['seed'])
        scene = load_configuration(directory/'scene.toml')
        cfg = task_configuration(snapshot['task'],settings,snapshot['model'],scene)
        event(phase='initializing',pid=os.getpid(),algorithm='ppo',message='Preparing PPO and physics')
        environments = {}
        def environment(width):
            if width not in environments:
                local_settings = dict(settings,parallel_environments=width)
                environments[width] = make_environment(local_settings,snapshot,cfg,scene,
                    progress=lambda x:event(**dict(x,parallel_environments=width)))
            return environments[width]

        env = environment(min(settings['parallel_environments'],episode_budget(settings)))
        evaluation_env = make_environment(dict(settings,parallel_environments=1),snapshot,cfg,scene)
        agent = PPO(settings,env.actor_columns,env.frame_size,env.device,env.action_size,
                    privileged_size=env.critic_state_size,
                    **({'encoder_options': env.policy_encoder_options}))
        ppo_rollout = None
        request = read_json(directory/'request.json')
        if request['resume']:
            saved = torch.load(project_path(request['resume']),map_location=env.device,weights_only=True)
            if saved.get('schema') != checkpoint_schema(settings):
                raise ValueError('Resume requires a matching learner checkpoint; historical learners are retired')
            if saved['identity'] != snapshot['identity']: raise ValueError('Resume snapshot does not match checkpoint')
            agent.load_state_dict(saved['agent'])
            round_number,episodes,steps,evaluation_steps = (saved[k] for k in ('round','episodes','steps','evaluation_steps'))
            if agent.environment_steps != steps:
                raise ValueError('Checkpoint exploration clock differs from collected training steps')
            if agent.collections != round_number:
                raise ValueError('PPO checkpoint is not a completed collection boundary')
            if agent.entropy_clock == 'completed_episodes_v1' and agent.completed_episodes != episodes:
                raise ValueError('Checkpoint completed-episode entropy clock differs from training episodes')
            best,best_replay = saved['best_return'],saved.get('best_replay')
            best_rollout,best_event = saved.get('best_rollout'),saved.get('best_event')
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
                agent=agent.state_dict(),replay=None,round=round_number,
                episodes=episodes,steps=steps,evaluation_steps=evaluation_steps,
                best_return=best,best_replay=best_replay,best_rollout=best_rollout,best_event=best_event,rng_cpu=torch.get_rng_state(),
                rng_cuda=torch.cuda.get_rng_state_all() if env.device=='cuda' else [])
            save_checkpoint(directory/name,saved)
            last_saved_round = round_number

        def collect(deterministic=False, *, evaluation=False):
            nonlocal steps,evaluation_steps,ppo_rollout
            env.reset()
            previous_raw = previous_index = None
            if not deterministic and (not evaluation):
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
                if not deterministic and (not evaluation):
                    action,*saved_policy = agent.collect_action(env.frames,env.indices(),env.critic_states[:,env.decision],
                        previous_raw=previous_raw, previous_index=previous_index)
                    previous_raw, previous_index = saved_policy[0], env.indices().clone()
                    for key, value in zip(('raw','logp','mean','log_std','values'),saved_policy):
                        ppo_rollout[key][:,env.decision].copy_(value)
                elif not deterministic and agent.noise_rho is not None:
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




        checkpoint()  # Stop during the first cohort still leaves a resumable policy.
        from training.configuration import collection_limit
        while episodes < episode_budget(settings) and round_number < collection_limit(settings) and not stop():
            # A smaller final graph avoids padded/fictitious episodes and preserves
            # the exact user budget without changing the environment's physics.
            env = environment(min(settings['parallel_environments'],episode_budget(settings)-episodes))
            if not collect():
                event(phase='discarded',message='Interrupted cohort omitted from learning data',decisions=env.decision)
                break
            returns,valid,score,cost = env.finish()
            event(**record_collection(directory,env,returns,valid,score,round_number+1))
            before_episodes = episodes
            episodes += env.batch
            losses = {}
            agent.set_completed_episodes(episodes)
            ppo_rollout.update(rewards=env.transition_rewards, lengths=env.lengths)
            updates = settings['ppo']['epochs']*((int(env.lengths.sum())+settings['batch_size']-1)//settings['batch_size'])
            losses = agent.optimize(ppo_rollout, progress=lambda n,m:event(
                phase='optimizing',round=round_number+1,gradient_update=n,gradient_updates=agent.updates,
                value_gradient_updates=agent.value_updates,**m))
            ppo_rollout = None  # Never carry this on-policy data to a later collection.
            round_number += 1
            components = {k:float(torch.nanmean(v.double())) for k,v in score.components.items()}
            event(phase='round',round=round_number,total_rounds=total_rounds,episodes=episodes,
                total_episodes=episode_budget(settings),collection_episodes=env.batch,scheduled_gradient_updates=updates,
                steps=steps,gradient_updates=agent.updates,return_mean=float(returns.mean()),
                return_max=float(returns.max()),valid_fraction=float(valid.float().mean()),
                reward_sum_max_error=float((env.transition_rewards.double().sum(1)-returns.double()).abs().max()),
                nonzero_reward_transitions=int(torch.count_nonzero(env.transition_rewards)),
                workspace_cost=float(cost.mean()),components=components,
                workspace={k:float(v.double().mean()) for k,v in env.workspace_metrics().items()},
                **(losses))
            event(phase='optimizer_clocks',round=round_number,gradient_updates=agent.updates,
                  value_gradient_updates=agent.value_updates)
            event(phase='exploration',round=round_number,steps=steps,
                  completed_episodes=agent.completed_episodes,entropy_clock=agent.entropy_clock,
                  entropy_episode_budget=agent.entropy_episode_budget,
                  entropy_weight=agent.entropy_weight,entropy=losses.get('entropy'),
                  entropy_measurement='last PPO update minibatch; bounded-action MC estimate')
            if not stop() and evaluation_due(settings, round_number, before_episodes, episodes):
                env = evaluation_env
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
            checkpoint()
            if (directory/'save.request').exists():
                (directory/'save.request').unlink()
                event(phase='saved',round=round_number,checkpoint='latest.pt')
        checkpoint()
        status = 'stopped' if stop() else 'completed'
        summary = dict(phase=status,round=round_number,episodes=episodes,steps=steps,
            evaluation_steps=evaluation_steps,gradient_updates=agent.updates,best_return=best,
            reward_scale=reward_scale,scaled_best_return=reward_scale*best,
            best_replay=best_replay,latest_checkpoint='latest.pt',elapsed_s=perf_counter()-started)
        summary['value_gradient_updates'] = agent.value_updates
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
