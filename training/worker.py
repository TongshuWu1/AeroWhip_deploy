"""PPO training worker for one prepared run folder; cooperative stop and save.

    python -m training.worker --run runs/ppo/<run>

Each collection simulates parallel_environments episodes with the stochastic policy,
retains every outcome, and applies one PPO update. Deterministic evaluation runs
after the first collection, at each evaluation interval and at the end; best.pt is
the checkpoint with the highest evaluation return. The run folder must have been
created by training.artifacts.create_run with the current source.
"""
import argparse
import os
from pathlib import Path
from time import perf_counter
import traceback
import numpy as np
import torch

from envs.environment import ACTION_INTERFACE, make_environment
from experiment import from_snapshot
from fitting.artifacts import atomic_save
from project_paths import project_path
from simulation.configuration import load_configuration
from training.artifacts import (CHECKPOINT_SCHEMA, EventLog, read_json, save_json, execution_hashes,
                                save_checkpoint, checkpoint_replay, replay_from_checkpoint)
from training.ppo import PPO


def make_agent(experiment, env):
    """PPO learner for an experiment, shaped by the environment's frames and critic state."""
    return PPO(experiment.ppo, columns=env.actor_columns, frame_size=env.frame_size,
               privileged_size=env.critic_state_size, device=env.device,
               policy_rate_hz=experiment.environment.policy_rate_hz,
               entropy_episode_budget=experiment.run.training_episodes, action_size=env.action_size,
               history_samples=env.samples,
               interface=dict(action_interface=ACTION_INTERFACE,
                              acceleration_limit_mps2=list(experiment.environment.acceleration_limit_mps2)))


def save_arrays(path, arrays):
    def writer(target):
        with target.open('wb') as stream: np.savez_compressed(stream, **arrays)
    atomic_save(path, writer)


def record_collection(directory, env, returns, valid, components, number):
    """Retain every outcome of one collection, plus replays of representative lanes."""
    prefix = f'collection-{number:05d}'
    save_arrays(directory/f'{prefix}.npz', dict(
        returns=returns.cpu().numpy(), valid=valid.cpu().numpy(), lengths=env.lengths.cpu().numpy(),
        stored_reward_sum=env.transition_rewards.sum(1).cpu().numpy(),
        **{'component_'+k: v.cpu().numpy() for k, v in components.items()},
        # Every attempted trajectory, including failures, so accounting can be checked independently.
        time_s=env.time.cpu().numpy(),
        cable_positions=env.history['cable_positions'].cpu().numpy(),
        cable_velocities=env.history['cable_velocities'].cpu().numpy(),
        drone_positions=env.history['drone_positions'].cpu().numpy(),
        state_valid=env.numerical_valid.cpu().numpy(),
        terminal_time_s=env.terminal_times.cpu().numpy(),
        transition_rewards=env.transition_rewards.cpu().numpy()))
    save_json(directory/f'{prefix}-events.json',
              dict(task=env.task.name, lanes=env.episode_records(), failures=env.failure_records))
    ordered = returns.argsort()
    selected = {int(ordered[0]), int(ordered[len(ordered)//2]), int(ordered[-1])}
    for mask in (valid, ~valid):
        lanes = mask.nonzero().flatten()
        if len(lanes): selected.add(int(lanes[0]))
    retained = []
    for lane in sorted(selected):
        arrays = env.replay_arrays(lane)
        if arrays is not None:
            name = f'{prefix}-lane-{lane:03d}.npz'
            save_arrays(directory/name, arrays)
            retained.append(dict(lane=lane, replay=name, valid=bool(valid[lane]), episode_return=float(returns[lane])))
    save_json(directory/f'{prefix}-motions.json', dict(samples=retained))
    return dict(phase='collection_evidence', round=number, collection_episodes=len(returns),
                invalid_count=int((~valid).sum()), accounting=f'{prefix}.npz', motions=retained,
                policy_measurement='stochastic pre-update collection')


def run(directory):
    directory = project_path(directory)
    # Never attach another worker to a run, even when a stale lock needs review.
    lock = directory/'worker.lock'
    with lock.open('x', encoding='utf-8') as stream: stream.write(str(os.getpid()))
    started = perf_counter()
    log = EventLog(directory)
    round_number = episodes = steps = evaluation_steps = 0
    best, best_replay, best_rollout, best_event = -float('inf'), None, None, None
    agent = last_saved_round = None
    reward_scale = 1.0
    def event(**value):
        value['reward_scale'] = reward_scale
        for key in ('return_mean', 'return_max', 'evaluation_return', 'best_return'):
            if key in value:
                value['scaled_'+key] = reward_scale*value[key]
        log(dict(value, elapsed_s=perf_counter()-started))
    def stop(): return (directory/'stop.request').exists()
    try:
        snapshot = read_json(directory/'configuration.json')
        if snapshot['execution_hashes'] != execution_hashes():
            raise ValueError('Source changed after run preparation; prepare a new run')
        scene = load_configuration(directory/'scene.toml')
        experiment = from_snapshot(snapshot, scene)
        run_config = experiment.run
        reward_scale = experiment.ppo.reward_scale
        if run_config.device == 'cuda' and not torch.cuda.is_available():
            raise RuntimeError('CUDA is unavailable. Select CPU explicitly; no silent fallback')
        torch.set_num_threads(min(8, os.cpu_count() or 1))
        torch.manual_seed(run_config.seed)
        if torch.cuda.is_available(): torch.cuda.manual_seed_all(run_config.seed)
        event(phase='initializing', pid=os.getpid(), algorithm='ppo', message='Preparing PPO and physics')
        environments = {}
        def environment(width):
            if width not in environments:
                environments[width] = make_environment(experiment, snapshot, scene, lanes=width)
            return environments[width]

        env = environment(min(run_config.parallel_environments, run_config.training_episodes))
        evaluation_env = make_environment(experiment, snapshot, scene, lanes=1)
        agent = make_agent(experiment, env)
        rollout = None
        request = read_json(directory/'request.json')
        if request['resume']:
            saved = torch.load(project_path(request['resume']), map_location=env.device, weights_only=True)
            if saved.get('schema') != CHECKPOINT_SCHEMA:
                raise ValueError('Resume requires a current learner checkpoint')
            if saved['identity'] != snapshot['identity']: raise ValueError('Resume snapshot does not match checkpoint')
            agent.load_state_dict(saved['agent'])
            round_number, episodes, steps, evaluation_steps = (saved[k] for k in ('round', 'episodes', 'steps', 'evaluation_steps'))
            if agent.environment_steps != steps:
                raise ValueError('Checkpoint exploration clock differs from collected training steps')
            if agent.collections != round_number:
                raise ValueError('PPO checkpoint is not a completed collection boundary')
            if agent.completed_episodes != episodes:
                raise ValueError('Checkpoint completed-episode entropy clock differs from training episodes')
            best, best_replay = saved['best_return'], saved.get('best_replay')
            best_rollout, best_event = saved.get('best_rollout'), saved.get('best_event')
            if best_rollout is not None:
                best_replay = 'inherited-best.npz'
                save_arrays(directory/best_replay, replay_from_checkpoint(best_rollout))
                event(**dict(best_event, replay=best_replay, inherited=True))
            torch.set_rng_state(saved['rng_cpu'].cpu())
            if env.device == 'cuda': torch.cuda.set_rng_state_all([s.cpu() for s in saved['rng_cuda']])
            event(phase='resumed', round=round_number, episodes=episodes, steps=steps)

        remaining = max(0, run_config.training_episodes-episodes)
        total_rounds = round_number+(remaining+run_config.parallel_environments-1)//run_config.parallel_environments

        def checkpoint(name='latest.pt'):
            nonlocal last_saved_round
            save_checkpoint(directory/name, dict(schema=CHECKPOINT_SCHEMA, identity=snapshot['identity'],
                snapshot=snapshot, agent=agent.state_dict(), round=round_number,
                episodes=episodes, steps=steps, evaluation_steps=evaluation_steps,
                best_return=best, best_replay=best_replay, best_rollout=best_rollout, best_event=best_event,
                rng_cpu=torch.get_rng_state(),
                rng_cuda=torch.cuda.get_rng_state_all() if env.device == 'cuda' else []))
            last_saved_round = round_number

        def collect(deterministic=False):
            """Run one batch to completion; False if a stop was requested first."""
            nonlocal steps, evaluation_steps, rollout
            env.reset()
            previous_raw = previous_index = None
            if not deterministic:
                shape = (env.batch, env.decisions)
                rollout = dict(collection=agent.collections+1, stride=env.stride,
                    frames=env.frames, privileged=env.critic_states,
                    raw=torch.zeros((*shape, env.action_size), device=env.device),
                    mean=torch.zeros((*shape, env.action_size), device=env.device),
                    log_std=torch.zeros((*shape, env.action_size), device=env.device),
                    logp=torch.zeros(shape, device=env.device), values=torch.zeros(shape, device=env.device))
            last_notice = perf_counter()
            while True:
                if stop(): return False
                active = int(env.active.sum())
                if deterministic:
                    action = env.policy_action(lambda h, i: agent.action(h, i, True))
                    evaluation_steps += active
                else:
                    action, *policy = agent.collect_action(env.frames, env.indices(), env.critic_states[:, env.decision],
                        previous_raw=previous_raw, previous_index=previous_index)
                    previous_raw, previous_index = policy[0], env.indices().clone()
                    for key, value in zip(('raw', 'logp', 'mean', 'log_std', 'values'), policy):
                        rollout[key][:, env.decision].copy_(value)
                _, _, done = env.step(action)
                if not deterministic:
                    steps += active
                    agent.set_environment_steps(steps)
                if perf_counter()-last_notice >= 2:
                    event(phase='evaluating' if deterministic else 'collecting',
                          round=round_number if deterministic else round_number+1, total_rounds=total_rounds,
                          simulation_s=env.index*.01, active=int(env.active.sum()), steps=steps, episodes=episodes)
                    last_notice = perf_counter()
                if done: return True

        checkpoint()  # Stop during the first cohort still leaves a resumable policy.
        while episodes < run_config.training_episodes and round_number < run_config.collections and not stop():
            # A narrower final batch keeps the exact episode budget without padded episodes.
            env = environment(min(run_config.parallel_environments, run_config.training_episodes-episodes))
            if not collect():
                event(phase='discarded', message='Interrupted cohort omitted from learning data', decisions=env.decision)
                break
            returns, valid, components = env.finish()
            event(**record_collection(directory, env, returns, valid, components, round_number+1))
            before_episodes = episodes
            episodes += env.batch
            agent.set_completed_episodes(episodes)
            rollout.update(rewards=env.transition_rewards, lengths=env.lengths)
            batch_size = experiment.ppo.batch_size
            updates = experiment.ppo.epochs*((int(env.lengths.sum())+batch_size-1)//batch_size)
            losses = agent.optimize(rollout, progress=lambda n, m: event(
                phase='optimizing', round=round_number+1, gradient_update=n, gradient_updates=agent.updates,
                value_gradient_updates=agent.value_updates, **m))
            rollout = None  # Never carry on-policy data to a later collection.
            round_number += 1
            event(phase='round', round=round_number, total_rounds=total_rounds, episodes=episodes,
                total_episodes=run_config.training_episodes, collection_episodes=env.batch,
                scheduled_gradient_updates=updates, steps=steps, gradient_updates=agent.updates,
                return_mean=float(returns.mean()), return_max=float(returns.max()),
                valid_fraction=float(valid.float().mean()),
                reward_sum_max_error=float((env.transition_rewards.double().sum(1)-returns.double()).abs().max()),
                nonzero_reward_transitions=int(torch.count_nonzero(env.transition_rewards)),
                components={k: float(torch.nanmean(v.double())) for k, v in components.items()},
                workspace={k: float(v.double().mean()) for k, v in env.workspace_metrics().items()},
                **losses)
            event(phase='exploration', round=round_number, steps=steps, completed_episodes=agent.completed_episodes,
                  entropy_clock=agent.entropy_clock, entropy_episode_budget=agent.entropy_episode_budget,
                  entropy_weight=agent.entropy_weight, entropy=losses.get('entropy'),
                  entropy_measurement='last PPO update minibatch; bounded-action MC estimate')
            if not stop() and run_config.evaluation_due(round_number, before_episodes, episodes):
                env = evaluation_env
                if collect(deterministic=True):
                    evaluated, valid, components = env.finish()
                    # One physical case repeated in the actor batch layout, not independent trials.
                    result = float(evaluated[0])
                    arrays = env.replay_arrays(0)
                    filename = f'round-{round_number:05d}.npz' if arrays is not None else None
                    if arrays is not None: save_arrays(directory/filename, arrays)
                    improved = result > best
                    if improved: best, best_replay = result, filename
                    evaluation_event = dict(phase='evaluation', round=round_number, evaluation_return=result,
                        best_return=best, valid=bool(valid[0]), replay=filename,
                        components={k: float(v[0]) for k, v in components.items()},
                        evaluation_steps=evaluation_steps, **env.provenance(),
                        workspace={k: float(v[0]) for k, v in env.workspace_metrics().items()})
                    if improved:
                        best_event, best_rollout = evaluation_event, checkpoint_replay(arrays)
                    event(**evaluation_event)
                    if improved: checkpoint('best.pt')
            checkpoint()
            if (directory/'save.request').exists():
                (directory/'save.request').unlink()
                event(phase='saved', round=round_number, checkpoint='latest.pt')
        checkpoint()
        status = 'stopped' if stop() else 'completed'
        summary = dict(phase=status, round=round_number, episodes=episodes, steps=steps,
            evaluation_steps=evaluation_steps, gradient_updates=agent.updates,
            value_gradient_updates=agent.value_updates, best_return=best, reward_scale=reward_scale,
            scaled_best_return=reward_scale*best, best_replay=best_replay, latest_checkpoint='latest.pt',
            elapsed_s=perf_counter()-started)
        save_json(directory/'summary.json', summary); log(summary)
    except Exception as exc:
        failure = dict(phase='failed', error=str(exc), round=round_number, last_saved_round=last_saved_round,
                       elapsed_s=perf_counter()-started)
        save_json(directory/'summary.json', failure); log(failure)
        traceback.print_exc()
        raise
    finally:
        lock.unlink()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--run', required=True, type=Path)
    run(parser.parse_args().run)
