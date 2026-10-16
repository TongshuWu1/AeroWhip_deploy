"""Actor-only supervised initialization from actual simulated drone commands.

No imitation bonus enters PPO. Prefix masking is the ordinary policy encoder's
contract; source frames cannot expose future samples or privileged simulator state.
"""
import argparse
from copy import deepcopy
import os

import numpy as np
import torch

from project_paths import ROOT, project_path, project_reference
from training.artifacts import (execution_hashes, identity, save_json, save_checkpoint,
                                validate_actor_source, create_run)
from training.configuration import load, checkpoint_schema
from training.ppo import PPO
from tasks.whip.deterministic_evaluation import DeterministicEvaluationEnvironment
from evaluation.motion_search import file_hash
from viewer.replay import StateReplay


def load_demonstration(path, env):
    if env.settings.get('action_interface') != 'velocity_v1':
        raise ValueError('Retained demonstrations contain velocity actions, not acceleration actions')
    with np.load(path, allow_pickle=False) as source:
        frames = source['frames'].copy()
        actions = source['normalized_actions'].copy()
        indices = source['policy_indices'].copy()
        if (frames.shape != (env.samples,env.frame_size) or actions.shape != (env.decisions,env.action_size)
                or not np.array_equal(indices,np.arange(env.decisions)*env.stride)
                or not np.array_equal(source['actor_columns'],env.actor_columns)
                or not np.array_equal(source['goal_columns'],env.policy_encoder_options['goal_columns'])
                or int(source['policy_rate_hz']) != env.settings['policy_rate_hz']
                or not np.allclose(source['velocity_limit_mps'],env.settings['velocity_limit_mps'],rtol=0,atol=0)
                or not np.isfinite(frames).all() or not np.isfinite(actions).all()
                or (np.abs(actions)>1).any()):
            raise ValueError('Demonstration does not match the deployable actor/control interface')
    return (torch.from_numpy(frames).to(env.device), torch.from_numpy(actions).to(env.device),
            torch.from_numpy(indices).to(env.device))


def evaluate_actor(env, agent, directory, name):
    env.reset()
    for _ in range(env.decisions):
        _,_,done = env.step(env.policy_action(lambda h,i:agent.action(h,i,True)))
        if done:
            break
    returns, valid, score, _ = env.finish()
    replay = env.replay_arrays()
    if replay is not None:
        StateReplay(replay)
        np.savez_compressed(directory/(name+'.npz'),**replay)
    return dict(raw_return=float(returns[0]),valid_full_horizon=bool(valid[0]),
        components={k:float(v[0]) for k,v in score.components.items()},
        workspace={k:v[0].item() for k,v in env.workspace_metrics().items()},
        replay=name+'.npz' if replay is not None else None)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--search',required=True)
    parser.add_argument('--source-actor',required=True)
    parser.add_argument('--settings',default='configs/learning/whip_stage1_ppo_demo_outreach_v7.json')
    args = parser.parse_args()
    settings,snapshot,cfg,scene = load(args.settings)
    directory = project_path(args.search)/'cloning'
    directory.mkdir(exist_ok=False)
    assessment = __import__('json').loads((directory.parent/'assessment.json').read_text())
    demo_path = directory.parent/'demonstration.npz'
    if file_hash(demo_path) != assessment['demonstration_sha256']:
        raise ValueError('Demonstration artifact changed')
    snapshot['execution_hashes'] = execution_hashes()
    snapshot['demonstration_initialization'] = dict(source=project_reference(demo_path),
        sha256=file_hash(demo_path), source_actor=project_reference(args.source_actor),
        source_actor_sha256=file_hash(project_path(args.source_actor)),
        loss='Mean squared error of deterministic bounded commands on causal teacher observations',
        steps=1200,learning_rate=.0003,scale_head='weights frozen; output can change with shared features',
        critic_and_ppo_optimizers='fresh',reward_bonus=False)
    snapshot['identity'] = identity(snapshot)
    torch.set_num_threads(min(8,os.cpu_count() or 1))
    torch.manual_seed(settings['seed'])
    env = DeterministicEvaluationEnvironment(settings,snapshot,cfg,scene)
    agent = PPO(settings,env.actor_columns,env.frame_size,env.device,env.action_size,
                privileged_size=env.critic_state_size,encoder_options=env.policy_encoder_options)
    saved = torch.load(project_path(args.source_actor),map_location='cpu',weights_only=True)
    validate_actor_source(saved,snapshot)
    agent.initialize_actor(saved['agent'])
    fresh_value = deepcopy(agent.value.state_dict())
    baseline = evaluate_actor(env,agent,directory,'source-actor')
    frames,targets,indices = load_demonstration(demo_path,env)
    with torch.no_grad():
        prepared = agent.actor.encoder.prepare(frames[None].expand(len(indices),-1,-1),indices)
    agent.actor.scale.requires_grad_(False)
    optimizer = torch.optim.Adam((p for p in agent.actor.parameters() if p.requires_grad),lr=.0003)
    log = []
    best_loss,best_actor = float('inf'),None
    for step in range(1201):
        mean,_ = agent.actor.distribution(prepared,None)
        predicted = mean.tanh()
        loss = (predicted-targets).square().mean()
        if not bool(torch.isfinite(loss)):
            raise RuntimeError('Nonfinite imitation loss')
        if float(loss.detach()) < best_loss:
            best_loss = float(loss.detach())
            best_actor = deepcopy(agent.actor.state_dict())
        if step%100==0:
            row = dict(step=step,mse=float(loss.detach()),max_command_error=float((predicted-targets).abs().max().detach()))
            log.append(row)
            print(__import__('json').dumps(dict(phase='cloning',**row)),flush=True)
        if step==1200:
            break
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_((p for p in agent.actor.parameters() if p.requires_grad),.5)
        optimizer.step()
    agent.actor.load_state_dict(best_actor)
    agent.actor.scale.requires_grad_(True)
    cloned = evaluate_actor(env,agent,directory,'cloned-actor')
    value_unchanged = all(torch.equal(v,agent.value.state_dict()[k]) for k,v in fresh_value.items())
    fresh = (not agent.actor_optimizer.state and not agent.value_optimizer.state
             and agent.updates==agent.value_updates==agent.collections==agent.completed_episodes==0)
    if not value_unchanged or not fresh:
        raise RuntimeError('Imitation changed the critic or PPO clocks/optimizers')
    checkpoint = directory/'actor.pt'
    save_checkpoint(checkpoint,dict(schema=checkpoint_schema(settings),identity=snapshot['identity'],snapshot=snapshot,
        agent=agent.state_dict(),round=0,episodes=0,steps=0,evaluation_steps=0,
        best_return=float('-inf'),best_replay=None,replay=None,rng_cpu=torch.get_rng_state(),
        rng_cuda=torch.cuda.get_rng_state_all(),demonstration=project_reference(demo_path)))
    run = create_run(settings,initialize_actor=checkpoint)
    save_json(directory/'assessment.json',dict(protocol='stage1_behavior_cloning_v1',configuration=snapshot,
        loss_history=log,best_teacher_state_command_mse=best_loss,
        source_actor_evaluation=baseline,cloned_actor_evaluation=cloned,
        fresh_critic_and_ppo_optimizers=value_unchanged and fresh,
        checkpoint_sha256=file_hash(checkpoint),ppo_run=project_reference(run),
        limitations='One searched nominal demonstration; low supervised error does not establish closed-loop reproduction, robustness or real-flight validity'))
    print(__import__('json').dumps(dict(phase='cloning_complete',ppo_run=project_reference(run),
        teacher_state_mse=best_loss,cloned_actor=cloned)),flush=True)


if __name__=='__main__':
    main()
