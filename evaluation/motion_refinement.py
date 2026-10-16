"""Refine scripted strokes on separate geometry/speed diagnostics, without learning.

The frontier compares the already defined r*H factor and measured forward tip
speed. It is a descriptive search heuristic, not a reward, threshold or physical
whip classifier. All scorer outputs keep the original reward-selected event.
"""
import argparse
import json
import os
from time import perf_counter

import numpy as np
import torch

from project_paths import ROOT, project_path, project_reference
from simulation.configuration import load_configuration
from tasks.whip.sac_environment import Stage1Environment
from tasks.whip.deterministic_evaluation import DeterministicEvaluationEnvironment
from training.configuration import task_configuration
from training.artifacts import execution_hashes, save_json
from viewer.replay import StateReplay
from .motion_search import candidates, program, rollout, motion_metrics, file_hash, ranked


def frontier(directory, rows, length, width, tip_lag):
    """Unweighted non-dominated r*H / tip-vx samples with actual root reversal.

    Only full-horizon valid, contained attempts enter this descriptive frontier.
    Event time lies in the scripted drive/pull plus the existing tip lag. No
    mechanics score is inferred at these alternative geometry-only samples.
    """
    points = []
    for batch in sorted({r['batch'] for r in rows}):
        eligible = [r for r in rows if r['batch']==batch and r['valid_full_horizon']
                    and not r['workspace']['boundary_breached']]
        with np.load(directory/f'batch-{batch:03d}.npz',allow_pickle=False) as d:
            t, q, v, p = (d[k] for k in ('time_s','cable_positions','cable_velocities','drone_positions'))
            for row in eligible:
                lane, b = row['lane'], row['program']
                reach = q[lane,:,-1,0]-q[lane,:,0,0]
                height = q[lane,:,-1,2]-p[lane,:,2]
                quality = np.clip(reach/length,0,1)*np.exp(-.5*(height/width)**2)
                mask = ((t>=b['drive_start_s']) & (t<=b['pull_end_s']+tip_lag)
                        & (v[lane,:,0,0]<0) & (v[lane,:,-1,0]>0) & d['state_valid'][lane])
                for i in np.flatnonzero(mask):
                    points.append(dict(id=row['id'],sample=int(i),time_s=float(t[i]),
                        reach_m=float(reach[i]),tip_drone_height_error_m=float(height[i]),
                        existing_joint_reach_height_factor=float(quality[i]),
                        tip_forward_velocity_mps=float(v[lane,i,-1,0]),
                        attachment_forward_velocity_mps=float(v[lane,i,0,0])))
    result, best_speed = [], -np.inf
    for point in sorted(points,key=lambda p:(p['existing_joint_reach_height_factor'],
                                            p['tip_forward_velocity_mps']),reverse=True):
        if point['tip_forward_velocity_mps']>best_speed:
            result.append(point)
            best_speed=point['tip_forward_velocity_mps']
    return result


def parents_from_frontier(rows, points):
    if not points:
        return ranked(rows,'level_outreach_bonus')
    ids = [points[i]['id'] for i in np.unique(np.linspace(0,len(points)-1,16).round().astype(int))]
    by_id = {row['id']:row for row in rows}
    return [by_id[i] for i in dict.fromkeys(ids)]


def retain_batch(directory, env, commands, cs, batch, rows):
    returns, valid, score, _ = rollout(env,commands)
    workspace = {k:v.cpu().numpy() for k,v in env.workspace_metrics().items()}
    components = {k:v.cpu().numpy() for k,v in score.components.items()}
    arrays = {k:v.cpu().numpy() for k,v in env.history.items()}
    arrays.update(time_s=env.time.cpu().numpy(),state_valid=env.numerical_valid.cpu().numpy(),
        cable_marker_indices=np.asarray(env.model.marker_indices),normalized_actions=commands.cpu().numpy(),
        raw_returns=returns.cpu().numpy(),valid_full_horizon=valid.cpu().numpy(),
        event_scores=score.score.cpu().numpy(),event_indices=score.event_indices.cpu().numpy(),
        transition_rewards=env.transition_rewards.cpu().numpy())
    arrays.update({'component_'+k:v for k,v in components.items()})
    arrays.update({'workspace_'+k:v for k,v in workspace.items()})
    filename = f'batch-{batch:03d}.npz'
    np.savez_compressed(directory/filename,**arrays)
    for lane,c in enumerate(cs):
        _, boundaries = program(c,env.decisions,env.settings['policy_rate_hz'],env.velocity_limit.cpu().numpy())
        row = dict(id=len(rows),batch=batch,lane=lane,artifact=filename,
            phase='pareto_local_refinement',parameters=c,program=boundaries,
            raw_return=float(returns[lane]),event_score=float(score.score[lane]),
            valid_full_horizon=bool(valid[lane]),components={k:float(v[lane]) for k,v in components.items()},
            workspace={k:v[lane].item() for k,v in workspace.items()})
        replay = env.replay_arrays(lane)
        if replay is not None:
            row['motion'] = motion_metrics(replay,row['components'],boundaries,
                env.scorer.direction.cpu().numpy(),float(env.model.rest.sum()),float(env.scorer.outreach_height_width))
        rows.append(row)
    save_json(directory/'attempts.json',rows)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source',default='runs/motion_search/20261010-forward-pullback-v1')
    parser.add_argument('--output',required=True)
    parser.add_argument('--batches',type=int,default=3)
    parser.add_argument('--batch-size',type=int,default=128)
    parser.add_argument('--seed',type=int,default=1)
    args = parser.parse_args()
    if args.batches<1 or args.batch_size<1:
        parser.error('Positive counts required')
    started = perf_counter()
    source = project_path(args.source)
    parent_configuration = json.loads((source/'configuration.json').read_text())
    snapshot = parent_configuration['snapshot']
    settings = snapshot['settings']
    scene = load_configuration(source/'scene.toml')
    cfg = task_configuration(snapshot['task'],settings,snapshot['model'],scene)
    if (cfg.matching.reward_mode!='directed_level_outreach_whip_v1'
            or cfg.request[:2]!=(0.,0.) or settings['policy_rate_hz']!=10):
        raise ValueError('Requires the nominal +X 10 Hz level-outreach experiment')
    before = execution_hashes()
    if before!=parent_configuration['execution_hashes']:
        raise ValueError('Parent execution sources changed')
    directory = project_path(args.output)
    directory.mkdir(parents=True,exist_ok=False)
    protected = dict(parent_configuration['protected_inputs'])
    protected.update({project_reference(p):file_hash(p) for p in source.rglob('*') if p.is_file()})
    if not all(file_hash(ROOT/name)==digest for name,digest in protected.items()):
        raise ValueError('A source experiment input changed')
    save_json(directory/'configuration.json',dict(protocol='stage1_pareto_velocity_refinement_v1',
        snapshot=snapshot,source=project_reference(source),execution_hashes=before,
        diagnostic_sources={project_reference(__file__):file_hash(__file__),
            'evaluation/motion_search.py':file_hash(ROOT/'evaluation/motion_search.py')},
        protected_inputs=protected,seed=args.seed,attempts=args.batch_size*args.batches,
        parent_selection='Unweighted Pareto frontier of existing r*H and forward tip speed at geometry-only samples; not a reward event or success',
        mutation='Same timing/amplitude bounds as first search; every fourth candidate additionally samples all allowed phase Z requests',
        training=False,imitation=False,model_promotion=False))
    (directory/'scene.toml').write_text(snapshot['scene_text'],encoding='utf-8')
    print(json.dumps(dict(phase='starting',directory=project_reference(directory))),flush=True)
    torch.set_num_threads(min(8,os.cpu_count() or 1))
    rng = np.random.default_rng(args.seed)
    env = Stage1Environment(dict(settings,parallel_environments=args.batch_size),snapshot,cfg,scene,
        progress=lambda value:print(json.dumps(value),flush=True))
    length, width = float(env.model.rest.sum()),float(env.scorer.outreach_height_width)
    initial_rows = json.loads((source/'attempts.json').read_text())
    initial_frontier = frontier(source,initial_rows,length,width,cfg.mechanics.tip_lag_s)
    save_json(directory/'initial-frontier.json',initial_frontier)
    parents = parents_from_frontier(initial_rows,initial_frontier)
    rows = []
    rate, limits = settings['policy_rate_hz'],np.asarray(settings['velocity_limit_mps'])
    for batch in range(args.batches):
        cs = candidates(rng,args.batch_size,env.decisions,rate,limits,parents)
        for i in range(0,len(cs),4):
            c = cs[i]
            prior = dict(c)
            c['family']='prepared_oblique'
            for name in ('prep_z_mps','drive_z_mps','pull_z_mps'):
                c[name]=float(rng.uniform(-.6,.6))
            try:
                program(c,env.decisions,rate,limits)
            except ValueError:
                cs[i]=prior
        commands = torch.from_numpy(np.stack([program(c,env.decisions,rate,limits)[0] for c in cs])).to(env.device)
        retain_batch(directory,env,commands,cs,batch,rows)
        points = frontier(directory,rows,length,width,cfg.mechanics.tip_lag_s)
        save_json(directory/'frontier.json',points)
        # Preserve exploration around the parent's frontier as well as newly
        # found examples; no scalar reward is invented to combine the axes.
        parents = parents_from_frontier(rows,points)+parents_from_frontier(initial_rows,initial_frontier)
        print(json.dumps(dict(phase='batch_completed',batch=batch,attempts=len(rows),
            valid=sum(r['valid_full_horizon'] for r in rows),
            contained=sum(r['valid_full_horizon'] and not r['workspace']['boundary_breached'] for r in rows),
            frontier_points=len(points),geometry_extreme=points[0] if points else None,
            elapsed_s=perf_counter()-started)),flush=True)
    sample_positions=np.unique(np.linspace(0,len(points)-1,4).round().astype(int)) if points else []
    selected = list(dict.fromkeys(points[i]['id'] for i in sample_positions))
    selected += [r['id'] for r in ranked(rows,'level_outreach_bonus',1) if r['id'] not in selected]
    by_id = {r['id']:r for r in rows}
    native = DeterministicEvaluationEnvironment(dict(settings,device='cpu'),snapshot,cfg,scene)
    checks = []
    for identity in selected:
        row = by_id[identity]
        p,_ = program(row['parameters'],native.decisions,rate,limits)
        returns,valid,score,_ = rollout(native,torch.from_numpy(p[None]))
        replay = native.replay_arrays()
        StateReplay(replay)
        filename = f'native-candidate-{identity:04d}.npz'
        np.savez_compressed(directory/filename,**replay)
        with np.load(directory/row['artifact']) as f:
            errors={k:float(np.max(np.abs(replay[k]-f[k][row['lane'],:len(replay['time_s'])])))
                    for k in ('drone_positions','drone_velocities','drone_rotations','cable_positions','cable_velocities')}
        checks.append(dict(candidate=identity,replay=filename,valid_full_horizon=bool(valid[0]),
            raw_return=float(returns[0]),components={k:float(v[0]) for k,v in score.components.items()},
            frontier_samples=[p for p in points if p['id']==identity],
            workspace={k:v[0].item() for k,v in native.workspace_metrics().items()},
            batched_native_max_absolute_errors=errors,numerical_failure=native.numerical_failure,
            replay_validation='passed'))
        print(json.dumps(dict(phase='native_checked',candidate=identity,errors=errors)),flush=True)
    preserved = all(file_hash(ROOT/name)==digest for name,digest in protected.items())
    unchanged = execution_hashes()==before
    save_json(directory/'assessment.json',dict(protocol='stage1_pareto_velocity_refinement_v1',
        attempts=len(rows),valid_full_horizon=sum(r['valid_full_horizon'] for r in rows),
        valid_contained=sum(r['valid_full_horizon'] and not r['workspace']['boundary_breached'] for r in rows),
        frontier_points=len(points),native_checks=checks,
        top_existing_outreach_bonus=[r['id'] for r in ranked(rows,'level_outreach_bonus')],
        protected_file_count=len(protected),protected_files_preserved=preserved,
        execution_sources_unchanged=unchanged,elapsed_s=perf_counter()-started,
        limitations='Fixed nominal hanging-cable start; adaptive scripted search, no demonstrated physical whip definition, policy learning, varied-state validation or real flight. Alternative frontier samples have no mechanics credit assigned.'))
    if not preserved or not unchanged:
        raise RuntimeError('Protected source data changed')
    print(json.dumps(dict(phase='complete',directory=project_reference(directory))),flush=True)


if __name__=='__main__':
    main()
