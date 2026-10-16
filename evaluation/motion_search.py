"""Diagnostic Stage1 velocity-stroke search; no learner, reward or plant changes.

Run as ``python -m evaluation.motion_search``. Candidate programs use the
existing 10 Hz velocity interface. All attempts keep full-horizon validity and
workspace accounting. Rankings expose the existing return and outreach bonus
separately; neither is a new success definition or model promotion criterion.
"""
import argparse
from datetime import datetime
import hashlib
import json
import os
from pathlib import Path
from time import perf_counter

import numpy as np
import torch

from project_paths import ROOT, project_path, project_reference
from tasks.whip.sac_environment import Stage1Environment
from tasks.whip.deterministic_evaluation import DeterministicEvaluationEnvironment
from training.artifacts import execution_hashes, save_json
from training.configuration import load
from viewer.replay import StateReplay


TICK_BOUNDS = dict(prep_ticks=(2, 6), pause_ticks=(0, 6),
                   drive_ticks=(3, 9), pull_ticks=(2, 7))
SPEED_BOUNDS = dict(prep_speed_mps=(.4, 2.), drive_speed_mps=(.8, 2.),
                    pull_speed_mps=(.8, 2.), prep_z_mps=(-.6, .6),
                    drive_z_mps=(-.6, .6), pull_z_mps=(-.6, .6))


def file_hash(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def program(candidate, decisions, rate, limits):
    """One backward preparation / forward drive / pullback / return program.

    Return requests cancel the requested velocity integral, not measured drone
    displacement. The existing motor/controller dynamics determine actual motion.
    The last 0.5 s is zero velocity; this adds no position servo or state reset.
    """
    result = np.zeros((decisions, 3), dtype=np.float64)
    cursor = 2
    boundaries = dict(wait_end_s=cursor/rate)
    for phase, sign in (('prep', -1), ('drive', 1), ('pull', -1)):
        if phase == 'drive':
            cursor += candidate['pause_ticks']
        ticks = candidate[phase+'_ticks']
        boundaries[phase+'_start_s'] = cursor/rate
        result[cursor:cursor+ticks, 0] = sign*candidate[phase+'_speed_mps']
        result[cursor:cursor+ticks, 2] = candidate[phase+'_z_mps']
        cursor += ticks
        boundaries[phase+'_end_s'] = cursor/rate
    recovery_ticks = rate
    if cursor+recovery_ticks > decisions-round(.5*rate):
        raise ValueError('Program must leave the configured final hold')
    recovery_velocity = -result.sum(0)/recovery_ticks
    result[cursor:cursor+recovery_ticks] = recovery_velocity
    boundaries.update(return_start_s=cursor/rate,
                      return_end_s=(cursor+recovery_ticks)/rate,
                      return_velocity_mps=recovery_velocity.tolist())
    if not np.isfinite(result).all() or (np.abs(result)>limits+1e-12).any():
        raise ValueError('Program exceeds the unchanged velocity limits')
    if not np.allclose(result.sum(0), 0, atol=1e-12, rtol=0):
        raise ValueError('Requested velocity integral did not balance')
    return result/limits, boundaries


def candidates(rng, count, decisions, rate, limits, parents=None):
    selected, seen = [], set()
    while len(selected) < count:
        if parents:
            c = dict(parents[len(selected) % len(parents)]['parameters'])
            for name, (low, high) in TICK_BOUNDS.items():
                c[name] = int(np.clip(c[name]+rng.integers(-2, 3), low, high))
            for name, (low, high) in SPEED_BOUNDS.items():
                width = .15 if '_z_' in name else .3
                c[name] = float(np.clip(c[name]+rng.uniform(-width, width), low, high))
        else:
            family = ('direct_horizontal', 'prepared_horizontal', 'prepared_oblique')[len(selected) % 3]
            c = dict(family=family)
            c.update({name:int(rng.integers(low, high+1)) for name,(low,high) in TICK_BOUNDS.items()})
            c.update({name:float(rng.uniform(low, high)) for name,(low,high) in SPEED_BOUNDS.items()})
        if c['family'] == 'direct_horizontal':
            c['prep_ticks'], c['pause_ticks'] = 0, 0
        if c['family'] != 'prepared_oblique':
            for name in ('prep_z_mps', 'drive_z_mps', 'pull_z_mps'):
                c[name] = 0.
        identity = json.dumps(c, sort_keys=True)
        if identity in seen:
            continue
        try:
            program(c, decisions, rate, limits)
        except ValueError:
            continue
        seen.add(identity)
        selected.append(c)
    return selected


def motion_metrics(arrays, components, boundaries, direction, length, height_width):
    t, p, v = (arrays[k] for k in ('time_s', 'drone_positions', 'drone_velocities'))
    q, cv = (arrays[k] for k in ('cable_positions', 'cable_velocities'))
    forward = cv @ direction
    reach = (q[:, -1]-q[:, 0]) @ direction
    dz = q[:, -1, 2]-p[:, 2]
    quality = np.clip(reach/length, 0, 1)*np.exp(-.5*(dz/height_width)**2)
    mask = ((t >= boundaries['drive_start_s']) &
            (t <= boundaries['pull_end_s']+.15) &
            (forward[:, 0]<0) & (forward[:, -1]>0))
    result = dict(max_drone_radius_m=float(np.linalg.norm(p[:, :2]-p[0, :2],axis=1).max()),
                  drone_height_range_m=[float(p[:, 2].min()),float(p[:, 2].max())])
    release = components.get('release_time_s')
    if release is not None and np.isfinite(release):
        i = int(round(release/.01))
        if i < len(t):
            result['selected_event'] = dict(time_s=float(t[i]), reach_m=float(reach[i]),
                tip_drone_height_error_m=float(dz[i]),
                attachment_forward_velocity_mps=float(forward[i, 0]),
                tip_forward_velocity_mps=float(forward[i, -1]),
                drone_forward_velocity_mps=float(v[i] @ direction),
                attachment_reversed_while_tip_forward=bool(forward[i,0]<0<forward[i,-1]),
                during_primary_stroke_or_existing_tip_lag=bool(
                    boundaries['drive_start_s'] <= t[i] <= boundaries['pull_end_s']+.15))
    if mask.any():
        i = int(np.argmax(np.where(mask, quality, -np.inf)))
        result['primary_stroke_geometry_diagnostic'] = dict(time_s=float(t[i]),
            reach_m=float(reach[i]), tip_drone_height_error_m=float(dz[i]),
            existing_joint_reach_height_factor=float(quality[i]),
            attachment_forward_velocity_mps=float(forward[i,0]),
            tip_forward_velocity_mps=float(forward[i,-1]),
            definition='Geometry-only maximum during drive/pullback plus existing 0.15 s tip lag; not a scored whip event or success')
    return result


def ranked(rows, key, count=8):
    eligible = [r for r in rows if r['valid_full_horizon'] and not r['workspace']['boundary_breached']
                and r['components'].get(key) is not None and np.isfinite(r['components'][key])]
    return sorted(eligible,key=lambda r:r['components'][key],reverse=True)[:count]


def rollout(env, commands):
    env.reset()
    for k in range(env.decisions):
        _, _, done = env.step(commands[:, k])
        if done:
            break
    return env.finish()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--settings',default='configs/learning/whip_stage1_ppo_level_outreach_v6.json')
    parser.add_argument('--output')
    parser.add_argument('--batch-size',type=int,default=128)
    parser.add_argument('--coarse-batches',type=int,default=2)
    parser.add_argument('--refine-batches',type=int,default=1)
    parser.add_argument('--seed',type=int,default=0)
    args = parser.parse_args()
    if args.batch_size<1 or args.coarse_batches<1 or args.refine_batches<0:
        parser.error('Positive batch/coarse counts and nonnegative refinement required')
    started = perf_counter()
    settings, snapshot, cfg, scene = load(args.settings)
    if settings['policy_rate_hz'] != 10 or cfg.matching.reward_mode != 'directed_level_outreach_whip_v1':
        raise ValueError('This diagnostic requires the current 10 Hz level-outreach preset')
    directory = project_path(args.output or ROOT/'runs/motion_search'/datetime.now().strftime('%Y%m%d-%H%M%S-%f'))
    directory.mkdir(parents=True,exist_ok=False)
    print(json.dumps(dict(phase='starting',directory=project_reference(directory))),flush=True)
    protected = {}
    for root in ('models/milestone-20261009','runs/ppo_stage1/20261009-235853-518123',
                 'runs/ppo_stage1/20261010-134743-636058'):
        protected.update({project_reference(p):file_hash(p) for p in (ROOT/root).rglob('*') if p.is_file()})
    before = execution_hashes()
    save_json(directory/'configuration.json',dict(protocol='stage1_scripted_velocity_search_v1',
        snapshot=snapshot,execution_hashes=before,diagnostic_source_sha256=file_hash(__file__),
        seed=args.seed,attempts=args.batch_size*(args.coarse_batches+args.refine_batches),
        tick_bounds=TICK_BOUNDS,speed_bounds=SPEED_BOUNDS,initial_wait_ticks=2,return_ticks=10,
        selection='Existing reward-selected-event level_outreach_bonus and score, among valid fully contained attempts; no new reward or success criterion',
        training=False,imitation=False,model_promotion=False,protected_inputs=protected))
    (directory/'scene.toml').write_text(snapshot['scene_text'],encoding='utf-8')
    torch.set_num_threads(min(8,os.cpu_count() or 1))
    rng = np.random.default_rng(args.seed)
    env = Stage1Environment(dict(settings,parallel_environments=args.batch_size),snapshot,cfg,scene,
        progress=lambda x:print(json.dumps(x),flush=True))
    rate, limits = settings['policy_rate_hz'],np.asarray(settings['velocity_limit_mps'])
    length, width = float(env.model.rest.sum()),float(env.scorer.outreach_height_width)
    direction = env.scorer.direction.cpu().numpy()
    all_rows = []
    for batch in range(args.coarse_batches+args.refine_batches):
        parents = None
        if batch >= args.coarse_batches:
            pool = ranked(all_rows,'level_outreach_bonus',4)+ranked(all_rows,'mechanics_credit',4)
            parents = list({r['id']:r for r in pool}.values()) or None
        cs = candidates(rng,args.batch_size,env.decisions,rate,limits,parents)
        programs = [program(c,env.decisions,rate,limits) for c in cs]
        commands = torch.as_tensor(np.stack([p[0] for p in programs]),device=env.device)
        returns,valid,score,cost = rollout(env,commands)
        workspace = env.workspace_metrics()
        components = {k:v.cpu().numpy() for k,v in score.components.items()}
        workspace = {k:v.cpu().numpy() for k,v in workspace.items()}
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
            component = {k:float(v[lane]) for k,v in components.items()}
            row = dict(id=len(all_rows),batch=batch,lane=lane,artifact=filename,
                phase='coarse' if parents is None else 'local_refinement',parameters=c,
                program=programs[lane][1],raw_return=float(returns[lane]),
                event_score=float(score.score[lane]),valid_full_horizon=bool(valid[lane]),
                components=component,workspace={k:v[lane].item() for k,v in workspace.items()})
            replay = env.replay_arrays(lane)
            if replay is not None:
                row['motion'] = motion_metrics(replay,component,row['program'],direction,length,width)
            all_rows.append(row)
        save_json(directory/'attempts.json',all_rows)
        best = ranked(all_rows,'level_outreach_bonus',1)
        print(json.dumps(dict(phase='batch_completed',batch=batch,attempts=len(all_rows),
            valid=int(valid.sum()),contained=int((valid.cpu().numpy() & ~workspace['boundary_breached']).sum()),
            best_outreach_candidate=best[0]['id'] if best else None,elapsed_s=perf_counter()-started)),flush=True)
    by_outreach = ranked(all_rows,'level_outreach_bonus')
    by_return = sorted([r for r in all_rows if r['valid_full_horizon'] and not r['workspace']['boundary_breached']],
                       key=lambda r:r['raw_return'],reverse=True)[:8]
    selected = list({r['id']:r for r in by_outreach[:2]+by_return[:1]}.values())
    native = DeterministicEvaluationEnvironment(dict(settings,device='cpu'),snapshot,cfg,scene)
    checks = []
    for row in selected:
        p,_ = program(row['parameters'],native.decisions,rate,limits)
        returns,valid,score,cost = rollout(native,torch.as_tensor(p[None],device='cpu'))
        replay = native.replay_arrays()
        StateReplay(replay)
        name = f'native-candidate-{row["id"]:04d}.npz'
        np.savez_compressed(directory/name,**replay)
        native_components = {k:float(v[0]) for k,v in score.components.items()}
        with np.load(directory/row['artifact']) as original:
            errors = {k:float(np.max(np.abs(replay[k]-original[k][row['lane'],:len(replay['time_s'])])))
                for k in ('drone_positions','drone_velocities','drone_rotations','cable_positions','cable_velocities')}
        check = dict(candidate=row['id'],replay=name,valid_full_horizon=bool(valid[0]),
            raw_return=float(returns[0]),components=native_components,
            workspace={k:v[0].item() for k,v in native.workspace_metrics().items()},
            motion=motion_metrics(replay,native_components,row['program'],direction,length,width),
            batched_native_max_absolute_errors=errors,
            numerical_failure=native.numerical_failure,replay_validation='passed')
        checks.append(check)
        print(json.dumps(dict(phase='native_checked',candidate=row['id'],errors=errors)),flush=True)
    preserved = all(file_hash(ROOT/p)==h for p,h in protected.items())
    unchanged = before==execution_hashes()
    save_json(directory/'assessment.json',dict(protocol='stage1_scripted_velocity_search_v1',
        attempts=len(all_rows),valid_full_horizon=sum(r['valid_full_horizon'] for r in all_rows),
        valid_contained=sum(r['valid_full_horizon'] and not r['workspace']['boundary_breached'] for r in all_rows),
        top_existing_outreach_bonus=[r['id'] for r in by_outreach],
        top_existing_return=[r['id'] for r in by_return],native_checks=checks,
        execution_sources_unchanged=unchanged,protected_files_preserved=preserved,
        protected_file_count=len(protected),elapsed_s=perf_counter()-started,
        limitations='One fixed initial state, scripted bounded XZ motions, adaptively selected candidates; no policy learning, robustness or real-flight evidence. Positive score is not whip success.'))
    if not preserved or not unchanged:
        raise RuntimeError('Protected inputs or execution sources changed during diagnostic')
    print(json.dumps(dict(phase='complete',directory=project_reference(directory),attempts=len(all_rows))),flush=True)


if __name__ == '__main__':
    main()
