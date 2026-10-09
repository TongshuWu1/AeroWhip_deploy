"""One deterministic checkpoint rollout. No SAC optimizers or learning replay."""
import argparse
import os
from pathlib import Path
import platform
from time import perf_counter
import traceback

import numpy as np
import torch

from evaluation.records import (load_checkpoint, verify_contract, compare_reference,
                                envelope_observation, require_evaluation_idle, task_stage)
from fitting.artifacts import atomic_save
from simulation.configuration import load_configuration
from tasks.whip.deterministic_evaluation import DeterministicEvaluationEnvironment
from training.artifacts import EventLog, clean, read_json, save_json
from training.configuration import task_configuration, validate, evaluation_settings
from evaluation.policy_test import frozen_actor
from training.curriculum import TargetCurriculum
from tasks.whip.learned_recovery import make_scorer
from tasks.whip.stage1 import Stage1Scorer


@torch.no_grad()
def run(directory):
    from project_paths import project_path
    directory = project_path(directory)
    lock = directory/'worker.lock'
    with lock.open('x', encoding='utf-8') as f:
        f.write(str(os.getpid()))
    started, log = perf_counter(), EventLog(directory)
    def emit(**value):
        log(dict(value, elapsed_s=perf_counter()-started))
    def stopped():
        return (directory/'stop.request').exists()
    try:
        request = read_json(directory/'request.json')
        info = load_checkpoint(request['checkpoint'])
        if info['sha256'] != request['checkpoint_sha256']:
            raise ValueError('Checkpoint changed after selection; refresh and evaluate again')
        snapshot = read_json(directory/'configuration.json')
        if snapshot != info['snapshot'] or (directory/'scene.toml').read_text(encoding='utf-8') != snapshot['scene_text']:
            raise ValueError('Evaluation snapshot differs from the checkpoint')
        contract = verify_contract(snapshot, request)
        require_evaluation_idle(owned_directory=directory)
        settings = snapshot['settings']
        validate(settings)
        if settings['device'] == 'cuda' and not torch.cuda.is_available():
            raise RuntimeError('This checkpoint uses CUDA, which is unavailable. Saved replay needs no CUDA.')
        save_json(directory/'provenance.json', dict(checkpoint_sha256=info['sha256'],
            source_identity=snapshot['identity'], torch_version=str(torch.__version__),
            python_version=platform.python_version(), **contract, task_stage=task_stage(snapshot),
            device=settings['device'], parallel_environments=1,
            evaluation_policy_lanes=evaluation_settings(settings)['parallel_environments'],
            evaluation_backend='native_fixed_state_v1',
            protocol=request.get('protocol','standard_full_horizon_v1'),
            unique_initial_states=1, deterministic=True, optimizer_updates=0))
        emit(phase='initializing', pid=os.getpid(), round=info['round'], message='Loading frozen policy and physics')
        if stopped():
            summary = dict(phase='cancelled', message='Cancelled before physics initialization')
        else:
            torch.set_num_threads(min(8, os.cpu_count() or 1))
            torch.manual_seed(settings['seed'])
            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(settings['seed'])
            scene = load_configuration(directory/'scene.toml')
            cfg = task_configuration(snapshot['task'], settings, snapshot['model'], scene)
            current_cfg, curriculum = None, None
            if task_stage(snapshot) in (2,3) and 'target_curriculum' in settings:
                if info['target_curriculum'] is None:
                    raise ValueError('Target-curriculum checkpoint has no saved level; level will not be guessed')
                curriculum = TargetCurriculum(**info['target_curriculum'])
                current_cfg = curriculum.configuration(cfg, settings)
            recovery = request.get('protocol') == 'strike_scripted_return_v1'
            if request.get('protocol') not in (None,'strike_scripted_return_v1'):
                raise ValueError('Unknown evaluation protocol')
            if recovery:
                from evaluation.recovery import StrikeReturnEnvironment, RecoverySettings, evaluate_return
                if contract['source_mode'] != 'current_code' or request.get('recovery_target') not in ('current','final'):
                    raise ValueError('Recovery requires explicit target and current-code mode')
                recovery_settings = RecoverySettings(**request['recovery_settings'])
                selected_cfg = current_cfg if current_cfg is not None and request['recovery_target']=='current' else cfg
                env = StrikeReturnEnvironment(settings, snapshot, selected_cfg, scene, progress=lambda e: emit(**e))
            else:
                env = DeterministicEvaluationEnvironment(settings, snapshot, cfg, scene, progress=lambda e: emit(**e))
            columns = info['actor']['encoder.columns']
            if not torch.equal(columns, torch.tensor(env.actor_columns)):
                raise ValueError('Actor observation columns do not match this environment')
            actor = frozen_actor(info, env)
            env.reset()
            last_notice, done = perf_counter(), False
            while not stopped():
                action = env.policy_action(lambda h,i:actor(h,i,deterministic=True)[0])
                _, _, done = env.step(action)
                if done or perf_counter()-last_notice >= 1:
                    emit(phase='evaluating', simulation_s=env.index*.01, duration_s=(env.samples-1)*.01,
                         active=int(env.active.sum()), round=info['round'])
                    last_notice = perf_counter()
                if done:
                    break
            if recovery:
                # New variable-horizon diagnostic: do not reuse full-horizon
                # training reward or append recovery to the historical score.
                arrays, result = evaluate_return(env,recovery_settings,stopped=stopped,emit=emit) if done else (env.replay_arrays(0),dict(status='cancelled_before_hit',success=False))
                filename = 'rollout.npz' if arrays is not None else None
                if arrays is not None:
                    def writer(path):
                        with path.open('wb') as f:
                            np.savez_compressed(f, **arrays)
                    atomic_save(directory/filename,writer)
                summary = clean(dict(phase='cancelled' if stopped() else 'completed',round=info['round'],
                    protocol=request['protocol'],task_stage=task_stage(snapshot),replay=filename,
                    hit=env.hit,recovery=result,recovery_target=request['recovery_target'],
                    target_width_m=env.configuration.strike_surface.width_m,
                    target_height_m=env.configuration.strike_surface.height_m,
                    source_mode=contract['source_mode'],exact_source_reproduction=False,
                    scoring='unscored_variable_horizon_diagnostic',evaluation_return=None,
                    valid=done and not bool(env.failed.any()),components={},
                    strike_phase_workspace={k:float(v[0]) for k,v in env.workspace_metrics().items()},**env.provenance(),
                    elapsed_s=perf_counter()-started,
                    comparison=dict(status='Different episode protocol; historical reward comparison not applicable'),
                    observation='Strike ends at first qualified crossing endpoint. Recovery is separately assessed; no reward is added.'))
                if env.hit is not None and task_stage(snapshot)==2:
                    from tasks.whip.strike import surface_events
                    i=env.index
                    final=surface_events(env.history['cable_positions'][:,i-1:i+1],
                        env.history['cable_velocities'][:,i-1:i+1],env.time[i-1:i+1],
                        env.scorer.center,env.scorer.direction,cfg.strike_surface,env.scorer.speed)
                    summary['strict_final_qualified_hit']=bool(final['qualified'][0,0])
                verify_contract(snapshot,request)
                save_json(directory/'summary.json',summary)
                log(summary)
                return
            if not done:
                summary = dict(phase='cancelled', message='Incomplete episode was not scored', simulation_s=env.index*.01)
            else:
                returns, valid, score, cost = env.finish()
                arrays = env.replay_arrays(0)
                filename = 'rollout.npz' if arrays is not None else None
                if arrays is not None:
                    def writer(path):
                        with path.open('wb') as f:
                            np.savez_compressed(f, **arrays)
                    atomic_save(directory/filename, writer)
                summary = clean(dict(phase='completed', round=info['round'], evaluation_return=float(returns[0]),
                    valid=bool(valid[0]), replay=filename, acceleration_cost=float(cost[0]),
                    workspace={k:float(v[0]) for k,v in env.workspace_metrics().items()},
                    components={k:float(v[0]) for k,v in score.components.items()}, **env.provenance(),
                    observation=envelope_observation(arrays, snapshot) if arrays is not None else 'No valid numerical replay prefix'))
                summary.update(source_mode=contract['source_mode'],
                    exact_source_reproduction=contract['exact_source_reproduction'], task_stage=task_stage(snapshot),
                    scoring=('nominal_precision_and_learned_recovery' if cfg.recovery is not None
                        else 'fixed_broad_target_delivery' if cfg.matching.reward_mode == 'directed_delivery_v1'
                        else 'nominal_crossing_precision_full_horizon' if cfg.matching.reward_mode == 'directed_precision_v1'
                        else 'strict_final_target' if task_stage(snapshot) >= 2 else 'stage1_broad_wall'),
                    termination=cfg.termination)
                if curriculum is not None:
                    env.configuration = current_cfg
                    scorer_factory=make_scorer if env.configuration.recovery is not None else Stage1Scorer
                    env.scorer = scorer_factory(env.model.rest, env.time, env.configuration, initial_position=env.origin)
                    current_return, current_valid, current_score, _ = env.finish()
                    current = clean(dict(stage=curriculum.level+1,
                        target_width_m=env.configuration.strike_surface.width_m,
                        target_height_m=env.configuration.strike_surface.height_m,
                        evaluation_return=float(current_return[0]), valid=bool(current_valid[0]),
                        qualified=bool(current_score.components['strike_success'][0] == 1),
                        components={k:float(v[0]) for k,v in current_score.components.items()},
                        physical_rollout='same as strict-final evaluation', unique_initial_states=1))
                    summary['current_target_evaluation'] = current
                summary['comparison'] = compare_reference(arrays, summary, request['reference'])
                summary['comparison']['interpretation'] = ('New current-code result, not historical reproduction'
                    if contract['source_mode'] == 'current_code' else 'Exact-source deterministic comparison')
                verify_contract(snapshot, request)
        summary['elapsed_s'] = perf_counter()-started
        save_json(directory/'summary.json', summary)
        log(summary)
    except Exception as exc:
        failure = dict(phase='failed', error=str(exc), elapsed_s=perf_counter()-started)
        save_json(directory/'summary.json', failure)
        log(failure)
        traceback.print_exc()
        raise
    finally:
        lock.unlink()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', type=Path, required=True)
    run(parser.parse_args().run)
