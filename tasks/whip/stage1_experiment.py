"""Visible fixed-request CEM experiment using the unchanged Stage-1 objective."""

from dataclasses import dataclass
from datetime import datetime, timezone
import json
import math
from pathlib import Path
import shutil
from time import perf_counter
import tomllib

import numpy as np
import torch

from simulation.batched_rollout import BatchedRollout
from simulation.configuration import load_configuration
from simulation.dynamics.coupled import CoupledDynamics
from training.cem import CEMConfiguration, DiagonalCEM
from trajectories.exploration import smooth_action_matrix
from trajectories.full_state import FullStateTrajectory
from .experiment import ROOT, digest, finite_float, notify, synchronize, write_json
from .resume import inspect_resume
from .stage1 import Stage1Reward
from .stage1_configuration import Stage1Configuration
from .workspace import WorkspaceWhipEnvironment


def load_training(path):
    with Path(path).open('rb') as stream:
        raw = tomllib.load(stream)
    cfg = Stage1Configuration.from_mapping(raw)
    exp = raw['experiment']
    if exp['mode'] != 'training' or exp['device'] not in ('cpu', 'cuda'):
        raise ValueError('Select the Stage-1 training preset with an explicit CPU/CUDA device')
    for key in ('batch_size', 'updates', 'evaluation_interval'):
        if type(exp[key]) is not int or exp[key] < 1:
            raise ValueError(f'{key} must be a positive integer')
    if (not isinstance(exp['seeds'], list) or not exp['seeds'] or len(set(exp['seeds'])) != len(exp['seeds'])
            or any(type(seed) is not int or not 0 <= seed < 2**63 for seed in exp['seeds'])):
        raise ValueError('Provide distinct nonnegative integer seeds')
    cem = CEMConfiguration(**raw['cem'])
    if cem.elites > exp['batch_size']:
        raise ValueError('Batch size must cover the CEM elite set')
    if (set(raw['exploration']) != {'smoothness_time_s', 'amplitude_scale'}
            or any(not math.isfinite(x) or x <= 0 for x in raw['exploration'].values())):
        raise ValueError('Provide positive smooth exploration settings')
    if not all(a < 0 < b for a, b in zip(cfg.trajectory.offset_lower_m, cfg.trajectory.offset_upper_m)):
        raise ValueError('The offset box must contain hover strictly inside it')
    return raw, cfg, cem, None


@dataclass(frozen=True)
class Stage1Episode:
    actions: torch.Tensor
    commands: torch.Tensor
    command_valid: torch.Tensor
    speed_bounds: torch.Tensor
    acceleration_bounds: torch.Tensor
    rollout: object

    @property
    def valid(self):
        return self.command_valid & self.rollout.valid

    def trajectory(self, index):
        if not 0 <= index < len(self.commands) or not bool(self.valid[index]):
            raise ValueError('Cannot export an invalid episode')
        command = self.commands[index].detach().cpu().numpy()
        rows = np.zeros((len(command), 14))
        rows[:, 0] = np.arange(len(command))*.01
        if np.any(command[:, 10] != 0):
            raise ValueError('CSV export requires zero yaw-rate feedforward')
        rows[:, 1:11] = command[:, :10]
        return FullStateTrajectory(rows)


class Stage1Task:
    """Fixed initial state/request. Reuse spline decoding without a fake target."""

    def __init__(self, model, scene, configuration, batch_size):
        cfg, initial = configuration, scene.initial_state
        if (not np.allclose(initial.quaternion_body_to_world_xyzw, [0., 0., 0., 1.], rtol=0, atol=1e-12)
                or np.any(initial.linear_velocity_world_mps != 0) or np.any(initial.angular_velocity_body_radps != 0)
                or initial.cable_node_positions_world_m is not None or initial.cable_node_velocities_world_mps is not None):
            raise ValueError('Stage 1 starts with a level stationary drone and hanging cable')
        if (not np.allclose(scene.platform.attachment_offset_body_m, model.config['attachment']['position_tracking_m'], atol=1e-12, rtol=0)
                or not np.allclose(scene.platform.cable_segment_lengths_m, model.config['cable']['marker_interval_lengths_m'], atol=1e-12, rtol=0)
                or not math.isclose(scene.timestep_s, model.dt, rel_tol=0, abs_tol=1e-12)):
            raise ValueError('Scene geometry/timestep must agree with the frozen model')
        if not cfg.workspace.height_lower_world_m <= initial.position_world_m[2] <= cfg.workspace.height_upper_world_m:
            raise ValueError('Initial drone height must be inside the cylinder')
        simulator = BatchedRollout(model, [model.initial_state(initial.position_world_m)]*batch_size)
        delay = simulator.delay_steps*model.dt
        if (not math.isclose(cfg.mechanics.event_start_s, delay, abs_tol=1e-12, rel_tol=0)
                or not math.isclose(cfg.mechanics.event_end_s, delay+cfg.trajectory.duration_s, abs_tol=1e-12, rel_tol=0)):
            raise ValueError('Stage-1 event bounds must match the configured model command delay')
        self.environment = WorkspaceWhipEnvironment(cfg.trajectory, simulator, cfg.workspace)
        time = torch.arange(round((cfg.trajectory.duration_s+cfg.trajectory.post_hold_s)/model.dt)+1,
                            device=model.rest.device, dtype=model.rest.dtype)*model.dt
        self.reward = Stage1Reward(model.rest, time, cfg, initial_position=initial.position_world_m)
        lower, upper = self.environment.lower, self.environment.upper
        self.initial_action = (-1-2*lower/(upper-lower)).repeat(cfg.trajectory.control_count-5)

    @torch.no_grad()
    def evaluate(self, actions):
        env = self.environment
        _, commands, _, valid, speed, acceleration = env.decode(actions.double())
        if env.post_hold_steps:
            commands = torch.cat((commands, commands[:, -1:].expand(-1, env.post_hold_steps, -1)), 1)
        rollout = env.simulator.run(commands[:, :-1], admissible=valid)
        episode = Stage1Episode(actions.detach().clone(), commands, valid, speed, acceleration, rollout)
        result = self.reward(rollout, commands, admissible=valid)
        if not bool(torch.isfinite(result.returns).all()):
            raise FloatingPointError('Stage 1 must return one finite return per episode')
        return episode, result


def statistics(episode, result):
    valid, event = result.valid, result.valid & result.score.has_event
    avg = lambda value, mask: finite_float(value.masked_fill(~mask, float('nan')).nanmean())
    info = dict(episodes=len(valid), valid_fraction=float(valid.double().mean()),
        command_admissible_fraction=float(episode.command_valid.double().mean()),
        numerical_valid_fraction=float(result.numerical_valid.double().mean()),
        return_mean=float(result.returns.mean()), return_min=float(result.returns.min()), return_max=float(result.returns.max()),
        valid_return_mean=avg(result.returns, valid), acceleration_cost_mean=avg(result.acceleration_cost, valid),
        score_mean=avg(result.score.score, valid), event_fraction=float(event.double().mean()))
    info.update({key+'_mean': avg(value, event) for key, value in result.score.components.items()})
    if 'strike_success' in result.score.components:
        info['strike_fraction'] = avg(result.score.components['strike_success'], valid)
        info['strike_count_mean'] = avg(result.score.components['strike_count'], valid)
        info['first_strike_time_s_mean'] = avg(result.score.components['first_strike_time_s'], valid)
    return info


def save_arrays(path, values):
    with Path(path).open('xb') as stream:
        np.savez_compressed(stream, **{key: value.detach().cpu().numpy() for key, value in values.items()})


def save_records(folder, name, episode, result, *, dense_index=None):
    save_arrays(folder/f'{name}.npz', dict(actions=episode.actions, returns=result.returns, valid=result.valid,
        command_valid=episode.command_valid, numerical_valid=result.numerical_valid, envelope_valid=result.envelope_valid,
        cost=result.acceleration_cost, score=result.score.score, event_indices=result.score.event_indices,
        **result.score.components))
    if dense_index is not None:
        r = episode.rollout
        arrays = {key: getattr(r, key)[dense_index] for key in ('cable_positions', 'cable_velocities', 'drone_positions',
            'drone_velocities', 'drone_rotations', 'drone_angular_velocities', 'state_valid', 'applied_commands')}
        arrays.update(time_s=r.time_s, cable_rest_lengths=r.cable_rest_lengths, cable_marker_indices=r.cable_marker_indices,
                      actions=episode.actions[dense_index], commands=episode.commands[dense_index])
        save_arrays(folder/f'{name}_motion.npz', arrays)
        if bool(result.valid[dense_index]):
            episode.trajectory(dense_index).save(folder/f'{name}.csv')


def evaluate_mean(search, task, folder, label):
    episode, result = task.evaluate(search.act())
    info = statistics(episode, result)
    save_records(folder, label, episode, result, dense_index=0)
    write_json(folder/f'{label}.json', info)
    return info


def train(raw, task, evaluation_task, output, metadata, *, progress, stop_requested, resumed_search=None):
    exp, summaries = raw['experiment'], []
    matrix = smooth_action_matrix(task.environment.configuration.control_count,
        task.environment.configuration.duration_s, **raw['exploration'])
    for seed in exp['seeds']:
        if stop_requested(): break
        folder = output/f'seed_{seed}'; folder.mkdir()
        search = resumed_search or DiagonalCEM(CEMConfiguration(**raw['cem']), task.initial_action, matrix,
            seed=seed, device=exp['device'], metadata=metadata)
        start = last_saved = search.updates
        search.save(folder/f'policy_{start:06d}.pt')
        best, best_update, best_sample = None, None, -math.inf
        if not stop_requested():
            notify(progress, 'phase', phase='CEM · Evaluating', seed=seed, update=start)
            best = evaluate_mean(search, evaluation_task, folder, f'policy_{start:06d}'); best_update = start
            notify(progress, 'evaluation', seed=seed, update=start, metrics=best)
        for update in range(start+1, exp['updates']+1):
            if stop_requested(): break
            notify(progress, 'phase', phase='CEM · Searching', seed=seed, update=update)
            synchronize(exp['device']); began = perf_counter()
            latent, actions = search.sample(exp['batch_size'])
            episode, result = task.evaluate(actions)
            diagnostics = search.update(latent, result.returns)
            synchronize(exp['device'])
            record = dict(seed=seed, update=update, samples=statistics(episode, result), learner=diagnostics,
                          elapsed_s=perf_counter()-began)
            save_records(folder, f'batch_{update:06d}', episode, result)
            if bool(result.valid.any()):
                index = int(result.returns.masked_fill(~result.valid, -float('inf')).argmax())
                if float(result.returns[index]) > best_sample:
                    best_sample = float(result.returns[index])
                    save_records(folder, f'best_sample_{update:06d}', episode, result, dense_index=index)
            write_json(folder/f'update_{update:06d}.json', record)
            notify(progress, 'update', record=record)
            if update % exp['evaluation_interval'] == 0 or update == exp['updates'] or stop_requested():
                search.save(folder/f'policy_{update:06d}.pt'); last_saved = update
                if stop_requested(): break
                notify(progress, 'phase', phase='CEM · Evaluating', seed=seed, update=update)
                evaluated = evaluate_mean(search, evaluation_task, folder, f'policy_{update:06d}')
                notify(progress, 'evaluation', seed=seed, update=update, metrics=evaluated)
                if best is None or evaluated['return_mean'] > best['return_mean']:
                    best, best_update = evaluated, update
        if search.updates != last_saved:
            search.save(folder/f'policy_{search.updates:06d}.pt')
        info = dict(seed=seed, start_update=start, updates=search.updates,
            training_episodes=(search.updates-start)*exp['batch_size'], cumulative_training_episodes=search.updates*exp['batch_size'],
            status='completed' if search.updates == exp['updates'] and not stop_requested() else 'stopped',
            latest_checkpoint=f'policy_{search.updates:06d}.pt', best_evaluated_policy_update=best_update,
            best_evaluated_policy=best, best_sample_return=None if best_sample == -math.inf else best_sample)
        write_json(folder/'summary.json', info); summaries.append(info)
        notify(progress, 'seed_finished', metrics=info)
    completed = len(summaries) == len(exp['seeds']) and all(s['status'] == 'completed' for s in summaries)
    return dict(mode='training', status='completed' if completed else 'stopped', seeds=summaries,
        training_episodes=sum(s['training_episodes'] for s in summaries),
        cumulative_training_episodes=sum(s['cumulative_training_episodes'] for s in summaries))


def run(configuration_path, *, progress=None, stop_requested=lambda: False, expected_configuration_sha256=None,
        total_updates=None, resume_checkpoint=None, expected_resume_sha256=None):
    path = Path(configuration_path).resolve()
    before = digest(path)
    if expected_configuration_sha256 is not None and before != expected_configuration_sha256:
        raise ValueError('Configuration changed after preview')
    raw, cfg, cem, _ = load_training(path); exp = raw['experiment']
    if total_updates is not None:
        if type(total_updates) is not int or total_updates < 1: raise ValueError('Total updates must be positive')
        exp['updates'] = total_updates
    resume = inspect_resume(resume_checkpoint, raw) if resume_checkpoint is not None else None
    if expected_resume_sha256 is not None and (resume is None or resume['sha256'] != expected_resume_sha256):
        raise ValueError('Checkpoint changed after preview')
    device = exp['device']
    if device == 'cuda' and not torch.cuda.is_available(): raise RuntimeError('CUDA unavailable; no CPU fallback')
    inputs = dict(configuration=path, scene=ROOT/cfg.scene, model=ROOT/cfg.model)
    model_config = json.loads(inputs['model'].read_text(encoding='utf-8'))
    residual = model_config['residual']; inputs['residual'] = ROOT/residual['checkpoint']
    if not residual['enabled'] or digest(inputs['residual']) != residual['sha256']:
        raise ValueError('The fitted residual must remain enabled with its recorded identity')
    if resume is not None: inputs['resume_checkpoint'] = Path(resume['checkpoint'])
    sources = {key: dict(path=str(value), sha256=digest(value)) for key, value in inputs.items()}
    if sources['configuration']['sha256'] != before: raise ValueError('Configuration changed while loading')
    output = ROOT/'runs/whip_stage1'/datetime.now(timezone.utc).strftime('%Y%m%d-%H%M%S-%f')
    output.mkdir(parents=True, exist_ok=False)
    for key, value in inputs.items():
        shutil.copy2(value, output/(key+value.suffix))
        if digest(output/(key+value.suffix)) != sources[key]['sha256']: raise ValueError(f'Input changed while copying: {key}')
    code = {}
    paths = [ROOT/'run_training.py']
    for package in ('tasks/whip', 'training', 'training/ui', 'trajectories', 'simulation', 'simulation/dynamics'):
        paths.extend((ROOT/package).glob('*.py'))
    for source in paths:
        relative = source.relative_to(ROOT); destination = output/'source_snapshot'/relative
        destination.parent.mkdir(parents=True, exist_ok=True); shutil.copy2(source, destination)
        code[str(relative)] = digest(destination)
    metadata = dict(schema='whip_stage1_v1', experiment=raw, sources=sources, code_sha256=code,
        method='Smoothed diagonal latent CEM; fixed initial state/request, not a contextual policy or iCEM',
        score=('Multiplicative directed-surface Stage1Reward; no hit bonus; crossing diagnostics are not whip labels'
               if cfg.strike_surface is not None else
               'Approved Stage1Reward; experimental shaping, not a whip classifier'),
        evaluation='Batch-one transformed CEM mean; best sample stored separately; no held-out/generalization claim',
        versions=dict(torch=str(torch.__version__), numpy=np.__version__),
        gpu=torch.cuda.get_device_name() if device == 'cuda' else None)
    if resume is not None:
        metadata['resume'] = {key: value for key, value in resume.items() if key not in ('history', 'protocol')}
        write_json(output/'resume_history.json', resume['history']); write_json(output/'parent_protocol.json', resume['protocol'])
    write_json(output/'protocol.json', metadata)

    def emit(event):
        with (output/'events.jsonl').open('a', encoding='utf-8') as stream:
            stream.write(json.dumps(event, allow_nan=False)+'\n')
        if progress is not None: progress(event)

    notify(emit, 'run', path=str(output), configuration=raw, resume=metadata.get('resume'))
    began = perf_counter()
    try:
        restored = None
        if resume is not None:
            if resume['protocol']['versions'] != metadata['versions'] or resume['protocol']['gpu'] != metadata['gpu']:
                raise ValueError('Resume requires the saved library versions and GPU type')
            restored = DiagonalCEM.load(output/'resume_checkpoint.pt', device=device, expected_metadata=resume['protocol'])
            if restored.updates != resume['update'] or restored.configuration != cem or restored.seed != resume['seed']:
                raise ValueError('Checkpoint state does not match the selected continuation')
            restored.metadata = metadata
        if stop_requested():
            info = dict(mode='training', status='stopped', seeds=[], training_episodes=0)
        else:
            notify(emit, 'phase', phase='Preparing')
            model_config['residual']['checkpoint'] = str(output/'residual.pt')
            model = CoupledDynamics(model_config, device=device, batched=True)
            scene = load_configuration(output/'scene.toml')
            task = Stage1Task(model, scene, cfg, exp['batch_size'])
            evaluation_task = Stage1Task(model, scene, cfg, 1)
            info = train(raw, task, evaluation_task, output, metadata, progress=emit,
                         stop_requested=stop_requested, resumed_search=restored)
        info['elapsed_seconds'] = perf_counter()-began
        for key, value in inputs.items():
            if digest(value) != sources[key]['sha256']: raise RuntimeError(f'Input changed during experiment: {key}')
        write_json(output/'summary.json', info)
        if info['status'] == 'stopped':
            write_json(output/'cancellation.json', dict(status='stopped', reason='UI Stop or UI disconnected',
                checkpoints=[str(p.relative_to(output)) for p in output.glob('seed_*/policy_*.pt')]))
    except Exception as error:
        write_json(output/'failure.json', dict(status='aborted', type=type(error).__name__, message=str(error)))
        raise
    finally:
        write_json(output/'artifacts.json', {str(p.relative_to(output)): digest(p) for p in output.rglob('*') if p.is_file()})
    # Final event is not appended after the artifact manifest was finalized.
    notify(progress, 'finished', path=str(output), summary=info)
    return output
