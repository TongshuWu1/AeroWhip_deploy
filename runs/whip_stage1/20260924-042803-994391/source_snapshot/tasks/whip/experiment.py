"""Single-scenario experiment with optional progress and cooperative stopping."""

from datetime import datetime, timezone
import hashlib
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
from training.configuration import LearnerConfiguration
from training.episodic import EpisodicLearner
from trajectories.exploration import smooth_action_matrix
from .configuration import TrainingConfiguration
from .environment import WhipEnvironment
from .reward import WhipReturnConfiguration, WhipReward
from .scoring import WhipScoreConfiguration
from .resume import inspect_resume
from .workspace import CylinderWorkspace, WorkspaceWhipEnvironment, WorkspaceWhipReward
from .reach import ReachConfiguration, ReachReward, ReachReturn
from . import reach as reach_metrics


ROOT = Path(__file__).resolve().parents[2]


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write_json(path, value):
    with Path(path).open('x', encoding='utf-8') as stream:
        json.dump(value, stream, indent=2, allow_nan=False)
        stream.write('\n')


def load_experiment(path):
    with Path(path).open('rb') as stream:
        raw = tomllib.load(stream)
    if raw.get('experiment', {}).get('name') == 'whip_stage1':
        from .stage1_experiment import load_training
        return load_training(path)
    required = {'experiment', 'trajectory', 'reward', 'return', 'learner'}
    if not required <= set(raw) or set(raw)-required-{'exploration', 'workspace', 'reach'}:
        raise ValueError('Provide experiment, trajectory, reward, return, learner and optional exploration/workspace/reach sections')
    if 'reach' in raw:
        ReachConfiguration(**raw['reach'])
        if 'workspace' not in raw:
            raise ValueError('Reach-first training requires an explicit cylindrical workspace')
    if 'exploration' in raw:
        exploration = raw['exploration']
        if (set(exploration) != {'smoothness_time_s', 'amplitude_scale'}
                or any(not math.isfinite(x) or x <= 0 for x in exploration.values())):
            raise ValueError('Provide positive exploration smoothness_time_s and amplitude_scale')
    exp = raw['experiment']
    expected = {'name', 'mode', 'device', 'scene', 'model', 'batch_size', 'updates', 'seeds',
                'evaluation_interval', 'target_position_world_m'}
    if set(exp) != expected or exp['name'] != 'whip_single_scenario':
        raise ValueError('Unexpected single-scenario experiment fields/name')
    if exp['mode'] not in ('initialization', 'training') or exp['device'] not in ('cpu', 'cuda'):
        raise ValueError('Mode must be initialization/training and device cpu/cuda')
    for key in ('batch_size', 'updates', 'evaluation_interval'):
        if type(exp[key]) is not int or exp[key] < 1:
            raise ValueError(f'{key} must be a positive integer')
    if (not isinstance(exp['seeds'], list) or not exp['seeds']
            or len(set(exp['seeds'])) != len(exp['seeds'])
            or any(type(seed) is not int or not 0 <= seed < 2**63 for seed in exp['seeds'])):
        raise ValueError('Provide distinct nonnegative integer seeds')
    target = np.asarray(exp['target_position_world_m'], dtype=float)
    if target.shape != (3,) or not np.isfinite(target).all():
        raise ValueError('Provide one finite world-XYZ target')
    trajectory = TrainingConfiguration(**raw['trajectory'])
    scoring = WhipScoreConfiguration(**raw['reward'])
    returns = WhipReturnConfiguration(**raw['return'])
    if 'workspace' in raw:
        workspace = CylinderWorkspace(**raw['workspace'])
        scene = load_configuration(ROOT/exp['scene'])
        workspace.validate_placement(scene.initial_state.position_world_m, target)
    if not all(a < 0 < b for a, b in zip(trajectory.offset_lower_m, trajectory.offset_upper_m)):
        raise ValueError('The offset box must contain hover strictly inside it')
    for seed in exp['seeds']:
        LearnerConfiguration(context_size=3, action_size=trajectory.action_size, seed=seed, **raw['learner'])
    return raw, trajectory, scoring, returns


class SingleScenarioTask:
    def __init__(self, model, scene, trajectory, scoring, returns, target, batch_size, *, workspace=None, reach=None):
        initial = scene.initial_state
        if workspace is not None:
            workspace.validate_placement(initial.position_world_m, target)
        # This protocol fixes the initial state, not a deployable state encoder.
        if (not np.allclose(initial.quaternion_body_to_world_xyzw, [0., 0., 0., 1.], rtol=0, atol=1e-12)
                or np.any(initial.linear_velocity_world_mps != 0)
                or np.any(initial.angular_velocity_body_radps != 0)
                or initial.cable_node_positions_world_m is not None
                or initial.cable_node_velocities_world_mps is not None):
            raise ValueError('Single-scenario initialization requires a level stationary drone and hanging cable')
        if (not np.allclose(scene.platform.attachment_offset_body_m,
                            model.config['attachment']['position_tracking_m'], atol=1e-12, rtol=0)
                or not np.allclose(scene.platform.cable_segment_lengths_m,
                                   model.config['cable']['marker_interval_lengths_m'], atol=1e-12, rtol=0)
                or not math.isclose(scene.timestep_s, model.dt, rel_tol=0, abs_tol=1e-12)):
            raise ValueError('Scene geometry/timestep must agree with the frozen physical model')
        state = model.initial_state(initial.position_world_m)
        simulator = BatchedRollout(model, [state]*batch_size)
        if workspace is None:
            self.environment = WhipEnvironment(trajectory, simulator)
            self.reward = WhipReward(self.environment, scoring, returns)
        else:
            self.environment = WorkspaceWhipEnvironment(trajectory, simulator, workspace)
            self.reward = WorkspaceWhipReward(self.environment, scoring, returns)
        if reach is not None:
            if workspace is None:
                raise ValueError('Reach-first training requires a cylindrical workspace')
            self.reward = ReachReward(self.reward, reach)
        self.targets = model.tensor(target)[None].expand(batch_size, -1).clone()
        # Dimensionless target offset; the complete initial state is FIXED in
        # this experiment. This must not be called initial-state generalization.
        self.contexts = (self.targets-self.environment.start_position)/model.rest.sum()
        lower, upper = self.environment.lower, self.environment.upper
        self.initial_action = (-1-2*lower/(upper-lower)).repeat(trajectory.control_count-5)

    @torch.no_grad()
    def evaluate(self, actions):
        episode = self.environment.run(actions.double(), self.targets)
        result = self.reward(episode)
        if not bool(torch.isfinite(result.returns).all()):
            raise FloatingPointError('The task did not produce one finite return per episode')
        return episode, result


def finite_float(value):
    value = float(value)
    return value if math.isfinite(value) else None


def statistics(episode, result):
    valid = result.valid

    def average(value, mask):
        return finite_float(value.masked_fill(~mask, float('nan')).nanmean())

    info = dict(episodes=len(valid), valid_fraction=float(valid.double().mean()),
        command_admissible_fraction=float((episode.command_valid & episode.context_valid).double().mean()),
        numerical_valid_fraction=float(result.numerical_valid.double().mean()),
        envelope_valid_fraction_among_numerical=average(result.envelope_valid.double(), result.numerical_valid),
        return_mean=float(result.returns.mean()), return_min=float(result.returns.min()),
        return_max=float(result.returns.max()), valid_return_mean=average(result.returns, valid),
        acceleration_cost_mean=average(result.acceleration_cost, valid),
        closest_distance_mean_m=average(result.closest_distance_m, valid))
    if isinstance(result, ReachReturn):
        info.update(reach_metrics.statistics(result))
        return info
    event = valid & result.score.has_event
    info.update(event_fraction=float(event.double().mean()),
        score_mean=average(result.score.score, valid), bend_mean=average(result.score.bend_score, event),
        speed_score_mean=average(result.score.speed_score, event),
        target_score_mean=average(result.score.target_score, event),
        selected_distance_mean_m=average(result.score.target_distance_m, event),
        selected_tip_speed_mean_mps=average(result.selected_tip_speed_mps, event),
        event_times_mean_s=[average(result.score.event_times_s[:, j], event) for j in range(4)])
    return info


def save_episode_records(folder, name, episode, result):
    # Per-sample diagnostics, not every trajectory/state history at every update.
    values = dict(actions=episode.actions, returns=result.returns, valid=result.valid,
        command_valid=episode.command_valid, numerical_valid=result.numerical_valid,
        envelope_valid=result.envelope_valid, score_valid=result.score_valid,
        speed_bounds=episode.speed_bounds, acceleration_bounds=episode.acceleration_bounds,
        cost=result.acceleration_cost, closest_distance_m=result.closest_distance_m,
        selected_tip_speed_mps=result.selected_tip_speed_mps)
    if isinstance(result, ReachReturn):
        values.update(reach_metrics.record_arrays(result))
    else:
        values.update(score=result.score.score, bend=result.score.bend_score, speed=result.score.speed_score,
            target=result.score.target_score, event_indices=result.score.event_indices,
            event_times_s=result.score.event_times_s, selected_distance_m=result.score.target_distance_m)
    with (folder/f'{name}.npz').open('xb') as stream:
        np.savez_compressed(stream, **{key: value.detach().cpu().numpy() for key, value in values.items()})


def synchronize(device):
    if device == 'cuda':
        torch.cuda.synchronize()


def make_learner(raw, task, seed, metadata):
    cfg = LearnerConfiguration(context_size=3, action_size=task.environment.configuration.action_size,
                               seed=seed, **raw['learner'])
    matrix = None
    if 'exploration' in raw:
        trajectory = task.environment.configuration
        matrix = smooth_action_matrix(trajectory.control_count, trajectory.duration_s, **raw['exploration'])
    return EpisodicLearner(cfg, device=raw['experiment']['device'],
        initial_action=task.initial_action.cpu(), metadata=metadata, action_matrix=matrix)


def restore_learner(checkpoint, raw, trajectory, resume, metadata):
    """Restore the complete learner; only replace its run-provenance metadata."""
    if (resume['protocol']['versions'] != metadata['versions']
            or resume['protocol']['gpu'] != metadata['gpu']):
        raise ValueError('Resume requires the saved library versions and GPU type')
    learner = EpisodicLearner.load(checkpoint, device=raw['experiment']['device'],
                                  expected_metadata=resume['protocol'])
    expected = LearnerConfiguration(context_size=3, action_size=trajectory.action_size,
                                    seed=resume['seed'], **raw['learner'])
    if learner.configuration != expected or learner.updates != resume['update']:
        raise ValueError('Checkpoint learner settings/update do not match the experiment')
    # Match the fresh learner's JSON normalization (torch.__version__ can be a
    # str subclass). Saved metadata must remain weights-only-load compatible.
    learner.metadata = json.loads(json.dumps(metadata, allow_nan=False))
    return learner


def evaluate_policy(learner, task, folder, label):
    # One-lane deterministic evaluation. Batch-one and training batches are
    # separately recorded; cross-batch floating-point equivalence is not assumed.
    episode, result = task.evaluate(learner.act(task.contexts))
    summary = statistics(episode, result)
    write_json(folder/f'{label}.json', summary)
    save_episode_records(folder, label, episode, result)
    if bool(result.valid[0]):
        episode.trajectory(0).save(folder/f'{label}.csv')
    return summary


def notify(progress, event, **values):
    if progress is not None:
        progress(dict(event=event, **values))


def initialization(raw, task, evaluation_task, output, metadata, *, progress=None, stop_requested=lambda: False):
    device = raw['experiment']['device']
    seed = raw['experiment']['seeds'][0]
    learner = make_learner(raw, task, seed, metadata)
    notify(progress, 'phase', phase='Evaluating', seed=seed, update=0)
    initial_parameters = {k: v.detach().clone() for k, v in learner.policy.state_dict().items()}
    deterministic = evaluate_policy(learner, evaluation_task, output, 'initial_policy')
    notify(progress, 'evaluation', seed=seed, update=0, metrics=deterministic)
    if stop_requested():
        learner.save(output/'initial_policy.pt')
        return dict(mode='initialization', status='stopped', seed=seed, optimizer_updates=0,
                    training_episodes=0, deterministic_policy=deterministic)
    if device == 'cuda':
        torch.cuda.reset_peak_memory_stats()
    synchronize(device)
    started = perf_counter()
    notify(progress, 'phase', phase='Sampling', seed=seed, update=0)
    batch = learner.sample(task.contexts)
    episode, result = task.evaluate(batch.actions)
    synchronize(device)
    elapsed = perf_counter()-started
    info = statistics(episode, result)
    info.update(seed=seed, optimizer_updates=learner.updates, mode='initialization',
        sample_rollout_scoring_seconds=elapsed, attempted_episodes_per_second=len(batch.actions)/elapsed,
        cuda_peak_allocated_bytes=torch.cuda.max_memory_allocated() if device == 'cuda' else None,
        cuda_peak_reserved_bytes=torch.cuda.max_memory_reserved() if device == 'cuda' else None,
        deterministic_policy=deterministic)
    save_episode_records(output, 'initial_samples', episode, result)
    if bool(result.valid.any()):
        selected = int(result.returns.masked_fill(~result.valid, -float('inf')).argmax())
        episode.trajectory(selected).save(output/'best_initial_sample.csv')
        info['best_initial_sample_index'] = selected
    learner.discard_batch()
    assert learner.updates == 0
    for key, value in learner.policy.state_dict().items():
        if not torch.equal(value, initial_parameters[key]):
            raise RuntimeError('Initialization must not change policy parameters')
    learner.save(output/'initial_policy.pt')
    info['status'] = 'stopped' if stop_requested() else 'completed'
    notify(progress, 'initialization', metrics=info)
    return info


def train(raw, task, evaluation_task, output, metadata, *, progress=None, stop_requested=lambda: False,
          resumed_learner=None):
    exp = raw['experiment']
    all_seeds = []
    for seed in exp['seeds']:
        if stop_requested():
            break
        folder = output/f'seed_{seed}'
        folder.mkdir()
        learner = resumed_learner if resumed_learner is not None else make_learner(raw, task, seed, metadata)
        start_update = learner.updates
        label = f'policy_{start_update:06d}'
        learner.save(folder/f'{label}.pt')
        notify(progress, 'phase', phase='Evaluating', seed=seed, update=start_update)
        best = evaluate_policy(learner, evaluation_task, folder, label)
        notify(progress, 'evaluation', seed=seed, update=start_update, metrics=best)
        best_update = start_update
        # Best is within this continuation (including its starting checkpoint),
        # not a claim that a previous run's older best policy was overwritten.
        last_checkpoint = start_update
        for update in range(start_update+1, exp['updates']+1):
            if stop_requested():
                break
            notify(progress, 'phase', phase='Training', seed=seed, update=update)
            synchronize(exp['device']); started = perf_counter()
            batch = learner.sample(task.contexts)
            episode, result = task.evaluate(batch.actions)
            sample_summary = statistics(episode, result)
            diagnostics = learner.update(batch, result.returns)
            synchronize(exp['device'])
            record = dict(seed=seed, update=update, samples=sample_summary, learner=diagnostics,
                          elapsed_s=perf_counter()-started)
            # Failed lanes are retained in both records and updates.
            save_episode_records(folder, f'batch_{update:06d}', episode, result)
            write_json(folder/f'update_{update:06d}.json', record)
            if progress is None:
                print(json.dumps(record, allow_nan=False), flush=True)
            notify(progress, 'update', record=record)
            if stop_requested():
                break
            if update % exp['evaluation_interval'] == 0 or update == exp['updates']:
                label = f'policy_{update:06d}'
                learner.save(folder/f'{label}.pt')
                last_checkpoint = update
                if stop_requested():
                    break
                notify(progress, 'phase', phase='Evaluating', seed=seed, update=update)
                evaluated = evaluate_policy(learner, evaluation_task, folder, label)
                notify(progress, 'evaluation', seed=seed, update=update, metrics=evaluated)
                if evaluated['return_mean'] > best['return_mean']:
                    best, best_update = evaluated, update
        # Stop at an update boundary, with no outstanding on-policy batch.
        # Save before returning even when this is not an evaluation interval.
        if learner.updates != last_checkpoint:
            learner.save(folder/f'policy_{learner.updates:06d}.pt')
        info = dict(seed=seed, start_update=start_update, updates=learner.updates,
                    training_episodes=(learner.updates-start_update)*exp['batch_size'],
                    cumulative_training_episodes=learner.updates*exp['batch_size'],
                    status='completed' if learner.updates == exp['updates'] and not stop_requested() else 'stopped',
                    latest_checkpoint=f'policy_{learner.updates:06d}.pt',
                    best_evaluated_policy_update=best_update, best_evaluated_policy=best)
        write_json(folder/'summary.json', info)
        all_seeds.append(info)
        notify(progress, 'seed_finished', metrics=info)
    completed = len(all_seeds) == len(exp['seeds']) and all(s['status'] == 'completed' for s in all_seeds)
    return dict(mode='training', status='completed' if completed else 'stopped', seeds=all_seeds,
                training_episodes=sum(x['training_episodes'] for x in all_seeds),
                cumulative_training_episodes=sum(x['cumulative_training_episodes'] for x in all_seeds))


def run_experiment(configuration_path, *, progress=None, stop_requested=lambda: False,
                   expected_configuration_sha256=None, total_updates=None, resume_checkpoint=None,
                   expected_resume_sha256=None):
    configuration_path = Path(configuration_path).resolve()
    if expected_configuration_sha256 is not None and digest(configuration_path) != expected_configuration_sha256:
        raise ValueError('Configuration changed after preview; reload it before starting')
    raw, trajectory, scoring, returns = load_experiment(configuration_path)
    if raw['experiment']['name'] == 'whip_stage1':
        from .stage1_experiment import run
        return run(configuration_path, progress=progress, stop_requested=stop_requested,
            expected_configuration_sha256=expected_configuration_sha256, total_updates=total_updates,
            resume_checkpoint=resume_checkpoint, expected_resume_sha256=expected_resume_sha256)
    exp = raw['experiment']; device = exp['device']
    if total_updates is not None:
        if exp['mode'] != 'training' or type(total_updates) is not int or total_updates < 1:
            raise ValueError('Total updates must be a positive integer for training mode')
        exp['updates'] = total_updates
    resume = inspect_resume(resume_checkpoint, raw) if resume_checkpoint is not None else None
    if expected_resume_sha256 is not None and (resume is None or resume['sha256'] != expected_resume_sha256):
        raise ValueError('Checkpoint changed after preview; select it again')
    if device == 'cuda' and not torch.cuda.is_available():
        raise RuntimeError('This experiment requests CUDA; no silent CPU fallback or batch-size change')
    scene_path, model_path = ((ROOT/exp[key]).resolve() for key in ('scene', 'model'))
    scene = load_configuration(scene_path)
    model_config = json.loads(model_path.read_text(encoding='utf-8'))
    model_reference = model_config.get('residual', {})
    if not model_reference.get('enabled', False):
        raise ValueError('The approved experiment requires the fitted residual to remain enabled')
    residual_path = (ROOT/model_reference['checkpoint']).resolve()
    if digest(residual_path) != model_reference['sha256']:
        raise ValueError('Residual checkpoint hash does not match model')
    stamp = datetime.now(timezone.utc).strftime('%Y%m%d-%H%M%S-%f')
    output = ROOT/'runs'/exp['name']/stamp
    output.mkdir(parents=True, exist_ok=False)
    if progress is None:
        print(f'Experiment: {exp["mode"]}; output: {output}', flush=True)
    input_paths = dict(configuration=configuration_path, scene=scene_path, model=model_path, residual=residual_path)
    if resume is not None:
        input_paths['resume_checkpoint'] = Path(resume['checkpoint'])
    sources = {name: dict(path=str(path), sha256=digest(path)) for name, path in input_paths.items()}
    if expected_configuration_sha256 is not None and sources['configuration']['sha256'] != expected_configuration_sha256:
        raise ValueError('Configuration changed while loading')
    for name, path in input_paths.items():
        shutil.copy2(path, output/(name+path.suffix))
    if resume is not None:
        if digest(output/'resume_checkpoint.pt') != resume['sha256']:
            raise ValueError('Checkpoint changed while copying')
        write_json(output/'resume_history.json', resume['history'])
        write_json(output/'parent_protocol.json', resume['protocol'])
    code_paths = [ROOT/'run_training.py']
    for package in ('tasks/whip', 'training', 'training/ui', 'trajectories', 'simulation'):
        code_paths.extend((ROOT/package).glob('*.py'))
    code_paths.extend((ROOT/'simulation/dynamics').glob('*.py'))
    code_hashes = {}
    for path in code_paths:
        relative = path.relative_to(ROOT)
        destination = output/'source_snapshot'/relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(path, destination)
        code_hashes[str(relative)] = digest(path)
    metadata = dict(schema='whip_single_scenario_v1', sources=sources, code_sha256=code_hashes,
        launch_options=dict(total_updates=total_updates),
        experiment=raw, context='(target_world - initial_tracking_position_world) / cable_rest_length; '
                                 'three dimensionless XYZ features; all other initial state fixed',
        score='PeakWhipScorer with explicit experiment configuration; not a physical whip classifier',
        envelope='100-Hz state-sample checks over maneuver and final hold; no continuous collision certificate',
        cost='Mean squared command acceleration / max_acceleration^2 over maneuver left-endpoint samples',
        event_window='[command_delay, command_delay + duration]; tip allowed another configured tip_window_s',
        exploration=raw.get('exploration', 'independent; identity latent-to-control map'),
        action_transform='tanh(offset + matrix @ latent); full-rank fixed float64 matrix and offset saved in policy'
            if 'exploration' in raw else 'tanh(latent)',
        evaluation='Batch-one deterministic transformed latent mean, every configured interval and at initialization/final; '
                   'not guaranteed numerically identical to training batch; same scenario, not held-out generalization',
        versions=dict(torch=torch.__version__, numpy=np.__version__),
        gpu=torch.cuda.get_device_name() if device == 'cuda' else None)
    if 'workspace' in raw:
        metadata['workspace'] = CylinderWorkspace(**raw['workspace']).description(scene.initial_state.position_world_m)
        metadata['workspace_constraint'] = (
            'Fixed initial-XY cylinder for tracking-origin motion; all B-spline controls must be inside '
            '(conservative continuous command bound), and all 100-Hz drone states including hold must be inside; '
            'violations receive the configured failure return; no wall forces, clipping or cable confinement')
    if 'reach' in raw:
        metadata['score'] = ('Reach-first: max_t P(t)*(1+beta*Q(t)); P=max(0,1-d_window(t)/d_window(0)); '
            'Q is the existing same-sample peak B*V proxy, not a physical whip classifier')
        metadata['reach'] = raw['reach']
        metadata['reach_window'] = ('Finite vertical rectangle centered on target; horizontal normal from initial '
            'drone XY to target XY. Full width/height in metres. No contact or binary hit model; no curriculum.')
    if resume is not None:
        metadata['resume'] = {k: v for k, v in resume.items() if k not in ('protocol', 'history')}
    write_json(output/'protocol.json', metadata)
    notify(progress, 'run', path=str(output), configuration=raw, resume=metadata.get('resume'))
    run_started = perf_counter()
    try:
        started = perf_counter()
        notify(progress, 'phase', phase='Preparing')
        resumed_learner = None
        if resume is not None:
            resumed_learner = restore_learner(output/'resume_checkpoint.pt', raw, trajectory, resume, metadata)
        # Read only retained assets, not a checkpoint that could change mid-run.
        model_config['residual']['checkpoint'] = str(output/'residual.pt')
        model = CoupledDynamics(model_config, device=device, batched=True)
        if progress is None:
            print('Preparing fixed-batch CUDA graphs and reward evaluation...', flush=True)
        workspace = CylinderWorkspace(**raw['workspace']) if 'workspace' in raw else None
        reach = ReachConfiguration(**raw['reach']) if 'reach' in raw else None
        task = SingleScenarioTask(model, scene, trajectory, scoring, returns,
                                  exp['target_position_world_m'], exp['batch_size'], workspace=workspace, reach=reach)
        evaluation_task = SingleScenarioTask(model, scene, trajectory, scoring, returns,
                                             exp['target_position_world_m'], 1, workspace=workspace, reach=reach)
        synchronize(device)
        setup_seconds = perf_counter()-started
        if progress is None:
            print(f'Setup complete ({setup_seconds:.2f} s); running {exp["mode"]}.', flush=True)
        if stop_requested():
            info = dict(mode=exp['mode'], status='stopped', seeds=[], training_episodes=0)
        else:
            options = {} if exp['mode'] == 'initialization' else dict(resumed_learner=resumed_learner)
            info = (initialization if exp['mode'] == 'initialization' else train)(
                raw, task, evaluation_task, output, metadata, progress=progress, stop_requested=stop_requested, **options)
        info['setup_seconds'] = setup_seconds
        info['elapsed_seconds'] = perf_counter()-run_started
        for name, path in input_paths.items():
            if digest(path) != sources[name]['sha256']:
                raise RuntimeError(f'Input changed during experiment: {name}')
        write_json(output/'summary.json', info)
        if info['status'] == 'stopped':
            checkpoints = [str(p.relative_to(output)) for p in output.glob('seed_*/policy_*.pt')]
            if (output/'initial_policy.pt').exists():
                checkpoints.append('initial_policy.pt')
            write_json(output/'cancellation.json', dict(status='stopped', reason='Requested by controller or controller disconnected',
                stopped_utc=datetime.now(timezone.utc).isoformat(), training_episodes=info['training_episodes']
                if 'training_episodes' in info else 0, checkpoints=checkpoints))
        if progress is None:
            print(json.dumps(info, allow_nan=False), flush=True)
    except Exception as exc:
        write_json(output/'failure.json', dict(type=type(exc).__name__, message=str(exc),
                                               mode=exp['mode'], status='aborted'))
        raise
    finally:
        write_json(output/'artifacts.json', {str(p.relative_to(output)): digest(p)
            for p in output.rglob('*') if p.is_file()})
    notify(progress, 'finished', path=str(output), summary=info)
    return output
