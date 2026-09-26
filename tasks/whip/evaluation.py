"""Read saved policies and explicitly evaluate one episode; never train."""

import hashlib
import json
import math
import os
from pathlib import Path, PureWindowsPath
import re
import shutil
import sys
import traceback
from zipfile import BadZipFile

import numpy as np


ROOT = Path(__file__).resolve().parents[2]


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read_json(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def relative_record_name(value):
    """Interpret retained Windows/POSIX relative keys without changing hashes."""
    path = PureWindowsPath(value)
    if path.drive or path.root or '..' in path.parts or not path.parts:
        raise ValueError(f'Expected a relative artifact path: {value}')
    return path.as_posix()


def discover_runs(folder):
    """Find supported run protocols at any depth, including folder itself."""
    runs = []
    if not Path(folder).exists():
        return runs
    def unreadable(error):
        raise error
    for directory, children, files in os.walk(folder, onerror=unreadable):
        children[:] = [name for name in children if not name.startswith('.')
                       and name not in ('__pycache__', 'source_snapshot', 'snapshot_fixture', 'dependencies')]
        if 'protocol.json' not in files:
            continue
        path = Path(directory)
        try:
            protocol = read_json(path/'protocol.json')
            supported = (protocol.get('schema') in ('whip_single_scenario_v1', 'whip_stage1_v1')
                         and protocol['experiment']['experiment']['mode'] == 'training')
        except (OSError, ValueError, KeyError, TypeError, AttributeError):
            continue
        if supported:
            runs.append(path.resolve())
            children.clear()  # Do not search checkpoint/source copies within a run.
    return sorted(runs, key=lambda path: (path.name, str(path)), reverse=True)


def verify_artifacts(folder, paths, *, required=False):
    manifest_path = Path(folder)/'artifacts.json'
    if not required and not manifest_path.exists():
        return
    original = read_json(manifest_path)
    manifest = {relative_record_name(key): value for key, value in original.items()}
    if len(manifest) != len(original):
        raise ValueError('Duplicate artifact paths in manifest')
    for path in paths:
        path = Path(path)
        if manifest.get(path.relative_to(folder).as_posix()) != digest(path):
            raise ValueError(f'Saved artifact hash mismatch: {path.name}')


def policy_record(checkpoint):
    """Only expose complete batch-one evaluation records, including during training."""
    path = Path(checkpoint).resolve()
    match = re.fullmatch(r'policy_(\d+)\.pt', path.name)
    seed = re.fullmatch(r'seed_(\d+)', path.parent.name)
    if match is None or seed is None or not path.is_file():
        raise ValueError('Select an evaluated policy from a saved seed folder')
    metrics = read_json(path.with_suffix('.json'))
    if metrics.get('episodes') != 1 or not math.isfinite(metrics['return_mean']):
        raise ValueError('The saved evaluation must contain one finite deterministic return')
    # JSON is written before NPZ; opening every array also detects partial ZIP writes.
    with np.load(path.with_suffix('.npz'), allow_pickle=False) as data:
        values = {key: data[key] for key in data.files}
    if (values['returns'].shape != (1,) or values['actions'].shape[0] != 1
            or float(values['returns'][0]) != metrics['return_mean']):
        raise ValueError('Saved evaluation records disagree')
    return dict(checkpoint=str(path), seed=int(seed[1]), update=int(match[1]), metrics=metrics)


def evaluated_policies(run):
    records = []
    for path in Path(run).glob('seed_*/policy_*.pt'):
        try:
            records.append(policy_record(path))
        except (OSError, ValueError, KeyError, IndexError, BadZipFile, EOFError):
            # A training worker may still be writing this evaluation. Never select it.
            continue
    return sorted(records, key=lambda row: (row['seed'], row['update']))


def best_policy(records):
    if not records:
        raise ValueError('This run has no completed policy evaluations yet')
    # Stable tie-break: keep the earlier checkpoint, as the experiment does.
    return max(records, key=lambda row: (row['metrics']['return_mean'], -row['update'], -row['seed']))


def inspect_saved_policy(checkpoint):
    """Validate retained inputs independently of today's execution backend."""
    record = policy_record(checkpoint)
    checkpoint = Path(record['checkpoint'])
    run = checkpoint.parent.parent
    protocol = read_json(run/'protocol.json')
    if (protocol.get('schema') not in ('whip_single_scenario_v1', 'whip_stage1_v1')
            or protocol['experiment']['experiment']['mode'] != 'training'
            or record['seed'] not in protocol['experiment']['experiment']['seeds']):
        raise ValueError('Unsupported training protocol or seed')
    paths = dict(checkpoint=checkpoint, metrics=checkpoint.with_suffix('.json'),
                 diagnostics=checkpoint.with_suffix('.npz'), protocol=run/'protocol.json',
                 scene=run/'scene.toml', model=run/'model.json', residual=run/'residual.pt')
    hashes = {name: digest(path) for name, path in paths.items()}
    for name in ('scene', 'model', 'residual'):
        if hashes[name] != protocol['sources'][name]['sha256']:
            raise ValueError(f'Saved {name} hash does not match the training protocol')
    verify_artifacts(run, paths.values())
    record.update(run=str(run), protocol=protocol, hashes=hashes,
                  paths={key: str(path) for key, path in paths.items()})
    return record


def inspect_policy(checkpoint):
    """Re-evaluation still requires compatible current mathematical source."""
    from .resume import source_compatible
    record = inspect_saved_policy(checkpoint)
    protocol = record['protocol']
    orchestration = {'run_training.py', 'tasks/whip/experiment.py',
                     'tasks/whip/worker.py', 'tasks/whip/resume.py', 'tasks/whip/evaluation.py'}
    for name, expected in protocol['code_sha256'].items():
        name = relative_record_name(name)
        if name in orchestration or name.startswith('training/ui/'):
            continue
        if not source_compatible(name, expected, digest(ROOT/name), protocol):
            raise ValueError(f'Training implementation changed: {name}')
    return record


def saved_motion(checkpoint):
    """Read a retained Stage-1 motion; no Torch, physics or reward execution."""
    from simulation.configuration import load_configuration
    record = inspect_saved_policy(checkpoint)
    path = Path(record['checkpoint'])
    motion = path.with_name(path.stem+'_motion.npz')
    if not motion.is_file():
        raise ValueError('No saved motion for this policy. Open an existing replay or use Evaluate on a compatible computer.')
    verify_artifacts(Path(record['run']), [motion])
    with np.load(motion, allow_pickle=False) as data:
        arrays = {key: data[key] for key in data.files}
    with np.load(record['paths']['diagnostics'], allow_pickle=False) as data:
        if not np.array_equal(arrays['actions'], data['actions'][0]):
            raise ValueError('Saved motion action does not match the evaluation record')
        indices = data['event_indices'][0].tolist()
        reasons = [label for label, key in (('command', 'command_valid'),
                   ('numerical', 'numerical_valid'), ('envelope', 'envelope_valid'))
                   if key in data and not bool(data[key][0])]
        if not bool(data['valid'][0]) and not reasons:
            reasons.append('score')
    raw = record['protocol']['experiment']
    if record['protocol']['schema'] != 'whip_stage1_v1':
        raise ValueError('Use Open replay for this historical evaluation format')
    scene = load_configuration(record['paths']['scene'])
    initial = scene.initial_state.position_world_m
    summary = dict(schema='whip_policy_evaluation_v1', source_run=record['run'],
        seed=record['seed'], update=record['update'], metrics=record['metrics'],
        saved_only=True, rejection_reasons=reasons,
        playable=bool(arrays['state_valid'].all()), target_position_world_m=None,
        stage1=raw['request'], event_indices=indices)
    if 'workspace' in raw:
        summary['workspace'] = dict(raw['workspace'], center_xy_world_m=initial[:2].tolist())
    if 'strike_surface' in raw:
        surface, request = raw['strike_surface'], raw['request']
        # Presentation of the saved vertical region; no event search/reward.
        normal = [math.cos(request['heading_rad']), math.sin(request['heading_rad']), 0.]
        summary['strike_surface'] = dict(
            center_world_m=[float(initial[0])+surface['distance_m']*normal[0],
                            float(initial[1])+surface['distance_m']*normal[1], request['release_height_world_m']],
            normal_world=normal, width_m=surface['width_m'], height_m=surface['height_m'],
            maximum_entry_angle_deg=surface['maximum_entry_angle_deg'],
            minimum_forward_speed_mps=request['tip_speed_mps'])
        summary['event_fraction'] = record['metrics'].get('crossing_fraction_mean')
    return summary, arrays, Path(record['paths']['scene'])


def saved_replay(folder):
    """A completed standalone replay is self-contained after relocation."""
    folder = Path(folder).resolve()
    paths = [folder/name for name in ('summary.json', 'rollout.npz', 'scene.toml')]
    verify_artifacts(folder, paths, required=True)
    summary = read_json(paths[0])
    if summary.get('schema') != 'whip_policy_evaluation_v1':
        raise ValueError('Unsupported saved replay')
    summary['saved_only'] = True
    with np.load(paths[1], allow_pickle=False) as data:
        arrays = {key: data[key] for key in data.files}
    return summary, arrays, paths[2]


def evaluate(checkpoint, output, expected_hashes, progress=print):
    """Use the production batch-one evaluation path and retain its full motion."""
    record = inspect_policy(checkpoint)
    if record['hashes'] != expected_hashes:
        raise ValueError('Selected evaluation inputs changed; refresh the selection')
    progress('Loading saved model and policy…', flush=True)
    import torch
    from simulation.configuration import load_configuration
    from simulation.dynamics.coupled import CoupledDynamics
    from training.configuration import LearnerConfiguration
    from training.episodic import EpisodicLearner
    from .configuration import TrainingConfiguration
    from .experiment import SingleScenarioTask, statistics, write_json
    from .reward import WhipReturnConfiguration
    from .scoring import WhipScoreConfiguration
    from .workspace import CylinderWorkspace
    from .reach import ReachConfiguration

    protocol = record['protocol']
    raw = protocol['experiment']
    stage1 = protocol['schema'] == 'whip_stage1_v1'
    device = raw['experiment']['device']
    if protocol['versions'] != dict(torch=torch.__version__, numpy=np.__version__):
        raise ValueError('Evaluation requires the saved Torch and NumPy versions')
    if device == 'cuda' and (not torch.cuda.is_available()
                            or torch.cuda.get_device_name() != protocol['gpu']):
        raise ValueError('The saved CUDA device is unavailable; no CPU fallback is applied')
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    try:
        scene = load_configuration(record['paths']['scene'])
        model_config = read_json(record['paths']['model'])
        model_config['residual']['checkpoint'] = record['paths']['residual']
        trajectory = TrainingConfiguration(**raw['trajectory'])
        if stage1:
            from training.cem import CEMConfiguration, DiagonalCEM
            from .stage1_configuration import Stage1Configuration
            from .stage1_experiment import Stage1Task, statistics as stage1_statistics
            learner = DiagonalCEM.load(checkpoint, device=device, expected_metadata=protocol)
            expected = CEMConfiguration(**raw['cem'])
            if learner.seed != record['seed']:
                raise ValueError('CEM seed does not match its saved evaluation')
        else:
            learner = EpisodicLearner.load(checkpoint, device=device, expected_metadata=protocol)
            expected = LearnerConfiguration(context_size=3, action_size=trajectory.action_size,
                                            seed=record['seed'], **raw['learner'])
            learner.policy.eval()
        if learner.updates != record['update'] or learner.configuration != expected:
            raise ValueError('Policy update/configuration does not match its saved evaluation')
        progress('Preparing one simulation episode…', flush=True)
        model = CoupledDynamics(model_config, device=device, batched=True)
        workspace = CylinderWorkspace(**raw['workspace']) if 'workspace' in raw else None
        reach = ReachConfiguration(**raw['reach']) if 'reach' in raw else None
        if stage1:
            task = Stage1Task(model, scene, Stage1Configuration.from_mapping(raw), 1)
        else:
            task = SingleScenarioTask(model, scene, trajectory, WhipScoreConfiguration(**raw['reward']),
                                      WhipReturnConfiguration(**raw['return']),
                                      raw['experiment']['target_position_world_m'], 1, workspace=workspace, reach=reach)
        progress('Evaluating at 100 Hz…', flush=True)
        episode, result = task.evaluate(learner.act() if stage1 else learner.act(task.contexts))
        metrics = stage1_statistics(episode, result) if stage1 else statistics(episode, result)
        rollout = episode.rollout
        arrays = {name: getattr(rollout, name)[0].detach().cpu().numpy() for name in (
            'drone_positions', 'drone_velocities', 'drone_rotations', 'drone_angular_velocities',
            'cable_positions', 'cable_velocities', 'state_valid')}
        arrays.update(time_s=rollout.time_s.cpu().numpy(), commands=episode.commands[0].cpu().numpy(),
                      command_times_s=episode.command_times_s.cpu().numpy(),
                      command_events=episode.command_events[0].cpu().numpy(),
                      cable_marker_indices=rollout.cable_marker_indices.cpu().numpy())
        with np.load(record['paths']['diagnostics'], allow_pickle=False) as saved:
            action_equal = np.array_equal(episode.actions.cpu().numpy(), saved['actions'])
        command_valid = episode.command_valid[0] if stage1 else episode.command_valid[0] & episode.context_valid[0]
        score_valid = result.score.valid[0] if stage1 else result.score_valid[0]
        reasons = [label for label, flag in (
            ('command', command_valid),
            ('numerical', result.numerical_valid[0]), ('envelope', result.envelope_valid[0]),
            ('score', score_valid)) if not bool(flag)]
        # Verify the pinned inputs again before publishing a completed replay.
        if {name: digest(path) for name, path in record['paths'].items()} != expected_hashes:
            raise ValueError('An evaluation input changed during the episode')
        with (output/'rollout.npz').open('xb') as stream:
            np.savez_compressed(stream, **arrays)
        shutil.copy2(record['paths']['scene'], output/'scene.toml')
        if bool(result.valid[0]):
            episode.trajectory(0).save(output/'commands.csv')
        summary = dict(schema='whip_policy_evaluation_v1', checkpoint=str(checkpoint),
            source_run=record['run'], seed=record['seed'], update=record['update'],
            target_position_world_m=None if stage1 else raw['experiment']['target_position_world_m'],
            target_width_m=None if stage1 else raw['reward']['target_width_m'], metrics=metrics,
            saved_metrics=record['metrics'], saved_action_equal=action_equal,
            metrics_equal=metrics == record['metrics'],
            return_difference=metrics['return_mean']-record['metrics']['return_mean'],
            rejection_reasons=reasons,
            playable=bool(command_valid & rollout.valid[0]),
            inputs={name: dict(path=record['paths'][name], sha256=value)
                    for name, value in record['hashes'].items()},
            evaluation_source_sha256=digest(__file__),
            experiment_source_sha256=digest(ROOT/'tasks/whip/experiment.py'),
            training_episodes=0, optimizer_updates=0)
        if stage1:
            summary['stage1'] = raw['request']
            summary['stage1_speed_mode'] = raw['matching'].get('speed_mode', 'match')
            summary['event_indices'] = result.score.event_indices[0].cpu().tolist()
            if task.reward.scorer.surface is not None:
                cfg = task.reward.scorer.configuration
                summary['strike_surface'] = cfg.strike_surface.description(scene.initial_state.position_world_m, cfg.request)
                summary['event_fraction'] = metrics['crossing_fraction_mean']
        if workspace is not None:
            summary['workspace'] = workspace.description(scene.initial_state.position_world_m)
            summary['workspace_source_sha256'] = digest(ROOT/'tasks/whip/workspace.py')
            summary['workspace_satisfied'] = bool(workspace.contains(
                rollout.drone_positions, task.environment.start_position).all())
            if not summary['workspace_satisfied']:
                summary['rejection_reasons'].append('workspace')
        if reach is not None:
            summary['reach'] = raw['reach']
            summary['reach_initial_position_world_m'] = scene.initial_state.position_world_m.tolist()
            summary['reach_source_sha256'] = digest(ROOT/'tasks/whip/reach.py')
        write_json(output/'summary.json', summary)
        write_json(output/'artifacts.json', {p.name: digest(p) for p in output.iterdir() if p.is_file()})
        progress('Evaluation complete.', flush=True)
        return summary
    except Exception as exc:
        write_json(output/'failure.json', dict(error=str(exc)))
        raise


def main():
    try:
        if len(sys.argv) != 4:
            raise ValueError('Open run_evaluation.py to select and evaluate a policy')
        evaluate(Path(sys.argv[1]), Path(sys.argv[2]), json.loads(sys.argv[3]))
    except Exception:
        traceback.print_exc()
        return 1
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
