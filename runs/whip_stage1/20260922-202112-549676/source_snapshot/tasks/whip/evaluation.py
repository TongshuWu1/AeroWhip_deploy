"""Read saved policies and explicitly evaluate one episode; never train."""

import hashlib
import json
import math
from pathlib import Path
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


def inspect_policy(checkpoint):
    """Read-only provenance check; no Torch import or CUDA initialization."""
    from .resume import source_compatible
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
    manifest_path = run/'artifacts.json'
    if manifest_path.exists():
        manifest = {Path(k).as_posix(): v for k, v in read_json(manifest_path).items()}
        for name, path in paths.items():
            if manifest.get(path.relative_to(run).as_posix()) != hashes[name]:
                raise ValueError(f'Saved artifact hash mismatch: {path.name}')
    orchestration = {'run_training.py', 'tasks/whip/experiment.py',
                     'tasks/whip/worker.py', 'tasks/whip/resume.py', 'tasks/whip/evaluation.py'}
    for name, expected in protocol['code_sha256'].items():
        name = Path(name).as_posix()
        if name in orchestration or name.startswith('training/ui/'):
            continue
        if not source_compatible(name, expected, digest(ROOT/name), protocol):
            raise ValueError(f'Training implementation changed: {name}')
    record.update(run=str(run), protocol=protocol, hashes=hashes,
                  paths={key: str(path) for key, path in paths.items()})
    return record


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
