"""Discover PPO/SAC results and prepare independent evaluation artifacts."""
from dataclasses import dataclass
from datetime import datetime
import hashlib
import io
import os
from pathlib import Path
import pickle
import shlex
import subprocess
import tomllib

import numpy as np
import torch

from fitting.artifacts import atomic_save
from project_paths import ROOT, project_path, project_reference, contained_child
from model_catalog import packaged_stages
from training.artifacts import events, execution_hashes, read_json, save_json
from training.configuration import digest

RUNS = ROOT/'runs/ppo_stage1'
RESULTS = ROOT/'runs/sac_evaluation'
PPO_RESULTS = ROOT/'runs/ppo_evaluation'
STAGE_ROOTS = (RUNS, ROOT/'runs/ppo_stage2', ROOT/'runs/ppo_stage3', ROOT/'runs/ppo_stage4', ROOT/'runs/sac_stage1', PPO_RESULTS, RESULTS)


def task_stage(snapshot):
    # PPO's outer schema is shared by both stages; folder names are not evidence.
    from training.configuration import training_stage
    return training_stage(snapshot['settings'])


def stage_label(snapshot):
    return f'{snapshot["settings"].get("algorithm", "sac").upper()} · Stage {task_stage(snapshot)}'


def scoring_label(snapshot):
    if snapshot['settings'].get('stage4_role') == 'learned_recovery_v1':
        return 'Precision whip + learned recovery (nominal, event-relative)'
    if snapshot['task'].get('matching', {}).get('reward_mode') == 'directed_precision_v1':
        return 'Precision +/-5cm (nominal, full horizon)'
    if snapshot['task'].get('matching', {}).get('reward_mode') == 'directed_delivery_v1':
        return 'Fixed broad-target delivery'
    return 'Strict-final target' if task_stage(snapshot) >= 2 else 'Stage 1 broad wall'


def discover_stages(stage=None, root=None):
    paths, warnings = [], []
    for directory in (STAGE_ROOTS if root is None else (root,)):
        found, notes = discover(directory)
        warnings.extend(notes)
        for path in found:
            if stage is None or task_stage(read_json(path/'configuration.json')) == stage:
                paths.append(path)
    if root is None:
        paths.extend(entry['directory'] for entry in packaged_stages()
                     if stage is None or entry['stage']['stage'] == stage)
    return sorted(set(paths), key=lambda p: p.name, reverse=True), warnings


def active_workers():
    """Inspect actual Python commands, not stale locks or GUI process names."""
    if os.name == 'nt':
        script = ("Get-CimInstance Win32_Process | Where-Object { $_.Name -like 'python*' -and "
                  "$_.CommandLine -match '\\b(training\\.worker|evaluation\\.(worker|policy_test))\\b' } | "
                  "Select-Object ProcessId,CommandLine | ConvertTo-Json -Compress")
        result = subprocess.run(['powershell', '-NoProfile', '-Command', script],
            capture_output=True, text=True, check=True, timeout=15,
            creationflags=subprocess.CREATE_NO_WINDOW)
        import json
        rows = json.loads(result.stdout) if result.stdout.strip() else []
        rows = [rows] if isinstance(rows, dict) else rows
        return [r for r in rows if r['ProcessId'] != os.getpid()]
    result = subprocess.run(['ps', '-eo', 'pid=,args='], capture_output=True,
                            text=True, check=True, timeout=15)
    import re
    rows = []
    for line in result.stdout.splitlines():
        parts = line.strip().split(None, 1)
        if len(parts) == 2 and int(parts[0]) != os.getpid() and re.search(
                r'\bpython[\w.]*\b.*\s-m\s+(training\.worker|evaluation\.(worker|policy_test))\b', parts[1]):
            rows.append(dict(ProcessId=int(parts[0]), CommandLine=parts[1]))
    return rows


def require_evaluation_idle(owned_directory=None):
    """Allow training alongside evaluation; reject another evaluation worker."""
    workers = []
    for row in active_workers():
        args = [a.strip('"') for a in shlex.split(row['CommandLine'], posix=False)]
        if '-m' in args and args.index('-m')+1 < len(args) and args[args.index('-m')+1] in ('evaluation.worker', 'evaluation.policy_test'):
            workers.append(row)
    if owned_directory is not None:
        def own_launcher(row):
            if row['ProcessId'] != os.getppid():
                return False
            args = [a.strip('"') for a in shlex.split(row['CommandLine'], posix=False)]
            try:
                return (args[args.index('-m')+1] == 'evaluation.worker' and
                        Path(args[args.index('--run')+1]).resolve() == Path(owned_directory).resolve())
            except (ValueError, IndexError):
                return False
        workers = [w for w in workers if not own_launcher(w)]
    if workers:
        raise ValueError('Fresh evaluation waits until other evaluation workers are idle '
                         f'(active PIDs: {", ".join(str(w["ProcessId"]) for w in workers)}). '
                         'View saved remains available; no worker was stopped.')
    for folder in ('policy_tests', 'context_evaluation'):
        if project_path(ROOT/'runs'/folder/'worker.lock').exists():
            raise ValueError('Evaluation lock exists in runs/'+folder+'; inspect ownership before starting. No lock was removed.')


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def child(directory, filename):
    path = contained_child(directory, filename)
    if path.parent != project_path(directory):
        raise ValueError('Replay must be a file inside its selected run')
    return path


def discover(root):
    root = project_path(root)
    candidates = [root] if (root/'configuration.json').exists() else (
        sorted(root.iterdir(), reverse=True) if root.is_dir() else [])
    result, warnings = [], []
    for path in candidates:
        if not path.is_dir() or not (path/'configuration.json').exists():
            continue
        try:
            if read_json(path/'configuration.json').get('schema') in ('ppo_stage1_velocity_v1','sac_stage1_ctbr_v1','sac_stage1_ctbr_v2','sac_stage1_acceleration_v1','sac_stage1_velocity_v1','sac_stage1_velocity_filtered_v1','sac_stage1_velocity_v2'):
                result.append(path)
        except (OSError, ValueError) as exc:
            warnings.append(f'{path.name}: {exc}')
    return result, warnings


def load_checkpoint(path):
    # Read a single immutable byte sequence even if a live trainer replaces the file.
    payload = project_path(path).read_bytes()
    digest = hashlib.sha256(payload).hexdigest()
    saved = torch.load(io.BytesIO(payload), map_location='cpu', weights_only=True)
    if saved.get('schema') not in ('ppo_stage1_checkpoint_v1','sac_stage1_checkpoint_v1','sac_stage1_checkpoint_v2','sac_stage1_checkpoint_v3','sac_stage1_checkpoint_v4','sac_stage1_checkpoint_v5','sac_stage1_checkpoint_v6','sac_stage1_checkpoint_v7'):
        raise ValueError('Select a PPO/SAC Stage 1 or Stage 2 checkpoint, not a historical CEM checkpoint')
    snapshot = saved['snapshot']
    is_ppo = saved['schema'] == 'ppo_stage1_checkpoint_v1'
    if is_ppo != (snapshot.get('schema') == 'ppo_stage1_velocity_v1') or (
            is_ppo != (snapshot['settings'].get('algorithm', 'sac') == 'ppo')):
        raise ValueError('Checkpoint learner and snapshot schemas differ')
    if (snapshot.get('schema') not in ('ppo_stage1_velocity_v1','sac_stage1_ctbr_v1','sac_stage1_ctbr_v2','sac_stage1_acceleration_v1','sac_stage1_velocity_v1','sac_stage1_velocity_filtered_v1','sac_stage1_velocity_v2')
            or checkpoint_identity(snapshot) != saved['identity']
            or snapshot.get('identity') != saved['identity']):
        raise ValueError('Checkpoint configuration identity is inconsistent')
    return dict(snapshot=snapshot, round=saved['round'], actor=saved['agent']['actor'], sha256=digest,
                target_curriculum=saved.get('target_curriculum'))


def checkpoint_identity(snapshot):
    """Read-only archive compatibility; never migrate settings into the learner."""
    snapshot = dict(snapshot, settings=dict(snapshot['settings']))
    snapshot.pop('identity', None)
    settings = snapshot['settings']
    key = 'training_episodes' if 'training_episodes' in settings else 'collection_rounds'
    settings.pop(key)
    return digest(snapshot)


def changed_sources(snapshot, current=None):
    expected = snapshot['execution_hashes']
    current = execution_hashes() if current is None else current
    return sorted(k for k in expected.keys() | current.keys() if expected.get(k) != current.get(k))


def require_compatible(snapshot):
    changed = changed_sources(snapshot)
    if changed:
        raise ValueError('Fresh evaluation requires the saved implementation. Changed files: '
                         +', '.join(changed[:6])+'; saved replay is still available.')


def evaluator_hashes():
    return {p.name:sha256(p) for p in Path(__file__).parent.glob('*.py')}


def evaluation_contract(snapshot, mode):
    if mode not in ('exact_source', 'current_code'):
        raise ValueError('Unknown evaluation source mode')
    current = execution_hashes()
    changed = changed_sources(snapshot, current)
    if mode == 'exact_source':
        require_compatible(snapshot)
    else:
        # This opt-in permits task/learning revisions, NOT a different plant.
        physical = [k for k in changed if k.split('/')[0] in ('simulation', 'controllers', 'native')]
        if physical:
            raise ValueError('Current-code reevaluation requires unchanged physical execution: '+', '.join(physical))
        from training.configuration import validate
        validate(snapshot['settings'])  # No silent settings/schema migration.
        if snapshot['settings'].get('action_interface') != 'velocity_v1':
            raise ValueError('Current-code reevaluation supports the current velocity interface only; use saved replay')
    return dict(source_mode=mode, execution_hashes=current,
                source_execution_hashes=snapshot['execution_hashes'], changed_files=changed,
                evaluator_hashes=evaluator_hashes(), exact_source_reproduction=(mode == 'exact_source'))


def verify_contract(snapshot, request):
    mode = request.get('source_mode', 'exact_source')
    actual = evaluation_contract(snapshot, mode)
    for key in ('execution_hashes', 'source_execution_hashes', 'changed_files', 'evaluator_hashes',
                'exact_source_reproduction'):
        if key in request and request[key] != actual[key]:
            raise ValueError('Evaluation source contract changed after confirmation: '+key)
        if mode == 'current_code' and key not in request:
            raise ValueError('Current-code reevaluation requires frozen provenance: '+key)
    return actual


@dataclass
class Selection:
    label: str
    directory: Path
    snapshot: dict
    checkpoint: Path | None = None
    event: dict | None = None
    curriculum_event: dict | None = None

    @property
    def replay_path(self):
        name = (self.event or {}).get('replay')
        return child(self.directory, name) if name else None


def selections(directory):
    directory = project_path(directory)
    snapshot = read_json(directory/'configuration.json')
    all_rows = events(directory)
    rows = [e for e in all_rows if e.get('phase') == 'evaluation']
    current_rows = {e['round']:e for e in all_rows if e.get('phase') == 'target_curriculum_evaluation'}
    result, warnings = [], []
    for name in ('latest', 'best'):
        checkpoint = directory/(name+'.pt')
        if not checkpoint.exists():
            continue
        try:
            info = load_checkpoint(checkpoint)
            if info['snapshot']['identity'] != snapshot['identity']:
                raise ValueError('Checkpoint does not match the run configuration')
            event = next((e for e in reversed(rows) if e['round'] == info['round']), None)
            result.append(Selection(f'{name.title()} · Iteration {info["round"]}', directory,
                                    snapshot, checkpoint, event, current_rows.get(info['round'])))
        except (OSError, ValueError, RuntimeError, KeyError, EOFError, pickle.UnpicklingError) as exc:
            warnings.append(f'{checkpoint.name}: {exc}')
    for event in reversed(rows):
        if event.get('replay'):
            result.append(Selection(f'Saved · Iteration {event["round"]} · Return {event["evaluation_return"]:.5g}',
                                    directory, snapshot, event=event, curriculum_event=current_rows.get(event['round'])))
    # Independent evaluations can also be reopened with Folder.
    if not result and (directory/'summary.json').exists():
        event = read_json(directory/'summary.json')
        if event.get('replay'):
            result.append(Selection('Saved evaluation', directory, snapshot, event=event,
                                    curriculum_event=event.get('current_target_evaluation')))
    return result, warnings


def prepare(checkpoint, *, root=None, mode='exact_source', recovery_target=None):
    if recovery_target not in (None,'current','final'):
        raise ValueError('Recovery target must be explicitly current or final')
    if recovery_target is not None and mode != 'current_code':
        raise ValueError('Strike/return is a new protocol; select current-code evaluation')
    checkpoint = project_path(checkpoint)
    info = load_checkpoint(checkpoint)
    snapshot = info['snapshot']
    contract = evaluation_contract(snapshot, mode)
    require_evaluation_idle()
    if root is None:
        root = PPO_RESULTS if snapshot['settings'].get('algorithm', 'sac') == 'ppo' else RESULTS
    path = project_path(root)/datetime.now().strftime('%Y%m%d-%H%M%S-%f')
    path.mkdir(parents=True, exist_ok=False)
    save_json(path/'configuration.json', snapshot)
    atomic_save(path/'scene.toml', lambda p: p.write_text(snapshot['scene_text'], encoding='utf-8'))
    reference = None
    config = checkpoint.parent/'configuration.json'
    if config.exists() and read_json(config).get('identity') == snapshot['identity']:
        event = next((e for e in reversed(events(checkpoint.parent))
                      if e.get('phase') == 'evaluation' and e.get('round') == info['round']), None)
        if event and event.get('replay'):
            replay = child(checkpoint.parent, event['replay'])
            if replay.exists():
                reference = dict(path=project_reference(replay), sha256=sha256(replay), event=event)
    request = dict(checkpoint=project_reference(checkpoint), checkpoint_sha256=info['sha256'],
                   round=info['round'], reference=reference, **contract)
    if recovery_target is not None:
        from dataclasses import asdict
        from evaluation.recovery import RecoverySettings
        request.update(protocol='strike_scripted_return_v1', recovery_target=recovery_target,
                       recovery_settings=asdict(RecoverySettings()))
    save_json(path/'request.json', request)
    return path


def envelope_observation(arrays, snapshot):
    """Report sampled boundary evidence, not an inferred controller failure cause."""
    task = snapshot['task']
    w, r = task['workspace'], task['return']
    p, q, rotation = (arrays[k] for k in ('drone_positions', 'cable_positions', 'drone_rotations'))
    origin = np.asarray(tomllib.loads(snapshot['scene_text'])['initial_state']['position_world_m'])
    failures = {
        'cylinder radius': np.square(p[:, :2]-origin[:2]).sum(-1) > w['radius_m']**2,
        'lower height': p[:, 2] < w['height_lower_world_m'],
        'upper height': p[:, 2] > w['height_upper_world_m'],
        'tilt': rotation[:, 2, 2] < np.cos(np.deg2rad(r['maximum_tilt_deg'])),
        'cable floor': q[:, :, 2].min(-1) < r['minimum_cable_height_m'],
    }
    invalid = np.flatnonzero(np.stack(list(failures.values())).any(0))
    if not len(invalid):
        return 'No envelope violation in the saved samples; this alone does not establish episode validity.'
    i = int(invalid[0])
    reasons = ', '.join(name for name, mask in failures.items() if mask[i])
    return f'First sampled violation: {reasons} at {arrays["time_s"][i]:.2f} s.'


def compare_reference(arrays, event, reference):
    if reference is None:
        return dict(status='No matching saved evaluation for this checkpoint iteration')
    path = project_path(reference['path'])
    if not path.exists() or sha256(path) != reference['sha256']:
        return dict(status='Reference missing or changed; comparison not performed')
    with np.load(path, allow_pickle=False) as source:
        old = dict(source)
    errors, exact = {}, arrays is not None and old.keys() == arrays.keys()
    for key in old.keys() & (arrays or {}).keys():
        a, b = old[key], arrays[key]
        if a.shape != b.shape:
            errors[key] = dict(saved_shape=list(a.shape), fresh_shape=list(b.shape))
            exact = False
        else:
            same = np.array_equal(a, b)
            exact &= same
            errors[key] = (float(np.max(np.abs(a-b))) if a.dtype.kind == 'f' else bool(same))
    previous = reference['event']
    metrics_equal = all(event.get(k) == previous.get(k) for k in ('evaluation_return', 'valid', 'components'))
    return dict(status='Compared', arrays_exact=bool(exact), metrics_exact=metrics_equal, differences=errors,
                reference_sha256=reference['sha256'])
