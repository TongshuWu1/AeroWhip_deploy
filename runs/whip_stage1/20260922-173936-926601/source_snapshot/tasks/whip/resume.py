"""Read-only continuation checks for retained single-scenario experiments."""

from copy import deepcopy
import hashlib
import json
from pathlib import Path
import re


ROOT = Path(__file__).resolve().parents[2]


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def source_compatible(name, expected, actual, protocol):
    """Exact hashes, plus the checked additive Stage-1 minimum-speed revision.

    This exception applies only to old exact-matching runs and these exact
    old/new file pairs. Any other edit still invalidates evaluation/resume.
    """
    if actual == expected:
        return True
    reviewed = {
        'tasks/whip/stage1.py': (
            '88d555ab8ec7948f9d3a37a5d1b9eaf6bc065450a472d4450dd1059bf696625d',
            '863edda4b3568801204c72bd44d8f25efb86ca31b9b55451a2f9060e7d256bf0'),
        'tasks/whip/stage1_configuration.py': (
            '7336a70e3692501a2e997e08533e8731d90dfba6d6f26e03daebc3e039e8b740',
            '4e360ceceb606fb3ecd4fa4eec1968357f3c12d6ca11ba7b5ba1a58a41c83671'),
    }
    return (protocol.get('schema') == 'whip_stage1_v1'
            and protocol['experiment']['matching'].get('speed_mode', 'match') == 'match'
            and reviewed.get(Path(name).as_posix()) == (expected, actual))


def inspect_resume(checkpoint, configuration, *, root=ROOT):
    """Validate provenance without loading Torch, a network or the simulator.

    Resume one seed from a finalized run, with only its total update budget
    changed. UI/orchestration source may evolve; dynamics, task math, action
    generation and learner implementation must match the retained source.
    """
    checkpoint = Path(checkpoint).resolve()
    match = re.fullmatch(r'policy_(\d+)\.pt', checkpoint.name)
    seed_match = re.fullmatch(r'seed_(\d+)', checkpoint.parent.name)
    if match is None or seed_match is None:
        raise ValueError('Select a policy checkpoint from a saved training seed folder')
    update, seed = int(match[1]), int(seed_match[1])
    run = checkpoint.parent.parent
    manifest = json.loads((run/'artifacts.json').read_text(encoding='utf-8'))

    def verified(path):
        relative = path.relative_to(run).as_posix()
        expected = {Path(k).as_posix(): v for k, v in manifest.items()}.get(relative)
        if expected is None or digest(path) != expected:
            raise ValueError(f'Saved artifact hash mismatch: {relative}')
        return path

    protocol_path = verified(run/'protocol.json')
    protocol = json.loads(protocol_path.read_text(encoding='utf-8'))
    verified(checkpoint)
    if protocol.get('schema') not in ('whip_single_scenario_v1', 'whip_stage1_v1'):
        raise ValueError('Unsupported experiment protocol')
    previous = deepcopy(protocol['experiment'])
    exp = configuration['experiment']
    if (exp['mode'] != 'training' or previous['experiment']['mode'] != 'training'
            or exp['seeds'] != [seed] or seed not in previous['experiment']['seeds']):
        raise ValueError('Resume requires training mode and exactly the selected checkpoint seed')
    if not 0 <= update < exp['updates'] or update > previous['experiment']['updates']:
        raise ValueError(f'Total updates must exceed checkpoint update {update}')
    previous['experiment'].update(updates=exp['updates'], seeds=[seed])
    if previous != configuration:
        raise ValueError('Resume must preserve all settings except total updates and seed selection')

    root = Path(root)
    for name in ('scene', 'model'):
        if digest(root/exp[name]) != protocol['sources'][name]['sha256']:
            raise ValueError(f'Resume {name} differs from the saved experiment')
    model = json.loads((root/exp['model']).read_text(encoding='utf-8'))
    if digest(root/model['residual']['checkpoint']) != protocol['sources']['residual']['sha256']:
        raise ValueError('Resume residual differs from the saved experiment')
    orchestration = {'run_training.py', 'tasks/whip/experiment.py', 'tasks/whip/worker.py',
                     'tasks/whip/resume.py', 'tasks/whip/evaluation.py'}
    for name, expected in protocol['code_sha256'].items():
        name = Path(name).as_posix()
        if name in orchestration or name.startswith('training/ui/'):
            continue
        if not source_compatible(name, expected, digest(root/name), protocol):
            raise ValueError(f'Resume implementation changed: {name}')

    # Retain the curve prefix as data, never as new training episodes. A new
    # continuation will not need the parent directory to display that prefix.
    updates, evaluations = {}, {}
    history_path = run/'resume_history.json'
    if history_path.exists():
        history = json.loads(verified(history_path).read_text(encoding='utf-8'))
        if history['seed'] != seed or history['update'] > update:
            raise ValueError('Checkpoint predates the retained continuation history')
        updates.update({r['update']: r for r in history['updates']})
        evaluations.update({r['update']: r['metrics'] for r in history['evaluations']})
    for path in sorted(checkpoint.parent.glob('update_*.json')):
        number = int(path.stem[7:])
        if number <= update:
            row = json.loads(verified(path).read_text(encoding='utf-8'))
            if row['update'] != number or row['seed'] != seed or row['samples']['episodes'] != exp['batch_size']:
                raise ValueError('Saved update record does not match its checkpoint seed/batch')
            updates[number] = row
    for path in sorted(checkpoint.parent.glob('policy_*.json')):
        number = int(path.stem[7:])
        if number <= update:
            evaluations[number] = json.loads(verified(path).read_text(encoding='utf-8'))
    if set(updates) != set(range(1, update+1)):
        raise ValueError('The selected checkpoint is missing its training history')
    return dict(checkpoint=str(checkpoint), sha256=digest(checkpoint), source_run=str(run),
                source_protocol_sha256=digest(protocol_path), seed=seed, update=update, protocol=protocol,
                history=dict(seed=seed, update=update, updates=[updates[k] for k in sorted(updates)],
                             evaluations=[dict(update=k, metrics=evaluations[k]) for k in sorted(evaluations)]))
