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
    """Exact hashes plus narrowly reviewed compatibility revisions."""
    if actual == expected:
        return True
    name = Path(name).as_posix()
    # Exact reviewed execution revisions only. Retained full-motion checks:
    # runs/physics_efficiency/20260922 and runs/rollout_efficiency/20260922.
    # Model/configuration checks and unknown-source rejection remain intact.
    execution_revisions = {
        'simulation/dynamics/coupled.py': (
            '0558a72266d47cae9826c60c91fa2a1a53b40e3486f49d62cb6f884ab977d363',
            'f6fd664b1fe47cbf9c2d671c651868efc30ebf3d01ca3b3443244752e09430a3'),
        'simulation/batched_rollout.py': (
            'e1172536887263bafac2d219fee32c332daa403d5e6301a152d114d10384602f',
            'cfeecf2305bbcb7490068df061b504b8bc84b9f5e787ece99a887c6184dd97c4'),
    }
    if (protocol.get('schema') in ('whip_single_scenario_v1', 'whip_stage1_v1')
            and execution_revisions.get(name) == (expected, actual)):
        return True
    # Bitwise-equal forward/reward execution checks, with unchanged physics:
    # runs/rollout_performance/20260923. Only these exact revisions qualify.
    block_revisions = {
        'simulation/batched_rollout.py': (
            '746f7a9155a2054774b00790bf56309211f54c0b58a8f289a321253513584326',
            '47e24574e928b9c78dab2a7cd4945aaa0240fb4f1dbbf6359358630c08d2f22b'),
        'tasks/whip/stage1.py': (
            '96734e1b96cbf1f9ba295b9ed42892814f45ef1108cba4692496f3276a0d0a89',
            '47592ca995e5772532c83382c67b5d52ba55a9fa4428343386c3689903bb6c59'),
    }
    if (protocol.get('schema') in ('whip_single_scenario_v1', 'whip_stage1_v1')
            and block_revisions.get(name) == (expected, actual)):
        return True
    reviewed = {
        'tasks/whip/stage1.py': (
            '88d555ab8ec7948f9d3a37a5d1b9eaf6bc065450a472d4450dd1059bf696625d',
            '863edda4b3568801204c72bd44d8f25efb86ca31b9b55451a2f9060e7d256bf0'),
        'tasks/whip/stage1_configuration.py': (
            '7336a70e3692501a2e997e08533e8731d90dfba6d6f26e03daebc3e039e8b740',
            '4e360ceceb606fb3ecd4fa4eec1968357f3c12d6ca11ba7b5ba1a58a41c83671'),
    }
    if protocol.get('schema') != 'whip_stage1_v1':
        return False
    raw = protocol['experiment']
    directed_revision = {
        'tasks/whip/stage1.py': (
            '863edda4b3568801204c72bd44d8f25efb86ca31b9b55451a2f9060e7d256bf0',
            '96734e1b96cbf1f9ba295b9ed42892814f45ef1108cba4692496f3276a0d0a89'),
        'tasks/whip/stage1_configuration.py': (
            '4e360ceceb606fb3ecd4fa4eec1968357f3c12d6ca11ba7b5ba1a58a41c83671',
            '354ec37f33cf1bc7b5a0db235b645435ffbe2924e8cb8f83d2a02558eb48781a'),
        'tasks/whip/stage1_experiment.py': (
            'e9189d1378aa3d97db07929a9a2ca9a5c4bad78419230ad4681c88ebfbf7dc90',
            '3f496f740930f224e60ea8448d390df063e38579fedff7858a226d464498f0d7'),
    }
    # Retained numerical checks: runs/directed_stage1_checks/20260922.
    # Only configurations without a surface use the unchanged legacy branch.
    pair = directed_revision.get(name)
    if 'strike_surface' not in raw and pair is not None and actual == pair[1]:
        if expected == pair[0]:
            return True
        actual = pair[0]  # Permit only the earlier, already-reviewed exact chain.
    return (raw['matching'].get('speed_mode', 'match') == 'match'
            and reviewed.get(name) == (expected, actual))


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
