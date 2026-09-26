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
    # Automatic preset dependencies leave complete saved configurations and
    # task execution unchanged. Only reviewed revisions with explicit saved
    # bounds qualify; new settings still fail the configuration comparison.
    # Evidence: runs/configuration_dependencies/20260924.
    dependency_revisions = {
        'tasks/whip/stage1_configuration.py': (
            '10843be0376872f398e944e12155e72a9865495e34ecf893c16d93f3e0c5e61a',
            '5bb6f17b57ba9241bec710799d9eb247db0f46c8f2a7824c677461d71ac0de6a'),
        'tasks/whip/stage1_experiment.py': (
            'ca4e672d48e6b92976fca6ef15356dda6d36f6d1b9dbc8941c9e535c016ddc40',
            '5fe39d094c3f4903f572e92eb8ea5689d06cdca83e923c9e2c14b87aa5f01d57'),
    }
    raw = protocol.get('experiment', {})
    if (protocol.get('schema') == 'whip_stage1_v1'
            and {'event_start_s', 'event_end_s'} <= raw.get('mechanics', {}).keys()
            and {'offset_lower_m', 'offset_upper_m'} <= raw.get('trajectory', {}).keys()
            and {'drone_lower_world_m', 'drone_upper_world_m'} <= raw.get('return', {}).keys()):
        pair = dependency_revisions.get(name)
        if pair is not None and actual == pair[1]:
            if expected == pair[0]:
                return True
            actual = pair[0]
    # Optional strike bonus: exact legacy execution when the saved weight is
    # absent/zero. Changed-reward Resume still fails the configuration check.
    # Saved-motion/independent-reference checks: runs/strike_reward/20260924.
    strike_revisions = {
        'tasks/whip/stage1.py': (
            'e5cd1c705ca4dce9fd731cfac3a5ade4b4ac0a53c42e0b11358d1eb0ba3f1a9d',
            '839c9c6bfc45289b9bb3b257fcede1586a1b2c732980cc4883dcc017dca69e0e'),
        'tasks/whip/stage1_configuration.py': (
            '20b27644826996684e8df5b756f51b5aa4b8e750b4d68f56b1ea0318936511d1',
            '10843be0376872f398e944e12155e72a9865495e34ecf893c16d93f3e0c5e61a'),
        'tasks/whip/stage1_experiment.py': (
            'aeff9a5183c2d51e020fcab568b313bbd4e8194c0fff7b697508d96ea3b115a7',
            'ca4e672d48e6b92976fca6ef15356dda6d36f6d1b9dbc8941c9e535c016ddc40'),
    }
    raw = protocol.get('experiment', {})
    if (protocol.get('schema') == 'whip_stage1_v1'
            and raw.get('strike_surface', {}).get('strike_bonus_weight', 0) == 0):
        pair = strike_revisions.get(name)
        if pair is not None and actual == pair[1]:
            if expected == pair[0]:
                return True
            actual = pair[0]  # Continue only the already-reviewed exact chain.
    # Exact legacy-branch bridge for cylinder mapping / valid-only CEM.
    # New experiment settings still cannot resume an old trajectory search.
    # Checks: runs/training_setup/20260924.
    setup_revisions = {
        'tasks/whip/configuration.py': (
            '5cd43d27ab17bb9c2d7b458b1c8d28d1ef2cf3e5a6bc9858a6bc33f1582bec1c',
            '313643c863275c260b37bd218bddbf4ac28ab7b778aed420dda51d70e2be169c'),
        'tasks/whip/environment.py': (
            '40967d22e63506aef44339389ef619ce6449853c46f46637696d00fb9c62ec3b',
            'b8ec15890a55e75e7bed45be9eac0395cf01fb202b7d134a5f601884d37c8263'),
        'tasks/whip/workspace.py': (
            '78847c85fb0251c5da7f9c85ddec2945fd492fda34f70515da1e79b4341d894d',
            'c145ddb5fd5f2f4f02ab03347ca5d76f7b7f1f29dc1cb89810f6d92564bcd475'),
        'tasks/whip/stage1_experiment.py': (
            '678c8988f767f4ede1c7880531a6eb9d04afce4bb7dde4ca5cb92599d0d3eafb',
            'aeff9a5183c2d51e020fcab568b313bbd4e8194c0fff7b697508d96ea3b115a7'),
        'training/cem.py': (
            'c3261b4b4264f86ea173e725394cdac54bd5f669b3a27013645e4596594dc8b6',
            '48239a9f724e1bf0e462bf5150b371155389e207d5d590f87ad810131b8b92a7'),
    }
    raw = protocol.get('experiment', {})
    if (protocol.get('schema') == 'whip_stage1_v1'
            and raw.get('trajectory', {}).get('control_mapping', 'box') == 'box'
            and not raw.get('cem', {}).get('valid_elites_only', False)):
        pair = setup_revisions.get(name)
        if pair is not None and actual == pair[1]:
            if expected == pair[0]:
                return True
            actual = pair[0]  # Only the already-reviewed exact earlier chain.
    # Optional acceleration rejection and independent effort scale. Only the
    # reviewed legacy branch qualifies; changed experiment settings still fail
    # inspect_resume. CPU checks: runs/command_acceleration/20260924.
    acceleration_revisions = {
        'tasks/whip/configuration.py': (
            '585304764429e83181485ff3401283a420d3cbf33f6d6faa680adc6a46434002',
            '5cd43d27ab17bb9c2d7b458b1c8d28d1ef2cf3e5a6bc9858a6bc33f1582bec1c'),
        'tasks/whip/environment.py': (
            'e2e88b20963dc024b24f096e3989be1fa134bbe755bedd9f1289cff6490ecc19',
            '40967d22e63506aef44339389ef619ce6449853c46f46637696d00fb9c62ec3b'),
        'tasks/whip/reward.py': (
            '0a90fe68cb2658e0d2ee0fc21e93c180149d72ac282e6d26aa285d0b1e6552f0',
            '80a98b828251105eae996692ce8e276e74db0334012781526c84f62bc6435010'),
        'tasks/whip/stage1.py': (
            '47592ca995e5772532c83382c67b5d52ba55a9fa4428343386c3689903bb6c59',
            'e5cd1c705ca4dce9fd731cfac3a5ade4b4ac0a53c42e0b11358d1eb0ba3f1a9d'),
        'tasks/whip/stage1_configuration.py': (
            '354ec37f33cf1bc7b5a0db235b645435ffbe2924e8cb8f83d2a02558eb48781a',
            '20b27644826996684e8df5b756f51b5aa4b8e750b4d68f56b1ea0318936511d1'),
    }
    raw = protocol.get('experiment', {})
    if (protocol.get('schema') == 'whip_stage1_v1'
            and raw.get('trajectory', {}).get('max_acceleration_mps2') == 15.
            and 'acceleration_cost_scale_mps2' not in raw.get('return', {})
            and acceleration_revisions.get(name) == (expected, actual)):
        return True
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
