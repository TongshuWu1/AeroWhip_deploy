"""Discover current PPO runs and verify checkpoints used for evaluation."""
import hashlib
import io
import os
from pathlib import Path
import re
import subprocess

import torch

from aerowhip.paths import project_path
from aerowhip.experiment import Experiment, RUN_SCHEMA, identity
from aerowhip.learning.artifacts import CHECKPOINT_SCHEMA, RUNS, execution_hashes, read_json


def discover_runs(root=RUNS):
    """Current-schema run folders, newest first; other folders are reported."""
    root = project_path(root)
    candidates = [root] if (root/'configuration.json').exists() else (
        sorted(root.iterdir(), reverse=True) if root.is_dir() else [])
    result, warnings = [], []
    for path in candidates:
        if not path.is_dir() or not (path/'configuration.json').exists():
            continue
        try:
            if read_json(path/'configuration.json').get('schema') == RUN_SCHEMA:
                result.append(path)
        except (OSError, ValueError, KeyError) as exc:
            warnings.append(f'{path.name}: {exc}')
    return result, warnings


def active_workers():
    """Inspect actual Python commands, not stale locks or GUI process names."""
    pattern = r'\bpython[\w.]*\b.*\s-m\s+(aerowhip\.learning\.worker|aerowhip\.evaluation\.policy_test)\b'
    if os.name == 'nt':
        script = ("Get-CimInstance Win32_Process | Where-Object { $_.Name -like 'python*' -and "
                  "$_.CommandLine -match '\\b(training\\.worker|evaluation\\.policy_test)\\b' } | "
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
    rows = []
    for line in result.stdout.splitlines():
        parts = line.strip().split(None, 1)
        if len(parts) == 2 and int(parts[0]) != os.getpid() and re.search(pattern, parts[1]):
            rows.append(dict(ProcessId=int(parts[0]), CommandLine=parts[1]))
    return rows


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def load_checkpoint(path):
    # Read a single immutable byte sequence even if a live trainer replaces the file.
    payload = project_path(path).read_bytes()
    digest_value = hashlib.sha256(payload).hexdigest()
    saved = torch.load(io.BytesIO(payload), map_location='cpu', weights_only=True)
    snapshot = saved.get('snapshot', {})
    if saved.get('schema') != CHECKPOINT_SCHEMA or snapshot.get('schema') != RUN_SCHEMA:
        raise ValueError('Select a current PPO checkpoint')
    environment = snapshot['experiment']['environment']
    interface = saved['agent'].get('interface', {})
    for key in ('acceleration_limit_mps2', 'policy_rate_hz'):
        if interface.get(key) != environment[key]:
            raise ValueError('Checkpoint control interface differs from its experiment: '+key)
    if identity(snapshot) != saved['identity'] or snapshot.get('identity') != saved['identity']:
        raise ValueError('Checkpoint configuration identity is inconsistent')
    return dict(snapshot=snapshot, round=saved['round'], actor=saved['agent']['actor'], sha256=digest_value)


def changed_sources(snapshot, current=None):
    expected = snapshot['execution_hashes']
    current = execution_hashes() if current is None else current
    return sorted(k for k in expected.keys() | current.keys() if expected.get(k) != current.get(k))


def evaluator_hashes():
    return {p.name:sha256(p) for p in Path(__file__).parent.glob('*.py')}


def evaluation_contract(snapshot, mode):
    if mode not in ('exact_source', 'current_code'):
        raise ValueError('Unknown evaluation source mode')
    current = execution_hashes()
    changed = changed_sources(snapshot, current)
    if mode == 'exact_source':
        if changed:
            raise ValueError('Fresh evaluation requires the saved implementation. Changed files: '
                             +', '.join(changed[:6])+'; saved replay is still available.')
    else:
        # This opt-in permits task/learning revisions, NOT a different plant.
        physical = [k for k in changed if k.split('/')[0] in ('simulation', 'controllers', 'native')]
        if physical:
            raise ValueError('Current-code reevaluation requires unchanged physical execution: '+', '.join(physical))
        Experiment.from_dict(snapshot['experiment'])  # No silent settings/schema migration.
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
