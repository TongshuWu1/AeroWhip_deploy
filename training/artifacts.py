"""Portable run snapshots, durable checkpoints and append-only live events."""
from datetime import datetime
import hashlib
import json
import math
from pathlib import Path
import numpy as np
import torch

from fitting.artifacts import atomic_save
from project_paths import ROOT, project_path, project_reference
from experiment import Experiment, identity, load

RUNS = ROOT/'runs/ppo'
CHECKPOINT_SCHEMA = 'aerowhip_ppo_checkpoint_v1'


def clean(value):
    if isinstance(value, dict): return {k:clean(v) for k,v in value.items()}
    if isinstance(value, (list,tuple)): return [clean(v) for v in value]
    if isinstance(value, float) and not math.isfinite(value): return None
    return value


def save_json(path, value):
    payload = json.dumps(clean(value),indent=2,allow_nan=False)
    atomic_save(path,lambda p:p.write_text(payload,encoding='utf-8'))


def read_json(path):
    return json.loads(project_path(path).read_text(encoding='utf-8-sig'))


def execution_hashes():
    paths = []
    for directory in ('simulation','controllers','envs','tasks','training','native'):
        for suffix in ('*.py','*.cpp','*.h','*.hpp','*.inc','*.pyd','*.so'):
            paths.extend((ROOT/directory).rglob(suffix))
    paths.append(ROOT/'experiment.py')
    return {p.relative_to(ROOT).as_posix():hashlib.sha256(p.read_bytes()).hexdigest()
            for p in sorted(set(paths)) if not any(x in p.parts for x in ('build','__pycache__'))}


def create_run(raw, resume=None, *, root=None):
    """Freeze a new run folder from an experiment dict.

    A fresh run validates and snapshots raw. An exact resume reuses the checkpoint's
    snapshot and takes only run.training_episodes (the extended budget) from raw.
    """
    parent = None
    if resume:
        parent = project_path(resume)
        saved = torch.load(parent,map_location='cpu',weights_only=True)
        if saved.get('schema') != CHECKPOINT_SCHEMA:
            raise ValueError('Resume requires a current learner checkpoint; start a new run.')
        snapshot = saved['snapshot']
        if snapshot['execution_hashes'] != execution_hashes():
            raise ValueError('Implementation changed since this checkpoint; exact continuation is not supported')
        snapshot['experiment']['run']['training_episodes'] = raw['run']['training_episodes']
        Experiment.from_dict(snapshot['experiment'])
        if raw['run']['training_episodes'] <= saved['episodes']:
            raise ValueError('The total training budget must exceed the saved episodes')
    else:
        _,snapshot,_ = load(raw)
        snapshot['execution_hashes'] = execution_hashes()
    snapshot['identity'] = identity(snapshot)
    root = project_path(RUNS if root is None else root)
    path = root/datetime.now().strftime('%Y%m%d-%H%M%S-%f')
    path.mkdir(parents=True,exist_ok=False)
    save_json(path/'configuration.json',snapshot)
    (path/'scene.toml').write_text(snapshot['scene_text'],encoding='utf-8')
    save_json(path/'request.json',dict(resume=project_reference(parent) if parent else None))
    return path


class EventLog:
    def __init__(self, path): self.path = project_path(path)

    def __call__(self, event):
        event = clean(dict(event,wall_time=datetime.now().isoformat(timespec='seconds')))
        line = json.dumps(event,allow_nan=False)
        with (self.path/'events.jsonl').open('a',encoding='utf-8') as stream:
            stream.write(line+'\n'); stream.flush()
        print(line,flush=True)
        try: save_json(self.path/'progress.json',event)
        except PermissionError as exc:
            print('Progress display is locked; durable event was saved: '+str(exc),flush=True)


def save_checkpoint(path, value):
    atomic_save(path,lambda p:torch.save(value,p))


def checkpoint_replay(arrays):
    """Numeric tensors and Unicode metadata, safe for weights_only loading."""
    if arrays is None:
        return None
    return {key: value.tolist() if value.dtype.kind == 'U'
            else torch.as_tensor(value).clone() for key, value in arrays.items()}


def replay_from_checkpoint(saved):
    """Restore historical numeric tensors and scalar/array Unicode metadata."""
    def is_text(value):
        return isinstance(value, str) or (isinstance(value, list) and all(is_text(v) for v in value))

    return {key: np.asarray(value, dtype=str) if is_text(value) else value.cpu().numpy()
            for key, value in saved.items()}


def events(path):
    source = project_path(Path(path)/'events.jsonl')
    if not source.exists(): return []
    result = []
    for line in source.read_text(encoding='utf-8').splitlines():
        try: result.append(json.loads(line))
        except json.JSONDecodeError: pass  # Writer may be appending the final line.
    return result
