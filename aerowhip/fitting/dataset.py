"""Explicit recording selection for refitting; no disk search or error ranking."""
import json

from aerowhip.paths import project_path, project_reference
from aerowhip.takes.take_data import digest


def validate_split(split):
    if not isinstance(split,dict) or set(split) != {'training','comparison'}:
        raise ValueError('Split requires exactly training and comparison lists')
    result = {}
    identities = set()
    for role,rows in split.items():
        if not isinstance(rows,(list,tuple)) or not rows:
            raise ValueError('Each split role requires at least one recording')
        result[role] = []
        for row in rows:
            if not isinstance(row,(list,tuple)) or len(row)!=3:
                raise ValueError('Split rows must be [maneuver, batch_folder, take_stem]')
            if not all(isinstance(v,str) and v.strip() for v in row):
                raise ValueError('Split fields must be nonempty strings')
            for value in row[1:]:
                if any(c in value for c in ('/','\\',':')) or value in ('.','..'):
                    raise ValueError('Batch and take names must be simple local names, not paths')
            identity=tuple(row[1:])
            if identity in identities:
                raise ValueError('Duplicate recording or train/comparison overlap')
            identities.add(identity)
            result[role].append(tuple(row))
    if {r[1] for r in result['training']} & {r[1] for r in result['comparison']}:
        raise ValueError('Recording batches must not cross training/comparison roles')
    return result


def load_split(path, default):
    if path is None:
        return validate_split(default), {}
    path=project_path(path)
    before=digest(path)
    result=validate_split(json.loads(path.read_text(encoding='utf-8-sig')))
    if digest(path)!=before:
        raise RuntimeError('Dataset split changed while reading')
    return result, {'dataset_split': {'path':project_reference(path),'sha256':before}}
