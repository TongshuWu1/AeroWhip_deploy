"""Compose the canonical controller–motor–plant setup, without inheritance.

Historical artifacts remain readable as data but cannot select retired models.
"""
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import tomllib

from aerowhip.paths import ROOT, project_path, project_reference


DEFAULT_MODEL = ROOT/'configs/simulation.json'
DEFAULT_FIT_MODEL = DEFAULT_MODEL
PHYSICAL_SECTIONS = ('drone', 'attachment', 'cable', 'simulation')


def _read(value, role, sources, root=ROOT):
    if isinstance(value, dict):
        return deepcopy(value)
    if not isinstance(value, (str, Path)):
        raise ValueError(f'{role} must be a configuration path or mapping')
    path = project_path(value)  # Validate native/foreign absolute path semantics.
    relative = Path(str(value).replace('\\', '/'))
    if not relative.is_absolute():
        path = project_path(project_path(root)/relative)
    data = path.read_bytes()
    sources[role] = {'path': project_reference(path), 'sha256': hashlib.sha256(data).hexdigest()}
    text = data.decode('utf-8-sig')
    result = tomllib.loads(text) if path.suffix.lower() == '.toml' else json.loads(text)
    if not isinstance(result, dict):
        raise ValueError(f'{role} configuration must be a mapping')
    return result


def verify_configuration_sources(sources):
    for role, identity in sources.items():
        if hashlib.sha256(project_path(identity['path']).read_bytes()).hexdigest() != identity['sha256']:
            raise RuntimeError(f'{role} configuration changed during use')


def configuration_digest(config):
    """Identity of resolved settings, not a hash of one component file."""
    return hashlib.sha256(json.dumps(config, sort_keys=True, separators=(',', ':'),
                                     allow_nan=False).encode('utf-8')).hexdigest()


def load_model_configuration(source, *, with_sources=False, root=ROOT):
    sources = {}
    value = _read(source, 'model', sources, root)
    if 'physics' in value:
        if set(value) != {'physics', 'controller', 'actuator'}:
            raise ValueError('Setup requires physics, controller and actuator')
        physics = _read(value['physics'], 'physics', sources, root)
        if set(physics) != set(PHYSICAL_SECTIONS):
            raise ValueError('Physics contains only drone, attachment, cable and simulation settings')
        controller = _read(value['controller'], 'controller', sources, root)
        result = {**physics, 'controller': controller,
                  'actuator': _read(value['actuator'], 'actuator', sources, root)}
    else:
        # Saved experiments are complete snapshots, never overlaid by live defaults. Older snapshots
        # carry the settings of a learned drag correction that no longer exists (never fitted): ignored.
        result = {k: v for k, v in value.items() if k != 'residual'}
    if not all(isinstance(result.get(key), dict) for key in (*PHYSICAL_SECTIONS, 'controller')):
        raise ValueError('Incomplete physical/controller configuration')
    if 'command_delay_s' not in result['controller']:
        raise ValueError('Controller settings must retain an explicit command_delay_s')
    if not isinstance(result.get('actuator'), dict) or 'controller_period_s' not in result['controller']:
        raise ValueError('Historical model is not an active setup: use the onboard controller and motor configuration')
    if result["cable"].get("constitutive_model") != "der_twist_free_kelvin_voigt_v2":
        raise ValueError("Historical model is not the twist-free cable setup; no implicit migration")
    verify_configuration_sources(sources)
    return (result, sources) if with_sources else result
