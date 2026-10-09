"""Compose the canonical controller–motor–plant setup, without inheritance.

Historical artifacts remain readable as data but cannot select retired models.
"""
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import tomllib

from project_paths import ROOT, project_path


DEFAULT_MODEL = ROOT/'configs/simulation.json'
DEFAULT_FIT_MODEL = DEFAULT_MODEL
PHYSICAL_SECTIONS = ('drone', 'attachment', 'cable', 'simulation')


def _read(value, role, sources, root=ROOT):
    if isinstance(value, dict):
        return deepcopy(value)
    if not isinstance(value, (str, Path)):
        raise ValueError(f'{role} must be a configuration path or mapping')
    path = project_path(value)  # Validate native/foreign absolute path semantics.
    relative = Path(str(value).replace('\\', '/')).expanduser()
    if not relative.is_absolute():
        path = project_path(project_path(root)/relative)
    data = path.read_bytes()
    sources[role] = {'path': str(path), 'sha256': hashlib.sha256(data).hexdigest()}
    text = data.decode('utf-8-sig')
    result = tomllib.loads(text) if path.suffix.lower() == '.toml' else json.loads(text)
    if not isinstance(result, dict):
        raise ValueError(f'{role} configuration must be a mapping')
    return result


def verify_configuration_sources(sources):
    for role, identity in sources.items():
        if hashlib.sha256(Path(identity['path']).read_bytes()).hexdigest() != identity['sha256']:
            raise RuntimeError(f'{role} configuration changed during use')


def configuration_digest(config):
    """Identity of resolved settings, not a hash of one component file."""
    return hashlib.sha256(json.dumps(config, sort_keys=True, separators=(',', ':'),
                                     allow_nan=False).encode('utf-8')).hexdigest()


def load_model_configuration(source, *, with_sources=False, root=ROOT):
    sources = {}
    value = _read(source, 'model', sources, root)
    if 'physics' in value:
        if set(value) != {'physics', 'controller', 'actuator', 'residual'}:
            raise ValueError('Setup requires physics, controller, actuator and residual')
        physics = _read(value['physics'], 'physics', sources, root)
        if set(physics) != set(PHYSICAL_SECTIONS):
            raise ValueError('Physics contains only drone, attachment, cable and simulation settings')
        controller = _read(value['controller'], 'controller', sources, root)
        result = {**physics, 'controller': controller,
                  'actuator': _read(value['actuator'], 'actuator', sources, root),
                  'residual': _read(value['residual'], 'residual_config', sources, root)}
    else:
        # Saved experiments are complete snapshots, never overlaid by live defaults.
        result = value
    if not all(isinstance(result.get(key), dict) for key in (*PHYSICAL_SECTIONS, 'controller')):
        raise ValueError('Incomplete physical/controller configuration')
    if 'residual' in result and not isinstance(result['residual'], dict):
        raise ValueError('Residual settings must be a mapping')
    if 'command_delay_s' not in result['controller']:
        raise ValueError('Controller settings must retain an explicit command_delay_s')
    if not isinstance(result.get('actuator'), dict) or 'controller_period_s' not in result['controller']:
        raise ValueError('Historical model is not an active setup: use the geometric controller and motor configuration')
    if result["cable"].get("constitutive_model") != "der_material_kelvin_voigt_v1":
        raise ValueError("Historical model is not the material-frame cable setup; no implicit migration")
    from controllers.full_state_controller.geometric_controller import load_gains
    load_gains(result['controller'])
    residual = result.get('residual', {})
    if residual.get('schema') != 'structured_aerodynamic_residual_v1' or residual.get('status') not in ('unfitted', 'fitted'):
        raise ValueError('Expected structured aerodynamic residual schema; historical models need explicit reconstruction')
    required = {'schema','status'} | ({'checkpoint','sha256'} if residual['status'] == 'fitted' else set())
    if set(residual) != required:
        raise ValueError('Residual settings must contain schema/status and, only when fitted, checkpoint/sha256')
    verify_configuration_sources(sources)
    return (result, sources) if with_sources else result
