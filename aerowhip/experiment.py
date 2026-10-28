"""Experiment files: one JSON file per experiment, parsed into typed sections.

    model, scene   paths to the shared physical model and initial scene
    run            compute and episode budget          (training.configuration.RunConfig)
    environment    horizon, control and validity limits (envs.configuration.EnvironmentConfig)
    task           reward definition selected by name  (tasks registry)
    ppo            learner settings                    (training.configuration.PPOConfig)

A run freezes a snapshot: the experiment dict, the resolved model and the scene text.
"""
from copy import deepcopy
from dataclasses import dataclass
import hashlib
import json

from aerowhip.envs.configuration import EnvironmentConfig
from aerowhip.paths import ROOT, project_path
from aerowhip.physics.configuration import load_configuration
from aerowhip.physics.model_configuration import load_model_configuration
from aerowhip.tasks import task_config
from aerowhip.learning.configuration import PPOConfig, RunConfig

DEFAULT = ROOT/'configs/experiments/strike_x_v1.json'
SECTIONS = {'model', 'scene', 'run', 'environment', 'task', 'ppo'}
RUN_SCHEMA = 'aerowhip_run_v1'


@dataclass(frozen=True)
class Experiment:
    model: str
    scene: str
    run: RunConfig
    environment: EnvironmentConfig
    task: object
    ppo: PPOConfig

    @classmethod
    def from_dict(cls, raw):
        if not isinstance(raw, dict) or raw.keys() != SECTIONS:
            found = set(raw) if isinstance(raw, dict) else set()
            raise ValueError(f'Experiment sections must be {sorted(SECTIONS)}; '
                             f'missing={sorted(SECTIONS-found)}, unknown={sorted(found-SECTIONS)}')
        if not all(isinstance(raw[k], str) and raw[k] for k in ('model', 'scene')):
            raise ValueError('model and scene must be file paths')
        return cls(raw['model'], raw['scene'], RunConfig.from_dict(raw['run']),
                   EnvironmentConfig.from_dict(raw['environment']), task_config(raw['task']),
                   PPOConfig.from_dict(raw['ppo']))


def check_model(experiment, model):
    """Model-dependent limits: the tasks' ten-link mesh and positive thrust. The command delay may take
    any value: the native runtime applies it exactly (the environment's 100 Hz record of commands in
    flight rounds it up)."""
    if model['simulation']['segments_per_marker_interval'] != [1]*10:
        raise ValueError('The tasks require the ten-link mesh (one segment per marker interval)')
    mass = model['drone']['mass_kg']
    trim = -(mass+model['cable']['total_mass_kg'])*model['simulation']['gravity_world_m_s2'][2]/mass
    if experiment.environment.acceleration_limit_mps2[2] >= trim:
        raise ValueError('Downward acceleration limit must keep desired vertical force positive')
    if model['controller']['command_delay_s'] < 0:
        raise ValueError('The command delay must not be negative')


def load(source=DEFAULT):
    """Experiment file path or dict -> (Experiment, snapshot dict, scene)."""
    raw = deepcopy(source) if isinstance(source, dict) else json.loads(
        project_path(source).read_text(encoding='utf-8-sig'))
    experiment = Experiment.from_dict(raw)
    model, sources = load_model_configuration(experiment.model, with_sources=True)
    check_model(experiment, model)
    scene_text = project_path(experiment.scene).read_text(encoding='utf-8-sig')
    scene = load_configuration(project_path(experiment.scene))
    snapshot = dict(schema=RUN_SCHEMA, experiment=raw, model=model, model_sources=sources, scene_text=scene_text)
    return experiment, snapshot, scene


def from_snapshot(snapshot, scene):
    """Typed experiment of a frozen run, checked against its own saved model."""
    if snapshot.get('schema') != RUN_SCHEMA:
        raise ValueError('Run snapshot predates the current experiment format')
    experiment = Experiment.from_dict(snapshot['experiment'])
    check_model(experiment, snapshot['model'])
    return experiment


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, allow_nan=False).encode()).hexdigest()


def identity(snapshot):
    """Run identity; only the total episode budget may change on an exact resume."""
    snapshot = deepcopy(snapshot)
    snapshot.pop('identity', None)
    snapshot['experiment']['run'].pop('training_episodes')
    return digest(snapshot)
