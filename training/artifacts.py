"""Portable run snapshots, durable checkpoints and append-only live events."""
from datetime import datetime
import hashlib
import json
import math
from pathlib import Path
import torch

from fitting.artifacts import atomic_save
from project_paths import ROOT
from training.configuration import digest, load, checkpoint_schema, algorithm, episode_budget, validate, training_stage, require_launchable

RUNS = ROOT/'runs/sac_stage1'
PPO_RUNS = ROOT/'runs/ppo_stage1'
PPO_STAGE2_RUNS = ROOT/'runs/ppo_stage2'
PPO_STAGE3_RUNS = ROOT/'runs/ppo_stage3'
PPO_STAGE4_RUNS = ROOT/'runs/ppo_stage4'


def run_root(settings):
    if training_stage(settings) == 4:
        return PPO_STAGE4_RUNS
    if training_stage(settings) == 3:
        return PPO_STAGE3_RUNS
    if training_stage(settings) == 2:
        return PPO_STAGE2_RUNS
    return PPO_RUNS if algorithm(settings) == 'ppo' else RUNS


def clean(value):
    if isinstance(value, dict): return {k:clean(v) for k,v in value.items()}
    if isinstance(value, (list,tuple)): return [clean(v) for v in value]
    if isinstance(value, float) and not math.isfinite(value): return None
    return value


def save_json(path, value):
    payload = json.dumps(clean(value),indent=2,allow_nan=False)
    atomic_save(path,lambda p:p.write_text(payload,encoding='utf-8'))


def read_json(path):
    return json.loads(Path(path).read_text(encoding='utf-8-sig'))


def execution_hashes():
    paths = []
    for directory in ('simulation','controllers','tasks','training','native'):
        for suffix in ('*.py','*.cpp','*.h','*.hpp','*.inc','*.pyd','*.so'):
            paths.extend((ROOT/directory).rglob(suffix))
    return {p.relative_to(ROOT).as_posix():hashlib.sha256(p.read_bytes()).hexdigest()
            for p in sorted(set(paths)) if not any(x in p.parts for x in ('build','__pycache__'))}


def identity(snapshot):
    # Only the total budget can change when resuming an experiment.
    snapshot = dict(snapshot,settings=dict(snapshot['settings']))
    snapshot.pop('identity',None)
    snapshot['settings'].pop('training_episodes')
    return digest(snapshot)


def validate_actor_source(saved, snapshot):
    """Allow explicit task transfer only, never weaken exact-resume identity."""
    old = saved.get('snapshot', {})
    if (saved.get('schema') != 'ppo_stage1_checkpoint_v1' or
            old.get('schema') != 'ppo_stage1_velocity_v1' or
            old.get('identity') != saved.get('identity') or identity(old) != saved.get('identity')):
        raise ValueError('Actor initialization requires an identity-consistent PPO checkpoint')
    if algorithm(snapshot['settings']) != 'ppo':
        raise ValueError('PPO actor initialization cannot initialize SAC')
    if old['model'] != snapshot['model'] or old['scene_text'] != snapshot['scene_text']:
        raise ValueError('Actor transfer requires the same physical model and initial scene')
    current = execution_hashes()
    for key, value in current.items():
        if key.split('/')[0] in ('simulation', 'controllers', 'native') and old['execution_hashes'].get(key) != value:
            raise ValueError('Physical execution changed since the actor checkpoint: '+key)
    for key in ('policy_rate_hz', 'action_interface', 'velocity_limit_mps', 'hidden_units'):
        if old['settings'][key] != snapshot['settings'][key]:
            raise ValueError('Actor control interface changed: '+key)
    if (snapshot['settings']['ppo'].get('transfer_entropy_clock', 'inherit') == 'inherit'
            and old['settings']['ppo']['entropy'] != snapshot['settings']['ppo']['entropy']):
        raise ValueError('Actor transfer preserves the existing exploration schedule')
    actor = saved['agent'].get('actor', {})
    if not actor or any(not bool(torch.isfinite(v).all()) for v in actor.values()):
        raise ValueError('Cannot initialize from nonfinite actor weights')


def create_run(settings, resume=None, *, task=None, root=None, initialize_actor=None):
    if resume and initialize_actor:
        raise ValueError('Select exact resume OR actor-only initialization, not both')
    parent = None
    initialization_payload = None
    if resume:
        parent = Path(resume).resolve()
        saved = torch.load(parent,map_location='cpu',weights_only=True)
        if saved.get('schema') != checkpoint_schema(saved['snapshot']['settings']):
            raise ValueError('Resume requires a current learner checkpoint. '
                             'Older checkpoints remain replayable; start a new run.')
        snapshot = saved['snapshot']
        if snapshot['execution_hashes'] != execution_hashes():
            raise ValueError('Implementation changed since this checkpoint; exact continuation is not supported')
        snapshot['settings']['training_episodes'] = settings['training_episodes']
        validate(snapshot['settings'])
        if episode_budget(snapshot['settings']) <= saved['episodes']:
            raise ValueError('The total training budget must exceed the saved episodes')
    else:
        require_launchable(settings)
        _,snapshot,_,scene = load(settings)
        if task is not None:
            from training.configuration import task_configuration
            task_configuration(task,settings,snapshot['model'],scene)
            snapshot['task'] = task
        snapshot['execution_hashes'] = execution_hashes()
        if training_stage(settings) == 2 and not initialize_actor:
            raise ValueError('New Stage 2 requires an explicit Stage 1 actor checkpoint')
        if settings.get('stage3_role') == 'precision_v1' and not initialize_actor:
            raise ValueError('New precision Stage3 requires an explicit Stage2 actor or precision Stage3 actor checkpoint')
        if settings.get('stage4_role') == 'learned_recovery_v1' and not initialize_actor:
            raise ValueError('New recovery Stage4 requires an explicit precision Stage3 actor checkpoint')
        if initialize_actor:
            import io
            source = Path(initialize_actor).resolve()
            initialization_payload = source.read_bytes()
            saved_actor = torch.load(io.BytesIO(initialization_payload),map_location='cpu',weights_only=True)
            if settings.get('stage3_role') == 'precision_v1':
                source_settings = saved_actor['snapshot']['settings']
                source_stage = training_stage(source_settings)
                if not (source_stage == 2 or
                        source_stage == 3 and source_settings.get('stage3_role') == 'precision_v1'):
                    raise ValueError('Precision Stage3 must initialize from Stage2 or a precision Stage3 actor')
            if settings.get('stage4_role') == 'learned_recovery_v1' and (
                    training_stage(saved_actor['snapshot']['settings']) != 3
                    or saved_actor['snapshot']['settings'].get('stage3_role') != 'precision_v1'):
                raise ValueError('Recovery Stage4 must initialize from a precision Stage3 actor')
            validate_actor_source(saved_actor,snapshot)
            snapshot['policy_initialization'] = dict(
                source_checkpoint=str(source), source_sha256=hashlib.sha256(initialization_payload).hexdigest(),
                source_identity=saved_actor['identity'], source_round=saved_actor['round'],
                source_episodes=saved_actor['episodes'], source_steps=saved_actor['steps'],
                transfer='actor_only_v1', value_and_optimizers='fresh',
                exploration_clock=('stage-local transitions; source offset intentionally reset'
                    if settings['ppo'].get('transfer_entropy_clock', 'inherit') == 'stage_local'
                    else 'source elapsed offset plus new-run transitions'))
    require_launchable(snapshot['settings'])
    snapshot['identity'] = identity(snapshot)
    root = run_root(snapshot['settings']) if root is None else Path(root)
    # identity must not recursively include itself in subsequent comparisons.
    path = root/datetime.now().strftime('%Y%m%d-%H%M%S-%f')
    path.mkdir(parents=True,exist_ok=False)
    save_json(path/'configuration.json',snapshot)
    (path/'scene.toml').write_text(snapshot['scene_text'],encoding='utf-8')
    if initialization_payload is not None:
        atomic_save(path/'initial-actor.pt', lambda p:p.write_bytes(initialization_payload))
    save_json(path/'request.json',dict(resume=str(parent) if parent else None,
        initialize_actor='initial-actor.pt' if initialization_payload is not None else None))
    if parent and saved.get('curriculum', {}).get('review_round') is not None:
        # A paused first-stage review must remain reviewable after Resume.
        review = read_json(parent.parent/'curriculum_review.json')
        if review['round'] != saved['curriculum']['review_round'] or review['identity'] != snapshot['identity']:
            raise ValueError('Pending review does not match the resumed checkpoint')
        for row in review['samples']:
            if row['replay']:
                name = row['replay']
                if Path(name).name != name: raise ValueError('Invalid review replay filename')
                payload = (parent.parent/name).read_bytes()
                atomic_save(path/name, lambda p, payload=payload:p.write_bytes(payload))
        save_json(path/'curriculum_review.json', review)
    return path


class EventLog:
    def __init__(self, path): self.path = path

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


def events(path):
    source = Path(path)/'events.jsonl'
    if not source.exists(): return []
    result = []
    for line in source.read_text(encoding='utf-8').splitlines():
        try: result.append(json.loads(line))
        except json.JSONDecodeError: pass  # Writer may be appending the final line.
    return result
