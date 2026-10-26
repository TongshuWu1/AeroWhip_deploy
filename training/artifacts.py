"""Portable run snapshots, durable checkpoints and append-only live events."""
from datetime import datetime
import hashlib
import json
import math
from pathlib import Path
import numpy as np
import torch

from fitting.artifacts import atomic_save
from project_paths import ROOT, project_path, project_reference, contained_child
from training.configuration import digest, load, checkpoint_schema, algorithm, episode_budget, validate, training_stage, require_launchable, run_schema, action_limit_key

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
    return json.loads(project_path(path).read_text(encoding='utf-8-sig'))


def execution_hashes():
    paths = []
    for directory in ('simulation','controllers','tasks','training','native'):
        for suffix in ('*.py','*.cpp','*.h','*.hpp','*.inc','*.pyd','*.so'):
            paths.extend((ROOT/directory).rglob(suffix))
    # These pure kinematic helpers are used by the versioned shape-release task.
    paths.extend(ROOT/'evaluation'/name for name in (
        'stage1_candidate.py', 'whip_measurements.py', 'whip_output_decomposition.py'))
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
            old.get('schema') not in ('ppo_stage1_velocity_v1', 'ppo_stage1_acceleration_direct_v1') or
            old.get('schema') != run_schema(old['settings']) or
            old.get('identity') != saved.get('identity') or identity(old) != saved.get('identity')):
        raise ValueError('Actor initialization requires an identity-consistent PPO checkpoint')
    if algorithm(snapshot['settings']) != 'ppo':
        raise ValueError('PPO actor initialization cannot initialize SAC')
    reviewed_shape_handoff = (old['task']['matching'].get('reward_mode') == 'directed_shape_release_v1'
        and training_stage(old['settings']) == 1 and training_stage(snapshot['settings']) == 2
        and snapshot['task']['matching'].get('reward_mode') == 'directed_shape_delivery_v1')
    reviewed_flexible_handoff = (old['task']['matching'].get('reward_mode') == 'flexible_excitation_v1'
        and training_stage(old['settings']) == 1 and training_stage(snapshot['settings']) == 2
        and old['settings'].get('stage1_return_transform') == 'negative_net_rational_v1'
        and snapshot['task']['matching'].get('reward_mode') in ('flexible_delivery_v1','first_hit_flexible_delivery_v2')
        and old['settings'].get('target_observation') == snapshot['settings'].get('target_observation'))
    if (old['task']['matching'].get('reward_mode') in ('continuous_horizontal_whip_v1', 'continuous_horizontal_whip_soft_workspace_v2', 'continuous_horizontal_whip_distance_v3','continuous_horizontal_whip_full_horizon_v4', 'continuous_directional_whip_full_horizon_v5', 'continuous_directional_whip_four_second_v6', 'directed_discovery_bounded_v1', 'directed_shape_release_v1', 'directed_unified_fold_v1', 'flexible_excitation_v1', 'directed_fold_formation_v1','directed_fold_release_aligned_v1')
            and training_stage(snapshot['settings']) != 1 and not (reviewed_shape_handoff or reviewed_flexible_handoff)):
        raise ValueError('Bounded Stage1 discovery requires motion review and an approved '
                         'later-stage handoff; only registered shape or flexible delivery handoffs are allowed')
    if old['model'] != snapshot['model'] or old['scene_text'] != snapshot['scene_text']:
        raise ValueError('Actor transfer requires the same physical model and initial scene')
    current = execution_hashes()
    previous = old['execution_hashes']
    for key in previous.keys() | current.keys():
        if key.split('/')[0] in ('simulation', 'controllers', 'native') and previous.get(key) != current.get(key):
            raise ValueError('Physical execution changed since the actor checkpoint: '+key)
    for key in ('action_interface', 'policy_rate_hz', action_limit_key(snapshot['settings']), 'hidden_units'):
        if old['settings'].get(key) != snapshot['settings'].get(key):
            raise ValueError('Actor control interface changed: '+key)
    if (snapshot['settings']['ppo'].get('transfer_entropy_clock', 'inherit') == 'inherit'
            and old['settings']['ppo']['entropy'] != snapshot['settings']['ppo']['entropy']):
        raise ValueError('Actor transfer preserves the existing exploration schedule')
    if snapshot['task']['matching'].get('reward_mode')=='continuous_horizontal_whip_soft_workspace_v2':
        if (old['task']['matching'].get('reward_mode')!='continuous_horizontal_whip_v1'
                or training_stage(old['settings'])!=1 or training_stage(snapshot['settings'])!=1
                or snapshot['settings']['training_episodes']!=saved['agent']['entropy_episode_budget']-saved['episodes']
                or snapshot['settings']['parallel_environments']!=old['settings']['parallel_environments']):
            raise ValueError('Soft workspace handoff must preserve the stopped continuous Stage1 campaign budget and lanes')
    if snapshot['task']['matching'].get('reward_mode')=='continuous_horizontal_whip_distance_v3':
        spent=saved['agent']['entropy_episode_offset']+saved['episodes']
        if (old['task']['matching'].get('reward_mode')!='continuous_horizontal_whip_soft_workspace_v2'
                or training_stage(old['settings'])!=1 or training_stage(snapshot['settings'])!=1
                or snapshot['settings']['training_episodes']!=saved['agent']['entropy_episode_budget']-spent
                or snapshot['settings']['parallel_environments']!=old['settings']['parallel_environments']):
            raise ValueError('Distance-cost handoff requires stopped v2 actor, same lanes and remaining original budget')
    if snapshot['task']['matching'].get('reward_mode')=='continuous_horizontal_whip_full_horizon_v4':
        spent=saved['agent']['entropy_episode_offset']+saved['episodes']
        if (old['task']['matching'].get('reward_mode')!='continuous_horizontal_whip_distance_v3'
                or training_stage(old['settings'])!=1 or training_stage(snapshot['settings'])!=1
                or snapshot['settings']['training_episodes']!=saved['agent']['entropy_episode_budget']-spent
                or snapshot['settings']['parallel_environments']!=old['settings']['parallel_environments']):
            raise ValueError('Full-horizon handoff requires v3 actor, same lanes and remaining original budget')
    actor = saved['agent'].get('actor', {})
    if not actor or any(not bool(torch.isfinite(v).all()) for v in actor.values()):
        raise ValueError('Cannot initialize from nonfinite actor weights')


def validate_fold_handoff(saved,snapshot,checkpoint_sha256,source_path=None):
    """Require review tied to the exact warm-up actor and remaining total budget."""
    if snapshot['task']['matching'].get('reward_mode')!='directed_fold_release_aligned_v1':return
    if saved['snapshot']['task']['matching'].get('reward_mode')!='directed_fold_formation_v1':
        raise ValueError('Fold release must use the reviewed formation actor')
    for key in ('movement_penalty', 'workspace_penalty'):
        if saved['snapshot']['settings'].get(key) != snapshot['settings'].get(key):
            raise ValueError('Fold handoff must preserve the reviewed movement/workspace contract')
    if source_path is None:raise ValueError('Fold handoff requires its completed warm-up run')
    summary=read_json(project_path(source_path).parent/'summary.json')
    if (summary.get('phase') not in ('stopped','completed')
            or summary.get('episodes')!=saved['episodes'] or summary.get('round')!=saved['round']):
        raise ValueError('Fold handoff uses the final stopped warm-up actor; earlier best cannot rewind the campaign budget or entropy clock')
    review=read_json(snapshot['settings']['fold_curriculum']['motion_review'])
    if (review.get('schema')!='stage1_fold_formation_review_v1' or review.get('approved') is not True
            or review.get('numeric_gate_passed') is not True or review.get('motion_review_passed') is not True
            or review.get('checkpoint_sha256')!=checkpoint_sha256
            or review.get('formation_episodes')!=saved['episodes']
            or not isinstance(review.get('evidence'),list) or not review['evidence']):
        raise ValueError('Exact formation checkpoint requires retained numeric AND motion approval')
    if (not 0<saved['episodes']<=3840
            or snapshot['settings']['training_episodes']!=15360-saved['episodes']
            or snapshot['settings']['ppo'].get('transfer_entropy_clock')!='inherit'
            or saved['agent'].get('entropy_episode_budget')!=15360):
        raise ValueError('Fold release must preserve the120-collection total and original entropy clock')
    snapshot['fold_formation_review']=dict(path=snapshot['settings']['fold_curriculum']['motion_review'],
        sha256=hashlib.sha256(project_path(snapshot['settings']['fold_curriculum']['motion_review']).read_bytes()).hexdigest(),
        **review)


def create_run(settings, resume=None, *, task=None, root=None, initialize_actor=None):
    if resume and initialize_actor:
        raise ValueError('Select exact resume OR actor-only initialization, not both')
    parent = None
    initialization_payload = None
    if resume:
        parent = project_path(resume)
        saved = torch.load(parent,map_location='cpu',weights_only=True)
        if settings.get('action_interface') != saved['snapshot']['settings'].get('action_interface'):
            raise ValueError('Exact resume cannot change the policy action interface')
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
        fold_mode=snapshot['task']['matching'].get('reward_mode')
        if fold_mode in ('continuous_directional_whip_full_horizon_v5', 'continuous_directional_whip_four_second_v6') and initialize_actor:
            raise ValueError('Directed +X restart requires a fresh actor and critic')
        if fold_mode in ('continuous_horizontal_whip_soft_workspace_v2','continuous_horizontal_whip_distance_v3','continuous_horizontal_whip_full_horizon_v4') and not initialize_actor:
            raise ValueError('Soft workspace continuation requires its stopped Stage1 actor')
        if fold_mode=='retained_fold_delivery_v1' and initialize_actor:
            raise ValueError('Retained fold-delivery restart requires a fresh random actor and critic')
        if fold_mode in ('continuous_horizontal_whip_v1','horizontal_fold_outreach_v2','directed_fold_formation_v1','directed_unified_fold_v1','flexible_excitation_v1') and initialize_actor:
            raise ValueError('Approved fresh Stage1 restart requires a random actor, not initialization')
        if fold_mode=='directed_fold_release_aligned_v1' and not initialize_actor:
            raise ValueError('Fold release requires an explicitly reviewed formation actor')
        if training_stage(settings) == 2 and not initialize_actor and fold_mode!='retained_fold_delivery_v1':
            raise ValueError('New Stage2 requires an explicit Stage1 or Stage2 actor checkpoint')
        if settings.get('stage3_role') == 'precision_v1' and not initialize_actor:
            raise ValueError('New precision Stage3 requires an explicit Stage2 actor or precision Stage3 actor checkpoint')
        if settings.get('stage4_role') == 'learned_recovery_v1' and not initialize_actor:
            raise ValueError('New recovery Stage4 requires an explicit precision Stage3 actor checkpoint')
        if initialize_actor:
            import io
            source = project_path(initialize_actor)
            initialization_payload = source.read_bytes()
            saved_actor = torch.load(io.BytesIO(initialization_payload),map_location='cpu',weights_only=True)
            if training_stage(settings) == 2 and training_stage(saved_actor['snapshot']['settings']) not in (1, 2):
                raise ValueError('Stage2 must initialize from a Stage1 or Stage2 actor')
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
            validate_fold_handoff(saved_actor,snapshot,hashlib.sha256(initialization_payload).hexdigest(),source)
            snapshot['policy_initialization'] = dict(
                source_checkpoint=project_reference(source), source_sha256=hashlib.sha256(initialization_payload).hexdigest(),
                source_identity=saved_actor['identity'], source_round=saved_actor['round'],
                source_episodes=saved_actor['episodes'], source_steps=saved_actor['steps'],
                transfer='actor_only_v1', value_and_optimizers='fresh',
                exploration_clock=('stage-local transitions; source offset intentionally reset'
                    if settings['ppo'].get('transfer_entropy_clock', 'inherit') == 'stage_local'
                    else 'source elapsed offset plus new-run transitions'))
    require_launchable(snapshot['settings'])
    snapshot['identity'] = identity(snapshot)
    root = project_path(run_root(snapshot['settings']) if root is None else root)
    # identity must not recursively include itself in subsequent comparisons.
    path = root/datetime.now().strftime('%Y%m%d-%H%M%S-%f')
    path.mkdir(parents=True,exist_ok=False)
    save_json(path/'configuration.json',snapshot)
    (path/'scene.toml').write_text(snapshot['scene_text'],encoding='utf-8')
    if initialization_payload is not None:
        atomic_save(path/'initial-actor.pt', lambda p:p.write_bytes(initialization_payload))
    save_json(path/'request.json',dict(resume=project_reference(parent) if parent else None,
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
                payload = contained_child(parent.parent,name).read_bytes()
                atomic_save(path/name, lambda p, payload=payload:p.write_bytes(payload))
        save_json(path/'curriculum_review.json', review)
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
