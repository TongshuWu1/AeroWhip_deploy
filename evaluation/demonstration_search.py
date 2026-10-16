"""Search bounded drone commands under the approved Stage1 experiment reward."""
import argparse
import json
import os
from time import perf_counter

import numpy as np
import torch

from project_paths import ROOT, project_path, project_reference
from training.configuration import load
from training.artifacts import execution_hashes, save_json
from tasks.whip.sac_environment import Stage1Environment
from tasks.whip.deterministic_evaluation import DeterministicEvaluationEnvironment
from viewer.replay import StateReplay
from .motion_search import candidates, program, rollout, file_hash
from .motion_refinement import retain_batch


def contained(rows):
    return sorted((r for r in rows if r['valid_full_horizon']
                   and not r['workspace']['boundary_breached']),
                  key=lambda r:r['raw_return'], reverse=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--settings', default='configs/learning/whip_stage1_ppo_demo_outreach_v7.json')
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    settings, snapshot, cfg, scene = load(args.settings)
    if cfg.matching.reward_mode != 'directed_outreach_speed_whip_v1':
        raise ValueError('Requires the approved speed/outreach experiment')
    directory = project_path(args.output)
    directory.mkdir(parents=True, exist_ok=False)
    protected = json.loads((ROOT/'tmp/stage1-demo-speed-20261010/protected-inputs.json').read_text())
    if any(file_hash(ROOT/p) != h for p,h in protected['protected_inputs'].items()):
        raise ValueError('Earlier artifacts changed')
    before = execution_hashes()
    physical = {k:v for k,v in before.items() if k.startswith(('simulation/','controllers/','native/'))}
    if physical != protected['physical_hashes']:
        raise ValueError('Physical execution changed')
    source = ROOT/'runs/motion_search/20261010-forward-pullback-pareto-v2'
    previous = json.loads((source/'attempts.json').read_text())
    # Retained counterfactual leader and complementary geometry/speed examples.
    parent_ids = [149,368,206,262,92,59,39]
    parents = [previous[i] for i in parent_ids]
    save_json(directory/'configuration.json', dict(protocol='stage1_demo_search_v1',
        snapshot=snapshot, execution_hashes=before, protected_inputs=protected,
        parent_source=project_reference(source), parent_ids=parent_ids,
        attempts=384, seed=2, selection='Maximum approved return among full-horizon valid contained attempts',
        search_source_sha256=file_hash(__file__), model_promotion=False))
    (directory/'scene.toml').write_text(snapshot['scene_text'])
    torch.set_num_threads(min(8, os.cpu_count() or 1))
    rng = np.random.default_rng(2)
    env = Stage1Environment(settings, snapshot, cfg, scene)
    rate, limits = settings['policy_rate_hz'], np.asarray(settings['velocity_limit_mps'])
    rows, started = [], perf_counter()
    for batch in range(3):
        cs = candidates(rng, 128, env.decisions, rate, limits, parents)
        # Retain the previous best as an integrated control in the 384 attempts.
        if batch == 0:
            cs[0] = dict(previous[149]['parameters'])
        for i in range(4, len(cs), 4):
            trial = dict(cs[i], family='prepared_oblique')
            for name in ('prep_z_mps','drive_z_mps','pull_z_mps'):
                trial[name] = float(rng.uniform(-.6,.6))
            try:
                program(trial, env.decisions, rate, limits)
                cs[i] = trial
            except ValueError:
                pass
        commands = torch.from_numpy(np.stack([program(c,env.decisions,rate,limits)[0] for c in cs])).to(env.device)
        retain_batch(directory, env, commands, cs, batch, rows)
        for row in rows:
            row['phase'] = 'approved_reward_demonstration_search'
        save_json(directory/'attempts.json', rows)
        leaders = contained(rows)
        if not leaders:
            raise RuntimeError('No contained demonstration found')
        parents = leaders[:12]+[previous[149],previous[206]]
        print(json.dumps(dict(phase='search_batch', attempts=len(rows),
            contained=len(leaders), best=leaders[0]['raw_return'], elapsed_s=perf_counter()-started)), flush=True)
    winner = contained(rows)[0]
    native = DeterministicEvaluationEnvironment(dict(settings,device='cpu'), snapshot, cfg, scene)
    command, boundaries = program(winner['parameters'], native.decisions, rate, limits)
    returns, valid, score, _ = rollout(native, torch.from_numpy(command[None]))
    replay = native.replay_arrays()
    StateReplay(replay)
    np.savez_compressed(directory/'teacher-replay.npz', **replay)
    # The actor sees only causal prefixes of these actual environment frames.
    np.savez_compressed(directory/'demonstration.npz', frames=native.frames[0].cpu().numpy(),
        normalized_actions=command.astype(np.float32),
        policy_indices=np.arange(native.decisions,dtype=np.int64)*native.stride,
        actor_columns=np.asarray(native.actor_columns,dtype=np.int64),
        goal_columns=np.asarray(native.policy_encoder_options['goal_columns'],dtype=np.int64),
        velocity_limit_mps=limits, policy_rate_hz=np.array(rate),
        frame_size=np.array(native.frame_size), privileged_size=np.array(native.critic_state_size))
    with np.load(directory/winner['artifact']) as saved:
        errors = {k:float(np.max(np.abs(replay[k]-saved[k][winner['lane']]))) for k in
                  ('drone_positions','drone_velocities','drone_rotations','cable_positions','cable_velocities')}
    workspace = {k:v[0].item() for k,v in native.workspace_metrics().items()}
    preserved = all(file_hash(ROOT/p)==h for p,h in protected['protected_inputs'].items())
    unchanged = execution_hashes()==before
    if not bool(valid[0]) or workspace['boundary_breached'] or max(errors.values())>1e-8 or not preserved or not unchanged:
        raise RuntimeError('Demonstration physical/provenance verification failed')
    save_json(directory/'assessment.json', dict(attempts=len(rows), valid=sum(r['valid_full_horizon'] for r in rows),
        contained=len(contained(rows)), winner=winner, native_return=float(returns[0]),
        native_components={k:float(v[0]) for k,v in score.components.items()}, workspace=workspace,
        batch_native_max_absolute_errors=errors, replay_validation='passed',
        demonstration_sha256=file_hash(directory/'demonstration.npz'),
        protected_file_count=len(protected['protected_inputs']), protected_files_preserved=preserved,
        execution_sources_unchanged=unchanged, elapsed_s=perf_counter()-started,
        limitations='Nominal simulated start and adaptive search; reward leader is not a physical whip success or real-flight demonstration'))
    print(json.dumps(dict(phase='search_complete', directory=project_reference(directory),
        winner=winner['id'], reward=float(returns[0]), components={k:float(v[0]) for k,v in score.components.items()})),flush=True)


if __name__ == '__main__':
    main()
