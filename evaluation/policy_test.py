"""One explicit stationary-target case, frozen actor, native plant, no learning."""
import argparse
from dataclasses import replace
from datetime import datetime
import os
from pathlib import Path
import platform
import subprocess
import sys
from time import perf_counter
import traceback

import numpy as np
import torch

from evaluation.records import load_checkpoint, evaluation_contract, verify_contract, task_stage
from fitting.artifacts import atomic_save
from project_paths import ROOT
from simulation.configuration import load_configuration
from simulation.state import quaternion_xyzw_to_rotation_matrix, rotation_matrix_to_quaternion_xyzw
from tasks.whip.deterministic_evaluation import DeterministicEvaluationEnvironment
from tasks.whip.target_sampling import configure_targets, target_surface
from training.artifacts import EventLog, clean, read_json, save_json
from training.configuration import digest, task_configuration, validate

RESULTS = ROOT/'runs/policy_tests'
PROTOCOL = 'stationary_target_policy_test_v1'
CASE_FIELDS = {'target_position_world_m','initial_position_world_m',
               'initial_quaternion_body_to_world_xyzw','initial_linear_velocity_world_mps',
               'initial_angular_velocity_body_radps'}


def vector(value, length=3):
    a = np.asarray(value, dtype=np.float64)
    if a.shape != (length,) or not np.isfinite(a).all():
        raise ValueError(f'Expected {length} finite coordinates')
    return a


def rpy_quaternion(degrees):
    """Body-to-world Rz(yaw) Ry(pitch) Rx(roll), world Z up; UI degrees."""
    r, p, y = np.radians(vector(degrees))
    cr,sr,cp,sp,cy,sy = np.cos(r),np.sin(r),np.cos(p),np.sin(p),np.cos(y),np.sin(y)
    rotation = np.array([[cy*cp,cy*sp*sr-sy*cr,cy*sp*cr+sy*sr],
                         [sy*cp,sy*sp*sr+cy*cr,sy*sp*cr-cy*sr],[-sp,cp*sr,cp*cr]])
    return rotation_matrix_to_quaternion_xyzw(rotation)


def initial_case(scene, cfg):
    state = scene.initial_state
    return dict(target_position_world_m=cfg.strike_surface.description(
                    state.position_world_m, cfg.request)['center_world_m'],
                initial_position_world_m=state.position_world_m.tolist(),
                initial_quaternion_body_to_world_xyzw=state.quaternion_body_to_world_xyzw.tolist(),
                initial_linear_velocity_world_mps=state.linear_velocity_world_mps.tolist(),
                initial_angular_velocity_body_radps=state.angular_velocity_body_radps.tolist())


def case_scene(scene, case):
    if set(case) != CASE_FIELDS:
        raise ValueError('Unknown or missing policy-test case fields')
    target = vector(case['target_position_world_m'])
    position = vector(case['initial_position_world_m'])
    q = vector(case['initial_quaternion_body_to_world_xyzw'], 4)
    if not np.isclose(np.linalg.norm(q), 1., rtol=0, atol=1e-9):
        raise ValueError('Initial quaternion must be unit body-to-world XYZW')
    velocity = vector(case['initial_linear_velocity_world_mps'])
    omega = vector(case['initial_angular_velocity_body_radps'])
    old = scene.initial_state
    modified = not (np.array_equal(old.position_world_m,position)
                    and np.allclose(quaternion_xyzw_to_rotation_matrix(old.quaternion_body_to_world_xyzw),
                                    quaternion_xyzw_to_rotation_matrix(q),atol=1e-14,rtol=0)
                    and np.array_equal(old.linear_velocity_world_mps,velocity)
                    and np.array_equal(old.angular_velocity_body_radps,omega))
    if modified and any(x is not None for x in (old.cable_node_positions_world_m,
            old.cable_node_velocities_world_mps,old.cable_material_angles_rad)):
        raise ValueError('Explicit cable initial states cannot be silently rebuilt for a new drone state')
    initial = replace(old,position_world_m=position,quaternion_body_to_world_xyzw=q,
                      linear_velocity_world_mps=velocity,angular_velocity_body_radps=omega)
    return replace(scene,initial_state=initial), target


def case_warnings(snapshot, scene, cfg, case):
    selected, target = case_scene(scene, case)
    cfg.workspace.validate_placement(selected.initial_state.position_world_m, target)
    warnings = []
    if snapshot['settings'].get('target_observation') != 'initial_relative_position_v1':
        warnings.append('This historical whip policy has NO target input: no learned goal adaptation. Target still affects scoring/termination and Stage4 hit/recovery timing.')
    if 'target_sampler' not in snapshot['settings']:
        warnings.append('Checkpoint uses fixed-target training; no validated varied-target adaptation.')
    fields = ('position_world_m','quaternion_body_to_world_xyzw','linear_velocity_world_mps','angular_velocity_body_radps')
    if any(not np.allclose(getattr(selected.initial_state,k),getattr(scene.initial_state,k),rtol=0,atol=1e-12) for k in fields):
        warnings.append('Initial drone state differs from the fixed training start: outside the sampled training domain.')
    nominal = np.asarray(cfg.strike_surface.description(scene.initial_state.position_world_m,cfg.request)['center_world_m'])
    normal = np.asarray(cfg.strike_surface.description(scene.initial_state.position_world_m,cfg.request)['normal_world'])
    delta = target-nominal
    side = np.array([-normal[1],normal[0],0.])
    if 'target_sampler' in snapshot['settings']:
        if abs(delta@normal)>1e-9 or abs(delta@side)>.2000000001 or abs(delta[2])>.2000000001:
            warnings.append('Target is outside the archived sampler domain (same plane, lateral/vertical ±0.20 m).')
    elif not np.allclose(target,nominal,rtol=0,atol=1e-9):
        warnings.append('Target differs from the checkpoint fixed training target; adaptation is unverified.')
    warnings.append('One deterministic simulation case is not robustness, a moving-target test, or flight evidence.')
    return warnings


def require_idle(directory=None):
    """No competing training/evaluation; exclude only this run's venv parent."""
    pattern = r'\b(?:training\.worker|evaluation\.(?:worker|policy_test)|run_goal_stages_comparison\.py)\b'
    if os.name == 'nt':
        script = ("Get-CimInstance Win32_Process | Where-Object { $_.Name -like 'python*' -and $_.CommandLine -match '"
                  +pattern+"' } | Select-Object ProcessId,CommandLine | ConvertTo-Json -Compress")
        import json
        out = subprocess.run(['powershell','-NoProfile','-Command',script],check=True,capture_output=True,
                             text=True,timeout=15,creationflags=subprocess.CREATE_NO_WINDOW).stdout
        rows = json.loads(out) if out.strip() else []
        rows = [rows] if isinstance(rows,dict) else rows
    else:
        import re
        out = subprocess.check_output(['ps','-eo','pid=,args='],text=True,timeout=15)
        rows = [dict(ProcessId=int(a),CommandLine=b) for line in out.splitlines()
                if len(parts:=line.strip().split(None,1))==2 for a,b in [parts] if re.search(pattern,b)]
    def own(row):
        if row['ProcessId']==os.getpid(): return True
        if directory is None or row['ProcessId'] != os.getppid(): return False
        import shlex
        args = [x.strip('"') for x in shlex.split(row['CommandLine'],posix=False)]
        try:
            return (args[args.index('-m')+1]=='evaluation.policy_test'
                    and Path(args[args.index('--run')+1]).resolve()==Path(directory).resolve())
        except (ValueError,IndexError): return False
    active = [row for row in rows if not own(row)]
    if active:
        raise RuntimeError('Policy test waits for training/evaluation workers to exit: '+str(active))
    if (RESULTS/'worker.lock').exists() and directory is None:
        raise RuntimeError('Policy tester lock exists; inspect its owner instead of deleting or relaunching')


def checkpoint_context(checkpoint):
    """Read saved model/task/state; never substitute today's training preset."""
    info = load_checkpoint(checkpoint)
    snapshot = info['snapshot']
    import tempfile
    with tempfile.TemporaryDirectory(prefix='aerowhip-case-') as temp:
        path = Path(temp)/'scene.toml'
        path.write_text(snapshot['scene_text'],encoding='utf-8')
        scene = load_configuration(path)
    cfg = task_configuration(snapshot['task'],snapshot['settings'],snapshot['model'],scene)
    return info, scene, cfg


def prepare(checkpoint, case, *, root=RESULTS):
    checkpoint = Path(checkpoint).resolve()
    info, scene, cfg = checkpoint_context(checkpoint)
    snapshot = info['snapshot']
    contract = evaluation_contract(snapshot,'current_code')
    require_idle()
    warnings = case_warnings(snapshot,scene,cfg,case)
    path = Path(root)/datetime.now().strftime('%Y%m%d-%H%M%S-%f')
    path.mkdir(parents=True,exist_ok=False)
    save_json(path/'configuration.json',snapshot)
    atomic_save(path/'scene.toml',lambda p:p.write_text(snapshot['scene_text'],encoding='utf-8'))
    save_json(path/'case.json',case)
    save_json(path/'request.json',dict(protocol=PROTOCOL,checkpoint=str(checkpoint),
        checkpoint_sha256=info['sha256'],round=info['round'],case_sha256=digest(case),
        warnings=warnings,**contract))
    return path


def frozen_actor(info, env):
    from training.ppo import make_ppo_actor
    from training.sac import Actor
    settings = info['snapshot']['settings']
    actor = (make_ppo_actor(settings,env.actor_columns,env.action_size,**env.policy_encoder_options)
             if settings.get('algorithm')=='ppo' else Actor(env.actor_columns,settings['hidden_units'],
                 settings['initial_log_std'],env.action_size,**env.policy_encoder_options))
    expected = actor.state_dict()
    for name in ('encoder.columns','encoder.phase_columns','encoder.goal_columns','recovery_limits'):
        if name in expected and (name not in info['actor'] or not torch.equal(info['actor'][name],expected[name])):
            raise ValueError('Checkpoint observation/control interface differs: '+name)
    actor.load_state_dict(info['actor'],strict=True)
    if not all(bool(torch.isfinite(p).all()) for p in actor.parameters()):
        raise ValueError('Nonfinite frozen actor')
    return actor.to(env.device).eval().requires_grad_(False)


@torch.no_grad()
def run(directory):
    directory = Path(directory).resolve()
    RESULTS.mkdir(parents=True,exist_ok=True)
    global_lock, lock = RESULTS/'worker.lock',directory/'worker.lock'
    started = perf_counter()
    log = EventLog(directory)
    def emit(**event): log(dict(event,elapsed_s=perf_counter()-started))
    def stopped(): return (directory/'stop.request').exists()
    owned = []
    try:
        for p in (global_lock,lock):
            with p.open('x',encoding='utf-8') as stream:
                stream.write(str(os.getpid()))
            owned.append(p)
        require_idle(directory)
        request, snapshot, case = (read_json(directory/f) for f in ('request.json','configuration.json','case.json'))
        if request['protocol'] != PROTOCOL or request['case_sha256'] != digest(case):
            raise ValueError('Policy-test case/protocol changed after selection')
        info = load_checkpoint(request['checkpoint'])
        if info['sha256'] != request['checkpoint_sha256'] or snapshot != info['snapshot']:
            raise ValueError('Frozen checkpoint changed after case preparation')
        if (directory/'scene.toml').read_text(encoding='utf-8') != snapshot['scene_text']:
            raise ValueError('Saved source scene changed')
        contract = verify_contract(snapshot,request)
        settings = snapshot['settings']; validate(settings)
        if settings['device']=='cuda' and not torch.cuda.is_available():
            raise RuntimeError('Checkpoint CUDA device is unavailable; no silent CPU fallback')
        torch.set_num_threads(min(8,os.cpu_count() or 1))
        scene = load_configuration(directory/'scene.toml')
        cfg = task_configuration(snapshot['task'],settings,snapshot['model'],scene)
        warnings = case_warnings(snapshot,scene,cfg,case)
        scene, target = case_scene(scene,case)
        cfg = replace(cfg,request=(*cfg.request[:2],float(target[2]),cfg.request[3]))
        emit(phase='initializing',round=info['round'],pid=os.getpid())
        env = DeterministicEvaluationEnvironment(settings,snapshot,cfg,scene)
        configure_targets(env,[target])
        actor = frozen_actor(info,env)
        save_json(directory/'provenance.json',dict(protocol=PROTOCOL,pid=os.getpid(),
            executable=sys.executable,python=platform.python_version(),torch=str(torch.__version__),
            device=settings['device'],gpu=torch.cuda.get_device_name() if settings['device']=='cuda' else None,
            checkpoint_sha256=info['sha256'],source_identity=snapshot['identity'],case=case,
            deterministic=True,optimizer_updates=0,**contract,**env.provenance()))
        env.reset(); done=False; last=perf_counter()
        while not stopped():
            action = env.policy_action(lambda h,i:actor(h,i,deterministic=True)[0])
            _,_,done = env.step(action)
            if done or perf_counter()-last>1:
                emit(phase='evaluating',round=info['round'],simulation_s=env.index*.01,
                     duration_s=(env.samples-1)*.01); last=perf_counter()
            if done: break
        if not done:
            summary=dict(phase='cancelled',round=info['round'],message='Incomplete episode was not scored',replay=None)
        else:
            returns,valid,score,cost=env.finish()
            goal_contract = None
            if env.relative_target_input:
                goal = env.goal_relative[0].tolist()
                observed = env.frames[:,:,-3:][env.numerical_valid]
                expected = env.goal_relative.expand(len(observed),3)
                if not torch.equal(observed,expected):
                    raise ValueError('Stationary goal input changed during the episode')
                goal_contract = dict(relative_position_over_cable_length=goal,
                                     fixed_at_all_valid_samples=True,world_axis_aligned=True)
            arrays=env.replay_arrays(0)
            filename='rollout.npz' if arrays is not None else None
            if arrays is not None:
                def writer(path):
                    with path.open('wb') as f: np.savez_compressed(f,**arrays)
                atomic_save(directory/filename,writer)
            summary=clean(dict(phase='completed',round=info['round'],protocol=PROTOCOL,
                valid=bool(valid[0]),evaluation_return=float(returns[0]),replay=filename,
                acceleration_cost=float(cost[0]),components={k:float(v[0]) for k,v in score.components.items()},
                workspace={k:float(v[0]) for k,v in env.workspace_metrics().items()},
                target_surface=target_surface(env,0),case=case,warnings=warnings,goal_observation=goal_contract,
                task_stage=task_stage(snapshot),source_mode='current_code',exact_source_reproduction=False,
                scoring='saved_task_final_target_size',checkpoint_sha256=info['sha256'],**env.provenance()))
        verify_contract(snapshot,request)
        if load_checkpoint(request['checkpoint'])['sha256']!=request['checkpoint_sha256']:
            raise ValueError('Source checkpoint changed during the test')
        summary['elapsed_s']=perf_counter()-started
        save_json(directory/'summary.json',summary); log(summary)
    except Exception as exc:
        summary=dict(phase='failed',error=str(exc),elapsed_s=perf_counter()-started)
        save_json(directory/'summary.json',summary); log(summary)
        traceback.print_exc(); raise
    finally:
        for p in reversed(owned): p.unlink()


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run',type=Path,required=True)
    run(parser.parse_args().run)
