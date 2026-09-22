"""PyCharm: bounded candidate study, with printed progress and zero policy updates."""

from datetime import datetime, timezone
import json
import platform
from pathlib import Path
import shutil
import subprocess
import sys
from time import perf_counter

import numpy as np
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[3]
if __package__ in (None, ''):
    sys.path.insert(0, str(ROOT))

from simulation.configuration import load_configuration
from simulation.dynamics.coupled import CoupledDynamics
from tasks.whip.analysis.candidates import candidate_actions, verify_candidates
from tasks.whip.analysis.propagation import curvature_advection, verify_advection
from tasks.whip.analysis.run_reach_comparison import CONFIGURATION, HISTORICAL, verified_arrays
from tasks.whip.experiment import (load_experiment, SingleScenarioTask, digest,
                                   write_json, save_episode_records, statistics)
from tasks.whip.metrics import WhipMetrics
from tasks.whip.reach import ReachConfiguration, window_distance
from tasks.whip.workspace import CylinderWorkspace

SEEDS = (0, 1, 2)
BATCH_SIZE = 256
DEVICE = 'cuda'


def serializable(value):
    if isinstance(value, dict):
        return {key: serializable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [serializable(item) for item in value]
    if isinstance(value, torch.Tensor):
        return serializable(value.detach().cpu().tolist())
    if isinstance(value, np.ndarray):
        return serializable(value.tolist())
    if isinstance(value, (float, np.floating)):
        return float(value) if np.isfinite(value) else None
    return value


def save_arrays(path, **arrays):
    with path.open('xb') as stream:
        np.savez_compressed(stream, **{key: value.detach().cpu().numpy()
            if isinstance(value, torch.Tensor) else value for key, value in arrays.items()})


@torch.no_grad()
def diagnose(positions, velocities, sample_valid, time, rest, start, targets, raw):
    """One fixed whole-motion window and one pre-closest-approach window.

    Neither window is selected by maximizing the new diagnostic. No success
    threshold, physical energy claim, or new reward is returned.
    """
    batch = len(positions)
    measure = WhipMetrics(rest, time, region_boundaries_m=rest.new_tensor(
        raw['reward']['region_fractions'])*rest.sum())
    measured = measure(positions, velocities, sample_valid)
    window = ReachConfiguration(**raw['reach'])
    distance = window_distance(positions[:, :, -1], start, targets, window)
    delay = float(raw['_study_delay_s'])
    end = delay+raw['trajectory']['duration_s']
    permitted = (time >= delay-1e-12) & (time <= end+raw['reward']['tip_window_s']+1e-12)
    finite = measured.sample_valid.all(-1) & torch.isfinite(distance).all(-1)
    index = distance.masked_fill(~permitted[None] | ~finite[:, None], float('inf')).argmin(-1)
    event = time[index]
    normal = targets[:, :2]-start[:, :2]
    normal = normal/normal.norm(dim=-1, keepdim=True)
    forward = (velocities[:, :, -1, :2]*normal[:, None]).sum(-1)
    progress = (1-distance/distance[:, :1]).clamp(0, 1)
    whole = rest.new_tensor([delay, end])[None].expand(batch, -1)
    event_windows = torch.stack(((event-raw['reward']['maximum_progression_s']).clamp_min(delay), event), -1)
    # An early closest approach can have no preceding cells. This is undefined,
    # not evidence against whipping. Preserve that distinction in saved results.
    whole_fit = curvature_advection(measured.curvature_per_m, time, measure.s[1:-1],
                                    measured.sample_valid, whole)
    event_fit = curvature_advection(measured.curvature_per_m, time, measure.s[1:-1],
                                    measured.sample_valid, event_windows)

    def selected(history):
        return history.gather(1, index[:, None])[:, 0].masked_fill(~finite, float('nan'))

    info = dict(history_valid=finite, event_index=index,
        event_time_s=event.masked_fill(~finite, float('nan')),
        closest_window_distance_m=selected(distance), reach_progress=selected(progress),
        forward_speed_at_event_mps=selected(forward), tip_speed_at_event_mps=selected(measured.tip_speed_mps))
    info.update({f'whole_{key}': value for key, value in whole_fit.items()})
    info.update({f'event_{key}': value for key, value in event_fit.items()})
    return info, measured, distance, forward


def plot_case(folder, title, time, measured, distance, forward, info, index=0):
    t = time.cpu().numpy()
    curvature = measured.curvature_per_m[index].cpu().numpy()
    fig, axes = plt.subplots(3, 1, figsize=(10, 8), sharex=True, layout='constrained')
    finite = curvature[np.isfinite(curvature)]
    ceiling = max(float(finite.max()) if finite.size else 0., 1e-6)
    mesh = axes[0].pcolormesh(t, measured.node_arclength_m[1:-1].cpu(), curvature.T,
                              shading='nearest', vmin=0, vmax=ceiling, cmap='viridis')
    fig.colorbar(mesh, ax=axes[0], label='Curvature (1/m)')
    axes[0].set_ylabel('Material position (m)')
    axes[1].plot(t, distance[index].cpu(), label='Target-window distance')
    axes[1].set_ylabel('Distance (m)')
    axes[2].plot(t, forward[index].cpu(), label='Forward tip speed')
    axes[2].plot(t, measured.tip_speed_mps[index].cpu(), label='World tip speed', alpha=.6)
    axes[2].set_ylabel('Speed (m/s)'); axes[2].set_xlabel('Time (s)')
    event = float(info['event_time_s'][index])
    for axis in axes:
        if np.isfinite(event):
            axis.axvline(event, color='black', linestyle='--', linewidth=.8)
        axis.set_xlim(t[0], t[-1])
    axes[1].legend(); axes[2].legend()
    rho = serializable(info['event_rho'][index])
    fig.suptitle(f'{title}\nPre-event advection coherence: {rho}; not a whip classification')
    fig.savefig(folder/'diagnostics.png', dpi=140)
    plt.close(fig)


def summarize(info, validity):
    result = dict(attempts=len(validity), valid_count=int(validity.sum()))
    for key in ('reach_progress', 'closest_window_distance_m', 'forward_speed_at_event_mps',
                'event_rho', 'whole_rho', 'event_speed_mps'):
        value = info[key][validity & torch.isfinite(info[key])]
        result[key] = dict(defined_count=len(value), median=float(value.median()) if len(value) else None,
            q90=float(torch.quantile(value, .9)) if len(value) else None,
            minimum=float(value.min()) if len(value) else None,
            maximum=float(value.max()) if len(value) else None)
    return result


def retained_comparison(output, raw, rest, start, target, model):
    rows = []
    inputs = []
    time = torch.arange(351, dtype=torch.float64)*.01
    cases = []
    for name in ('M3', 'M4', 'M5', 'M6', 'Gentle'):
        path = HISTORICAL/name/'measurements.npz'
        arrays = verified_arrays(path, HISTORICAL)
        np.testing.assert_allclose(arrays['cable_rest_lengths'], rest.numpy(), rtol=0, atol=1e-15)
        cases.append((name, arrays, 'Retained simulated command; not recorded flight or admissible-task ground truth'))
        inputs.append(dict(name=name, path=str(path), sha256=digest(path)))
    u = time.clamp_max(3.)
    amplitude = (torch.pi*u/3).sin().square()
    rate = torch.pi/3*(2*torch.pi*u/3).sin()
    s = torch.cat((rest.new_zeros(1), rest.cumsum(0)))
    for name, sign in (('Rigid_forward', 1.), ('Rigid_backward', -1.)):
        theta, omega = sign*1.4*amplitude, sign*1.4*rate
        drone = start.expand(len(time), -1).clone(); drone[:, 0] += sign*.25*amplitude
        root = drone+rest.new_tensor(model['attachment']['position_tracking_m'])
        direction = torch.stack((theta.sin(), torch.zeros_like(theta), -theta.cos()), -1)
        derivative = omega[:, None]*torch.stack((theta.cos(), torch.zeros_like(theta), theta.sin()), -1)
        root_velocity = torch.zeros_like(root); root_velocity[:, 0] = sign*.25*rate
        cases.append((name, dict(time_s=time.numpy(), drone_positions=drone.numpy(),
            drone_rotations=np.broadcast_to(np.eye(3), (len(time), 3, 3)),
            cable_positions=(root[:, None]+s[None, :, None]*direction[:, None]).numpy(),
            cable_velocities=(root_velocity[:, None]+s[None, :, None]*derivative[:, None]).numpy(),
            state_valid=np.ones(len(time), dtype=bool)), 'Prescribed rigid kinematics; no dynamic feasibility claim'))
    workspace = CylinderWorkspace(**raw['workspace'])
    for name, arrays, evidence in cases:
        folder = output/name; folder.mkdir()
        np.testing.assert_allclose(arrays['time_s'][:len(time)], time.numpy(), rtol=0, atol=1e-12)
        shift = start.numpy()-arrays['drone_positions'][0]
        position = torch.tensor(arrays['cable_positions'][:len(time)]+shift)[None]
        velocity = torch.tensor(arrays['cable_velocities'][:len(time)])[None]
        valid = torch.tensor(arrays['state_valid'][:len(time)], dtype=torch.bool)[None]
        info, measured, distance, forward = diagnose(position, velocity, valid, time, rest,
                                                    start[None], target[None], raw)
        inside = workspace.contains(torch.tensor(arrays['drone_positions'][:len(time)]+shift)[None], start[None]).all()
        row = dict(name=name, evidence=evidence, translation_applied_m=shift.tolist(),
                   workspace_satisfied=bool(inside), diagnostics={k: serializable(v[0]) for k, v in info.items()})
        rows.append(row)
        save_arrays(folder/'measurements.npz', **info, time_s=time, curvature_per_m=measured.curvature_per_m,
                    material_s_m=measured.node_arclength_m[1:-1], window_distance_m=distance, forward_speed_mps=forward)
        plot_case(folder, name, time, measured, distance, forward, info)
        print(f'Retained {name}: {json.dumps(row["diagnostics"])}', flush=True)
    return dict(inputs=inputs, cases=rows, physics_rollouts=0)


def main(*, simulate=True):
    torch.set_num_threads(2)
    raw, trajectory, scoring, returns = load_experiment(CONFIGURATION)
    scene = load_configuration(ROOT/raw['experiment']['scene'])
    model_path = ROOT/raw['experiment']['model']
    model_config = json.loads(model_path.read_text(encoding='utf-8'))
    raw['_study_delay_s'] = model_config['controller']['command_delay_s']
    rest = torch.tensor(np.concatenate([np.full(n, length/n) for length, n in zip(
        model_config['cable']['marker_interval_lengths_m'], model_config['simulation']['segments_per_marker_interval'])]))
    start = torch.tensor(scene.initial_state.position_world_m, dtype=torch.float64)
    target = torch.tensor(raw['experiment']['target_position_world_m'], dtype=torch.float64)
    normal = target-start; normal[2] = 0.; normal /= normal.norm()
    output = ROOT/'runs/exploration_comparison'/datetime.now(timezone.utc).strftime('%Y%m%d-%H%M%S-%f')
    output.mkdir(parents=True, exist_ok=False)
    print(f'Exploration comparison: {output}\nZero optimizer updates; production settings unchanged.', flush=True)
    # Capture inputs/source identities before any work; never overwrite a run.
    protected = [model_path, ROOT/raw['experiment']['scene'], CONFIGURATION]
    residual = Path(model_config['residual']['checkpoint'])
    if not residual.is_absolute(): residual = ROOT/residual
    protected.append(residual)
    protected += [Path(__file__), ROOT/'tasks/whip/analysis/candidates.py',
                  ROOT/'tasks/whip/analysis/propagation.py', ROOT/'tasks/whip/analysis/run_reach_comparison.py']
    protected += [p for top in ('simulation', 'trajectories', 'training', 'tasks/whip')
                  for p in (ROOT/top).rglob('*.py') if 'analysis' not in p.parts]
    before = {str(path.relative_to(ROOT) if path.is_relative_to(ROOT) else path): digest(path)
              for path in protected}
    for path in (Path(__file__), ROOT/'tasks/whip/analysis/candidates.py', ROOT/'tasks/whip/analysis/propagation.py',
                 ROOT/'tasks/whip/analysis/run_reach_comparison.py'):
        shutil.copy2(path, output/path.name)
    write_json(output/'protocol.json', dict(configuration=serializable(raw), seeds=SEEDS, batch_size=BATCH_SIZE,
        mode='bounded_forward_comparison' if simulate else 'retained_history_analysis_only',
        planned_raw_candidates_per_arm=len(SEEDS)*BATCH_SIZE, device=DEVICE if simulate else 'cpu',
        protected_sha256=before, optimizer_updates=0,
        versions=dict(python=platform.python_version(), torch=str(torch.__version__), numpy=np.__version__,
            matplotlib=matplotlib.__version__, cuda=torch.version.cuda),
        gpu=torch.cuda.get_device_name() if simulate else None,
        gpu_driver=subprocess.check_output(['nvidia-smi', '--query-gpu=driver_version', '--format=csv,noheader'],
                                           text=True).strip() if simulate else None,
        windows='Whole: command delay to delay+3s. Event: preceding 0.40s at closest window approach, clipped to delay.',
        selection='One valid closest-window candidate per arm/seed. No selection by new diagnostic.',
        rejection='Raw proposals retained; no resampling, clipping, repair, or rewarding invalid episodes.',
        limitations=['Advection is a constant-speed transport fit, not connected-pulse tracking or a whip classifier.',
            'Prior comparison changes mean and command amplitude; not causal isolation of preparation.',
            'Swing candidates can themselves deform/whip; they are not labelled negatives.',
            'Equal raw proposal budgets; command-rejected lanes do not constitute admitted full physical episodes.',
            'No policy optimization, learning-curve comparison, promotion, or physical validation.']))
    checks = dict(advection=verify_advection(), candidates=verify_candidates(
        trajectory, raw['exploration'], raw['learner'], normal))
    write_json(output/'verification.json', serializable(checks))
    retained = retained_comparison(output, raw, rest, start, target, model_config)
    write_json(output/'retained.json', serializable(retained))
    results = []
    if simulate:
        if DEVICE != 'cuda' or not torch.cuda.is_available():
            raise RuntimeError('This explicit GPU study requires CUDA; no silent CPU fallback')
        model = CoupledDynamics(model_config, device=DEVICE, batched=True)
        task = SingleScenarioTask(model, scene, trajectory, scoring, returns, target.numpy(), BATCH_SIZE,
            workspace=CylinderWorkspace(**raw['workspace']), reach=ReachConfiguration(**raw['reach']))
        for seed in SEEDS:
            arms, details = candidate_actions(trajectory, raw['exploration'], raw['learner'], normal, seed, BATCH_SIZE)
            write_json(output/f'candidates_seed_{seed}.json', serializable(details))
            # Rotate order across seeds to reduce simple order/thermal confounding.
            names = list(arms); offset = seed % len(names); names = names[offset:]+names[:offset]
            for name in names:
                folder = output/f'{name}_seed_{seed}'; folder.mkdir()
                print(f'Running {name}, seed {seed}: {BATCH_SIZE} raw candidates, 3.5s each...', flush=True)
                torch.cuda.synchronize(); began = perf_counter()
                episode, result = task.evaluate(arms[name].to(DEVICE))
                torch.cuda.synchronize(); rollout_seconds = perf_counter()-began
                rollout = episode.rollout
                began = perf_counter()
                info, measured, distance, forward = diagnose(rollout.cable_positions, rollout.cable_velocities,
                    rollout.state_valid, rollout.time_s, rollout.cable_rest_lengths,
                    task.environment.start_position, task.targets, raw)
                torch.cuda.synchronize(); diagnostic_seconds = perf_counter()-began
                save_episode_records(folder, 'episodes', episode, result)
                save_arrays(folder/'diagnostics.npz', **info)
                stats = dict(arm=name, seed=seed, rollout_seconds=rollout_seconds,
                    diagnostic_seconds=diagnostic_seconds, production=statistics(episode, result),
                    measurements=summarize(info, result.valid))
                if bool(result.valid.any()):
                    selected = int(info['closest_window_distance_m'].masked_fill(~result.valid, float('inf')).argmin())
                    stats['selected_index'] = selected
                    stats['selected'] = {k: serializable(v[selected]) for k, v in info.items()}
                    episode.trajectory(selected).save(folder/'command.csv')
                    save_arrays(folder/'rollout.npz', time_s=rollout.time_s,
                        cable_rest_lengths=rollout.cable_rest_lengths, cable_positions=rollout.cable_positions[selected],
                        cable_velocities=rollout.cable_velocities[selected], drone_positions=rollout.drone_positions[selected],
                        drone_rotations=rollout.drone_rotations[selected], state_valid=rollout.state_valid[selected],
                        commands=episode.commands[selected], action=episode.actions[selected])
                    plot_case(folder, f'{name}, seed {seed}, candidate {selected}', rollout.time_s,
                              measured, distance, forward, info, selected)
                write_json(folder/'summary.json', serializable(stats)); results.append(stats)
                print(json.dumps(dict(arm=name, seed=seed, seconds=rollout_seconds,
                    valid=int(result.valid.sum()), best_distance_m=stats.get('selected', {}).get('closest_window_distance_m'))), flush=True)
                del episode, result, info, measured, distance, forward, rollout
    after = {name: digest(ROOT/name) for name in before}
    if before != after:
        raise RuntimeError('Protected input/source changed during the study; comparison must be reviewed')
    summary = dict(status='completed' if simulate else 'gpu_comparison_not_run', optimizer_updates=0,
        attempted_candidate_rollouts=len(results)*BATCH_SIZE, protected_files_unchanged=True,
        results=results, retained_cases=len(retained['cases']))
    write_json(output/'summary.json', serializable(summary))
    write_json(output/'artifacts.json', {str(path.relative_to(output)): digest(path)
        for path in output.rglob('*') if path.is_file()})
    print(f'Finished: {output}', flush=True)
    return output


if __name__ == '__main__':
    main()
