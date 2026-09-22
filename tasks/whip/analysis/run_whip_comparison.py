"""Run in PyCharm: compare M3-M6 plans and a gentle swing candidate, without RL."""

from dataclasses import fields
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import platform
import re
import shutil
import sys

import time


import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[3]
if __package__ in (None, ""):
    sys.path.insert(0, str(ROOT))

from trajectories.full_state import FullStateTrajectory
from simulation.batched_rollout import BatchedRollout
from simulation.configuration import DEFAULT_MODEL, load_configuration
from simulation.dynamics.coupled import CoupledDynamics
from tasks.whip.metrics import WhipMetrics

# Analysis settings, not training/reward defaults. No command-line options needed.
POST_HOLD_S = .5
OVERVIEW_END_S = 3.5
REGION_FRACTIONS = (0., 1/3, 2/3, 1.)
DT = .01


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write_json(path, value):
    path.write_text(json.dumps(value, indent=2, allow_nan=False)+'\n', encoding='utf-8')


def load_cases():
    catalog = json.loads((ROOT/'data/take_processing/catalog.json').read_text(encoding='utf-8'))
    cases = []
    for name in ('M3', 'M4', 'M5', 'M6'):
        references = {(e['sources']['reference']['path'], e['sources']['reference']['sha256'])
                      for e in catalog['takes'].values() if e['name'].startswith(name+'_')}
        if len(references) != 1:
            raise ValueError(f'{name}: expected exactly one catalog-linked reference archive')
        path, expected_hash = next(iter(references))
        path = Path(path)
        if digest(path) != expected_hash:
            raise ValueError(f'{name}: reference hash does not match catalog')
        with np.load(path, allow_pickle=False) as source:
            t, command, phase = (source[k].copy() for k in ('command_time_s', 'commands', 'command_phase'))
        if (t.ndim != 1 or len(t) < 2 or command.shape != (len(t), 11)
                or phase.shape != t.shape or not np.isfinite(t).all()
                or not np.isfinite(command).all() or t[0] != 0 or not np.all(np.diff(t) > 0)
                or np.any(command[:, 10] != 0) or not np.isin(phase, [1, 2, 3, 4]).all()
                or not np.allclose(command[0, 3:9], 0, atol=1e-10, rtol=0)
                or not np.allclose(command[-1, 3:9], 0, atol=1e-10, rtol=0)):
            raise ValueError(f'{name}: unsupported source command contract')
        transitions = np.flatnonzero(np.diff(phase) != 0)+1
        cases.append(dict(name=name, source_path=str(path), source_sha256=expected_hash,
                          source_time=t, source_commands=command, source_phase=phase,
                          transitions=[dict(time_s=float(t[k]), phase=int(phase[k])) for k in transitions],
                          kind='saved planned reference, not a recorded flight-command log'))
    gentle = FullStateTrajectory.gentle_move((0., 0., 2.))
    t = np.arange(gentle.sample_count)*DT
    command = np.stack([gentle.command(k) for k in range(len(t))])
    cases.append(dict(name='Gentle', source_path=None, source_sha256=None,
                      source_time=t, source_commands=command, source_phase=np.zeros(len(t), dtype=int),
                      transitions=[dict(time_s=1., phase='move'), dict(time_s=3., phase='hold')],
                      kind='built-in 0.25 m X move over 2 s; intended swing counterexample, not a certified non-whip'))
    for case in cases:
        # A source row is held until the next source timestamp. Coincident grid
        # times are resolved with 1 ps roundoff allowance, not interpolation.
        case['steps'] = math.ceil((case['source_time'][-1]+POST_HOLD_S)/DT-1e-10)
        t = np.arange(case['steps']+1)*DT
        index = np.searchsorted(case['source_time'], t+1e-12, side='right')-1
        case['commands'] = case['source_commands'][index].copy()
        case['sampled_phase'] = case['source_phase'][index].copy()
        case['source_indices'] = index
    return cases


def export_replay(case, folder):
    rows = np.zeros((len(case['commands']), 14))
    rows[:, 0] = np.arange(len(rows))*DT
    rows[:, 1:11] = case['commands'][:, :10]
    trajectory = FullStateTrajectory(rows)
    trajectory.save(folder/'commands.csv')
    scene = (ROOT/'configs/scenes/default.toml').read_text(encoding='utf-8')
    yaw = case['commands'][0, 9]
    replacements = dict(position_world_m=case['commands'][0, :3].tolist(),
                        quaternion_body_to_world_xyzw=[0., 0., math.sin(yaw/2), math.cos(yaw/2)])
    for key, value in replacements.items():
        scene, count = re.subn(r'^'+key+r'\s*=.*$', key+' = '+json.dumps(value), scene, flags=re.MULTILINE)
        if count != 1:
            raise ValueError(f'Expected exactly one {key} in scene template')
    (folder/'scene.toml').write_text(scene, encoding='utf-8')
    reloaded = FullStateTrajectory.load(folder/'commands.csv')
    reloaded.validate_initial(load_configuration(folder/'scene.toml'))
    np.testing.assert_allclose(np.stack([reloaded.command(k) for k in range(len(rows))]),
                               case['commands'], atol=1e-10, rtol=0)


def annotate_phases(ax, case, delay, end):
    for change in case['transitions']:
        t = math.ceil((change['time_s']-1e-12)/DT)*DT+delay
        if t < end:
            ax.axvline(t, color='#777777', linestyle=':', linewidth=.7, alpha=.65)


def heatmap(ax, case, key, *, end, vmax, speed=False):
    a = case['arrays']
    use = a['time_s'] <= end+1e-12
    t = a['time_s'][use]
    te = np.r_[t[0], (t[:-1]+t[1:])/2, t[-1]]
    s = a['node_arclength_m']
    se = np.r_[s[0], (s[:-1]+s[1:])/2, s[-1]] if speed else (s[:-1]+s[1:])/2
    return ax.pcolormesh(te, se, a[key][use].T, shading='flat', vmin=0, vmax=vmax, cmap='magma')


def plot_comparison(cases, output, delay):
    colors = ('#2676b8', '#ee9434', '#28a178')
    plt.rcParams.update({'font.size': 9, 'axes.spines.top': False, 'axes.spines.right': False})
    max_curvature = max(float(np.nanmax(c['arrays']['curvature_per_m'][c['arrays']['time_s'] <= OVERVIEW_END_S])) for c in cases)
    max_node_speed = max(float(np.nanmax(c['arrays']['node_relative_speed_mps'][c['arrays']['time_s'] <= OVERVIEW_END_S])) for c in cases)
    max_tip_speed = max(float(np.nanmax(c['arrays']['tip_relative_speed_mps'][c['arrays']['time_s'] <= OVERVIEW_END_S])) for c in cases)
    fig, axes = plt.subplots(len(cases), 3, figsize=(14, 12), layout='constrained')
    for row, case in enumerate(cases):
        a = case['arrays']; t = a['time_s']
        images = []
        for col, (key, vmax, speed) in enumerate((('curvature_per_m', max_curvature, False),
                                                  ('node_relative_speed_mps', max_node_speed, True))):
            images.append(heatmap(axes[row, col], case, key, end=OVERVIEW_END_S, vmax=vmax, speed=speed))
            axes[row, col].set_ylim(0, a['node_arclength_m'][-1])
            axes[row, col].set_ylabel(f"{case['name']}  |  s (m)")
        for j, color in enumerate(colors):
            axes[row, 2].plot(t, a['region_relative_rms_speed_mps'][:, j], color=color,
                              label=('Proximal', 'Middle', 'Distal')[j])
        axes[row, 2].plot(t, a['tip_relative_speed_mps'], color='#27272a', linewidth=1.4, label='Tip')
        axes[row, 2].set_ylim(0, max(max_tip_speed, max_node_speed)*1.05)
        axes[row, 2].set_ylabel('Relative speed (m/s)')
        for ax in axes[row]:
            ax.set_xlim(0, OVERVIEW_END_S)
            ax.set_xlabel('Time (s)')
            annotate_phases(ax, case, delay, OVERVIEW_END_S)
    for ax, title in zip(axes[0], ('Bending along the cable', 'Motion along the cable', 'Regional and tip motion')):
        ax.set_title(title, fontweight='bold')
    axes[0, 2].legend(ncol=2, fontsize=8)
    fig.colorbar(images[0], ax=axes[:, 0], location='bottom', shrink=.8, label='Turning density (1/m), shared linear scale')
    fig.colorbar(images[1], ax=axes[:, 1], location='bottom', shrink=.8, label='Speed relative to attachment (m/s), shared linear scale')
    fig.suptitle('Current-model replay of saved plans\nFirst 3.5 s; equal material thirds; dotted lines = delayed command-phase changes', fontsize=14)
    fig.savefig(output/'comparison.png', dpi=160)
    plt.close(fig)
    for case in cases:
        a = case['arrays']; t = a['time_s']; end = t[-1]
        fig, ax = plt.subplots(4, 1, figsize=(11, 10), sharex=True, layout='constrained')
        im = heatmap(ax[0], case, 'curvature_per_m', end=end, vmax=max(float(np.nanmax(a['curvature_per_m'])), 1e-12))
        ax[0].set_ylabel('Material s (m)')
        fig.colorbar(im, ax=ax[0], label='Turning density (1/m)')
        for i, key in enumerate(('region_rms_speed_mps', 'region_relative_rms_speed_mps'), start=1):
            for j, color in enumerate(colors):
                ax[i].plot(t, a[key][:, j], color=color, label=('Proximal', 'Middle', 'Distal')[j])
            ax[i].plot(t, a['tip_speed_mps' if i == 1 else 'tip_relative_speed_mps'], color='#27272a', label='Tip')
            ax[i].set_ylabel(('World' if i == 1 else 'Relative')+' speed (m/s)')
            ax[i].legend(ncol=4, fontsize=8)
        ax[3].plot(t, a['bend_centroid_m'], label='Bend centroid')
        ax[3].fill_between(t, a['bend_centroid_m']-a['bend_spread_m'],
                          a['bend_centroid_m']+a['bend_spread_m'], alpha=.15, label='Centroid +/- spread')
        ax[3].set_ylabel('Material s (m)'); ax[3].set_ylim(0, a['node_arclength_m'][-1]); ax[3].legend()
        for axis in ax:
            axis.set_xlim(0, end)
            annotate_phases(axis, case, delay, end)
        ax[-1].set_xlabel('Time (s)')
        fig.suptitle(case['name']+' | Full simulated history\nCentroid is not a tracked wave; near-zero bends can give meaningless centroids', fontsize=13)
        fig.savefig(output/case['name']/'diagnostics.png', dpi=150)
        plt.close(fig)


def main():
    cases = load_cases()
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    stamp = datetime.now(timezone.utc).strftime('%Y%m%d-%H%M%S-%f')
    output = ROOT/'runs/whip_comparison'/stamp
    output.mkdir(parents=True, exist_ok=False)
    print(f'Output: {output}', flush=True)
    source_files = [Path(__file__), ROOT/'tasks/whip/metrics.py', ROOT/'trajectories/full_state.py',
                    ROOT/'simulation/batched_rollout.py', ROOT/'simulation/configuration.py',
                    ROOT/'configs/scenes/default.toml', ROOT/'data/take_processing/catalog.json']
    source_files += list((ROOT/'simulation/dynamics').glob('*.py'))
    source_hashes = {}
    for path in source_files:
        relative = path.relative_to(ROOT)
        destination = output/'source_snapshot'/relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(path, destination)
        source_hashes[str(relative)] = digest(path)
    shutil.copy2(DEFAULT_MODEL, output/'model.json')
    config = json.loads(DEFAULT_MODEL.read_text(encoding='utf-8'))
    checkpoint = Path(config['residual']['checkpoint'])
    if not checkpoint.is_absolute():
        checkpoint = ROOT/checkpoint
    shutil.copy2(checkpoint, output/'residual.pt')
    protocol = dict(schema='whip_measurement_comparison_v1', created_utc=stamp,
        model_path=str(DEFAULT_MODEL), model_sha256=digest(DEFAULT_MODEL), residual_sha256=digest(checkpoint),
        source_sha256=source_hashes, device=device, dtype='float64', timestep_s=DT,
        region_fractions=REGION_FRACTIONS, post_hold_s=POST_HOLD_S, overview_end_s=OVERVIEW_END_S,
        sampling='Source-command zero-order hold on 100 Hz grid; 1e-12 s timestamp roundoff allowance',
        initial_state='First command tracking position/yaw, zero velocities, level drone, hanging current-model cable',
        comparison_scope='Saved plans, not recorded flight logs. All source phases retained. No source predictions reused.',
        exclusions='No whip labels, classifier, reward, thresholds, target score, fit update, or training',
        versions=dict(python=platform.python_version(), torch=torch.__version__, numpy=np.__version__, matplotlib=matplotlib.__version__),
        gpu=torch.cuda.get_device_name() if device == 'cuda' else None, cases=[])
    for case in cases:
        folder = output/case['name']; folder.mkdir()
        export_replay(case, folder)
        if case['source_path']:
            shutil.copy2(case['source_path'], folder/'source.npz')
        else:
            np.savez_compressed(folder/'source.npz', command_time_s=case['source_time'],
                                commands=case['source_commands'], command_phase=case['source_phase'])
        np.savez_compressed(folder/'sampling.npz', source_indices=case['source_indices'], phase=case['sampled_phase'])
        protocol['cases'].append({k:case[k] for k in ('name','kind','source_path','source_sha256','steps','transitions')})
    write_json(output/'protocol.json', protocol)
    # Run from the retained checkpoint copy, without rewriting the original
    # model JSON or its recorded hash. Relative references in the original
    # JSON otherwise resolve against the live repository's model directory.
    runtime_config = json.loads((output/'model.json').read_text(encoding='utf-8'))
    runtime_config['residual']['checkpoint'] = str(output/'residual.pt')
    model = CoupledDynamics(runtime_config, device=device, batched=True)
    if not math.isclose(model.dt, DT, rel_tol=0, abs_tol=1e-12):
        raise ValueError('This analysis requires 100 Hz physics')
    initial = []
    for case in cases:
        yaw = case['commands'][0, 9]
        rotation = model.tensor([[math.cos(yaw), -math.sin(yaw), 0], [math.sin(yaw), math.cos(yaw), 0], [0,0,1]])
        initial.append(model.initial_state(case['commands'][0, :3], rotation=rotation))
    runner = BatchedRollout(model, initial)
    steps = max(c['steps'] for c in cases)
    commands = np.stack([np.concatenate((c['commands'][:-1], np.repeat(c['commands'][-1:], steps-c['steps'], axis=0))) for c in cases])
    if device == 'cuda': torch.cuda.synchronize()
    start = time.perf_counter()
    result = runner.run(model.tensor(commands))
    if device == 'cuda': torch.cuda.synchronize()
    elapsed = time.perf_counter()-start
    print(f'Completed {len(cases)} lanes / {steps} intervals in {elapsed:.2f} s', flush=True)
    metrics = WhipMetrics(result.cable_rest_lengths, result.time_s,
                          region_boundaries_m=model.tensor(REGION_FRACTIONS)*model.rest.sum())
    measured = metrics(result.cable_positions, result.cable_velocities, result.state_valid)
    shared = {'time_s','interval_time_s','node_arclength_m','region_boundaries_m','region_bend_support_m'}
    values = {f.name:getattr(measured, f.name).cpu().numpy() for f in fields(measured)}
    summary = dict(rollout_wall_seconds=elapsed, batch_intervals=steps, cases=[])
    for lane, case in enumerate(cases):
        count = case['steps']+1
        a = {}
        for key, value in values.items():
            a[key] = value if key in shared else value[lane, :count-1 if key == 'bend_centroid_rate_mps' else count]
        a['time_s'] = values['time_s'][:count]; a['interval_time_s'] = values['interval_time_s'][:count-1]
        for key in ('drone_positions','drone_velocities','drone_rotations','drone_angular_velocities',
                    'cable_positions','cable_velocities','state_valid','requested_commands','applied_commands',
                    'command_valid','step_finite','step_rotation_valid'):
            limit = count-1 if key in ('requested_commands','applied_commands','command_valid','step_finite','step_rotation_valid') else count
            a[key] = getattr(result,key)[lane,:limit].cpu().numpy()
        a['cable_rest_lengths'] = result.cable_rest_lengths.cpu().numpy()
        a['cable_marker_indices'] = result.cable_marker_indices.cpu().numpy()
        a['node_relative_speed_mps'] = np.linalg.norm(a['cable_velocities']-a['cable_velocities'][:,:1],axis=-1)
        a['node_relative_speed_mps'][~a['sample_valid']] = np.nan
        case['arrays'] = a
        np.savez_compressed(output/case['name']/'measurements.npz', **a)
        good = a['sample_valid']; invalid = np.flatnonzero(~good)
        if not good.any():
            raise RuntimeError(f"{case['name']}: no valid measurement samples; artifacts retained")
        peak = int(np.nanargmax(a['tip_speed_mps']))
        window = (a['time_s'] <= OVERVIEW_END_S) & good
        error = np.linalg.norm(np.diff(a['cable_positions'],axis=1),axis=-1)-a['cable_rest_lengths']
        info = dict(name=case['name'], duration_s=float(a['time_s'][-1]), all_samples_valid=bool(good.all()),
                    first_invalid_time_s=float(a['time_s'][invalid[0]]) if len(invalid) else None,
                    full_peak_tip_speed_mps=float(a['tip_speed_mps'][peak]), full_peak_tip_time_s=float(a['time_s'][peak]),
                    overview_peak_tip_speed_mps=float(np.nanmax(a['tip_speed_mps'][window])),
                    overview_peak_relative_tip_speed_mps=float(np.nanmax(a['tip_relative_speed_mps'][window])),
                    peak_length_error_m=float(np.max(np.abs(error[good]))))
        summary['cases'].append(info)
        print(json.dumps(info), flush=True)
    plot_comparison(cases, output, float(config['controller']['command_delay_s']))
    write_json(output/'summary.json', summary)
    write_json(output/'artifacts.json', {str(p.relative_to(output)):digest(p) for p in output.rglob('*') if p.is_file()})
    print(f'Figures, measurements, source snapshots and replay CSV/scenes saved: {output}', flush=True)
    return output


if __name__ == '__main__':
    main()
