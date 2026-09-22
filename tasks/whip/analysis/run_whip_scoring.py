"""Run in PyCharm: score the retained comparison; no simulation or training."""

from dataclasses import asdict
from datetime import datetime, timezone
import json
import math
from pathlib import Path
import shutil
import sys

import numpy as np
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[3]
if __package__ in (None, ''):
    sys.path.insert(0, str(ROOT))

from tasks.whip.scoring import WhipScoreConfiguration, WhipScorer
from tasks.whip.analysis import ARCHIVE_ROOT
from tasks.whip.analysis.run_whip_comparison import digest, heatmap, write_json

# Explicit pilot settings; editing these does not alter the physical model.
SOURCE_RUN = ARCHIVE_ROOT/'runs/whip_comparison/20260921-062943-373533'
CONFIGURATION = WhipScoreConfiguration()


def evaluate(arrays, window, target, device, *, scorer_class=WhipScorer):
    tensor = lambda x: torch.as_tensor(x, device=device, dtype=torch.float64)
    length = arrays['cable_rest_lengths'].sum()
    np.testing.assert_allclose(arrays['region_boundaries_m']/length,
                               CONFIGURATION.region_fractions, atol=1e-12, rtol=0)
    scorer = scorer_class(tensor(arrays['time_s']), CONFIGURATION)
    return scorer(tensor(arrays['region_rms_curvature_per_m']*length)[None],
        tensor(arrays['cable_positions'][:, -1])[None],
        tensor(arrays['cable_velocities'][:, -1])[None],
        torch.as_tensor(arrays['sample_valid'][None], device=device), tensor([window]),
        None if target is None else tensor(target)[None])


def describe(result):
    def scalar(value):
        if value is None:
            return None
        number = float(value[0])
        return number if math.isfinite(number) else None
    return dict(valid=bool(result.valid[0]), has_event=bool(result.has_event[0]),
        score=scalar(result.score), bend_score=scalar(result.bend_score),
        speed_score=scalar(result.speed_score), quality_score=scalar(result.quality_score),
        target_score=scalar(result.target_score), target_distance_m=scalar(result.target_distance_m),
        event_times_s=[float(t) if math.isfinite(float(t)) else None for t in result.event_times_s[0]],
        event_indices=result.event_indices[0].cpu().tolist())


def plot(case, result, quality_result, window, target, output):
    a = case['arrays']; t = a['time_s']
    end = min(float(t[-1]), 3.5)
    colors = ('#2676b8', '#ee9434', '#28a178', '#92278f')
    labels = ('Proximal', 'Middle', 'Distal', 'Tip event')
    fig, axes = plt.subplots(4, 1, figsize=(11, 10), sharex=True, layout='constrained')
    image = heatmap(axes[0], case, 'curvature_per_m', end=end,
                    vmax=max(float(np.nanmax(a['curvature_per_m'][t <= end])), 1e-12))
    fig.colorbar(image, ax=axes[0], label='Turning density (1/m)')
    axes[0].set_ylabel('Material s (m)')
    prominence = result.prominence[0].cpu().numpy()
    for j in range(3):
        axes[1].plot(t, prominence[:, j], color=colors[j], label=labels[j])
    axes[1].set_ylabel('Bend increase (L × curvature)')
    axes[1].legend(ncol=3, loc='upper right')
    axes[2].plot(t, a['tip_speed_mps'], color='#27272a', label='World tip speed')
    axes[2].set_ylabel('Tip speed (m/s)')
    axes[2].legend(loc='upper left')
    if target is not None:
        right = axes[2].twinx()
        right.plot(t, np.linalg.norm(a['cable_positions'][:, -1]-target, axis=-1),
                   color='#777777', alpha=.65, label='Target distance')
        right.set_ylabel('Target distance (m)')
        right.legend(loc='upper right')
    axes[3].plot(t, quality_result.score_history[0].cpu(), color='#777777',
                 label='Quality B × V (best event at each time)')
    if target is not None:
        axes[3].plot(t, result.score_history[0].cpu(), color='#92278f', label='Target-weighted score')
    axes[3].set_ylabel('Score'); axes[3].set_ylim(0, 1)
    axes[3].legend(loc='upper right', fontsize=8)
    info = describe(result)
    if info['has_event']:
        for j, event_time in enumerate(info['event_times_s']):
            for ax in axes:
                ax.axvline(event_time, color=colors[j], linewidth=1, linestyle='--', alpha=.8)
            if j < 3:
                index = info['event_indices'][j]
                axes[1].scatter(event_time, prominence[index, j], color=colors[j], s=35, zorder=5)
        axes[2].scatter(info['event_times_s'][3], a['tip_speed_mps'][info['event_indices'][3]],
                        color=colors[3], s=35, zorder=5)
    for ax in axes:
        ax.axvspan(window[0], window[1], color='#387ca3', alpha=.06)
        ax.axvspan(window[1]+.01, end, color='#777777', alpha=.09)
        ax.set_xlim(0, end)
    axes[-1].set_xlabel('Time (s) — blue: delayed maneuver; gray: recovery/hold')
    target_text = 'target-weighted selection' if target is not None else 'quality only; no target'
    fig.suptitle(f"{case['name']} | {target_text} | score {info['score']:.4f}\n"
                 'Dashed lines: selected P / M / D / tip samples; not a tracked travelling bend', fontsize=12)
    fig.savefig(output/f"{case['name']}.png", dpi=150)
    plt.close(fig)


def main():
    protocol = json.loads((SOURCE_RUN/'protocol.json').read_text(encoding='utf-8'))
    artifacts = json.loads((SOURCE_RUN/'artifacts.json').read_text(encoding='utf-8'))
    artifacts = {key.replace('\\', '/'): value for key, value in artifacts.items()}
    input_hashes = {}
    # Verify retained inputs before analysis, including the archived targets.
    for relative in ['protocol.json', 'model.json'] + [f'{n}/{f}' for n in
            ('M3', 'M4', 'M5', 'M6', 'Gentle') for f in ('measurements.npz', 'source.npz')]:
        actual = digest(SOURCE_RUN/relative)
        if actual != artifacts[relative]:
            raise ValueError(f'Retained artifact hash mismatch: {relative}')
        input_hashes[relative] = actual
    delay = json.loads((SOURCE_RUN/'model.json').read_text(encoding='utf-8'))['controller']['command_delay_s']
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    stamp = datetime.now(timezone.utc).strftime('%Y%m%d-%H%M%S-%f')
    output = ROOT/'runs/whip_scoring'/stamp
    output.mkdir(parents=True, exist_ok=False)
    print(f'Output: {output}', flush=True)
    source_hashes = {}
    for path in (Path(__file__), ROOT/'tasks/whip/scoring.py',
                 ROOT/'tasks/whip/analysis/run_whip_comparison.py'):
        relative = path.relative_to(ROOT)
        destination = output/'source_snapshot'/relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(path, destination)
        source_hashes[str(relative)] = digest(path)
    summary = dict(schema='whip_scoring_pilot_v1', source_run=str(SOURCE_RUN),
        configuration=asdict(CONFIGURATION), device=device, dtype='float64',
        input_sha256=input_hashes, source_sha256=source_hashes,
        protocol='Regional events inside delayed phase-1 whip maneuver (Gentle: move). '
                 'Last regional sample is one 100-Hz step before recovery. '
                 'Tip window may extend 0.15 s beyond that sample. Baselines may precede maneuver. '
                 'Discrete 100-Hz target distance and speed at the SAME sample; no interpolation. '
                 'No command penalties, success labels, collision, physics rerun or training.', cases=[])
    for entry in protocol['cases']:
        name = entry['name']
        with np.load(SOURCE_RUN/name/'measurements.npz', allow_pickle=False) as source:
            a = {key: source[key] for key in source.files}
        if name == 'Gentle':
            start, stop = 1., 3.
            target = None
        else:
            start = 0.
            stop = next(v['time_s'] for v in entry['transitions'] if v['phase'] == 2)
            with np.load(SOURCE_RUN/name/'source.npz', allow_pickle=False) as source:
                target = source['target_position_m'].copy()
                phase = source['command_phase']
                source_time = source['command_time_s']
                if not (np.all(phase[source_time < stop] == 1)
                        and phase[np.searchsorted(source_time, stop)] == 2):
                    raise ValueError(f'{name}: unexpected archived whip/braking phase boundary')
            if target.shape != (3,) or not np.isfinite(target).all():
                raise ValueError(f'{name}: expected one finite archived world-XYZ target')
        window = [math.ceil((start-1e-12)/.01)*.01+delay,
                  math.ceil((stop-1e-12)/.01)*.01+delay-.01]
        result = evaluate(a, window, target, device)
        quality = evaluate(a, window, None, device) if target is not None else result
        cpu = evaluate(a, window, target, 'cpu')
        for key in ('score', 'bend_score', 'speed_score', 'quality_score', 'target_score',
                    'target_distance_m', 'event_times_s', 'prominence', 'score_history'):
            value, expected = getattr(result, key), getattr(cpu, key)
            if value is not None:
                torch.testing.assert_close(value.cpu(), expected, atol=1e-11, rtol=1e-11, equal_nan=True)
        torch.testing.assert_close(result.event_indices.cpu(), cpu.event_indices, rtol=0, atol=0)
        if not bool(result.valid.all()):
            raise RuntimeError(f'{name}: invalid history; inspect inputs instead of treating it as a score')
        info = dict(name=name, event_window_s=window, target_position_m=None if target is None else target.tolist(),
                    selected=describe(result), quality_only=describe(quality))
        summary['cases'].append(info)
        np.savez_compressed(output/f'{name}.npz', time_s=a['time_s'],
            prominence=result.prominence[0].cpu().numpy(),
            score_history=result.score_history[0].cpu().numpy(),
            quality_history=quality.score_history[0].cpu().numpy(),
            event_indices=result.event_indices[0].cpu().numpy())
        plot(dict(name=name, arrays=a), result, quality, window, target, output)
        print(json.dumps(info, allow_nan=False), flush=True)
    summary['cpu_device_agreement'] = 'All five scores, components and histories within 1e-11; selected indices exact'
    write_json(output/'summary.json', summary)
    write_json(output/'artifacts.json', {str(p.relative_to(output)):digest(p) for p in output.rglob('*') if p.is_file()})
    print(f'Saved scores and event plots: {output}', flush=True)
    return output


if __name__ == '__main__':
    main()
