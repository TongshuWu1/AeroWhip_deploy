"""Run directly in PyCharm: compare sample and peak association, without RL."""

from dataclasses import asdict
from datetime import datetime, timezone
import json
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

from tasks.whip.peak_scoring import PeakWhipScorer, regional_peak_mask
from tasks.whip.scoring import WhipScorer
from tasks.whip.analysis import ARCHIVE_ROOT
from tasks.whip.analysis.run_whip_scoring import CONFIGURATION, describe, evaluate
from tasks.whip.analysis.run_whip_comparison import digest, write_json

BASELINE_RUN = ARCHIVE_ROOT/'runs/whip_scoring/20260921-164833-565507'


def compare(actual, expected):
    for key in ('score', 'bend_score', 'speed_score', 'quality_score', 'target_score',
                'target_distance_m', 'event_times_s', 'prominence', 'score_history'):
        value, reference = getattr(actual, key), getattr(expected, key)
        if value is not None:
            torch.testing.assert_close(value.cpu(), reference.cpu(), atol=1e-11,
                                       rtol=1e-11, equal_nan=True)
    for key in ('event_indices', 'valid', 'has_event'):
        torch.testing.assert_close(getattr(actual, key).cpu(), getattr(expected, key).cpu(),
                                   rtol=0, atol=0)


def plot(name, arrays, window, results, qualities, mask, output):
    time = arrays['time_s']
    end = window[1]+CONFIGURATION.tip_window_s+.05
    colors = ('#2676b8', '#ee9434', '#28a178')
    fig, axes = plt.subplots(3, 2, figsize=(12, 8), sharex=True, sharey='row', layout='constrained')
    for column, (mode, result) in enumerate(results.items()):
        prominence = result.prominence[0].cpu().numpy()
        for j, label in enumerate(('Proximal', 'Middle', 'Distal')):
            axes[0, column].plot(time, prominence[:, j], color=colors[j], label=label)
            if mode == 'peaks':
                indices = np.flatnonzero(mask[:, j] & (time >= window[0]) & (time <= window[1]+1e-12))
                axes[0, column].scatter(time[indices], prominence[indices, j], color=colors[j], s=10)
        axes[1, column].plot(time, arrays['tip_speed_mps'], color='#292929')
        axes[2, column].plot(time, qualities[mode].score_history[0].cpu(),
                            color='#777777', label='Quality B × V')
        if result.target_score is not None:
            axes[2, column].plot(time, result.score_history[0].cpu(),
                                color='#92278f', label='Target-weighted score')
        info = describe(result)
        if info['has_event']:
            for j, (index, color) in enumerate(zip(info['event_indices'], (*colors, '#92278f'))):
                for ax in axes[:, column]:
                    ax.axvline(time[index], color=color, linestyle='--', linewidth=1)
                if j < 3:
                    axes[0, column].scatter(time[index], prominence[index, j], color=color,
                                            edgecolors='black', s=45, zorder=5)
                else:
                    axes[1, column].scatter(time[index], arrays['tip_speed_mps'][index],
                                            color=color, s=35, zorder=5)
        axes[0, column].set_title(f"{mode.capitalize()} | score {info['score']:.6f}")
        axes[0, column].legend(ncol=3, fontsize=8)
        axes[2, column].legend(fontsize=8)
        for ax in axes[:, column]:
            ax.axvspan(window[1]+.01, end, color='#777777', alpha=.10)
            ax.set_xlim(max(0, window[0]-.05), end)
            ax.set_ylim(bottom=0)
        axes[2, column].set_xlabel('Time (s); gray = recovery/hold')
    axes[0, 0].set_ylabel('Bend increase (dimensionless)')
    axes[1, 0].set_ylabel('World tip speed (m/s)')
    axes[2, 0].set_ylabel('Score')
    visible = (time >= max(0, window[0]-.05)) & (time <= end)
    axes[0, 0].set_ylim(0, max(float(prominence[visible].max())*1.1, 1e-6))
    axes[1, 0].set_ylim(0, max(float(arrays['tip_speed_mps'][visible].max())*1.1, 1e-6))
    suffix = 'target-weighted event selection' if results['samples'].target_score is not None else 'quality only; no target'
    fig.suptitle(f'{name} | {suffix}\nDashed lines: selected P / M / D / tip; dots: peak candidates', fontsize=12)
    fig.savefig(output/f'{name}.png', dpi=150)
    plt.close(fig)


def main():
    manifest = json.loads((BASELINE_RUN/'artifacts.json').read_text(encoding='utf-8'))
    for name, expected in manifest.items():
        if digest(BASELINE_RUN/name) != expected:
            raise ValueError(f'Baseline artifact changed: {name}')
    baseline = json.loads((BASELINE_RUN/'summary.json').read_text(encoding='utf-8'))
    if json.loads(json.dumps(asdict(CONFIGURATION))) != baseline['configuration']:
        raise ValueError('Comparison must use the unchanged pilot configuration')
    source = ARCHIVE_ROOT/Path(baseline['source_run']).relative_to(ROOT)
    for name, expected in baseline['input_sha256'].items():
        if digest(source/name) != expected:
            raise ValueError(f'Input history/target changed: {name}')
    stamp = datetime.now(timezone.utc).strftime('%Y%m%d-%H%M%S-%f')
    output = ROOT/'runs/whip_peak_comparison'/stamp
    output.mkdir(parents=True, exist_ok=False)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f'Output: {output}', flush=True)
    source_hashes = {}
    for path in (Path(__file__), ROOT/'tasks/whip/scoring.py', ROOT/'tasks/whip/peak_scoring.py',
                 ROOT/'tasks/whip/analysis/run_whip_scoring.py',
                 ROOT/'tasks/whip/analysis/run_whip_comparison.py'):
        relative = path.relative_to(ROOT)
        destination = output/'source_snapshot'/relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(path, destination)
        source_hashes[str(relative)] = digest(path)
    summary = dict(schema='whip_peak_comparison_v1', baseline_run=str(BASELINE_RUN),
        baseline_summary_sha256=digest(BASELINE_RUN/'summary.json'), source_run=str(source),
        input_sha256=baseline['input_sha256'], source_sha256=source_hashes,
        configuration=asdict(CONFIGURATION), device=device, dtype='float64',
        versions=dict(torch=torch.__version__, numpy=np.__version__, matplotlib=matplotlib.__version__),
        protocol=baseline['protocol'],
        difference='Only P/M/D candidate eligibility changes: positive local maxima of the same '
                   'bend-increase signal. Strict rise and fall across any equal plateau; '
                   'first plateau sample selected; history endpoints excluded. Future samples '
                   'may confirm a peak outside the scoring window. No smoothing or threshold.',
        limitations='Peak timing is not continuous under perturbations. Ordered regional peaks '
                    'need not represent one travelling bend. Neither scorer is adopted for training.', cases=[])
    for case in baseline['cases']:
        name = case['name']; window = case['event_window_s']
        target = case['target_position_m']
        with np.load(source/name/'measurements.npz', allow_pickle=False) as saved:
            arrays = {key: saved[key] for key in saved.files}
        results, qualities = {}, {}
        for mode, scorer in (('samples', WhipScorer), ('peaks', PeakWhipScorer)):
            results[mode] = evaluate(arrays, window, target, device, scorer_class=scorer)
            qualities[mode] = evaluate(arrays, window, None, device, scorer_class=scorer)
            compare(results[mode], evaluate(arrays, window, target, 'cpu', scorer_class=scorer))
            compare(qualities[mode], evaluate(arrays, window, None, 'cpu', scorer_class=scorer))
            if not bool(results[mode].valid.all()):
                raise ValueError(f'{name}: invalid measurement history')
        with np.load(BASELINE_RUN/f'{name}.npz', allow_pickle=False) as saved:
            np.testing.assert_array_equal(results['samples'].score_history[0].cpu(), saved['score_history'])
            np.testing.assert_array_equal(qualities['samples'].score_history[0].cpu(), saved['quality_history'])
            np.testing.assert_array_equal(results['samples'].event_indices[0].cpu(), saved['event_indices'])
        mask = regional_peak_mask(results['peaks'].prominence)[0].cpu().numpy()
        np.testing.assert_array_equal(results['samples'].prominence.cpu(), results['peaks'].prominence.cpu())
        for variant in (results, qualities):
            if not bool((variant['peaks'].score_history <= variant['samples'].score_history+1e-12).all()):
                raise AssertionError('Restricting candidate tuples cannot increase the score')
            peak_result = variant['peaks']
            if bool(peak_result.has_event[0]):
                selected = peak_result.event_indices[0, :3].cpu().numpy()
                if not mask[selected, np.arange(3)].all():
                    raise AssertionError('Selected regional samples must be peak candidates')
        info = dict(name=name, event_window_s=window, target_position_m=target,
                    samples=dict(selected=describe(results['samples']), quality_only=describe(qualities['samples'])),
                    peaks=dict(selected=describe(results['peaks']), quality_only=describe(qualities['peaks'])))
        summary['cases'].append(info)
        arrays_out = dict(time_s=arrays['time_s'], prominence=results['samples'].prominence[0].cpu().numpy(),
                          peak_mask=mask)
        for mode in results:
            arrays_out[mode+'_score_history'] = results[mode].score_history[0].cpu().numpy()
            arrays_out[mode+'_quality_history'] = qualities[mode].score_history[0].cpu().numpy()
            arrays_out[mode+'_event_indices'] = results[mode].event_indices[0].cpu().numpy()
        np.savez_compressed(output/f'{name}.npz', **arrays_out)
        plot(name, arrays, window, results, qualities, mask, output)
        print(json.dumps(info, allow_nan=False), flush=True)
    summary['checks'] = ['Original score/quality histories and selected indices match retained baseline exactly',
                         'Both variants: CPU/device components and histories agree within 1e-11, indices exact',
                         'Peak-restricted history never exceeds unrestricted history',
                         'Every selected peak event uses eligible regional peaks']
    write_json(output/'summary.json', summary)
    write_json(output/'artifacts.json', {str(p.relative_to(output)):digest(p) for p in output.rglob('*') if p.is_file()})
    print(f'Comparison saved: {output}', flush=True)
    return output


if __name__ == '__main__':
    main()
