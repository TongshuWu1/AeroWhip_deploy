"""Run bend diagnostics on controlled geometry and existing flight records.

Run directly in PyCharm. This reads saved data; it does not simulate, fit a
model, change a reward, or train a policy. Source recordings are never edited.
"""
from datetime import datetime
from pathlib import Path
import hashlib
import json
import shutil
import sys

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from take_processing.take_data import load_take
from take_processing.baseline import load_baseline
from tasks.whip.analysis import ARCHIVE_ROOT
from tasks.whip.analysis.bend_tracking import measure_bends
from tasks.whip.analysis.synthetic_cases import controlled_cases

# Explicit saved reconstruction selection; no automatic model/run promotion.
SOURCE_RUN = ARCHIVE_ROOT / 'runs/residual_fit/20260921-000505/after'
TIP_LAG_S = .15  # Descriptive post-track inspection window, not a whip threshold.


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def serializable(value):
    if isinstance(value, np.ndarray):
        return serializable(value.tolist())
    if isinstance(value, (np.floating, float)):
        return float(value) if np.isfinite(value) else None
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.bool_):
        return bool(value)
    if isinstance(value, dict):
        return {key: serializable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [serializable(item) for item in value]
    return value


def write_json(path, value):
    path.write_text(json.dumps(serializable(value), indent=2, allow_nan=False) + '\n', encoding='utf-8')


def recorded_pairs():
    paths = sorted(SOURCE_RUN.glob('*_M[34]_*.json'))
    if not paths:
        raise FileNotFoundError(f'No saved M3/M4 reconstructions in {SOURCE_RUN}')
    for path in paths:
        meta = json.loads(path.read_text(encoding='utf-8'))
        take = load_take(Path(meta['take']))
        baseline = load_baseline(path, take)  # Verifies sources and exact saved measured targets.
        start = int(np.flatnonzero(take.frame == meta['start_frame'])[0])
        stop = start + len(baseline.time)
        segment = json.loads(Path(meta['sources']['segment']['path']).read_text(encoding='utf-8'))
        time = take.source_time[start:stop] + meta['alignment']['offset_s'] - segment['onset_s']
        if not np.allclose(np.diff(time), .01, rtol=0, atol=1e-8):
            raise ValueError(f'Non-native 100-Hz window: {path}')
        material_s = np.cumsum(meta['model_config']['cable']['marker_interval_lengths_m'])
        common = dict(time_s=time, material_s_m=material_s,
                      description='Native c1..c10 positions; no inferred attachment or spatial upsampling.')
        cases = [dict(common, name='recorded', title='Recorded', positions_m=take.cable[start:stop]),
                 dict(common, name='saved_simulation', title='Saved simulation', positions_m=baseline.markers)]
        provenance = dict(take=take.path.stem, role=meta['role'], source_metadata=str(path),
                          source_metadata_sha256=digest(path), source_arrays_sha256=digest(path.with_suffix('.npz')),
                          sources=meta['sources'], model_sha256=meta['model_sha256'],
                          source_hashes_verified=True, saved_measured_targets_equal_raw_recording=True,
                          clock_verified=meta['alignment']['clock_verified'],
                          time_convention='Native recording time + saved estimated offset - command onset; no additional delay.',
                          samples=len(time), time_window_s=time[[0, -1]])
        yield take.path.stem, cases, provenance


def measure(case):
    return measure_bends(case['time_s'], case['positions_m'], case['material_s_m'], tip_lag_s=TIP_LAG_S)


def summary(result):
    tracks = result['tracks']
    linked = [track for track in tracks if track['resolved_link_count']]
    return dict(samples=len(result['time_s']), valid_samples=int(result['sample_valid'].sum()),
                tracks_including_singletons=len(tracks), tracks_with_links=len(linked),
                ambiguous_frames=int(result['ambiguity_by_frame'].sum()),
                boundary_censored_features=sum(feature['boundary_censored']
                    for frame in result['features_by_frame'] for feature in frame),
                max_abs_track_travel_m=max((abs(t['net_travel_m']) for t in linked), default=None),
                max_track_excess_turn_rad=max((float(t['excess_turn_rad'].max()) for t in tracks), default=None))


def save_measurements(folder, case, result):
    folder.mkdir()
    np.savez_compressed(folder / 'measurements.npz', positions_m=case['positions_m'],
        input_material_s_m=case['material_s_m'], **{k: v for k, v in result.items() if isinstance(v, np.ndarray)})
    write_json(folder / 'tracks.json', dict(description=case['description'], metadata=result['metadata'],
        summary=summary(result), features_by_frame=result['features_by_frame'], tracks=result['tracks']))


def motion_panel(ax, case, bounds):
    positions, time = case['positions_m'], case['time_s']
    finite = np.isfinite(positions).all((1, 2))
    ids = np.flatnonzero(finite)
    if not len(ids):
        return
    selected = ids[np.unique(np.linspace(0, len(ids) - 1, min(9, len(ids))).round().astype(int))]
    norm = plt.Normalize(time[0], time[-1])
    for index in selected:
        p = positions[index]
        color = plt.cm.viridis(norm(time[index]))
        ax.plot(p[:, 0], p[:, 1], p[:, 2], color=color, lw=1.2)
        ax.scatter(*p[-1], color=color, s=16)
    low, high = bounds
    center, radius = (low + high) / 2, max(float(np.max(high - low)) / 2, .05)
    ax.set(xlim=(center[0] - radius, center[0] + radius), ylim=(center[1] - radius, center[1] + radius),
           zlim=(center[2] - radius, center[2] + radius), xlabel='X (m)', ylabel='Y (m)', zlabel='Z (m)',
           title=case['title'] + '\nTime-coloured cable snapshots')
    ax.set_box_aspect((1, 1, 1))
    ax.view_init(elev=22, azim=-65)
    return plt.cm.ScalarMappable(norm=norm, cmap='viridis')


def heat_panel(ax, case, result, maximum):
    im = ax.pcolormesh(result['time_s'], result['material_s_m'], result['curvature_per_m'].T,
                      shading='nearest', cmap='magma', vmin=0, vmax=maximum)
    for track in result['tracks']:
        ax.plot(track['time_s'], track['peak_s_m'], color='cyan', lw=.8,
                marker='.' if len(track['time_s']) == 1 else None, ms=2)
    ax.set(xlabel='Time (s)', ylabel='Material coordinate (m)', title='Curvature and resolved track fragments',
           ylim=(case['material_s_m'][0], case['material_s_m'][-1]))
    return im


def strength_panel(ax, result):
    for track in result['tracks']:
        ax.plot(track['time_s'], track['excess_turn_rad'], lw=1,
                marker='.' if len(track['time_s']) == 1 else None, ms=2)
    ax.set(xlabel='Time (s)', ylabel='Excess turning angle (rad)', title='Integrated local bend strength', ylim=(0, None))
    ax.grid(alpha=.15)


def plot_cases(path, title, cases, results, limits=None):
    maximum = max(float(np.nanmax(r['curvature_per_m'])) for r in results)
    maximum = max(maximum, max(float(np.nanmax(r['numerical_floor_per_m'])) for r in results))
    if limits is not None:
        maximum = limits['curvature']
    fig = plt.figure(figsize=(19, 4.9 * len(cases)), layout='constrained')
    grid = fig.add_gridspec(len(cases), 4, width_ratios=[1.15, 1.35, 1, 1])
    xyz = np.concatenate([case['positions_m'][np.isfinite(case['positions_m']).all((1, 2))].reshape(-1, 3)
                          for case in cases])
    bounds = xyz.min(0), xyz.max(0)
    heat_axes, strength_axes, speed_axes = [], [], []
    for row, (case, result) in enumerate(zip(cases, results)):
        ax = fig.add_subplot(grid[row, 0], projection='3d')
        colors = motion_panel(ax, case, bounds)
        if colors is not None:
            fig.colorbar(colors, ax=ax, label='Time (s)', orientation='horizontal', shrink=.65, pad=.10)
        heat_axes.append(fig.add_subplot(grid[row, 1]))
        im = heat_panel(heat_axes[-1], case, result, maximum)
        strength_axes.append(fig.add_subplot(grid[row, 2]))
        strength_panel(strength_axes[-1], result)
        speed_axes.append(fig.add_subplot(grid[row, 3]))
        speed_axes[-1].plot(result['time_s'], result['tip_speed_mps'], color='#146a9e')
        speed_axes[-1].set(xlabel='Time (s)', ylabel='Tip speed (m/s)', title='Position-derived world tip speed', ylim=(0, None))
        speed_axes[-1].grid(alpha=.15)
        for axis in (heat_axes[-1], strength_axes[-1], speed_axes[-1]):
            axis.set_xlim(case['time_s'][0], case['time_s'][-1])
    # Paired recordings/simulations use identical axes and full-range colour scales.
    for key, axes in (('strength', strength_axes), ('speed', speed_axes)):
        top = limits[key] if limits is not None else max(ax.get_ylim()[1] for ax in axes)
        for ax in axes:
            ax.set_ylim(0, top)
    fig.colorbar(im, ax=heat_axes, label='Curvature (1/m)', orientation='horizontal', shrink=.7, pad=.10)
    fig.suptitle(title + '\nDescriptive measurements; track fragments are not whip labels', fontsize=13)
    fig.savefig(path, dpi=145)
    plt.close(fig)


def check_controlled(cases, results):
    """Finite analytic checks for this diagnostic, not physical classification."""
    by_name = {c['name']: (c, r) for c, r in zip(cases, results)}
    checks = {}

    def check(name, condition, measured):
        checks[name] = dict(passed=bool(condition), measured=measured)
        if not condition:
            raise AssertionError(f'Controlled geometry check failed: {name}: {measured}')

    edge_error = max(float(np.nanmax(np.abs(np.linalg.norm(np.diff(c['positions_m'], axis=1), axis=-1)
                                           - np.diff(c['material_s_m'])))) for c in cases)
    check('inextensible_geometry_1e-12_m', edge_error < 1e-12, edge_error)
    check('straight_rigid_swing_no_tracks', len(by_name['straight_rigid_swing'][1]['tracks']) == 0,
          len(by_name['straight_rigid_swing'][1]['tracks']))
    for name, count in (('curved_rigid_swing', 1), ('stationary_strength_change', 1),
                        ('separated_lobes_swap', 2), ('disjoint_reappearance', 2)):
        tracks = by_name[name][1]['tracks']
        variation = max((float(np.ptp(t['peak_s_m'])) for t in tracks), default=float('inf'))
        check(name + '_stationary_tracks_1e-9_m', len(tracks) == count and variation < 1e-9,
              dict(tracks=len(tracks), maximum_position_variation_m=variation))
    fast = by_name['travelling_bend_fast'][1]
    slow = by_name['travelling_bend_slow'][1]
    tiny = by_name['tiny_travelling_ripple'][1]
    for name, result, rate, strength in (('fast', fast, .4, 1.), ('slow', slow, .2, 1.), ('tiny', tiny, .4, .001)):
        tracks = result['tracks']
        check(name + '_single_full_track', len(tracks) == 1 and len(tracks[0]['time_s']) == 101,
              [len(t['time_s']) for t in tracks])
        track = tracks[0]
        # Centers at both endpoints coincide with material samples; integrated
        # turning is exact for the chosen compact tangent construction.
        check(name + '_travel_rate_strength_1e-8',
              abs(track['net_travel_m'] - .4) < 1e-8 and abs(track['mean_rate_mps'] - rate) < 1e-8
              and np.max(np.abs(track['excess_turn_rad'] - strength)) < 1e-8,
              dict(travel_m=track['net_travel_m'], rate_mps=track['mean_rate_mps'],
                   maximum_strength_error_rad=float(np.max(np.abs(track['excess_turn_rad'] - strength)))))
    speed_error = float(np.max(np.abs(fast['tip_velocity_mps'] - 2 * slow['tip_velocity_mps'])))
    check('time_doubling_velocity_halves_1e-10_mps', speed_error < 1e-10, speed_error)
    missing = by_name['missing_frame'][1]
    tracks = missing['tracks']
    check('missing_frame_not_bridged', len(tracks) == 2 and not missing['sample_valid'][50]
          and np.isnan(missing['tip_velocity_mps'][50]).all()
          and all(not (t['frame_indices'][0] < 50 < t['frame_indices'][-1]) for t in tracks),
          [t['frame_indices'][[0, -1]] for t in tracks])
    overlap = by_name['overlapping_lobes_swap'][1]
    checks['overlap_interpretation'] = dict(observed_travel_m=[t['net_travel_m'] for t in overlap['tracks']],
        interpretation='Combined peak progression is allowed; no claim about causal wave identity.')
    return checks


def main():
    output = ROOT / 'runs/bend_analysis' / datetime.now().strftime('%Y%m%d-%H%M%S-%f')
    output.mkdir(parents=True, exist_ok=False)
    print(f'Bend analysis: {output}', flush=True)
    sources = [Path(__file__), Path(__file__).with_name('bend_tracking.py'),
               Path(__file__).with_name('synthetic_cases.py'), ROOT / 'take_processing/take_data.py',
               ROOT / 'take_processing/baseline.py', ROOT / 'take_processing/library.py']
    source_folder = output / 'sources'
    source_folder.mkdir()
    for path in sources:
        shutil.copyfile(path, source_folder / path.name)
    protocol = dict(scope='Analysis only. No dynamics rollout, fitting, reward change or training.',
        source_run=str(SOURCE_RUN), tip_lag_s=TIP_LAG_S, python=sys.version, numpy=np.__version__,
        matplotlib=matplotlib.__version__, source_hashes={str(p.relative_to(ROOT)): digest(p) for p in sources},
        controlled_cases='Prescribed inextensible geometry, not dynamically validated commands or ground-truth whips.',
        recorded_cases='Every available M3/M4 pair in the explicitly selected saved residual reconstruction run.',
        limitations=['No take, including M3, is a known positive whip label.',
            'Historical training/comparison roles do not imply a new independently held-out evaluation.',
            'Ten-marker chord curvature cannot resolve every narrow bend; no resolved track does not mean non-whip.',
            'Sparse samples, noise and local maxima can fragment tracking; no smoothing or physical amplitude threshold.',
            'Tip speed association and bend propagation do not prove causal energy transfer.',
            'Saved clock offsets are estimated; no claim of hardware-verified time alignment.'])
    write_json(output / 'protocol.json', protocol)
    report = dict(controlled=[], recordings=[])
    controlled = controlled_cases()
    control_results = [measure(case) for case in controlled]
    write_json(output / 'controlled_checks.json', check_controlled(controlled, control_results))
    limits = dict(curvature=max(float(np.nanmax(r['curvature_per_m'])) for r in control_results),
                  strength=1.05 * max(float(t['excess_turn_rad'].max()) for r in control_results for t in r['tracks']),
                  speed=1.05 * max(float(np.nanmax(r['tip_speed_mps'])) for r in control_results))
    for case, result in zip(controlled, control_results):
        save_measurements(output / case['name'], case, result)
        plot_cases(output / case['name'] / 'diagnostics.png', case['title'] + ' (shared controlled-case scales)', [case], [result], limits)
        report['controlled'].append(dict(name=case['name'], expected=case['expected'], **summary(result)))
        print(f"Controlled: {case['name']} ({len(result['tracks'])} fragments)", flush=True)
    for name, cases, provenance in recorded_pairs():
        folder = output / name
        folder.mkdir()
        results = [measure(case) for case in cases]
        for case, result in zip(cases, results):
            save_measurements(folder / case['name'], case, result)
        write_json(folder / 'provenance.json', provenance)
        plot_cases(folder / 'diagnostics.png', name, cases, results)
        report['recordings'].append(dict(take=name, role=provenance['role'],
            **{case['name']: summary(result) for case, result in zip(cases, results)}))
        print(f'Recorded/saved comparison: {name}', flush=True)
    write_json(output / 'summary.json', report)
    write_json(output / 'manifest.json', {str(p.relative_to(output)): digest(p)
        for p in sorted(output.rglob('*')) if p.is_file()})
    print(f'Completed: {len(controlled)} controlled cases and {len(report["recordings"])} recorded pairs.\n{output}', flush=True)


if __name__ == '__main__':
    main()
