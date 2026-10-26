"""Offline event measurements on pinned replays; no reward or whip classifier.

python -m evaluation.whip_event_review --manifest inputs.json --output new-folder

Automatic intervals are speed-peak basins, not identified physical mechanisms.
Explicit sample intervals in the manifest allow review of alternative boundaries.
Optional release intervals carry sampled output speed and manual review provenance.
Automatic associations are temporal evidence, never causal claims.
"""
import argparse
import csv
import json
import math
from pathlib import Path
import shlex
import sys
import tomllib

import numpy as np

from evaluation.whip_measurements import extract_measurements, track_curvature
from viewer.replay_data import sha256


SCHEMA = 'offline_whip_event_review_v1'
INPUT_SCHEMA = 'offline_whip_event_inputs_v1'


def review_surface(initial_position_world_m):
    """Virtual review board 1.1 m along world +X from the initial drone position.

    Measurement geometry only (formerly stored in the Stage 1 task file): no
    contact, reward or success definition. Units are metres; normal is world +X.
    """
    x, y = (float(v) for v in initial_position_world_m[:2])
    return dict(center_world_m=[x+1.1, y, 2.], normal_world=[1., 0., 0.], width_m=1., height_m=.6,
                maximum_entry_angle_deg=30., minimum_forward_speed_mps=3.)


def snapshot_geometry(snapshot):
    """Use the frozen source mesh and the review board, not live defaults."""
    model = snapshot['model']
    lengths = np.asarray(model['cable']['marker_interval_lengths_m'], dtype=float)
    counts = np.asarray(model['simulation']['segments_per_marker_interval'])
    if (lengths.ndim != 1 or counts.shape != lengths.shape or not np.isfinite(counts).all()
            or (counts < 1).any() or (counts != np.floor(counts)).any()):
        raise ValueError('Frozen mesh requires positive integer subdivisions per marker interval')
    rest = np.concatenate([np.full(int(n), length/n) for length, n in zip(lengths, counts)])
    initial = tomllib.loads(snapshot['scene_text'])['initial_state']['position_world_m']
    return rest, review_surface(initial)


def _extrema(values, maximum):
    """Retain plateau extents so quiet time cannot extend a preceding event."""
    values = np.asarray(values)
    eps = 64*np.finfo(float).eps*max(1., float(np.max(np.abs(values))))
    starts = np.r_[0, np.flatnonzero(np.abs(np.diff(values)) > eps)+1]
    ends = np.r_[starts[1:]-1, len(values)-1]
    sign = 1 if maximum else -1
    found = []
    for start, end in zip(starts, ends):
        value = sign*float(np.mean(values[start:end+1]))
        left = start == 0 or value > sign*values[start-1]+eps
        right = end == len(values)-1 or value > sign*values[end+1]+eps
        if left and right and not (start == 0 and end == len(values)-1):
            found.append((int(start), int(end)))
    return found


def candidate_windows(measurements):
    """All local relative-tip speed peaks; adjacent troughs bound each interval.

    No physical amplitude, duration or success threshold is applied. The full
    recording is required: these are offline, future-dependent review intervals.
    """
    speed = measurements['relative_speed_mps'][:, -1]
    minima = _extrema(speed, False)
    windows = []
    for first, last in _extrema(speed, True):
        peak = (first+last)//2
        windows.append(dict(start_index=max([0]+[end for _, end in minima if end < first]),
                            peak_index=peak,
                            end_index=min([len(speed)-1]+[start for start, _ in minima if start > last])))
    return windows


def _vector_metrics(velocity, attachment_velocity, normal):
    relative = velocity-attachment_velocity
    speed = float(np.linalg.norm(velocity))
    forward = float(velocity@normal)
    return dict(velocity_world_mps=velocity.tolist(),
                attachment_velocity_world_mps=attachment_velocity.tolist(),
                velocity_relative_mps=relative.tolist(), speed_world_mps=speed,
                speed_relative_mps=float(np.linalg.norm(relative)),
                forward_world_mps=forward, forward_relative_mps=float(relative@normal),
                entry_angle_deg=None if speed == 0 else float(np.degrees(np.arctan2(
                    np.linalg.norm(np.cross(velocity, normal)), forward))))


def _point(arrays, index, fraction, surface):
    p, v, t = (arrays[k] for k in ('cable_positions', 'cable_velocities', 'time_s'))
    position = p[index, -1]+fraction*(p[index+1, -1]-p[index, -1])
    velocity = v[index, -1]+fraction*(v[index+1, -1]-v[index, -1])
    attachment = v[index, 0]+fraction*(v[index+1, 0]-v[index, 0])
    normal, center = np.asarray(surface['normal_world']), np.asarray(surface['center_world_m'])
    delta = position-center
    side = np.array([-normal[1], normal[0], 0.])
    value = _vector_metrics(velocity, attachment, normal)
    value.update(interval_index=int(index), fraction=float(fraction),
                 time_s=float(t[index]+fraction*(t[index+1]-t[index])),
                 position_world_m=position.tolist(), center_distance_m=float(np.linalg.norm(delta)),
                 lateral_error_m=float(delta@side), vertical_error_m=float(delta[2]))
    return value


def measure_crossings(arrays, surface):
    tip = arrays['cable_positions'][:, -1]
    normal, center = np.asarray(surface['normal_world']), np.asarray(surface['center_world_m'])
    signed = (tip-center)@normal
    crossings = []
    for i, (old, new) in enumerate(zip(signed[:-1], signed[1:])):
        if not (old < 0 <= new or old > 0 >= new):
            continue
        confirmation = i+1
        if new == 0:
            # Reaching the plane and returning to the same side is a touch, not
            # an established passage. A recording ending on the plane is unresolved.
            while confirmation < len(signed) and signed[confirmation] == 0:
                confirmation += 1
            if confirmation == len(signed) or old*signed[confirmation] >= 0:
                continue
        point = _point(arrays, i, float(-old/(new-old)), surface)
        eps = 32*np.finfo(float).eps*max(1., float(np.max(np.abs(tip[i:i+2]))),
                                        float(np.max(np.abs(center))))
        point.update(id=len(crossings), passage='forward' if new > old else 'backward',
                     confirmed_at_s=float(arrays['time_s'][confirmation]),
                     inside_target_rectangle=bool(abs(point['lateral_error_m']) <= surface['width_m']/2+eps
                                                   and abs(point['vertical_error_m']) <= surface['height_m']/2+eps))
        crossings.append(point)
    return crossings


def _closest(arrays, start, end, surface):
    tip = arrays['cable_positions'][:, -1]
    center = np.asarray(surface['center_world_m'])
    candidates = []
    for i in range(start, end):
        delta = tip[i+1]-tip[i]
        norm = float(delta@delta)
        fraction = 0. if norm == 0 else float(np.clip((center-tip[i])@delta/norm, 0, 1))
        candidates.append(_point(arrays, i, fraction, surface))
    return min(candidates, key=lambda point: point['center_distance_m'])


def _motion_distribution(measurements, rest):
    # Length-weighted squared relative speed describes spatial kinematics.
    # It is deliberately NOT called kinetic energy: marker masses are absent.
    weights = np.r_[rest[0], rest[:-1]+rest[1:], rest[-1]]/2
    q = measurements['relative_speed_mps']**2*weights
    total = q.sum(axis=1)
    present = total > (64*np.finfo(float).eps)**2*weights.sum()
    center = np.full(len(total), np.nan)
    spread = np.full(len(total), np.nan)
    center[present] = (q[present]*measurements['node_s_m']).sum(axis=1)/total[present]
    spread[present] = np.sqrt((q[present]*(measurements['node_s_m']-center[present, None])**2
                              ).sum(axis=1)/total[present])
    return center, spread


def _number(value):
    return float(value) if np.isfinite(value) else None


def _annotation_fields(value, fields, label):
    if not isinstance(value, dict) or set(value) != set(fields):
        raise ValueError(f'{label} requires exactly: '+', '.join(fields))
    return value


def _review_provenance(value):
    value = _annotation_fields(value, ('reviewer', 'method', 'notes'), 'Review provenance')
    if any(not isinstance(item, str) or not item.strip() for item in value.values()):
        raise ValueError('Review provenance requires nonempty reviewer, method and notes strings')
    # Preserve the supplied attribution; the evaluator does not verify its truth.
    return dict(value)


def _event_core(event, window, arrays, measurements, crossings, normal):
    """Add explicit release measurements and separately attributed manual review.

    Bounds must be supplied independently of this calculation. This function
    neither finds a physical release nor certifies the reviewer's interpretation.
    Q is the largest saved attachment-relative tip speed inside the inclusive
    release bounds; exact ties use the earliest sample. The baseline is one raw
    prerelease sample, and signed gain is Q minus that sample, without clipping.
    """
    event['mechanism_review'] = dict(status='unresolved', provenance=None)
    delivery = event['delivery']
    delivery.update(release_crossing_ids=[], reviewed_linked_crossing_ids=[],
                    review_provenance=None, link_status='unresolved', linked_quality=[])
    if 'release_interval' not in window:
        if 'mechanism_review' in window or 'delivery_review' in window:
            raise ValueError('Mechanism and delivery review require an explicit release_interval')
        return
    release = _annotation_fields(window['release_interval'],
        ('start_index', 'end_index', 'baseline_index', 'provenance'), 'release_interval')
    start, end, baseline = (release[key] for key in ('start_index', 'end_index', 'baseline_index'))
    if (any(type(i) is not int for i in (start, end, baseline))
            or not event['start_index'] <= baseline < start <= end <= event['end_index']):
        raise ValueError('Release needs integer event_start <= baseline < release_start <= release_end <= event_end')
    provenance = _review_provenance(release['provenance'])
    time, speed = measurements['time_s'], measurements['relative_speed_mps'][:, -1]
    peak = start+int(np.argmax(speed[start:end+1]))
    event['release_measurement'] = dict(start_index=start, end_index=end,
        start_s=float(time[start]), end_s=float(time[end]), baseline_index=baseline,
        baseline_time_s=float(time[baseline]), baseline_relative_tip_speed_mps=float(speed[baseline]),
        q_relative_tip_speed_mps=float(speed[peak]), q_index=peak, q_time_s=float(time[peak]),
        signed_gain_mps=float(speed[peak]-speed[baseline]),
        peak_velocity=_vector_metrics(arrays['cable_velocities'][peak, -1],
                                      arrays['cable_velocities'][peak, 0], normal),
        selection_provenance=provenance)
    event['notes'].append('Legacy deformation and response fields still use candidate start..peak, '
        'not the explicit release or its Q sample. Inspect the full traces for release context.')
    if 'mechanism_review' in window:
        review = _annotation_fields(window['mechanism_review'],
                                    ('status', 'provenance'), 'mechanism_review')
        if review['status'] not in ('supported', 'unsupported', 'unresolved'):
            raise ValueError('Mechanism review status must be supported, unsupported or unresolved')
        event['mechanism_review'] = dict(status=review['status'],
                                         provenance=_review_provenance(review['provenance']))
    release_crossings = [crossing for crossing in crossings
        if time[start] <= crossing['time_s'] <= crossing['confirmed_at_s'] <= time[end]]
    delivery['release_crossing_ids'] = [crossing['id'] for crossing in release_crossings]
    if 'delivery_review' in window:
        review = _annotation_fields(window['delivery_review'],
            ('linked_crossing_ids', 'provenance'), 'delivery_review')
        ids = review['linked_crossing_ids']
        if (not isinstance(ids, list) or not ids or any(type(i) is not int for i in ids)
                or len(set(ids)) != len(ids)):
            raise ValueError('Delivery review needs a nonempty list of unique integer linked_crossing_ids')
        if event['mechanism_review']['status'] != 'supported':
            raise ValueError('Reviewed delivery links require a supported manual mechanism review')
        by_id = {crossing['id']: crossing for crossing in release_crossings}
        if any(i not in by_id for i in ids):
            raise ValueError('Linked crossing arrival and confirmation must both lie inside the release interval')
        delivery.update(reviewed_linked_crossing_ids=list(ids),
            review_provenance=_review_provenance(review['provenance']), link_status='review_linked',
            linked_quality=[dict(by_id[i]) for i in ids])


def evaluate(arrays, snapshot, *, event_windows=None, target_surface=None):
    """Return evidence, optional explicit-release Q and traces; no automatic labels.

    Explicit event dictionaries may include release_interval, mechanism_review
    and delivery_review. Their annotations are attributed manual input, not an
    algorithmic physical classification. Legacy candidate fields stay unchanged.
    """
    rest, surface = snapshot_geometry(snapshot)
    if target_surface is not None:
        surface = dict(target_surface)
    normal = np.asarray(surface['normal_world'], dtype=float)
    center = np.asarray(surface['center_world_m'], dtype=float)
    if (normal.shape != (3,) or center.shape != (3,) or not np.isfinite(center).all()
            or not np.isfinite(normal).all() or not np.isclose(np.linalg.norm(normal), 1., atol=1e-12, rtol=0)
            or normal[2] != 0 or not all(math.isfinite(surface[k]) and surface[k] > 0
                                        for k in ('width_m', 'height_m'))):
        raise ValueError('Finite horizontal unit normal, center and positive target dimensions required')
    measurements = extract_measurements(arrays, rest, normal)
    time = measurements['time_s']
    if len(time) < 2:
        raise ValueError('Event review requires at least two time samples')
    tracks = track_curvature(measurements)
    crossings = measure_crossings(arrays, surface)
    motion_center, motion_spread = _motion_distribution(measurements, rest)
    measurements.update(relative_motion_center_m=motion_center, relative_motion_spread_m=motion_spread)
    automatic = event_windows is None
    windows = candidate_windows(measurements) if automatic else event_windows
    events = []
    seen = set()
    for window in windows:
        start, peak, end = (window[k] for k in ('start_index', 'peak_index', 'end_index'))
        if (any(type(i) is not int for i in (start, peak, end))
                or not 0 <= start <= peak <= end < len(time) or start == end):
            raise ValueError('Event needs ordered integer start/peak/end samples within the recording')
        if (start, peak, end) in seen:
            raise ValueError('Duplicate event interval')
        seen.add((start, peak, end))
        support = []
        for track in tracks:
            indices = np.asarray(track['indices'])
            keep = (indices >= start) & (indices <= peak)
            if keep.sum() < 2:
                continue
            indices = indices[keep]
            position = np.asarray(track['position_m'])[keep]
            amplitude = np.asarray(track['amplitude_per_m'])[keep]
            nodes = np.asarray(track['node_indices'])[keep]
            # Sample association only. No speed or amplitude gate qualifies a whip.
            support.append(dict(track_id=track['id'], start_index=int(indices[0]),
                end_index=int(indices[-1]), start_s=float(time[indices[0]]), end_s=float(time[indices[-1]]),
                start_material_m=float(position[0]), end_material_m=float(position[-1]),
                signed_travel_m=float(position[-1]-position[0]),
                outward_travel_m=float(np.maximum(np.diff(position), 0).sum()),
                inward_travel_m=float(np.maximum(-np.diff(position), 0).sum()),
                maximum_curvature_per_m=float(amplitude.max()),
                reaches_last_interior_sample=bool((nodes == len(rest)-1).any()),
                flags=list(track['flags'])))
        forward_tracks = [s for s in support if s['signed_travel_m'] > 0]
        distal_tracks = [s for s in forward_tracks if s['reaches_last_interior_sample']]
        current_crossings = [c['id'] for c in crossings
                             if time[start] <= c['time_s'] <= c['confirmed_at_s'] <= time[end]]
        notes = ['Unreviewed kinematic candidate; temporal association does not establish causation.']
        if not distal_tracks:
            notes.append('No outward track reaches the last interior sample before the peak. This is a sampling observation, not a required whip condition.')
        if any(s['flags'] for s in support):
            notes.append('Track boundary/ambiguity flags require inspection; no unique physical wave identity is asserted.')
        if start == 0 or end == len(time)-1:
            notes.append('Interval touches a recording boundary; preparation or decay may be incomplete.')
        if measurements['world_forward_mps'][peak, -1] > 0 >= measurements['relative_forward_mps'][peak, -1]:
            notes.append('World-forward tip motion is not forward relative to the attachment at the selected peak.')
        sl = slice(start, peak+1)
        response = dict(relative_tip_at_start_mps=float(measurements['relative_speed_mps'][start, -1]),
            relative_tip_at_peak_mps=float(measurements['relative_speed_mps'][peak, -1]),
            relative_tip_increase_mps=float(measurements['relative_speed_mps'][peak, -1]
                                           -measurements['relative_speed_mps'][start, -1]),
            nonrigid_tip_at_peak_mps=float(measurements['nonrigid_speed_mps'][peak, -1]),
            node_relative_speed_at_start_mps=measurements['relative_speed_mps'][start].tolist(),
            node_relative_speed_at_peak_mps=measurements['relative_speed_mps'][peak].tolist(),
            motion_center_start_m=_number(motion_center[start]), motion_center_peak_m=_number(motion_center[peak]),
            motion_spread_start_m=_number(motion_spread[start]), motion_spread_peak_m=_number(motion_spread[peak]))
        response.update(_vector_metrics(arrays['cable_velocities'][peak, -1],
                                       arrays['cable_velocities'][peak, 0], normal))
        event = dict(id=f'event-{len(events):03d}', start_index=start, peak_index=peak, end_index=end,
            start_s=float(time[start]), peak_s=float(time[peak]), end_s=float(time[end]),
            selection='relative_tip_speed_basin' if automatic else 'explicit_review_interval',
            review_status='unreviewed', association='temporal_support_only', notes=notes,
            deformation=dict(tracks=support, outward_track_count=len(forward_tracks),
                distal_outward_track_count=len(distal_tracks),
                longest_outward_travel_m=max([0.]+[s['signed_travel_m'] for s in forward_tracks]),
                furthest_outward_endpoint_m=max([s['end_material_m'] for s in forward_tracks], default=None),
                maximum_turn_rad=float(measurements['turn_rad'][sl].max()),
                maximum_curvature_per_m=float(measurements['curvature_per_m'][sl].max()),
                chord_arc_recovery_fraction=float(measurements['chord_ratio'][peak]
                                                  -measurements['chord_ratio'][sl].min())),
            response=response, delivery=dict(crossing_ids=current_crossings,
                                             closest_approach=_closest(arrays, start, end, surface)))
        _event_core(event, window, arrays, measurements, crossings, normal)
        events.append(event)
    for crossing in crossings:
        crossing['associated_event_ids'] = [e['id'] for e in events if crossing['id'] in e['delivery']['crossing_ids']]
    default = (max(range(len(events)), key=lambda i: events[i]['response']['relative_tip_at_peak_mps'])
               if events else None)
    measured_lengths = np.linalg.norm(np.diff(arrays['cable_positions'], axis=1), axis=-1)
    result = dict(target_surface=surface, tracks=tracks, events=events, crossings=crossings,
                  default_event_index=default,
                  default_event_selection='Largest sampled relative-tip speed, for display only; not a whip ranking',
                  sampling=dict(samples=len(time), nodes=len(rest)+1, interior_curvature_samples=len(rest)-1,
                      minimum_dt_s=float(np.diff(time).min()), maximum_dt_s=float(np.diff(time).max()),
                      max_edge_length_error_m=float(np.abs(measured_lengths-rest).max())),
                  summary=dict(maximum_curvature_per_m=float(measurements['curvature_per_m'].max()),
                      maximum_turn_rad=float(measurements['turn_rad'].max()),
                      peak_relative_tip_speed_mps=float(measurements['relative_speed_mps'][:, -1].max()),
                      peak_world_tip_speed_mps=float(measurements['world_speed_mps'][:, -1].max()),
                      peak_nonrigid_tip_speed_mps=float(measurements['nonrigid_speed_mps'][:, -1].max())))
    return result, measurements


def plot_case(case, measurements, destination):
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
    except ImportError as error:
        raise RuntimeError('Install the review extra: python -m pip install -e ".[review]"') from error
    t, length = measurements['time_s'], measurements['node_s_m'][-1]
    fig, axes = plt.subplots(4, 1, figsize=(12, 11), sharex=True, constrained_layout=True)
    fields = [('curvature_per_m', 'material_s_m', 'Bending (1/m)', 'magma'),
              ('relative_speed_mps', 'node_s_m', 'Speed relative to attachment (m/s)', 'viridis')]
    for ax, (field, material, title, color) in zip(axes, fields):
        mesh = ax.pcolormesh(t, measurements[material]/length, measurements[field].T,
                            shading='nearest', cmap=color, vmin=0)
        fig.colorbar(mesh, ax=ax, label=title, pad=.015)
        ax.set_ylabel('Material position / length')
        ax.set_ylim(0, 1)
    for track in case['tracks']:
        if len(track['indices']) > 1:
            axes[0].plot(t[track['indices']], np.asarray(track['position_m'])/length,
                         color='cyan', linewidth=.7, alpha=.7, linestyle='--' if track['flags'] else '-')
    for name, field, node, color in [('Tip, world', 'world_speed_mps', -1, '#0072B2'),
            ('Attachment, world', 'world_speed_mps', 0, '#D55E00'),
            ('Tip, relative', 'relative_speed_mps', -1, '#009E73'),
            ('Tip, rigid-fit residual', 'nonrigid_speed_mps', -1, '#CC79A7')]:
        axes[2].plot(t, measurements[field][:, node], label=name, color=color)
    axes[2].set_ylabel('Speed (m/s)')
    axes[2].legend(loc='upper left', fontsize=8, ncol=2)
    axes[3].plot(t, measurements['world_forward_mps'][:, -1], label='Tip toward target, world', color='#0072B2')
    axes[3].plot(t, measurements['relative_forward_mps'][:, -1], label='Tip toward target, relative', color='#009E73')
    axes[3].axhline(0, color='#555555', linewidth=.7)
    axes[3].set_ylabel('Signed forward speed (m/s)')
    axes[3].set_xlabel('Recorded time (s)')
    axes[3].legend(loc='upper left', fontsize=8)
    selected = case['default_event_index']
    event = None if selected is None else case['events'][selected]
    release = None if event is None else event.get('release_measurement')
    for ax in axes:
        if event:
            ax.axvspan(event['start_s'], event['end_s'], color='#f2a900', alpha=.13)
            ax.axvline(event['peak_s'], color='#d22d2d', linewidth=1.1)
        if release:
            ax.axvspan(release['start_s'], release['end_s'], color='#009E73', alpha=.12)
            ax.axvline(release['q_time_s'], color='#762A83', linewidth=1.1,
                       label='Explicit release Q time' if ax is axes[2] else None)
            ax.axvline(release['baseline_time_s'], color='#555555', linestyle='-.', linewidth=.9,
                       label='Prerelease baseline time' if ax is axes[2] else None)
        for crossing in case['crossings']:
            ax.axvline(crossing['time_s'], color='#666666', linestyle=':', linewidth=.8)
        ax.set_xlim(t[0], t[-1])
    fig.suptitle(f"{case['name']}\n{case['evidence_kind'].replace('_', ' ')} · unreviewed kinematic evidence", fontsize=13)
    title = 'Cyan: tracks (dashed: flags); amber: candidate; red: candidate peak; dotted: passages. Color scales are per case.'
    if release:
        title += '\nGreen: explicit release; purple: Q time; gray dash-dot: baseline. Legacy deformation is candidate start..peak.'
        axes[2].scatter([release['q_time_s'], release['baseline_time_s']],
            [release['q_relative_tip_speed_mps'], release['baseline_relative_tip_speed_mps']],
            color=['#762A83', '#555555'], s=22, zorder=5)
        axes[2].legend(loc='upper left', fontsize=8, ncol=2)
    axes[0].set_title(title, fontsize=9)
    fig.savefig(destination, dpi=140)
    plt.close(fig)


def _viewer_command(report, case_index, event_index=None):
    root = Path(__file__).resolve().parents[1]
    args = [sys.executable, str(root/'run_replay.py'), '--measurement-report', str(report), '--case', str(case_index)]
    if event_index is not None:
        args += ['--event', str(event_index)]
    return shlex.join(args)


def _write_review(report, output):
    lines = ['# Whipping-event measurement review', '',
        'Descriptive evidence only. No automatic whip labels, reward, policy ranking or energy-transfer claim.', '',
        'Automatic intervals are relative-tip speed basins between adjacent troughs. All candidates remain in report.json.',
        'A default interval is selected only for display. Track association and target passage do not prove a common physical cause.', '',
        'Optional explicit release intervals report Q in m/s, a raw prerelease baseline and signed gain. '
        'Mechanism and delivery-link reviews are attributed input, not automatic physical labels. '
        'Legacy candidate review_status remains unreviewed independently of these annotations.', '',
        '## Cases', '',
        '| Case | Evidence | Max bend (1/m) | Peak relative tip (m/s) | Candidates |',
        '|---|---|---:|---:|---:|']
    for case in report['cases']:
        s = case['summary']
        name = case['name'].replace('|', '\\|').replace('\n', ' ')
        lines.append(f"| {name} | {case['evidence_kind']} | {s['maximum_curvature_per_m']:.4g} | "
                     f"{s['peak_relative_tip_speed_mps']:.4g} | {len(case['events'])} |")
    for i, case in enumerate(report['cases']):
        lines += ['', f"## {case['name']}", '', case.get('notes', ''), '',
                  f"![Synchronized motion evidence]({(output/case['plot']).as_posix()})", '',
                  'Open in the existing replay viewer:', '', '```bash', case['viewer_command'], '```', '',
                  '| Event | Start / peak / end (s) | Relative tip increase (m/s) | Longest outward track travel (m) | Forward world / relative at peak (m/s) | Crossings |',
                  '|---|---|---:|---:|---|---:|']
        for event in case['events']:
            r = event['response']
            lines.append(f"| {event['id']} | {event['start_s']:.3f} / {event['peak_s']:.3f} / {event['end_s']:.3f} "
                         f"| {r['relative_tip_increase_mps']:.4g} | {event['deformation']['longest_outward_travel_m']:.4g} "
                         f"| {r['forward_world_mps']:.4g} / {r['forward_relative_mps']:.4g} | {len(event['delivery']['crossing_ids'])} |")
        releases = [event for event in case['events'] if 'release_measurement' in event]
        if releases:
            lines += ['', 'Explicit release measurements (Q is sampled output speed, not a success score):', '',
                '| Event | Release (s) | Q / baseline / signed gain (m/s) | Q / baseline time (s) | Mechanism review | Delivery link review |',
                '|---|---|---|---|---|---|']
            for event in releases:
                r = event['release_measurement']
                lines.append(f"| {event['id']} | {r['start_s']:.3f} / {r['end_s']:.3f} "
                    f"| {r['q_relative_tip_speed_mps']:.4g} / {r['baseline_relative_tip_speed_mps']:.4g} / {r['signed_gain_mps']:.4g} "
                    f"| {r['q_time_s']:.3f} / {r['baseline_time_s']:.3f} | {event['mechanism_review']['status']} "
                    f"| {event['delivery']['link_status']} ({event['delivery']['reviewed_linked_crossing_ids']}) |")
            lines += ['', 'Supplied review provenance (full annotations and raw linked delivery quality are in report.json):']
            for event in releases:
                entries = [('release selection', event['release_measurement']['selection_provenance']),
                           ('mechanism', event['mechanism_review']['provenance']),
                           ('delivery link', event['delivery']['review_provenance'])]
                for label, provenance in entries:
                    if provenance is not None:
                        lines.append(f"- {event['id']} {label}: "+json.dumps(provenance, ensure_ascii=False))
        if not case['events']:
            lines += ['', 'No sampled relative-tip speed peak was detected. This is not a universal no-whip classification.']
        else:
            selected = case['events'][case['default_event_index']]
            lines += ['', f"Displayed: {selected['id']} — {selected['review_status']}.", '']
            lines += ['- '+note for note in selected['notes']]
    lines += ['', '## Interpretation limits', '']+['- '+s for s in report['limitations']]
    (output/'review.md').write_text('\n'.join(lines)+'\n')
    fields = ['case', 'event', 'start_s', 'peak_s', 'end_s', 'review_status', 'relative_tip_increase_mps',
              'forward_world_mps', 'forward_relative_mps', 'longest_outward_travel_m',
              'distal_outward_track_count', 'crossing_count', 'viewer_command',
              'release_start_s', 'release_end_s', 'q_relative_tip_speed_mps', 'q_time_s',
              'baseline_relative_tip_speed_mps', 'baseline_time_s', 'signed_gain_mps',
              'mechanism_status', 'delivery_link_status', 'reviewed_linked_crossing_count']
    with (output/'events.csv').open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for i, case in enumerate(report['cases']):
            for j, event in enumerate(case['events']):
                row = {k: event[k] for k in ('start_s', 'peak_s', 'end_s', 'review_status')}
                row.update({k: event['response'][k] for k in ('relative_tip_increase_mps', 'forward_world_mps', 'forward_relative_mps')})
                row.update(case=case['name'], event=event['id'],
                    longest_outward_travel_m=event['deformation']['longest_outward_travel_m'],
                    distal_outward_track_count=event['deformation']['distal_outward_track_count'],
                    crossing_count=len(event['delivery']['crossing_ids']),
                    viewer_command=_viewer_command(output/'report.json', i, j))
                row.update(mechanism_status=event['mechanism_review']['status'],
                    delivery_link_status=event['delivery']['link_status'],
                    reviewed_linked_crossing_count=len(event['delivery']['reviewed_linked_crossing_ids']))
                if 'release_measurement' in event:
                    release = event['release_measurement']
                    row.update(release_start_s=release['start_s'], release_end_s=release['end_s'])
                    row.update({key: release[key] for key in ('q_relative_tip_speed_mps', 'q_time_s',
                        'baseline_relative_tip_speed_mps', 'baseline_time_s', 'signed_gain_mps')})
                writer.writerow(row)


def run_manifest(manifest_path, output):
    manifest_path, output = Path(manifest_path).resolve(), Path(output).resolve()
    raw = json.loads(manifest_path.read_text())
    if raw.get('schema') != INPUT_SCHEMA or not isinstance(raw.get('cases'), list) or not raw['cases']:
        raise ValueError('Expected '+INPUT_SCHEMA+' with nonempty cases')
    def pinned(item):
        path = (manifest_path.parent/item['path']).resolve()
        if sha256(path) != item['sha256']:
            raise ValueError('Input hash mismatch: '+str(path))
        return path
    prepared, names = [], set()
    for case in raw['cases']:
        if not isinstance(case.get('name'), str) or not case['name'].strip() or case['name'] in names:
            raise ValueError('Each case needs a unique nonempty name')
        names.add(case['name'])
        if case.get('evidence_kind') not in ('saved_simulation', 'synthetic_kinematic_control'):
            raise ValueError('Separate saved simulations from synthetic kinematic controls')
        path, source = pinned(case), pinned(case['source_snapshot'])
        snapshot = json.loads(source.read_text())
        with np.load(path, allow_pickle=False) as saved:
            arrays = dict(saved)
        from viewer.replay import StateReplay
        StateReplay(arrays)  # All emitted cases must also open in the existing viewer.
        result, trace = evaluate(arrays, snapshot, event_windows=case.get('events'))
        result.update(name=case['name'], evidence_kind=case['evidence_kind'], notes=case.get('notes', ''),
            source_replay=str(path), replay_sha256=sha256(path), source_snapshot=str(source),
            source_snapshot_sha256=sha256(source))
        json.dumps(result, allow_nan=False)  # Validate before creating any output artifacts.
        prepared.append((result, trace))
    # Validate all inputs before reserving the output; never overwrite a review.
    output.mkdir(parents=True, exist_ok=False)
    report = dict(schema=SCHEMA, manifest=str(manifest_path), manifest_sha256=sha256(manifest_path),
        scope='Offline kinematic event review; no physics, policy inference, reward or classifier',
        runtime=dict(python=sys.version.split()[0], numpy=np.__version__),
        source_hashes={str(p): sha256(p) for p in (Path(__file__).resolve(),
                      Path(__file__).with_name('whip_measurements.py'),
                      Path(__file__).resolve().parents[1]/'tasks/whip/stage1_configuration.py',
                      Path(__file__).resolve().parents[1]/'viewer/replay.py')},
        definitions=dict(event_selection='All sampled relative-tip speed maxima bounded by adjacent troughs; explicit intervals override',
            tracking='Consecutive local curvature peaks within one spatial sample; ambiguous associations split tracks',
            association='Tracks clipped to start..peak; target arrival and opposite-side confirmation restricted to start..end. Temporal support is not causation',
            curvature='Turning angle / average adjacent measured edge length; not the native constitutive strain',
            rigid_residual='Material-length weighted best-fit translation and rotation; not kinetic energy',
            localization='Material-length weighted relative-speed-squared centroid and spread; not energy',
            review_status='All generated candidates unreviewed; separate mechanism_review is attributed manual input',
            release_measurement='Q is maximum sampled attachment-relative tip speed within independently supplied inclusive release bounds; exact ties select earliest sample. Baseline is the explicit earlier raw sample; signed gain is Q minus baseline',
            mechanism_review='Supported/unsupported/unresolved are supplied review judgements with provenance, never inferred or verified by this evaluator',
            delivery_review='Release arrival and opposite-side confirmation must both lie inside release bounds. Review-linked quality requires a supported manual mechanism review and explicit crossing IDs; geometric fields are not task success'),
        limitations=['Sampling can miss narrow bends and brief peaks; interpolation adds no physical resolution.',
            'Each plot uses its own color scales; compare amplitudes using units and the case table.',
            'The tracker can confuse overlapping/fading waves; flags and raw maps require review.',
            'Track-wide ambiguity flags can originate outside a clipped candidate interval.',
            'A missing distal track is unresolved evidence, not proof that whipping did not occur.',
            'Neither relative speed nor a rigid-fit residual measures energy transfer.',
            'Explicit interval provenance records the supplied selection method; it cannot establish that selection was independent of speed.',
            'Q is a sampled event-output measurement, not a physical-whip classifier, episode ranking, impact quality or training reward.',
            'Manual mechanism and delivery-link annotations are not automatically validated physical truth; no minimum acceptable quality is defined.',
            'Legacy event deformation and response fields use candidate start..peak, independently of explicit release bounds and Q time; full raw traces retain release context.',
            'Crossings and nearest approaches use linear saved-state interpolation; no contact/impact physics.',
            'A plane touch or recording ending on the plane is not an established crossing; inspect closest approach.',
            'Synthetic controls validate measurement logic, not dynamically feasible flight.',
            'Selected historical policies are not independent whip labels or matched policy comparisons.',
            'Automatic intervals require future samples and are not a causal training signal.'], cases=[])
    for i, (case, trace) in enumerate(prepared):
        case['trace'], case['plot'] = f'trace-{i:03d}.npz', f'case-{i:03d}.png'
        case['viewer_command'] = _viewer_command(output/'report.json', i, case['default_event_index'])
        np.savez_compressed(output/case['trace'], **trace)
        plot_case(case, trace, output/case['plot'])
        report['cases'].append(case)
        print(f"{case['name']}: {len(case['events'])} candidate intervals; unreviewed", flush=True)
    (output/'report.json').write_text(json.dumps(report, indent=2, allow_nan=False)+'\n')
    _write_review(report, output)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--manifest', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    run_manifest(args.manifest, args.output)


if __name__ == '__main__':
    main()
