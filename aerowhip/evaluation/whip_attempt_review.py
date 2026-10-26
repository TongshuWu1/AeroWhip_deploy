"""Whole-attempt excitation, distal response and geometric delivery measurements.

This offline API does not select physical events, compute a reward, estimate
energy transfer or assign whip/success labels. Explicit interval provenance is
recorded, not verified. Full saved traces and failed trial outcomes are retained.
"""
import numpy as np

from aerowhip.evaluation.whip_event_review import measure_crossings
from aerowhip.evaluation.whip_measurements import extract_measurements, track_curvature
from aerowhip.evaluation.whip_output_decomposition import decompose
from aerowhip.evaluation.whip_peak_timing import describe_speed_peaks


def _bounds(start, end, count):
    if (any(isinstance(i, (bool, np.bool_)) or not isinstance(i, (int, np.integer))
            for i in (start, end)) or not 0 <= start < end < count):
        raise ValueError('Need integer 0 <= start_index < end_index < samples')
    return int(start), int(end)


def _provenance(value):
    keys = {'reviewer', 'method', 'notes'}
    if (not isinstance(value, dict) or set(value) != keys
            or any(not isinstance(v, str) or not v.strip() for v in value.values())):
        raise ValueError('Provenance needs nonempty reviewer, method and notes strings')
    return dict(value)


def _surface(value):
    try:
        normal = np.asarray(value['normal_world'])
        center = np.asarray(value['center_world_m'])
        dimensions = np.asarray([value['width_m'], value['height_m']])
        if any(a.dtype.kind not in 'fiu' for a in (normal, center, dimensions)):
            raise ValueError('Target fields must be real numbers')
        normal, center, dimensions = (a.astype(float) for a in (normal, center, dimensions))
        if (normal.shape != (3,) or center.shape != (3,) or dimensions.shape != (2,)
                or not all(np.isfinite(a).all() for a in (normal, center, dimensions))
                or not np.isclose(np.linalg.norm(normal), 1., atol=1e-12, rtol=0)
                or normal[2] != 0 or np.any(dimensions <= 0)):
            raise ValueError('Invalid horizontal rectangle')
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError('Need finite horizontal unit normal, center and positive width/height') from exc
    return dict(normal_world=normal.tolist(), center_world_m=center.tolist(),
                width_m=float(dimensions[0]), height_m=float(dimensions[1]))


def evaluate_attempt(arrays, rest_lengths, surface, *, start_index, end_index,
                     selection_provenance, release_interval=None, trial_outcome=None):
    """Return JSON-compatible measurements and full-record NumPy traces.

    Bounds are inclusive saved sample indices and must be supplied before this
    calculation. Peak timing always uses the same whole-attempt bounds. An
    optional release_interval has start_index, end_index and provenance; only
    the display sample and release crossing associations use its narrower bounds.
    Neither interval is inferred from a desired propagation pattern.

    Saved positions/velocities are world metres/metres per second; node zero is
    the attachment. Only horizontal target normals are currently supported.
    Positions and crossings use linear saved-sample interpolation; confirmation
    of opposite-side arrival must also fall within the attributed interval.

    trial_outcome records a supplied status ('valid', 'failed', 'unknown') and
    reason, independently of measurements. It is never inferred from finite
    arrays. Invalid/padded samples are rejected rather than silently removed.
    """
    surface = _surface(surface)
    provenance = _provenance(selection_provenance)
    m = extract_measurements(arrays, rest_lengths, surface['normal_world'])
    time = m['time_s']
    start, end = _bounds(start_index, end_index, len(time))
    outcome = (dict(status='unknown', reason='No trial outcome supplied')
               if trial_outcome is None else trial_outcome)
    if (not isinstance(outcome, dict) or set(outcome) != {'status', 'reason'}
            or outcome['status'] not in ('valid', 'failed', 'unknown')
            or not isinstance(outcome['reason'], str) or not outcome['reason'].strip()):
        raise ValueError('trial_outcome needs valid/failed/unknown status and a nonempty reason')

    release = None
    if release_interval is not None:
        if (not isinstance(release_interval, dict)
                or set(release_interval) != {'start_index', 'end_index', 'provenance'}):
            raise ValueError('release_interval needs start_index, end_index and provenance')
        a, b = _bounds(release_interval['start_index'], release_interval['end_index'], len(time))
        if not start <= a < b <= end:
            raise ValueError('Release must be contained in the attempt')
        release = dict(start_index=a, end_index=b, start_time_s=float(time[a]),
                       end_time_s=float(time[b]), provenance=_provenance(release_interval['provenance']))

    p = np.asarray(arrays['cable_positions'], float)
    v = np.asarray(arrays['cable_velocities'], float)
    shape = decompose(p, v, rest_lengths)
    speeds = dict(world=m['world_speed_mps'], attachment_relative=m['relative_speed_mps'],
                  attachment_anchored_shape=shape['shape_speed_mps'])
    timing = {name: describe_speed_peaks(time, speed, start_index=start, end_index=end)
              for name, speed in speeds.items()}
    a, b = (start, end) if release is None else (release['start_index'], release['end_index'])
    display = describe_speed_peaks(time, m['relative_speed_mps'][:, -1:], start_index=a, end_index=b)
    peak = display['nodes'][0]
    q = peak['first_peak_index']
    sample = dict(index=q, time_s=float(time[q]), selection='earliest_relative_tip_maximum_for_display_only',
                  interval='attempt' if release is None else 'explicit_release',
                  selection_start_index=a, selection_end_index=b,
                  peak_indices=peak['peak_indices'], boundary_peak=peak['boundary_peak'],
                  peak_timing_identifiable=peak['peak_timing_identifiable'],
                  world_velocity_mps=v[q, -1].tolist(), attachment_velocity_mps=v[q, 0].tolist(),
                  relative_velocity_mps=shape['relative_velocity_world_mps'][q, -1].tolist(),
                  shape_velocity_mps=shape['shape_velocity_world_mps'][q, -1].tolist(),
                  world_speed_mps=float(m['world_speed_mps'][q, -1]),
                  relative_speed_mps=float(m['relative_speed_mps'][q, -1]),
                  shape_speed_mps=float(shape['shape_speed_mps'][q, -1]),
                  world_forward_mps=float(m['world_forward_mps'][q, -1]),
                  relative_forward_mps=float(m['relative_forward_mps'][q, -1]))

    # Recompute tracks only inside the complete attempt, never truncate at Q.
    local = dict(m, time_s=time[start:end+1], curvature_per_m=m['curvature_per_m'][start:end+1])
    tracks = track_curvature(local)
    for track in tracks:
        track['indices'] = [i+start for i in track['indices']]

    crossings = measure_crossings(dict(cable_positions=p, cable_velocities=v, time_s=time), surface)
    for crossing in crossings:
        lateral = max(0., abs(crossing['lateral_error_m'])-surface['width_m']/2)
        vertical = max(0., abs(crossing['vertical_error_m'])-surface['height_m']/2)
        crossing['rectangle_miss_distance_m'] = float(np.hypot(lateral, vertical))

    def contained(a, b):
        return [c['id'] for c in crossings if time[a] <= c['time_s'] <= c['confirmed_at_s'] <= time[b]]

    attempt_ids = contained(start, end)
    release_ids = [] if release is None else contained(release['start_index'], release['end_index'])

    def forward_rectangle(ids):
        return [c['id'] for c in crossings if c['id'] in ids and c['passage'] == 'forward'
                and c['inside_target_rectangle']]

    sl = slice(start, end+1)
    report = dict(schema='offline_whip_attempt_review_v1', trial_outcome=dict(outcome),
        attempt=dict(start_index=start, end_index=end, start_time_s=float(time[start]),
                     end_time_s=float(time[end]), selection_provenance=provenance,
                     recording_boundary=bool(start == 0 or end == len(time)-1)),
        release_interval=release, target_surface=surface,
        excitation=dict(attachment_peak_speed_mps=float(m['world_speed_mps'][sl, 0].max()),
            attachment_sampled_path_m=float(np.linalg.norm(np.diff(p[sl, 0], axis=0), axis=1).sum()),
            attachment_displacement_world_m=(p[end, 0]-p[start, 0]).tolist(),
            total_turn_range_rad=[float(m['turn_rad'][sl].min()), float(m['turn_rad'][sl].max())],
            chord_ratio_range=[float(m['chord_ratio'][sl].min()), float(m['chord_ratio'][sl].max())],
            per_node_curvature_range_per_m=np.ptp(m['curvature_per_m'][sl], axis=0).tolist()),
        distal_response=dict(peak_timing=timing, output_sample=sample, curvature_tracks=tracks,
                             mechanism_status='unresolved'),
        delivery=dict(crossings=crossings, attempt_crossing_ids=attempt_ids,
            attempt_forward_rectangle_crossing_ids=forward_rectangle(attempt_ids),
            release_crossing_ids=release_ids, geometric_hit_ids=forward_rectangle(release_ids),
            association='temporal_containment_only', mechanism_link_status='unresolved'),
        sampling=dict(full_record_samples=len(time), attempt_samples=end-start+1, nodes=p.shape[1],
            minimum_dt_s=float(np.diff(time).min()), maximum_dt_s=float(np.diff(time).max()),
            maximum_edge_length_error_m=float(np.abs(np.linalg.norm(np.diff(p, axis=1), axis=-1)
                                                     -np.asarray(rest_lengths)).max())),
        limitations=[
            'No physical-whip label, scalar score, energy estimate, success threshold or stage promotion.',
            'Attempt/release provenance is supplied attribution, not verified independence or causality.',
            'A common interval can still contain unrelated oscillations; peak order is descriptive only.',
            'Rigid-fit residual uses geometric rest-length weights, not physical masses or work.',
            'Temporal rectangle passages alone do not establish a cable-mediated strike.',
            'Missing release bounds leave release association unresolved; attempt hits remain separate.',
            'Curvature tracking and interpolation cannot recover unresolved spatial or temporal detail.',
            'Full-record boundary peaks, flat traces and supplied failed outcomes require explicit review.'])
    traces = dict(m, shape_speed_mps=shape['shape_speed_mps'],
                  shape_velocity_world_mps=shape['shape_velocity_world_mps'],
                  fitted_rigid_velocity_world_mps=shape['fitted_rigid_velocity_world_mps'],
                  relative_velocity_world_mps=shape['relative_velocity_world_mps'],
                  fitted_angular_velocity_world_radps=shape['angular_velocity_world_radps'],
                  cable_positions=p.copy(), cable_velocities=v.copy())
    return report, traces
