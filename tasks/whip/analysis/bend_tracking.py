"""Conservative descriptive bend tracking; no whip classifier or reward.

Coordinates are material/rest arclength, not instantaneous geometric length.
No smoothing, physical amplitude threshold, or motion bound is applied. Explicit
invalid-row gaps are not bridged; finite timestamp gaps are not auto-detected.
An unresolved link is not evidence of absent propagation. Tracking
identity may change discontinuously at feature birth, merger, or splitting.
"""

import numpy as np


def _groups(values, floor):
    """Numerically equal plateaus; the tolerance is roundoff-only."""
    groups = []
    start = 0
    for stop in range(1, len(values) + 1):
        if stop == len(values) or abs(values[stop] - values[start]) > floor:
            groups.append((start, stop - 1, float(np.mean(values[start:stop]))))
            start = stop
    return groups


def _features(curvature, coordinate, floor):
    groups = _groups(curvature, floor)
    found = []
    for group, (first, last, height) in enumerate(groups):
        if height <= floor:
            continue
        if group > 0 and height <= groups[group - 1][2] + floor:
            continue
        if group + 1 < len(groups) and height <= groups[group + 1][2] + floor:
            continue
        endpoint = group == 0 or group == len(groups) - 1
        if endpoint:
            found.append(dict(peak_s_m=float((coordinate[first] + coordinate[last]) / 2),
                sampled_peak_left_s_m=float(coordinate[first]),
                sampled_peak_right_s_m=float(coordinate[last]),
                peak_curvature_per_m=height, boundary_censored=True,
                exclusion_reason='endpoint_maximum', peak_estimate='plateau_midpoint',
                plateau_indices=(first, last)))
            continue
        left_group, right_group = group - 1, group + 1
        while left_group > 0:
            value = groups[left_group][2]
            if value < groups[left_group - 1][2] - floor and value < groups[left_group + 1][2] - floor:
                break
            left_group -= 1
        while right_group < len(groups) - 1:
            value = groups[right_group][2]
            if value < groups[right_group - 1][2] - floor and value < groups[right_group + 1][2] - floor:
                break
            right_group += 1
        baseline = max(groups[left_group][2], groups[right_group][2])
        prominence = height - baseline
        if prominence <= floor:
            continue
        # Intersections with the higher-trough baseline define ONE connected lobe.
        left, right = first, last
        while left > 0 and curvature[left - 1] > baseline:
            left -= 1
        while right + 1 < len(curvature) and curvature[right + 1] > baseline:
            right += 1
        if left == 0:
            left_cross = coordinate[0]
        else:
            fraction = (baseline - curvature[left - 1]) / (curvature[left] - curvature[left - 1])
            left_cross = coordinate[left - 1] + fraction * (coordinate[left] - coordinate[left - 1])
        if right == len(curvature) - 1:
            right_cross = coordinate[-1]
        else:
            fraction = (baseline - curvature[right + 1]) / (curvature[right] - curvature[right + 1])
            right_cross = coordinate[right + 1] + fraction * (coordinate[right] - coordinate[right + 1])
        interior = (coordinate > left_cross) & (coordinate < right_cross)
        sample_s = np.r_[left_cross, coordinate[interior], right_cross]
        excess = np.r_[0., np.maximum(curvature[interior] - baseline, 0.), 0.]
        area = float(np.sum(np.diff(sample_s) * (excess[:-1] + excess[1:]) / 2))
        peak = float((coordinate[first] + coordinate[last]) / 2)
        estimate = 'plateau_midpoint'
        if first == last:
            index = first
            dl, dr = coordinate[index] - coordinate[index - 1], coordinate[index + 1] - coordinate[index]
            slope_l = (curvature[index] - curvature[index - 1]) / dl
            slope_r = (curvature[index + 1] - curvature[index]) / dr
            quadratic = (slope_r - slope_l) / (dl + dr)
            linear = slope_l + quadratic * dl
            if quadratic < -floor / (dl + dr) ** 2:
                vertex = coordinate[index] - linear / (2 * quadratic)
                if np.isfinite(vertex) and coordinate[index - 1] <= vertex <= coordinate[index + 1] and left_cross <= vertex <= right_cross:
                    peak, estimate = float(vertex), 'three_point_quadratic_estimate'
        # A flat trough extending into the observed interior is resolved even
        # when its plateau also reaches a boundary. A lone endpoint trough is
        # not: there is no observed interior turn/flat confirming its minimum.
        left_boundary_only = left_group == 0 and groups[left_group][1] == 0
        right_boundary_only = right_group == len(groups) - 1 and groups[right_group][0] == len(curvature) - 1
        censored = left_boundary_only or right_boundary_only or left == 0 or right == len(curvature) - 1
        found.append(dict(peak_s_m=peak, peak_curvature_per_m=height,
            sampled_peak_left_s_m=float(coordinate[first]),
            sampled_peak_right_s_m=float(coordinate[last]),
            excess_turn_rad=area, width_m=float(right_cross - left_cross),
            prominence_per_m=float(prominence), baseline_per_m=float(baseline),
            left_s_m=float(left_cross), right_s_m=float(right_cross),
            boundary_censored=censored,
            exclusion_reason='unobserved_boundary_trough' if censored else None,
            peak_estimate=estimate, plateau_indices=(first, last)))
    return found


def _tip_velocity(time, positions, valid):
    velocity = np.full((len(time), 3), np.nan)
    start = 0
    while start < len(time):
        if not valid[start]:
            start += 1
            continue
        stop = start + 1
        while stop < len(time) and valid[stop]:
            stop += 1
        if stop - start >= 2:
            velocity[start:stop] = np.gradient(positions[start:stop, -1],
                time[start:stop], axis=0, edge_order=2 if stop - start >= 3 else 1)
        start = stop
    return velocity


def _coordinate_roundoff_bound(positions, lengths, dual):
    """Position-quantization contribution to curvature uncertainty, in 1/m.

    Assume each supplied coordinate has one rounding of relative size u, not
    unknown measurement/integration error. Edge subtraction adds another such
    rounding. Thus ||delta edge|| <= gamma1*(||p0||+||p1||+||edge||).
    An uncertainty ball of radius delta around a nonzero edge changes its
    direction by at most asin(delta/length) when delta<length. Adjacent angular
    uncertainties add in their turning angle, then divide by rest dual length.
    Recentring cannot undo quantization already present in world coordinates.
    Ordinary tangent/cross/atan2 arithmetic is handled by the separate eps term.
    """
    u = np.finfo(float).eps / 2
    gamma1 = u / (1 - u)
    point_norm = np.linalg.norm(positions, axis=-1)
    edge_error = gamma1 * (point_norm[:-1] + point_norm[1:] + lengths)
    ratio = edge_error / lengths
    angular = np.where(ratio < 1, np.arcsin(np.minimum(ratio, 1)), np.pi)
    return (angular[:-1] + angular[1:]) / dual, bool(np.any(ratio >= 1))


def measure_bends(time, positions, material_s, *, tip_lag_s=.15):
    """Measure (T,N,3) centerlines with fixed, strictly increasing (N,) rest s.

    Returns curvature/turning-angle histories on interior nodes, finite sample
    validity, position-derived tip velocity/speed, all detected frame features,
    and conservatively matched INTERIOR tracks. Boundary-censored features are
    preserved in features_by_frame but excluded from interior tracking.
    Invalid position rows are retained as NaN and split tracks/derivatives.
    Finite, nonuniform timestamps are differentiated as supplied, not interpreted
    as missing rows. Matching uses sampled peak/plateau INTERVALS; quadratic
    subgrid locations are descriptive only, not branch-identity decisions.
    Tip speed uses nonuniform central differences within contiguous valid runs,
    one-sided differences at run endpoints; isolated valid rows remain NaN.

    Track arrays: frame_indices, time_s, peak_s_m, excess_turn_rad, width_m,
    prominence_per_m, left_s_m, right_s_m, peak_estimate. Scalar fields include
    start/end reasons, net travel/rate, and same-end-event tip-speed summaries.
    Every track, including singleton fragments, is returned. No track is called
    a whip, and no physical coherence/success threshold is selected.
    """
    time = np.asarray(time, dtype=np.float64)
    positions = np.asarray(positions, dtype=np.float64)
    material_s = np.asarray(material_s, dtype=np.float64)
    if time.ndim != 1 or len(time) < 2 or not np.isfinite(time).all() or not np.isfinite(np.diff(time)).all() or not (np.diff(time) > 0).all():
        raise ValueError('time must be a finite strictly increasing vector with >=2 samples')
    if material_s.ndim != 1 or len(material_s) < 3 or not np.isfinite(material_s).all() or not np.isfinite(np.diff(material_s)).all() or not (np.diff(material_s) > 0).all():
        raise ValueError('material_s must be finite and strictly increasing with >=3 nodes')
    if positions.shape != (len(time), len(material_s), 3):
        raise ValueError('positions must have shape (len(time), len(material_s), 3)')
    if not np.isfinite(tip_lag_s) or tip_lag_s < 0:
        raise ValueError('tip_lag_s must be finite and nonnegative')
    dual = (material_s[2:] - material_s[:-2]) / 2
    coordinate = material_s[1:-1]
    with np.errstate(invalid='ignore', over='ignore', divide='ignore'):
        edges = np.diff(positions, axis=1)
        lengths = np.linalg.norm(edges, axis=-1)
        valid = np.isfinite(positions).all((1, 2)) & np.isfinite(lengths).all(1) & (lengths > 0).all(1)
        tangent = edges / lengths[..., None]
        cross = np.linalg.norm(np.cross(tangent[:, :-1], tangent[:, 1:]), axis=-1)
        dot = np.sum(tangent[:, :-1] * tangent[:, 1:], axis=-1)
        angle = np.arctan2(cross, dot)
        curvature = angle / dual
    valid &= np.isfinite(curvature).all(1)
    angle[~valid], curvature[~valid] = np.nan, np.nan
    velocity = _tip_velocity(time, positions, valid)
    speed = np.linalg.norm(velocity, axis=-1)
    floor = np.full(len(time), np.nan)
    coordinate_uncertainty = np.full_like(curvature, np.nan)
    roundoff_unresolved = np.zeros(len(time), dtype=bool)
    feature_frames, tracks = [], []
    ambiguity = np.zeros(len(time), dtype=bool)
    active = []
    spatial_tolerance = 128 * np.finfo(float).eps * max(1., np.max(np.abs(material_s)))

    def start_track(frame, feature, reason):
        track = dict(track_id=len(tracks), observations=[(frame, feature)],
                     start_reason=reason, end_reason=None, ambiguous=reason == 'ambiguous_matching')
        tracks.append(track)
        return track

    for frame in range(len(time)):
        if not valid[frame]:
            feature_frames.append([])
            for track in active:
                track['end_reason'] = 'invalid_gap'
            active = []
            continue
        bound, unresolved = _coordinate_roundoff_bound(positions[frame], lengths[frame], dual)
        coordinate_uncertainty[frame] = bound
        roundoff_unresolved[frame] = unresolved
        arithmetic = 128 * np.finfo(float).eps * max(1 / dual.min(), np.max(np.abs(curvature[frame])))
        # A comparison of two curvature values has BOTH uncertainty budgets.
        floor[frame] = arithmetic + 2 * np.max(bound)
        features = _features(curvature[frame], coordinate, floor[frame])
        feature_frames.append(features)
        current = [feature for feature in features if not feature['boundary_censored']]
        candidates = np.zeros((len(active), len(current)), dtype=bool)
        for previous_index, track in enumerate(active):
            previous = track['observations'][-1][1]
            for current_index, feature in enumerate(current):
                # Use the observed peak/plateau, not an ill-conditioned subgrid
                # vertex, for identity. Interval overlap preserves ambiguity if
                # more than one support contains part of a sampled plateau.
                candidates[previous_index, current_index] = (
                    previous['sampled_peak_right_s_m'] >= feature['left_s_m'] - spatial_tolerance
                    and previous['sampled_peak_left_s_m'] <= feature['right_s_m'] + spatial_tolerance
                    and feature['sampled_peak_right_s_m'] >= previous['left_s_m'] - spatial_tolerance
                    and feature['sampled_peak_left_s_m'] <= previous['right_s_m'] + spatial_tolerance)
        current_degree = candidates.sum(0)
        next_active = []
        assigned = set()
        for previous_index, track in enumerate(active):
            choices = np.flatnonzero(candidates[previous_index])
            if len(choices) == 1 and current_degree[choices[0]] == 1:
                index = int(choices[0])
                track['observations'].append((frame, current[index]))
                assigned.add(index)
                next_active.append(track)
            else:
                ambiguous = len(choices) > 0
                track['end_reason'] = 'ambiguous_matching' if ambiguous else 'no_mutual_support'
                track['ambiguous'] |= ambiguous
                if ambiguous:
                    ambiguity[frame - 1:frame + 1] = True
        for index, feature in enumerate(current):
            if index in assigned:
                continue
            ambiguous = current_degree[index] > 0
            reason = 'ambiguous_matching' if ambiguous else ('record_start' if frame == 0 else 'new_or_unresolved_feature')
            next_active.append(start_track(frame, feature, reason))
            ambiguity[frame] |= ambiguous
        active = next_active
    for track in active:
        track['end_reason'] = 'record_end'

    for track in tracks:
        observations = track.pop('observations')
        indices = np.array([item[0] for item in observations], dtype=np.int64)
        track['frame_indices'], track['time_s'] = indices, time[indices].copy()
        for key in ('peak_s_m', 'sampled_peak_left_s_m', 'sampled_peak_right_s_m', 'excess_turn_rad', 'width_m', 'prominence_per_m', 'left_s_m', 'right_s_m', 'peak_estimate'):
            track[key] = np.array([item[1][key] for item in observations])
        travel = float(track['peak_s_m'][-1] - track['peak_s_m'][0])
        duration = float(time[indices[-1]] - time[indices[0]])
        end, stop = int(indices[-1]), int(indices[-1]) + 1
        while stop < len(time) and valid[stop] and time[stop] <= time[end] + tip_lag_s + 16 * np.finfo(float).eps * max(1., abs(time[end])):
            stop += 1
        lag_speed = speed[end:stop]
        finite = np.isfinite(lag_speed)
        peak_index = end + int(np.nanargmax(lag_speed)) if finite.any() else None
        track.update(net_travel_m=travel, duration_s=duration,
            mean_rate_mps=travel / duration if duration > 0 else np.nan,
            resolved_link_count=len(indices) - 1,
            tip_speed_at_end_mps=float(speed[end]),
            tip_speed_end_lag_max_mps=float(speed[peak_index]) if peak_index is not None else np.nan,
            tip_speed_end_lag_peak_time_s=float(time[peak_index]) if peak_index is not None else np.nan,
            tip_lag_observed_duration_s=float(time[stop - 1] - time[end]))
    return dict(time_s=time.copy(), curvature_per_m=curvature, turning_angle_rad=angle,
        material_s_m=coordinate.copy(), sample_valid=valid, tip_velocity_mps=velocity,
        tip_speed_mps=speed, features_by_frame=feature_frames, tracks=tracks,
        ambiguity_by_frame=ambiguity, numerical_floor_per_m=floor,
        coordinate_roundoff_bound_per_m=coordinate_uncertainty,
        roundoff_unresolved_by_frame=roundoff_unresolved,
        metadata=dict(tip_lag_s=float(tip_lag_s), smoothing='none',
            curvature='turning angle / rest dual length',
            peak_position='three-point quadratic estimate or plateau midpoint; original piecewise-linear lobe integration',
            numerical_floor='128*eps*max(1/min(rest_dual),max(abs(curvature))) + 2*max(coordinate_roundoff_bound), per valid frame; not a noise or physical threshold',
            coordinate_roundoff_assumption='one-rounding input coordinate/subtraction uncertainty gamma1*(norm(p0)+norm(p1)+edge_length), gamma1=(eps/2)/(1-eps/2); direction uncertainty asin(delta_edge/edge_length), pi if unresolved; neighboring angles add and divide by rest dual. Does not bound unknown upstream error.',
            association='unique mutual-support matches of sampled peak/plateau intervals on adjacent valid rows; quadratic positions descriptive only; finite timestamp gaps not auto-detected',
            boundary_policy='endpoint maxima/truncated lobes/lone endpoint troughs excluded; flat trough plateaus observed in the interior are allowed',
            interpretation='descriptive features only; no features/no resolved link does not mean non-whip; ambiguity is not bridged'))
