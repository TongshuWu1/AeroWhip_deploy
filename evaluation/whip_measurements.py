"""Descriptive cable kinematics from saved samples, without a whip classifier.

Material coordinates and rigid-fit quadrature weights use the supplied rest
lengths. Curvature instead uses the *measured* edge lengths. No smoothing,
resampling, prescribed sample rate, target reward, or physical success threshold
is applied. Node zero is the cable attachment, which need not be the drone CoM.
"""

import numpy as np


def _finite_array(value, name):
    """Convert real numeric inputs without silently accepting complex values."""
    raw = np.asarray(value)
    if raw.dtype.kind not in 'fiu':
        raise ValueError(f'{name} must contain real numbers')
    result = np.asarray(raw, dtype=np.float64)
    if not np.isfinite(result).all():
        raise ValueError(f'{name} must be finite')
    return result


def _norm(value, axis=-1):
    # hypot avoids avoidable overflow/underflow from squaring the components.
    return np.hypot.reduce(value, axis=axis)


def extract_measurements(arrays, rest_lengths, direction):
    """Return NumPy measurements of one saved ``(time, node, xyz)`` trajectory.

    Required fields are ``time_s``, ``cable_positions`` and
    ``cable_velocities``. Optional ``state_valid`` must be an all-true boolean
    time vector: invalid/padded data must be dealt with explicitly by the caller.
    At least one time sample and three cable nodes are required. Supplied times
    are preserved, including their origin and any nonuniform sampling.

    ``node_s_m`` contains all N cumulative rest coordinates; ``material_s_m``
    contains the N-2 interior coordinates where curvature is defined. Curvature
    is turning angle / adjacent-edge mean measured length. ``turn_rad`` is its
    spatial integral with those dual lengths, i.e. the sum of turning angles.
    ``curvature_dual_length_m`` records those measured integration weights.
    Curvature at the physical clamp and free endpoint is not estimated.

    All speed and forward-component arrays have shape (T,N). Relative fields
    subtract attachment velocity. ``nonrigid_speed_mps`` is the residual from a
    per-frame least-squares rigid velocity field v = V + omega cross (p-c),
    weighted by material-length trapezoidal weights. These are geometric length
    weights, not physical masses. ``rigid_fit_rank`` includes translation plus
    rotation; a straight cable has an unobservable axial rotation and rank five,
    but its fitted nodal rigid velocities remain well-defined.
    """
    try:
        time = _finite_array(arrays['time_s'], 'time_s')
        positions = _finite_array(arrays['cable_positions'], 'cable_positions')
        velocities = _finite_array(arrays['cable_velocities'], 'cable_velocities')
    except KeyError as exc:
        raise ValueError(f'Missing replay field: {exc.args[0]}') from exc
    if time.ndim != 1 or len(time) < 1 or not np.all(np.diff(time) > 0):
        raise ValueError('time_s must be a nonempty strictly increasing vector')
    if (positions.ndim != 3 or positions.shape[0] != len(time)
            or positions.shape[1] < 3 or positions.shape[2] != 3):
        raise ValueError('cable_positions must have shape (time, nodes>=3, 3)')
    if velocities.shape != positions.shape:
        raise ValueError('cable_velocities must match cable_positions shape')
    if 'state_valid' in arrays:
        valid = np.asarray(arrays['state_valid'])
        if valid.dtype != np.bool_ or valid.shape != time.shape or not valid.all():
            raise ValueError('state_valid must be an all-true boolean time vector')
    rest = _finite_array(rest_lengths, 'rest_lengths')
    if rest.shape != (positions.shape[1]-1,) or not np.all(rest > 0):
        raise ValueError('rest_lengths must contain one positive length per edge')
    forward = _finite_array(direction, 'direction')
    if forward.shape != (3,) or np.max(np.abs(forward)) == 0:
        raise ValueError('direction must be a nonzero three-vector')
    forward = forward / np.max(np.abs(forward))
    forward = forward / _norm(forward)

    edges = np.diff(positions, axis=1)
    lengths = _norm(edges)
    if not np.isfinite(lengths).all() or not np.all(lengths > 0):
        raise ValueError('Every measured cable edge must have positive finite length')
    tangents = edges / lengths[..., None]
    cross = _norm(np.cross(tangents[:, :-1], tangents[:, 1:]))
    dot = np.einsum('tni,tni->tn', tangents[:, :-1], tangents[:, 1:])
    angles = np.arctan2(cross, dot)
    dual_lengths = .5*lengths[:, :-1] + .5*lengths[:, 1:]
    arc = np.sum(lengths, axis=1)
    node_s = np.concatenate(([0.], np.cumsum(rest)))
    if (not np.isfinite(arc).all() or not np.isfinite(node_s).all()
            or not np.all(np.diff(node_s) > 0) or not np.all(dual_lengths > 0)):
        raise ValueError('Cable length scale exceeds usable floating-point range')
    relative = velocities - velocities[:, :1]

    normalized_rest = rest/node_s[-1]
    weights = np.concatenate((normalized_rest[:1]/2,
                              normalized_rest[:-1]/2+normalized_rest[1:]/2,
                              normalized_rest[-1:]/2))
    weights /= np.sum(weights)
    sqrt_weights = np.repeat(np.sqrt(weights), 3)
    residual = np.empty(positions.shape[:2], dtype=np.float64)
    ranks = np.empty(len(time), dtype=np.int64)
    for frame, (p, v) in enumerate(zip(positions, velocities)):
        # Subtract the attachment first to improve conditioning far from origin.
        offsets = p-p[0]
        centered = offsets-np.sum(weights[:, None]*offsets, axis=0)
        mean_v = np.sum(weights[:, None]*v, axis=0)
        x, y, z = centered.T
        design = np.zeros((len(p), 3, 3), dtype=np.float64)
        design[:, 0, 1], design[:, 0, 2] = z, -y
        design[:, 1, 0], design[:, 1, 2] = -z, x
        design[:, 2, 0], design[:, 2, 1] = y, -x
        omega, _, rank, _ = np.linalg.lstsq(
            design.reshape(-1, 3)*sqrt_weights[:, None],
            (v-mean_v).reshape(-1)*sqrt_weights, rcond=None)
        residual[frame] = _norm(v-mean_v-np.cross(omega, centered))
        ranks[frame] = rank+3

    result = dict(
        time_s=time.copy(), material_s_m=node_s[1:-1].copy(), node_s_m=node_s,
        curvature_per_m=angles/dual_lengths,
        curvature_dual_length_m=dual_lengths, turn_rad=np.sum(angles, axis=1),
        arc_length_m=arc, chord_ratio=_norm(positions[:, -1]-positions[:, 0])/arc,
        world_speed_mps=_norm(velocities), relative_speed_mps=_norm(relative),
        world_forward_mps=velocities@forward, relative_forward_mps=relative@forward,
        nonrigid_speed_mps=residual, rigid_fit_rank=ranks)
    if any(not np.isfinite(value).all() for value in result.values()):
        raise ValueError('Measurements exceed finite floating-point range')
    return result


def _peaks(row, tolerance):
    """Return (index, flags) candidates without selecting a plateau center.

    A positive, flat entire profile has no identifiable maximum location. Local
    plateaus produce separate, explicitly ambiguous candidates at all samples;
    these cannot continue tracks. Boundary maxima are one-sided and flagged.
    """
    if np.max(row) <= tolerance:
        return []
    if len(row) > 1 and np.max(row)-np.min(row) <= tolerance:
        return []
    groups = []
    start = 0
    for index in range(1, len(row)+1):
        if index == len(row) or abs(row[index]-row[index-1]) > tolerance:
            groups.append((start, index))
            start = index
    result = []
    for start, end in groups:
        value = np.min(row[start:end])
        if value <= tolerance:
            continue
        if start and value <= row[start-1]+tolerance:
            continue
        if end < len(row) and value <= row[end]+tolerance:
            continue
        for index in range(start, end):
            flags = []
            if end-start > 1:
                flags.append('ambiguous_plateau')
            if index == 0 or index == len(row)-1:
                flags.append('boundary_peak')
            result.append((index, flags))
    return result


def track_curvature(measurements):
    """Track identifiable local curvature maxima conservatively, not whip events.

    A match can join only consecutive recorded frames and move at most one
    interior spatial sample. It must be unique in both directions. Entire
    bipartite components containing a split/merge are broken and flagged, so a
    nearest-neighbor tie never silently invents continuity. Local plateaus are
    retained as isolated ambiguous samples. Gaps never reconnect; reflected or
    backward tracks are retained without preference for propagation direction.

    The sole amplitude/comparison tolerance is a floating-point algorithmic
    floor: 32*float64 epsilon*max(profile amplitude, 1/minimum material dual
    length). It suppresses zero/flat numerical profiles; it is not a physical
    bend, significance, success, or noise threshold. No smoothing is used.

    Each output dictionary has JSON-compatible lists: ``indices`` are frame
    indices; ``node_indices`` are original full-cable indices (1 through N-2);
    ``position_m`` uses material position; ``amplitude_per_m`` is raw curvature.
    ``flags`` combines track-wide cautions and ``start_reason``/``end_reason``
    identify why continuity was not assumed. A one-frame track is valid output.
    """
    time = _finite_array(measurements['time_s'], 'time_s')
    material = _finite_array(measurements['material_s_m'], 'material_s_m')
    nodes = _finite_array(measurements['node_s_m'], 'node_s_m')
    curvature = _finite_array(measurements['curvature_per_m'], 'curvature_per_m')
    if time.ndim != 1 or not len(time) or not np.all(np.diff(time) > 0):
        raise ValueError('time_s must be a nonempty strictly increasing vector')
    if (nodes.ndim != 1 or len(nodes) < 3 or not np.all(np.diff(nodes) > 0)
            or material.shape != nodes[1:-1].shape
            or not np.array_equal(material, nodes[1:-1])):
        raise ValueError('material_s_m must be the interior node_s_m coordinates')
    if curvature.shape != (len(time), len(material)) or np.any(curvature < 0):
        raise ValueError('curvature_per_m must be nonnegative with shape (time, interiors)')
    min_dual = np.min((nodes[2:]-nodes[:-2])/2)
    inverse_length = 1/min_dual
    tracks = []
    previous = []

    def add_flags(track, flags):
        track['flags'] = sorted(set(track['flags']).union(flags))

    def append_sample(track, frame, index, flags):
        track['indices'].append(int(frame))
        track['node_indices'].append(int(index+1))
        track['position_m'].append(float(material[index]))
        track['amplitude_per_m'].append(float(curvature[frame, index]))
        add_flags(track, flags)

    for frame, row in enumerate(curvature):
        tolerance = 32*np.finfo(np.float64).eps*max(float(np.max(row)), inverse_length)
        current = _peaks(row, tolerance)
        left = [[j for j, (index, _) in enumerate(current) if abs(old[0]-index) <= 1]
                for old in previous]
        right = [[i for i, neighbors in enumerate(left) if j in neighbors]
                 for j in range(len(current))]
        # Flood candidate components to propagate ambiguity to every member,
        # including nodes with degree one attached to a larger split/merge.
        left_flags, right_flags = {}, {}
        seen = set()
        for seed in range(len(previous)):
            if seed in seen or not left[seed]:
                continue
            group_left, group_right, pending = set(), set(), [seed]
            while pending:
                i = pending.pop()
                if i in group_left:
                    continue
                group_left.add(i)
                for j in left[i]:
                    group_right.add(j)
                    pending.extend(k for k in right[j] if k not in group_left)
            seen.update(group_left)
            flags = []
            if any(len(left[i]) > 1 for i in group_left):
                flags.append('ambiguous_split')
            if any(len(right[j]) > 1 for j in group_right):
                flags.append('ambiguous_merge')
            if any('ambiguous_plateau' in current[j][1] for j in group_right):
                flags.append('ambiguous_plateau')
            if any('ambiguous_plateau' in previous[i][2] for i in group_left):
                flags.append('ambiguous_plateau')
            for i in group_left:
                left_flags[i] = flags
            for j in group_right:
                right_flags[j] = flags

        next_previous = []
        continued = set()
        for j, (index, sample_flags) in enumerate(current):
            flags = list(set(sample_flags).union(right_flags.get(j, [])))
            unique = len(right[j]) == 1 and len(left[right[j][0]]) == 1
            ambiguous = any(flag.startswith('ambiguous_') for flag in flags)
            if unique and not ambiguous:
                old_index = right[j][0]
                track = previous[old_index][1]
                continued.add(old_index)
            else:
                reason = ('ambiguous_association' if ambiguous else
                          'recording_start' if frame == 0 else 'no_adjacent_peak')
                track = dict(id=len(tracks), indices=[], node_indices=[], position_m=[],
                             amplitude_per_m=[], flags=[], start_reason=reason,
                             end_reason=None)
                tracks.append(track)
            append_sample(track, frame, index, flags)
            next_previous.append((index, track, sample_flags))
        for i, (_, track, _) in enumerate(previous):
            if i not in continued:
                flags = left_flags.get(i, [])
                add_flags(track, flags)
                track['end_reason'] = ('ambiguous_association' if flags
                                       else 'no_adjacent_peak')
        previous = next_previous
    for _, track, _ in previous:
        track['end_reason'] = 'recording_end'
    return tracks
