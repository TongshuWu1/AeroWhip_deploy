"""Offline nodal speed-peak timing in one explicit shared sample interval.

These are sampled kinematic descriptions, not a wave detector, physical-whip
classifier, energy-transfer estimate or training signal. No amplitudes are
normalized, smoothed or discarded. The caller chooses the speed frame and the
interval; timing results from different frames are not interchangeable.
"""
import numpy as np


def _real_finite_array(value, name):
    raw = np.asarray(value)
    if raw.dtype.kind not in 'iuf':
        raise ValueError(f'{name} must contain real numeric values')
    result = np.asarray(raw, dtype=np.float64)
    if not np.isfinite(result).all():
        raise ValueError(f'{name} must contain finite values')
    return result


def describe_speed_peaks(time_s, speed_mps, *, start_index, end_index):
    """Describe every node's exact maxima in inclusive ``[start_index,end_index]``.

    ``speed_mps`` has shape (time, nodes), contains nonnegative speeds, and uses
    the same frame for every node. Supplied seconds are retained, including
    nonuniform sampling and nonzero time origins. All exact maximum sample
    indices/times are returned, including separated equal maxima. Boundary
    maxima are marked because the true peak may lie outside the chosen window.

    A numerical-constancy flag uses 64 float64 eps times the larger of 1 m/s and
    the maximum speed across all nodes in this window. This is solely a
    machine-precision diagnostic: it neither changes values nor defines a
    minimum physical speed. Near-constant nodes cannot identify peak timing.

    Adjacent ordering is conservative over all maximum samples. ``distal_later``
    requires the earliest distal maximum to follow the latest proximal maximum;
    ``distal_earlier`` is the reverse. ``tied`` denotes unique maxima at the same
    sample; all other intersecting peak-time spans are ``overlapping``. A flat,
    resting or numerically constant member makes the pair ``not_identifiable``.
    These orders describe sampled maxima, not propagation or causal transfer.
    """
    time = _real_finite_array(time_s, 'time_s')
    speed = _real_finite_array(speed_mps, 'speed_mps')
    if time.ndim != 1 or not len(time) or not np.all(time[1:] > time[:-1]):
        raise ValueError('time_s must be a nonempty strictly increasing vector')
    if (speed.ndim != 2 or speed.shape[0] != len(time) or speed.shape[1] < 1
            or np.any(speed < 0)):
        raise ValueError('speed_mps must be nonnegative with shape (time,nodes>=1)')
    if (any(isinstance(i, (bool, np.bool_)) or not isinstance(i, (int, np.integer))
            for i in (start_index, end_index))
            or not 0 <= start_index <= end_index < len(time)):
        raise ValueError('Need integer 0 <= start_index <= end_index < samples')
    start, end = int(start_index), int(end_index)
    selected = speed[start:end+1]
    maximum = float(np.max(selected))
    tolerance = float((64*np.finfo(np.float64).eps)*max(maximum, 1.))
    nodes = []
    for node, values in enumerate(selected.T):
        low, high = float(np.min(values)), float(np.max(values))
        indices = np.flatnonzero(values == high)+start
        flat = low == high
        constant = high-low <= tolerance
        nodes.append(dict(
            node_index=node, peak_speed_mps=high, minimum_speed_mps=low,
            speed_range_mps=high-low, peak_indices=indices.tolist(),
            peak_times_s=time[indices].tolist(), peak_count=int(len(indices)),
            first_peak_index=int(indices[0]), last_peak_index=int(indices[-1]),
            first_peak_time_s=float(time[indices[0]]),
            last_peak_time_s=float(time[indices[-1]]),
            flat=flat, zero=high == 0., numerically_constant=constant,
            boundary_peak=bool(indices[0] == start or indices[-1] == end),
            multiple_maxima=len(indices) > 1, peak_timing_identifiable=not constant))

    adjacent = []
    for proximal, distal in zip(nodes[:-1], nodes[1:]):
        identifiable = (proximal['peak_timing_identifiable']
                        and distal['peak_timing_identifiable'])
        lower = distal['first_peak_time_s']-proximal['last_peak_time_s']
        upper = distal['last_peak_time_s']-proximal['first_peak_time_s']
        if not identifiable:
            order = 'not_identifiable'
        elif distal['first_peak_index'] > proximal['last_peak_index']:
            order = 'distal_later'
        elif distal['last_peak_index'] < proximal['first_peak_index']:
            order = 'distal_earlier'
        elif proximal['peak_count'] == distal['peak_count'] == 1:
            order = 'tied'
        else:
            order = 'overlapping'
        adjacent.append(dict(
            proximal_node_index=proximal['node_index'],
            distal_node_index=distal['node_index'], order=order,
            minimum_peak_lag_s=lower if identifiable else None,
            maximum_peak_lag_s=upper if identifiable else None,
            peak_amplitude_difference_mps=distal['peak_speed_mps']-proximal['peak_speed_mps']))
    return dict(
        schema='offline_nodal_speed_peak_timing_v1', start_index=start, end_index=end,
        start_time_s=float(time[start]), end_time_s=float(time[end]),
        sample_count=end-start+1, numerical_constancy_tolerance_mps=tolerance,
        maximum_speed_in_window_mps=maximum, nodes=nodes, adjacent=adjacent)
