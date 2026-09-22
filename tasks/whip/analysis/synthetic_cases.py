"""Controlled inextensible kinematics; none are dynamically validated commands."""

import numpy as np


def _smoothstep(value):
    """Compact C2 transition with exact constant plateaus outside [0, 1]."""
    q = np.clip(value, 0.0, 1.0)
    return q**3 * (10.0 + q * (-15.0 + 6.0 * q))


def _positions(material_s, frame_count, bends=(), rotation=0.0):
    """Integrate edge tangents, keeping every finite edge exactly its rest length.

    A bend is (center[T], integrated_turn[T], full_support_width). Its
    continuum curvature is the compact derivative of a quintic transition.
    The returned polyline is a spatial discretization, not a dynamics rollout.
    """
    edge_s = (material_s[:-1] + material_s[1:]) / 2.0
    angles = np.broadcast_to(np.asarray(rotation), (frame_count,)).copy()[:, None]
    angles = np.broadcast_to(angles, (frame_count, len(edge_s))).copy()
    for center, strength, width in bends:
        center = np.broadcast_to(np.asarray(center), (frame_count,))
        strength = np.broadcast_to(np.asarray(strength), (frame_count,))
        q = (edge_s[None, :] - center[:, None]) / width + 0.5
        angles += strength[:, None] * _smoothstep(q)
    edges = np.stack((np.sin(angles), np.zeros_like(angles), -np.cos(angles)), axis=-1)
    edges *= np.diff(material_s)[None, :, None]
    positions = np.zeros((frame_count, len(material_s), 3), dtype=np.float64)
    positions[:, 1:] = np.cumsum(edges, axis=1)
    positions += np.array([0.0, 0.0, 2.0])
    return positions


def controlled_cases():
    """Return independent dictionaries with time[T], positions[T,101,3], s[101].

    All arrays are float64, seconds/metres. `expected` is JSON-serializable
    analytic ground truth, not a tolerance or a preassigned whip label.
    All cases use 10-ms samples except the exact time-doubled slow comparison.
    The missing-frame case deliberately contains one all-NaN position frame.
    """
    material_s = np.linspace(0.0, 1.0, 101, dtype=np.float64)
    time = np.arange(101, dtype=np.float64) * 0.01
    phase = time / time[-1]
    cases = []

    def add(name, title, positions, description, expected, *, time_s=time):
        cases.append(dict(name=name, title=title, time_s=time_s.copy(),
            positions_m=positions.copy(), material_s_m=material_s.copy(),
            description=description + ' Prescribed kinematics; dynamic feasibility is not established.',
            expected=dict(kinematic_only=True, rest_edge_length_m=0.01,
                          spatial_step_m=0.01, **expected)))

    rotation = 0.7 * np.sin(2.0 * np.pi * phase)
    add('straight_rigid_swing', 'Straight rigid swing',
        _positions(material_s, len(time), rotation=rotation),
        'A straight cable rotates rigidly about its first node; tip motion alone is not bend transport.',
        dict(integrated_turn_rad=0.0, bend_transport=False))
    add('curved_rigid_swing', 'Curved rigid swing',
        _positions(material_s, len(time), [(0.5, 0.8, 0.20)], rotation),
        'An existing curved shape rotates rigidly; its bend stays at fixed material position.',
        dict(bend_centers_m=[0.5], integrated_turn_rad=0.8,
             center_travel_m=0.0, center_speed_mps=0.0))
    add('stationary_strength_change', 'Stationary bend changing strength',
        _positions(material_s, len(time), [(0.5, 0.3 + 0.7 * _smoothstep(phase), 0.20)]),
        'One fixed-center bend grows smoothly without travelling along the cable.',
        dict(bend_centers_m=[0.5], integrated_turn_range_rad=[0.3, 1.0],
             center_travel_m=0.0, center_speed_mps=0.0))

    swap = _smoothstep(phase)
    add('separated_lobes_swap', 'Separated stationary lobes swapping strength',
        _positions(material_s, len(time), [(0.3, 1.0 - 0.6 * swap, 0.16),
                                           (0.7, 0.4 + 0.6 * swap, 0.16)]),
        'Both lobes remain present. The strongest peak switches identity; it must not become one travelling track.',
        dict(bend_centers_m=[0.3, 0.7], integrated_turn_rad=1.4,
             component_center_travel_m=[0.0, 0.0], strongest_peak_switch=True))
    add('overlapping_lobes_swap', 'Overlapping fixed lobes swapping strength',
        _positions(material_s, len(time), [(0.46, 1.0 - 0.85 * swap, 0.24),
                                           (0.54, 0.15 + 0.85 * swap, 0.24)]),
        'Fixed component centers can produce a genuinely moving combined curvature maximum. '
        'Kinematic peak progression here is not automatically a false positive or evidence of a transported physical wave.',
        dict(component_centers_m=[0.46, 0.54], integrated_turn_rad=1.15,
             component_center_travel_m=[0.0, 0.0], combined_peak_may_travel=True))

    centers = 0.3 + 0.4 * phase
    travel = _positions(material_s, len(time), [(centers, 1.0, 0.16)])
    travel_expected = dict(center_start_m=0.3, center_end_m=0.7,
        center_travel_m=0.4, integrated_turn_rad=1.0, full_support_width_m=0.16,
        continuum_peak_curvature_per_m=1.875 / 0.16)
    add('travelling_bend_fast', 'Travelling finite-width bend', travel,
        'A constant-strength compact bend travels from material position 0.30 to 0.70 m in one second.',
        dict(**travel_expected, center_speed_mps=0.4, sample_interval_s=0.01))
    add('travelling_bend_slow', 'Same bend with doubled time', travel,
        'Exactly the same spatial/phase samples as the fast case, with doubled timestamps; '
        'sampling is 20 ms here, not a resampled 100-Hz record.',
        dict(**travel_expected, center_speed_mps=0.2, sample_interval_s=0.02,
             identical_geometry_to='travelling_bend_fast'), time_s=2.0 * time)
    add('tiny_travelling_ripple', 'Tiny travelling ripple',
        _positions(material_s, len(time), [(centers, 0.001, 0.16)]),
        'The same bend position, width and speed with 0.001 times the integrated turn; '
        'normalization can hide this absolute-strength difference.',
        dict(center_start_m=0.3, center_end_m=0.7, center_travel_m=0.4,
             center_speed_mps=0.4, integrated_turn_rad=0.001,
             full_support_width_m=0.16, continuum_peak_curvature_per_m=0.001 * 1.875 / 0.16,
             strength_ratio_to_fast=0.001))

    missing = travel.copy()
    missing[50] = np.nan
    add('missing_frame', 'Travelling bend with one missing frame', missing,
        'Frame 50 is missing rather than interpolated; a track must not silently bridge that unavailable observation.',
        dict(nonfinite_frame_indices=[50], required_track_break_frames=[50],
             inextensible_on_finite_frames=True))
    first = 1.0 - _smoothstep((phase - 0.25) / 0.15)
    second = _smoothstep((phase - 0.60) / 0.15)
    add('disjoint_reappearance', 'Separated disappearance and reappearance',
        _positions(material_s, len(time), [(0.3, first, 0.16), (0.7, second, 0.16)]),
        'A stationary bend fades away, followed by a straight interval, then a different stationary bend appears downstream. '
        'There is no observed connection across the interval.',
        dict(component_centers_m=[0.3, 0.7], component_center_travel_m=[0.0, 0.0],
             straight_frame_interval_inclusive=[40, 60], must_not_join_components=True))
    return cases
