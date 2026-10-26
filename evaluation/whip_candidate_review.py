"""Offline challenge of a provisional release/focusing hypothesis.

This module is not imported by training. Candidate support is NOT a physical
whip label or permission to promote a policy. In particular, a boundary peak
may expose a bad event window, rather than prove absence of whipping.
"""
import numpy as np

from evaluation.whip_measurements import extract_measurements
from evaluation.whip_output_decomposition import decompose
from tasks.whip.shape_release import score_events


VERSION = 'shape_release_focus_candidate_v1'
DISTAL_FRACTION = .75


def distal_motion_fraction(shape_velocity, rest_lengths):
    """Length-weighted squared shape-speed share, NOT physical kinetic energy."""
    u, rest = np.asarray(shape_velocity, float), np.asarray(rest_lengths, float)
    if (u.ndim != 3 or u.shape[-1] != 3 or rest.shape != (u.shape[1]-1,)
            or not np.isfinite(u).all() or not np.isfinite(rest).all()
            or np.any(rest <= 0)):
        raise ValueError('Finite time/node/xyz velocities and positive rest edges required')
    weights = np.r_[rest[0], rest[:-1]+rest[1:], rest[-1]]/(2*rest.sum())
    material = np.r_[0.,np.cumsum(rest)]/rest.sum()
    motion = np.square(u).sum(-1)*weights
    denominator = motion.sum(1)
    numerator = motion[:,material >= DISTAL_FRACTION].sum(1)
    return np.divide(numerator,denominator,out=np.zeros_like(denominator),
                     where=denominator > np.finfo(float).tiny)


def event_checks(event, fraction):
    """Return explicit reasons; no threshold fitted to a case name or Y motion."""
    if not event['finalized'] or not event['provisional_eligible'] or event['score'] <= 0:
        return dict(candidate_support=False, reasons=['no_positive_finalized_shape_release'])
    baseline, arm, q, end = (event[k] for k in
                            ('baseline_index','arming_index','q_shape_index','end_index'))
    if not 0 <= baseline < arm <= q <= end < len(fraction):
        raise ValueError('Invalid frozen event indices')
    change = float(fraction[q]-fraction[baseline])
    eps = 64*np.finfo(float).eps
    reasons = []
    if q == arm or q == end:
        reasons.append('boundary_censored_release_peak')
    if change <= eps:
        reasons.append('no_increase_in_distal_shape_motion_fraction')
    return dict(candidate_support=not reasons, reasons=reasons,
                baseline_distal_fraction=float(fraction[baseline]),
                release_distal_fraction=float(fraction[q]),
                distal_fraction_change=change, peak_strictly_inside_window=arm < q < end)


def evaluate(arrays, rest_lengths, direction, *, complete=True, failed=False):
    """Inspect only the finite observed prefix, retaining the old score separately."""
    if type(complete) is not bool or type(failed) is not bool:
        raise ValueError('complete and failed must be explicit booleans')
    time = np.asarray(arrays['time_s'],float)
    p, v = (np.asarray(arrays[k],float) for k in ('cable_positions','cable_velocities'))
    if time.ndim != 1 or p.ndim != 3 or p.shape != v.shape or p.shape[0] != len(time):
        raise ValueError('Matching time/node/xyz position and velocity histories required')
    observed = np.asarray(arrays.get('state_valid',np.ones(len(time),bool)))
    if observed.dtype != np.bool_ or observed.shape != time.shape:
        raise ValueError('state_valid must be a boolean time vector')
    finite = observed & np.isfinite(p).all((1,2)) & np.isfinite(v).all((1,2))
    count = int(np.cumprod(finite).sum())
    if count == 0:
        raise ValueError('A finite observed initial state is required')
    prefix = dict(time_s=time[:count],cable_positions=p[:count],cable_velocities=v[:count])
    m = extract_measurements(prefix,rest_lengths,direction)
    d = decompose(prefix['cable_positions'],prefix['cable_velocities'],rest_lengths)
    core = score_events(m,d,direction,complete=complete and count==len(time) and not failed,
                        failed=failed)
    fraction = distal_motion_fraction(d['shape_velocity_world_mps'],rest_lengths)
    events = []
    best, selected = 0., None
    for e in core['events']:
        checks = event_checks(e,fraction)
        value = e['score'] if checks['candidate_support'] else 0.
        events.append(dict(e,**checks,candidate_output=value))
        if value > best:
            best, selected = value,len(events)-1
    usable = complete and count==len(time) and not failed
    return dict(schema=VERSION, observed_samples=count, complete=usable, failed=failed,
                legacy_proxy_score=core['best_event_score'],
                candidate_output=best if usable else 0.,
                candidate_support=bool(usable and best > 0), selected_candidate_event=selected,
                legacy_selected_event=core['selected_event'],events=events,
                physical_whip_status='unvalidated', goal_acceptance='unresolved',
                limitations=['Boundary rejection can reflect timing censoring, not non-whipping.',
                    'Distal share is a geometric velocity diagnostic, not energy transfer.',
                    'These conditions have no independently reviewed physical positive set.',
                    'No sideways-motion penalty or automatic physical-whip label is applied.'])
