"""Offline implementation of the provisional 2026-10-22 Stage1 contract.

This is a diagnostic, not an enabled training reward or a physical whip label.
Thresholds are fixed research choices. Complete valid horizons may clip release
windows; failed/incomplete recordings cannot finalize an unfinished release.
"""
from collections import Counter

import numpy as np

from evaluation.whip_measurements import extract_measurements, _peaks


RULES = dict(version='stage1_outward_release_candidate_v1', observation_period_s=.01,
             maximum_node_hop=1, start_fraction=.55, arrival_fraction=.75,
             minimum_turn_rad=.10, minimum_distinct_nodes=3,
             maximum_segment_s=.60, minimum_nonrigid_rms_mps=.10,
             release_window_s=.20, minimum_gain_mps=.25,
             speed_unit_mps=1., failure_return=-10.)


def detect(measurements, *, complete=True, failed=False):
    """Causal peak associations and frozen event evidence, on the 100-Hz mesh.

    Returns diagnostic segments/events plus offline 100-Hz reward increments.
    No past event is removed when a later association becomes ambiguous. No
    automatic mechanism_review='supported' is assigned. Input measurements are
    produced by extract_measurements; this lower-level API also permits unit
    fixtures that isolate timing/accounting independently of physical motion.
    """
    if type(complete) is not bool or type(failed) is not bool:
        raise ValueError('complete and failed must be explicit booleans')
    t = np.asarray(measurements['time_s'], float)
    nodes = np.asarray(measurements['node_s_m'], float)
    curvature = np.asarray(measurements['curvature_per_m'], float)
    dual = np.asarray(measurements['curvature_dual_length_m'], float)
    speed = np.asarray(measurements['relative_speed_mps'], float)[:, -1]
    forward = np.asarray(measurements['world_forward_mps'], float)[:, -1]
    residual = np.asarray(measurements['nonrigid_speed_mps'], float)
    if (t.ndim != 1 or not len(t) or not np.isfinite(t).all()
            or not np.allclose(np.diff(t), .01, rtol=0, atol=1e-10)):
        raise ValueError('Candidate requires consecutive 100-Hz observations')
    if (nodes.ndim != 1 or len(nodes) < 5 or nodes[0] != 0
            or not np.isfinite(nodes).all() or np.any(np.diff(nodes) <= 0)):
        raise ValueError('Need positive material edges and at least three interior nodes')
    if (curvature.shape != (len(t), len(nodes)-2) or dual.shape != curvature.shape
            or residual.shape != (len(t), len(nodes))
            or speed.shape != t.shape or forward.shape != t.shape
            or not all(np.isfinite(x).all() for x in (curvature, dual, residual, speed, forward))
            or np.any(curvature < 0) or np.any(dual <= 0)
            or np.any(speed < 0) or np.any(residual < 0)):
        raise ValueError('Invalid finite kinematic measurement shapes/values')
    rest = np.diff(nodes)
    weights = np.r_[rest[0], rest[:-1]+rest[1:], rest[-1]]/(2*nodes[-1])
    rms = np.sqrt((residual**2*weights).sum(-1))
    fractions = nodes/nodes[-1]
    inverse_length = 1/np.min((nodes[2:]-nodes[:-2])/2)
    segments, events, previous = [], [], []
    rejected = Counter()

    def end(segment, reason):
        if segment is not None and segment['end_reason'] is None:
            segment['end_reason'] = reason
            if segment['event_id'] is None:
                rejected[reason] += 1

    for frame, row in enumerate(curvature):
        tolerance = 32*np.finfo(float).eps*max(float(row.max()), inverse_length)
        peaks = _peaks(row, tolerance)
        left = [[j for j, (node, _) in enumerate(peaks) if abs(old[0]-node) <= 1]
                for old in previous]
        right = [[i for i, links in enumerate(left) if j in links] for j in range(len(peaks))]
        next_previous, continued = [], set()
        for j, (node, flags) in enumerate(peaks):
            incoming = right[j]
            unique = len(incoming) == 1 and len(left[incoming[0]]) == 1
            plateau = 'ambiguous_plateau' in flags
            ambiguous = plateau or (bool(incoming) and not unique)
            if unique and 'ambiguous_plateau' in previous[incoming[0]][1]:
                ambiguous = True
            segment = previous[incoming[0]][2] if unique and not ambiguous else None
            if segment is not None:
                continued.add(incoming[0])
            turn = row[node]*dual[frame, node]
            if ambiguous:
                rejected['ambiguous_peak_samples'] += 1
            if segment is not None:
                reason = ('backward_step' if node+1 < segment['node_indices'][-1] else
                          'turn_below_floor' if turn < RULES['minimum_turn_rad'] else
                          'duration_limit' if t[frame]-t[segment['start_index']] > .60+1e-10 else None)
                if reason:
                    end(segment, reason)
                    segment = None
            if segment is None and not ambiguous and turn >= .10 and fractions[node+1] <= .55:
                segment = dict(id=len(segments), start_index=frame, indices=[], node_indices=[],
                               maximum_nonrigid_rms_mps=0., event_id=None, end_reason=None)
                segments.append(segment)
            if segment is not None:
                segment['indices'].append(frame)
                segment['node_indices'].append(node+1)
                segment['maximum_nonrigid_rms_mps'] = max(segment['maximum_nonrigid_rms_mps'], float(rms[frame]))
                if (segment['event_id'] is None and fractions[node+1] >= .75
                        and len(set(segment['node_indices'])) >= 3
                        and segment['maximum_nonrigid_rms_mps'] >= .10):
                    segment['event_id'] = len(events)
                    events.append(dict(id=len(events), segment_id=segment['id'],
                        baseline_index=segment['start_index'], arming_index=frame,
                        evidence_indices=segment['indices'].copy(),
                        evidence_node_indices=segment['node_indices'].copy(),
                        evidence_nonrigid_rms_mps=segment['maximum_nonrigid_rms_mps']))
            next_previous.append((node, flags, segment))
        for i, (_, _, segment) in enumerate(previous):
            if i not in continued:
                end(segment, 'ambiguous_association' if left[i] else 'lost_peak')
        previous = next_previous
    for _, _, segment in previous:
        end(segment, 'recording_end')

    increments = np.zeros(len(t))
    best = 0.
    # Events are armed chronologically and all release windows have equal length.
    for event in events:
        start, baseline = event['arming_index'], event['baseline_index']
        requested_end = start+20
        event['requested_end_index'] = requested_end
        event['truncated'] = requested_end >= len(t)
        event['finalized'] = not event['truncated'] or (complete and not failed)
        event['provisional_eligible'] = False
        event['score'] = 0.
        if not event['finalized']:
            event['rejection_reason'] = 'unfinished_release'
            rejected['unfinished_release'] += 1
            continue
        stop = min(requested_end, len(t)-1)
        peak = start+int(np.argmax(speed[start:stop+1]))
        gain = float(speed[peak]-speed[baseline])
        eligible = gain >= .25
        score = float(min(speed[peak], max(0., forward[peak]))) if eligible else 0.
        reason = None if eligible else 'insufficient_relative_speed_gain'
        if reason:
            rejected[reason] += 1
        event.update(end_index=stop, q_index=peak, q_relative_tip_speed_mps=float(speed[peak]),
                     baseline_relative_tip_speed_mps=float(speed[baseline]), signed_gain_mps=gain,
                     forward_world_mps=float(forward[peak]), provisional_eligible=eligible,
                     score=score, rejection_reason=reason)
        improved = max(best, score)
        increments[stop] += improved-best
        best = improved
    if failed:
        increments[-1] += -10.-best
    return dict(schema=RULES['version'], rules=dict(RULES), events=events, segments=segments,
                rejection_counts=dict(rejected), best_event_score=best,
                episode_return=(-10. if failed else best if complete else None),
                reward_increments_100hz=increments.tolist(), complete=complete, failed=failed,
                mechanism_status='unresolved')


def evaluate(arrays, rest_lengths, direction, *, complete=True, failed=False):
    return detect(extract_measurements(arrays, rest_lengths, direction), complete=complete, failed=failed)
