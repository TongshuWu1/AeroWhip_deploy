"""One experimental fold attempt: formation, propagation, aligned release.

Partial preparation credit is never a release-success certificate. The release
detector and output definition are shared with the preserved aligned-fold task.
"""
import numpy as np

from .fold_formation import aligned_release_events, reversal_angles
from .fold_release import fold_geometry
from .shape_release import ShapeReleaseScorer


MODE = 'directed_unified_fold_v1'


def score_events(arrays, measurements, decomposition, direction, *, complete, failed):
    core = aligned_release_events(arrays, measurements, decomposition, direction,
                                  complete=complete, failed=failed)
    t = np.asarray(measurements['time_s'])
    nodes = np.asarray(measurements['node_s_m'])
    fractions = nodes/nodes[-1]
    root = np.flatnonzero(fractions[1:-1] <= .35)
    if not len(root):
        raise ValueError('Unified fold needs a root-side interior node')
    angles = reversal_angles(arrays['cable_positions'])
    reversal, turn = fold_geometry(arrays['cable_positions'])
    speed = decomposition['shape_speed_mps'][:, -1]
    d = np.asarray(direction, float); d = d/np.linalg.norm(d)
    forward = decomposition['shape_velocity_world_mps'][:, -1] @ d
    world = measurements['world_forward_mps'][:, -1]
    causal = np.zeros(len(t)); records = []

    def record(baseline, end, formation, progress, *, track_id=-1, path=(), release=None):
        q = release['q_shape_index'] if release is not None and release['finalized'] else end
        output = release['score'] if release is not None else 0.
        row = dict(id=len(records), segment_id=track_id, baseline_index=baseline,
                   arming_index=(release['arming_index'] if release is not None else end),
                   requested_end_index=end, end_index=end, q_shape_index=q,
                   evidence_indices=[i for i, j in path], evidence_node_indices=[j for i, j in path],
                   truncated=False, finalized=True, provisional_eligible=output > 0,
                   formation_fraction=float(formation), propagation_fraction=float(progress),
                   aligned_release_score=float(output), completed_aligned_release=output > 0,
                   score=float(.5*formation+.5*progress+output), release_evidence=release,
                   q_shape_mps=float(speed[q]), shape_baseline_mps=float(speed[baseline]),
                   shape_gain_mps=float(speed[q]-speed[baseline]),
                   shape_forward_mps=float(forward[q]), world_forward_mps=float(world[q]),
                   relative_tip_speed_mps=float(measurements['relative_speed_mps'][q, -1]),
                   rejection_reason=None if output > 0 else 'preparation_credit_only')
        records.append(row)
        return row

    # Each root observation is a formation-only candidate. Keep new records
    # only on improvement; the causal guide retains every observed prefix.
    formation = np.clip(angles[:, root].max(1)/(np.pi/2), 0., 1.)
    causal[:] = np.maximum.accumulate(.5*formation)
    previous = -1.
    for frame, value in enumerate(formation):
        if value > previous:
            record(frame, frame, value, 0.)
            previous = float(value)

    releases = {(e['segment_id'], e['baseline_index'], e['arming_index']): e for e in core['events']}
    # Walk the SAME conservative material tracks and reset rules as fold_release.
    # Incomplete attempts now earn observed progress without inventing a release.
    for track in core['segments']:
        baseline = None; path = []; distal = False; previous_node = None

        def finish_attempt(release=None):
            if not path:
                return
            start_node = path[0][1]
            progress = float(np.clip((nodes[path[-1][1]]-nodes[start_node]) /
                                    (nodes[-2]-nodes[start_node]), 0., 1.))
            end = release['end_index'] if release is not None and release['finalized'] else path[-1][0]
            row = record(baseline, end, 1., progress, track_id=track['id'], path=path, release=release)
            causal[end:] = np.maximum(causal[end:], row['score'])

        for frame, node in zip(track['indices'], track['node_indices']):
            if (previous_node is not None and node < previous_node
                    or turn[frame, node-1] < .10
                    or baseline is not None and t[frame]-t[baseline] > .60+1e-10):
                finish_attempt()
                baseline = None; path = []; distal = False
            previous_node = node
            if baseline is None:
                if fractions[node] > .35 or not reversal[frame, node-1] or turn[frame, node-1] < .10:
                    continue
                baseline = frame; path = []; distal = False
            path.append((frame, node))
            progress = float(np.clip((nodes[node]-nodes[path[0][1]])/(nodes[-2]-nodes[path[0][1]]), 0., 1.))
            causal[frame:] = np.maximum(causal[frame:], .5+.5*progress)
            distal |= bool(fractions[node] >= .65 and reversal[frame, node-1])
            if node == len(nodes)-2 and distal:
                finish_attempt(releases[(track['id'], baseline, frame)])
                baseline = None; path = []; distal = False
        finish_attempt()

    records.sort(key=lambda e: (e['end_index'], e['baseline_index'], e['segment_id']))
    for index, row in enumerate(records): row['id'] = index
    selected = int(np.argmax([e['score'] for e in records])) if records else None
    best = records[selected]['score'] if selected is not None else 0.
    if not np.isclose(causal[-1], best, rtol=0, atol=1e-12):
        raise AssertionError('Attempt records and causal utility disagree')
    # Initial preparation, if present, is paid on the first transition.
    causal[0] = 0.
    return dict(schema=MODE, events=records, selected_event=selected, segments=core['segments'],
                best_event_score=best, causal_best=causal, complete=complete, failed=failed,
                episode_return=-10. if failed else best if complete else None,
                maximum_fold_progress=core['maximum_fold_progress'],
                completed_aligned_release=bool(selected is not None and records[selected]['completed_aligned_release']),
                mechanism_status='unresolved',
                rules=dict(version=MODE, utility='max_same_attempt(0.5F+0.5P+S)',
                           preparation_is_not_success=True, release_rules=core['rules']))


class UnifiedFoldScorer(ShapeReleaseScorer):
    def evaluate_events(self, arrays, measurements, decomposition, direction, *, complete, failed):
        return score_events(arrays, measurements, decomposition, direction, complete=complete, failed=failed)

    def __call__(self, *args, **kwargs):
        score = super().__call__(*args, **kwargs)
        score.components.pop('shape_release_score')
        score.components['unified_fold_score'] = score.score
        for key in ('formation_fraction', 'propagation_fraction', 'aligned_release_score', 'completed_aligned_release'):
            values = [r['events'][r['selected_event']][key] if r['selected_event'] is not None else 0.
                      for r in self.event_records]
            score.components[key] = score.score.new_tensor(values).masked_fill(~score.valid, float('nan'))
        score.components['completed_release_fraction_all'] = (
            score.valid & (score.components['completed_aligned_release'] == 1)).to(score.score.dtype)
        return score
