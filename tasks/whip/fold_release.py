"""Experimental outward-fold opening utility; not a physical-whip certificate.

Versioned separately from the preserved shape-release task. All event geometry
is material ordered, and output and opening belong to the same frozen event.
"""
import numpy as np

from evaluation.whip_measurements import track_curvature
from .shape_release import ShapeReleaseScorer


RULES = dict(version='directed_fold_release_v1', observation_period_s=.01,
             root_fraction=.35, distal_fold_fraction=.65,
             minimum_apex_turn_rad=.10, maximum_propagation_s=.60,
             output_lookback_s=.10, opening_window_s=.20,
             maximum_remaining_tail_turn_fraction=.50,
             minimum_shape_gain_mps=.25, speed_unit_mps=1., failure_return=-10.)


def fold_geometry(positions):
    """Local tangent reversal and actual turns; invariant to rigid pose changes."""
    p = np.asarray(positions, float)
    if p.ndim != 3 or p.shape[-1] != 3 or p.shape[1] < 5 or not np.isfinite(p).all():
        raise ValueError('Fold geometry requires finite time/node/xyz positions')
    edge = np.diff(p, axis=1)
    length = np.linalg.norm(edge,axis=-1)
    if np.any(length <= 0):
        raise ValueError('Fold geometry requires positive measured edges')
    tangent = edge/length[...,None]
    turns = np.arctan2(np.linalg.norm(np.cross(tangent[:,:-1],tangent[:,1:]),axis=-1),
                      np.einsum('tni,tni->tn',tangent[:,:-1],tangent[:,1:]))
    reversal = np.zeros_like(turns,dtype=bool)
    for j in range(1,p.shape[1]-1):
        left,right=max(0,j-2),min(len(tangent[0])-1,j+1)
        reversal[:,j-1] = np.einsum('ti,ti->t',tangent[:,left],tangent[:,right]) < -64*np.finfo(float).eps
    return reversal, turns


def score_events(arrays, measurements, decomposition, direction, *, complete, failed):
    """Finalize a release only after its observed opening window, without clipping."""
    if type(complete) is not bool or type(failed) is not bool:
        raise ValueError('complete and failed must be explicit booleans')
    time = np.asarray(measurements['time_s'],float)
    if len(time)<1 or not np.allclose(np.diff(time),.01,rtol=0,atol=1e-10):
        raise ValueError('Fold events require consecutive100-Hz observations')
    reversal,turn = fold_geometry(arrays['cable_positions'])
    if len(turn)!=len(time):
        raise ValueError('Geometry and measurements must have matching histories')
    nodes=np.asarray(measurements['node_s_m'],float); fraction=nodes/nodes[-1]
    speed=decomposition['shape_speed_mps'][:,-1]
    direction=np.asarray(direction,float);direction=direction/np.linalg.norm(direction)
    forward=decomposition['shape_velocity_world_mps'][:,-1]@direction
    world=measurements['world_forward_mps'][:,-1]
    tracks=track_curvature(measurements); events=[];maximum_progress=0.
    for track in tracks:
        baseline=None;path=[];distal=False;previous_node=None
        for frame,node in zip(track['indices'],track['node_indices']):
            if (previous_node is not None and node<previous_node
                    or turn[frame,node-1]<RULES['minimum_apex_turn_rad']
                    or baseline is not None and time[frame]-time[baseline]>.60+1e-10):
                baseline=None;path=[];distal=False
            previous_node=node
            if baseline is None:
                if fraction[node]>.35 or not reversal[frame,node-1] or turn[frame,node-1]<.10:
                    continue
                baseline=frame;path=[];distal=False
            path.append((frame,node))
            maximum_progress=max(maximum_progress,float((fraction[node]-fraction[path[0][1]])/(fraction[-2]-fraction[path[0][1]])))
            distal |= bool(fraction[node]>=.65 and reversal[frame,node-1])
            if node != len(nodes)-2 or not distal:
                continue
            end=frame+20
            event=dict(id=len(events),segment_id=track['id'],baseline_index=baseline,
                arming_index=frame,evidence_indices=[i for i,j in path],
                evidence_node_indices=[j for i,j in path],requested_end_index=end,
                truncated=end>=len(time),finalized=end<len(time),provisional_eligible=False,
                score=0.,rejection_reason='unfinished_opening_window')
            if event['finalized']:
                begin=max(baseline,frame-10);q=begin+int(np.argmax(speed[begin:end+1]))
                gain=float(speed[q]-speed[baseline])
                tail_start=max(0,node-3)
                before,after=float(turn[frame,tail_start:].sum()),float(turn[end,tail_start:].sum())
                opening=before>0 and after<=.5*before
                eligible=bool(opening and gain>=.25)
                value=float(min(max(0.,forward[q]),max(0.,world[q]),max(0.,gain))) if eligible else 0.
                event.update(end_index=end,q_shape_index=q,q_shape_mps=float(speed[q]),
                    shape_baseline_mps=float(speed[baseline]),shape_gain_mps=gain,
                    shape_forward_mps=float(forward[q]),world_forward_mps=float(world[q]),
                    relative_tip_speed_mps=float(measurements['relative_speed_mps'][q,-1]),
                    fold_tail_turn_at_arrival_rad=before,fold_tail_turn_after_rad=after,
                    fold_opening_fraction=(1-after/before if before>0 else 0.),
                    opening_verified=bool(opening),provisional_eligible=eligible,score=value,
                    rejection_reason=('fold_did_not_open' if not opening else 'insufficient_shape_speed_gain'
                                      if gain<.25 else 'nonpositive_forward_release' if value<=0 else None))
            events.append(event)
            # One frozen event per outward section. A later unrelated wave cannot
            # revise its baseline, output interval or opening measurement.
            baseline=None;path=[];distal=False
    events.sort(key=lambda e:(e['requested_end_index'],e['baseline_index'],e['segment_id']))
    best=0.;selected=None;increments=np.zeros(len(time))
    for i,e in enumerate(events):
        e['id']=i
        if e['score']>best:
            increments[e['end_index']]+=e['score']-best;best=e['score'];selected=i
    return dict(schema=RULES['version'],rules=dict(RULES),events=events,segments=tracks,
                best_event_score=best,selected_event=selected,causal_best=increments.cumsum(),
                complete=complete,failed=failed,episode_return=(-10. if failed else best if complete else None),
                maximum_fold_progress=maximum_progress,mechanism_status='unresolved')


class FoldReleaseScorer(ShapeReleaseScorer):
    def evaluate_events(self,arrays,measurements,decomposition,direction,*,complete,failed):
        return score_events(arrays,measurements,decomposition,direction,complete=complete,failed=failed)

    def __call__(self,*args,**kwargs):
        score=super().__call__(*args,**kwargs)
        score.components['fold_release_score']=score.score
        score.components['maximum_fold_progress']=score.score.new_tensor(
            [r['maximum_fold_progress'] for r in self.event_records])
        return score
