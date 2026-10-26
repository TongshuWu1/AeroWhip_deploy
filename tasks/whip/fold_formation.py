"""Approved Stage1 warm-up: fold shape is a guide, never a release certificate."""
import numpy as np

from .shape_release import ShapeReleaseScorer
from .fold_release import score_events as release_events


FORMATION_RULES=dict(version='directed_fold_formation_v1',root_fraction=.35,
                     angle_unit_rad=1.,formed_angle_rad=float(np.pi/2),
                     minimum_consecutive_observations=2,failure_return=-10.)


def reversal_angles(positions):
    p=np.asarray(positions,float)
    if p.ndim!=3 or p.shape[-1]!=3 or p.shape[1]<5 or not np.isfinite(p).all():
        raise ValueError('Finite time/node/xyz cable positions required')
    edge=np.diff(p,axis=1);length=np.linalg.norm(edge,axis=-1)
    if np.any(length<=0):raise ValueError('Positive measured cable edges required')
    tangent=edge/length[...,None];angles=[]
    for node in range(1,p.shape[1]-1):
        left,right=max(0,node-2),min(tangent.shape[1]-1,node+1)
        a,b=tangent[:,left],tangent[:,right]
        angles.append(np.arctan2(np.linalg.norm(np.cross(a,b),axis=-1),np.einsum('ti,ti->t',a,b)))
    return np.stack(angles,axis=1)


def aligned_release_events(arrays,measurements,decomposition,direction,*,complete,failed):
    """Same-event fold output, with smooth3D alignment; no lateral output bonus."""
    r=release_events(arrays,measurements,decomposition,direction,complete=complete,failed=failed)
    d=np.asarray(direction,float);d=d/np.linalg.norm(d)
    best=0.;selected=None;increments=np.zeros(len(measurements['time_s']))
    for i,e in enumerate(r['events']):
        if not e['finalized']:continue
        q=e['q_shape_index'];u=decomposition['shape_velocity_world_mps'][q,-1]
        v=np.asarray(arrays['cable_velocities'])[q,-1]
        cosine=lambda x:max(0.,float(x@d))/float(np.linalg.norm(x)) if np.linalg.norm(x)>0 else 0.
        factor=float(min(cosine(u),cosine(v))**2)
        e.update(unaligned_fold_release_score=e['score'],direction_alignment=factor,
                 score=e['score']*factor)
        if e['score']>best:
            increments[e['end_index']]+=e['score']-best;best=e['score'];selected=i
    r.update(schema='directed_fold_release_aligned_v1',
             rules=dict(r['rules'],version='directed_fold_release_aligned_v1',
                        alignment='min(positive cosine(shape velocity,d),positive cosine(world velocity,d)) squared at SAME q'),
             best_event_score=best,selected_event=selected,causal_best=increments.cumsum(),
             episode_return=(-10. if failed else best if complete else None))
    return r


class FoldFormationScorer(ShapeReleaseScorer):
    def evaluate_events(self,arrays,measurements,decomposition,direction,*,complete,failed):
        angles=reversal_angles(arrays['cable_positions'])
        nodes=np.asarray(measurements['node_s_m']);root=np.flatnonzero(nodes[1:-1]/nodes[-1]<=.35)
        if not len(root):raise ValueError('Mesh requires an interior sample in attachment-side35%')
        per_frame=angles[:,root].max(1);causal=np.maximum.accumulate(per_frame)
        q=int(np.argmax(per_frame));node=int(root[np.argmax(angles[q,root])]+1)
        present=angles[:,root]>np.pi/2
        formed=bool((present[1:] & present[:-1]).any())
        value=float(causal[-1]);speed=decomposition['shape_speed_mps'][:,-1]
        # The environment differences a potential starting at zero. Any
        # initially present guide angle is paid on the first transition;
        # otherwise a curved initial cable silently loses part of its return.
        causal[0]=0.
        d=np.asarray(direction,float);d=d/np.linalg.norm(d)
        event=dict(id=0,baseline_index=0,arming_index=q,q_shape_index=q,
            requested_end_index=q,end_index=q,truncated=False,finalized=True,
            provisional_eligible=formed,score=value,
            q_shape_mps=float(speed[q]),shape_baseline_mps=float(speed[0]),
            shape_gain_mps=float(speed[q]-speed[0]),
            shape_forward_mps=float(decomposition['shape_velocity_world_mps'][q,-1]@d),
            world_forward_mps=float(measurements['world_forward_mps'][q,-1]),
            relative_tip_speed_mps=float(measurements['relative_speed_mps'][q,-1]),
            root_fold_peak_node=node,root_fold_peak_angle_rad=value,
            rejection_reason=None if formed else 'formation_guide_only_no_persistent_fold')
        release=aligned_release_events(arrays,measurements,decomposition,direction,complete=complete,failed=failed)
        return dict(schema=FORMATION_RULES['version'],rules=dict(FORMATION_RULES),
            events=[event] if value>0 else [],selected_event=0 if value>0 else None,
            segments=[],best_event_score=value,causal_best=causal,root_fold_formed=formed,
            complete=complete,failed=failed,episode_return=-10. if failed else value if complete else None,
            release_diagnostic={k:release[k] for k in ('schema','events','best_event_score','selected_event','maximum_fold_progress')},
            mechanism_status='unresolved',full_stage1_success=False)

    def __call__(self,*args,**kwargs):
        score=super().__call__(*args,**kwargs)
        score.components.pop('shape_release_score')
        score.components['fold_formation_score']=score.score
        score.components['root_fold_formed']=score.score.new_tensor([r['root_fold_formed'] for r in self.event_records]).masked_fill(~score.valid,float('nan'))
        score.components['fold_release_score']=score.score.new_tensor([r['release_diagnostic']['best_event_score'] for r in self.event_records]).masked_fill(~score.valid,float('nan'))
        return score


class AlignedFoldReleaseScorer(ShapeReleaseScorer):
    def evaluate_events(self,arrays,measurements,decomposition,direction,*,complete,failed):
        return aligned_release_events(arrays,measurements,decomposition,direction,complete=complete,failed=failed)

    def __call__(self,*args,**kwargs):
        score=super().__call__(*args,**kwargs);score.components.pop('shape_release_score')
        score.components['fold_release_score']=score.score
        score.components['release_direction_alignment']=score.score.new_tensor([
            r['events'][r['selected_event']]['direction_alignment'] if r['selected_event'] is not None else 0.
            for r in self.event_records]).masked_fill(~score.valid,float('nan'))
        return score
