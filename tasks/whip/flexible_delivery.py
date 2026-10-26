"""Fixed-direction broad delivery; geometric quality is not a whip classifier.

Both speed terms belong to the same state. Crossing credit is causal only after
strict positive-side confirmation; exact-zero plateaus use their last departure.
"""
import numpy as np
import torch

from evaluation.whip_output_decomposition import decompose
from .stage1 import Stage1Scorer, Stage1Score

MODE = 'flexible_delivery_v1'


def confirmed_passages(offset):
    """Yield (left, right, fraction, confirmation) for forward passages.

    A zero plateau does not earn credit on arrival. Retreat cancels it. Starting
    on the plane or ending there never substitutes for a complete passage.
    """
    negative = None
    for i, x in enumerate(offset):
        if x < 0:
            negative = i
        elif x > 0:
            if negative is not None:
                if i == negative+1:
                    yield negative, i, float(-offset[negative]/(x-offset[negative])), i
                else:
                    yield i-1, i-1, 0., i
            negative = None


def score_trace(positions, velocities, times, rest, center, width, height, *, first_hit=False):
    """Float64 finite-prefix scoring for the explicitly fixed +X task."""
    p, v, t, rest, center = (np.asarray(x, dtype=np.float64)
                             for x in (positions, velocities, times, rest, center))
    if (p.ndim != 3 or p.shape != v.shape or p.shape[-1] != 3
            or t.shape != (len(p),) or not len(t) or np.any(np.diff(t) <= 0)
            or center.shape != (3,) or width <= 0 or height <= 0
            or not all(np.isfinite(x).all() for x in (p,v,t,rest,center))):
        raise ValueError('Delivery requires finite matching states and increasing times')
    shape = decompose(p,v,rest)['shape_speed_mps'][:, -1]
    q = np.minimum(np.maximum(v[:, -1, 0],0),shape)
    delta = p[:, -1]-center
    distance = np.linalg.norm(np.column_stack((delta[:,0],
        np.maximum(np.abs(delta[:,1])-width/2,0),
        np.maximum(np.abs(delta[:,2])-height/2,0))),axis=1)
    guide = q/(1+q)*np.exp(-.5*(distance/.7)**2)
    guide[0] = 0.
    crossings = []
    by_confirmation = {}
    for left,right,fraction,confirmation in confirmed_passages(delta[:,0]):
        cp = p[left]+fraction*(p[right]-p[left])
        cv = v[left]+fraction*(v[right]-v[left])
        cs = float(decompose(cp[None],cv[None],rest)['shape_speed_mps'][0,-1])
        forward = float(cv[-1,0])
        quality = min(max(forward,0.),cs)
        error = cp[-1]-center
        hit = bool(abs(error[1]) <= width/2 and abs(error[2]) <= height/2 and forward > 0)
        speed = float(np.linalg.norm(cv[-1]))
        angle = float(np.degrees(np.arccos(np.clip(forward/speed,-1,1)))) if speed else 180.
        crossing = dict(left_index=left,right_index=right,fraction=fraction,
            confirmation_index=confirmation,time_s=float(t[left]+fraction*(t[right]-t[left])),
            confirmation_time_s=float(t[confirmation]),point_world_m=cp[-1].tolist(),
            velocity_world_mps=cv[-1].tolist(),shape_speed_mps=cs,forward_speed_mps=forward,
            quality=quality,geometric_hit=hit,quality_hit=bool(hit and quality >= 1.),
            strict_hit=bool(hit and speed >= 3. and angle <= 30.),
            center_error_m=float(np.linalg.norm(error)),entry_angle_deg=angle,
            utility=5*quality/(1+quality) if hit else 0.)
        crossings.append(crossing)
        by_confirmation[confirmation] = len(crossings)-1
    causal = np.zeros(len(t)); events=[]; best=0.; selected=None
    for i in range(1,len(t)):
        candidates=[('approach',float(guide[i]),None)]
        if i in by_confirmation:
            j=by_confirmation[i]
            if not first_hit or crossings[j]['quality_hit']:
                candidates.append(('crossing',crossings[j]['utility'],j))
        for kind,value,j in candidates:
            if value > best:
                best=value; selected=len(events)
                events.append(dict(id=selected,kind=kind,score=value,
                    baseline_index=0,arming_index=i,q_shape_index=i,end_index=i,
                    finalized=True,truncated=False,provisional_eligible=True,
                    crossing_index=j,time_s=float(t[i])))
        causal[i]=best
    eligible=[i for i,c in enumerate(crossings) if c['geometric_hit']]
    best_crossing=max(eligible,key=lambda i:crossings[i]['utility']) if eligible else None
    return dict(schema=MODE,events=events,crossings=crossings,selected_event=selected,
        best_crossing=best_crossing,best_event_score=best,causal_best=causal,
        approach_max=float(guide.max()),crossing_max=(crossings[best_crossing]['utility']
            if best_crossing is not None else 0.),mechanism_status='unresolved')


class FlexibleDeliveryScorer(Stage1Scorer):
    first_hit = False
    @torch.no_grad()
    def __call__(self,positions,velocities,state_valid,*,failed):
        b,s=positions.shape[:2]
        if (positions.shape != (b,len(self.time),len(self.rest)+1,3)
                or velocities.shape != positions.shape or state_valid.shape != (b,s)
                or failed.shape != (b,) or state_valid.dtype != torch.bool or failed.dtype != torch.bool):
            raise ValueError('Delivery needs batched histories and explicit failure flags')
        finite=state_valid & torch.isfinite(positions).flatten(2).all(-1) & torch.isfinite(velocities).flatten(2).all(-1)
        prefix=finite.long().cumprod(1).bool(); valid=finite.all(1) & ~failed
        p,v,t,rest,center=(x.detach().cpu().numpy() for x in
            (positions,velocities,self.time,self.rest,self.center))
        causal=np.zeros((b,s)); scores=np.zeros(b); indices=np.full((b,3),-1,dtype=np.int64)
        fields=('delivery_utility','approach_utility','crossing_utility','geometric_hit_success',
                'quality_hit_success','strike_success','strict_strike_success','best_crossing_quality',
                'best_crossing_forward_mps','best_crossing_shape_mps','best_crossing_center_error_m')
        components={k:np.zeros(b) for k in fields}; self.event_records=[]
        for lane,count in enumerate(prefix.sum(1).tolist()):
            if count == 0: raise ValueError('Delivery requires finite initial state')
            r=score_trace(p[lane,:count],v[lane,:count],t[:count],rest,center,
                          self.surface.width_m,self.surface.height_m,first_hit=self.first_hit)
            if self.first_hit:
                qualified=[c for c in r['crossings'] if c['quality_hit']]
                if qualified and qualified[0]['confirmation_index'] != count-1:
                    raise ValueError('First-hit trace continued beyond its first qualifying confirmation')
                valid[lane] = not bool(failed[lane]) and (bool(qualified) or count == s)
                r['schema']='first_hit_flexible_delivery_v2'
            scores[lane]=r['best_event_score']; causal[lane,:count]=r.pop('causal_best')
            causal[lane,count:]=scores[lane]
            if r['selected_event'] is not None:
                e=r['events'][r['selected_event']];indices[lane]=[0,e['arming_index'],e['q_shape_index']]
            c=r['crossings']; j=r['best_crossing']
            components['delivery_utility'][lane]=scores[lane]
            components['approach_utility'][lane]=r['approach_max']
            components['crossing_utility'][lane]=r['crossing_max']
            for key,flag in (('geometric_hit_success','geometric_hit'),('quality_hit_success','quality_hit'),
                             ('strike_success','strict_hit'),('strict_strike_success','strict_hit')):
                components[key][lane]=any(x[flag] for x in c)
            for key,field in (('best_crossing_quality','quality'),('best_crossing_forward_mps','forward_speed_mps'),
                              ('best_crossing_shape_mps','shape_speed_mps'),('best_crossing_center_error_m','center_error_m')):
                components[key][lane]=c[j][field] if j is not None else np.nan
            self.event_records.append(dict(lane=lane,observed_samples=count,complete=bool(valid[lane]),
                failed=bool(failed[lane]),episode_return=(-10. if not valid[lane] else scores[lane]),**r))
        tensor=lambda x:torch.as_tensor(x,device=positions.device,dtype=positions.dtype)
        # Raw events survive failures; success metrics count only full-horizon valid attempts.
        comp={k:tensor(x).masked_fill(~valid,float('nan')) for k,x in components.items()}
        result=tensor(scores).masked_fill(~valid,float('nan'))
        return Stage1Score(result,valid,valid & (comp['quality_hit_success']==1),
            torch.as_tensor(indices,device=positions.device),comp,tensor(causal))
