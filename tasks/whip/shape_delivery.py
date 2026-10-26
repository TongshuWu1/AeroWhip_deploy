"""Experimental Stage2: same-q target guidance on the frozen shape-release core.

Actual same-release forward crossings are evidence, not an extra reward. The
temporal association is not a claim of force-mediated causality.
"""
import copy
import math
import numpy as np
import torch

from .shape_release import ShapeReleaseScorer
from .stage1 import Stage1Score


def delivery_events(core, time, tip_position, tip_velocity, *, center, width,
                    height, sigma, minimum_speed=3., maximum_angle_deg=30.):
    """Score supplied frozen events on an observed finite prefix (+X wall)."""
    t = np.asarray(time, dtype=float)
    p, v, c = (np.asarray(x, dtype=float) for x in (tip_position, tip_velocity, center))
    if (t.ndim != 1 or not len(t) or p.shape != (len(t), 3) or v.shape != p.shape
            or c.shape != (3,) or not all(np.isfinite(x).all() for x in (t,p,v,c))
            or (np.diff(t) <= 0).any()
            or not all(math.isfinite(x) and x > 0 for x in
                       (width,height,sigma,minimum_speed,maximum_angle_deg))):
        raise ValueError('Delivery requires finite observed histories and positive geometry')
    crossings = []
    for k in range(len(t)-1):
        before, after = p[k,0]-c[0], p[k+1,0]-c[0]
        if not before < 0 <= after:
            continue
        fraction = -before/(after-before)
        point, velocity = p[k]+fraction*(p[k+1]-p[k]), v[k]+fraction*(v[k+1]-v[k])
        if velocity[0] <= 0:
            continue
        when = t[k]+fraction*(t[k+1]-t[k])
        side, vertical = point[1]-c[1], point[2]-c[2]
        angle = math.degrees(math.atan2(np.linalg.norm(velocity[1:]), velocity[0]))
        eps = 16*np.finfo(float).eps
        position_eps = eps*max(1., np.abs(point).max(), np.abs(c).max())
        geometric = bool(abs(side) <= width/2+position_eps and abs(vertical) <= height/2+position_eps)
        crossings.append(dict(interval_index=k, fraction=float(fraction), time_s=float(when),
            lateral_error_m=float(side), vertical_error_m=float(vertical),
            forward_mps=float(velocity[0]), angle_deg=angle, geometric_hit=geometric,
            strict_hit=bool(geometric and velocity[0] >= minimum_speed-eps*max(1,minimum_speed)
                            and angle <= maximum_angle_deg+math.degrees(eps))))
    result = copy.deepcopy(core)
    result.update(schema='directed_shape_delivery_v1', shape_core_selected_event=core['selected_event'],
                  shape_core_best_event_score=core['best_event_score'], global_forward_crossings=crossings)
    best, selected = 0., None
    causal = np.zeros(len(t))
    for index, e in enumerate(result['events']):
        e.update(delivery_utility=0., target_guidance=0., target_distance_m=None,
                 associated_crossings=[])
        if not e['finalized'] or not e['provisional_eligible'] or e['score'] <= 0:
            continue
        arm, end, q = e['arming_index'], e['end_index'], e['q_shape_index']
        if not 0 <= arm <= q <= end < len(t):
            raise ValueError('Finalized event exceeds its observed prefix')
        delta = p[q]-c
        distance = float(np.linalg.norm([delta[0], max(abs(delta[1])-width/2,0),
                                         max(abs(delta[2])-height/2,0)]))
        guidance = math.exp(-.5*(distance/sigma)**2)
        utility = e['score']*(1+guidance)
        time_eps = 16*np.finfo(float).eps*max(1.,abs(t[arm]),abs(t[end]))
        associated = [x for x in crossings if t[arm]-time_eps <= x['time_s'] <= t[end]+time_eps]
        e.update(delivery_utility=utility, target_guidance=guidance,
                 target_distance_m=distance, associated_crossings=associated)
        # Maximum available utility at each FINALIZATION time, not at peak q.
        causal[end:] = np.maximum(causal[end:], utility)
        if utility > best:
            best, selected = utility, index
    result.update(best_event_score=best, selected_event=selected, causal_best=causal,
                  selected_shape_score=(result['events'][selected]['score'] if selected is not None else 0.),
                  episode_return=(-10. if core['failed'] else best if core['complete'] else None),
                  delivery_rules=dict(version='directed_shape_delivery_v1', sigma_m=sigma,
                      guidance='S_e*(1+exp(-0.5*(distance_rectangle_at_same_q/sigma)^2))',
                      association='actual forward crossing in same finalized positive-S [arm,end]'))
    return result


class ShapeDeliveryScorer(ShapeReleaseScorer):
    @torch.no_grad()
    def __call__(self, positions, velocities, state_valid, *, failed):
        base = super().__call__(positions, velocities, state_valid, failed=failed)
        if not torch.allclose(self.direction, self.direction.new_tensor([1.,0.,0.]), rtol=0, atol=1e-14):
            raise ValueError('Shape delivery v1 implements the fixed +X target only')
        t, center = self.time.cpu().numpy(), self.center.cpu().numpy()
        p, v = positions.detach().cpu().numpy(), velocities.detach().cpu().numpy()
        batch, samples = positions.shape[:2]
        scores, causal, indices = np.zeros(batch), np.zeros((batch,samples)), np.full((batch,3),-1,dtype=np.int64)
        fields = ('selected_shape_release_score','target_distance_m','target_guidance',
                  'selected_event_geometric_hit_success','selected_event_strict_hit_success',
                  'eligible_event_geometric_hit_success','eligible_event_strict_hit_success')
        extra = {k:np.zeros(batch) for k in fields}
        extra['target_distance_m'][:] = np.nan
        records = []
        components = dict(base.components)
        detail = ('q_shape_mps','shape_baseline_mps','shape_gain_mps','shape_forward_mps',
                  'world_forward_mps','relative_tip_speed_mps')
        selected_detail = {k:np.full(batch,np.nan) for k in detail}
        for lane, core in enumerate(self.event_records):
            count = core['observed_samples']
            r = delivery_events(core,t[:count],p[lane,:count,-1],v[lane,:count,-1],
                center=center,width=self.surface.width_m,height=self.surface.height_m,
                sigma=self.configuration.matching.delivery_sigma_m, minimum_speed=float(self.speed),
                maximum_angle_deg=self.surface.maximum_entry_angle_deg)
            scores[lane] = r['best_event_score']
            causal[lane,:count] = r.pop('causal_best')
            causal[lane,count:] = scores[lane]
            extra['selected_shape_release_score'][lane] = r['selected_shape_score']
            for label, key in (('geometric','geometric_hit'),('strict','strict_hit')):
                extra[f'eligible_event_{label}_hit_success'][lane] = any(
                    x[key] for e in r['events'] for x in e['associated_crossings'])
            if r['selected_event'] is not None:
                e = r['events'][r['selected_event']]
                indices[lane] = [e['baseline_index'],e['arming_index'],e['q_shape_index']]
                for key in detail: selected_detail[key][lane] = e[key]
                for key in ('target_distance_m','target_guidance'): extra[key][lane] = e[key]
                for label, key in (('geometric','geometric_hit'),('strict','strict_hit')):
                    extra[f'selected_event_{label}_hit_success'][lane] = any(x[key] for x in e['associated_crossings'])
            records.append(r)
        self.event_records = records
        tensor = lambda x: torch.as_tensor(x,device=positions.device,dtype=positions.dtype)
        result = tensor(scores).masked_fill(~base.valid,float('nan'))
        components.update({k:tensor(x).masked_fill(~base.valid,float('nan')) for k,x in extra.items()})
        components.update({k:tensor(x) for k,x in selected_detail.items()})
        components['shape_delivery_score'] = result
        # shape_release_score remains the untouched core maximum; selected S is separate.
        return Stage1Score(result,base.valid,base.valid & (result > 0),
            torch.as_tensor(indices,device=positions.device),components,tensor(causal))
