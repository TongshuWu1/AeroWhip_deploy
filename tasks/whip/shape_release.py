"""Approved experimental Stage1 kinematic utility, versioned independently.

The frozen event detector is shared with the retained offline diagnostic. Its
old relative-speed utility is discarded; only its mechanism evidence and release
bounds are reused. No force-mediated causation or physical-whip label is implied.
"""
import numpy as np
import torch

from evaluation.stage1_candidate import detect
from evaluation.whip_measurements import extract_measurements
from evaluation.whip_output_decomposition import decompose
from .stage1 import Stage1Scorer, Stage1Score
from .strike import surface_events


RULES = dict(version='directed_shape_release_v1', minimum_shape_gain_mps=.25,
             speed_unit_mps=1., failure_return=-10.)


def score_events(measurements, decomposition, direction, *, complete, failed):
    """Finalize only frozen windows; select earliest maximum SHAPE speed.

    The free-translation rigid residual in measurements gates mechanism evidence.
    The independently attachment-anchored decomposition determines release output.
    """
    detected = detect(measurements, complete=complete, failed=failed)
    speed = decomposition['shape_speed_mps'][:, -1]
    forward = decomposition['shape_velocity_world_mps'][:, -1] @ np.asarray(direction)
    world = measurements['world_forward_mps'][:, -1]
    increments = np.zeros(len(speed))
    best, selected, events = 0., None, []
    for old in detected['events']:
        # Do not leak the superseded relative-speed eligibility/score into records.
        e = {k:old[k] for k in ('id', 'segment_id', 'baseline_index', 'arming_index',
            'evidence_indices', 'evidence_node_indices', 'evidence_nonrigid_rms_mps',
            'requested_end_index', 'truncated', 'finalized')}
        e.update(provisional_eligible=False, score=0., rejection_reason='unfinished_release')
        if e['finalized']:
            start, stop, baseline = e['arming_index'], old['end_index'], e['baseline_index']
            q = start+int(np.argmax(speed[start:stop+1]))
            gain = float(speed[q]-speed[baseline])
            eligible = gain >= RULES['minimum_shape_gain_mps']
            value = float(min(max(0., forward[q]), max(0., world[q]))/RULES['speed_unit_mps']) if eligible else 0.
            e.update(end_index=stop, q_shape_index=q, q_shape_mps=float(speed[q]),
                shape_baseline_mps=float(speed[baseline]), shape_gain_mps=gain,
                shape_forward_mps=float(forward[q]), world_forward_mps=float(world[q]),
                relative_tip_speed_mps=float(measurements['relative_speed_mps'][q, -1]),
                provisional_eligible=bool(eligible), score=value,
                rejection_reason=('insufficient_shape_speed_gain' if not eligible else
                                  'nonpositive_delivery' if value == 0 else None))
            if value > best:
                increments[stop] += value-best
                best, selected = value, len(events)
        events.append(e)
    # causal_best is pre-failure credit; the environment supplies terminal clawback.
    return dict(schema=RULES['version'], rules=dict(RULES), events=events,
                segments=detected['segments'], best_event_score=best, selected_event=selected,
                causal_best=increments.cumsum(), complete=complete, failed=failed,
                episode_return=(-10. if failed else best if complete else None),
                mechanism_status='unresolved')


class ShapeReleaseScorer(Stage1Scorer):
    """Float64 CPU reference scorer; invoked once per completed cohort.

    Only finite executed prefixes enter either geometric fit. Full-horizon
    failure must be supplied explicitly: an all-finite trace may still violate
    a physical limit at the last observation. Events remain diagnostic on failure.
    """
    def evaluate_events(self,arrays,measurements,decomposition,direction,*,complete,failed):
        return score_events(measurements,decomposition,direction,complete=complete,failed=failed)

    @torch.no_grad()
    def __call__(self, positions, velocities, state_valid, *, failed):
        if (positions.ndim != 4 or positions.shape[1:] != (len(self.time),len(self.rest)+1,3)
                or velocities.shape != positions.shape or state_valid.shape != positions.shape[:2]
                or state_valid.dtype != torch.bool or failed.shape != positions.shape[:1]
                or failed.dtype != torch.bool):
            raise ValueError('Shape release needs complete batched histories and explicit failure flags')
        finite = (state_valid & torch.isfinite(positions).flatten(2).all(-1)
                  & torch.isfinite(velocities).flatten(2).all(-1))
        prefix = finite.long().cumprod(1).bool()
        valid = finite.all(1) & ~failed
        p, v = positions.detach().cpu().numpy(), velocities.detach().cpu().numpy()
        rest, direction, times = (x.cpu().numpy() for x in (self.rest,self.direction,self.time))
        counts, failures = prefix.sum(1).tolist(), failed.tolist()
        batch, samples = positions.shape[:2]
        scores = np.zeros(batch)
        causal = np.zeros((batch,samples))
        indices = np.full((batch,3), -1, dtype=np.int64)
        names = ('q_shape_mps','shape_baseline_mps','shape_gain_mps','shape_forward_mps',
                 'world_forward_mps','relative_tip_speed_mps')
        components = {k:np.full(batch,np.nan) for k in names}
        components.update({k:np.zeros(batch) for k in
            ('armed_event_count','finalized_event_count','eligible_event_count','truncated_event_count')})
        self.event_records = []
        for lane, count in enumerate(counts):
            if count == 0:
                raise ValueError('Shape release requires a finite initial cable state')
            arrays = dict(time_s=times[:count], cable_positions=p[lane,:count], cable_velocities=v[lane,:count])
            m = extract_measurements(arrays,rest,direction)
            d = decompose(arrays['cable_positions'],arrays['cable_velocities'],rest)
            r = self.evaluate_events(arrays,m,d,direction,
                complete=(count == samples and not failures[lane]),failed=failures[lane])
            scores[lane] = r['best_event_score']
            causal[lane,:count] = r.pop('causal_best')
            causal[lane,count:] = scores[lane]
            events = r['events']
            for key, value in zip(('armed_event_count','finalized_event_count','eligible_event_count','truncated_event_count'),
                    (len(events),sum(e['finalized'] for e in events),sum(e['provisional_eligible'] for e in events),sum(e['truncated'] for e in events))):
                components[key][lane] = value
            if r['selected_event'] is not None:
                e = events[r['selected_event']]
                indices[lane] = [e['baseline_index'],e['arming_index'],e['q_shape_index']]
                for k in names: components[k][lane] = e[k]
            self.event_records.append(dict(lane=lane,observed_samples=count,**r))
        tensor = lambda x: torch.as_tensor(x,device=positions.device,dtype=positions.dtype)
        result = tensor(scores).masked_fill(~valid,float('nan'))
        components = {k:tensor(x) for k,x in components.items()}
        # Existing interpolated crossing measurements remain diagnostic only.
        safe_p = torch.where(prefix[:,:,None,None],positions,torch.zeros_like(positions))
        safe_v = torch.where(prefix[:,:,None,None],velocities,torch.zeros_like(velocities))
        surface = surface_events(safe_p,safe_v,self.time,self.center,self.direction,self.surface,self.speed)
        ready = (torch.arange(samples,device=positions.device)+1).clamp_max(samples-1)
        crossing = surface['crossing'] & prefix & prefix[:,ready]
        for name, mask in (('strike_success',surface['qualified']),
                           ('geometric_hit_success',surface['geometric_hit']),
                           ('strict_strike_success',surface['qualified'])):
            components[name] = (crossing & mask).any(1).to(positions.dtype).masked_fill(~valid,float('nan'))
        components['shape_release_score'] = result
        return Stage1Score(result,valid,valid & (result > 0),
                           torch.as_tensor(indices,device=positions.device),components,tensor(causal))
