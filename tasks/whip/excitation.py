"""Experimental direction-free flexible-excitation guide for Stage 1.

Credit is the best same-sample minimum of world tip speed and the existing
attachment-anchored nonrigid tip speed, in units of 1 m/s. The initial state is
an unpaid baseline. Improvement records are not detected physical releases,
whip classifications, energy measurements or target-success certificates.
"""
import numpy as np
import torch

from .shape_release import ShapeReleaseScorer


MODE = 'flexible_excitation_v1'


def negative_net_rational(value):
    """Approved v3 return map: identity above zero, negative limit -5.

    Transform the net score, never individual workspace charges. Clamping the
    unused branch's denominator avoids a singularity at positive value=5.
    """
    negative = value.clamp_max(0)
    return torch.where(value >= 0, value, 5*negative/(5-negative))


RULES = dict(version=MODE, speed_unit_mps=1., failure_return=-10.,
             utility='max_post_initial_sample(min(world_tip_speed, anchored_shape_tip_speed)/(1 m/s))',
             initial_sample='unpaid_baseline', direction_in_reward=False,
             record_kind='excitation_record_not_physical_event')


def _finite(value, name):
    raw = np.asarray(value)
    if raw.dtype.kind not in 'fiu':
        raise ValueError(f'{name} must be real numeric values')
    result = np.asarray(raw, dtype=np.float64)
    if not np.isfinite(result).all():
        raise ValueError(f'{name} must be finite')
    return result


def score_events(measurements, decomposition, direction, *, complete, failed):
    """Return running-record credit compatible with the existing scorer adapter.

    Every record uses one observed sample for both speed magnitudes. Direction,
    baseline/gain and relative-speed fields are diagnostics only. Strict record
    improvements preserve earliest ties. A one-sample prefix has no executed
    transition and receives zero credit. Failed prefixes retain their raw
    causal_best; the environment separately claws it back and applies costs.
    """
    if not isinstance(complete, (bool, np.bool_)) or not isinstance(failed, (bool, np.bool_)):
        raise ValueError('complete and failed must be explicit boolean flags')
    try:
        time = _finite(measurements['time_s'], 'time_s')
        world = _finite(measurements['world_speed_mps'], 'world_speed_mps')
        relative = _finite(measurements['relative_speed_mps'], 'relative_speed_mps')
        world_forward = _finite(measurements['world_forward_mps'], 'world_forward_mps')
        shape = _finite(decomposition['shape_speed_mps'], 'shape_speed_mps')
        shape_velocity = _finite(decomposition['shape_velocity_world_mps'], 'shape_velocity_world_mps')
    except KeyError as exc:
        raise ValueError(f'Missing excitation field: {exc.args[0]}') from exc
    if (time.ndim != 1 or not len(time) or not np.all(np.diff(time) > 0)
            or world.ndim != 2 or world.shape[0] != len(time) or world.shape[1] < 3
            or any(a.shape != world.shape for a in (relative, world_forward, shape))
            or shape_velocity.shape != (*world.shape, 3)
            or any(np.any(a < 0) for a in (world, relative, shape))):
        raise ValueError('Need increasing times and matching nonnegative (time,nodes>=3) speed arrays')
    d = _finite(direction, 'direction')
    if d.shape != (3,) or not np.any(d):
        raise ValueError('direction must be a nonzero three-vector')
    d = d/np.max(np.abs(d)); d = d/np.linalg.norm(d)
    shape_forward = shape_velocity[:, -1] @ d
    observed = np.minimum(world[:, -1], shape[:, -1])/RULES['speed_unit_mps']
    observed[0] = 0.
    causal = np.maximum.accumulate(observed)
    events = []
    previous = 0.
    for q in range(1, len(time)):
        value = float(causal[q])
        if value <= previous:
            continue
        previous = value
        events.append(dict(id=len(events), segment_id=-1,
            kind='excitation_record_not_physical_event', baseline_index=0,
            arming_index=q, requested_end_index=q, end_index=q, q_shape_index=q,
            evidence_indices=[q], evidence_node_indices=[world.shape[1]-1],
            evidence_nonrigid_rms_mps=float(shape[q, -1]),
            truncated=False, finalized=True, provisional_eligible=True,
            eligibility_scope='positive_excitation_guide_only', score=value,
            q_shape_mps=float(shape[q, -1]), shape_baseline_mps=float(shape[0, -1]),
            shape_gain_mps=float(shape[q, -1]-shape[0, -1]),
            shape_forward_mps=float(shape_forward[q]),
            world_forward_mps=float(world_forward[q, -1]),
            relative_tip_speed_mps=float(relative[q, -1]),
            world_speed_at_q=float(world[q, -1]), shape_speed_at_q=float(shape[q, -1]),
            rejection_reason=None))
    best = float(causal[-1])
    return dict(schema=MODE, rules=dict(RULES), events=events, segments=[],
                best_event_score=best, selected_event=len(events)-1 if events else None,
                causal_best=causal, complete=bool(complete), failed=bool(failed),
                episode_return=(-10. if failed else best if complete else None),
                mechanism_status='unresolved',
                event_semantics='causal_sample_improvements_not_detected_whip_events')


class ExcitationScorer(ShapeReleaseScorer):
    """Reuse finite-prefix geometry and crossing diagnostics, without a detector."""
    def evaluate_events(self, arrays, measurements, decomposition, direction, *, complete, failed):
        return score_events(measurements, decomposition, direction, complete=complete, failed=failed)

    @torch.no_grad()
    def __call__(self, *args, **kwargs):
        result = super().__call__(*args, **kwargs)
        result.components.pop('shape_release_score')
        result.components['excitation_score'] = result.score
        for key in ('world_speed_at_q', 'shape_speed_at_q'):
            values = [r['events'][r['selected_event']][key] if r['selected_event'] is not None else np.nan
                      for r in self.event_records]
            result.components[key] = result.score.new_tensor(values).masked_fill(~result.valid, float('nan'))
        result.components['excitation_positive_guide'] = (result.valid & (result.score > 0)).to(result.score.dtype)
        return result
