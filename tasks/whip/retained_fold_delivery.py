"""Approved experimental retained-progress fold delivery, with unchanged physics.

Progress is an episode maximum of one coherent candidate's sequential score.
It is intentionally retained at terminal states, not potential shaping.
"""
from copy import deepcopy
import math
import numpy as np
from .horizontal_fold import FoldGeometry, output_sample, CONTRACT as FOLD_CONTRACT

MODE = 'retained_fold_delivery_v1'
FEATURE_SIZE = 24
CONTRACT = dict(FOLD_CONTRACT, version=MODE, time_cost_per_s=0.,
    progress_weight=5., hit_bonus=10., miss_penalty=2.,
    target_center_m=[1.1, 0., 2.], target_width_m=1., target_height_m=.6,
    delivery_sigma_m=.7, useful_hit_forward_mps=1.)


def _clip(value):
    return float(np.clip(value, 0., 1.))


def _persist(values):
    return len(values) / 3. * min(values) if values else 0.


def _turns(positions):
    edge = np.diff(positions, axis=0)
    norm = np.linalg.norm(edge, axis=1)
    if np.any(norm <= 0):
        raise ValueError('Collapsed interpolated cable segment')
    tangent = edge / norm[:, None]
    return np.arctan2(np.linalg.norm(np.cross(tangent[:-1], tangent[1:]), axis=1),
                      np.sum(tangent[:-1] * tangent[1:], axis=1))


def delivery_sample(positions, velocities, length):
    """All guide factors use this one sample in world coordinates."""
    sample = output_sample(positions, velocities, length)
    tip = positions[-1]
    excess = np.array([tip[0] - 1.1, max(abs(tip[1]) - .5, 0.),
                       max(abs(tip[2] - 2.) - .3, 0.)])
    distance = float(np.linalg.norm(excess))
    world = velocities[-1]
    relative = world - velocities[0]
    cw = _clip(world[0] / np.linalg.norm(world)) if np.linalg.norm(world) else 0.
    cr = _clip(relative[0] / np.linalg.norm(relative)) if np.linalg.norm(relative) else 0.
    alignment = min(cw, cr)
    proximity = math.exp(-.5 * (distance / .7) ** 2)
    sample.update(rectangle_distance_m=distance, target_alignment=alignment,
                  target_world_cosine=cw, target_relative_cosine=cr,
                  proximity=proximity,
                  delivery_guide=sample['H'] ** 2 * proximity * alignment ** 2)
    return sample


class RetainedFoldDeliveryAttempt:
    """One seven-second attempt. Native safety guards call fail before observe."""
    def __init__(self, rest):
        self.geometry = FoldGeometry(rest)
        self.L = self.geometry.L
        self.time = self.total = self.progress = 0.
        self.progress_credit = self.outcome_credit = 0.
        self.progress_source_event_id = self.terminal_event_id = None
        self.progress_source = None
        self.terminal = self.active = self.selected = None
        self.formed = self.propagated = self.opened = False
        self.f = self.p = self.o = self.d = 0.
        self.histories = {}
        self.last_regions = []
        self.log = []
        self.crossings = []
        self._previous = None
        self._plane_touch = None

    @staticmethod
    def _arrays(positions, velocities):
        p = np.asarray(positions, dtype=float)
        v = np.asarray(velocities, dtype=float)
        if p.shape != (11, 3) or v.shape != (11, 3) or not np.isfinite(p).all() or not np.isfinite(v).all():
            raise ValueError('Finite eleven-node positions and velocities required')
        return p, v

    def initialize(self, positions, velocities):
        if self.time != 0. or self._previous is not None or self.terminal is not None:
            raise RuntimeError('Initialize reset geometry exactly once before observations')
        p, v = self._arrays(positions, velocities)
        self._previous = (0., p.copy(), v.copy())

    def _credit(self, score, event_id, time, phase):
        score = _clip(score)
        if score > self.progress:
            increment = 5. * (score - self.progress)
            self.progress = score
            self.total += increment
            self.progress_credit = 5. * score
            self.progress_source_event_id = int(event_id)
            self.progress_source = dict(time_s=time, track_id=int(event_id), phase=phase,
                                        score=score, f=self.f, p=self.p, o=self.o, d=self.d)
            self.log.append(dict(kind='progress', increment=increment, **self.progress_source))

    def _end(self, reason, time):
        self.terminal_event_id = None if self.active is None else int(self.active['id'])
        if reason == 'hit':
            self._credit(1., self.active['id'], time, 'verified_hit')
            self.outcome_credit = 10.
        else:
            self.outcome_credit = -2. if reason in ('event_end', 'timeout') else -10.
        self.total = self.progress_credit + self.outcome_credit
        self.time = float(time)
        self.terminal = reason
        self.log.append(dict(kind=reason, time_s=time, return_value=self.total,
                             progress=self.progress, terminal_event_id=self.terminal_event_id,
                             selected=deepcopy(self.selected)))

    def fail(self, time, reason='physical_failure'):
        if self.terminal is not None:
            raise RuntimeError('Cannot alter a terminated attempt')
        if not self.time <= time <= 7. + 1e-10:
            raise ValueError('Failure must follow valid observed prefix')
        if reason in ('hit', 'event_end', 'timeout'):
            raise ValueError('Failure reason cannot be a successful/safe terminal')
        old = self.total
        self._end(reason, float(time))
        return self.total - old

    def _search(self, rows, time):
        histories = {}
        ready = []
        best = (0., None)
        for b in rows:
            if (b['centre_m'] > .5 * self.L or not b['joint_indices'] or b['unresolved']
                    or set(b['flags']) & {'plateau', 'ambiguous_identity'}):
                continue
            ident = b['track_id']
            raw = _clip(b['theta'] / math.radians(150)) * _clip(b['coherence'] / .8)
            values = (self.histories.get(ident, []) + [raw])[-3:]
            histories[ident] = values
            fraction = _persist(values)
            if fraction > best[0]:
                best = (fraction, ident)
            if len(values) == 3 and min(values) == 1.:
                ready.append(b)
        self.histories = histories
        self.f, self.p, self.o, self.d = best[0], 0., 0., 0.
        if best[1] is not None:
            self._credit(self.f / 4., best[1], time, 'formation')
        if not ready:
            return
        b = sorted(ready, key=lambda b: (-b['theta'] * b['coherence'], b['centre_m'], b['track_id']))[0]
        self.active = dict(id=b['track_id'], s_a=b['centre_m'], max_s=b['centre_m'],
            tail=min(b['joint_indices']) - 1, propagated=False, opened=False,
            low=0, bad=0, identity_lost=False, opening_values=[], buffer=[],
            pending_hits=[], output_start=None, opening_confirmed_s=None,
            tail_turn=0.)
        self.formed = True
        self.f = 1.
        self.selected = None
        self.log.append(dict(kind='armed', time_s=time, track_id=b['track_id'],
                             s_a=b['centre_m'], tail_joint=self.active['tail'] + 1))
        self._credit(.25, b['track_id'], time, 'formed')

    def _forward_crossing(self, time, positions, velocities):
        """Confirm negative-to-positive passage, including an exact plane sample.

        A touch alone is pending; a return to the negative side cancels it.
        Starting on the plane cannot create a negative-side history.
        """
        prev_time, prev_p, prev_v = self._previous
        before, after = prev_p[-1, 0] - 1.1, positions[-1, 0] - 1.1
        crossing = None
        if before < 0. and after > 0.:
            fraction = -before / (after - before)
            crossing = (prev_time + fraction * (time - prev_time),
                        prev_p + fraction * (positions - prev_p),
                        prev_v + fraction * (velocities - prev_v),
                        prev_time, time)
        elif after == 0. and before < 0.:
            # Keep the original arrival, never retime a stationary plane touch
            # into a later fold or replace its velocity with a later peak.
            self._plane_touch = (time, positions.copy(), velocities.copy(), prev_time, time)
        elif before == 0. and after > 0. and self._plane_touch is not None:
            crossing = self._plane_touch
        if after != 0.:
            self._plane_touch = None
        return crossing

    def _crossing_record(self, crossing, time):
        if crossing is None:
            return None
        when, p, v, bracket_start, bracket_end = crossing
        sample = delivery_sample(p, v, self.L)
        a = self.active
        reasons = []
        if a is None or not a['propagated'] or a['output_start'] is None or when < a['output_start'] - 1e-12:
            reasons.append('outside_same_opening')
        tail = None
        if a is not None:
            try:
                tail = float(_turns(p)[a['tail']:].sum())
            except ValueError:
                reasons.append('collapsed_interpolated_geometry')
            if tail is not None and tail > math.pi / 3:
                reasons.append('tail_not_open_at_crossing')
        if abs(p[-1, 1]) > .5 or abs(p[-1, 2] - 2.) > .3:
            reasons.append('outside_rectangle')
        if sample['H'] < .75:
            reasons.append('insufficient_reach')
        if not sample['outward'] or not sample['aligned']:
            reasons.append('output_velocity_ineligible')
        if v[-1, 0] < 1.:
            reasons.append('forward_speed_below_floor')
        record = dict(sample, time_s=when, observed_positive_s=time,
            bracket_start_s=bracket_start, bracket_end_s=bracket_end,
            track_id=None if a is None else int(a['id']),
            tail_turn_rad=tail, positions_m=p.tolist(), velocities_mps=v.tolist(),
            eligible=not reasons, reasons=reasons, opening_confirmed=False)
        self.crossings.append(record)
        return record

    def _accept_outputs(self, samples, time):
        a = self.active
        for sample in samples:
            if sample['tail_turn_rad'] <= math.pi / 3 and sample['outward'] and sample['aligned']:
                if self.selected is None or sample['delivery_guide'] > self.selected['delivery_guide']:
                    self.selected = deepcopy(sample)
                self.d = max(self.d, sample['delivery_guide'])
                self._credit((3. + sample['delivery_guide']) / 4., a['id'], time, 'delivery')

    def observe(self, time, positions, velocities):
        if self.terminal is not None:
            raise RuntimeError('No observations after termination')
        if self._previous is None:
            raise RuntimeError('Call initialize with reset geometry before observing')
        time = float(time)
        if not math.isclose(time - self.time, .01, abs_tol=1e-8) or time > 7. + 1e-9:
            raise ValueError('Observe causal100Hz endpoints through seven seconds')
        p, v = self._arrays(positions, velocities)
        old_total = self.total
        turn, rows = self.geometry.observe(p)
        self.last_regions = rows
        crossing = self._forward_crossing(time, p, v)
        a = self.active
        completed = False
        if a is not None:
            match = next((b for b in rows if b['track_id'] == a['id']), None)
            unique = match is not None and not set(match['flags']) & {'plateau', 'ambiguous_identity'}
            if unique:
                s = match['centre_m']
                a['max_s'] = max(a['max_s'], s)
                self.p = _clip((s - a['s_a']) / max(.2 * self.L, .65 * self.L - a['s_a']))
                if not a['propagated']:
                    self._credit((1. + self.p) / 4., a['id'], time, 'propagation')
                if s - a['s_a'] >= .2 * self.L and s >= .65 * self.L:
                    if not a['propagated']:
                        self.log.append(dict(kind='propagated', time_s=time, track_id=a['id']))
                    a['propagated'] = self.propagated = True
            tail = float(turn[a['tail']:].sum())
            a['tail_turn'] = tail
            low = tail <= math.pi / 3
            if a['propagated']:
                self.p = 1.
                raw = _clip((math.radians(150) - tail) / math.radians(90))
                a['opening_values'] = (a['opening_values'] + [raw])[-3:]
                self.o = _persist(a['opening_values'])
                if unique and not a['identity_lost'] and not a['opened']:
                    self._credit((2. + self.o) / 4., a['id'], time, 'opening')
            if a['opened'] or (a['propagated'] and low):
                sample = dict(delivery_sample(p, v, self.L), time_s=time,
                    tail_turn_rad=tail, track_id=int(a['id']), positions_m=p.tolist())
                if not a['opened']:
                    if a['low'] == 0:
                        a['output_start'] = time
                    a['low'] += 1
                    a['identity_lost'] |= not unique
                    a['buffer'].append(sample)
                    if a['low'] >= 3:
                        a['opened'] = self.opened = True
                        a['opening_confirmed_s'] = time
                        self.o = 1.
                        self.log.append(dict(kind='opening_confirmed', time_s=time, track_id=a['id']))
                        self._credit(.75, a['id'], time, 'opened')
                        self._accept_outputs(a['buffer'], time)
                        a['buffer'] = []
                else:
                    self.o = 1.
                    self._accept_outputs([sample], time)
                continuing = low and sample['outward']
                a['bad'] = 0 if continuing else a['bad'] + 1
                completed = a['opened'] and a['bad'] >= 3
            else:
                a['low'] = a['bad'] = 0
                a['buffer'] = []
                a['pending_hits'] = []
                a['output_start'] = None
                if not unique or a['identity_lost']:
                    self.log.append(dict(kind='discarded', time_s=time, track_id=a['id']))
                    self.active = a = None
                    self.histories = {}
                    self.selected = None
                    self.f = self.p = self.o = self.d = 0.
        if a is None:
            self._search(rows, time)
            a = self.active
        record = self._crossing_record(crossing, time)
        if record is not None and record['eligible']:
            a['pending_hits'].append(record)
        if a is not None and a['opened'] and a['pending_hits']:
            hit = a['pending_hits'][0]
            hit['opening_confirmed'] = True
            hit['opening_confirmed_s'] = a['opening_confirmed_s']
            self.selected = deepcopy(hit)
            self._end('hit', time)
        elif completed:
            self._end('event_end', time)
        elif time >= 7. - 1e-10:
            self._end('timeout', 7.)
        else:
            self.time = time
        self._previous = (time, p.copy(), v.copy())
        return self.total - old_total

    @property
    def qualified(self):
        return self.terminal == 'hit'

    def features(self):
        a = self.active
        selected = self.selected or {}
        values = [self.f, self.p, self.o, self.d, self.progress,
            a is not None, self.formed, self.propagated, self.opened,
            0. if a is None else a['s_a'] / self.L,
            0. if a is None else a['max_s'] / self.L,
            0. if a is None else a['tail'] / 9.,
            0. if a is None else a['tail_turn'] / (2 * math.pi),
            0. if a is None else a['low'] / 3.,
            0. if a is None else a['bad'] / 3.,
            False if a is None else a['identity_lost'],
            selected.get('H', 0.), selected.get('delivery_guide', 0.),
            selected.get('rectangle_distance_m', 0.), selected.get('target_alignment', 0.),
            False if a is None else bool(a['pending_hits']),
            0. if a is None or a['output_start'] is None else a['output_start'] / 7.,
            self.time / 7., self.progress_credit / 5.]
        assert len(values) == FEATURE_SIZE
        return np.asarray(values, dtype=np.float32)

    def record(self):
        return dict(terminal=self.terminal, terminal_time_s=self.time,
            selected=deepcopy(self.selected), events=deepcopy(self.log),
            crossings=deepcopy(self.crossings), qualified=self.qualified,
            episode_return=self.total, progress=self.progress,
            progress_credit=self.progress_credit, outcome_credit=self.outcome_credit,
            progress_source_event_id=self.progress_source_event_id,
            terminal_event_id=self.terminal_event_id,
            progress_source=deepcopy(self.progress_source))
