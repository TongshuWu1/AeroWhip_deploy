"""Provisional replay scores, not a whip classifier or complete training reward."""

from dataclasses import dataclass
import math

import torch
import torch.nn.functional as F


@dataclass(frozen=True)
class WhipScoreConfiguration:
    region_fractions: tuple[float, float, float, float] = (0., 1/3, 2/3, 1.)
    baseline_window_s: float = .30
    minimum_region_delay_s: float = .02
    maximum_region_delay_s: float = .25
    maximum_progression_s: float = .40
    tip_window_s: float = .15
    bend_reference: float = 1.
    speed_reference_mps: float = 5.
    target_width_m: float = .30
    whip_weight: float = .2
    target_weight: float = .8

    def __post_init__(self):
        fractions = tuple(self.region_fractions)
        if (len(fractions) != 4 or not all(math.isfinite(x) for x in fractions)
                or fractions[0] != 0 or fractions[-1] != 1
                or not all(a < b for a, b in zip(fractions, fractions[1:]))):
            raise ValueError('Four region fractions must increase from zero to one')
        object.__setattr__(self, 'region_fractions', fractions)
        for name in ('baseline_window_s', 'minimum_region_delay_s', 'maximum_region_delay_s',
                     'maximum_progression_s', 'tip_window_s', 'bend_reference',
                     'speed_reference_mps', 'target_width_m', 'whip_weight', 'target_weight'):
            value = getattr(self, name)
            if not math.isfinite(value) or value <= 0:
                raise ValueError(f'{name} must be positive and finite')
        if (self.minimum_region_delay_s > self.maximum_region_delay_s
                or self.maximum_progression_s < 2*self.minimum_region_delay_s):
            raise ValueError('Inconsistent progression timing bounds')


@dataclass(frozen=True)
class WhipScore:
    valid: torch.Tensor
    has_event: torch.Tensor
    score: torch.Tensor
    bend_score: torch.Tensor
    speed_score: torch.Tensor
    quality_score: torch.Tensor
    target_score: torch.Tensor | None
    target_distance_m: torch.Tensor | None
    event_indices: torch.Tensor  # P, M, D, tip; -1 if no event
    event_times_s: torch.Tensor  # NaN if no event
    prominence: torch.Tensor
    score_history: torch.Tensor


class WhipScorer:
    """Cache a uniform time grid and evaluate batches entirely on their device.

    Inputs: dimensionless regional bending (B,T,3), WORLD tip velocity and
    position (B,T,3), sample validity (B,T), regional-event windows (B,2).
    Windows are closed; tip sampling may extend tip_window_s beyond their end.
    Baselines use preceding recorded samples, including before the window.
    Truncated baselines at the beginning of the history use available samples.

    All supplied samples must be valid: no bridging gaps or scoring frozen
    invalid states. Invalid scores are NaN, NOT learner-ready failure returns.
    Valid histories with no positive event score zero and have no event times.
    Missing targets return unweighted B*V quality, with target fields None.

    The exact discrete search maximizes over ordered sample tuples, NOT local
    peaks or a tracked curvature ridge. Equal scores choose the first maximum
    at each reduction. No temporal filtering, command cost, safety gate,
    target collision model, or curriculum is implicit in this call.
    """

    def __init__(self, time_s, configuration=WhipScoreConfiguration()):
        if (not isinstance(time_s, torch.Tensor) or time_s.ndim != 1 or len(time_s) < 2
                or time_s.dtype not in (torch.float32, torch.float64)):
            raise ValueError('Provide a float32/float64 time vector with at least two samples')
        self.time = time_s.detach().clone()
        self.configuration = configuration
        delta = torch.diff(self.time)
        self.dt = float(delta[0])
        tolerance = 16*torch.finfo(time_s.dtype).eps*max(1., float(time_s.abs().max()))
        if (not bool(torch.isfinite(time_s).all()) or self.dt <= 0
                or not bool((delta > 0).all())
                or not bool(torch.allclose(delta, delta[:1].expand_as(delta), rtol=0, atol=tolerance))):
            raise ValueError('Scoring requires finite, strictly increasing, uniformly sampled times')

        def steps(value):
            count = round(value/self.dt)
            if not math.isclose(count*self.dt, value, abs_tol=tolerance, rel_tol=0):
                raise ValueError('Scoring windows must be exact multiples of the sample interval')
            return count

        self.baseline = steps(configuration.baseline_window_s)
        self.lo = steps(configuration.minimum_region_delay_s)
        self.hi = steps(configuration.maximum_region_delay_s)
        self.total = steps(configuration.maximum_progression_s)
        self.tip = steps(configuration.tip_window_s)
        if self.lo < 1:
            raise ValueError('Progression delay must span at least one sample')
        self.index = torch.arange(len(time_s), device=time_s.device)
        self.tolerance = tolerance

    def _check(self, value, shape, name, dtype=None):
        if (not isinstance(value, torch.Tensor) or tuple(value.shape) != shape
                or value.device != self.time.device or value.dtype != (dtype or self.time.dtype)):
            raise ValueError(f'{name} must have shape {shape} and the configured dtype/device')

    def _candidate_mask(self, prominence):
        """No restriction in the original scorer; analysis variants may restrict candidates."""
        return None

    @torch.no_grad()
    def __call__(self, bending, tip_positions, tip_velocities, sample_valid,
                 event_windows_s, target_positions=None):
        if not isinstance(bending, torch.Tensor) or bending.ndim != 3 or len(bending) < 1:
            raise ValueError('Bending must be a nonempty (batch,times,3) tensor')
        batch, count = len(bending), len(self.time)
        for name, value in (('Bending', bending), ('Tip position', tip_positions),
                            ('Tip velocity', tip_velocities)):
            self._check(value, (batch, count, 3), name)
        self._check(sample_valid, (batch, count), 'Sample validity', torch.bool)
        self._check(event_windows_s, (batch, 2), 'Event windows')
        if target_positions is not None:
            self._check(target_positions, (batch, 3), 'Targets')
        valid = (sample_valid.all(-1) & torch.isfinite(bending).flatten(1).all(-1)
                 & (bending >= 0).flatten(1).all(-1)
                 & torch.isfinite(tip_positions).flatten(1).all(-1)
                 & torch.isfinite(tip_velocities).flatten(1).all(-1)
                 & torch.isfinite(event_windows_s).all(-1)
                 & (event_windows_s[:, 0] <= event_windows_s[:, 1])
                 & (event_windows_s[:, 0] >= self.time[0]-self.tolerance)
                 & (event_windows_s[:, 1] <= self.time[-1]+self.tolerance))
        if target_positions is not None:
            valid &= torch.isfinite(target_positions).all(-1)
        # Sanitize only to keep failed lanes arithmetically isolated. They are
        # still rejected and returned as NaN, never converted to good episodes.
        clean = torch.where(valid[:, None, None], bending, 0.)
        series = clean.transpose(1, 2)
        minimum = -F.max_pool1d(F.pad(-series, (self.baseline, 0), value=-float('inf')),
                                self.baseline+1, stride=1)
        prominence = (series-minimum).transpose(1, 2).clamp_min(0)
        cfg = self.configuration
        amplitude = prominence/(prominence+cfg.bend_reference)
        inside = ((self.time[None] >= event_windows_s[:, :1]-self.tolerance)
                  & (self.time[None] <= event_windows_s[:, 1:]+self.tolerance)
                  & valid[:, None])
        amplitude = amplitude.masked_fill(~inside[..., None], -1.)
        candidate_mask = self._candidate_mask(prominence)
        if candidate_mask is not None:
            amplitude = amplitude.masked_fill(~candidate_mask, -1.)
        best = bending.new_full((batch, count), -1.)
        proximal = torch.full_like(best, -1, dtype=torch.long)
        middle = proximal.clone()
        # For each middle->distal lag, select the best admissible proximal
        # sample by a rolling maximum. This avoids a B*T*T*T candidate tensor.
        for lag in range(self.lo, min(self.hi, self.total-self.lo, count-1)+1):
            upper = min(self.hi, self.total-lag)
            windows = F.pad(amplitude[..., 0], (upper, 0), value=-1.).unfold(
                -1, upper-self.lo+1, 1)[:, :count]
            pvalue, offset = windows.max(-1)
            pindex = self.index[None]-upper+offset
            mid_index = self.index-lag
            m = mid_index.clamp_min(0)
            candidate = pvalue[:, m]*amplitude[:, m, 1]*amplitude[..., 2]
            allowed = ((mid_index[None] >= 0) & (pvalue[:, m] >= 0)
                       & (amplitude[:, m, 1] >= 0) & (amplitude[..., 2] >= 0))
            candidate = candidate.masked_fill(~allowed, -1.)
            improve = candidate > best
            best = torch.where(improve, candidate, best)
            proximal = torch.where(improve, pindex[:, m], proximal)
            middle = torch.where(improve, mid_index[None], middle)
        bend = best.clamp_min(0).pow(1/3)
        # Best distal event within the preceding tip window. Future tip
        # speed/accuracy do not depend on P/M, so these reductions are exact.
        recent, offset = F.pad(bend, (self.tip, 0), value=-1.).unfold(
            -1, self.tip+1, 1).max(-1)
        distal = self.index[None]-self.tip+offset
        speed = torch.linalg.vector_norm(torch.where(valid[:, None, None], tip_velocities, 0.), dim=-1)
        valid &= torch.isfinite(speed).all(-1)
        speed_score = speed/(speed+cfg.speed_reference_mps)
        quality = recent*speed_score
        distance = accuracy = None
        history = quality
        if target_positions is not None:
            distance = torch.linalg.vector_norm(tip_positions-target_positions[:, None], dim=-1)
            valid &= torch.isfinite(distance).all(-1)
            accuracy = torch.exp(-.5*(distance/cfg.target_width_m).square())
            history = quality*(cfg.whip_weight+cfg.target_weight*accuracy)
        permitted_tip = ((self.time[None] >= event_windows_s[:, :1]-self.tolerance)
                         & (self.time[None] <= event_windows_s[:, 1:]+cfg.tip_window_s+self.tolerance))
        history = torch.where(permitted_tip & valid[:, None], history, 0.)
        score, tip_index = history.max(-1)
        has_event = valid & (score > 0)
        rows = torch.arange(batch, device=bending.device)
        d = distal[rows, tip_index].clamp_min(0)
        indices = torch.stack((proximal[rows, d], middle[rows, d], d, tip_index), dim=-1)
        indices = indices.masked_fill(~has_event[:, None], -1)
        times = self.time[indices.clamp_min(0)].masked_fill(~has_event[:, None], float('nan'))

        def selected(values):
            return torch.where(has_event, values[rows, tip_index], float('nan'))

        return WhipScore(valid, has_event, score.masked_fill(~valid, float('nan')),
            selected(recent), selected(speed_score), selected(quality),
            None if accuracy is None else selected(accuracy),
            None if distance is None else selected(distance), indices, times,
            prominence.masked_fill(~valid[:, None, None], float('nan')),
            history.masked_fill(~valid[:, None], float('nan')))
