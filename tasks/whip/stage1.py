"""Isolated Stage-1 shaping objective; not a whip classifier or training runner."""

from dataclasses import dataclass
import math

import torch

from .stage1_configuration import Stage1Configuration


def directed_alignment(forward, tip_speed, configuration):
    """Same-event alignment, never a geometric hit gate.

    The explicit cone version is continuous (not C1 at the plateau boundary).
    Historical mode deliberately retains the original operations exactly.
    """
    if configuration.alignment_mode == 'squared_cosine_v1':
        return (forward/tip_speed.clamp_min(1e-12)).clamp(0, 1).square()
    valid = torch.isfinite(forward) & torch.isfinite(tip_speed) & (tip_speed > 1e-12)
    safe_forward = torch.where(valid, forward, torch.zeros_like(forward))
    safe_speed = torch.where(valid, tip_speed, torch.ones_like(tip_speed))
    cosine2 = (safe_forward/safe_speed).clamp(0, 1).square()
    return (cosine2/math.cos(math.radians(configuration.alignment_cone_deg))**2).clamp_max(1)


def _sat(value, scale):
    value = value.clamp_min(0)
    return value/(value+scale)


def _priority_event_components(mechanical, matching, qualified, configuration,
                               strike_weight, factor):
    """Bounded current credit; no new physical success/whip threshold.

    A miss is bounded by the configured support weight. Current Stage1 uses
    support 1 / strike 3; Stage2/3 use support .2 / strike 1. For M>=.1,
    their minimum centered/timed bonuses exceed the corresponding miss bound.
    Boundary costs still apply and can reverse NET return ordering.
    Zero mechanics earns no hit bonus, so rigid translation is not rewarded.
    """
    weighted=mechanical*factor*configuration.shaping_weight
    return (weighted*configuration.mechanics_credit_weight,
            weighted*(1-configuration.mechanics_credit_weight)*matching,
            _sat(mechanical,configuration.strike_mechanics_scale)*factor*strike_weight*qualified)


@dataclass(frozen=True)
class Stage1Score:
    score: torch.Tensor
    valid: torch.Tensor
    has_event: torch.Tensor
    event_indices: torch.Tensor
    components: dict[str, torch.Tensor]
    # Best complete event available at each 100-Hz observation. Invalid future
    # samples never erase or contribute to an already observed valid prefix.
    causal_best: torch.Tensor


class Stage1Scorer:
    """Score fixed-request batches on a cached 100 Hz mesh/time grid.

    Explicit directed_whip_v1 selects max_e M(e)*U(e)*D(e), where U is
    recovered chord/arc slack from the minimum within that bend window and
    D is squared positive forward velocity alignment. No wall interpolation,
    matching, strike or time credit enters that mode; wall crossings remain
    diagnostics. This experimental event proxy is not a physical whip classifier.

    Explicit directed_delivery_v1 retains W=M*U*D at the same actual event and
    selects max_e W(e)*(1+P(e)), P=exp(-.5*(rectangle_distance/approach_width)^2).
    P is uniform on the whole rectangle, independent of speed/angle thresholds.
    This is broad target guidance, not a contact/hit bonus or target curriculum.

    directed_precision_v1 selects max_e W(e)*(1+I_forward_cross(e)*C(e)),
    C=exp(-(lateral^2+vertical^2)/(2*center_sigma^2)). The same M/U/D
    formula is evaluated on the interpolated whole-cable crossing state;
    noncrossing samples retain mechanics support. Crossing credit is delayed
    until its interval endpoint is observed. Positive bound2, no hard speed/
    angle gate, repeated-event accumulation, contact model or early termination.

    max_e M(e) * [shaping_weight*(lambda + (1-lambda)*A(e)) + strike_weight*I_strike(e)]
        * (1-beta*t_e/T).
    A is smooth request matching. The optional directed-wall bonus is earned
    only at a qualified crossing, with mechanics measured at that SAME event.
    Maximum selection does not accumulate credit for repeated crossings.
    lambda is mechanics_credit_weight: partial mechanics credit despite a miss.
    Missing/zero lambda preserves saved tasks' original objective. This is an
    explicit objective change, not policy-invariant reward redistribution.
    M is the existing experimental transport/release/forward proxy, including
    its cubic root and known limitations. Every term uses the SAME event.
    The Gaussian is outside the cubic root. Direction is requested tip velocity,
    not drone yaw. Timing is a preference, not an exact timestamp request.

    Current explicit strike_mechanics_scale selects
    [shaping_weight*M*A + strike_weight*I_strike*M/(M+scale)]*center*time
    (with the configured independent mechanics credit if nonzero). Legacy
    snapshots without that selection retain the above arithmetic/search bounds.
    Current search includes the final horizon endpoint. The minimum .2-s bend
    window still needs observed history: very early hits are not fabricated
    into developed mechanics. Neither the scale nor score defines a physical
    whip threshold, and boundary costs are accounted separately by the caller.

    Positions/velocities are (B,T,N,3), state_valid is bool (B,T). Any invalid
    sample invalidates the whole episode, including failures after the event,
    unless an explicit terminal_samples prefix is supplied. Staged first-hit
    mode excludes the unexecuted suffix and selects terminal credit at that hit.
    A valid zero-score episode has has_event=False and indices=-1, not failure.
    Components are NaN when no positive event exists. No physical qualification
    or success threshold is implied by has_event or a positive score.
    causal_best retains the running maximum of complete, observed events even
    for a subsequently failed episode; it is not a validity override. The caller
    must reconcile intermediate credit with the original terminal failure rule.

    Construction validates values and can synchronize CUDA. Calls are batched,
    forward-only and device-resident; no rollout, target or learner is required.
    The inherited GPU pseudoinverse is not graph-capturable on the checked
    backend; ordinary CUDA evaluation works. No graph/synchronization guarantee.
    """

    @torch.no_grad()
    def __init__(self, rest_lengths, time_s, configuration: Stage1Configuration, *, initial_position=None):
        if (not isinstance(rest_lengths, torch.Tensor) or rest_lengths.ndim != 1
                or rest_lengths.numel() < 3 or rest_lengths.dtype not in (torch.float32, torch.float64)
                or not bool((torch.isfinite(rest_lengths) & (rest_lengths > 0)).all())):
            raise ValueError('Provide at least three positive finite rest edges as a floating tensor')
        self.configuration = configuration
        self.horizon = configuration.episode.duration_s+configuration.episode.post_hold_s
        count = round(self.horizon/.01)+1
        if (not isinstance(time_s, torch.Tensor) or time_s.shape != (count,)
                or time_s.dtype != rest_lengths.dtype or time_s.device != rest_lengths.device):
            raise ValueError('Time grid must cover the configured horizon on the mesh dtype/device')
        expected = torch.arange(count, dtype=time_s.dtype, device=time_s.device)*.01
        if not bool(torch.isfinite(time_s).all()) or not torch.allclose(
                time_s-time_s[0], expected, rtol=0, atol=32*torch.finfo(time_s.dtype).eps*max(1, self.horizon)):
            raise ValueError('Stage-1 objective requires uniformly sampled 100 Hz states')
        self.rest, self.time = rest_lengths.detach().clone(), time_s.detach().clone()
        self.elapsed = expected
        self.length = self.rest.sum()
        if not bool(torch.isfinite(self.length)):
            raise ValueError('Cable length must be finite')
        self.s = self.rest.cumsum(0)[:-1]
        self.dual = (self.rest[:-1]+self.rest[1:])/2
        self.ds = self.s.diff()
        cfg = configuration.mechanics
        kernel = torch.exp(-.5*((self.s[:, None]-self.s[None, :])/(cfg.smoothing_sigma_fraction*self.length))**2)
        self.kernel = kernel/(kernel*self.dual[None]).sum(-1, keepdim=True)
        self.weights = torch.cat((self.rest[:1], self.rest[:-1]+self.rest[1:], self.rest[-1:]))/(2*self.length)
        self.eye = torch.eye(3, device=self.rest.device, dtype=self.rest.dtype)
        material = torch.cat((self.rest.new_zeros(1), self.rest.cumsum(0)))
        self.fallback = torch.stack((torch.zeros_like(material), torch.zeros_like(material), -material), -1)
        heading, elevation, self.height, self.speed = configuration.request
        self.direction = self.rest.new_tensor([math.cos(elevation)*math.cos(heading),
                                              math.cos(elevation)*math.sin(heading), math.sin(elevation)])
        self.surface = configuration.strike_surface
        if self.surface is not None:
            if initial_position is None:
                raise ValueError('The wall requires the fixed initial tracking position')
            initial = torch.as_tensor(initial_position, dtype=self.rest.dtype, device=self.rest.device)
            if initial.shape != (3,) or not bool(torch.isfinite(initial).all()):
                raise ValueError('The wall requires a finite initial tracking position, not a moving origin')
            self.center = initial+self.surface.distance_m*self.direction
            self.center[2] = self.height
        lag = round(cfg.tip_lag_s/.01)
        start, stop = round(cfg.event_start_s/.01), round(cfg.event_end_s/.01)
        self.windows = []
        for seconds in cfg.windows_s:
            h = round(seconds/.01)
            begins = torch.arange(start, stop-h+1, round(cfg.window_stride_s/.01), device=self.rest.device)
            if (configuration.matching.strike_mechanics_scale is not None
                    or configuration.matching.reward_mode in ('directed_whip_v1', 'directed_delivery_v1', 'directed_precision_v1')) and int(begins[-1]) != stop-h:
                # The 50-ms stride need not land on the final search endpoint.
                # Include it explicitly so the last accepted crossing interval
                # has an eligible SAME-event bend window and tip lag.
                begins=torch.cat((begins,begins.new_tensor([stop-h])))
            ends = begins+h
            tips = ends[:, None]+torch.arange(lag+1, device=self.rest.device)
            self.windows.append((h, begins, ends, tips))

    def _nonrigid_tip_speed(self, positions, velocities):
        # Same material-weighted rigid least-squares fit as the research score;
        # this is a kinematic residual, NOT an energy estimate or dynamics MLP.
        r = positions-(positions*self.weights[:, None]).sum(-2, keepdim=True)
        dv = velocities-(velocities*self.weights[:, None]).sum(-2, keepdim=True)
        inertia = ((r.square().sum(-1)[..., None, None]*self.eye-r[..., :, None]*r[..., None, :])
                   *self.weights[:, None, None]).sum(-3)
        rhs = (torch.linalg.cross(r, dv, dim=-1)*self.weights[:, None]).sum(-2)
        finite = torch.isfinite(inertia).flatten(2).all(-1) & torch.isfinite(rhs).all(-1)
        safe = torch.where(finite[..., None, None], inertia, self.eye).reshape(-1, 3, 3)
        # Bound eigensolver workspace without changing any per-matrix math.
        # Larger chunks reserve several GiB on CUDA during training scoring.
        inverse = torch.cat([torch.linalg.pinv(part, hermitian=True,
            rtol=self.configuration.mechanics.rigid_pinv_rtol) for part in safe.split(2048)], 0).reshape_as(inertia)
        omega = (inverse @ torch.where(finite[..., None], rhs, 0)[..., None])[..., 0]
        residual = dv[..., -1, :]-torch.linalg.cross(omega, r[..., -1, :], dim=-1)
        return torch.linalg.vector_norm(residual, dim=-1).masked_fill(~finite, float('nan'))

    def _directed_hit_score(self, surface, prefix_valid, valid, initial_wall_distance):
        """Simple experimental task reward, not a physical-whip classifier.

        Pay only new records of spatial proximity, bounded by shaping_weight;
        no per-cycle credit, velocity/angle/mechanics conjunction or retreat
        charge. directed_hit_v1 preserves strict crossing qualification;
        geometric_hit_quality_v1 uses first geometric forward target crossing
        with bounded speed and straightness credit, strict qualification diagnostic only.
        Elapsed-time and workspace charges are delivered separately by finish.
        No potential-based optimal-policy-invariance claim is made.
        """
        match = self.configuration.matching
        count = prefix_valid.shape[1]
        ready = (torch.arange(count, device=prefix_valid.device)+1).clamp_max(count-1)
        available = prefix_valid[:, ready]
        quality_mode = match.reward_mode == 'geometric_hit_quality_v1'
        qualified = surface['geometric_hit' if quality_mode else 'qualified'] & available
        hit = qualified.any(1)
        first_hit = surface['elapsed'].masked_fill(~qualified, float('inf')).argmin(1)
        if quality_mode:
            # Even a saved full-horizon diagnostic must not credit later hits or
            # improve the approach after the first geometric target crossing.
            available &= ~hit[:, None] | (ready[None] <= (first_hit+1)[:, None])
            qualified &= torch.arange(count, device=qualified.device)[None] == first_hit[:, None]
        # Use the actual initial position, not a future interpolated event.
        proximity = torch.exp(-.5*(surface['distance']/self.surface.approach_width_m).square())
        initial = torch.exp(-.5*(initial_wall_distance/self.surface.approach_width_m).square())
        progress = match.shaping_weight*(proximity-initial[:, None]).clamp_min(0)
        progress = torch.where(available, progress, 0.)
        observed_credit = torch.zeros_like(progress)
        observed_credit.scatter_reduce_(1, ready[None].expand_as(progress), progress,
                                        reduce='amax', include_self=True)
        best_progress = observed_credit.cummax(1).values
        hit_credit = torch.zeros_like(progress)
        speed_credit = torch.zeros_like(progress)
        straight_credit = torch.zeros_like(progress)
        if quality_mode:
            forward = surface['forward'].clamp_min(0)
            speed_credit = match.hit_speed_weight*(forward/(forward+match.hit_speed_scale_mps))*qualified
            straight_credit = match.hit_straightness_weight*surface['angle'].cos().clamp_min(0).square()*qualified
        event_credit = self.surface.strike_bonus_weight*qualified.to(progress.dtype)+speed_credit+straight_credit
        hit_credit.scatter_reduce_(1, ready[None].expand_as(progress),
            event_credit,
            reduce='amax', include_self=True)
        causal = best_progress+hit_credit.cummax(1).values
        terminal = prefix_valid.long().sum(1).clamp_min(1)-1
        approach = best_progress.gather(1, terminal[:, None])[:, 0]
        bonus = self.surface.strike_bonus_weight*hit.to(progress.dtype)
        score = (approach+bonus+speed_credit.sum(1)+straight_credit.sum(1)).masked_fill(~valid, float('nan'))
        selected = torch.where(hit, first_hit, progress.argmax(1))
        take = lambda value: value.gather(1, selected[:, None])[:, 0]
        has_event = valid & ((approach > 0) | hit)
        diagnostic = lambda value: value.masked_fill(~has_event, float('nan'))
        components = dict(
            mechanics_credit=torch.zeros_like(score).masked_fill(~valid, float('nan')),
            approach_reward=approach.masked_fill(~valid, float('nan')),
            strike_bonus=bonus.masked_fill(~valid, float('nan')),
            matching=diagnostic(take(proximity)),
            mechanical=torch.full_like(score, float('nan')),  # saved-motion diagnostic only
            wall_distance_m=diagnostic(take(surface['distance'])),
            plane_crossing=diagnostic(take(surface['crossing'].to(progress.dtype))),
            directed_strike=diagnostic(take(qualified.to(progress.dtype))),
            target_lateral_error_m=diagnostic(take(surface['lateral'])),
            target_vertical_error_m=diagnostic(take(surface['vertical'])),
            forward_mps=diagnostic(take(surface['forward'])),
            tip_speed_mps=diagnostic(take(surface['speed'])),
            direction_error_rad=diagnostic(take(surface['angle'])),
            release_time_s=diagnostic(take(surface['elapsed'])),
            release_height_world_m=diagnostic(take(surface['positions'][:, :, -1, 2])),
            strike_success=hit.to(progress.dtype).masked_fill(~valid, float('nan')),
            strike_count=qualified.sum(1).to(progress.dtype).masked_fill(~valid, float('nan')),
            first_strike_time_s=surface['elapsed'].masked_fill(~qualified, float('inf')).amin(1)
                .masked_fill(~valid | ~hit, float('nan')))
        if quality_mode:
            components.update(
                hit_speed_bonus=speed_credit.sum(1).masked_fill(~valid, float('nan')),
                hit_straightness_bonus=straight_credit.sum(1).masked_fill(~valid, float('nan')),
                geometric_hit_success=hit.to(progress.dtype).masked_fill(~valid, float('nan')),
                strict_strike_success=(surface['qualified'] & qualified).any(1).to(progress.dtype)
                    .masked_fill(~valid, float('nan')))
        # No invented bend-window indices for a spatial approach event.
        indices = torch.stack((-torch.ones_like(selected), -torch.ones_like(selected), selected), -1)
        return Stage1Score(score, valid, has_event, indices.masked_fill(~has_event[:, None], -1),
                           components, causal)

    @torch.no_grad()
    def __call__(self, positions, velocities, state_valid, *, terminal_samples=None):
        if self.configuration.termination in ('first_qualified_hit_v1', 'first_geometric_hit_v1') and terminal_samples is None:
            raise ValueError('First-hit tasks require explicit executed terminal_samples; padded suffixes are not episodes')
        if (not isinstance(positions, torch.Tensor) or positions.ndim != 4 or positions.shape[0] < 1
                or positions.shape[1:] != (len(self.time), len(self.rest)+1, 3)
                or positions.dtype != self.rest.dtype or positions.device != self.rest.device):
            raise ValueError('Positions must be (batch, times, nodes, 3) on the cached mesh dtype/device')
        if (not isinstance(velocities, torch.Tensor) or velocities.shape != positions.shape
                or velocities.dtype != positions.dtype or velocities.device != positions.device):
            raise ValueError('Velocities must match positions')
        if (not isinstance(state_valid, torch.Tensor) or state_valid.shape != positions.shape[:2]
                or state_valid.dtype != torch.bool or state_valid.device != positions.device):
            raise ValueError('State validity must be bool (batch, times) on the history device')
        lengths = torch.linalg.vector_norm(positions.diff(dim=2), dim=-1)
        sample_valid = (state_valid & torch.isfinite(positions).flatten(2).all(-1)
                        & torch.isfinite(velocities).flatten(2).all(-1)
                        & (torch.isfinite(lengths) & (lengths > 0)).all(-1))
        prefix_valid = sample_valid.to(torch.int64).cumprod(1).bool()
        if terminal_samples is not None:
            if (terminal_samples.shape != (len(positions),) or terminal_samples.device != positions.device
                    or terminal_samples.dtype != torch.int64
                    or bool(((terminal_samples < 1) | (terminal_samples > len(self.time))).any())):
                raise ValueError('Terminal samples must give each executed inclusive prefix length')
            valid = prefix_valid.gather(1, (terminal_samples-1)[:, None])[:, 0]
            prefix_valid &= torch.arange(len(self.time), device=positions.device)[None] < terminal_samples[:, None]
        else:
            valid = prefix_valid[:, -1]
        # Sanitize invalid suffixes BEFORE linear algebra, preserving only the
        # observed prefix for causal credit. The full episode is still invalid.
        p = torch.where(prefix_valid[..., None, None], positions, self.fallback)
        v = torch.where(prefix_valid[..., None, None], velocities, 0)
        if self.configuration.matching.reward_mode in ('directed_hit_v1', 'geometric_hit_quality_v1'):
            from .strike import surface_events
            surface = surface_events(p, v, self.time, self.center, self.direction, self.surface, self.speed)
            # Per-call/per-lane baseline supports physically different starts.
            delta = p[:, 0, -1]-self.center
            side = torch.stack((-self.direction[1], self.direction[0], self.direction.new_zeros(())))
            initial_wall_distance = torch.stack(((delta*self.direction).sum(-1),
                ((delta*side).sum(-1).abs()-self.surface.width_m/2).clamp_min(0),
                (delta[:, 2].abs()-self.surface.height_m/2).clamp_min(0)), -1).norm(dim=-1)
            return self._directed_hit_score(surface, prefix_valid, valid, initial_wall_distance)
        edges = p.diff(dim=2)
        tangents = edges/torch.linalg.vector_norm(edges, dim=-1, keepdim=True).clamp_min(1e-12)
        theta = torch.atan2(torch.linalg.vector_norm(torch.linalg.cross(tangents[:, :, :-1],
            tangents[:, :, 1:], dim=-1), dim=-1), (tangents[:, :, :-1]*tangents[:, :, 1:]).sum(-1))
        q = theta @ self.kernel.T
        turn = (q*self.dual).sum(-1)
        mu = (q*self.dual*self.s).sum(-1)/turn.clamp_min(1e-12)
        spread = ((q*self.dual*(self.s-mu[..., None]).square()).sum(-1)/turn.clamp_min(1e-12)).sqrt()
        qt = (q[:, 1:, 1:]+q[:, 1:, :-1]-q[:, :-1, 1:]-q[:, :-1, :-1])/.02
        qs = (q[:, 1:, 1:]+q[:, :-1, 1:]-q[:, 1:, :-1]-q[:, :-1, :-1])/(2*self.ds)
        abc = torch.stack(((qt.square()*self.ds).sum(-1), (qs.square()*self.ds).sum(-1),
                           -(qt*qs*self.ds).sum(-1)), -1)
        prefix = torch.cat((torch.zeros_like(abc[:, :1]), abc.cumsum(1)), 1)
        event_p, event_v = p, v
        event_time = self.elapsed[None].expand(len(p), -1)
        delivery_mode = self.configuration.matching.reward_mode == 'directed_delivery_v1'
        precision_mode = self.configuration.matching.reward_mode == 'directed_precision_v1'
        whip_mode = self.configuration.matching.reward_mode in ('directed_whip_v1', 'directed_delivery_v1', 'directed_precision_v1')
        interpolated_mode = precision_mode or not whip_mode
        surface = None
        if self.surface is not None:
            from .strike import surface_events
            surface = surface_events(p, v, self.time, self.center, self.direction, self.surface, self.speed)
            if terminal_samples is not None:
                ready = (torch.arange(len(self.time), device=p.device)+1).clamp_max(len(self.time)-1)
                observed_event = prefix_valid[:, ready]
                surface['qualified'] &= observed_event
                surface['crossing'] &= observed_event
            if interpolated_mode:
                event_p, event_v, event_time = surface['positions'], surface['velocities'], surface['elapsed']
        # Geometric unfolding is recovered chord/arc slack within the SAME
        # bend window and release. Translation/rotation cannot change this ratio.
        # Use actual arc length, not rest length: extension is not unfolding.
        chord_ratio = (p[:, :, -1]-p[:, :, 0]).norm(dim=-1)/edges.norm(dim=-1).sum(-1).clamp_min(1e-12)
        event_chord_ratio = chord_ratio
        if precision_mode:
            event_edges = event_p[:, :, 1:]-event_p[:, :, :-1]
            event_chord_ratio = (event_p[:, :, -1]-event_p[:, :, 0]).norm(dim=-1)/event_edges.norm(dim=-1).sum(-1).clamp_min(1e-12)
        nonrigid = self._nonrigid_tip_speed(event_p, event_v)
        relative_speed = torch.linalg.vector_norm(v[:, :, -1]-v[:, :, 0], dim=-1)
        event_relative = (relative_speed if surface is None else
                          torch.linalg.vector_norm(event_v[:, :, -1]-event_v[:, :, 0], dim=-1))
        tip_speed = torch.linalg.vector_norm(event_v[:, :, -1], dim=-1)
        forward = (event_v[:, :, -1]*self.direction).sum(-1)
        unit_velocity = event_v[:, :, -1]/tip_speed.clamp_min(1e-12)[..., None]
        angle = torch.atan2(torch.linalg.vector_norm(torch.linalg.cross(
            unit_velocity, self.direction.expand_as(unit_velocity), dim=-1), dim=-1),
            (unit_velocity*self.direction).sum(-1))
        release_angle = angle
        cfg, match = self.configuration.mechanics, self.configuration.matching
        precision_quality = None
        if precision_mode:
            forward_cross = surface['crossing'] & (surface['forward'] > 0)
            precision_quality = torch.exp(-.5*(surface['lateral'].square()+surface['vertical'].square())/match.center_sigma_m**2)
            precision_quality = precision_quality.masked_fill(~forward_cross, 0)
        target_proximity = None
        target_distance = None
        if delivery_mode:
            # Use the actual observed release sample, NOT an interpolated future
            # plane crossing from surface_events. Every factor shares this event.
            delta = event_p[:, :, -1]-self.center
            side = torch.stack((-self.direction[1], self.direction[0], self.direction.new_zeros(())))
            target_distance = torch.stack(((delta*self.direction).sum(-1),
                ((delta*side).sum(-1).abs()-self.surface.width_m/2).clamp_min(0),
                (delta[:, :, 2].abs()-self.surface.height_m/2).clamp_min(0)), -1).norm(dim=-1)
            target_proximity = torch.exp(-.5*(target_distance/self.surface.approach_width_m).square())
        height_error, speed_error = event_p[:, :, -1, 2]-self.height, tip_speed-self.speed
        # Preserve signed speed error in diagnostics; minimum mode penalizes
        # only shortfall, and still chooses ALL terms at the same event.
        matching_speed_error = speed_error.clamp_max(0) if match.speed_mode == 'minimum' else speed_error
        matching = torch.exp(-.5*((height_error/match.height_width_m).square()
            +(matching_speed_error/match.speed_width_mps).square()+(angle/math.radians(match.direction_width_deg)).square()))
        matching = matching.masked_fill(tip_speed <= 0, 0)
        if surface is not None:
            # Spatial matching itself is uniform inside the rectangle; the
            # separately configured center_factor below adds center preference.
            speed_error = forward-self.speed
            angle = surface['angle']
            spatial = (surface['distance']/self.surface.approach_width_m).square()
            if self.surface.overflow_width_m is not None:
                spatial = ((surface['normal_error']/self.surface.approach_width_m).square()
                    +((surface['lateral'].abs()-self.surface.width_m/2).clamp_min(0)/self.surface.overflow_width_m).square()
                    +((surface['vertical'].abs()-self.surface.height_m/2).clamp_min(0)/self.surface.overflow_width_m).square())
            matching = torch.exp(-.5*(spatial
                +(speed_error.clamp_max(0)/match.speed_width_mps).square()
                +(angle/math.radians(match.direction_width_deg)).square()))
            matching = matching.masked_fill(tip_speed <= 0, 0)
        if delivery_mode:
            # Delivery selection/reward uses the observed release, so its angle
            # diagnostic must not inherit an interpolated wall-crossing angle.
            angle = release_angle
        strike_weight = self.surface.strike_bonus_weight if surface is not None else 0.0
        credit = match.mechanics_credit_weight
        # Preserve the legacy arithmetic when loading snapshots without credit.
        reward_matching = matching if credit == 0 else credit+(1-credit)*matching
        reward_matching = match.shaping_weight*reward_matching
        if strike_weight > 0:
            reward_matching = reward_matching+strike_weight*surface['qualified'].to(p.dtype)
        center_factor = torch.ones_like(matching)
        if match.center_weight > 0:
            if surface is None:
                raise ValueError('Center preference requires a wall surface')
            center_factor = (1-match.center_weight)+match.center_weight*torch.exp(
                -.5*(surface['lateral'].square()+surface['vertical'].square())/match.center_sigma_m**2)
            reward_matching = reward_matching*center_factor
        first_hit = None
        if terminal_samples is not None and self.configuration.termination == 'first_qualified_hit_v1':
            first_hit = surface['elapsed'].masked_fill(~surface['qualified'], float('inf')).argmin(1)
            hit_exists = surface['qualified'].any(1)
        time_factor = 1-match.time_preference*event_time/self.horizon
        for value in (turn, mu, spread, prefix, nonrigid, relative_speed, tip_speed, matching):
            valid = valid & torch.isfinite(value).flatten(1).all(-1)
        fields = []
        event_scores = torch.zeros_like(state_valid, dtype=p.dtype)
        for h, begins, ends, tips in self.windows:
            a, b, c = (prefix[:, ends]-prefix[:, begins]).unbind(-1)
            coherence = (c/(a*b).clamp_min(1e-24).sqrt()).clamp(0, 1)
            transport = (c/b.clamp_min(1e-12)).clamp_min(0)
            travel = torch.minimum((mu[:, ends]-mu[:, begins]).clamp_min(0), transport*(h*.01))
            minimum_turn = turn.unfold(1, h+1, 1)[:, begins].amin(-1)
            maximum_spread = spread.unfold(1, h+1, 1)[:, begins].amax(-1)
            bend = (coherence.square()*_sat(minimum_turn, cfg.turn_reference_rad)
                *_sat(travel, cfg.travel_reference_fraction*self.length)
                *torch.exp(-(maximum_spread/(cfg.spread_scale_fraction*self.length))**4))
            pre_speed = relative_speed.unfold(1, max(1, h//5), 1)[:, begins].mean(-1)
            release = (event_relative[:, tips]-pre_speed[..., None]).clamp_min(0)
            pulse = _sat(nonrigid[:, tips], cfg.nonrigid_reference_mps)*_sat(release, cfg.release_reference_mps)
            mechanical = (bend[..., None]*pulse*_sat(forward[:, tips], cfg.forward_reference_mps)).clamp_min(0).pow(1/3)
            objective = mechanical*reward_matching[:, tips]*time_factor[:, tips]
            if whip_mode:
                folded = chord_ratio.unfold(1, h+1, 1)[:, begins].amin(-1)
                unfolding = ((event_chord_ratio[:, tips]-folded[..., None]).clamp_min(0)
                    /(1-folded[..., None]).clamp_min(1e-12)).clamp_max(1)
                direction = directed_alignment(forward[:, tips], tip_speed[:, tips], match)
                # One bounded event, without wall matching, hit, time or speed
                # surplus bonuses. Smooth direction preference has no angle gate.
                objective = mechanical*unfolding*direction
                if delivery_mode:
                    objective = objective*(1+target_proximity[:, tips])
                if precision_mode:
                    objective = objective*(1+precision_quality[:, tips])
            priority_components=None
            if match.strike_mechanics_scale is not None:
                priority_components=_priority_event_components(mechanical,matching[:,tips],
                    surface['qualified'][:,tips].to(p.dtype),match,strike_weight,
                    center_factor[:,tips]*time_factor[:,tips])
                objective=sum(priority_components)
            if surface is not None and interpolated_mode:
                # Interpolation must not extend the configured bend-to-tip lag.
                permitted = event_time[:, tips] <= self.elapsed[ends][None, :, None]+cfg.tip_lag_s+1e-12
                objective = objective.masked_fill(~permitted, 0)
            valid = valid & torch.isfinite(objective).flatten(1).all(-1)
            # A surface event stored at i may interpolate i -> i+1. Do not
            # credit it until that next observation exists, even for intervals
            # ultimately classified as non-crossings. No future argmax is used.
            ready = (tips+int(surface is not None and interpolated_mode)).clamp_max(len(self.time)-1)
            ready = ready.flatten()[None].expand(len(p), -1)
            available = prefix_valid.gather(1, ready)
            values_at_event = objective.flatten(1)
            values_at_event = torch.where(available, values_at_event, 0)
            event_scores.scatter_reduce_(1, ready, values_at_event, reduce='amax', include_self=True)
            if terminal_samples is not None:
                objective = torch.where(available.reshape_as(objective), objective, 0)
                if first_hit is not None:
                    # Terminal credit is measured at the actual first qualified
                    # hit, not at a different, better-looking approach event.
                    objective = objective.masked_fill(hit_exists[:, None, None]
                        & (tips[None] != first_hit[:, None, None]), 0)
            best, lag = objective.max(-1)
            chosen = tips[None].expand(len(p), -1, -1).gather(2, lag[..., None])[..., 0]
            take = lambda value: value.gather(2, lag[..., None])[..., 0]
            fields.append(dict(score=best, mechanical=take(mechanical), matching=matching.gather(1, chosen),
                center_factor=center_factor.gather(1, chosen),
                time_factor=time_factor.gather(1, chosen), bend=bend, pulse=take(pulse), coherence=coherence,
                travel_m=travel, turn_rad=minimum_turn, spread_m=maximum_spread, transport_mps=transport,
                nonrigid_mps=nonrigid.gather(1, chosen), release_mps=take(release), forward_mps=forward.gather(1, chosen),
                tip_speed_mps=tip_speed.gather(1, chosen), height_error_m=height_error.gather(1, chosen),
                speed_error_mps=speed_error.gather(1, chosen), direction_error_rad=angle.gather(1, chosen),
                release_time_s=event_time.gather(1, chosen), release_height_world_m=event_p[:, :, -1, 2].gather(1, chosen),
                start_index=begins[None].expand(len(p), -1), end_index=ends[None].expand(len(p), -1), tip_index=chosen))
            # Additive contributions at the SAME selected event; not independent
            # maxima. Their sum is the event score before effort/failure handling.
            weighted_mechanics = take(mechanical)*time_factor.gather(1, chosen)
            if match.center_weight > 0:
                weighted_mechanics = weighted_mechanics*center_factor.gather(1, chosen)
            fields[-1]['mechanics_credit'] = weighted_mechanics*match.shaping_weight*credit
            fields[-1]['approach_reward'] = weighted_mechanics*match.shaping_weight*(1-credit)*matching.gather(1, chosen)
            fields[-1]['strike_bonus'] = (weighted_mechanics*strike_weight
                *surface['qualified'].to(p.dtype).gather(1, chosen)
                if surface is not None else torch.zeros_like(weighted_mechanics))
            if priority_components is not None:
                for key,value in zip(('mechanics_credit','approach_reward','strike_bonus'),priority_components):
                    fields[-1][key]=take(value)
            if whip_mode:
                selected_whip = take(mechanical*unfolding*direction)
                fields[-1].update(mechanics_credit=selected_whip, approach_reward=torch.zeros_like(best),
                    strike_bonus=torch.zeros_like(best), unfolding_fraction=take(unfolding),
                    folded_chord_arc_ratio=folded, release_chord_arc_ratio=event_chord_ratio.gather(1, chosen),
                    forward_alignment=take(direction))
                if delivery_mode:
                    proximity=target_proximity.gather(1,chosen)
                    fields[-1].update(approach_reward=selected_whip*proximity,
                        matching=proximity,target_proximity=proximity,
                        wall_distance_m=target_distance.gather(1,chosen))
                if precision_mode:
                    quality=precision_quality.gather(1,chosen)
                    fields[-1].update(approach_reward=selected_whip*quality, matching=quality,
                        crossing_center_quality=quality,precision_sigma_m=torch.full_like(best,match.center_sigma_m))
            if surface is not None and interpolated_mode:
                for name, key in (('wall_distance_m', 'distance'), ('crossing_fraction', 'fraction'),
                                  ('plane_crossing', 'crossing'), ('directed_strike', 'qualified'),
                                  ('target_lateral_error_m', 'lateral'), ('target_vertical_error_m', 'vertical')):
                    fields[-1][name] = surface[key].to(p.dtype).gather(1, chosen)
        joined = {key: torch.cat([item[key] for item in fields], 1) for key in fields[0]}
        selected = joined['score'].argmax(1)[:, None]
        values = {key: value.gather(1, selected)[:, 0] for key, value in joined.items()}
        score = values.pop('score').masked_fill(~valid, float('nan'))
        has_event = valid & (score > 0)
        indices = torch.stack([values.pop(key) for key in ('start_index', 'end_index', 'tip_index')], -1)
        components = {key: value.masked_fill(~has_event, float('nan')) for key, value in values.items()}
        if surface is not None:
            # Whole-episode target metrics are independent of the selected
            # mechanics event. They do not label a crossing as a physical whip.
            success = surface['qualified'].any(-1)
            first = surface['elapsed'].masked_fill(~surface['qualified'], float('inf')).amin(-1)
            components.update(strike_success=success.to(p.dtype).masked_fill(~valid, float('nan')),
                              strike_count=surface['qualified'].sum(-1).to(p.dtype).masked_fill(~valid, float('nan')),
                              first_strike_time_s=first.masked_fill(~valid | ~success, float('nan')))
            if whip_mode:
                geometric = surface['geometric_hit'].any(-1)
                components.update(geometric_hit_success=geometric.to(p.dtype).masked_fill(~valid, float('nan')),
                    strict_strike_success=success.to(p.dtype).masked_fill(~valid, float('nan')),
                    first_geometric_hit_time_s=surface['elapsed'].masked_fill(~surface['geometric_hit'], float('inf'))
                        .amin(-1).masked_fill(~valid | ~geometric, float('nan')))
            if precision_mode:
                from .strike import crossing_precision_diagnostics
                components.update(crossing_precision_diagnostics(surface,valid))
        return Stage1Score(score, valid, has_event, indices.masked_fill(~has_event[:, None], -1),
                           components, event_scores.cummax(1).values)
