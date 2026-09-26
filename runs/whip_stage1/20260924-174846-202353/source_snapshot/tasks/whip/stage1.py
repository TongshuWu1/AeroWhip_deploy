"""Isolated Stage-1 shaping objective; not a whip classifier or training runner."""

from dataclasses import dataclass
import math

import torch

from .stage1_configuration import Stage1Configuration


def _sat(value, scale):
    value = value.clamp_min(0)
    return value/(value+scale)


@dataclass(frozen=True)
class Stage1Score:
    score: torch.Tensor
    valid: torch.Tensor
    has_event: torch.Tensor
    event_indices: torch.Tensor
    components: dict[str, torch.Tensor]


@dataclass(frozen=True)
class Stage1Return:
    returns: torch.Tensor
    valid: torch.Tensor
    numerical_valid: torch.Tensor
    envelope_valid: torch.Tensor
    acceleration_cost: torch.Tensor
    score: Stage1Score


class Stage1Scorer:
    """Score fixed-request batches on a cached 100 Hz mesh/time grid.

    max_e M(e) * [A(e) + strike_weight*I_strike(e)] * (1-beta*t_e/T).
    A is smooth request matching. The optional directed-wall bonus is earned
    only at a qualified crossing, with mechanics measured at that SAME event.
    Maximum selection does not accumulate credit for repeated crossings.
    Missing/zero strike weight preserves the original matching-only objective.
    M is the existing experimental transport/release/forward proxy, including
    its cubic root and known limitations. Every term uses the SAME event.
    The Gaussian is outside the cubic root. Direction is requested tip velocity,
    not drone yaw. Timing is a preference, not an exact timestamp request.

    Positions/velocities are (B,T,N,3), state_valid is bool (B,T). Any invalid
    sample invalidates the whole episode, including failures after the event.
    A valid zero-score episode has has_event=False and indices=-1, not failure.
    Components are NaN when no positive event exists. No physical qualification
    or success threshold is implied by has_event or a positive score.

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
        self.horizon = configuration.trajectory.duration_s+configuration.trajectory.post_hold_s
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

    @torch.no_grad()
    def __call__(self, positions, velocities, state_valid):
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
        valid = (state_valid.all(-1) & torch.isfinite(positions).flatten(1).all(-1)
                 & torch.isfinite(velocities).flatten(1).all(-1)
                 & (torch.isfinite(lengths) & (lengths > 0)).flatten(1).all(-1))
        # Sanitize invalid lanes BEFORE linear algebra; never promote them to valid.
        p = torch.where(valid[:, None, None, None], positions, self.fallback)
        v = torch.where(valid[:, None, None, None], velocities, 0)
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
        surface = None
        if self.surface is not None:
            from .strike import surface_events
            surface = surface_events(p, v, self.time, self.center, self.direction, self.surface, self.speed)
            event_p, event_v, event_time = surface['positions'], surface['velocities'], surface['elapsed']
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
        cfg, match = self.configuration.mechanics, self.configuration.matching
        height_error, speed_error = event_p[:, :, -1, 2]-self.height, tip_speed-self.speed
        # Preserve signed speed error in diagnostics; minimum mode penalizes
        # only shortfall, and still chooses ALL terms at the same event.
        matching_speed_error = speed_error.clamp_max(0) if match.speed_mode == 'minimum' else speed_error
        matching = torch.exp(-.5*((height_error/match.height_width_m).square()
            +(matching_speed_error/match.speed_width_mps).square()+(angle/math.radians(match.direction_width_deg)).square()))
        matching = matching.masked_fill(tip_speed <= 0, 0)
        if surface is not None:
            # Smooth approach shaping is uniform inside the finite rectangle;
            # no center-point accuracy is requested.
            speed_error = forward-self.speed
            angle = surface['angle']
            matching = torch.exp(-.5*((surface['distance']/self.surface.approach_width_m).square()
                +(speed_error.clamp_max(0)/match.speed_width_mps).square()
                +(angle/math.radians(match.direction_width_deg)).square()))
            matching = matching.masked_fill(tip_speed <= 0, 0)
        strike_weight = self.surface.strike_bonus_weight if surface is not None else 0.0
        reward_matching = matching
        if strike_weight > 0:
            reward_matching = matching+strike_weight*surface['qualified'].to(p.dtype)
        time_factor = 1-match.time_preference*event_time/self.horizon
        for value in (turn, mu, spread, prefix, nonrigid, relative_speed, tip_speed, matching):
            valid = valid & torch.isfinite(value).flatten(1).all(-1)
        fields = []
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
            if surface is not None:
                # Interpolation must not extend the configured bend-to-tip lag.
                permitted = event_time[:, tips] <= self.elapsed[ends][None, :, None]+cfg.tip_lag_s+1e-12
                objective = objective.masked_fill(~permitted, 0)
            valid = valid & torch.isfinite(objective).flatten(1).all(-1)
            best, lag = objective.max(-1)
            chosen = tips[None].expand(len(p), -1, -1).gather(2, lag[..., None])[..., 0]
            take = lambda value: value.gather(2, lag[..., None])[..., 0]
            fields.append(dict(score=best, mechanical=take(mechanical), matching=matching.gather(1, chosen),
                time_factor=time_factor.gather(1, chosen), bend=bend, pulse=take(pulse), coherence=coherence,
                travel_m=travel, turn_rad=minimum_turn, spread_m=maximum_spread, transport_mps=transport,
                nonrigid_mps=nonrigid.gather(1, chosen), release_mps=take(release), forward_mps=forward.gather(1, chosen),
                tip_speed_mps=tip_speed.gather(1, chosen), height_error_m=height_error.gather(1, chosen),
                speed_error_mps=speed_error.gather(1, chosen), direction_error_rad=angle.gather(1, chosen),
                release_time_s=event_time.gather(1, chosen), release_height_world_m=event_p[:, :, -1, 2].gather(1, chosen),
                start_index=begins[None].expand(len(p), -1), end_index=ends[None].expand(len(p), -1), tip_index=chosen))
            if surface is not None:
                for name, key in (('wall_distance_m', 'distance'), ('crossing_fraction', 'fraction'),
                                  ('plane_crossing', 'crossing'), ('directed_strike', 'qualified')):
                    fields[-1][name] = surface[key].to(p.dtype).gather(1, chosen)
                if strike_weight > 0:
                    # Components at the selected event, not independent maxima.
                    weighted_mechanics = take(mechanical)*time_factor.gather(1, chosen)
                    fields[-1]['approach_reward'] = weighted_mechanics*matching.gather(1, chosen)
                    fields[-1]['strike_bonus'] = (weighted_mechanics*strike_weight
                        *surface['qualified'].to(p.dtype).gather(1, chosen))
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
            first = event_time.masked_fill(~surface['qualified'], float('inf')).amin(-1)
            components.update(strike_success=success.to(p.dtype).masked_fill(~valid, float('nan')),
                              strike_count=surface['qualified'].sum(-1).to(p.dtype).masked_fill(~valid, float('nan')),
                              first_strike_time_s=first.masked_fill(~valid | ~success, float('nan')))
        return Stage1Score(score, valid, has_event, indices.masked_fill(~has_event[:, None], -1), components)


class Stage1Reward:
    """Full-episode envelope/cost wrapper; does not invoke physics or training.

    Pass a BatchedRolloutResult, aligned full-state commands (B,T,11), and the
    caller's bool (B,) command/action admissibility. No target context is needed.
    Rebuild for a new configuration, mesh or time grid. Existing trajectory
    decoding must still enforce continuous command/control-point constraints.
    """

    def __init__(self, rest_lengths, time_s, configuration: Stage1Configuration, *, initial_position=None):
        self.scorer = Stage1Scorer(rest_lengths, time_s, configuration, initial_position=initial_position)
        self.configuration = configuration
        self.intervals = round(configuration.trajectory.duration_s/.01)
        self.acceleration_scale = configuration.returns.acceleration_scale(configuration.trajectory)
        self.lower = rest_lengths.new_tensor(configuration.returns.drone_lower_world_m)
        self.upper = rest_lengths.new_tensor(configuration.returns.drone_upper_world_m)
        self.tilt_cosine = math.cos(math.radians(configuration.returns.maximum_tilt_deg))

    @torch.no_grad()
    def __call__(self, rollout, commands, *, admissible):
        p = rollout.cable_positions
        batch, samples = p.shape[:2]
        if (not isinstance(commands, torch.Tensor) or commands.shape != (batch, samples, 11)
                or commands.device != p.device or commands.dtype != p.dtype):
            raise ValueError('Commands must be aligned (batch, state samples, 11) on the history dtype/device')
        if (not isinstance(admissible, torch.Tensor) or admissible.shape != (batch,)
                or admissible.dtype != torch.bool or admissible.device != p.device):
            raise ValueError('Explicit command/action admissibility must be bool (batch,) on the history device')
        if rollout.time_s.shape != self.scorer.time.shape or rollout.cable_rest_lengths.shape != self.scorer.rest.shape:
            raise ValueError('Rollout mesh/time grid must match the cached objective')
        score = self.scorer(p, rollout.cable_velocities, rollout.state_valid)
        numerical = (rollout.valid & admissible & score.valid
            & (rollout.time_s == self.scorer.time).all() & (rollout.cable_rest_lengths == self.scorer.rest).all())
        cfg = self.configuration
        drone, rotations = rollout.drone_positions, rollout.drone_rotations
        envelope = (((drone >= self.lower) & (drone <= self.upper)).flatten(1).all(-1)
            & torch.isfinite(rotations).flatten(1).all(-1) & (rotations[:, :, 2, 2] >= self.tilt_cosine).all(-1)
            & (p[..., 2] >= cfg.returns.minimum_cable_height_m).flatten(1).all(-1)
            & cfg.workspace.contains(drone, drone[:, 0]).all(-1))
        cost = cfg.returns.acceleration_cost_weight*(commands[:, :self.intervals, 6:9]
            /self.acceleration_scale).square().sum(-1).mean(-1)
        valid = numerical & envelope & torch.isfinite(commands).flatten(1).all(-1) & torch.isfinite(cost)
        returns = torch.where(valid, score.score-cost, cfg.returns.failure_return)
        return Stage1Return(returns, valid, numerical, envelope, cost, score)
