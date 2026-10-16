"""Versioned Stage4 task scoring; never a controller handoff or state reset.

The event-relative pilot freezes original precision mechanics at the causally
observed first hit (or miss timeout), then scores four seconds of recovery.
The original unlaunched fixed-window draft remains separately testable.
"""
from dataclasses import replace
import torch

from .configuration import EpisodeTiming
from .stage1 import Stage1Scorer, Stage1Score


def make_scorer(rest, time, configuration, *, initial_position):
    if configuration.recovery is None:
        return Stage1Scorer(rest,time,configuration,initial_position=initial_position)
    cls = (EventRecoveryScorer if configuration.recovery.timing_mode == 'after_first_geometric_hit_v1'
           else LearnedRecoveryScorer)
    return cls(rest,time,configuration,initial_position=initial_position)


class LearnedRecoveryScorer:
    def __init__(self, rest, time, configuration, *, initial_position):
        self.configuration, self.time = configuration,time
        r = configuration.recovery
        self.whip_end = round(r.whip_duration_s/.01)
        self.hold_begin = len(time)-1-round(r.final_hold_s/.01)
        if len(time)-1 != self.whip_end+round(r.recovery_duration_s/.01):
            raise ValueError('Recovery scorer requires the complete declared 100Hz horizon')
        # Original Stage3 duration4+post_hold.5 and event search are unchanged.
        original = replace(configuration,recovery=None,
            episode=EpisodeTiming(r.whip_duration_s-.5,.5))
        self.base = Stage1Scorer(rest,time[:self.whip_end+1],original,initial_position=initial_position)
        self.origin = torch.as_tensor(initial_position,dtype=rest.dtype,device=rest.device).detach().clone()

    def __getattr__(self, name):
        return getattr(self.base,name)

    @torch.no_grad()
    def __call__(self, positions, velocities, state_valid, *, drone_positions,
                 drone_velocities, terminal_samples=None):
        if terminal_samples is not None:
            raise ValueError('Learned recovery never uses first-hit termination')
        r, end = self.configuration.recovery,self.whip_end
        if drone_positions.shape != (*positions.shape[:2],3) or drone_velocities.shape != drone_positions.shape:
            raise ValueError('Recovery requires complete drone position/velocity histories')
        if positions.shape[1] != len(self.time) or state_valid.shape != positions.shape[:2]:
            raise ValueError('Recovery history horizon mismatch')
        base = self.base(positions[:,:end+1],velocities[:,:end+1],state_valid[:,:end+1],
            drone_positions=drone_positions[:,:end+1])
        finite = (state_valid & torch.isfinite(positions).flatten(2).all(-1)
            & torch.isfinite(velocities).flatten(2).all(-1)
            & torch.isfinite(drone_positions).all(-1) & torch.isfinite(drone_velocities).all(-1))
        prefix = finite.long().cumprod(1).bool()
        valid = base.valid & finite.all(-1)
        error = (drone_positions-self.origin).norm(dim=-1)
        speed = drone_velocities.norm(dim=-1)
        q = torch.exp(-.5*((error/r.position_scale_m).square()+(speed/r.velocity_scale_mps).square()))
        q = torch.where(prefix,torch.nan_to_num(q,nan=0.),0.)
        # A is the W*C actually earned at the selected Stage3 event, NOT an
        # unrelated maximum W times unrelated best accuracy or binary hit.
        gate = torch.nan_to_num(base.components['approach_reward'],nan=0.).clamp(0,1)
        weights = torch.zeros_like(q)
        weights[:,end+1:] = .5/(len(self.time)-1-end)
        weights[:,self.hold_begin+1:] += .5/(len(self.time)-1-self.hold_begin)
        quality_prefix = (q*weights).cumsum(1)
        recovery_prefix = r.weight*gate[:,None]*quality_prefix
        whip_prefix = torch.cat((base.causal_best,
            base.causal_best[:,-1:].expand(-1,len(self.time)-1-end)),1)
        causal = whip_prefix+recovery_prefix
        positive = base.score+recovery_prefix[:,-1]
        held = ((error[:,self.hold_begin+1:] <= r.position_tolerance_m)
            & (speed[:,self.hold_begin+1:] <= r.speed_tolerance_mps)
            & prefix[:,self.hold_begin+1:]).all(-1) & valid
        components = dict(base.components)
        components.update(whip_event_score=base.score,whip_score_valid=base.valid.to(q.dtype),
            recovery_gate=gate,recovery_quality=quality_prefix[:,-1],
            recovery_window_quality=q[:,end+1:].mean(1),
            recovery_final_hold_quality=q[:,self.hold_begin+1:].mean(1),
            recovery_credit=recovery_prefix[:,-1],
            recovery_success=held.to(q.dtype),
            whip_and_recovery_success=(held & (base.components['geometric_hit_success']==1)).to(q.dtype),
            final_position_error_m=error[:,-1],final_speed_mps=speed[:,-1],
            hold_max_position_error_m=error[:,self.hold_begin+1:].amax(1),
            hold_max_speed_mps=speed[:,self.hold_begin+1:].amax(1),
            hold_max_cable_speed_mps=velocities[:,self.hold_begin+1:].norm(dim=-1).flatten(1).amax(1))
        return Stage1Score(positive.masked_fill(~valid,float('nan')),valid,base.has_event,
            base.event_indices,components,causal)


class EventRecoveryScorer(LearnedRecoveryScorer):
    @torch.no_grad()
    def __call__(self, positions, velocities, state_valid, *, drone_positions,
                 drone_velocities, recovery_start_samples, recovery_hit,
                 terminal_samples=None):
        r = self.configuration.recovery
        b,t = state_valid.shape
        if positions.shape[1] != len(self.time) or drone_positions.shape != (b,t,3) or drone_velocities.shape != (b,t,3):
            raise ValueError('Event recovery needs complete padded histories with explicit prefix masks')
        start = recovery_start_samples.clamp_min(0)
        if start.shape != (b,) or start.dtype != torch.long or bool((start>self.whip_end).any()):
            raise ValueError('Recovery starts must be causal observation endpoints within whip deadline')
        if terminal_samples is None:
            terminal_samples = state_valid.long().sum(1)
        grid = torch.arange(t,device=positions.device)[None]
        first_end = torch.minimum(start,terminal_samples-1).clamp_min(0)
        whip_valid = state_valid[:,:self.whip_end+1] & (grid[:,:self.whip_end+1]<=first_end[:,None])
        base = self.base(positions[:,:self.whip_end+1],velocities[:,:self.whip_end+1],
                         whip_valid,terminal_samples=first_end+1,
                         drone_positions=drone_positions[:,:self.whip_end+1])
        finite = (state_valid & torch.isfinite(positions).flatten(2).all(-1)
            & torch.isfinite(velocities).flatten(2).all(-1)
            & torch.isfinite(drone_positions).all(-1) & torch.isfinite(drone_velocities).all(-1))
        prefix = finite.long().cumprod(1).bool()
        recovery_count = round(r.recovery_duration_s/.01)
        hold_count = round(r.final_hold_s/.01)
        end = start+recovery_count
        valid = base.valid & prefix.gather(1,end[:,None])[:,0] & (terminal_samples==end+1)
        error = (drone_positions-self.origin).norm(dim=-1)
        speed = drone_velocities.norm(dim=-1)
        norm = (error/r.position_scale_m).square()+(speed/r.velocity_scale_mps).square()
        hover = torch.exp(-.5*norm)
        returning = 1/(1+norm) if r.quality_mode=='reciprocal_return_gaussian_hold_v1' else hover
        returning = torch.where(prefix,torch.nan_to_num(returning,nan=0.),0.)
        hover = torch.where(prefix,torch.nan_to_num(hover,nan=0.),0.)
        settling = r.quality_mode=='return_and_cable_settle_v1'
        if settling:
            # Common drone translation is removed. Horizontal shape prevents a
            # wide swing at its zero-speed turning point looking settled.
            relative_velocity = velocities[:,:,1:]-velocities[:,:,:1]
            relative_position = positions[:,:,1:]-positions[:,:,:1]
            cable_speed = relative_velocity.square().sum(-1).mean(-1).sqrt()
            cable_horizontal = relative_position[...,:2].square().sum(-1).mean(-1).sqrt()
            cable_quality = 1/(1+(cable_speed/r.cable_velocity_scale_mps).square()
                               +(cable_horizontal/r.cable_horizontal_scale_m).square())
            cable_quality = torch.where(prefix,torch.nan_to_num(cable_quality,nan=0.),0.)
            return_quality = 1/(1+(error/r.return_position_scale_m).square()
                                 +(speed/r.return_velocity_scale_mps).square())
            return_quality = torch.where(prefix,torch.nan_to_num(return_quality,nan=0.),0.)
            returning = .5*return_quality+.5*cable_quality
            hover = hover*cable_quality
        window = (grid>start[:,None]) & (grid<=end[:,None])
        hold = (grid>end[:,None]-hold_count) & (grid<=end[:,None])
        quality_steps = .5*returning*window/recovery_count+.5*hover*hold/hold_count
        quality = quality_steps.cumsum(1)
        gate = torch.nan_to_num(base.components['approach_reward'],nan=0.).clamp(0,1)
        if settling:
            # Recovery is not attenuated again by frozen W*C. A miss invokes
            # the return backbone but earns no recovery credit.
            gate = recovery_hit.to(error.dtype)
        credit = r.weight*gate[:,None]*quality
        base_index = torch.minimum(grid.expand(b,-1),start[:,None]).clamp_max(self.whip_end)
        causal = base.causal_best.gather(1,base_index)+credit
        held = (~hold | ((error<=r.position_tolerance_m) & (speed<=r.speed_tolerance_mps) & prefix)).all(1) & valid
        stable = ((error<=r.position_tolerance_m) & (speed<=r.speed_tolerance_mps) & prefix)
        completed = stable.unfold(1,hold_count,1).all(-1)
        completed_grid = torch.arange(hold_count-1,t,device=positions.device)[None]
        completed &= (completed_grid>=start[:,None]+hold_count) & (completed_grid<=end[:,None])
        first_hold = torch.where(completed,completed_grid.expand(b,-1),t+1).amin(1)
        first_hold_time = ((first_hold-start).to(error.dtype)*.01).masked_fill(first_hold>t,float('nan'))
        take = lambda v:v.gather(1,end[:,None])[:,0]
        maximum = lambda v:torch.where(hold & prefix,v,0.).amax(1)
        components = dict(base.components)
        if settling:
            quiet = (~hold | ((cable_speed<=r.cable_speed_tolerance_mps)
                & (cable_horizontal<=r.cable_horizontal_tolerance_m) & prefix)).all(1) & valid
            components.update(cable_settling_success=quiet.to(error.dtype),
                whip_hover_and_cable_success=(quiet & held & recovery_hit).to(error.dtype),
                hold_max_cable_relative_rms_speed_mps=maximum(cable_speed),
                hold_max_cable_horizontal_rms_m=maximum(cable_horizontal),
                cable_window_quality=(cable_quality*window).sum(1)/recovery_count,
                drone_return_window_quality=(return_quality*window).sum(1)/recovery_count)
        components.update(whip_event_score=base.score,whip_score_valid=base.valid.to(error.dtype),
            recovery_gate=gate,recovery_quality=quality[:,-1],
            recovery_window_quality=(returning*window).sum(1)/recovery_count,
            recovery_final_hold_quality=(hover*hold).sum(1)/hold_count,
            recovery_credit=credit[:,-1],recovery_success=held.to(error.dtype),
            whip_and_recovery_success=(held & recovery_hit).to(error.dtype),
            recovery_trigger_hit=recovery_hit.to(error.dtype),
            recovery_first_hold_completed_s=first_hold_time,
            recovery_start_s=start.to(error.dtype)*.01,recovery_end_s=end.to(error.dtype)*.01,
            final_position_error_m=take(error),final_speed_mps=take(speed),
            hold_max_position_error_m=maximum(error),hold_max_speed_mps=maximum(speed),
            hold_max_cable_speed_mps=maximum(velocities.norm(dim=-1).amax(-1)),
            executed_duration_s=(terminal_samples-1).to(error.dtype)*.01)
        return Stage1Score((base.score+credit[:,-1]).masked_fill(~valid,float('nan')),
                          valid,base.has_event,base.event_indices,components,causal)
