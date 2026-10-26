"""Ten-link horizontal-fold adapter with substep safety and causal event rewards.

Physics remains native. The first failing substep is latched; subsequent batched
work is discarded, never evaluated as valid continuation. No old reward is called.
"""
import json
import numpy as np
import torch
from simulation.tensor_model import ControlledState,controlled_values
from .sac_environment import Stage1Environment
from .horizontal_fold import HorizontalFoldAttempt
from .stage1 import Stage1Score


class HorizontalFoldMetadata:
    def __init__(self,rest,time,configuration,*,initial_position):
        self.center=initial_position.clone();self.direction=rest.new_tensor([1.,0.,0.]);self.speed=3.
    def __call__(self,*args,**kwargs):
        raise RuntimeError('Horizontal fold is scored from causal task state, not legacy full-horizon scoring')


class HorizontalFoldEnvironment(Stage1Environment):
    def __init__(self,*args,**kwargs):
        capture=kwargs.get('capture',True)
        # Keep task guards outside solver replay. Reuse only the existing native
        # iteration acceleration, with unchanged convergence masks and limits.
        kwargs['capture']=False
        super().__init__(*args,**kwargs)
        horizon=4 if self.configuration.matching.reward_mode=='continuous_directional_whip_four_second_v6' else 7
        if self.model.n!=11 or self.samples!=100*horizon+1 or self.decisions!=50*horizon:
            raise ValueError('Fold task requires ten links and its registered horizon')
        self.execution_backend='native_substep_guard_v1'
        if self.device=='cuda' and capture:
            self.model.compile_batched_physics()
            from .sac_environment import _conditional_solver_enabled
            if _conditional_solver_enabled(kwargs.get('conditional')):
                from .conditional_graph import install
                self.conditional_slots=install(self.model)
                self.execution_backend='cuda_conditional_substep_guard_v1'
            else:
                from .iteration_graph import IterationGraphs
                self.model._batch_implicit_iteration=IterationGraphs(self.model._batch_implicit_iteration)
                self.execution_backend='cuda_iteration_substep_guard_v1'

    def reset(self):
        self.attempts=None
        result=super().reset()
        rest=self.model.rest.detach().cpu().numpy()
        self.attempts=[HorizontalFoldAttempt(rest) for _ in range(self.batch)]
        self.ledger=torch.zeros((self.batch,self.decisions),dtype=torch.float64,device=self.device)
        self.terminal_times=torch.zeros(self.batch,dtype=torch.float64,device=self.device)
        self.failure_reasons=['']*self.batch
        self.substep_records=[]
        self._record_critic_state()
        return result

    def _task_critic_features(self):
        values=(np.stack([a.features() for a in self.attempts]) if self.attempts is not None
                else np.zeros((self.batch,16),np.float32))
        return torch.as_tensor(values,device=self.device)

    def _substep_observer(self,state,elapsed):
        p,v,q,cv=self._observe(state)
        finite=self._result_finite(state)
        inside=self.configuration.workspace.contains(p[:,None],self.origin.expand(self.batch,3))[:,0]
        safe=self._envelope(p,q,state.rotation)
        # The registered soft-workspace task measures violations but does not end on them.
        soft=getattr(self,'soft_workspace',False)
        if soft:self.workspace_breached |= self._guard_active & ~inside
        # Only the explicitly registered full-horizon task observes physical
        # envelope violations without stopping. Numerical corruption still stops.
        physical_termination=getattr(self,'physical_termination',True)
        if not physical_termination:
            observed=self._guard_active & finite
            self.physical_violation |= observed & ~safe
            height=q[:,:,2].amin(-1)
            tilt=torch.rad2deg(torch.acos(state.rotation[:,2,2].clamp(-1,1)))
            self.minimum_observed_cable_height=torch.where(observed,
                torch.minimum(self.minimum_observed_cable_height,height),self.minimum_observed_cable_height)
            self.maximum_observed_tilt=torch.where(observed,
                torch.maximum(self.maximum_observed_tilt,tilt),self.maximum_observed_tilt)
        envelope=safe if physical_termination else torch.ones_like(safe)
        new=self._guard_active & ~(finite & (torch.ones_like(inside) if soft else inside) & envelope)
        self._guard_time=torch.where(new,torch.full_like(self._guard_time,elapsed),self._guard_time)
        reason=torch.where(~finite,1,torch.where(~inside & (not soft),2,3))
        self._guard_reason=torch.where(new,reason,self._guard_reason)
        # For nonfinite results retain last finite state. The failed elapsed time
        # is still charged; the corrupt endpoint is never passed to geometry.
        keep=self._guard_active & finite
        self._last_finite=ControlledState(*(torch.where(keep.reshape(-1,*([1]*(x.ndim-1))),x,y)
            for x,y in zip(controlled_values(state),controlled_values(self._last_finite))))
        self._guard_active &= ~new

    @torch.no_grad()
    def step(self,action):
        if self.decision>=self.decisions or not bool(self.active.any()):raise RuntimeError('Cohort finished')
        if action.shape!=(self.batch,3) or not bool(torch.isfinite(action).all()) or bool((action.abs()>1.000001).any()):
            raise ValueError('Expected normalized finite acceleration actions')
        action=action.detach().clamp(-1,1);was_decision=self.active.clone()
        self.actions[:,self.decision]=torch.where(self.active[:,None],action,0.)
        self.previous_action=torch.where(self.active[:,None],action,self.previous_action)
        command=torch.cat((action.double()*self.action_limit,self.hold[3].expand(self.batch,1)),1)
        for _ in range(self.stride):
            was=self.active.clone();applied=self.queue[:,0].clone() if self.delay else command
            if self.delay:self.queue=torch.where(was[:,None,None],torch.cat((self.queue[:,1:],command[:,None]),1),self.queue)
            self._guard_active=was.clone();self._guard_time=torch.zeros(self.batch,device=self.device,dtype=torch.float64)
            self._guard_reason=torch.zeros(self.batch,device=self.device,dtype=torch.long)
            self._last_finite=self.state
            self.model.step_unchecked(self.state,applied,self.controller,batch=True,observer=self._substep_observer)
            self.state=self._last_finite
            p,v,q,cv=self._observe(self.state)
            self.index+=1
            failed=was & ~self._guard_active
            self.active=self._guard_active
            self.failed |= failed;self.workspace_breached |= self._guard_reason==2
            violation=(~self.configuration.workspace.contains(p[:,None],self.origin.expand(self.batch,3))[:,0]
                       if getattr(self,'soft_workspace',False) else self._guard_reason==2)
            self._record(p,v,q,cv,command,applied,was & (self._guard_reason!=1),violation)
            # One bounded device transfer per100Hz observation, not per physics substep.
            pp=q.cpu().numpy();vv=cv.cpu().numpy();active=was.cpu().numpy()
            tracking=p.cpu().numpy() if getattr(self,'soft_workspace',False) else None
            times=self._guard_time.cpu().numpy();reasons=self._guard_reason.cpu().numpy()
            for lane in np.flatnonzero(active):
                a=self.attempts[lane]
                if reasons[lane]:
                    reason={1:'numerical_failure',2:'workspace_failure',3:'physical_failure'}[int(reasons[lane])]
                    elapsed=(self.index-1)*.01+float(times[lane])
                    reward=a.fail(elapsed,reason);self.failure_reasons[lane]=reason
                    self.substep_records.append(dict(lane=int(lane),time_s=elapsed,reason=reason,
                        drone_position_m=p[lane].cpu().tolist()))
                elif tracking is not None:reward=a.observe(self.index*.01,pp[lane],vv[lane],drone_position=tracking[lane])
                else:reward=a.observe(self.index*.01,pp[lane],vv[lane])
                self.ledger[lane,self.decision]+=reward
                if a.terminal is not None:
                    self.active[lane]=False;self.terminal_times[lane]=a.time
                    self.terminal_hit[lane]=a.qualified
            self.executed_samples+=was.long()
            if not bool(self.active.any()):break
        self.lengths+=was_decision.long();self.decision+=1;self._record_critic_state()
        return self.frames,self.indices(),not bool(self.active.any())

    @torch.no_grad()
    def finish(self):
        if any(a.terminal is None for a in self.attempts):raise RuntimeError('Interrupted cohort cannot be scored')
        dtype=torch.float64
        tensor=lambda x:torch.tensor(x,device=self.device,dtype=dtype)
        returns=tensor([a.total for a in self.attempts]);valid=~self.failed
        event=torch.tensor([a.terminal=='event_end' for a in self.attempts],device=self.device)
        H=tensor([a.selected['H'] if a.selected is not None else 0. for a in self.attempts])
        components=dict(horizontal_reach_fraction=H,qualified_outreach=tensor([a.qualified for a in self.attempts]),
            fold_formed=tensor([a.formed for a in self.attempts]),fold_propagated=tensor([a.propagated for a in self.attempts]),
            fold_opened=tensor([a.opened for a in self.attempts]),event_completed=event.double(),
            timeout=tensor([a.terminal=='timeout' for a in self.attempts]),executed_duration_s=self.terminal_times,
            time_cost=.1*self.terminal_times,workspace_cost=torch.zeros_like(returns),
            net_episode_return=returns,event_credit=torch.where(event,5*H.square(),0.))
        for name in ('world_outward_mps','relative_outward_mps','world_cosine','relative_cosine'):
            components[name]=tensor([a.selected[name] if a.selected is not None else 0. for a in self.attempts])
        indices=torch.tensor([round(a.selected['time_s']*100) if a.selected else -1 for a in self.attempts],device=self.device)
        score=Stage1Score(components['event_credit'],valid,event,indices,components,
            torch.zeros((self.batch,self.samples),device=self.device,dtype=dtype))
        if not torch.allclose(self.ledger.sum(1),returns,atol=1e-12,rtol=1e-12):raise RuntimeError('Fold ledger does not telescope')
        self.transition_rewards=self.ledger.float()
        return returns.float(),valid,score,torch.zeros_like(returns)

    def workspace_metrics(self):
        values=super().workspace_metrics()
        values['boundary_breached']=self.workspace_breached.clone()
        values['executed_duration_s']=self.terminal_times.clone()
        return values

    def event_records(self):
        return [dict(terminal=a.terminal,terminal_time_s=a.time,selected=a.selected,events=a.log,
                     qualified=a.qualified,episode_return=a.total) for a in self.attempts]

    def replay_arrays(self,lane=0):
        result=super().replay_arrays(lane)
        if result is not None:
            if self.failed[lane] and self.failure_reasons[lane]!='numerical_failure':
                result['time_s'][-1]=float(self.terminal_times[lane])
            result['fold_event_json']=np.array(json.dumps(self.event_records()[lane]))
            result['terminal_time_s']=np.array(float(self.terminal_times[lane]))
            result['task_valid']=np.array(not bool(self.failed[lane]))
        return result

    def policy_action(self,callback):
        width=self.settings.get('evaluation_environments',128)
        if self.batch!=1:raise ValueError('Deterministic fixed-state evaluation uses one lane')
        return callback(self.frames.repeat(width,1,1),self.indices().repeat(width))[:1]

    def provenance(self):
        return dict(evaluation_backend='native_tensor_substep_guard_v1',evaluation_environments=self.batch,
                    evaluation_policy_lanes=self.settings.get('evaluation_environments',128),unique_initial_states=1)
