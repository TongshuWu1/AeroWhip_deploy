"""Continuous reward adapter over unchanged native substep physics and guards."""
import numpy as np
import torch
from .horizontal_fold_environment import HorizontalFoldEnvironment, HorizontalFoldMetadata
from .sac_environment import Stage1Environment
from .stage1 import Stage1Score
from .continuous_whip import ContinuousWhipAttempt, FEATURE_SIZE, TRACE_FIELDS, SOFT_MODE, PLAIN_MODE, FULL_MODE, DIRECTED_MODE, SHORT_MODE, SOFT_TRACE_FIELDS

ContinuousWhipMetadata=HorizontalFoldMetadata


class ContinuousWhipEnvironment(HorizontalFoldEnvironment):
    def __init__(self,settings,snapshot,configuration,scene,**kwargs):
        self.soft_workspace=configuration.matching.reward_mode in (SOFT_MODE,PLAIN_MODE,FULL_MODE,DIRECTED_MODE,SHORT_MODE)
        self.plain_net=configuration.matching.reward_mode in (PLAIN_MODE,FULL_MODE,DIRECTED_MODE,SHORT_MODE)
        self.directed_x=configuration.matching.reward_mode in (DIRECTED_MODE,SHORT_MODE)
        self.horizon_s=configuration.episode.duration_s
        self.physical_termination=configuration.matching.reward_mode not in (FULL_MODE,DIRECTED_MODE,SHORT_MODE)
        super().__init__(settings,snapshot,configuration,scene,**kwargs)

    @torch.no_grad()
    def reset(self):
        self.attempts=None;result=Stage1Environment.reset(self)
        rest=self.model.rest.detach().cpu().numpy()
        self.attempts=[ContinuousWhipAttempt(rest,workspace_origin=self.origin.cpu().numpy() if self.soft_workspace else None,plain_net=self.plain_net,directed_x=self.directed_x,horizon_s=self.horizon_s) for _ in range(self.batch)]
        _,_,p,v=self._observe(self.state);p=p.cpu().numpy();v=v.cpu().numpy()
        for lane,a in enumerate(self.attempts):a.initialize(p[lane],v[lane])
        if not self.physical_termination:
            self.minimum_observed_cable_height=torch.as_tensor(p[:,:,2].min(axis=1),device=self.device).clone()
            self.maximum_observed_tilt=torch.rad2deg(torch.acos(self.state.rotation[:,2,2].clamp(-1,1)))
            self.physical_violation=((self.minimum_observed_cable_height<self.configuration.returns.minimum_cable_height_m)
                | (self.maximum_observed_tilt>self.configuration.returns.maximum_tilt_deg))
        self.ledger=torch.zeros((self.batch,self.decisions),dtype=torch.float64,device=self.device)
        self.terminal_times=torch.zeros(self.batch,dtype=torch.float64,device=self.device)
        self.failure_reasons=['']*self.batch;self.substep_records=[];self._record_critic_state()
        return result

    def _task_critic_features(self):
        values=(np.stack([a.features() for a in self.attempts]) if self.attempts is not None
                else np.zeros((self.batch,FEATURE_SIZE+2*self.soft_workspace),np.float32))
        return torch.as_tensor(values,device=self.device)

    @torch.no_grad()
    def finish(self):
        if any(a.terminal is None for a in self.attempts):raise RuntimeError('Interrupted cohort cannot be scored')
        tensor=lambda values:torch.tensor(values,device=self.device,dtype=torch.float64)
        returns=tensor([a.total for a in self.attempts]);valid=~self.failed
        if not self.physical_termination:valid &= ~self.physical_violation
        components=dict(continuous_whip_credit=tensor([a.accrued for a in self.attempts]),
            peak_reward_rate=tensor([a.peak_rate for a in self.attempts]),
            fold_support_integral_s=tensor([a.support_time for a in self.attempts]),
            positive_output_duration_s=tensor([a.output_time for a in self.attempts]),
            horizontal_reach_fraction=tensor([a.selected['horizontal_reach_fraction'] if a.selected else 0. for a in self.attempts]),
            completed_horizon=tensor([a.terminal=='horizon' for a in self.attempts]),
            failure=self.failed.double(),executed_duration_s=self.terminal_times,
            failure_adjustment=tensor([a.total-a.prefix_return() if a.terminal!='horizon' else 0. for a in self.attempts]),
            failure_cost=tensor([a.failure_cost for a in self.attempts]),
            return_transform_adjustment=tensor([a.prefix_return()-(a.accrued-a.cost) for a in self.attempts]),
            net_episode_return=returns,time_cost=torch.zeros_like(returns),workspace_cost=tensor([a.cost for a in self.attempts]))
        for key in ('world_outward_mps','relative_outward_mps','straightening_mps','alignment','fold_support'):
            components[key]=tensor([a.selected[key] if a.selected else 0. for a in self.attempts])
        if not self.physical_termination:
            components.update(physical_violation=self.physical_violation.double(),
                numerically_valid=(~self.failed).double(),
                minimum_cable_height_m=self.minimum_observed_cable_height,
                maximum_tilt_deg=self.maximum_observed_tilt)
        score=Stage1Score(components['continuous_whip_credit'],valid,torch.zeros_like(valid),
            torch.tensor([round(a.selected['time_s']*100) if a.selected else -1 for a in self.attempts],device=self.device),
            components,torch.zeros((self.batch,self.samples),device=self.device,dtype=torch.float64))
        if not torch.allclose(self.ledger.sum(1),returns,atol=1e-12,rtol=1e-12):
            raise RuntimeError('Continuous-whip ledger does not telescope')
        self.transition_rewards=self.ledger.float()
        return returns.float(),valid,score,components['workspace_cost']

    def event_records(self):
        records=[a.record() for a in self.attempts]
        if not self.physical_termination:
            for lane,row in enumerate(records):
                row.update(contract=self.configuration.matching.reward_mode,physical_termination=False,
                    physical_violation=bool(self.physical_violation[lane]),
                    numerically_valid=not bool(self.failed[lane]),
                    minimum_cable_height_m=float(self.minimum_observed_cable_height[lane]),
                    maximum_tilt_deg=float(self.maximum_observed_tilt[lane]))
        return records

    def replay_arrays(self,lane=0):
        result=super().replay_arrays(lane)
        if result is not None:
            if not self.physical_termination:
                result['physical_violation']=np.array(bool(self.physical_violation[lane]))
                result['numerically_valid']=np.array(not bool(self.failed[lane]))
                result['task_valid']=np.array(not bool(self.failed[lane] or self.physical_violation[lane]))
            # A failing substep can have a finite saved position but no valid
            # reward measurement. Represent that row as missing, not zero credit.
            fields=SOFT_TRACE_FIELDS if self.soft_workspace else TRACE_FIELDS
            values=np.full((len(result['time_s']),len(fields)),np.nan)
            trace=np.asarray(self.attempts[lane].trace);n=min(len(values),len(trace));values[:n]=trace[:n]
            result['continuous_whip_diagnostics']=values
            result['continuous_whip_fields']=np.array(fields)
        return result

    def provenance(self):
        return dict(evaluation_backend='native_tensor_substep_guard_v1',reward_mode=self.configuration.matching.reward_mode,
            evaluation_environments=self.batch,evaluation_policy_lanes=self.settings.get('evaluation_environments',128),
            unique_initial_states=1)
