"""Retained-progress Stage2 adapter over unchanged ten-link native physics.

Reuse the reviewed substep guards and per-lane freezing. Only the task ledger,
event metadata, and critic-only state differ from horizontal-fold Stage1.
"""
import numpy as np
import torch

from .horizontal_fold_environment import HorizontalFoldEnvironment
from .retained_fold_delivery import RetainedFoldDeliveryAttempt, FEATURE_SIZE
from .sac_environment import Stage1Environment
from .stage1 import Stage1Score


class RetainedFoldDeliveryMetadata:
    def __init__(self, rest, time, configuration, *, initial_position):
        self.center=rest.new_tensor([1.1,0.,2.])
        self.direction=rest.new_tensor([1.,0.,0.])
        self.speed=1.

    def __call__(self,*args,**kwargs):
        raise RuntimeError('Retained fold delivery is scored from its causal event ledger')


class RetainedFoldDeliveryEnvironment(HorizontalFoldEnvironment):
    @torch.no_grad()
    def reset(self):
        self.attempts=None
        result=Stage1Environment.reset(self)
        rest=self.model.rest.detach().cpu().numpy()
        self.attempts=[RetainedFoldDeliveryAttempt(rest) for _ in range(self.batch)]
        _,_,positions,velocities=self._observe(self.state)
        pp=positions.cpu().numpy();vv=velocities.cpu().numpy()
        for lane,a in enumerate(self.attempts):a.initialize(pp[lane],vv[lane])
        self.ledger=torch.zeros((self.batch,self.decisions),dtype=torch.float64,device=self.device)
        self.terminal_times=torch.zeros(self.batch,dtype=torch.float64,device=self.device)
        self.failure_reasons=['']*self.batch
        self.substep_records=[]
        self._record_critic_state()
        return result

    def _task_critic_features(self):
        values=(np.stack([a.features() for a in self.attempts]) if self.attempts is not None
                else np.zeros((self.batch,FEATURE_SIZE),np.float32))
        return torch.as_tensor(values,device=self.device)

    @torch.no_grad()
    def finish(self):
        if any(a.terminal is None for a in self.attempts):raise RuntimeError('Interrupted cohort cannot be scored')
        tensor=lambda values:torch.tensor(values,device=self.device,dtype=torch.float64)
        returns=tensor([a.total for a in self.attempts]);valid=~self.failed
        qualified=torch.tensor([a.qualified for a in self.attempts],device=self.device)&valid
        H=tensor([a.selected.get('H',0.) if a.selected else 0. for a in self.attempts])
        components=dict(progress=tensor([a.progress for a in self.attempts]),
            progress_credit=tensor([a.progress_credit for a in self.attempts]),
            outcome_credit=tensor([a.outcome_credit for a in self.attempts]),
            horizontal_reach_fraction=H,qualified_hit=qualified.double(),
            plane_crossing=qualified.double(),
            release_time_s=tensor([a.selected['time_s'] if a.selected else float('nan') for a in self.attempts]),
            selected_delivery_guide=tensor([a.selected.get('delivery_guide',0.) if a.selected else 0.
                                          for a in self.attempts]),
            fold_formed=tensor([a.formed for a in self.attempts]),
            fold_propagated=tensor([a.propagated for a in self.attempts]),
            fold_opened=tensor([a.opened for a in self.attempts]),
            event_completed=tensor([a.terminal in ('hit','event_end') for a in self.attempts]),
            timeout=tensor([a.terminal=='timeout' for a in self.attempts]),
            failure=self.failed.double(),executed_duration_s=self.terminal_times,
            time_cost=torch.zeros_like(returns),workspace_cost=torch.zeros_like(returns),
            net_episode_return=returns)
        for name in ('world_outward_mps','relative_outward_mps','world_cosine','relative_cosine'):
            components[name]=tensor([a.selected.get(name,0.) if a.selected else 0. for a in self.attempts])
        indices=torch.tensor([round(a.selected.get('time_s',0.)*100) if a.selected else -1
                              for a in self.attempts],device=self.device)
        score=Stage1Score(components['progress_credit'],valid,qualified,indices,components,
            torch.zeros((self.batch,self.samples),device=self.device,dtype=torch.float64))
        if not torch.allclose(self.ledger.sum(1),returns,atol=1e-12,rtol=1e-12):
            raise RuntimeError('Retained fold-delivery ledger does not telescope')
        self.transition_rewards=self.ledger.float()
        return returns.float(),valid,score,torch.zeros_like(returns)

    def event_records(self):
        return [a.record() for a in self.attempts]

    def provenance(self):
        return dict(evaluation_backend='native_tensor_substep_guard_v1',
                    reward_mode='retained_fold_delivery_v1',evaluation_environments=self.batch,
                    evaluation_policy_lanes=self.settings.get('evaluation_environments',128),
                    unique_initial_states=1)
