"""Causal first-useful-crossing termination for the approved five-second task."""
import numpy as np
import torch
from evaluation.whip_output_decomposition import decompose
from .flexible_delivery import FlexibleDeliveryScorer

MODE='first_hit_flexible_delivery_v2'


class FirstHitDeliveryScorer(FlexibleDeliveryScorer):
    first_hit=True


class FirstHitTracker:
    """Observe executed100Hz endpoints; require negative arrival before departure.

    Decompose only candidate crossings, using the same float64 CPU projection
    as offline scoring. Failed lanes are excluded by the caller before observe.
    """
    def __init__(self,positions,rest,center):
        self.center=center
        self.rest=rest.detach().cpu().numpy()
        self.negative_seen=positions[:,-1,0] < center[0]

    @torch.no_grad()
    def observe(self,previous_p,current_p,previous_v,current_v,active):
        old=previous_p[:,-1,0]-self.center[0]
        new=current_p[:,-1,0]-self.center[0]
        candidate=active & self.negative_seen & (old<=0) & (new>0)
        hit=torch.zeros_like(active)
        center=self.center.detach().cpu().numpy()
        for lane in candidate.nonzero().flatten().tolist():
            fraction=float(-old[lane]/(new[lane]-old[lane])) if old[lane]<0 else 0.
            p=(previous_p[lane]+fraction*(current_p[lane]-previous_p[lane])).detach().cpu().numpy()
            v=(previous_v[lane]+fraction*(current_v[lane]-previous_v[lane])).detach().cpu().numpy()
            delta=p[-1]-center
            if abs(delta[1])<=.5 and abs(delta[2])<=.3 and v[-1,0]>0:
                shape=decompose(p[None],v[None],self.rest)['shape_speed_mps'][0,-1]
                hit[lane]=bool(min(v[-1,0],shape)>=1.)
        self.negative_seen=torch.where(active & (new<0),True,self.negative_seen)
        self.negative_seen &= ~(active & (new>0))
        return hit
