"""Continuous experimental fold-transport/straightening reward, without attempts.

Material-region IDs associate measured bends, not policy attempts or whip counts.
All windows are causal. The score is kinematic, not energy or validated causality.
"""
from collections import deque
import math
import numpy as np

from .horizontal_fold import FoldGeometry

SHORT_MODE = 'continuous_directional_whip_four_second_v6'
FEATURE_SIZE = 16
SHORT_CONTRACT = {'version': 'continuous_directional_whip_four_second_v6',
 'links': 10,
 'horizon_s': 4.0,
 'observation_hz': 100,
 'history_window_s': 0.1,
 'support_fade_s': 0.1,
 'travel_reference_fraction': 0.1,
 'soft_fold_lower_deg': 90.0,
 'soft_fold_upper_deg': 150.0,
 'coherence_reference': 0.8,
 'rate_weight': 10.0,
 'reach_power': 2.0,
 'alignment_power': 2.0,
 'failure_return': None,
 'time_cost_per_s': 0.0,
 'effort_cost': 0.0,
 'path_cost': 0.0,
 'target_reward': False,
 'failure_cancels_credit': False,
 'per_whip_bonus': False,
 'early_completion': False,
 'minimum_radius_fraction': 1e-06,
 'rate_integration': 'right_endpoint_dt_v1',
 'geometry': 'existing_ten_link_FoldGeometry_v1',
 'output_speed': 'minimum_world_x_relative_x_end_to_end_extension_v1',
 'workspace_termination': False,
 'workspace_cost': '10_times_fixed_horizon_mean_squared_excess_m2_v1',
 'workspace_radius_m': 0.55,
 'workspace_height_lower_m': 1.5,
 'workspace_height_upper_m': 2.6,
 'negative_net_transform': 'identity',
 'failure_cancels_cost_and_credit': False,
 'failure_cost_per_remaining_s': 1.0,
 'failure_preserves_prefix': True,
 'physical_termination': False,
 'physical_violation_diagnostic': True,
 'numerical_termination': True,
 'output_direction_world': [1.0, 0.0, 0.0],
 'reach_measure': 'positive_world_x_attachment_relative_extension_v1'}
TRACE_FIELDS = ('fold_support', 'horizontal_reach_fraction', 'alignment', 'world_outward_mps', 'relative_outward_mps', 'straightening_mps', 'effective_outward_mps', 'reward_rate', 'accrued_credit', 'end_to_end_fraction', 'transport_signal', 'max_turn_rad')
SOFT_TRACE_FIELDS = TRACE_FIELDS + ('workspace_cost', 'raw_net_return', 'net_return')


def output_metrics(positions, velocities, length, *, directed_x=True):
    """Same-sample world +X output; preparation remains unconstrained."""
    if not directed_x: raise ValueError("Only world +X output is supported")
    r=positions[-1]-positions[0];relative=velocities[-1]-velocities[0]
    rho=float(np.linalg.norm(r[:2]));distance=float(np.linalg.norm(r))
    radial=np.r_[r[:2]/rho,0.] if rho>1e-6*length else np.zeros(3)
    if directed_x:
        radial=np.array([1.,0.,0.])
        rho=max(0.,float(r[0]))
    world=velocities[-1]
    uw=float(world@radial);ur=float(relative@radial)
    ws=float(np.linalg.norm(world));rs=float(np.linalg.norm(relative))
    cw=uw/ws if ws>0 else 0.;cr=ur/rs if rs>0 else 0.
    straightening=float(relative@r/distance) if distance>1e-6*length else 0.
    return dict(horizontal_reach_fraction=float(np.clip(rho/length,0.,1.)),
        alignment=float(np.clip(min(cw,cr),0.,1.)),world_outward_mps=uw,
        relative_outward_mps=ur,straightening_mps=straightening,
        effective_outward_mps=max(0.,min(uw,ur,straightening)),
        end_to_end_fraction=float(distance/length))


class ContinuousWhipAttempt:
    """One full-horizon episode, without event-count or preparation bonuses."""
    def __init__(self, rest, *, workspace_origin=None, plain_net=True, directed_x=True, horizon_s=4.):
        if horizon_s != 4. or not (directed_x and plain_net and workspace_origin is not None):
            raise ValueError('Four-second horizon requires the directed plain-workspace task')
        self.horizon_s=float(horizon_s)
        self.geometry=FoldGeometry(rest);self.L=self.geometry.L
        self.workspace_origin=None if workspace_origin is None else np.asarray(workspace_origin,float)
        if self.workspace_origin is not None and (self.workspace_origin.shape!=(3,) or not np.isfinite(self.workspace_origin).all()):
            raise ValueError('Workspace origin must be finite world XYZ')
        if plain_net and workspace_origin is None:raise ValueError('Plain net task requires a soft workspace')
        self.directed_x=directed_x
        self.plain_net=plain_net;self.failure_cost=0.
        self.cost=0.
        self.fields=TRACE_FIELDS if workspace_origin is None else SOFT_TRACE_FIELDS
        self.histories={};self.support=deque(maxlen=11)
        self.time=self.total=self.accrued=self.peak_rate=self.support_time=self.output_time=0.
        self.terminal=None;self.initialized=False;self.selected=None
        self.last={key:0. for key in self.fields};self.trace=[];self.log=[]

    @staticmethod
    def _arrays(p,v):
        p=np.asarray(p,float);v=np.asarray(v,float)
        if p.shape!=(11,3) or v.shape!=(11,3) or not np.isfinite(p).all() or not np.isfinite(v).all():
            raise ValueError('Continuous whip requires finite eleven-node positions and velocities')
        return p,v

    def _fold_support(self,time,rows):
        histories={};transport=0.;max_turn=0.
        for b in rows:
            if b['unresolved'] or set(b['flags'])&{'plateau','ambiguous_identity'}:continue
            ident=b['track_id'];samples=deque(self.histories.get(ident,()),maxlen=11)
            if samples and not math.isclose(time-samples[-1][0],.01,abs_tol=1e-8):samples.clear()
            samples.append((time,b['centre_m'],b['theta'],b['coherence']))
            histories[ident]=samples;max_turn=max(max_turn,b['theta'])
            if len(samples)<11:continue
            strength=float(np.clip((max(s[2] for s in samples)-math.pi/2)/(math.pi/3),0.,1.))
            coherence=float(np.clip(min(s[3] for s in samples)/.8,0.,1.))
            travel=float(np.clip((samples[-1][1]-samples[0][1])/(.1*self.L),0.,1.))
            transport=max(transport,strength*coherence*travel)
        self.histories=histories
        self.support.append((time,transport))
        support=max(value*max(0.,1.-(time-stamp)/.1) for stamp,value in self.support)
        return support,transport,max_turn

    def initialize(self,positions,velocities):
        if self.initialized:raise RuntimeError('Initialize exactly once')
        p,v=self._arrays(positions,velocities);_,rows=self.geometry.observe(p)
        support,transport,turn=self._fold_support(0.,rows)
        self.last.update(output_metrics(p,v,self.L,directed_x=self.directed_x),fold_support=support,transport_signal=transport,max_turn_rad=turn)
        self.trace.append([self.last[k] for k in self.fields]);self.initialized=True

    def observe(self,time,positions,velocities, *, drone_position=None):
        if not self.initialized or self.terminal is not None:raise RuntimeError('Episode is not active')
        if not math.isclose(time-self.time,.01,abs_tol=1e-8) or time>self.horizon_s+1e-9:
            raise ValueError('Observe causal 100Hz endpoints through the configured horizon')
        p,v=self._arrays(positions,velocities)
        charge=0.
        if self.workspace_origin is not None:
            tracking=np.asarray(drone_position,float)
            if tracking.shape!=(3,) or not np.isfinite(tracking).all():
                raise ValueError('Soft workspace cost requires finite drone tracking-point XYZ')
            radial=max(0.,float(np.linalg.norm(tracking[:2]-self.workspace_origin[:2]))-.55)
            vertical=max(0.,1.5-tracking[2])+max(0.,tracking[2]-2.6)
            charge=10.*(radial*radial+vertical*vertical)/(100.*self.horizon_s)
        previous=self.total
        _,rows=self.geometry.observe(p)
        support,transport,turn=self._fold_support(time,rows)
        m=output_metrics(p,v,self.L,directed_x=self.directed_x)
        rate=10.*support*m['horizontal_reach_fraction']**2*m['alignment']**2*m['effective_outward_mps']/self.L
        reward=rate*.01;self.accrued+=reward;self.cost+=charge
        self.total=self.prefix_return()
        if self.workspace_origin is not None:reward=self.total-previous
        self.time=float(time)
        self.last=dict(m,fold_support=support,transport_signal=transport,max_turn_rad=turn,
                       reward_rate=rate,accrued_credit=self.accrued)
        if self.workspace_origin is not None:
            self.last.update(workspace_cost=self.cost,raw_net_return=self.accrued-self.cost,failure_cost=self.failure_cost,net_return=self.total)
        self.trace.append([self.last[k] for k in self.fields])
        self.support_time+=.01*support;self.output_time+=.01*(rate>0.)
        if rate>self.peak_rate:
            self.peak_rate=rate;self.selected=dict(self.last,time_s=time)
        if time>=self.horizon_s-1e-10:
            self.terminal='horizon';self.log.append(dict(kind='horizon',time_s=time,return_value=self.total))
        return reward

    def prefix_return(self):
        if self.workspace_origin is None:return self.accrued
        raw=self.accrued-self.cost
        return raw

    def fail(self,time,reason='physical_failure'):
        if self.terminal is not None or not self.initialized:raise RuntimeError('Episode is not active')
        if not self.time<=time<=self.horizon_s+1e-9 or reason in ('horizon','hit','event_end','timeout'):
            raise ValueError('Failure must follow the valid prefix')
        self.failure_cost=max(0.,self.horizon_s-time)
        correction=-self.failure_cost
        self.total=self.prefix_return()-self.failure_cost
        entry=dict(kind=reason,time_s=time,return_value=self.total,
            failure_cost=self.failure_cost,preserved_credit=self.accrued,preserved_workspace_cost=self.cost)
        self.time=float(time);self.terminal=reason;self.log.append(entry)
        return correction

    @property
    def qualified(self):
        # No unvalidated binary success gate is introduced by a continuous score.
        return False

    def features(self):
        m=self.last
        values=[self.time/self.horizon_s,m['fold_support'],m['horizontal_reach_fraction'],m['alignment'],
            m['world_outward_mps']/10.,m['relative_outward_mps']/10.,m['straightening_mps']/10.,
            m['effective_outward_mps']/10.,m['reward_rate']/10.,self.accrued/10.,
            m['end_to_end_fraction'],m['transport_signal'],m['max_turn_rad']/math.pi,
            self.peak_rate/10.,self.support_time/self.horizon_s,self.output_time/self.horizon_s]
        if self.workspace_origin is not None:values.extend((self.cost/10.,self.total/10.))
        return np.asarray(values,np.float32)

    def record(self):
        return dict(terminal=self.terminal,terminal_time_s=self.time,selected=self.selected,
            events=list(self.log),qualified=None,episode_return=self.total,
            continuous_metrics=dict(accrued_credit=self.accrued,peak_reward_rate=self.peak_rate,
                fold_support_integral_s=self.support_time,positive_output_duration_s=self.output_time),
            workspace_cost=self.cost,raw_net_return=self.accrued-self.cost,failure_cost=self.failure_cost,
            output_direction_world=[1.,0.,0.] if self.directed_x else None,
            contract=SHORT_MODE,qualification_note='Continuous diagnostic; binary whip success is not defined.')
