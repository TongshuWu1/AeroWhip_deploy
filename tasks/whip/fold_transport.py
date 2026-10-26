"""Fold-transport task: continuous reward for a bend travelling toward the tip while
the tip moves outward along world +X (specification: doc/stage1.md).

Material-region IDs associate measured bends, not policy attempts or whip counts.
All windows are causal. The score is kinematic, not energy or validated causality.
"""
from collections import deque
import math
import numpy as np

from envs.configuration import OBSERVATION_PERIOD_S
from envs.task import Task
from .fold_geometry import FoldGeometry

FEATURE_SIZE = 18  # Critic task features, including workspace cost and net return.
_OUTPUT_FIELDS = ('fold_support', 'horizontal_reach_fraction', 'alignment', 'world_outward_mps', 'relative_outward_mps', 'straightening_mps', 'effective_outward_mps', 'reward_rate', 'accrued_credit', 'end_to_end_fraction', 'transport_signal', 'max_turn_rad')
TRACE_FIELDS = _OUTPUT_FIELDS + ('workspace_cost', 'raw_net_return', 'net_return')


def output_metrics(positions, velocities, length):
    """Same-sample world +X output; preparation remains unconstrained."""
    r=positions[-1]-positions[0];relative=velocities[-1]-velocities[0]
    distance=float(np.linalg.norm(r))
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


class FoldTransportEpisode:
    """One lane's full-horizon episode, without event-count or preparation bonuses.

    Reward constants come from FoldTransportConfig; horizon and workspace from the
    EnvironmentConfig.
    """
    def __init__(self, rest, config, environment, *, workspace_origin):
        self.config,self.environment=config,environment
        self.dt=OBSERVATION_PERIOD_S
        self.horizon_s=environment.duration_s
        self.steps=round(self.horizon_s/self.dt)
        fold=config.fold
        self.window=round(fold.history_window_s/self.dt)+1
        self.turn_lower=math.radians(fold.turn_lower_deg)
        self.turn_range=math.radians(fold.turn_upper_deg-fold.turn_lower_deg)
        self.geometry=FoldGeometry(rest);self.L=self.geometry.L
        self.workspace_origin=np.asarray(workspace_origin,float)
        if self.workspace_origin.shape!=(3,) or not np.isfinite(self.workspace_origin).all():
            raise ValueError('Workspace origin must be finite world XYZ')
        self.failure_cost=0.
        self.cost=0.
        self.fields=TRACE_FIELDS
        self.histories={};self.support=deque(maxlen=self.window)
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
        fold=self.config.fold
        histories={};transport=0.;max_turn=0.
        for b in rows:
            if b['unresolved'] or set(b['flags'])&{'plateau','ambiguous_identity'}:continue
            ident=b['track_id'];samples=deque(self.histories.get(ident,()),maxlen=self.window)
            if samples and not math.isclose(time-samples[-1][0],self.dt,abs_tol=1e-8):samples.clear()
            samples.append((time,b['centre_m'],b['theta'],b['coherence']))
            histories[ident]=samples;max_turn=max(max_turn,b['theta'])
            if len(samples)<self.window:continue
            strength=float(np.clip((max(s[2] for s in samples)-self.turn_lower)/self.turn_range,0.,1.))
            coherence=float(np.clip(min(s[3] for s in samples)/fold.coherence_reference,0.,1.))
            travel=float(np.clip((samples[-1][1]-samples[0][1])/(fold.travel_reference_fraction*self.L),0.,1.))
            transport=max(transport,strength*coherence*travel)
        self.histories=histories
        self.support.append((time,transport))
        support=max(value*max(0.,1.-(time-stamp)/fold.support_fade_s) for stamp,value in self.support)
        return support,transport,max_turn

    def initialize(self,positions,velocities):
        if self.initialized:raise RuntimeError('Initialize exactly once')
        p,v=self._arrays(positions,velocities);_,rows=self.geometry.observe(p)
        support,transport,turn=self._fold_support(0.,rows)
        self.last.update(output_metrics(p,v,self.L),fold_support=support,transport_signal=transport,max_turn_rad=turn)
        self.trace.append([self.last[k] for k in self.fields]);self.initialized=True

    def observe(self,time,positions,velocities, *, drone_position=None):
        if not self.initialized or self.terminal is not None:raise RuntimeError('Episode is not active')
        if not math.isclose(time-self.time,self.dt,abs_tol=1e-8) or time>self.horizon_s+1e-9:
            raise ValueError('Observe causal 100Hz endpoints through the configured horizon')
        p,v=self._arrays(positions,velocities)
        tracking=np.asarray(drone_position,float)
        if tracking.shape!=(3,) or not np.isfinite(tracking).all():
            raise ValueError('Soft workspace cost requires finite drone tracking-point XYZ')
        c,w=self.config,self.environment.workspace
        radial=max(0.,float(np.linalg.norm(tracking[:2]-self.workspace_origin[:2]))-w.radius_m)
        vertical=max(0.,w.height_lower_world_m-tracking[2])+max(0.,tracking[2]-w.height_upper_world_m)
        # Fixed-horizon mean of the weighted squared excess, integrated at right endpoints.
        charge=c.workspace_cost_weight*(radial*radial+vertical*vertical)/self.steps
        previous=self.total
        _,rows=self.geometry.observe(p)
        support,transport,turn=self._fold_support(time,rows)
        m=output_metrics(p,v,self.L)
        rate=(c.rate_weight*support*m['horizontal_reach_fraction']**c.reach_power
              *m['alignment']**c.alignment_power*m['effective_outward_mps']/self.L)
        reward=rate*self.dt;self.accrued+=reward;self.cost+=charge
        self.total=self.prefix_return()
        reward=self.total-previous
        self.time=float(time)
        self.last=dict(m,fold_support=support,transport_signal=transport,max_turn_rad=turn,
                       reward_rate=rate,accrued_credit=self.accrued,workspace_cost=self.cost,
                       raw_net_return=self.accrued-self.cost,failure_cost=self.failure_cost,net_return=self.total)
        self.trace.append([self.last[k] for k in self.fields])
        self.support_time+=self.dt*support;self.output_time+=self.dt*(rate>0.)
        if rate>self.peak_rate:
            self.peak_rate=rate;self.selected=dict(self.last,time_s=time)
        if time>=self.horizon_s-1e-10:
            self.terminal='horizon';self.log.append(dict(kind='horizon',time_s=time,return_value=self.total))
        return reward

    def prefix_return(self):
        return self.accrued-self.cost

    def fail(self,time,reason='physical_failure'):
        if self.terminal is not None or not self.initialized:raise RuntimeError('Episode is not active')
        if not self.time<=time<=self.horizon_s+1e-9 or reason in ('horizon','hit','event_end','timeout'):
            raise ValueError('Failure must follow the valid prefix')
        self.failure_cost=self.config.failure_cost_per_remaining_s*max(0.,self.horizon_s-time)
        correction=-self.failure_cost
        self.total=self.prefix_return()-self.failure_cost
        entry=dict(kind=reason,time_s=time,return_value=self.total,
            failure_cost=self.failure_cost,preserved_credit=self.accrued,preserved_workspace_cost=self.cost)
        self.time=float(time);self.terminal=reason;self.log.append(entry)
        return correction

    def features(self):
        m=self.last
        values=[self.time/self.horizon_s,m['fold_support'],m['horizontal_reach_fraction'],m['alignment'],
            m['world_outward_mps']/10.,m['relative_outward_mps']/10.,m['straightening_mps']/10.,
            m['effective_outward_mps']/10.,m['reward_rate']/10.,self.accrued/10.,
            m['end_to_end_fraction'],m['transport_signal'],m['max_turn_rad']/math.pi,
            self.peak_rate/10.,self.support_time/self.horizon_s,self.output_time/self.horizon_s,
            self.cost/10.,self.total/10.]
        return np.asarray(values,np.float32)

    def record(self):
        return dict(terminal=self.terminal,terminal_time_s=self.time,selected=self.selected,
            events=list(self.log),qualified=None,episode_return=self.total,
            continuous_metrics=dict(accrued_credit=self.accrued,peak_reward_rate=self.peak_rate,
                fold_support_integral_s=self.support_time,positive_output_duration_s=self.output_time),
            workspace_cost=self.cost,raw_net_return=self.accrued-self.cost,failure_cost=self.failure_cost,
            output_direction_world=list(self.config.output_direction_world),
            contract=self.config.name,qualification_note='Continuous diagnostic; binary whip success is not defined.')


class FoldTransportTask(Task):
    """Batch of FoldTransportEpisode lanes behind the environment's Task interface."""
    critic_feature_size = FEATURE_SIZE
    trace_fields = TRACE_FIELDS
    score_component = 'continuous_whip_credit'

    def __init__(self, config, environment, rest):
        self.config, self.environment = config, environment
        self.name = config.name
        self.rest = np.asarray(rest, float)
        self.episodes = None

    def reset(self, observation):
        self.episodes = [FoldTransportEpisode(self.rest, self.config, self.environment, workspace_origin=o)
                         for o in observation['workspace_origin']]
        for lane, episode in enumerate(self.episodes):
            episode.initialize(observation['cable_positions'][lane], observation['cable_velocities'][lane])

    def observe(self, lane, time_s, observation):
        return self.episodes[lane].observe(time_s, observation['cable_positions'], observation['cable_velocities'],
                                           drone_position=observation['drone_positions'])

    def fail(self, lane, time_s):
        return self.episodes[lane].fail(time_s, 'numerical_failure')

    def done(self, lane):
        return self.episodes[lane].terminal is not None

    def end_time(self, lane):
        return self.episodes[lane].time

    def episode_return(self, lane):
        return self.episodes[lane].total

    def critic_features(self):
        return np.stack([e.features() for e in self.episodes])

    def components(self):
        e = self.episodes
        selected = lambda key: [x.selected[key] if x.selected else 0. for x in e]
        result = dict(continuous_whip_credit=[x.accrued for x in e],
            peak_reward_rate=[x.peak_rate for x in e],
            fold_support_integral_s=[x.support_time for x in e],
            positive_output_duration_s=[x.output_time for x in e],
            horizontal_reach_fraction=selected('horizontal_reach_fraction'),
            completed_horizon=[x.terminal == 'horizon' for x in e],
            failure_cost=[x.failure_cost for x in e],
            workspace_cost=[x.cost for x in e])
        for key in ('world_outward_mps', 'relative_outward_mps', 'straightening_mps', 'alignment', 'fold_support'):
            result[key] = selected(key)
        return result

    def record(self, lane):
        return self.episodes[lane].record()

    def trace(self, lane):
        return np.asarray(self.episodes[lane].trace)

