"""Approved experimental ten-link fold/outreach task; no physics or target reward.

Streaming material-region association and one-event reward ledger. The thresholds
are engineering task choices, not a universal physical definition of whipping.
"""
from copy import deepcopy
import math
import numpy as np
from .fold_geometry import regions

MODE = 'horizontal_fold_outreach_v2'
CONTRACT = dict(version=MODE, links=10, horizon_s=7., observation_hz=100,
    bandwidth_reference_m=.075, reference_length_m=.9525, grid_points=1001,
    formation_turn_deg=150., coherence_min=.8, proximal_fraction=.5,
    propagation_fraction=.2, distal_fraction=.65, opening_turn_deg=60.,
    persistence_samples=3, radial_floor_mps=.05, output_cone_deg=45.,
    minimum_radius_fraction=1e-6, reach_weight=5., time_cost_per_s=.1,
    timeout_penalty=2., failure_penalty=10., qualified_reach_fraction=.75)


def output_sample(positions, velocities, length):
    r=np.asarray(positions[-1]-positions[0],float).copy();r[2]=0.
    rho=float(np.linalg.norm(r)); e=r/rho if rho>1e-6*length else np.zeros(3)
    v=np.asarray(velocities[-1]); relative=v-velocities[0]
    uw=float(v@e);ur=float(relative@e)
    cw=float(uw/np.linalg.norm(v)) if np.linalg.norm(v)>0 else 0.
    cr=float(ur/np.linalg.norm(relative)) if np.linalg.norm(relative)>0 else 0.
    outward=rho>1e-6*length and uw>.05 and ur>.05
    aligned=cw>=math.cos(math.pi/4) and cr>=math.cos(math.pi/4)
    return dict(H=float(np.clip(rho/length,0,1)),world_outward_mps=uw,
        relative_outward_mps=ur,world_cosine=cw,relative_cosine=cr,
        world_velocity_mps=v.tolist(),relative_velocity_mps=relative.tolist(),
        outward=bool(outward),aligned=bool(aligned))


class FoldGeometry:
    """Causal port of the reviewed unique half-height-core overlap tracker."""
    def __init__(self,rest):
        self.rest=np.asarray(rest,float)
        if self.rest.shape!=(10,) or not np.isfinite(self.rest).all() or np.any(self.rest<=0):
            raise ValueError('Horizontal fold requires ten positive finite rest lengths')
        self.L=float(self.rest.sum());self.nodes=np.r_[0.,np.cumsum(self.rest)]
        self.grid=np.linspace(self.nodes[1],self.nodes[-2],1001)
        self.bandwidth=.075*self.L/.9525
        kernel=np.exp(-.5*((self.grid[None]-self.nodes[1:-1,None])/self.bandwidth)**2)
        self.kernel=kernel/np.trapezoid(kernel,self.grid,axis=1)[:,None]
        self.previous=[];self.next_id=0

    def observe(self,p):
        p=np.asarray(p,float)
        if p.shape!=(11,3) or not np.isfinite(p).all():raise ValueError('Finite eleven-node geometry required')
        edge=np.diff(p,axis=0);length=np.linalg.norm(edge,axis=-1)
        if np.any(length<=0):raise ValueError('Collapsed cable segment')
        u=edge/length[:,None];cross=np.cross(u[:-1],u[1:]);norm=np.linalg.norm(cross,axis=-1)
        dot=np.sum(u[:-1]*u[1:],axis=-1);turn=np.arctan2(norm,dot)
        axes=np.divide(cross,norm[:,None],out=np.zeros_like(cross),where=norm[:,None]>1e-10)
        current=regions(self.grid,turn@self.kernel,self.bandwidth)
        left=[[j for j,b in enumerate(current) if max(a['core_m'][0],b['core_m'][0])<min(a['core_m'][1],b['core_m'][1])] for a in self.previous]
        right=[[i for i,js in enumerate(left) if j in js] for j in range(len(current))]
        for j,b in enumerate(current):
            unique=len(right[j])==1 and len(left[right[j][0]])==1
            clean=unique and 'plateau' not in b['flags'] and 'plateau' not in self.previous[right[j][0]]['flags']
            if clean:b['track_id']=self.previous[right[j][0]]['track_id']
            else:
                b['track_id']=self.next_id;self.next_id+=1
                if right[j]:b['flags'].append('ambiguous_identity')
            ix=np.flatnonzero((self.nodes[1:-1]>=b['basin_m'][0])&(self.nodes[1:-1]<=b['basin_m'][1]))
            theta=float(turn[ix].sum());unresolved=bool(np.any((norm[ix]<=1e-10)&(dot[ix]<0)))
            c=float(np.clip(np.linalg.norm((turn[ix,None]*axes[ix]).sum(0))/theta,0,1)) if theta>0 and not unresolved else 0.
            b.update(joint_indices=(ix+1).tolist(),theta=theta,coherence=c,unresolved=unresolved)
        self.previous=current
        return turn,current


class HorizontalFoldAttempt:
    """One causal attempt including temporary guidance and every terminal ledger."""
    def __init__(self,rest):
        self.geometry=FoldGeometry(rest);self.L=self.geometry.L
        self.counts={};self.active=None;self.time=0.;self.phi=0.;self.total=0.
        self.terminal=None;self.selected=None;self.log=[];self.last_regions=[]
        self.f=self.p=self.o=0.;self.formed=False;self.propagated=False;self.opened=False

    def _end(self,reason,time):
        H=0. if self.selected is None else self.selected['H']
        target=(5*H*H if reason=='event_end' else -2 if reason=='timeout' else -10)-.1*time
        reward=target-self.total
        self.phi=0.;self.total=target;self.time=time;self.terminal=reason
        self.log.append(dict(kind=reason,time_s=time,selected=deepcopy(self.selected),return_value=target))
        return reward

    def fail(self,time,reason='physical_failure'):
        if self.terminal is not None:raise RuntimeError('Cannot alter a terminated attempt')
        if not self.time<=time<=7+1e-10:raise ValueError('Failure time must follow observed prefix')
        return self._end(reason,float(time))

    def _candidate_potential(self,rows):
        return max((min(b['theta']/math.radians(150),1)*min(b['coherence']/.8,1)
            for b in rows if b['centre_m']<=.5*self.L and b['joint_indices'] and not b['unresolved']
            and not set(b['flags'])&{'plateau','ambiguous_identity'}),default=0.)

    def observe(self,time,positions,velocities):
        if self.terminal is not None:raise RuntimeError('No observations after task terminal')
        time=float(time)
        if not math.isclose(time-self.time,.01,abs_tol=1e-8) or time>7+1e-9:
            raise ValueError('Observe causal 100Hz endpoints, through seven seconds')
        if np.asarray(velocities).shape!=(11,3) or not np.isfinite(velocities).all():raise ValueError('Finite velocities required')
        turn,rows=self.geometry.observe(positions);self.last_regions=rows
        old_phi=self.phi;old_time=self.time;a=self.active
        if a is not None:
            match=next((b for b in rows if b['track_id']==a['id']),None)
            unique=match is not None and not set(match['flags'])&{'plateau','ambiguous_identity'}
            if unique:
                a['max_s']=max(a['max_s'],match['centre_m'])
                if match['centre_m']-a['s_a']>=.2*self.L and match['centre_m']>=.65*self.L:
                    if not a['propagated']:self.log.append(dict(kind='propagated',time_s=time,track_id=a['id']))
                    a['propagated']=True;self.propagated=True
            tail=float(turn[a['tail']:].sum());a['peak']=max(a['peak'],tail)
            low=tail<=math.pi/3
            if a['opened'] or (a['propagated'] and low):
                sample=output_sample(np.asarray(positions),np.asarray(velocities),self.L)
                if not a['opened']:
                    a['low']+=1;a['identity_lost'] |= not unique
                    if a['low']>=3:
                        a['opened']=True;self.opened=True
                        self.log.append(dict(kind='opening_confirmed',time_s=time,track_id=a['id']))
                continuing=low and sample['outward']
                a['bad']=0 if continuing else a['bad']+1
                if continuing and sample['aligned'] and (self.selected is None or sample['H']>self.selected['H']):
                    self.selected=dict(sample,time_s=time,tail_turn_rad=tail,track_id=a['id'],
                        positions_m=np.asarray(positions).tolist())
                if a['opened'] and a['bad']>=3:return self._end('event_end',time)
            else:
                a['low']=a['bad']=0;self.selected=None
                if not unique or a['identity_lost']:
                    self.log.append(dict(kind='discarded',time_s=time,track_id=a['id']))
                    self.active=a=None;self.counts={}
        if a is None:
            counts={};eligible=[]
            for b in rows:
                good=(b['theta']>=math.radians(150) and b['coherence']>=.8 and b['centre_m']<=.5*self.L
                    and b['joint_indices'] and not b['unresolved'] and not set(b['flags'])&{'plateau','ambiguous_identity'})
                if good:
                    ident=b['track_id'];counts[ident]=self.counts.get(ident,0)+1
                    if counts[ident]>=3:eligible.append(b)
            self.counts=counts
            if eligible:
                b=sorted(eligible,key=lambda b:(-b['theta']*b['coherence'],b['centre_m'],b['track_id']))[0]
                tail=min(b['joint_indices'])-1
                self.active=a=dict(id=b['track_id'],s_a=b['centre_m'],max_s=b['centre_m'],tail=tail,
                    peak=float(turn[tail:].sum()),propagated=False,opened=False,low=0,bad=0,identity_lost=False)
                self.formed=True;self.selected=None
                self.log.append(dict(kind='armed',time_s=time,track_id=a['id'],s_a=a['s_a'],tail_joint=tail+1))
        if time>=7-1e-10:return self._end('timeout',7.)
        if a is None:self.f,self.p,self.o=self._candidate_potential(rows),0.,0.
        else:
            self.f=1.;self.p=1. if a['propagated'] else float(np.clip((a['max_s']-a['s_a'])/(.2*self.L),0,1))
            self.o=float(np.clip(1-turn[a['tail']:].sum()/a['peak'],0,1)) if a['propagated'] and a['peak']>0 else 0.
        self.phi=.5*(self.f+self.p+self.o)
        reward=-.1*(time-old_time)+self.phi-old_phi
        self.total+=reward;self.time=time
        return reward

    def features(self):
        a=self.active
        return np.array([self.f,self.p,self.o,a is not None,self.propagated,self.opened,
            0 if a is None else a['s_a']/self.L,0 if a is None else a['max_s']/self.L,
            0 if a is None else a['tail']/9,0 if a is None else a['peak']/(2*np.pi),
            0 if a is None else a['low']/3,0 if a is None else a['bad']/3,
            0 if a is None else a['identity_lost'],0 if self.selected is None else self.selected['H'],
            0 if self.selected is None else self.selected['time_s']/7,self.time/7],dtype=np.float32)

    @property
    def qualified(self):
        return self.terminal=='event_end' and self.selected is not None and self.selected['H']>=.75
