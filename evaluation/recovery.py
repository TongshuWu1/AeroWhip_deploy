"""Opt-in strike/return diagnostic. No training reward or contact dynamics.

The 100-Hz endpoint following an interpolated qualified crossing hands control
to a rate-limited position-to-velocity return script. Delay, motors, cable and
BDF2 history are retained. Protected boxes are provisional visual envelopes,
NOT measured CAD or a collision-avoidance guarantee.
"""
from dataclasses import asdict, dataclass
import math

import numpy as np
import torch

from simulation.tensor_model import ControlledState, controlled_values
from tasks.whip.deterministic_evaluation import DeterministicEvaluationEnvironment
from tasks.whip.strike import surface_events


@dataclass(frozen=True)
class RecoverySettings:
    maximum_duration_s: float = 8.
    braking_duration_s: float = .5
    maximum_speed_mps: float = .5
    command_acceleration_mps2: float = 1.
    position_gain_per_s: float = 1.
    position_tolerance_m: float = .05
    speed_tolerance_mps: float = .1
    hover_duration_s: float = .5
    cable_radius_m: float = .001
    clearance_margin_m: float = .003
    clamp_exemption_m: float = .01

    def __post_init__(self):
        if any(isinstance(v, bool) or not math.isfinite(v) or v <= 0
               for v in asdict(self).values()):
            raise ValueError('Recovery settings must be positive and finite')
        if self.braking_duration_s+self.hover_duration_s >= self.maximum_duration_s:
            raise ValueError('Recovery horizon must leave time for return')


def segment_box_distance(a, b, lower, upper):
    """Exact Euclidean segment/AABB distance via piecewise quadratic minima."""
    a, b, lower, upper = map(lambda x: np.asarray(x, dtype=float), (a,b,lower,upper))
    if any(x.shape != (3,) or not np.isfinite(x).all() for x in (a,b,lower,upper)) or np.any(lower > upper):
        raise ValueError('Invalid segment or box')
    d, breaks = b-a, [0.,1.]
    for j in range(3):
        if d[j] != 0:
            breaks.extend(t for t in ((lower[j]-a[j])/d[j], (upper[j]-a[j])/d[j]) if 0 < t < 1)
    breaks = sorted(set(breaks))
    def squared(t):
        p = a+t*d
        return np.square(np.maximum(lower-p,0)+np.maximum(p-upper,0)).sum()
    result = min(squared(t) for t in breaks)
    for lo, hi in zip(breaks,breaks[1:]):
        p = a+(lo+hi)/2*d
        active = (p < lower)|(p > upper)
        bound = np.where(p < lower,lower,upper)
        denom = np.square(d[active]).sum()
        if denom > 0:
            t = np.clip(-np.dot(d[active], (a-bound)[active])/denom,lo,hi)
            result = min(result,squared(t))
    return float(np.sqrt(result))


def clearance(cable, position, rotation, settings):
    """Full segments versus provisional body/battery/rotor boxes in tracking frame.

    Only the first 1 cm from the attachment along edge zero is exempt. Later
    segments returning near the clamp are never exempt. Rotor squares enclose
    the visual disks; sampled clearance does not rule out between-step contact.
    """
    q, p, r = map(np.asarray,(cable,position,rotation))
    if q.ndim != 2 or q.shape[1] != 3 or len(q)<2 or p.shape!=(3,) or r.shape!=(3,3):
        raise ValueError('Invalid clearance state shapes')
    if not all(np.isfinite(x).all() for x in (q,p,r)):
        raise ValueError('Nonfinite clearance state')
    if not np.allclose(r@r.T,np.eye(3),rtol=0,atol=1e-8) or abs(np.linalg.det(r)-1)>1e-8:
        raise ValueError('Invalid clearance rotation')
    body = (q-p)@r
    boxes = [('body',[-.03,-.03,-.036],[.03,.03,0.]),
             ('battery',[-.031,-.015,-.051],[.031,.015,-.036])]
    for x in (-.05,.05):
        for y in (-.05,.05):
            boxes.append((f'rotor_{x}_{y}',[x-.038,y-.038,-.003],[x+.038,y+.038,.003]))
    best = dict(clearance_m=float('inf'),segment=None,protected_part=None)
    for i,(a,b) in enumerate(zip(body[:-1],body[1:])):
        if i == 0:
            length = float(np.linalg.norm(b-a))
            if length <= settings.clamp_exemption_m:
                continue
            a = a+(b-a)*(settings.clamp_exemption_m/length)
        for name,lower,upper in boxes:
            gap = segment_box_distance(a,b,lower,upper)-settings.cable_radius_m-settings.clearance_margin_m
            if gap < best['clearance_m']:
                best = dict(clearance_m=gap,segment=i,protected_part=name)
    return best


def return_command(position, origin, previous, elapsed, settings, dt=.01):
    target = np.zeros(3) if elapsed < settings.braking_duration_s else settings.position_gain_per_s*(origin-position)
    target *= min(1.,settings.maximum_speed_mps/max(float(np.linalg.norm(target)),1e-30))
    delta = target-previous
    delta *= min(1.,settings.command_acceleration_mps2*dt/max(float(np.linalg.norm(delta)),1e-30))
    return previous+delta


class StrikeReturnEnvironment(DeterministicEvaluationEnvironment):
    """One lane, stop issuing actor actions at first qualified sampled crossing."""
    @torch.no_grad()
    def reset(self):
        self.hit = None
        return super().reset()

    @torch.no_grad()
    def endpoint(self, command):
        applied = self.queue[:,0].clone() if self.delay else command
        if self.delay:
            self.queue = torch.cat((self.queue[:,1:],command[:,None]),1)
        result = self._advance(applied)
        values = controlled_values(result)
        finite = all(bool(torch.isfinite(v).all()) for v in values)
        if finite:
            r = result.rotation
            finite = (bool(((r@r.transpose(-1,-2)-self.model.eye3).abs() <= 1e-8).all())
                      and bool(((torch.linalg.det(r)-1).abs() <= 1e-8).all()))
        if finite:
            self.state = ControlledState(*(v.clone() for v in values))
        p,v,q,cv = self._observe(self.state)
        safe = finite and bool(self._envelope(p,q,self.state.rotation).all())
        return p,v,q,cv,applied,finite,safe

    @torch.no_grad()
    def step(self, action):
        if self.hit is not None or not bool(self.active.any()) or self.decision>=self.decisions:
            raise RuntimeError('Strike phase has ended')
        if action.shape != (1,3) or not bool(torch.isfinite(action).all()) or bool((action.abs()>1.000001).any()):
            raise ValueError('Expected normalized finite actor action')
        self.previous_action = action.detach().clamp(-1,1)
        command = torch.cat((self.previous_action.double()*self.velocity_limit,self.hold[3].reshape(1,1)),1)
        self.actions[:,self.decision] = self.previous_action
        for _ in range(self.stride):
            p,v,q,cv,applied,finite,safe = self.endpoint(command)
            self.index += 1
            inside = self.configuration.workspace.contains(p[:,None],self.origin[None])[:,0]
            self.workspace_breached |= ~inside
            self._record(p,v,q,cv,command,applied,torch.tensor([finite],device=self.device),~inside)
            self.active.fill_(safe)
            self.failed.fill_(not safe)
            if not safe:
                break
            i = self.index
            events = surface_events(self.history['cable_positions'][:,i-1:i+1],
                self.history['cable_velocities'][:,i-1:i+1],self.time[i-1:i+1],
                self.scorer.center,self.scorer.direction,self.configuration.strike_surface,self.scorer.speed)
            if bool(events['qualified'][0,0]):
                self.hit = dict(time_s=(i-1)*.01+float(events['elapsed'][0,0]),endpoint_time_s=i*.01,
                    endpoint_index=i, lateral_m=float(events['lateral'][0,0]),vertical_m=float(events['vertical'][0,0]),
                    forward_mps=float(events['forward'][0,0]),entry_angle_deg=math.degrees(float(events['angle'][0,0])),
                    position_world_m=events['positions'][0,0,-1].cpu().tolist(),
                    velocity_world_mps=events['velocities'][0,0,-1].cpu().tolist())
                break
        self.lengths += 1
        self.decision += 1
        return self.frames,self.indices(),self.hit is not None or not safe or self.decision==self.decisions


@torch.no_grad()
def evaluate_return(env, settings, *, stopped, emit):
    """No strike reward is evaluated under this new, shorter episode protocol."""
    arrays = env.replay_arrays(0)
    if arrays is None:
        return None,dict(status='not_started_no_valid_prefix')
    count = env.index+1
    arrays = {k:(v[:count].copy() if k!='cable_marker_indices' else v.copy()) for k,v in arrays.items()}
    arrays['phase_id'] = np.zeros(len(arrays['time_s']),dtype=np.int64)
    arrays['clearance_m'] = np.array([clearance(q,p,r,settings)['clearance_m'] for q,p,r in
        zip(arrays['cable_positions'],arrays['drone_positions'],arrays['drone_rotations'])])
    if env.hit is None:
        return arrays,dict(status='not_started_no_qualified_hit',success=False)
    rows = {k:list(v) for k,v in arrays.items() if k!='cable_marker_indices'}
    origin = env.origin.cpu().numpy()
    previous = arrays['command'][-1,:3].copy()
    hit_time, hover, status, last_clearance = arrays['time_s'][-1],0.,'timeout',None
    # Collision-envelope breach at handoff is a failure, not an avoided contact.
    minimum = float(arrays['clearance_m'][-1])
    if minimum <= 0:
        return arrays,dict(status='protected_envelope_intersection_at_handoff',success=False,
                           minimum_recovery_clearance_m=minimum,duration_s=0.)
    for j in range(round(settings.maximum_duration_s/.01)):
        if stopped():
            status='cancelled'
            break
        elapsed = j*.01
        p = rows['drone_positions'][-1]
        previous = return_command(p,origin,previous,elapsed,settings)
        command = env.model.tensor([*previous,float(env.hold[3])])[None]
        p,v,q,cv,applied,finite,safe = env.endpoint(command)
        if not finite:
            status='numerical_failure'
            break
        state = env.state
        host = lambda x: x[0].detach().cpu().numpy().copy()
        p,v,q,cv = map(host,(p,v,q,cv))
        r = host(state.rotation)
        last_clearance = clearance(q,p,r,settings)
        minimum = min(minimum,last_clearance['clearance_m'])
        inside = bool(env.configuration.workspace.contains(env.model.tensor(p)[None,None],env.origin[None])[0,0])
        position_error, speed = float(np.linalg.norm(p-origin)),float(np.linalg.norm(v))
        near = position_error<=settings.position_tolerance_m and speed<=settings.speed_tolerance_mps
        hover = hover+.01 if near and elapsed>=settings.braking_duration_s else 0.
        phase = 1 if elapsed<settings.braking_duration_s else (3 if near else 2)
        row = dict(time_s=hit_time+(j+1)*.01,state_valid=True,drone_positions=p,drone_velocities=v,
            drone_rotations=r,drone_angular_velocities=host(state.omega_body),cable_positions=q,cable_velocities=cv,
            cable_material_angles_rad=host(state.material_angles),motor_thrust_n=host(state.motor_thrust_n),
            requested_wrench=host(state.held_wrench),command=host(command),applied_command=host(applied),
            workspace_violation=float(not inside),phase_id=phase,clearance_m=last_clearance['clearance_m'])
        for k in rows:
            rows[k].append(row[k])
        if not safe:
            status='existing_tilt_or_cable_floor_failure'
            break
        if last_clearance['clearance_m'] <= 0:
            status='protected_envelope_intersection'
            break
        if hover+1e-12 >= settings.hover_duration_s:
            status='returned_and_hovered'
            break
        if j%100 == 0:
            emit(phase='recovering',message=f'Scripted recovery {elapsed:.2f} s; position error {position_error:.3f} m')
    result = {k:np.asarray(v) for k,v in rows.items()}
    result['cable_marker_indices'] = arrays['cable_marker_indices']
    return result,dict(status=status,success=status=='returned_and_hovered',
        duration_s=float(result['time_s'][-1]-hit_time),minimum_recovery_clearance_m=minimum,
        last_clearance=last_clearance,position_error_m=float(np.linalg.norm(result['drone_positions'][-1]-origin)),
        speed_mps=float(np.linalg.norm(result['drone_velocities'][-1])),
        geometry='provisional_tracking_frame_body_battery_rotor_boxes_v1',sample_period_s=.01,
        limitation='Sampled protected-envelope diagnostic, no contact dynamics or guaranteed avoidance')
