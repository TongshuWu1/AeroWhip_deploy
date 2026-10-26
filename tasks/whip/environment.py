"""Current Stage 1 task environment over the unchanged native C++ coupled plant.

Policy: 50 Hz acceleration commands; observations and reward: 100 Hz; servo:
model clock. Episodes finish independently. A lane stops at its first
numerically invalid substep; that corrupt state is never scored or continued.
Physical-envelope violations (tilt, cable height) and workspace exits are
recorded but do not stop the lane. Physics stays float64, networks float32.
"""
import json
import math
import os
import struct
import sys
import numpy as np
import torch

from controllers.full_state_controller.tensor_controller import GeometricController
from simulation.tensor_model import TensorModel, ControlledState, controlled_values, state_values
from simulation.state import quaternion_xyzw_to_rotation_matrix
from .configuration import OBSERVATION_PERIOD_S
from .continuous_whip import ContinuousWhipAttempt, FEATURE_SIZE, TRACE_FIELDS
from .stage1 import Stage1Score


def _conditional_solver_enabled(conditional=None):
    """Supported64-bit CUDA-task default; explicit choice wins over the environment."""
    if conditional is not None:
        return conditional
    cuda_version = tuple(int(v) for v in (torch.version.cuda or '0.0').split('.')[:2])
    supported = os.name == 'nt' or (sys.platform == 'linux' and cuda_version >= (12,8))
    default = '1' if supported and struct.calcsize('P') == 8 else '0'
    flag = os.environ.get('AEROWHIP_CONDITIONAL_SOLVER', default)
    if flag not in ('0', '1'):
        raise ValueError('AEROWHIP_CONDITIONAL_SOLVER must be 0 or 1')
    return flag == '1'


class Stage1Environment:
    @torch.no_grad()
    def __init__(self, settings, snapshot, configuration, scene, *, progress=None, capture=True,
                 conditional=None):
        self.settings, self.configuration, self.scene = settings, configuration, scene
        self.device, self.batch = settings['device'], settings['parallel_environments']
        self.model = m = TensorModel(snapshot['model'], device=self.device, batched=True)
        if m.dt != OBSERVATION_PERIOD_S or scene.timestep_s != OBSERVATION_PERIOD_S:
            raise ValueError('Stage 1 requires a 100 Hz observation clock')
        if m.n != 11:
            raise ValueError('The fold reward measures a ten-link (eleven-node) cable')
        self.action_interface = settings['action_interface']
        self.controller = GeometricController(m).acceleration
        initial = scene.initial_state
        self.origin = m.tensor(initial.position_world_m)
        seed = m.controller_state(m.initial_state(initial.position_world_m,
            rotation=quaternion_xyzw_to_rotation_matrix(initial.quaternion_body_to_world_xyzw),
            tracking_velocity=initial.linear_velocity_world_mps,
            omega_body=initial.angular_velocity_body_radps,
            cable_positions=initial.cable_node_positions_world_m,
            cable_velocities=initial.cable_node_velocities_world_mps,
            material_angles=initial.cable_material_angles_rad))
        self.seed = [v.expand(self.batch,*v.shape).clone() for v in controlled_values(seed)]
        self.stride = round(1/(OBSERVATION_PERIOD_S*settings['policy_rate_hz']))
        self.samples = round(configuration.duration_s/OBSERVATION_PERIOD_S)+1
        self.time_samples = self.samples-1
        if self.time_samples % self.stride:
            raise ValueError('The episode must contain a whole number of policy decisions')
        self.decisions = self.time_samples//self.stride
        self.time = torch.arange(self.samples,device=self.device,dtype=torch.float64)*OBSERVATION_PERIOD_S
        self.maximum_thrust = float(sum(m.config['actuator']['maximum_thrust_n']))
        self.action_size = 3
        self.action_limit = m.tensor(settings['acceleration_limit_mps2'])
        rotation = quaternion_xyzw_to_rotation_matrix(initial.quaternion_body_to_world_xyzw)
        yaw = float(np.arctan2(rotation[1,0],rotation[0,0]))
        self.hold = m.tensor([0.,0.,0.,yaw])
        self.delay = round(m.config['controller']['command_delay_s']/OBSERVATION_PERIOD_S)
        self.frame_size = 18+6*m.n+8+self.action_size+1
        self.policy_encoder_options = dict(history_samples=self.samples)
        # Deployable actor inputs are unchanged. Critic-only state is stored
        # separately at policy boundaries, never appended to actor histories.
        self.actor_columns = list(range(18))
        for offset in (18,18+3*m.n):
            self.actor_columns += [offset+3*int(i)+j for i in m.marker_indices for j in range(3)]
        self.actor_columns += list(range(18+6*m.n+8,self.frame_size))
        length = float(m.rest.sum())
        self.physical_scale = m.tensor([length]*3+[1.]*9+[4.]*6+
            [length]*(3*(m.n-1))+[4.]*(3*(m.n-1))+[math.pi]*(m.n-1))
        self.physical_origin = torch.zeros_like(self.physical_scale)
        self.physical_origin[:3] = self.origin
        self.physical_origin[18:18+3*(m.n-1)] = self.origin.repeat(m.n-1)
        self.wrench_scale = m.tensor([self.maximum_thrust,.1,.1,.02])
        self.queue_scale = torch.cat((self.action_limit,m.tensor([math.pi])))
        # Current/previous physical state, previous dt, motors/wrench, delay
        # queue, previous action, time, task features. Angles stay unwrapped.
        self.critic_state_size = (2*len(self.physical_scale)+1+8+4*self.delay+self.action_size+1
                                  +FEATURE_SIZE)
        self.execution_backend = 'native_substep_guard_v1'
        self.reset()
        if not bool(self.active.all()):
            raise ValueError('Initial state is outside the task envelope')
        # Keep task guards outside solver replay. Reuse only the existing native
        # iteration acceleration, with unchanged convergence masks and limits.
        if self.device == 'cuda' and capture:
            m.compile_batched_physics()
            if _conditional_solver_enabled(conditional):
                from .conditional_graph import install
                self.conditional_slots = install(m)
                self.execution_backend = 'cuda_conditional_substep_guard_v1'
            else:
                from .iteration_graph import IterationGraphs
                m._batch_implicit_iteration = IterationGraphs(m._batch_implicit_iteration)
                self.execution_backend = 'cuda_iteration_substep_guard_v1'

    def _observe(self, state):
        def one(*values):
            s = ControlledState(*values)
            return (*self.model.tracking_pose(s),*self.model.cable_state(s))
        return torch.vmap(one)(*controlled_values(state))

    def _envelope(self, p, q, rotation):
        c = self.configuration
        return ((rotation[:,2,2] >= math.cos(math.radians(c.maximum_tilt_deg)))
            & (q[:,:,2].amin(-1) >= c.minimum_cable_height_m))

    @torch.no_grad()
    def reset(self):
        self.attempts = None
        self.state = ControlledState(*(v.clone() for v in self.seed))
        self.index, self.decision = 0,0
        self.queue = self.hold.expand(self.batch,self.delay,4).clone()
        self.previous_action = torch.zeros((self.batch,self.action_size),device=self.device)
        self.frames = torch.zeros((self.batch,self.samples,self.frame_size),device=self.device)
        self.critic_states = torch.zeros((self.batch,self.decisions+1,self.critic_state_size),device=self.device)
        self.actions = torch.zeros((self.batch,self.decisions,self.action_size),device=self.device)
        self.lengths = torch.zeros(self.batch,device=self.device,dtype=torch.long)
        self.transition_rewards = None
        self.active = torch.ones(self.batch,device=self.device,dtype=torch.bool)
        self.failed = torch.zeros_like(self.active)
        self.executed_samples = torch.zeros(self.batch,device=self.device,dtype=torch.long)
        self.workspace_breached = torch.zeros_like(self.active)
        self.history = {name: torch.zeros((self.batch,self.samples,*shape),device=self.device,dtype=torch.float64)
            for name,shape in dict(drone_positions=(3,),drone_velocities=(3,),drone_rotations=(3,3),
                drone_angular_velocities=(3,),cable_positions=(self.model.n,3),cable_velocities=(self.model.n,3),
                cable_material_angles_rad=(self.model.n-1,),
                motor_thrust_n=(4,),requested_wrench=(4,),command=(4,),
                applied_command=(4,),workspace_violation=()).items()}
        self.numerical_valid = torch.zeros((self.batch,self.samples),device=self.device,dtype=torch.bool)
        p,v,q,cv = self._observe(self.state)
        self.active &= self._envelope(p,q,self.state.rotation)
        inside = self.configuration.workspace.contains(p[:,None],self.origin.expand(self.batch,3))[:,0]
        if not bool(inside.all()): raise ValueError('Initial state is outside the workspace')
        hold = self.hold.expand(self.batch,4)
        self._record(p,v,q,cv,hold,hold,self.active,~inside)
        self._record_critic_state()
        result = self.frames,self.indices()
        rest = self.model.rest.detach().cpu().numpy()
        self.attempts = [ContinuousWhipAttempt(rest,self.configuration,workspace_origin=self.origin.cpu().numpy())
                         for _ in range(self.batch)]
        _,_,p,v = self._observe(self.state);p=p.cpu().numpy();v=v.cpu().numpy()
        for lane,a in enumerate(self.attempts):a.initialize(p[lane],v[lane])
        self.minimum_observed_cable_height = torch.as_tensor(p[:,:,2].min(axis=1),device=self.device).clone()
        self.maximum_observed_tilt = torch.rad2deg(torch.acos(self.state.rotation[:,2,2].clamp(-1,1)))
        self.physical_violation = ((self.minimum_observed_cable_height<self.configuration.minimum_cable_height_m)
            | (self.maximum_observed_tilt>self.configuration.maximum_tilt_deg))
        self.ledger = torch.zeros((self.batch,self.decisions),dtype=torch.float64,device=self.device)
        self.terminal_times = torch.zeros(self.batch,dtype=torch.float64,device=self.device)
        self.failure_reasons = ['']*self.batch;self.substep_records = [];self._record_critic_state()
        return result

    def indices(self):
        return torch.full((self.batch,),self.index,device=self.device,dtype=torch.long)

    def _task_critic_features(self):
        values = (np.stack([a.features() for a in self.attempts]) if self.attempts is not None
                  else np.zeros((self.batch,FEATURE_SIZE),np.float32))
        return torch.as_tensor(values,device=self.device)

    def _record_critic_state(self):
        s = self.state
        physical = torch.cat([v.flatten(1) for v in state_values(s)],-1)
        features = torch.cat((
            (physical-self.physical_origin)/self.physical_scale,
            (s.previous_state-self.physical_origin)/self.physical_scale,
            s.previous_step_s/self.model.controller_period_s,
            s.motor_thrust_n/self.maximum_thrust,s.held_wrench/self.wrench_scale,
            (self.queue/self.queue_scale).flatten(1),self.previous_action,
            physical.new_full((self.batch,1),self.index/self.time_samples)),-1).float()
        features = torch.cat((features,self._task_critic_features()),-1)
        self.critic_states[:,self.decision] = features

    def _record(self,p,v,q,cv,command,applied,valid,violation):
        state, i, m = self.state,self.index,self.model
        for name,value in dict(drone_positions=p,drone_velocities=v,drone_rotations=state.rotation,
            drone_angular_velocities=state.omega_body,cable_positions=q,cable_velocities=cv,
            cable_material_angles_rad=state.material_angles,
            motor_thrust_n=state.motor_thrust_n,requested_wrench=state.held_wrench,
            command=command,applied_command=applied,workspace_violation=violation).items():
            self.history[name][:,i].copy_(value)
        self.numerical_valid[:,i] = valid
        frame = torch.cat(((p-self.origin)/m.rest.sum(),state.rotation.flatten(1),v/4,
            state.omega_body/4,((q-q[:,:1])/m.rest.sum()).flatten(1),((cv-cv[:,:1])/4).flatten(1),
            state.motor_thrust_n/self.maximum_thrust,state.held_wrench/self.wrench_scale,self.previous_action,
            torch.full((self.batch,1),i/self.time_samples,device=self.device,dtype=torch.float64)),-1).float()
        self.frames[:,i] = frame

    def _result_finite(self,result):
        values = controlled_values(result)
        finite = torch.stack([torch.isfinite(v).flatten(1).all(-1) for v in values]).all(0)
        r = result.rotation
        finite &= ((r@r.transpose(-1,-2)-self.model.eye3).abs().amax((-1,-2)) <= 1e-8)
        finite &= (torch.linalg.det(r)-1).abs() <= 1e-8
        return finite

    def _substep_observer(self,state,elapsed):
        p,v,q,cv=self._observe(state)
        finite=self._result_finite(state)
        inside=self.configuration.workspace.contains(p[:,None],self.origin.expand(self.batch,3))[:,0]
        safe=self._envelope(p,q,state.rotation)
        # Workspace exits and physical-envelope violations are measured, not terminal.
        self.workspace_breached |= self._guard_active & ~inside
        observed=self._guard_active & finite
        self.physical_violation |= observed & ~safe
        height=q[:,:,2].amin(-1)
        tilt=torch.rad2deg(torch.acos(state.rotation[:,2,2].clamp(-1,1)))
        self.minimum_observed_cable_height=torch.where(observed,
            torch.minimum(self.minimum_observed_cable_height,height),self.minimum_observed_cable_height)
        self.maximum_observed_tilt=torch.where(observed,
            torch.maximum(self.maximum_observed_tilt,tilt),self.maximum_observed_tilt)
        # Only numerical corruption stops a lane.
        new=self._guard_active & ~finite
        self._guard_time=torch.where(new,torch.full_like(self._guard_time,elapsed),self._guard_time)
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
            self._last_finite=self.state
            self.model.step_unchecked(self.state,applied,self.controller,batch=True,observer=self._substep_observer)
            self.state=self._last_finite
            p,v,q,cv=self._observe(self.state)
            self.index+=1
            failed=was & ~self._guard_active
            self.active=self._guard_active
            self.failed |= failed
            violation=~self.configuration.workspace.contains(p[:,None],self.origin.expand(self.batch,3))[:,0]
            self._record(p,v,q,cv,command,applied,was & ~failed,violation)
            # One bounded device transfer per100Hz observation, not per physics substep.
            pp=q.cpu().numpy();vv=cv.cpu().numpy();active=was.cpu().numpy()
            tracking=p.cpu().numpy();times=self._guard_time.cpu().numpy();lost=failed.cpu().numpy()
            for lane in np.flatnonzero(active):
                a=self.attempts[lane]
                if lost[lane]:
                    elapsed=(self.index-1)*OBSERVATION_PERIOD_S+float(times[lane])
                    reward=a.fail(elapsed,'numerical_failure');self.failure_reasons[lane]='numerical_failure'
                    self.substep_records.append(dict(lane=int(lane),time_s=elapsed,reason='numerical_failure',
                        drone_position_m=p[lane].cpu().tolist()))
                else:reward=a.observe(self.index*OBSERVATION_PERIOD_S,pp[lane],vv[lane],drone_position=tracking[lane])
                self.ledger[lane,self.decision]+=reward
                if a.terminal is not None:
                    self.active[lane]=False;self.terminal_times[lane]=a.time
            self.executed_samples+=was.long()
            if not bool(self.active.any()):break
        self.lengths+=was_decision.long();self.decision+=1;self._record_critic_state()
        return self.frames,self.indices(),not bool(self.active.any())

    @torch.no_grad()
    def finish(self):
        if any(a.terminal is None for a in self.attempts):raise RuntimeError('Interrupted cohort cannot be scored')
        tensor=lambda values:torch.tensor(values,device=self.device,dtype=torch.float64)
        returns=tensor([a.total for a in self.attempts]);valid=~self.failed & ~self.physical_violation
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
        components.update(physical_violation=self.physical_violation.double(),
            numerically_valid=(~self.failed).double(),
            minimum_cable_height_m=self.minimum_observed_cable_height,
            maximum_tilt_deg=self.maximum_observed_tilt)
        score=Stage1Score(components['continuous_whip_credit'],valid,components)
        if not torch.allclose(self.ledger.sum(1),returns,atol=1e-12,rtol=1e-12):
            raise RuntimeError('Continuous-whip ledger does not telescope')
        self.transition_rewards=self.ledger.float()
        return returns.float(),valid,score,components['workspace_cost']

    def event_records(self):
        records=[a.record() for a in self.attempts]
        for lane,row in enumerate(records):
            row.update(contract=self.configuration.reward,
                physical_violation=bool(self.physical_violation[lane]),
                numerically_valid=not bool(self.failed[lane]),
                minimum_cable_height_m=float(self.minimum_observed_cable_height[lane]),
                maximum_tilt_deg=float(self.maximum_observed_tilt[lane]))
        return records

    def workspace_metrics(self):
        mask = self.numerical_valid[:,1:]
        radial,vertical = self.configuration.workspace.excess(self.history['drone_positions'],
            self.origin.expand(self.batch,3))
        radial,vertical = radial[:,1:],vertical[:,1:]
        return dict(boundary_breached=self.workspace_breached.clone(),
            executed_duration_s=self.terminal_times.clone(),
            workspace_mean_square_excess_m2=torch.where(mask,radial.square()+vertical.square(),0.).sum(1)/mask.sum(1).clamp_min(1),
            max_radial_violation_m=torch.where(mask,radial,0.).amax(1),
            max_vertical_violation_m=torch.where(mask,vertical,0.).amax(1),
            outside_duration_s=(((radial>0)|(vertical>0))&mask).sum(1)*OBSERVATION_PERIOD_S,
            # The reward's own workspace cost C, not a second implementation.
            workspace_cost=torch.tensor([a.cost for a in self.attempts],device=self.device,dtype=torch.float64))

    def replay_arrays(self, lane=0):
        # Valid numerical prefix only, including a finite envelope violation.
        valid = self.numerical_valid[lane].cpu().numpy()
        invalid = np.flatnonzero(~valid)
        count = int(invalid[0]) if len(invalid) else self.samples
        if count < 2: return None
        arrays = {key:value[lane,:count].cpu().numpy() for key,value in self.history.items()}
        arrays.update(time_s=self.time[:count].cpu().numpy(),state_valid=valid[:count],
                      cable_marker_indices=self.model.marker_indices,
                      action_interface=np.array(self.action_interface),
                      command_units=np.array('world acceleration m/s^2; yaw rad'),
                      acceleration_limit_mps2=self.action_limit.cpu().numpy(),
                      policy_rate_hz=np.array(self.settings['policy_rate_hz']))
        arrays['fold_event_json']=np.array(json.dumps(self.event_records()[lane]))
        arrays['terminal_time_s']=np.array(float(self.terminal_times[lane]))
        arrays['physical_violation']=np.array(bool(self.physical_violation[lane]))
        arrays['numerically_valid']=np.array(not bool(self.failed[lane]))
        arrays['task_valid']=np.array(not bool(self.failed[lane] or self.physical_violation[lane]))
        # A failing substep can have a finite saved position but no valid
        # reward measurement. Represent that row as missing, not zero credit.
        values=np.full((count,len(TRACE_FIELDS)),np.nan)
        trace=np.asarray(self.attempts[lane].trace);n=min(len(values),len(trace));values[:n]=trace[:n]
        arrays['continuous_whip_diagnostics']=values
        arrays['continuous_whip_fields']=np.array(TRACE_FIELDS)
        return arrays

    def policy_action(self,callback):
        width=self.settings.get('evaluation_environments',128)
        if self.batch!=1:raise ValueError('Deterministic fixed-state evaluation uses one lane')
        return callback(self.frames.repeat(width,1,1),self.indices().repeat(width))[:1]

    def provenance(self):
        return dict(evaluation_backend='native_tensor_substep_guard_v1',reward_mode=self.configuration.reward,
            evaluation_environments=self.batch,evaluation_policy_lanes=self.settings.get('evaluation_environments',128),
            unique_initial_states=1)


def make_environment(settings, snapshot, configuration, scene, **kwargs):
    return Stage1Environment(settings,snapshot,configuration,scene,**kwargs)
