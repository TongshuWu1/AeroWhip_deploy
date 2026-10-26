"""Batched environment over the unchanged native C++ coupled drone-cable plant.

Policy: normalized world acceleration commands at policy_rate_hz; observations and
task reward: 100 Hz; controller: model clock with the model's command delay.
Episodes finish independently. A lane stops at its first numerically invalid
substep; that corrupt state is never scored or continued. Physical-validity
violations (tilt, cable height) and workspace exits are recorded but do not stop
the lane. Rewards come from a Task (envs.task). Physics float64, networks float32.
"""
import json
import math
import numpy as np
import torch

from aerowhip.control.tensor_controller import GeometricController
from aerowhip.physics.iteration_graph import accelerate
from aerowhip.physics.tensor_model import TensorModel, ControlledState, controlled_values, state_values
from aerowhip.physics.state import quaternion_xyzw_to_rotation_matrix
from aerowhip.tasks import make_task
from .configuration import OBSERVATION_PERIOD_S

ACTION_INTERFACE = 'acceleration_direct_v1'


class PlantEnvironment:
    @torch.no_grad()
    def __init__(self, config, task_config, model, scene, *, device, lanes, actor_batch=1,
                 capture=True, conditional=None):
        """config: EnvironmentConfig; task_config: typed task settings; model: resolved model dict.

        actor_batch is the actor batch layout used by policy_action() for one-lane evaluation.
        capture=True on CUDA replays the solver with CUDA graphs (same equations and limits).
        """
        self.config, self.scene = config, scene
        self.device, self.batch, self.actor_batch = device, lanes, actor_batch
        self.model = m = TensorModel(model, device=self.device, batched=True)
        if m.dt != OBSERVATION_PERIOD_S or scene.timestep_s != OBSERVATION_PERIOD_S:
            raise ValueError('Environments require a 100 Hz observation clock')
        if m.n != 11:
            raise ValueError('The fold reward measures a ten-link (eleven-node) cable')
        self.task = make_task(task_config, config, m.rest.detach().cpu().numpy())
        self.action_interface = ACTION_INTERFACE
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
        self.stride = round(1/(OBSERVATION_PERIOD_S*config.policy_rate_hz))
        self.samples = config.observations+1
        self.time_samples = self.samples-1
        if self.time_samples % self.stride:
            raise ValueError('The episode must contain a whole number of policy decisions')
        self.decisions = self.time_samples//self.stride
        self.time = torch.arange(self.samples,device=self.device,dtype=torch.float64)*OBSERVATION_PERIOD_S
        self.maximum_thrust = float(sum(m.config['actuator']['maximum_thrust_n']))
        self.action_size = 3
        self.action_limit = m.tensor(config.acceleration_limit_mps2)
        rotation = quaternion_xyzw_to_rotation_matrix(initial.quaternion_body_to_world_xyzw)
        yaw = float(np.arctan2(rotation[1,0],rotation[0,0]))
        self.hold = m.tensor([0.,0.,0.,yaw])
        self.delay = round(m.config['controller']['command_delay_s']/OBSERVATION_PERIOD_S)
        self.frame_size = 18+6*m.n+8+self.action_size+1
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
                                  +self.task.critic_feature_size)
        self.execution_backend = 'native_substep_guard_v1'
        self.reset()
        if not bool(self.active.all()):
            raise ValueError('Initial state is outside the task envelope')
        # Keep validity guards outside solver replay. Reuse only the existing native
        # iteration acceleration, with unchanged convergence masks and limits.
        if self.device == 'cuda' and capture:
            self.execution_backend, self.conditional_slots = accelerate(m, conditional)

    def _observe(self, state):
        def one(*values):
            s = ControlledState(*values)
            return (*self.model.tracking_pose(s),*self.model.cable_state(s))
        return torch.vmap(one)(*controlled_values(state))

    def _envelope(self, p, q, rotation):
        c = self.config
        return ((rotation[:,2,2] >= math.cos(math.radians(c.maximum_tilt_deg)))
            & (q[:,:,2].amin(-1) >= c.minimum_cable_height_m))

    @torch.no_grad()
    def reset(self):
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
        inside = self.config.workspace.contains(p[:,None],self.origin.expand(self.batch,3))[:,0]
        if not bool(inside.all()): raise ValueError('Initial state is outside the workspace')
        hold = self.hold.expand(self.batch,4)
        self._record(p,v,q,cv,hold,hold,self.active,~inside)
        self.task.reset(dict(cable_positions=q.cpu().numpy(),cable_velocities=cv.cpu().numpy(),
                             drone_positions=p.cpu().numpy(),
                             workspace_origin=self.origin.expand(self.batch,3).cpu().numpy()))
        self.minimum_observed_cable_height = q[:,:,2].amin(-1).clone()
        self.maximum_observed_tilt = torch.rad2deg(torch.acos(self.state.rotation[:,2,2].clamp(-1,1)))
        self.physical_violation = ((self.minimum_observed_cable_height<self.config.minimum_cable_height_m)
            | (self.maximum_observed_tilt>self.config.maximum_tilt_deg))
        self.ledger = torch.zeros((self.batch,self.decisions),dtype=torch.float64,device=self.device)
        self.terminal_times = torch.zeros(self.batch,dtype=torch.float64,device=self.device)
        self.failure_reasons = ['']*self.batch;self.failure_records = [];self._record_critic_state()
        return self.frames,self.indices()

    def indices(self):
        return torch.full((self.batch,),self.index,device=self.device,dtype=torch.long)

    def _task_critic_features(self):
        return torch.as_tensor(self.task.critic_features(),device=self.device)

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
        inside=self.config.workspace.contains(p[:,None],self.origin.expand(self.batch,3))[:,0]
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
            violation=~self.config.workspace.contains(p[:,None],self.origin.expand(self.batch,3))[:,0]
            self._record(p,v,q,cv,command,applied,was & ~failed,violation)
            # One bounded device transfer per100Hz observation, not per physics substep.
            pp=q.cpu().numpy();vv=cv.cpu().numpy();active=was.cpu().numpy()
            tracking=p.cpu().numpy();times=self._guard_time.cpu().numpy();lost=failed.cpu().numpy()
            for lane in np.flatnonzero(active):
                if lost[lane]:
                    elapsed=(self.index-1)*OBSERVATION_PERIOD_S+float(times[lane])
                    reward=self.task.fail(lane,elapsed);self.failure_reasons[lane]='numerical_failure'
                    self.failure_records.append(dict(lane=int(lane),time_s=elapsed,reason='numerical_failure',
                        drone_position_m=p[lane].cpu().tolist()))
                else:reward=self.task.observe(lane,self.index*OBSERVATION_PERIOD_S,
                    dict(cable_positions=pp[lane],cable_velocities=vv[lane],drone_positions=tracking[lane]))
                self.ledger[lane,self.decision]+=reward
                if self.task.done(lane):
                    self.active[lane]=False;self.terminal_times[lane]=self.task.end_time(lane)
            self.executed_samples+=was.long()
            if not bool(self.active.any()):break
        self.lengths+=was_decision.long();self.decision+=1;self._record_critic_state()
        return self.frames,self.indices(),not bool(self.active.any())

    @torch.no_grad()
    def finish(self):
        """After all lanes ended: (returns, valid, components). The per-decision rewards
        (transition_rewards) sum to each lane's return."""
        if not all(self.task.done(lane) for lane in range(self.batch)):
            raise RuntimeError('Interrupted cohort cannot be scored')
        tensor=lambda values:torch.tensor(values,device=self.device,dtype=torch.float64)
        returns=tensor([self.task.episode_return(lane) for lane in range(self.batch)])
        valid=~self.failed & ~self.physical_violation
        components={key:tensor(values) for key,values in self.task.components().items()}
        components.update(net_episode_return=returns,failure=self.failed.double(),
            executed_duration_s=self.terminal_times,physical_violation=self.physical_violation.double(),
            numerically_valid=(~self.failed).double(),minimum_cable_height_m=self.minimum_observed_cable_height,
            maximum_tilt_deg=self.maximum_observed_tilt)
        if not torch.allclose(self.ledger.sum(1),returns,atol=1e-12,rtol=1e-12):
            raise RuntimeError('Reward ledger does not telescope to the episode returns')
        self.transition_rewards=self.ledger.float()
        return returns.float(),valid,components

    def episode_records(self):
        """Task record per lane plus the environment's validity facts."""
        records=[self.task.record(lane) for lane in range(self.batch)]
        for lane,row in enumerate(records):
            row.update(task=self.task.name,
                physical_violation=bool(self.physical_violation[lane]),
                numerically_valid=not bool(self.failed[lane]),
                minimum_cable_height_m=float(self.minimum_observed_cable_height[lane]),
                maximum_tilt_deg=float(self.maximum_observed_tilt[lane]))
        return records

    def workspace_metrics(self):
        mask = self.numerical_valid[:,1:]
        radial,vertical = self.config.workspace.excess(self.history['drone_positions'],
            self.origin.expand(self.batch,3))
        radial,vertical = radial[:,1:],vertical[:,1:]
        return dict(boundary_breached=self.workspace_breached.clone(),
            executed_duration_s=self.terminal_times.clone(),
            workspace_mean_square_excess_m2=torch.where(mask,radial.square()+vertical.square(),0.).sum(1)/mask.sum(1).clamp_min(1),
            max_radial_violation_m=torch.where(mask,radial,0.).amax(1),
            max_vertical_violation_m=torch.where(mask,vertical,0.).amax(1),
            outside_duration_s=(((radial>0)|(vertical>0))&mask).sum(1)*OBSERVATION_PERIOD_S,
        )

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
                      policy_rate_hz=np.array(self.config.policy_rate_hz))
        arrays['task_record_json']=np.array(json.dumps(self.episode_records()[lane]))
        arrays['terminal_time_s']=np.array(float(self.terminal_times[lane]))
        arrays['physical_violation']=np.array(bool(self.physical_violation[lane]))
        arrays['numerically_valid']=np.array(not bool(self.failed[lane]))
        arrays['task_valid']=np.array(not bool(self.failed[lane] or self.physical_violation[lane]))
        # A failing substep can have a finite saved position but no valid
        # reward measurement. Represent that row as missing, not zero credit.
        fields=self.task.trace_fields
        values=np.full((count,len(fields)),np.nan)
        trace=self.task.trace(lane);n=min(len(values),len(trace));values[:n]=trace[:n]
        arrays['task_trace']=values
        arrays['task_trace_fields']=np.array(fields)
        return arrays

    def policy_action(self,callback):
        """Deterministic one-lane evaluation with the saved actor batch layout (not extra trials)."""
        if self.batch!=1:raise ValueError('Deterministic fixed-state evaluation uses one lane')
        return callback(self.frames.repeat(self.actor_batch,1,1),self.indices().repeat(self.actor_batch))[:1]

    def provenance(self):
        return dict(evaluation_backend='native_tensor_substep_guard_v1',task=self.task.name,
            environment_lanes=self.batch,evaluation_actor_batch=self.actor_batch,unique_initial_states=1)


def make_environment(experiment, snapshot, scene, *, lanes=None, device=None, **kwargs):
    """Environment for a typed Experiment and its frozen snapshot (model dict)."""
    return PlantEnvironment(experiment.environment, experiment.task, snapshot['model'], scene,
        device=device or experiment.run.device,
        lanes=experiment.run.parallel_environments if lanes is None else lanes,
        actor_batch=experiment.run.evaluation_actor_batch, **kwargs)
