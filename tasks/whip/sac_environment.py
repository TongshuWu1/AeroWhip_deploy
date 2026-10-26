"""Feedback task adapter over the unchanged C++ motor-driven plant.

Policy: configured velocity/acceleration rate; observations/reward: 100 Hz;
servo: model clock.
Episodes finish independently. Failed lanes are frozen only for batching and
never become valid replay continuation. Physics stays float64, networks float32.
"""
import math
import os
import struct
import sys
from contextlib import nullcontext
import numpy as np
import torch

from controllers.full_state_controller.tensor_controller import GeometricController
from simulation.tensor_model import TensorModel, ControlledState, controlled_values, state_values
from simulation.state import quaternion_xyzw_to_rotation_matrix
from .workspace import WorkspacePenalty
from .iteration_graph import IterationGraphs, PhysicsReplay
from training.configuration import action_limit_key


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
        if type(self) is Stage1Environment and configuration.matching.reward_mode in ('horizontal_fold_outreach_v2','retained_fold_delivery_v1','continuous_horizontal_whip_v1','continuous_horizontal_whip_soft_workspace_v2','continuous_horizontal_whip_distance_v3','continuous_horizontal_whip_full_horizon_v4', 'continuous_directional_whip_full_horizon_v5', 'continuous_directional_whip_four_second_v6'):
            raise ValueError('Use make_environment for the causal fold task')
        self.settings, self.configuration, self.scene = settings, configuration, scene
        self.device, self.batch = settings['device'], settings['parallel_environments']
        self.model = m = TensorModel(snapshot['model'], device=self.device, batched=True)
        if m.dt != .01 or scene.timestep_s != .01:
            raise ValueError('Stage 1 requires a 100 Hz observation clock')
        geometric = GeometricController(m)
        self.action_interface = settings['action_interface']
        limit_key = action_limit_key(settings)
        self.controller = (geometric.acceleration if self.action_interface == 'acceleration_direct_v1'
                           else geometric.velocity)
        if 'workspace_filter' in settings:
            raise ValueError('Workspace command filtering is retired; use explicit workspace_penalty')
        self.workspace_penalty = WorkspacePenalty(**settings['workspace_penalty'])
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
        self.stride = 100//settings['policy_rate_hz']
        self.samples = round((configuration.episode.duration_s+configuration.episode.post_hold_s)/.01)+1
        self.time_samples = self.samples-1
        self.decisions = (self.samples-1)//self.stride
        self.time = torch.arange(self.samples,device=self.device,dtype=torch.float64)*.01
        self.maximum_thrust = float(sum(m.config['actuator']['maximum_thrust_n']))
        self.action_size = 3
        self.action_limit = m.tensor(settings[limit_key])
        if self.action_interface == 'velocity_v1':
            self.velocity_limit = self.action_limit  # Retained velocity-only search/recovery tools.
        rotation = quaternion_xyzw_to_rotation_matrix(initial.quaternion_body_to_world_xyzw)
        yaw = float(np.arctan2(rotation[1,0],rotation[0,0]))
        self.hold = m.tensor([0.,0.,0.,yaw])
        self.delay = round(m.config['controller']['command_delay_s']/.01)
        self.frame_size = 18+6*m.n+8+self.action_size+1
        self.policy_encoder_options = dict(history_samples=401)
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
        # queue, previous action, time. Angles stay unwrapped (winding retained).
        self.critic_state_size = 2*len(self.physical_scale)+1+8+4*self.delay+self.action_size+1
        from .continuous_whip import FEATURE_SIZE
        from .horizontal_fold_environment import HorizontalFoldMetadata
        self.critic_state_size += FEATURE_SIZE + 2
        self.scorer = HorizontalFoldMetadata(m.rest,self.time,configuration,initial_position=self.origin)
        self.graph = None
        self.execution_backend = 'eager_v1'
        self.reset()
        if not bool(self.active.all()):
            raise ValueError('Initial state is outside the task envelope')
    def _observe(self, state):
        def one(*values):
            s = ControlledState(*values)
            return (*self.model.tracking_pose(s),*self.model.cable_state(s))
        return torch.vmap(one)(*controlled_values(state))

    def _envelope(self, p, q, rotation):
        c = self.configuration
        return ((rotation[:,2,2] >= math.cos(math.radians(c.returns.maximum_tilt_deg)))
            & (q[:,:,2].amin(-1) >= c.returns.minimum_cable_height_m))

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
        self.terminal_hit = torch.zeros_like(self.active)
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
        return self.frames,self.indices()

    def indices(self):
        return torch.full((self.batch,),self.index,device=self.device,dtype=torch.long)

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


    def scoring_prefix(self):
        return self.numerical_valid, torch.zeros(self.batch, device=self.device, dtype=torch.bool)


    def workspace_charges(self, observed):
        return self.workspace_penalty.charges(self.history['drone_positions'],
            self.origin.expand(self.batch,3),self.configuration.workspace,observed)

    def workspace_metrics(self):
        observed,_ = self.scoring_prefix()
        mask = observed[:,1:]
        radial,vertical = self.workspace_penalty.distances(self.history['drone_positions'],
            self.origin.expand(self.batch,3),self.configuration.workspace)
        radial,vertical = radial[:,1:],vertical[:,1:]
        return dict(boundary_breached=(((radial>0)|(vertical>0))&mask).any(1),
            executed_duration_s=mask.sum(1).to(torch.float64)*.01,
            workspace_mean_square_excess_m2=torch.where(mask,radial.square()+vertical.square(),0.).sum(1)/mask.sum(1).clamp_min(1),
            max_radial_violation_m=torch.where(mask,radial,0.).amax(1),
            max_vertical_violation_m=torch.where(mask,vertical,0.).amax(1),
            outside_duration_s=(((radial>0)|(vertical>0))&mask).sum(1)*.01,
            workspace_cost=self.workspace_charges(observed).sum(1))


    def replay_arrays(self, lane=0):
        # Valid numerical prefix only, including a finite envelope violation.
        valid = self.numerical_valid[lane].cpu().numpy()
        invalid = np.flatnonzero(~valid)
        count = int(invalid[0]) if len(invalid) else self.samples
        if count < 2: return None
        arrays = {key:value[lane,:count].cpu().numpy() for key,value in self.history.items()}
        arrays.update(time_s=self.time[:count].cpu().numpy(),state_valid=valid[:count],
                      cable_marker_indices=self.model.marker_indices)
        if self.action_interface == 'acceleration_direct_v1':
            arrays.update(action_interface=np.array(self.action_interface),
                          command_units=np.array('world acceleration m/s^2; yaw rad'),
                          acceleration_limit_mps2=self.action_limit.cpu().numpy(),
                          policy_rate_hz=np.array(self.settings['policy_rate_hz']))
        return arrays


def make_environment(settings, snapshot, configuration, scene, **kwargs):
    from .continuous_whip import SHORT_MODE
    from .continuous_whip_environment import ContinuousWhipEnvironment
    if configuration.matching.reward_mode != SHORT_MODE:
        raise ValueError('Retired training task; use the current four-second Stage 1 preset')
    return ContinuousWhipEnvironment(settings,snapshot,configuration,scene,**kwargs)
