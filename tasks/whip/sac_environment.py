"""Feedback task adapter over the unchanged C++ motor-driven plant.

Policy: configured rate (current curriculum 10 Hz); observations/reward: 100 Hz;
servo: model clock.
Episodes finish independently. Failed lanes are frozen only for batching and
never become valid replay continuation. Physics stays float64, networks float32.
"""
import math
import os
import struct
from contextlib import nullcontext
import numpy as np
import torch

from controllers.full_state_controller.tensor_controller import GeometricController
from simulation.tensor_model import TensorModel, ControlledState, controlled_values, state_values
from simulation.state import quaternion_xyzw_to_rotation_matrix
from .learned_recovery import make_scorer
from .workspace import WorkspacePenalty
from .iteration_graph import IterationGraphs, PhysicsReplay


def _conditional_solver_enabled(conditional=None):
    """Windows64 CUDA-task default; explicit choice wins over the environment."""
    if conditional is not None:
        return conditional
    default = '1' if os.name == 'nt' and struct.calcsize('P') == 8 else '0'
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
        if m.dt != .01 or scene.timestep_s != .01:
            raise ValueError('Stage 1 requires a 100 Hz observation clock')
        geometric = GeometricController(m)
        self.controller = geometric.velocity
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
        self.time_samples = (round(configuration.recovery.whip_duration_s/.01)
            if configuration.recovery is not None else self.samples-1)
        self.decisions = (self.samples-1)//self.stride
        self.time = torch.arange(self.samples,device=self.device,dtype=torch.float64)*.01
        self.maximum_thrust = float(sum(m.config['actuator']['maximum_thrust_n']))
        self.action_size = 3
        self.velocity_limit = m.tensor(settings['velocity_limit_mps'])
        rotation = quaternion_xyzw_to_rotation_matrix(initial.quaternion_body_to_world_xyzw)
        yaw = float(np.arctan2(rotation[1,0],rotation[0,0]))
        self.hold = m.tensor([0.,0.,0.,yaw])
        self.delay = round(m.config['controller']['command_delay_s']/.01)
        self.frame_size = 18+6*m.n+8+self.action_size+1
        self.event_recovery = (configuration.recovery is not None
            and configuration.recovery.timing_mode=='after_first_geometric_hit_v1')
        self.policy_encoder_options = {}
        original_frame_size = self.frame_size
        # Deployable actor inputs are unchanged. Critic-only state is stored
        # separately at policy boundaries, never appended to actor histories.
        self.actor_columns = list(range(18))
        for offset in (18,18+3*m.n):
            self.actor_columns += [offset+3*int(i)+j for i in m.marker_indices for j in range(3)]
        self.actor_columns += list(range(18+6*m.n+8,self.frame_size))
        if self.event_recovery:
            if original_frame_size != 96:
                raise ValueError('Stage4 source-preserving adapter requires the registered11-node interface')
            self.policy_encoder_options = dict(history_samples=451,phase_columns=[96,97])
            self.frame_size += 2
        self.relative_target_input = settings.get('target_observation') == 'initial_relative_position_v1'
        if self.relative_target_input:
            # Append a static task input; legacy physical/history columns stay exact.
            self.policy_encoder_options['goal_columns'] = list(range(self.frame_size,self.frame_size+3))
            self.frame_size += 3
        length = float(m.rest.sum())
        self.physical_scale = m.tensor([length]*3+[1.]*9+[4.]*6+
            [length]*(3*(m.n-1))+[4.]*(3*(m.n-1))+[math.pi]*(m.n-1))
        self.physical_origin = torch.zeros_like(self.physical_scale)
        self.physical_origin[:3] = self.origin
        self.physical_origin[18:18+3*(m.n-1)] = self.origin.repeat(m.n-1)
        self.wrench_scale = m.tensor([self.maximum_thrust,.1,.1,.02])
        self.queue_scale = torch.cat((self.velocity_limit,m.tensor([math.pi])))
        # Current/previous physical state, previous dt, motors/wrench, delay
        # queue, previous action, time. Angles stay unwrapped (winding retained).
        self.critic_state_size = 2*len(self.physical_scale)+1+8+4*self.delay+self.action_size+1
        if 'target_curriculum' in settings or configuration.recovery is not None:
            self.critic_state_size += 2  # Target dimensions, not actor/deployment observations.
        if self.event_recovery:
            self.critic_state_size += 2  # Observable recovery phase and elapsed time.
        self.scorer = make_scorer(m.rest,self.time,configuration,initial_position=self.origin)
        if self.relative_target_input:
            self.goal_relative = ((self.scorer.center-self.origin)/length).float()
        self.graph = None
        self.execution_backend = 'eager_v1'
        self.reset()
        if not bool(self.active.all()):
            raise ValueError('Initial state is outside the task envelope')
        if self.device == 'cuda' and capture:
            conditional = _conditional_solver_enabled(conditional)
            self.execution_backend = ('cuda_conditional_v1' if conditional
                                      else 'cuda_host_iterations_v1')
            if progress: progress(dict(phase='compiling', execution_backend=self.execution_backend,
                                       message='Preparing batched C++ physics ('+self.execution_backend+')'))
            m.compile_batched_physics()
            # Both schedulers replay the same compiled slots and convergence
            # masks. Never lower the simulator's20-slot hard maximum.
            if conditional:
                from .conditional_graph import install
                self.conditional_slots=install(m)
            else:
                m._batch_implicit_iteration = IterationGraphs(m._batch_implicit_iteration)
            self.static_state = [v.clone() for v in self.seed]
            self.static_command = self.hold.expand(self.batch,4).clone()
            # Conditional replay explicitly warms both mask layouts before
            # capture; it need not run all20 already-converged slots at startup.
            with nullcontext() if conditional else m.cuda_graph_warmup():
                for _ in range(2):
                    result = m.step_unchecked(ControlledState(*self.static_state),self.static_command,
                                              self.controller,batch=True)
                    self._result_finite(result)
            torch.cuda.synchronize()
            def replay_physics():
                self.graph_output = m.step_unchecked(ControlledState(*self.static_state),
                    self.static_command,self.controller,batch=True)
                # Same numerical gates; the smaller graphs end before these.
                self.graph_finite = self._result_finite(self.graph_output)
            self.graph = PhysicsReplay(replay_physics)
            if progress:
                runtime = (dict(cuda_driver_api=self.conditional_slots.driver.driver_version,
                                nvrtc_version=list(self.conditional_slots.driver.nvrtc_version))
                           if conditional else {})
                progress(dict(phase='ready', execution_backend=self.execution_backend,
                              message='Physics ready ('+self.execution_backend+')', **runtime))
        self.reset()

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
        self.recovery_start = torch.full((self.batch,),-1,device=self.device,dtype=torch.long)
        self.recovery_hit = torch.zeros_like(self.active)
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
        if 'target_curriculum' in self.settings or self.configuration.recovery is not None:
            surface = self.configuration.strike_surface
            dimensions = features.new_tensor([surface.width_m, surface.height_m])/self.model.rest.sum()
            features = torch.cat((features, dimensions.expand(self.batch,2)), -1)
        if self.event_recovery:
            features = torch.cat((features,self._recovery_phase().float()),-1)
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
        frame = torch.cat((frame,self._recovery_phase().float()),-1) if self.event_recovery else frame
        if self.relative_target_input:
            frame = torch.cat((frame,self.goal_relative.expand(self.batch,3)),-1)
        self.frames[:,i] = frame

    def _recovery_phase(self):
        started = self.recovery_start>=0
        elapsed = torch.where(started,(self.index-self.recovery_start).clamp_min(0),0)
        return torch.stack((started.float(),elapsed.float()/(100*self.configuration.recovery.recovery_duration_s)),1)

    def _advance_recovery_clock(self):
        # Geometric crossing is observed at100Hz; commands change only at the
        # next policy decision and retain the original actuator/controller delay.
        from .strike import surface_events
        i = self.index
        pending = self.active & (self.recovery_start<0)
        hit = surface_events(self.history['cable_positions'][:,i-1:i+1],
            self.history['cable_velocities'][:,i-1:i+1],self.time[i-1:i+1],
            self.scorer.center,self.scorer.direction,self.configuration.strike_surface,
            self.scorer.speed)['geometric_hit'][:,0]
        deadline = round(self.configuration.recovery.whip_duration_s/.01)
        hit = pending & hit & (i<=deadline)
        begin = hit | (pending & (i>=deadline))
        self.recovery_start = torch.where(begin,i,self.recovery_start)
        self.recovery_hit |= hit
        self.frames[:,i,self.policy_encoder_options['phase_columns']] = self._recovery_phase().float()
        self.active &= ~((self.recovery_start>=0)
            & (i>=self.recovery_start+round(self.configuration.recovery.recovery_duration_s/.01)))

    @torch.no_grad()
    def _advance(self, applied):
        if self.graph is None:
            return self.model.step_unchecked(self.state,applied,self.controller,batch=True)
        for buffer,value in zip(self.static_state,controlled_values(self.state)): buffer.copy_(value)
        self.static_command.copy_(applied)
        self.graph.replay()
        return self.graph_output

    def _result_finite(self,result):
        values = controlled_values(result)
        finite = torch.stack([torch.isfinite(v).flatten(1).all(-1) for v in values]).all(0)
        r = result.rotation
        finite &= ((r@r.transpose(-1,-2)-self.model.eye3).abs().amax((-1,-2)) <= 1e-8)
        finite &= (torch.linalg.det(r)-1).abs() <= 1e-8
        return finite

    @torch.no_grad()
    def step(self, action):
        if self.decision >= self.decisions or not bool(self.active.any()):
            raise RuntimeError('Episode cohort has finished; reset before stepping')
        if action.shape != (self.batch,self.action_size) or not bool(torch.isfinite(action).all()) or bool((action.abs()>1.000001).any()):
            raise ValueError('Expected three finite normalized velocity requests in [-1,1]')
        action = action.detach().clamp(-1,1)
        self.actions[:,self.decision] = torch.where(self.active[:,None], action, 0.)
        self.previous_action = torch.where(self.active[:,None], action, self.previous_action)
        command = torch.cat((action.double()*self.velocity_limit,self.hold[3].expand(self.batch,1)),1)
        active_at_start = self.active.clone()
        for _ in range(self.stride):
            was_active = self.active.clone()
            applied = self.queue[:,0].clone() if self.delay else command
            if self.delay:
                shifted = torch.cat((self.queue[:,1:],command[:,None]),1)
                self.queue = torch.where(was_active[:,None,None], shifted, self.queue)
            result = self._advance(applied)
            values = controlled_values(result)
            finite = (self.graph_finite if self.graph is not None and result is self.graph_output
                      else self._result_finite(result))
            keep = was_active & finite
            self.state = ControlledState(*(torch.where(keep.reshape(-1,*([1]*(a.ndim-1))),a,b)
                for a,b in zip(values,controlled_values(self.state))))
            p,v,q,cv = self._observe(self.state)
            safe = self._envelope(p,q,self.state.rotation)
            inside = self.configuration.workspace.contains(p[:,None],self.origin.expand(self.batch,3))[:,0]
            violation = keep & ~inside
            self.workspace_breached |= violation
            self.active &= finite & safe
            self.failed |= was_active & ~self.active
            self.index += 1
            self._record(p,v,q,cv,command,applied,keep,violation)
            if self.event_recovery:
                self._advance_recovery_clock()
                if not bool(self.active.any()):
                    break
            if self.configuration.termination in ('first_qualified_hit_v1', 'first_geometric_hit_v1'):
                from .strike import surface_events
                i = self.index
                crossing = surface_events(self.history['cable_positions'][:,i-1:i+1],
                    self.history['cable_velocities'][:,i-1:i+1], self.time[i-1:i+1],
                    self.scorer.center, self.scorer.direction, self.configuration.strike_surface,
                    self.scorer.speed)['geometric_hit' if self.configuration.termination == 'first_geometric_hit_v1' else 'qualified'][:,0]
                reached = self.active & crossing
                self.terminal_hit |= reached
                self.active &= ~reached
                if not bool(self.active.any()):
                    break
        self.lengths += active_at_start.long()
        self.decision += 1
        self._record_critic_state()
        done = self.decision == self.decisions or not bool(self.active.any())
        return self.frames,self.indices(),done

    @torch.no_grad()
    def scoring_prefix(self):
        observed = self.numerical_valid
        hit = torch.zeros(self.batch,device=self.device,dtype=torch.bool)
        if self.configuration.termination in ('first_qualified_hit_v1', 'first_geometric_hit_v1'):
            from .strike import surface_events
            h = self.history
            events = surface_events(h['cable_positions'],h['cable_velocities'],self.time,
                self.scorer.center,self.scorer.direction,self.configuration.strike_surface,self.scorer.speed)
            safe = self._envelope(h['drone_positions'].flatten(0,1),
                h['cable_positions'].flatten(0,1),h['drone_rotations'].flatten(0,1)).reshape(self.batch,self.samples)
            prefix_safe = (observed & safe).long().cumprod(1).bool()
            qualified = events['geometric_hit' if self.configuration.termination == 'first_geometric_hit_v1' else 'qualified'][:,:-1] & prefix_safe[:,1:]
            hit = qualified.any(1)
            endpoint = qualified.long().argmax(1)+1
            observed = observed & (~hit[:,None] | (torch.arange(self.samples,device=self.device)[None] <= endpoint[:,None]))
        return observed,hit

    @torch.no_grad()
    def finish(self):
        if self.decision < self.decisions and bool(self.active.any()):
            raise RuntimeError('Do not score an interrupted cohort as completed episodes')
        h, c = self.history,self.configuration
        first_hit_mode = c.termination in ('first_qualified_hit_v1', 'first_geometric_hit_v1')
        observed,hit = self.scoring_prefix()
        terminal_samples = observed.long().sum(1) if first_hit_mode else None
        score = self.scorer(h['cable_positions'],h['cable_velocities'],observed,
            **({'terminal_samples': terminal_samples} if first_hit_mode else {}),
            **({'drone_positions':h['drone_positions'],'drone_velocities':h['drone_velocities']}
                if c.recovery is not None else {}),
            **({'recovery_start_samples':torch.where(self.recovery_start>=0,self.recovery_start,
                       round(c.recovery.whip_duration_s/.01)),
                 'recovery_hit':self.recovery_hit,'terminal_samples':observed.long().sum(1)}
                if self.event_recovery else {}))
        count = round(c.episode.duration_s/.01)
        acceleration = h['drone_velocities'][:,1:count+1].diff(dim=1,prepend=h['drone_velocities'][:,:1])/.01
        if first_hit_mode:
            acceleration = torch.where(observed[:,1:count+1,None], acceleration, 0.)
        cost = c.returns.acceleration_cost_weight*(acceleration/c.returns.acceleration_cost_scale_mps2).square().sum(-1).mean(-1)
        valid = (~self.failed | hit) & score.valid if first_hit_mode else ~self.failed & score.valid
        if not first_hit_mode and not self.event_recovery:
            valid &= self.numerical_valid.all(-1)
        workspace_charge = self.workspace_charges(observed)
        workspace_cost = workspace_charge.sum(1)
        time_charge = torch.zeros_like(workspace_charge)
        if c.matching.reward_mode in ('directed_hit_v1', 'geometric_hit_quality_v1'):
            horizon = c.episode.duration_s+c.episode.post_hold_s
            time_charge = observed[:, 1:].to(torch.float64)*(.01*c.matching.time_preference/horizon)
            score.components['time_cost'] = time_charge.sum(1)
        # A failure cancels positive event credit, never erases accrued workspace cost.
        returns = (torch.where(valid,score.score-cost,c.returns.failure_return)
                   -workspace_cost-time_charge.sum(1))
        score.components['workspace_cost'] = workspace_cost
        if c.recovery is not None:
            end=(self.recovery_start.clamp_min(0) if self.event_recovery else
                 torch.full_like(self.lengths,round(c.recovery.whip_duration_s/.01)))
            before=torch.arange(workspace_charge.shape[1],device=self.device)[None]<end[:,None]
            score.components['whip_workspace_cost']=torch.where(before,workspace_charge,0.).sum(1)
            score.components['recovery_workspace_cost']=torch.where(~before,workspace_charge,0.).sum(1)
        if first_hit_mode:
            score.components['executed_duration_s'] = (terminal_samples-1).to(torch.float64)*.01
            if c.termination == 'first_geometric_hit_v1':
                score.components['terminal_geometric_hit'] = hit.to(torch.float64)
                score.components['terminal_qualified_hit'] = score.components['strict_strike_success']
            else:
                score.components['terminal_qualified_hit'] = hit.to(torch.float64)
        if not bool(torch.isfinite(returns).all()):
            raise FloatingPointError('Nonfinite terminal reward')
        # Redistribute the SAME undiscounted objective, not a sum of repeated
        # event scores. Compute once per collected cohort, then store in replay.
        # The cost denominator remains the full maneuver duration, including
        # for short failed prefixes; the failure terminal correction overrides it.
        charge = c.returns.acceleration_cost_weight*(
            acceleration/c.returns.acceleration_cost_scale_mps2).square().sum(-1)/count
        cumulative_cost = torch.cat((torch.zeros_like(charge[:, :1]), charge.cumsum(1)), 1)
        cumulative_cost = torch.cat((cumulative_cost, cumulative_cost[:, -1:].expand(
            -1, self.samples-cumulative_cost.shape[1])), 1)
        running_charge = workspace_charge+time_charge
        cumulative_cost = cumulative_cost+torch.cat((torch.zeros_like(running_charge[:,:1]),running_charge.cumsum(1)),1)
        potential = (score.causal_best-cumulative_cost)[:, ::self.stride]
        rewards = potential.diff(dim=1)
        scoring_lengths = ((terminal_samples-1+self.stride-1)//self.stride).clamp_min(1) if first_hit_mode else self.lengths
        terminal = scoring_lengths-1
        if bool((terminal < 0).any()):
            raise ValueError('Completed episodes must contain a policy transition')
        # This also claws back all earlier credit when a lane fails. Boundaries
        # after the terminal step (including frozen lanes) are never replayed.
        final_reward = returns-potential.gather(1, terminal[:, None])[:, 0]
        rewards.scatter_(1, terminal[:, None], final_reward[:, None])
        active = torch.arange(self.decisions, device=self.device)[None] < scoring_lengths[:, None]
        rewards = rewards.masked_fill(~active, 0)
        if (not bool(torch.isfinite(rewards).all()) or not torch.allclose(
                rewards.sum(1), returns, rtol=1e-12, atol=1e-12)):
            raise FloatingPointError('Redistributed rewards do not preserve episode return')
        self.transition_rewards = rewards.float()
        return returns.float(),valid,score,cost

    def workspace_charges(self, observed):
        p,origin,c = self.history['drone_positions'],self.origin.expand(self.batch,3),self.configuration
        if c.recovery is None:
            return self.workspace_penalty.charges(p,origin,c.workspace,observed)
        if self.event_recovery:
            # Fixed reference denominators, never actual episode length: short
            # hits/padded suffixes cannot dilute accrued boundary cost.
            radial,vertical=self.workspace_penalty.distances(p,origin,c.workspace)
            distance=(radial[:,1:].square()+vertical[:,1:].square())/self.workspace_penalty.distance_scale_m**2
            start=torch.where(self.recovery_start>=0,self.recovery_start,
                             round(c.recovery.whip_duration_s/.01))
            before=torch.arange(1,self.samples,device=self.device)[None]<=start[:,None]
            denominator=torch.where(before,c.recovery.whip_duration_s*100,c.recovery.recovery_duration_s*100)
            return torch.where(observed[:,1:],distance*self.workspace_penalty.weight/denominator,0.)
        end=round(c.recovery.whip_duration_s/.01)
        # Preserve original /450 charge; normalize added /400 independently.
        return torch.cat((self.workspace_penalty.charges(p[:,:end+1],origin,c.workspace,observed[:,:end+1]),
            self.workspace_penalty.charges(p[:,end:],origin,c.workspace,observed[:,end:])),1)

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

    def boundary_gate_cost(self):
        if 'training_stage' not in self.settings:
            return self.workspace_metrics()['workspace_cost']
        # Gate uses each maneuver's EXECUTED duration, unlike the fixed-reference
        # reward integral. Quiet padded suffixes cannot dilute containment.
        return (self.workspace_penalty.weight/self.workspace_penalty.distance_scale_m**2
                *self.workspace_metrics()['workspace_mean_square_excess_m2'])

    def replay_arrays(self, lane=0):
        # Valid numerical prefix only, including a finite envelope violation.
        valid = self.numerical_valid[lane].cpu().numpy()
        invalid = np.flatnonzero(~valid)
        count = int(invalid[0]) if len(invalid) else self.samples
        if count < 2: return None
        arrays = {key:value[lane,:count].cpu().numpy() for key,value in self.history.items()}
        arrays.update(time_s=self.time[:count].cpu().numpy(),state_valid=valid[:count],
                      cable_marker_indices=self.model.marker_indices)
        return arrays
