"""Batched environment over the native C++ twist-free plant (engine v2) and its flight controller.

Policy: normalized world acceleration commands (gravity excluded) at policy_rate_hz, held
between decisions. Each lane runs the real flight pipeline natively (firmware thrust vector,
tilt/voltage-scaled thrust map, attitude loop, command delay); lanes step in parallel on the CPU
(`BatchRuntime`, OpenMP) and are bit-identical to a single `Runtime`. Observations and task reward
at 100 Hz. Episodes finish independently. A lane the solver rejects stops at that step; its last
finite state is never scored or continued. Physical-validity violations (tilt, cable height) and
workspace exits are recorded at the 100 Hz samples and do not stop a lane. Rewards come from a
Task (envs.task). Physics float64 on the CPU, networks float32 on `device`.

The actor observes through the OptiTrack model (envs.sensing): the frame it sees carries the
delayed drone pose and the live cable filter's marker positions, velocities and confidences from
the delayed, partly hidden marker cloud; the critic and the history keep the true state. The
attribute layout (frames, critic_states, history, ledger, ...) is otherwise that of the earlier
tensor environment, so the learner and the evaluation tools are unchanged. The frame's critic-only
slots for the motor/wrench state hold the four motor thrusts and the applied (delayed) command.
"""
import json
import math

import numpy as np
import torch

from aerowhip.deployment.cable_filter import MARKERS
from aerowhip.fitting.flight_drone import firmware_config
from aerowhip.physics.model import Model, native_module
from aerowhip.physics.state import quaternion_xyzw_to_rotation_matrix
from aerowhip.tasks import make_task
from .configuration import OBSERVATION_PERIOD_S
from .sensing import OptiTrackSensor

ACTION_INTERFACE = 'acceleration_direct_v1'
COMMAND_WIDTH = 13      # position(3) velocity(3) acceleration(3) yaw ... as the native flight controller reads it


class BatchEnvironment:
    @torch.no_grad()
    def __init__(self, config, task_config, model, scene, *, device, lanes, actor_batch=1,
                 firmware=None, battery_voltage=None, sensing_seed=0, **unused):
        """config: EnvironmentConfig; task_config: typed task settings; model: resolved model dict.

        actor_batch is the actor batch layout used by policy_action() for one-lane evaluation.
        battery_voltage: per-lane voltages (lanes,) or one value; default the firmware reference.
        sensing_seed: the sensing model's random draws (dropout, noise, latency) per lane are a
        function of (sensing_seed, lane), so a lane's observations do not depend on the other lanes.
        `unused` accepts the tensor environment's GPU options (capture, conditional); ignored.
        """
        self.config, self.scene = config, scene
        self.device, self.batch, self.actor_batch = device, lanes, actor_batch
        self.model_config = model
        self.native = native_module()
        self.model = m = Model(model, load_weights=False)
        if abs(m.dt-OBSERVATION_PERIOD_S) > 1e-12 or abs(scene.timestep_s-OBSERVATION_PERIOD_S) > 1e-12:
            raise ValueError('Environments require a 100 Hz observation clock')
        if m.n != 11:
            raise ValueError('The fold reward measures a ten-link (eleven-node) cable')
        self.nodes = n = m.n
        rest = np.asarray(m.rest, float)
        self.task = make_task(task_config, config, rest)
        self.action_interface = ACTION_INTERFACE
        self.firmware = firmware_config() if firmware is None else firmware
        initial = scene.initial_state
        rotation = quaternion_xyzw_to_rotation_matrix(initial.quaternion_body_to_world_xyzw)
        self.origin_np = np.asarray(initial.position_world_m, float)
        self.origin = torch.as_tensor(self.origin_np, dtype=torch.float64, device=device)
        seed = m.native.initial_state(self.origin_np, rotation, np.asarray(initial.linear_velocity_world_mps, float),
            np.asarray(initial.angular_velocity_body_radps, float),
            None if initial.cable_node_positions_world_m is None else np.asarray(initial.cable_node_positions_world_m, float),
            None if initial.cable_node_velocities_world_mps is None else np.asarray(initial.cable_node_velocities_world_mps, float))
        self.seed = np.asarray(seed, float)
        self.stride = round(1/(OBSERVATION_PERIOD_S*config.policy_rate_hz))
        self.samples = config.observations+1
        self.time_samples = self.samples-1
        if self.time_samples % self.stride:
            raise ValueError('The episode must contain a whole number of policy decisions')
        self.decisions = self.time_samples//self.stride
        self.time = torch.arange(self.samples, dtype=torch.float64)*OBSERVATION_PERIOD_S
        self.maximum_thrust = float(sum(model['actuator']['maximum_thrust_n']))
        self.action_size = 3
        self.action_limit_np = np.asarray(config.acceleration_limit_mps2, float)
        self.action_limit = torch.as_tensor(self.action_limit_np, dtype=torch.float64, device=device)
        yaw = float(np.arctan2(rotation[1, 0], rotation[0, 0]))
        self.yaw = yaw
        self.hold = torch.tensor([0., 0., 0., yaw], dtype=torch.float64, device=device)
        self.delay = round(model['controller']['command_delay_s']/OBSERVATION_PERIOD_S)
        if list(m.marker_indices) != list(range(n)) or n != MARKERS+1:
            raise ValueError('The sensing model expects the attachment node and the ten marker nodes')
        # Frame: sensed pose (18), sensed cable nodes and velocities (6n, node 0 = attachment),
        # marker confidences (10), motors and applied command (8, critic only), previous action, time.
        self.confidence_offset = 18+6*n
        self.frame_size = 18+6*n+MARKERS+8+self.action_size+1
        # Deployable actor inputs: the sensed pose, markers and confidences, previous action, time.
        self.actor_columns = list(range(18+6*n+MARKERS))+list(range(18+6*n+MARKERS+8, self.frame_size))
        self.sensing = config.sensing
        arc = np.cumsum(model['cable']['marker_interval_lengths_m'])
        attachment = np.asarray(model['attachment']['position_tracking_m'], float)
        self.sensors = [OptiTrackSensor(config.sensing, arc, attachment, np.random.default_rng([int(sensing_seed), lane]))
                        for lane in range(lanes)]
        self.length = float(rest.sum())
        # Critic physical vector: drone (18) then the free cable nodes and their velocities.
        self.physical_size = 18+6*(n-1)
        self.physical_scale = torch.as_tensor(np.r_[[self.length]*3, [1.]*9, [4.]*6, [self.length]*(3*(n-1)),
                                                    [4.]*(3*(n-1))], dtype=torch.float64, device=device)
        self.physical_origin = torch.zeros_like(self.physical_scale)
        self.physical_origin[:3] = self.origin
        self.physical_origin[18:18+3*(n-1)] = self.origin.repeat(n-1)
        self.command_scale = torch.cat((self.action_limit, torch.tensor([math.pi], dtype=torch.float64, device=device)))
        self.critic_state_size = (2*self.physical_size+4+4+4*self.delay+self.action_size+1
                                  +self.task.critic_feature_size)
        voltage = self.firmware['reference_voltage_v'] if battery_voltage is None else battery_voltage
        self.voltage = np.broadcast_to(np.asarray(voltage, float), (lanes,)).copy()
        # Zero action means hover: a vertical feed-forward acceleration per lane makes the delivered
        # thrust (thrust map at that lane's voltage) equal the weight of drone and cable. The real
        # host PID supplies this trim in flight.
        weight = (model['drone']['mass_kg']+model['cable']['total_mass_kg'])*-model['simulation']['gravity_world_m_s2'][2]
        self.hover_trim = np.zeros((lanes, 3))
        for lane in range(lanes):
            probe = self.native.FlightController(self.firmware, model, float(self.voltage[lane]))
            low, high = -5., 12.
            for _ in range(60):
                mid = .5*(low+high)
                command = np.zeros(COMMAND_WIDTH); command[8] = mid; command[9] = yaw
                if probe(self.seed, command)[0] < weight:
                    low = mid
                else:
                    high = mid
            self.hover_trim[lane, 2] = .5*(low+high)
        probe = self.native.FlightController(self.firmware, model, float(self.voltage[0]))
        hold_row = self._command(self.hover_trim[:1])[0]
        actuator = {k: (list(v) if isinstance(v, (list, tuple)) else v) for k, v in model['actuator'].items()}
        thrust = float(probe(self.seed, hold_row)[0])
        low, high = actuator['minimum_thrust_n'], actuator['maximum_thrust_n']
        actuator['initial_thrust_n'] = [float(np.clip(thrust/4, low[i], high[i])) for i in range(4)]
        self.controllers = [self.native.FlightController(self.firmware, model, float(self.voltage[i])) for i in range(lanes)]
        self.runtime = self.native.BatchRuntime(m.native, self.seed, hold_row, model['controller']['controller_period_s'],
                                                model['controller']['command_delay_s'], self.native.Actuator(actuator),
                                                self.controllers)
        self.execution_backend = f'native_batch_openmp_{self.native.BatchRuntime.threads()}_threads'
        self.reset()
        if not bool(self.active.all()):
            raise ValueError('Initial state is outside the task envelope')

    def _command(self, acceleration):
        command = np.zeros((len(acceleration), COMMAND_WIDTH))
        command[:, 6:9] = acceleration
        command[:, 9] = self.yaw
        return command

    def _observe(self):
        """Tracking position/velocity, rotation, body rate, cable nodes and velocities of every lane."""
        o = self.runtime.observations()
        n = self.nodes
        return (o[:, :3], o[:, 3:6], o[:, 6:15].reshape(-1, 3, 3), o[:, 15:18],
                o[:, 18:18+3*n].reshape(-1, n, 3), o[:, 18+3*n:18+6*n].reshape(-1, n, 3))

    def _sense(self, observation, lanes):
        """What OptiTrack and the cable filter give each lane for this frame: the sensed pose
        (18), node positions relative to the sensed attachment (n, 3), node velocities relative
        to it (n, 3) and the marker confidences (10). Lanes that are not sensed keep zeros."""
        p, v, R, w, q, cv = observation
        pose = np.zeros((self.batch, 18)); nodes = np.zeros((self.batch, self.nodes, 3))
        rates = np.zeros((self.batch, self.nodes, 3)); confidence = np.zeros((self.batch, MARKERS))
        world = np.full((self.batch, MARKERS, 3), np.nan); latency = np.zeros(self.batch, int)
        for lane in np.flatnonzero(lanes):
            s = self.sensors[lane].observe(self.index, p[lane], R[lane], v[lane], w[lane], q[lane, 1:], cv[lane, 1:])
            pose[lane] = np.r_[s['position'], s['rotation'].reshape(9), s['velocity'], s['angular_velocity']]
            nodes[lane, 1:] = s['cable_positions']-s['root']
            rates[lane, 1:] = s['cable_velocities']-s['root_velocity']
            confidence[lane], world[lane], latency[lane] = s['confidence'], s['cable_positions'], s['latency']
        return dict(pose=pose, nodes=nodes, rates=rates, confidence=confidence, world=world, latency=latency)

    def _envelope(self, q, rotation):
        c = self.config
        return ((rotation[:, 2, 2] >= math.cos(math.radians(c.maximum_tilt_deg)))
                & (q[:, :, 2].min(-1) >= c.minimum_cable_height_m))

    def _inside(self, p):
        return self.config.workspace.contains(torch.as_tensor(p[:, None]), self.origin.cpu().expand(self.batch, 3))[:, 0].numpy()

    @torch.no_grad()
    def reset(self):
        self.runtime.reinitialize(np.tile(self.seed, (self.batch, 1)))
        self.runtime.set_voltage(self.voltage)
        self.index, self.decision = 0, 0
        self.queue = np.zeros((self.batch, self.delay, 4))
        self.queue[:, :, 3] = self.yaw
        self.previous_action = torch.zeros((self.batch, self.action_size), device=self.device)
        self.frames = torch.zeros((self.batch, self.samples, self.frame_size), device=self.device)
        self.critic_states = torch.zeros((self.batch, self.decisions+1, self.critic_state_size), device=self.device)
        self.actions = torch.zeros((self.batch, self.decisions, self.action_size), device=self.device)
        self.lengths = torch.zeros(self.batch, device=self.device, dtype=torch.long)
        self.transition_rewards = None
        self.active_np = np.ones(self.batch, bool)
        self.failed_np = np.zeros(self.batch, bool)
        self.executed_np = np.zeros(self.batch, int)
        self.workspace_breached_np = np.zeros(self.batch, bool)
        shapes = dict(drone_positions=(3,), drone_velocities=(3,), drone_rotations=(3, 3), drone_angular_velocities=(3,),
                      cable_positions=(self.nodes, 3), cable_velocities=(self.nodes, 3), motor_thrust_n=(4,),
                      requested_wrench=(4,), command=(4,), applied_command=(4,), workspace_violation=(),
                      sensed_cable_positions=(MARKERS, 3), sensed_confidence=(MARKERS,))
        self.history = {name: torch.zeros((self.batch, self.samples, *shape), dtype=torch.float64)
                        for name, shape in shapes.items()}
        self.numerical_valid = torch.zeros((self.batch, self.samples), dtype=torch.bool)
        for sensor in self.sensors:
            sensor.reset()
        self.sensing_latency_frames = np.array([s.latency for s in self.sensors])
        p, v, R, w, q, cv = self._observe()
        envelope = self._envelope(q, R)
        inside = self._inside(p)
        if not inside.all():
            raise ValueError('Initial state is outside the workspace')
        self.active_np &= envelope
        hold = np.tile(self.hold.cpu().numpy(), (self.batch, 1))
        sensed = self._sense((p, v, R, w, q, cv), np.ones(self.batch, bool))
        if not (sensed['confidence'] == 1.).all():
            raise ValueError('The cable filter must label the complete initial cable')
        self._record((p, v, R, w, q, cv), sensed, hold, hold, self.active_np.copy(), ~inside)
        self.task.reset(dict(cable_positions=q, cable_velocities=cv, drone_positions=p,
                             workspace_origin=np.tile(self.origin_np, (self.batch, 1))))
        self.minimum_observed_cable_height = q[:, :, 2].min(-1).copy()
        self.maximum_observed_tilt = np.degrees(np.arccos(np.clip(R[:, 2, 2], -1, 1)))
        self.physical_violation_np = ((self.minimum_observed_cable_height < self.config.minimum_cable_height_m)
                                      | (self.maximum_observed_tilt > self.config.maximum_tilt_deg))
        self.ledger = torch.zeros((self.batch, self.decisions), dtype=torch.float64, device=self.device)
        self.terminal_times = np.zeros(self.batch)
        self.failure_reasons = ['']*self.batch
        self.failure_records = []
        self.previous_physical = None
        self._record_critic_state((p, v, R, w, q, cv))
        return self.frames, self.indices()

    # Boolean per-lane state is held in numpy (the lanes are scored one at a time on the CPU);
    # these views give the learner and tools the tensor attributes of the earlier environment.
    @property
    def active(self):
        return torch.as_tensor(self.active_np, device=self.device)

    @property
    def failed(self):
        return torch.as_tensor(self.failed_np, device=self.device)

    @property
    def physical_violation(self):
        return torch.as_tensor(self.physical_violation_np, device=self.device)

    @property
    def workspace_breached(self):
        return torch.as_tensor(self.workspace_breached_np, device=self.device)

    def indices(self):
        return torch.full((self.batch,), self.index, device=self.device, dtype=torch.long)

    def _physical(self, observation):
        p, v, R, w, q, cv = observation
        return np.concatenate([p, v, R.reshape(self.batch, 9), w, q[:, 1:].reshape(self.batch, -1),
                               cv[:, 1:].reshape(self.batch, -1)], 1)

    def _record_critic_state(self, observation):
        physical = torch.as_tensor(self._physical(observation), device=self.device)
        previous = physical if self.previous_physical is None else self.previous_physical
        motors = torch.as_tensor(self.runtime.motors(), device=self.device)
        applied = torch.as_tensor(self.history['applied_command'][:, self.index].numpy(), device=self.device)
        queue = torch.as_tensor(self.queue, device=self.device)
        features = torch.cat((
            (physical-self.physical_origin)/self.physical_scale,
            (previous-self.physical_origin)/self.physical_scale,
            motors/self.maximum_thrust, applied/self.command_scale,
            (queue/self.command_scale).flatten(1), self.previous_action.double(),
            physical.new_full((self.batch, 1), self.index/self.time_samples)), -1).float()
        features = torch.cat((features, torch.as_tensor(self.task.critic_features(), device=self.device)), -1)
        self.critic_states[:, self.decision] = features
        self.previous_physical = physical

    def _record(self, observation, sensed, command, applied, valid, violation):
        p, v, R, w, q, cv = observation
        i, h = self.index, self.history
        motors = self.runtime.motors()
        for name, value in dict(drone_positions=p, drone_velocities=v, drone_rotations=R, drone_angular_velocities=w,
                                cable_positions=q, cable_velocities=cv, motor_thrust_n=motors,
                                requested_wrench=applied, command=command, applied_command=applied,
                                workspace_violation=violation, sensed_cable_positions=sensed['world'],
                                sensed_confidence=sensed['confidence']).items():
            h[name][:, i] = torch.as_tensor(np.asarray(value, float) if name != 'workspace_violation' else value)
        self.numerical_valid[:, i] = torch.as_tensor(valid)
        scale = self.length
        pose = sensed['pose']
        frame = np.concatenate(((pose[:, :3]-self.origin_np)/scale, pose[:, 3:12], pose[:, 12:15]/4, pose[:, 15:18]/4,
                                (sensed['nodes']/scale).reshape(self.batch, -1), (sensed['rates']/4).reshape(self.batch, -1),
                                sensed['confidence'],
                                motors/self.maximum_thrust, applied/self.command_scale.cpu().numpy(),
                                self.previous_action.double().cpu().numpy(),
                                np.full((self.batch, 1), i/self.time_samples)), 1)
        self.frames[:, i] = torch.as_tensor(frame, device=self.device).float()

    @torch.no_grad()
    def step(self, action):
        if self.decision >= self.decisions or not self.active_np.any():
            raise RuntimeError('Cohort finished')
        if action.shape != (self.batch, 3) or not bool(torch.isfinite(action).all()) or bool((action.abs() > 1.000001).any()):
            raise ValueError('Expected normalized finite acceleration actions')
        action = action.detach().clamp(-1, 1)
        was_decision = self.active_np.copy()
        self.actions[:, self.decision] = torch.where(self.active[:, None], action, 0.)
        self.previous_action = torch.where(self.active[:, None], action, self.previous_action)
        acceleration = action.double().cpu().numpy()*self.action_limit_np
        requested = np.concatenate((acceleration, np.full((self.batch, 1), self.yaw)), 1)   # policy part, as recorded
        sent = acceleration+self.hover_trim
        for _ in range(self.stride):
            was = self.active_np.copy()
            applied = self.queue[:, 0].copy() if self.delay else requested
            if self.delay:
                self.queue[was] = np.concatenate((self.queue[was][:, 1:], requested[was][:, None]), 1)
            self.runtime.step(self._command(sent))
            observation = self._observe()
            p, v, R, w, q, cv = observation
            lost = np.asarray(self.runtime.failed, bool)
            failed = was & lost
            self.index += 1
            inside = self._inside(p)
            observed = was & ~lost
            safe = self._envelope(q, R)
            self.workspace_breached_np |= observed & ~inside
            self.physical_violation_np |= observed & ~safe
            self.minimum_observed_cable_height = np.where(observed, np.minimum(self.minimum_observed_cable_height,
                                                                              q[:, :, 2].min(-1)), self.minimum_observed_cable_height)
            self.maximum_observed_tilt = np.where(observed, np.maximum(self.maximum_observed_tilt,
                np.degrees(np.arccos(np.clip(R[:, 2, 2], -1, 1)))), self.maximum_observed_tilt)
            self.failed_np |= failed
            self.active_np = was & ~lost
            self._record(observation, self._sense(observation, observed), requested, applied, was & ~failed, ~inside)
            for lane in np.flatnonzero(was):
                if lost[lane]:
                    elapsed = self.index*OBSERVATION_PERIOD_S
                    reward = self.task.fail(lane, elapsed)
                    self.failure_reasons[lane] = 'numerical_failure'
                    self.failure_records.append(dict(lane=int(lane), time_s=elapsed, reason='numerical_failure',
                        error=self.runtime.errors[lane], drone_position_m=p[lane].tolist()))
                else:
                    reward = self.task.observe(lane, self.index*OBSERVATION_PERIOD_S,
                        dict(cable_positions=q[lane], cable_velocities=cv[lane], drone_positions=p[lane]))
                self.ledger[lane, self.decision] += reward
                if self.task.done(lane):
                    self.active_np[lane] = False
                    self.terminal_times[lane] = self.task.end_time(lane)
            self.executed_np += was
            if not self.active_np.any():
                break
        self.lengths += torch.as_tensor(was_decision, device=self.device).long()
        self.decision += 1
        self._record_critic_state(observation)
        return self.frames, self.indices(), not self.active_np.any()

    @torch.no_grad()
    def finish(self):
        """After all lanes ended: (returns, valid, components). The per-decision rewards
        (transition_rewards) sum to each lane's return."""
        if not all(self.task.done(lane) for lane in range(self.batch)):
            raise RuntimeError('Interrupted cohort cannot be scored')
        tensor = lambda values: torch.tensor(values, device=self.device, dtype=torch.float64)
        returns = tensor([self.task.episode_return(lane) for lane in range(self.batch)])
        valid = ~self.failed & ~self.physical_violation
        components = {key: tensor(values) for key, values in self.task.components().items()}
        components.update(net_episode_return=returns, failure=self.failed.double(),
            executed_duration_s=tensor(self.terminal_times), physical_violation=self.physical_violation.double(),
            numerically_valid=(~self.failed).double(), minimum_cable_height_m=tensor(self.minimum_observed_cable_height),
            maximum_tilt_deg=tensor(self.maximum_observed_tilt))
        if not torch.allclose(self.ledger.sum(1), returns, atol=1e-12, rtol=1e-12):
            raise RuntimeError('Reward ledger does not telescope to the episode returns')
        self.transition_rewards = self.ledger.float()
        return returns.float(), valid, components

    def episode_records(self):
        """Task record per lane plus the environment's validity facts."""
        records = [self.task.record(lane) for lane in range(self.batch)]
        for lane, row in enumerate(records):
            row.update(task=self.task.name, physical_violation=bool(self.physical_violation_np[lane]),
                       numerically_valid=not bool(self.failed_np[lane]),
                       minimum_cable_height_m=float(self.minimum_observed_cable_height[lane]),
                       maximum_tilt_deg=float(self.maximum_observed_tilt[lane]))
        return records

    def workspace_metrics(self):
        mask = self.numerical_valid[:, 1:]
        radial, vertical = self.config.workspace.excess(self.history['drone_positions'], self.origin.cpu().expand(self.batch, 3))
        radial, vertical = radial[:, 1:], vertical[:, 1:]
        dev = self.device
        return dict(boundary_breached=self.workspace_breached.clone(),
            executed_duration_s=torch.as_tensor(self.terminal_times, device=dev),
            workspace_mean_square_excess_m2=(torch.where(mask, radial.square()+vertical.square(), 0.).sum(1)/mask.sum(1).clamp_min(1)).to(dev),
            max_radial_violation_m=torch.where(mask, radial, 0.).amax(1).to(dev),
            max_vertical_violation_m=torch.where(mask, vertical, 0.).amax(1).to(dev),
            outside_duration_s=((((radial > 0) | (vertical > 0)) & mask).sum(1)*OBSERVATION_PERIOD_S).to(dev))

    def replay_arrays(self, lane=0):
        # Valid numerical prefix only, including a finite envelope violation.
        valid = self.numerical_valid[lane].numpy()
        invalid = np.flatnonzero(~valid)
        count = int(invalid[0]) if len(invalid) else self.samples
        if count < 2:
            return None
        arrays = {key: value[lane, :count].numpy() for key, value in self.history.items()}
        arrays.update(time_s=self.time[:count].numpy(), state_valid=valid[:count],
                      cable_marker_indices=self.model.marker_indices, action_interface=np.array(self.action_interface),
                      command_units=np.array('world acceleration m/s^2; yaw rad'),
                      acceleration_limit_mps2=self.action_limit_np, policy_rate_hz=np.array(self.config.policy_rate_hz))
        arrays['task_record_json'] = np.array(json.dumps(self.episode_records()[lane]))
        arrays['sensing_latency_frames'] = np.array(int(self.sensing_latency_frames[lane]))
        arrays['terminal_time_s'] = np.array(float(self.terminal_times[lane]))
        arrays['physical_violation'] = np.array(bool(self.physical_violation_np[lane]))
        arrays['numerically_valid'] = np.array(not bool(self.failed_np[lane]))
        arrays['task_valid'] = np.array(not bool(self.failed_np[lane] or self.physical_violation_np[lane]))
        # A failing step can have a finite saved position but no valid reward measurement:
        # that row is missing, not zero credit.
        fields = self.task.trace_fields
        values = np.full((count, len(fields)), np.nan)
        trace = self.task.trace(lane)
        k = min(len(values), len(trace))
        values[:k] = trace[:k]
        arrays['task_trace'] = values
        arrays['task_trace_fields'] = np.array(fields)
        return arrays

    def policy_action(self, callback):
        """Deterministic one-lane evaluation with the saved actor batch layout (not extra trials)."""
        if self.batch != 1:
            raise ValueError('Deterministic fixed-state evaluation uses one lane')
        return callback(self.frames.repeat(self.actor_batch, 1, 1), self.indices().repeat(self.actor_batch))[:1]

    def provenance(self):
        return dict(evaluation_backend='native_batch_flight_controller_v2', task=self.task.name,
                    environment_lanes=self.batch, evaluation_actor_batch=self.actor_batch, unique_initial_states=1,
                    firmware=self.firmware, battery_voltage_v=self.voltage.tolist(),
                    hover_trim_mps2=self.hover_trim.tolist(),
                    sensing=dict(latency_frames=list(self.sensing.latency_frames), position_noise_m=self.sensing.position_noise_m,
                                 dropout_hazard=[list(row) for row in self.sensing.dropout_hazard],
                                 reappear_probability=self.sensing.reappear_probability,
                                 cable_filter='deployment.cable_filter.CableFilter'))


def make_batch_environment(experiment, snapshot, scene, *, lanes=None, device=None, **kwargs):
    """Environment for a typed Experiment and its frozen snapshot (model dict)."""
    return BatchEnvironment(experiment.environment, experiment.task, snapshot['model'], scene,
        device=device or experiment.run.device,
        lanes=experiment.run.parallel_environments if lanes is None else lanes,
        actor_batch=experiment.run.evaluation_actor_batch, **kwargs)
