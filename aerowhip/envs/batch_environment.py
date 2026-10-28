"""Batched environment over the native C++ twist-free plant (engine v2) and its flight controller.

Policy: normalized world acceleration commands (gravity excluded) at policy_rate_hz, held
between decisions. Each lane runs the real flight pipeline natively (firmware thrust vector,
tilt/voltage-scaled thrust map, attitude loop, command delay); lanes step in parallel on the CPU
(`BatchRuntime`, OpenMP) and are bit-identical to a single `Runtime`. Observations and task reward
at 100 Hz. Episodes finish independently. A lane the solver rejects stops at that step; its last
finite state is never scored or continued. Physical-validity violations (tilt, cable height) are
recorded at the 100 Hz samples and, with `terminate_on_violation`, end the lane with the task's
termination cost; workspace exits are recorded and never stop a lane. Rewards come from a Task
(envs.task). Physics float64 on the CPU, networks float32 on `device`.

The actor observes through the OptiTrack model (envs.sensing): the frame it sees carries the
delayed drone pose and the live cable filter's marker positions, velocities and confidences from
the delayed, partly hidden marker cloud, plus the task's target when it has one; the critic and
the history keep the true state. The attribute layout (frames, critic_states, history, ledger, ...)
is otherwise that of the earlier tensor environment, so the learner and the evaluation tools are
unchanged. The frame's critic-only slots for the motor/wrench state hold the four motor thrusts
and the applied (delayed) command.

Domain randomisation (`environment.randomisation`, when `randomised`): every reset draws, per
lane, the battery voltage and a constant world acceleration bias (native per-lane inputs) and,
per group of lanes, a model variant (body drag, bending damping, motor response, command delay);
one native batch runtime per group is built for the cohort. Reference-state starts: a fraction of
the lanes start from a recorded whip state (the task's curriculum reference bank) translated to
the scene origin, the rest from hover; a reference state the cable filter cannot label from its
first frame, or that the engine rejects, falls back to hover and is counted.
"""
from copy import deepcopy
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
from .observation import actor_columns, frame_row, frame_size
from .sensing import OptiTrackSensor
from .workspace import CylinderWorkspace

ACTION_INTERFACE = 'acceleration_direct_v1'
COMMAND_WIDTH = 13      # position(3) velocity(3) acceleration(3) yaw ... as the native flight controller reads it


class RuntimeGroup:
    """Several native batch runtimes (one model variant each) presented as one batch; lane
    order is the concatenation of the runtimes' lanes."""

    def __init__(self, runtimes):
        self.runtimes = list(runtimes)
        self.sizes = [r.size for r in self.runtimes]
        self.bounds = np.cumsum([0, *self.sizes])

    def _split(self, rows):
        return [rows[a:b] for a, b in zip(self.bounds[:-1], self.bounds[1:])]

    def reinitialize(self, states):
        for r, part in zip(self.runtimes, self._split(states)):
            r.reinitialize(part)

    def set_voltage(self, voltage):
        for r, part in zip(self.runtimes, self._split(voltage)):
            r.set_voltage(part)

    def set_bias(self, bias):
        for r, part in zip(self.runtimes, self._split(bias)):
            r.set_bias(part)

    def step(self, commands):
        for r, part in zip(self.runtimes, self._split(commands)):
            r.step(part)

    def observations(self):
        return np.concatenate([r.observations() for r in self.runtimes])

    def motors(self):
        return np.concatenate([r.motors() for r in self.runtimes])

    @property
    def failed(self):
        return [f for r in self.runtimes for f in r.failed]

    @property
    def errors(self):
        return [e for r in self.runtimes for e in r.errors]


class BatchEnvironment:
    @torch.no_grad()
    def __init__(self, config, task_config, model, scene, *, device, lanes, actor_batch=1,
                 firmware=None, battery_voltage=None, sensing_seed=0, randomised=False, randomisation_seed=None,
                 reference_states=True, lane_offset=0, total_lanes=None, **unused):
        """config: EnvironmentConfig; task_config: typed task settings; model: resolved model dict.

        actor_batch is the actor batch layout used by policy_action() for one-lane evaluation.
        battery_voltage: per-lane voltages (lanes,) or one value; default the firmware reference
        (ignored when randomised draws voltages).
        sensing_seed: the sensing model's random draws (dropout, noise, latency) per lane are a
        function of (sensing_seed, lane), so a lane's observations do not depend on the other lanes.
        randomised: draw the model, voltage and bias per reset from config.randomisation (a
        function of (randomisation_seed, cohort)); reference_states: start the task's reference
        fraction of lanes from its reference bank (training only).
        lane_offset, total_lanes: this batch holds lanes lane_offset .. lane_offset+lanes-1 of a cohort of
        total_lanes (a slice run in one worker process of envs.parallel_environment). Every random draw
        of a lane is a function of the seeds, the cohort and its global lane number, and model groups
        are formed over the whole cohort, so a lane is bit-identical however the cohort is split.
        `unused` accepts the tensor environment's GPU options (capture, conditional); ignored.
        """
        self.config, self.scene = config, scene
        self.device, self.batch, self.actor_batch = device, lanes, actor_batch
        self.lane_offset = int(lane_offset)
        self.total_lanes = lanes if total_lanes is None else int(total_lanes)
        if not 0 <= self.lane_offset <= self.total_lanes-lanes:
            raise ValueError('The lane slice must lie inside the cohort')
        self.model_config = model
        self.native = native_module()
        self.model = m = Model(model, load_weights=False)
        if abs(m.dt-OBSERVATION_PERIOD_S) > 1e-12 or abs(scene.timestep_s-OBSERVATION_PERIOD_S) > 1e-12:
            raise ValueError('Environments require a 100 Hz observation clock')
        if m.n != 11:
            raise ValueError('The tasks measure a ten-link (eleven-node) cable')
        self.nodes = n = m.n
        rest = np.asarray(m.rest, float)
        self.task = make_task(task_config, config, rest)
        self.workspace = config.workspace
        self.stage = None
        if hasattr(self.task, 'set_stage'):
            self.set_curriculum_stage(self.task.stage)
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
        # 100 Hz record of the commands still in flight (history and critic input): the delay rounded up
        # to whole samples. The runtime itself applies the exact delay, including randomised ones.
        self.delay = math.ceil(model['controller']['command_delay_s']/OBSERVATION_PERIOD_S-1e-9)
        if list(m.marker_indices) != list(range(n)) or n != MARKERS+1:
            raise ValueError('The sensing model expects the attachment node and the ten marker nodes')
        # Frame layout and actor columns: envs.observation (shared with the flight node).
        self.confidence_offset = 18+6*n
        self.frame_size = frame_size(n)
        self.actor_columns = actor_columns(n)
        self.sensing = config.sensing
        arc = np.cumsum(model['cable']['marker_interval_lengths_m'])
        attachment = np.asarray(model['attachment']['position_tracking_m'], float)
        self.sensors = [OptiTrackSensor(config.sensing, arc, attachment,
                                        np.random.default_rng([int(sensing_seed), self.lane_offset+lane]))
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
        self.weight = (model['drone']['mass_kg']+model['cable']['total_mass_kg'])*-model['simulation']['gravity_world_m_s2'][2]
        self.randomisation = config.randomisation
        self.randomised = bool(randomised) and self.randomisation.active
        self.randomisation_rng_seed = int(sensing_seed)+1 if randomisation_seed is None else int(randomisation_seed)
        self.cohort = 0
        self.nominal_voltage = self.firmware['reference_voltage_v'] if battery_voltage is None else battery_voltage
        self.voltage = np.broadcast_to(np.asarray(self.nominal_voltage, float), (lanes,)).copy()
        self.bias = np.zeros((lanes, 3))
        self.variants = []            # per lane: index of its model variant in self.variant_models
        self.variant_models = [model]
        self.variant_draws = []
        self.reference = None
        curriculum = getattr(task_config, 'curriculum', None)
        if reference_states and curriculum is not None and curriculum.reference_state_fraction > 0:
            self.reference = load_reference_bank(curriculum.reference_bank, self.seed.shape[0])
            self.reference_fraction = curriculum.reference_state_fraction
        self.reference_lanes = np.zeros(lanes, bool)
        self.reference_fallbacks = 0
        self._build_runtime([model], [lanes])
        self.reset()
        if not bool(self.active.all()):
            raise ValueError('Initial state is outside the task envelope')

    # ------------------------------------------------------------------ construction
    def _hover_trim(self, model, voltage):
        """Vertical feed-forward (m/s²) making the delivered thrust equal the weight of drone and
        cable at this voltage, so zero action means hover (the host PID supplies it in flight)."""
        probe = self.native.FlightController(self.firmware, model, float(voltage))
        low, high = -5., 12.
        for _ in range(60):
            mid = .5*(low+high)
            command = np.zeros(COMMAND_WIDTH); command[8] = mid; command[9] = self.yaw
            if probe(self.seed, command)[0] < self.weight:
                low = mid
            else:
                high = mid
        return .5*(low+high)

    def _build_runtime(self, models, sizes):
        """One native batch runtime per model variant; lanes are assigned in order."""
        self.variant_models = list(models)
        self.variants = np.repeat(np.arange(len(models)), sizes)
        self.hover_trim = np.zeros((self.batch, 3))
        for lane in range(self.batch):
            self.hover_trim[lane, 2] = self._hover_trim(models[self.variants[lane]], self.voltage[lane])
        runtimes, controllers, start = [], [], 0
        for model, size in zip(models, sizes):
            native_model = self.model.native if model is self.model_config else Model(model, load_weights=False).native
            probe = self.native.FlightController(self.firmware, model, float(self.voltage[start]))
            hold_row = self._command(self.hover_trim[start:start+1])[0]
            actuator = {k: (list(v) if isinstance(v, (list, tuple)) else v) for k, v in model['actuator'].items()}
            thrust = float(probe(self.seed, hold_row)[0])
            low, high = actuator['minimum_thrust_n'], actuator['maximum_thrust_n']
            actuator['initial_thrust_n'] = [float(np.clip(thrust/4, low[i], high[i])) for i in range(4)]
            lane_controllers = [self.native.FlightController(self.firmware, model, float(self.voltage[i]))
                                for i in range(start, start+size)]
            controllers.extend(lane_controllers)
            runtimes.append(self.native.BatchRuntime(native_model, self.seed, hold_row, model['controller']['controller_period_s'],
                                                     model['controller']['command_delay_s'], self.native.Actuator(actuator),
                                                     lane_controllers))
            start += size
        self.controllers = controllers
        self.runtime = runtimes[0] if len(runtimes) == 1 else RuntimeGroup(runtimes)
        self.execution_backend = f'native_batch_openmp_{self.native.BatchRuntime.threads()}_threads'

    def _lanes(self):
        """Global lane numbers of this batch."""
        return self.lane_offset+np.arange(self.batch)

    def _draw(self):
        """This cohort's randomisation: per-lane voltage and bias (one generator per global lane),
        per-group model variants (groups split the whole cohort; one generator per group)."""
        r, seed = self.randomisation, self.randomisation_rng_seed
        lanes = [np.random.default_rng([seed, self.cohort, int(lane)]) for lane in self._lanes()]
        self.voltage = np.array([g.uniform(*r.battery_voltage_v) for g in lanes])
        self.bias = np.array([g.normal(0., 1., 3) for g in lanes])*np.asarray(r.acceleration_bias_sd_mps2)
        groups = min(r.groups, self.total_lanes)
        group_of = np.concatenate([np.full(len(part), k) for k, part in
                                   enumerate(np.array_split(np.arange(self.total_lanes), groups))])[self._lanes()]
        mine, sizes = np.unique(group_of, return_counts=True)
        models, draws = [], []
        for group in mine:
            rng = np.random.default_rng([seed, self.cohort, 1_000_003, int(group)])
            drag = math.exp(rng.uniform(*np.log(r.body_drag_scale))) if r.body_drag_scale[0] > 0 else 1.
            damping = math.exp(rng.uniform(*np.log(r.bending_damping_scale))) if r.bending_damping_scale[0] > 0 else 1.
            response = rng.uniform(*r.motor_response_s)
            delay = rng.uniform(*r.command_delay_s)
            model = deepcopy(self.model_config)
            model['drone']['body_drag_s_inv'] = [float(v*drag) for v in model['drone']['body_drag_s_inv']]
            model['cable']['Cb_n_m2_s'] = float(model['cable']['Cb_n_m2_s']*damping)
            model['actuator']['response_time_s'] = [float(response)]*4
            model['controller'] = dict(model['controller'], command_delay_s=float(delay))
            models.append(model)
            draws.append(dict(body_drag_scale=drag, bending_damping_scale=damping, motor_response_s=float(response),
                              command_delay_s=float(delay)))
        self.variant_draws = draws
        self._build_runtime(models, [int(n) for n in sizes])

    def set_curriculum_stage(self, stage):
        """Workspace radius and the task's target/threshold for the next cohorts (strike task)."""
        w = self.config.workspace
        self.workspace = CylinderWorkspace(stage.workspace_radius_m, w.height_lower_world_m, w.height_upper_world_m)
        self.stage = stage
        self.task.set_stage(stage, self.workspace)

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
        """The Observer estimate (deployment.observer) each sensed lane receives for this frame:
        OptiTrack model, latency, then the same estimator as in flight. None for other lanes."""
        p, v, R, w, q, cv = observation
        return [self.sensors[lane].observe(self.index, p[lane], R[lane], q[lane, 1:], cv[lane, 1:]) if lanes[lane] else None
                for lane in range(self.batch)]

    def _envelope(self, q, rotation):
        c = self.config
        return ((rotation[:, 2, 2] >= math.cos(math.radians(c.maximum_tilt_deg)))
                & (q[:, :, 2].min(-1) >= c.minimum_cable_height_m))

    def _inside(self, p):
        return self.workspace.contains(torch.as_tensor(p[:, None]), self.origin.cpu().expand(self.batch, 3))[:, 0].numpy()

    def _initial_states(self):
        """Per-lane initial state rows: the hover seed, or a reference-bank state for the
        reference fraction of the lanes (translated to the scene origin); one generator per global lane."""
        rows = np.tile(self.seed, (self.batch, 1))
        self.reference_lanes[:] = False
        if self.reference is None:
            return rows
        for lane, number in enumerate(self._lanes()):
            rng = np.random.default_rng([self.randomisation_rng_seed, self.cohort, 7, int(number)])
            if rng.random() < self.reference_fraction:
                pick = int(rng.integers(0, len(self.reference['states'])))
                rows[lane] = translate_state(self.reference['states'][pick], self.origin_np, self.nodes-1)
                self.reference_lanes[lane] = True
        return rows

    def _start(self, rows):
        """Reinitialize the lanes; a reference state the engine rejects or the cable filter cannot
        label falls back to the hover seed (counted). Returns the observation and the sensed frames."""
        self.runtime.reinitialize(rows)
        self.runtime.set_voltage(self.voltage)
        self.runtime.set_bias(self.bias)
        for sensor in self.sensors:
            sensor.reset()
        rejected = np.asarray(self.runtime.failed, bool)
        observation = self._observe()
        sensed = self._sense(observation, ~rejected)
        complete = np.array([s is not None and bool((s['confidence'] == 1.).all()) for s in sensed])
        fallback = self.reference_lanes & (rejected | ~complete)
        if fallback.any():
            self.reference_fallbacks += int(fallback.sum())
            rows = rows.copy(); rows[fallback] = self.seed
            self.reference_lanes &= ~fallback
            return self._start(rows)
        if rejected.any():
            raise ValueError(f'Initial state rejected: {[e for e in self.runtime.errors if e][:1]}')
        if not complete.all():
            raise ValueError('The cable filter must label the complete initial cable')
        return observation, sensed

    @torch.no_grad()
    def reset(self):
        self.cohort += 1
        if self.randomised:
            self._draw()
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
        observation, sensed = self._start(self._initial_states())
        self.sensing_latency_frames = np.array([s.latency for s in self.sensors])
        p, v, R, w, q, cv = observation
        envelope = self._envelope(q, R)
        inside = self._inside(p)
        if not inside[~self.reference_lanes].all():
            raise ValueError('Initial state is outside the workspace')
        self.active_np &= envelope
        hold = np.tile(self.hold.cpu().numpy(), (self.batch, 1))
        self.task.reset(dict(cable_positions=q, cable_velocities=cv, drone_positions=p,
                             workspace_origin=np.tile(self.origin_np, (self.batch, 1))))
        self._record((p, v, R, w, q, cv), sensed, hold, hold, self.active_np.copy(), ~inside)
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
        world = np.full((self.batch, MARKERS, 3), np.nan); confidence = np.zeros((self.batch, MARKERS))
        for lane, s in enumerate(sensed):
            if s is not None:
                world[lane], confidence[lane] = s['cable_positions'], s['confidence']
        for name, value in dict(drone_positions=p, drone_velocities=v, drone_rotations=R, drone_angular_velocities=w,
                                cable_positions=q, cable_velocities=cv, motor_thrust_n=motors,
                                requested_wrench=applied, command=command, applied_command=applied,
                                workspace_violation=violation, sensed_cable_positions=world,
                                sensed_confidence=confidence).items():
            h[name][:, i] = torch.as_tensor(np.asarray(value, float) if name != 'workspace_violation' else value)
        self.numerical_valid[:, i] = torch.as_tensor(valid)
        critic = np.concatenate((motors/self.maximum_thrust, applied/self.command_scale.cpu().numpy()), 1)
        previous = self.previous_action.double().cpu().numpy()
        targets = self.task.targets() if hasattr(self.task, 'targets') else [None]*self.batch
        frame = np.zeros((self.batch, self.frame_size))
        for lane, s in enumerate(sensed):
            if s is not None:
                frame[lane] = frame_row(s, origin=self.origin_np, length=self.length, previous_action=previous[lane],
                                        progress=i/self.time_samples, nodes=self.nodes, critic=critic[lane],
                                        target=targets[lane])
        self.frames[:, i] = torch.as_tensor(frame, device=self.device).float()

    def _end(self, lane, elapsed, reason, position):
        self.active_np[lane] = False
        self.terminal_times[lane] = self.task.end_time(lane)
        self.failure_reasons[lane] = reason
        self.failure_records.append(dict(lane=int(lane), time_s=elapsed, reason=reason,
                                         error=self.runtime.errors[lane] if reason == 'numerical_failure' else '',
                                         drone_position_m=position.tolist()))

    @torch.no_grad()
    def step(self, action):
        """One policy decision: the action held for `stride` 10 ms sub-steps; the cohort stops early
        when every lane has ended."""
        self.begin_decision(action)
        for _ in range(self.stride):
            self.substep()
            if not self.active_np.any():
                break
        return self.end_decision()

    # A decision in parts, so that envs.parallel_environment can step worker slices sub-step by
    # sub-step and stop all of them exactly when the whole cohort has ended.
    def begin_decision(self, action, *, cohort_active=None):
        running = self.active_np.any() if cohort_active is None else cohort_active
        if self.decision >= self.decisions or not running:
            raise RuntimeError('Cohort finished')
        if action.shape != (self.batch, 3) or not bool(torch.isfinite(action).all()) or bool((action.abs() > 1.000001).any()):
            raise ValueError('Expected normalized finite acceleration actions')
        action = action.detach().clamp(-1, 1)
        self._was_decision = self.active_np.copy()
        self.actions[:, self.decision] = torch.where(self.active[:, None], action, 0.)
        self.previous_action = torch.where(self.active[:, None], action, self.previous_action)
        self._action_np = action.double().cpu().numpy()
        acceleration = self._action_np*self.action_limit_np
        self._requested = np.concatenate((acceleration, np.full((self.batch, 1), self.yaw)), 1)   # policy part, as recorded
        self._sent = acceleration+self.hover_trim

    def substep(self):
        """One 10 ms sample of the current decision."""
        requested, sent, action_np = self._requested, self._sent, self._action_np
        terminate = self.config.terminate_on_violation
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
        elapsed = self.index*OBSERVATION_PERIOD_S
        for lane in np.flatnonzero(was):
            if lost[lane]:
                reward = self.task.fail(lane, elapsed)
                self._end(lane, elapsed, 'numerical_failure', p[lane])
            else:
                reward = self.task.observe(lane, elapsed, dict(cable_positions=q[lane], cable_velocities=cv[lane],
                                                               drone_positions=p[lane], action=action_np[lane]))
                if terminate and not safe[lane] and not self.task.done(lane):
                    reward += self.task.fail(lane, elapsed, 'envelope_violation')
                    self._end(lane, elapsed, 'envelope_violation', p[lane])
                elif self.task.done(lane):
                    self.active_np[lane] = False
                    self.terminal_times[lane] = self.task.end_time(lane)
            self.ledger[lane, self.decision] += reward
        self.executed_np += was
        self._observation = observation

    def end_decision(self):
        self.lengths += torch.as_tensor(self._was_decision, device=self.device).long()
        self.decision += 1
        self._record_critic_state(self._observation)
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
            maximum_tilt_deg=tensor(self.maximum_observed_tilt),
            reference_start=tensor(self.reference_lanes.astype(float)))
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
                       maximum_tilt_deg=float(self.maximum_observed_tilt[lane]),
                       reference_start=bool(self.reference_lanes[lane]), terminated=self.failure_reasons[lane],
                       battery_voltage_v=float(self.voltage[lane]), acceleration_bias_mps2=self.bias[lane].tolist(),
                       model_variant=self.variant_draws[self.variants[lane]] if self.variant_draws else None)
        return records

    def workspace_metrics(self):
        mask = self.numerical_valid[:, 1:]
        radial, vertical = self.workspace.excess(self.history['drone_positions'], self.origin.cpu().expand(self.batch, 3))
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
        stage = None if self.stage is None else dict(target_distance_m=self.stage.target_distance_m,
                                                      speed_threshold_mps=self.stage.speed_threshold_mps,
                                                      workspace_radius_m=self.stage.workspace_radius_m)
        return dict(evaluation_backend='native_batch_flight_controller_v2', task=self.task.name,
                    environment_lanes=self.batch, evaluation_actor_batch=self.actor_batch,
                    unique_initial_states=int(1+self.reference_lanes.sum()),
                    firmware=self.firmware, battery_voltage_v=self.voltage.tolist(),
                    hover_trim_mps2=self.hover_trim.tolist(), randomised=self.randomised,
                    model_variants=list(self.variant_draws), curriculum_stage=stage,
                    reference_starts=int(self.reference_lanes.sum()), reference_fallbacks=self.reference_fallbacks,
                    sensing=dict(latency_frames=list(self.sensing.latency_frames), position_noise_m=self.sensing.position_noise_m,
                                 dropout_hazard=[list(row) for row in self.sensing.dropout_hazard],
                                 reappear_probability=self.sensing.reappear_probability,
                                 cable_filter='deployment.cable_filter.CableFilter'))


def translate_state(state, origin, nodes):
    """A reference-bank state (drone tracking origin at the bank's zero) moved so that its hover
    origin is at `origin`: the centre of mass and every free cable node (nodes = 10) are shifted."""
    s = np.array(state, float)
    s[:3] += origin
    s[18:18+3*nodes] += np.tile(origin, nodes)
    return s


def load_reference_bank(path, width):
    """Reference states written by scripts/build_reset_bank.py: 'states' (K, state width) relative to
    the recorded hover origin, with their provenance."""
    from aerowhip.paths import project_path
    bank = np.load(project_path(path))
    states = np.asarray(bank['states'], float)
    if states.ndim != 2 or states.shape[1] != width or not len(states):
        raise ValueError(f'{path}: expected reference states of width {width}')
    return dict(states=states, source=str(bank['source']) if 'source' in bank else '', count=len(states))


def make_batch_environment(experiment, snapshot, scene, *, lanes=None, device=None, workers=None, **kwargs):
    """Environment for a typed Experiment and its frozen snapshot (model dict): lanes in worker
    processes (envs.parallel_environment) when run.environment_workers > 0 and there is more than
    one lane, else in this process."""
    lanes = experiment.run.parallel_environments if lanes is None else lanes
    workers = experiment.run.environment_workers if workers is None else workers
    arguments = (experiment.environment, experiment.task, snapshot['model'], scene)
    options = dict(device=device or experiment.run.device, lanes=lanes, actor_batch=experiment.run.evaluation_actor_batch,
                   **kwargs)
    if workers > 0 and lanes > 1:
        from aerowhip.envs.parallel_environment import ParallelEnvironment
        return ParallelEnvironment(*arguments, workers=workers, **options)
    return BatchEnvironment(*arguments, **options)
