"""Thin viewer/controller binding to C++ state ownership and event scheduling.

The drone flies the current flight pipeline (`aerowhip.control.flight_pipeline`): once per output
step the host law turns the newest reference into an acceleration command, which reaches the
onboard firmware law (native FlightController) after the command delay."""
import math
from copy import deepcopy
import numpy as np

from aerowhip.control.flight_pipeline import FlightPipeline
from aerowhip.physics.actuator import Actuator
from aerowhip.physics.model import Model, native_module, pack_state, unpack_state
from aerowhip.physics.state import DroneState, ForceTelemetry, SimulationState, ActuatorTelemetry, quaternion_xyzw_to_rotation_matrix, rotation_matrix_to_quaternion_xyzw


class NativeRuntime:
    def __init__(self, configuration, *, model_config, device=None, command_delay_s=None):
        if device not in (None, 'cpu'):
            raise ValueError('The interactive C++ runtime runs on CPU.')
        self.configuration = configuration
        self.model = Model(model_config)
        self.controller = FlightPipeline(self.model)
        controller_period_s = self.controller.controller_period_s
        self.device, self.compiled, self.compilation_backend = 'cpu', True, 'native_cpu'
        self.dt = self.model.dt
        self.controller_period_s = controller_period_s
        self.delay_s = self.model.delay_s if command_delay_s is None else command_delay_s
        if not math.isclose(configuration.timestep_s, self.dt, abs_tol=1e-12):
            raise ValueError('Scene and physical model timesteps must match')
        if not np.allclose(configuration.platform.attachment_offset_body_m,
                           self.model.attachment_tracking, atol=1e-12, rtol=0):
            raise ValueError('Scene and physical model attachment positions must match')
        if not np.array_equal(configuration.platform.cable_segment_lengths_m,
                              self.model.config['cable']['marker_interval_lengths_m']):
            raise ValueError('Scene and physical model cable lengths must match')
        initial = configuration.initial_state
        R = quaternion_xyzw_to_rotation_matrix(initial.quaternion_body_to_world_xyzw)
        state = self.model.initial_state(initial.position_world_m, rotation=R,
            tracking_velocity=initial.linear_velocity_world_mps, omega_body=initial.angular_velocity_body_radps,
            cable_positions=initial.cable_node_positions_world_m, cable_velocities=initial.cable_node_velocities_world_mps)
        hold = np.zeros(13)
        hold[:3], hold[9] = initial.position_world_m, np.arctan2(R[1, 0], R[0, 0])
        self.initial_reference = hold
        self.initial_state = pack_state(state)
        thrust = self.controller.settle(self.initial_state, hold[9])
        actuator = deepcopy(self.model.config['actuator'])
        actuator['initial_thrust_n'] = [float(np.clip(thrust/4, actuator['minimum_thrust_n'][i], actuator['maximum_thrust_n'][i]))
                                        for i in range(4)]
        self.actuator = Actuator(actuator)
        self.reference = hold.copy()
        self.native = native_module().Runtime(self.model.native, self.initial_state,
                                              self.controller.command(hold[:3], hold[3:6], hold, 0.),
                                              controller_period_s, self.delay_s, self.actuator.native)

    def reset(self):
        self.native.reset()
        self.controller.settle(self.initial_state, self.initial_reference[9])
        self.reference = self.initial_reference.copy()

    def checkpoint(self):
        """Runtime continuation: plant state with BDF2 history, motors and command queues. The
        firmware's memory (attitude estimate, filters) and the host integral are not included.
        Physical telemetry alone is an initial condition, not a checkpoint.
        """
        return {'schema': 'aerowhip_runtime_bdf2_v1', 'model': deepcopy(self.model.config),
                'state': self.native.checkpoint().copy()}

    def restore(self, checkpoint):
        if (checkpoint.get('schema') != 'aerowhip_runtime_bdf2_v1'
                or checkpoint.get('model') != self.model.config):
            raise ValueError('Runtime checkpoint schema/model mismatch')
        self.native.restore(checkpoint['state'])

    @property
    def physics(self):
        return unpack_state(self.native.state())

    @property
    def actuation(self):
        values = self.native.actuation
        return None if values is None else ActuatorTelemetry(**{
            key: None if value is None else np.asarray(value, dtype=np.float64).copy()
            for key, value in values.items()})

    @property
    def requested_command(self):
        return self.native.requested

    @property
    def applied_command(self):
        return self.native.applied

    @property
    def command(self):
        return self.native.command

    @property
    def last_controller_update_times(self):
        return tuple(self.native.controller_updates)

    @property
    def last_requested_events(self):
        return tuple((t, np.asarray(c)) for t, c in self.native.requested_events)

    @property
    def last_applied_events(self):
        return tuple((t, np.asarray(c)) for t, c in self.native.applied_events)

    def step(self, command=None, *, command_events=()):
        """One output step. The newest reference given (a command, or the last timestamped event)
        replaces the held one; the host law runs once, at the start of the step, on the drone state."""
        if command is not None and len(command_events):
            raise ValueError('Supply a command or timestamped events, not both')
        events = [(self.native.time, command)] if command is not None else list(command_events)
        if events:
            self.reference = self.controller.prepare_reference(events[-1][1])
        observation = self.model.native.observation(self.native.state())
        sent = self.controller.command(observation[:3], observation[3:6], self.reference, self.dt)
        self.native.step([(self.native.time, sent)], self.controller.flight)

    def reference_events(self, source, start_s, end_s):
        if hasattr(source, 'reference_events'):
            return source.reference_events(start_s, end_s, self.controller.prepare_reference)
        return source.events(start_s, end_s)

    def snapshot(self, time_s, *, include_forces=False):
        values = self.native.snapshot(include_forces, self.controller.flight)
        n = self.model.n
        points = values[18:18+3*n].reshape(n, 3).copy()
        end = 18+6*n
        forces = (ForceTelemetry(values[end+3:end+6].copy(), values[end+6:end+9].copy(),
                                  values[end+9:end+12].copy()) if len(values) == end+12 else None)
        return SimulationState(time_s, DroneState(values[:3].copy(),
            rotation_matrix_to_quaternion_xyzw(values[6:15].reshape(3, 3)),
            values[3:6].copy(), values[15:18].copy(), values[end:end+3].copy()),
            points[0].copy(), points, values[18+3*n:end].reshape(n, 3).copy(),
            self.model.marker_indices.copy(), forces, self.actuation)
