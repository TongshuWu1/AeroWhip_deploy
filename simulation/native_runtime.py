"""Thin viewer/controller binding to C++ state ownership and event scheduling."""
import math
from copy import deepcopy
import numpy as np

from controllers.full_state_controller.geometric_controller import GeometricController
from simulation.actuator import Actuator
from simulation.model import Model, native_module, pack_state, unpack_state
from simulation.state import (DroneState, ForceTelemetry, SimulationState, ActuatorTelemetry,
                              quaternion_xyzw_to_rotation_matrix, rotation_matrix_to_quaternion_xyzw)


class NativeRuntime:
    def __init__(self, configuration, *, model_config, device=None, command_delay_s=None):
        if device not in (None, 'cpu'):
            raise ValueError('The interactive C++ runtime runs on CPU; GPU tensors are a separate fitting/batch interface.')
        self.configuration = configuration
        self.model = Model(model_config)
        self.controller = GeometricController(self.model)
        self.actuator = Actuator(self.model.config['actuator'])
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
            cable_positions=initial.cable_node_positions_world_m, cable_velocities=initial.cable_node_velocities_world_mps,
            material_angles=initial.cable_material_angles_rad)
        hold = np.zeros(11)
        hold[:3], hold[9] = initial.position_world_m, np.arctan2(R[1, 0], R[0, 0])
        hold = self.controller.prepare_reference(hold)
        self.native = native_module().Runtime(self.model.native, pack_state(state), hold,
                                              controller_period_s, command_delay_s,
                                              self.actuator.native)

    def reset(self):
        self.native.reset()

    def checkpoint(self):
        """Exact runtime continuation, including BDF2 history, motors and queues.

        Restore with the identical model and stateless geometric controller.
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
        if command is not None and len(command_events):
            raise ValueError('Supply a command or timestamped events, not both')
        events = [(self.native.time, command)] if command is not None else list(command_events)
        events = [(t,self.controller.prepare_reference(c,None)) for t,c in events]
        self.native.step(events, self.controller)

    def reference_events(self, source, start_s, end_s):
        if hasattr(source, 'reference_events'):
            return source.reference_events(start_s, end_s, self.controller.prepare_reference)
        return source.events(start_s, end_s)

    def snapshot(self, time_s, *, include_forces=False):
        values = self.native.snapshot(include_forces, self.controller)
        n = self.model.n
        points = values[18:18+3*n].reshape(n, 3).copy()
        end = 18+6*n
        forces = (ForceTelemetry(values[end+3:end+6].copy(), values[end+6:end+9].copy(),
                                  values[end+9:end+12].copy()) if len(values) == end+12 else None)
        return SimulationState(time_s, DroneState(values[:3].copy(),
            rotation_matrix_to_quaternion_xyzw(values[6:15].reshape(3, 3)),
            values[3:6].copy(), values[15:18].copy(), values[end:end+3].copy()),
            points[0].copy(), points, values[18+3*n:end].reshape(n, 3).copy(),
            self.model.marker_indices.copy(), forces, self.actuation,
            self.physics.material_angles.copy())
