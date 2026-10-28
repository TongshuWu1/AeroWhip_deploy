"""Configuration/bindings for the C++ four-motor actuator; no Python physics."""
from copy import deepcopy
import json
from pathlib import Path

import numpy as np
from aerowhip.paths import project_path
from aerowhip.physics.model import native_module


class Actuator:
    def __init__(self, config):
        self.config = (json.loads(project_path(config).read_text())
                       if isinstance(config, (str, Path)) else deepcopy(config))
        if self.config.get('allocation') != 'inverse_clip':
            raise ValueError('Actuator allocation must be inverse_clip; this is not a firmware mixer.')
        if self.config.get('calibration_status') not in ('uncalibrated', 'calibrated'):
            raise ValueError('Specify actuator calibration_status explicitly')
        required = ('motor_positions_com_m', 'yaw_torque_per_thrust_m', 'minimum_thrust_n',
                    'maximum_thrust_n', 'response_time_s', 'initial_thrust_n')
        for name in required:
            value = self.config.get(name)
            if value is None or not np.isfinite(np.asarray(value, dtype=float)).all():
                raise ValueError(f'Fill actuator {name}; no hardware value is assumed.')
        if 'controller_period_s' in self.config:
            raise ValueError('Set the single controller clock in controller settings, not in the motor configuration')
        self.native = native_module().Actuator(self.config)
        self.info = self.native.info()
