"""Renderer-independent playback of sampled simulation states; no physics or reward."""

import math

import numpy as np

from simulation.state import DroneState, SimulationState, rotation_matrix_to_quaternion_xyzw


class StateReplay:
    def __init__(self, arrays):
        self.arrays = {key: np.asarray(value).copy() for key, value in arrays.items()}
        self.time = self.arrays['time_s']
        if (self.time.ndim != 1 or len(self.time) < 2 or not np.isfinite(self.time).all()
                or self.time[0] != 0 or not np.all(np.diff(self.time) > 0)):
            raise ValueError('Replay requires increasing finite sample times starting at zero')
        count = len(self.time)
        cable = self.arrays['cable_positions']
        if cable.ndim != 3 or cable.shape[0] != count or cable.shape[1] < 2 or cable.shape[2] != 3:
            raise ValueError('Replay cable positions must have shape (times,nodes,3)')
        shapes = dict(drone_positions=(count, 3), drone_velocities=(count, 3),
                      drone_rotations=(count, 3, 3), drone_angular_velocities=(count, 3),
                      cable_positions=cable.shape, cable_velocities=cable.shape)
        for name, shape in shapes.items():
            value = self.arrays[name]
            if value.shape != shape or not np.isfinite(value).all():
                raise ValueError(f'Invalid replay array: {name}')
        valid = self.arrays['state_valid']
        if valid.shape != (count,) or valid.dtype != np.bool_ or not valid.all():
            raise ValueError('Cannot present invalid/frozen numerical states as simulated motion')
        markers = self.arrays['cable_marker_indices']
        if (markers.ndim != 1 or not np.issubdtype(markers.dtype, np.integer) or len(markers) < 2
                or markers[0] != 0 or markers[-1] >= cable.shape[1] or not np.all(np.diff(markers) > 0)):
            raise ValueError('Invalid replay cable marker indices')
        rotations = self.arrays['drone_rotations']
        if (not np.allclose(rotations @ rotations.transpose(0, 2, 1), np.eye(3), atol=1e-8, rtol=0)
                or not np.allclose(np.linalg.det(rotations), 1, atol=1e-8, rtol=0)):
            raise ValueError('Replay contains invalid body rotations')
        self.speed = 1.
        self.reset()

    @property
    def duration_s(self):
        return float(self.time[-1])

    def reset(self):
        self.running = False
        self.seek(0)

    def seek(self, index):
        if not 0 <= index < len(self.time):
            raise IndexError('Replay index out of range')
        self.index = int(index)
        self._cursor_s = float(self.time[self.index])
        self._update_state()

    def _update_state(self):
        a, i = self.arrays, self.index
        acceleration = None if i == 0 else (a['drone_velocities'][i]-a['drone_velocities'][i-1])/(self.time[i]-self.time[i-1])
        self.state = SimulationState(float(self.time[i]),
            DroneState(a['drone_positions'][i], rotation_matrix_to_quaternion_xyzw(a['drone_rotations'][i]),
                       a['drone_velocities'][i], a['drone_angular_velocities'][i], acceleration),
            a['cable_positions'][i, 0], a['cable_positions'][i], a['cable_velocities'][i],
            a['cable_marker_indices'])

    def toggle_running(self):
        if not self.running and self.index == len(self.time)-1:
            self.seek(0)
        self.running = not self.running

    def step(self):
        self.running = False
        self.seek(min(self.index+1, len(self.time)-1))
        return self.state

    def advance(self, elapsed_s):
        if not math.isfinite(elapsed_s) or elapsed_s < 0 or not math.isfinite(self.speed) or self.speed <= 0:
            raise ValueError('Invalid playback elapsed time or speed')
        if not self.running:
            return 0
        previous = self.index
        self._cursor_s = min(self.duration_s, self._cursor_s+elapsed_s*self.speed)
        self.index = min(len(self.time)-1, int(np.searchsorted(self.time, self._cursor_s, side='right')-1))
        if self.index == len(self.time)-1:
            self.running = False
        if self.index != previous:
            self._update_state()
        return self.index-previous
