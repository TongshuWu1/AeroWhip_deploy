"""Native 100 Hz P/V/A/yaw commands, not predicted motion.

World position refers to the top-of-drone tracking origin, not the COM.
Acceleration is kinematic feedforward (gravity is added by the inner controller).
Yaw is in radians; omega is body-frame rad/s and must currently be zero.
"""

import numpy as np


COLUMNS = ('time_s', 'x', 'y', 'z', 'vx', 'vy', 'vz', 'ax', 'ay', 'az',
           'yaw', 'omega_x', 'omega_y', 'omega_z')


class FullStateTrajectory:
    timestep_s = .01

    def __init__(self, samples):
        values = np.array(samples, dtype=np.float64, copy=True)
        if values.ndim != 2 or values.shape[1] != len(COLUMNS) or len(values) < 2:
            raise ValueError('Expected at least two rows of time, P/V/A, yaw and body angular velocity')
        if not np.isfinite(values).all():
            raise ValueError('Trajectory values must be finite')
        if not np.allclose(values[:, 0], np.arange(len(values))*self.timestep_s,
                           rtol=0, atol=1e-8):
            raise ValueError('Trajectory must start at 0 s with consecutive 0.01 s samples; no resampling')
        if np.any(values[:, 11:] != 0):
            raise ValueError('Nonzero body angular-rate feedforward is not supported by the fitted controller')
        if np.any(np.abs(values[-1, 4:10]) > 1e-10):
            raise ValueError('End the trajectory at rest: final velocity and acceleration must be zero')
        # Existing plant command: P(3), V(3), A(3), yaw, yaw-rate (zero).
        self._commands = np.column_stack((values[:, 1:11], np.zeros(len(values))))
        self._commands[-1, 3:9] = 0.0
        self._commands.flags.writeable = False

    @classmethod
    def load(cls, path):
        """CSV units: metres, seconds, radians; omega is body-frame rad/s."""
        rows = np.atleast_1d(np.genfromtxt(path, delimiter=',', names=True, encoding='utf-8-sig'))
        if rows.dtype.names != COLUMNS:
            raise ValueError('CSV header must be: ' + ','.join(COLUMNS))
        return cls(np.column_stack([rows[name] for name in COLUMNS]))

    def command(self, step_index):
        if not isinstance(step_index, (int, np.integer)) or step_index < 0:
            raise ValueError('Step index must be a nonnegative integer')
        # Hold the final position/yaw indefinitely; pause/reset need no cursor state.
        return self._commands[min(step_index, len(self._commands)-1)].copy()

    def save(self, path):
        """Export the same 100 Hz full-state format accepted by load()."""
        rows = np.zeros((self.sample_count, len(COLUMNS)))
        rows[:, 0] = np.arange(self.sample_count)*self.timestep_s
        rows[:, 1:11] = self._commands[:, :10]
        np.savetxt(path, rows, delimiter=',', header=','.join(COLUMNS), comments='', fmt='%.17g')

    @property
    def sample_count(self):
        return len(self._commands)

    @property
    def duration_s(self):
        return (self.sample_count-1)*self.timestep_s

    def preview(self, component):
        offset = {'Position': 0, 'Velocity': 3, 'Acceleration': 6}[component]
        return self._commands[:, offset:offset+3].copy()

    def validate_initial(self, config):
        from simulation.state import quaternion_xyzw_to_rotation_matrix
        if not np.isclose(config.timestep_s, self.timestep_s, rtol=0, atol=1e-12):
            raise ValueError('Simulation and trajectory timesteps must agree')
        initial = config.initial_state
        rotation = quaternion_xyzw_to_rotation_matrix(initial.quaternion_body_to_world_xyzw)
        yaw = float(np.arctan2(rotation[1, 0], rotation[0, 0]))
        first = self.command(0)
        if not np.allclose(first[:3], initial.position_world_m, rtol=0, atol=1e-8):
            raise ValueError('Trajectory must begin at the configured initial tracking position')
        if abs(np.arctan2(np.sin(first[9]-yaw), np.cos(first[9]-yaw))) > 1e-8:
            raise ValueError('Trajectory must begin at the configured initial yaw')

    @classmethod
    def gentle_move(cls, position, yaw=0.):
        """One-second hold, 0.25 m world-X move over 2 s, one-second hold."""
        t = np.arange(401)*cls.timestep_s
        s = np.clip((t-1.)/2., 0., 1.)
        shape = 10*s**3-15*s**4+6*s**5
        velocity = (30*s**2-60*s**3+30*s**4)/2.
        acceleration = (60*s-180*s**2+120*s**3)/4.
        rows = np.zeros((len(t), len(COLUMNS)))
        rows[:, 0] = t
        rows[:, 1:4] = position
        rows[:, 1] += .25*shape
        rows[:, 4] = .25*velocity
        rows[:, 7] = .25*acceleration
        rows[:, 10] = yaw
        return cls(rows)
