"""Timestamped P/V/A/yaw packets, independent of the feedback sample rate.

World position refers to the top-of-drone tracking origin, not the COM.
Acceleration is kinematic feedforward (gravity is added by the inner controller).
Yaw is in radians; omega is rad/s in the nominal desired body frame. The fitted
controller requires zero rates; the optional geometric controller consumes them.
"""

import numpy as np
from project_paths import project_path
from simulation.timing import TIME_EPS, validate_command_times


COLUMNS = ('time_s', 'x', 'y', 'z', 'vx', 'vy', 'vz', 'ax', 'ay', 'az',
           'yaw', 'omega_x', 'omega_y', 'omega_z')


class FullStateTrajectory:
    timestep_s = 1/30

    def __init__(self, samples, *, jerk_world=None):
        values = np.array(samples, dtype=np.float64, copy=True)
        if values.ndim != 2 or values.shape[1] != len(COLUMNS) or len(values) < 2:
            raise ValueError('Expected at least two rows of time, P/V/A, yaw and body angular velocity')
        if not np.isfinite(values).all():
            raise ValueError('Trajectory values must be finite')
        self.time_s = validate_command_times(values[:, 0])
        self.time_s.flags.writeable = False
        intervals = np.diff(self.time_s)
        self.timestep_s = (float(intervals[0]) if
            np.allclose(intervals, intervals[0], rtol=0, atol=TIME_EPS) else None)
        self.has_body_rates = bool(np.any(values[:, 11:] != 0))
        self.jerk_world = None
        if jerk_world is not None:
            jerk = np.array(jerk_world, dtype=np.float64, copy=True)
            if jerk.shape != (len(values),3) or not np.isfinite(jerk).all() or self.has_body_rates:
                raise ValueError('Analytic jerk requires finite XYZ rows and no explicit body rates')
            if np.any(np.abs(jerk[-1]) > 1e-10) or np.any(values[:,10] != values[0,10]):
                raise ValueError('Analytic jerk references require fixed yaw and zero final jerk for hold')
            jerk.flags.writeable = False
            self.jerk_world = jerk
        if np.any(np.abs(values[-1, 4:10]) > 1e-10):
            raise ValueError('End the trajectory at rest: final velocity and acceleration must be zero')
        if np.any(values[-1,11:] != 0):
            raise ValueError('End the trajectory at rest: final body rates must be zero')
        # Preserve legacy packet identity when rates are zero; no change to fits.
        self._commands = (values[:,1:14].copy() if self.has_body_rates else
                          np.column_stack((values[:, 1:11], np.zeros(len(values)))))
        self._commands[-1, 3:9] = 0.0
        self._commands.flags.writeable = False

    @classmethod
    def load(cls, path):
        """CSV units: metres, seconds, radians; omega is body-frame rad/s."""
        rows = np.atleast_1d(np.genfromtxt(project_path(path), delimiter=',', names=True, encoding='utf-8-sig'))
        legacy = ('time_s', 'px_m', 'py_m', 'pz_m', 'vx_m_s', 'vy_m_s', 'vz_m_s',
                  'ax_m_s2', 'ay_m_s2', 'az_m_s2', 'yaw_rad', 'yaw_rate_rad_s')
        if rows.dtype.names == legacy:
            if np.any(rows['yaw_rate_rad_s'] != 0):
                raise ValueError('Nonzero yaw-rate feedforward is not supported')
            values = np.zeros((len(rows), len(COLUMNS)))
            values[:, :11] = np.column_stack([rows[name] for name in legacy[:11]])
            return cls(values)
        if rows.dtype.names != COLUMNS:
            raise ValueError('CSV header must be: ' + ','.join(COLUMNS))
        return cls(np.column_stack([rows[name] for name in COLUMNS]))

    def command(self, step_index):
        if not isinstance(step_index, (int, np.integer)) or step_index < 0:
            raise ValueError('Step index must be a nonnegative integer')
        # Hold the final position/yaw indefinitely; pause/reset need no cursor state.
        return self._commands[min(step_index, len(self._commands)-1)].copy()

    def save(self, path, *, transform=None):
        """Preserve source timestamps; never resample commands during export."""
        rows = np.zeros((self.sample_count, len(COLUMNS)))
        rows[:, 0] = self.time_s
        rows[:, 1:11] = self._commands[:, :10]
        for i, command in enumerate(self._commands):
            if transform is not None:
                command = transform(command, None if self.jerk_world is None else self.jerk_world[i])
            if len(command) == 13:
                rows[i,11:14] = command[10:13]
        np.savetxt(project_path(path), rows, delimiter=',', header=','.join(COLUMNS), comments='', fmt='%.17g')

    @property
    def sample_count(self):
        return len(self._commands)

    @property
    def duration_s(self):
        return float(self.time_s[-1])

    @property
    def rate_label(self):
        return f'{1/self.timestep_s:g} Hz' if self.timestep_s is not None else 'Timestamped'

    def index_at(self, time_s):
        if not np.isfinite(time_s) or time_s < 0:
            raise ValueError('Time must be finite and nonnegative')
        return int(np.searchsorted(self.time_s, time_s+TIME_EPS, side='right')-1)

    def command_at(self, time_s):
        return self.command(self.index_at(time_s))

    def events(self, start_s, end_s):
        """Commands in [start, end); the runtime applies its delay exactly once."""
        first = np.searchsorted(self.time_s, start_s-TIME_EPS, side='left')
        last = np.searchsorted(self.time_s, end_s-TIME_EPS, side='left')
        return tuple((float(self.time_s[i]), self.command(i)) for i in range(first, last))

    def reference_events(self, start_s, end_s, transform):
        """Pure reference preparation; neither future state nor plant force is used."""
        first = np.searchsorted(self.time_s, start_s-TIME_EPS, side='left')
        last = np.searchsorted(self.time_s, end_s-TIME_EPS, side='left')
        return tuple((float(self.time_s[i]), transform(self.command(i),
            None if self.jerk_world is None else self.jerk_world[i])) for i in range(first,last))

    def validate_controller(self):
        if self._commands.shape[1] not in (11,13) or not np.isfinite(self._commands).all():
            raise ValueError('Expected finite full-state references')

    def preview(self, component):
        offset = {'Position': 0, 'Velocity': 3, 'Acceleration': 6}[component]
        return self._commands[:, offset:offset+3].copy()

    def validate_initial(self, config):
        from simulation.state import quaternion_xyzw_to_rotation_matrix
        initial = config.initial_state
        rotation = quaternion_xyzw_to_rotation_matrix(initial.quaternion_body_to_world_xyzw)
        yaw = float(np.arctan2(rotation[1, 0], rotation[0, 0]))
        first = self.command(0)
        if not np.allclose(first[:3], initial.position_world_m, rtol=0, atol=1e-8):
            raise ValueError('Trajectory must begin at the configured initial tracking position')
        if abs(np.arctan2(np.sin(first[9]-yaw), np.cos(first[9]-yaw))) > 1e-8:
            raise ValueError('Trajectory must begin at the configured initial yaw')

    @classmethod
    def gentle_move(cls, position, yaw=0., *, timestep_s=1/30):
        """One-second hold, 0.25 m world-X move over 2 s, one-second hold."""
        if not np.isfinite(timestep_s) or timestep_s <= 0 or abs(round(4/timestep_s)*timestep_s-4) > TIME_EPS:
            raise ValueError('The command timestep must divide the four-second trajectory')
        t = np.arange(round(4/timestep_s)+1)*timestep_s
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
        jerk = np.zeros((len(t),3))
        active = (t>=1.) & (t<3.)
        jerk[active,0] = .25*(60-360*s[active]+360*s[active]**2)/8.
        return cls(rows, jerk_world=jerk)
