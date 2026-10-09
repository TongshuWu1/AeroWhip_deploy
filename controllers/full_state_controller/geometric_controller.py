"""Nominal geometric PVA/rate feedback in SI units; not firmware emulation.

References use the top-plate tracking origin. Angular-rate feedforward is in
the nominal desired body frame obtained from acceleration, weight trim and yaw.
Position feedback adjusts the desired thrust direction; supplied nominal rates
do not include derivatives of that feedback correction. No integral, estimated
load cancellation, angular-acceleration feedforward or stability claim for the
loaded sampled plant is implied. Evaluation is pure, including telemetry reads.
"""
import tomllib

import numpy as np

from project_paths import ROOT, project_path


DEFAULT_CONFIGURATION = ROOT/'configs/controllers/geometric.toml'


def load_gains(path=DEFAULT_CONFIGURATION):
    if isinstance(path, dict):
        values = dict(path)
    else:
        values = tomllib.loads(project_path(path).read_text(encoding='utf-8'))
    keys = ('position_gain_s_inv2', 'velocity_gain_s_inv',
            'attitude_frequency_rad_s', 'attitude_damping_ratio')
    if set(values) != {*keys, 'controller_period_s', 'command_delay_s'}:
        raise ValueError('Geometric controller configuration has missing or unknown fields')
    for name in keys:
        value = np.array(values[name], dtype=np.float64, copy=True)
        if value.shape != (3,) or not np.isfinite(value).all() or (value <= 0).any():
            raise ValueError('Expected three positive finite gains: '+name)
        values[name] = value
    period = values['controller_period_s']
    if isinstance(period, bool) or not isinstance(period,(int,float)) or not np.isfinite(period) or period <= 1e-12:
        raise ValueError('Controller period must be finite and positive')
    values['controller_period_s'] = float(period)
    if not np.isfinite(values['command_delay_s']) or values['command_delay_s'] < 0:
        raise ValueError('Command delay must be finite and nonnegative')
    return values


def desired_rotation(force, yaw, force_rate=None):
    """Body +Z follows force. Optional analytic derivative, with fixed yaw."""
    size = np.linalg.norm(force)
    if not np.isfinite(size) or size < 1e-9:
        raise ValueError('Desired thrust direction is undefined at zero force')
    z = force/size
    heading = np.array([np.cos(yaw), np.sin(yaw), 0.])
    raw_y = np.cross(z, heading)
    length = np.linalg.norm(raw_y)
    if length < 1e-8:
        raise ValueError('Desired thrust direction is parallel to the yaw heading')
    y = raw_y/length
    R = np.column_stack((np.cross(y, z), y, z))
    if force_rate is None:
        return R
    dz = (force_rate-z*np.dot(z, force_rate))/size
    raw_dy = np.cross(dz, heading)
    dy = (raw_dy-y*np.dot(y, raw_dy))/length
    dR = np.column_stack((np.cross(dy,z)+np.cross(y,dz), dy, dz))
    skew = R.T@dR
    omega = .5*np.array([skew[2,1]-skew[1,2],skew[0,2]-skew[2,0],skew[1,0]-skew[0,1]])
    return R, omega


class GeometricController:
    def __init__(self, model):
        gains = load_gains(model.config['controller'])
        self.kp = gains['position_gain_s_inv2']
        self.kv = gains['velocity_gain_s_inv']
        frequency = gains['attitude_frequency_rad_s']
        self.kr = frequency**2
        self.kw = 2*gains['attitude_damping_ratio']*frequency
        self.controller_period_s = gains['controller_period_s']
        self.mass = float(model.mass)
        self.inertia = np.array(model.inertia, copy=True)
        self.com = np.array(model.com_tracking, copy=True)
        self.trim = (self.mass+float(model.node_mass.sum()))*np.array(model.gravity)
        self.set_command(np.zeros(13))

    def prepare_reference(self, command, jerk=None):
        """Expand legacy PVA or analytically supply rates; never alter PVA/yaw."""
        command = np.asarray(command, dtype=np.float64)
        if command.shape not in ((11,), (13,)) or not np.isfinite(command).all():
            raise ValueError('Expected finite legacy PVA/yaw or PVA/yaw/body rates')
        if command.shape == (11,) and command[10] != 0:
            raise ValueError('Legacy yaw-rate packets are unsupported; supply three body rates')
        result = np.zeros(13)
        result[:10] = command[:10]
        if command.shape == (13,):
            if jerk is not None:
                raise ValueError('Supply explicit rates or analytic jerk, not both')
            result[10:13] = command[10:13]
        elif jerk is not None:
            jerk = np.asarray(jerk, dtype=np.float64)
            if jerk.shape != (3,) or not np.isfinite(jerk).all():
                raise ValueError('Analytic jerk must be a finite world XYZ vector')
            _, result[10:13] = desired_rotation(self.mass*command[6:9]-self.trim,
                                                command[9], self.mass*jerk)
        return result

    def set_command(self, command):
        result = self.prepare_reference(command)
        reference_rotation = desired_rotation(self.mass*result[6:9]-self.trim, result[9])
        self.command, self.reference_rotation = result, reference_rotation

    def __call__(self, state):
        state = np.asarray(state, dtype=np.float64)
        if state.ndim != 1 or len(state) < 18 or not np.isfinite(state[:18]).all():
            raise ValueError('Expected a finite COM/rotation/velocity/body-rate state')
        p, R, v, omega = state[:3], state[3:12].reshape(3,3), state[12:15], state[15:18]
        tracking_p = p-R@self.com
        tracking_v = v-R@np.cross(omega,self.com)
        command = self.command
        acceleration = command[6:9]+self.kp*(command[:3]-tracking_p)+self.kv*(command[3:6]-tracking_v)
        force = self.mass*acceleration-self.trim
        desired = desired_rotation(force, command[9])
        skew = R.T@desired-desired.T@R
        e = .5*np.array([skew[2,1],skew[0,2],skew[1,0]])
        rate = R.T@self.reference_rotation@command[10:13]
        # Transport the nominal desired rate into the actual body frame.
        alpha = self.kr*e-self.kw*(omega-rate)-np.cross(omega,rate)
        torque = self.inertia@alpha+np.cross(omega,self.inertia@omega)
        return np.r_[max(float(force@R[:,2]),0.), torque]
