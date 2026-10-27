"""Drone-response objective on drone-only calibration flights.

The native coupled runtime (drone plus the model's cable, which is absent in these flights:
the drone-only setup is run with a model whose cable mass is negligible) is driven by the
acceleration commands that were sent, in direct mode (zero position and velocity gains, as
on the real drone where the Mellinger translation gains were zeroed). The loss compares the
simulated tracking position with the measured one over the window.

Parameters: a scale on the commanded acceleration (the real drone delivered 0.83–0.86 of
it without the cable), the attitude frequency, and a command delay, on top of the model.
"""
from copy import deepcopy
import math

import numpy as np

from aerowhip.fitting.calibration import _runtime
from aerowhip.fitting.prepare_window import initialize
from aerowhip.physics.tensor_model import TensorModel

NAMES = ('acceleration_gain', 'attitude_frequency_scale', 'extra_delay_s')


def drone_only_config(base, effective_mass_kg=.157):
    """The base model with the cable made negligible (drone-only flights) and the drone at
    the effective mass the flight trims encode (the sent command's vertical trim is the
    thrust the firmware's 0.144 kg mass model lacks: real drone + calibration)."""
    config = deepcopy(base)
    cable = config['cable']
    markers = len(cable['marker_interval_lengths_m'])
    cable['total_mass_kg'] = 1e-4
    cable['moving_marker_masses_kg'] = [1e-6]*markers
    config['drone']['mass_kg'] = float(effective_mass_kg)
    return config


def scaled_config(base, values):
    gain, freq, _ = values
    config = deepcopy(base)
    control = config['controller']
    control['attitude_frequency_rad_s'] = [float(v*freq) for v in control['attitude_frequency_rad_s']]
    return config


def window_initial(model, window):
    hist = np.zeros((len(window.history_time), len(model.marker_indices)-1, 3))
    # Drone-only: build a straight hanging (negligible) cable under the attachment for the state.
    attachment = model.attachment_tracking.detach().cpu().numpy()
    arc = np.cumsum(model.config['cable']['marker_interval_lengths_m'])
    for i, (p, R) in enumerate(zip(window.history_position, window.history_rotation)):
        root = p+R@attachment
        hist[i] = root-np.outer(arc, [0., 0., 1.])
    state, info = initialize(model, window.history_time, window.history_position, window.history_rotation, hist)
    return state, info


def window_loss(base, window, values, *, return_positions=False):
    gain, _, delay = values
    model = TensorModel(scaled_config(base, values))
    initial, _ = window_initial(model, window)
    runtime = _runtime(model, type('R', (), {'initial': initial})())
    runtime.controller.kp = np.zeros(3)
    runtime.controller.kv = np.zeros(3)
    positions = []
    yaw = math.atan2(window.rotation[0][1, 0], window.rotation[0][0, 0])
    t = window.time
    for i in range(len(t)-1):
        # command in force at t[i] minus the extra delay (zero-order hold on the log)
        j = int(np.clip(np.searchsorted(t, t[i]-delay, 'right')-1, 0, len(t)-1))
        command = np.zeros(11)
        command[:3] = window.position[0]
        # The real drone received the full command (trims included); its firmware mass model
        # (0.144 kg) turned it into thrust. The simulated drone uses the effective mass with
        # its own exact hover trim, so the vertical trim that compensated the mass error is
        # removed and only the measured sideways offsets and the planned motion remain.
        command[6:9] = gain*(window.sent[j]-np.array([0., 0., window.sent[j, 2]-window.command[j, 2]]))
        command[9] = yaw
        runtime.step(command)
        state = runtime.snapshot(runtime.native.time)
        positions.append(state.drone.position_world_m.copy())
    p = np.array(positions)
    error = np.linalg.norm(p-window.position[1:], axis=1)
    loss = float(np.sqrt(np.mean(error**2)))
    if not math.isfinite(loss):
        raise FloatingPointError('Nonfinite drone loss')
    return (loss, p) if return_positions else loss
