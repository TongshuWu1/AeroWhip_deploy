"""Native vehicle-response fitting with all cable parameters frozen.

Central differences are for three controller log-scales only, not neural
weights. The tensor implementation remains the independent derivative reference.
Recorded commands already include delay; do not apply that delay a second time.
"""
from types import SimpleNamespace
import numpy as np

from aerowhip.fitting.calibration import _runtime
from aerowhip.physics.tensor_model import TensorModel

EPSILON = 1e-5


def rollout(base, window, values):
    from aerowhip.fitting.coupled_fit import _scaled_config, NAMES
    values = np.asarray(values, dtype=np.float64)
    if values.shape != (len(NAMES),) or not np.isfinite(values).all():
        raise ValueError(f'{len(NAMES)} finite log multipliers required')
    model = TensorModel(_scaled_config(base, np.exp(values)))
    runtime = _runtime(model, SimpleNamespace(initial=window.initial))
    positions, rotations, markers = [], [], []
    for command in window.applied_commands:
        runtime.step(command)
        state = runtime.snapshot(runtime.native.time)
        positions.append(state.drone.position_world_m)
        rotations.append(runtime.physics.rotation)
        markers.append(state.cable_node_positions_world_m[model.marker_indices[1:]])
    p, R, q = map(np.asarray, (positions, rotations, markers))
    loss = (((p-window.tracking_positions)/.02)**2).mean()
    loss += (((R-window.tracking_rotations)/.05)**2).sum((-2,-1)).mean()/2
    if not np.isfinite(loss):
        raise FloatingPointError('Nonfinite native coupled fitting loss')
    return float(loss), (p, q, R)


def gradient(base, window, values, epsilon=EPSILON):
    from aerowhip.fitting.coupled_fit import NAMES
    values = np.asarray(values, dtype=np.float64)
    result = []
    for index in range(len(NAMES)):
        offset = np.zeros(len(NAMES))
        offset[index] = epsilon
        plus = rollout(base, window, values+offset)[0]
        minus = rollout(base, window, values-offset)[0]
        result.append((plus-minus)/(2*epsilon))
    if not np.isfinite(result).all():
        raise FloatingPointError('Nonfinite native finite-difference gradient')
    return result


def evaluate_window(args):
    import torch
    torch.set_num_threads(1)
    base, window, values, backward = args
    loss = rollout(base, window, values)[0]
    return loss, gradient(base, window, values) if backward else None


def check_window(args):
    """Step-size agreement, not a substitute for independent AD validation."""
    import torch
    torch.set_num_threads(1)
    base, window, values = args
    first = np.asarray(gradient(base, window, values, EPSILON))
    second = np.asarray(gradient(base, window, values, EPSILON/2))
    limit = 1e-8 + .02*np.maximum(np.abs(first),np.abs(second))
    return dict(recording_group=window.recording_group, maneuver=window.maneuver,
                epsilons=[EPSILON,EPSILON/2], gradients=[first.tolist(),second.tolist()],
                absolute_tolerance=1e-8, relative_tolerance=.02,
                passed=bool(np.all(np.abs(first-second)<=limit)))
