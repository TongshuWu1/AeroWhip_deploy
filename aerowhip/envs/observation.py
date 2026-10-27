"""The policy's frame: one row per 100 Hz observation, built from an Observer estimate
(deployment.observer). The training environment and the flight node both build their frames
here, so the actor sees the same numbers in simulation and in flight.

Columns: sensed tracking position relative to the episode origin / cable length (3), rotation
(9), velocity / 4 (3), body rate / 4 (3), cable nodes relative to the attachment / length (3n,
node 0 = attachment = 0), node velocities relative to the attachment / 4 (3n), marker
confidences (10), [critic only: motor thrusts / maximum (4), applied command / scale (4)],
previous action (3), time / horizon (1).
"""
import numpy as np

from aerowhip.deployment.cable_filter import MARKERS

POSE = 18
CRITIC_ONLY = 8
ACTION = 3


def frame_size(nodes):
    return POSE+6*nodes+MARKERS+CRITIC_ONLY+ACTION+1


def actor_columns(nodes):
    """Frame columns the actor reads: everything but the critic-only motor and command slots."""
    sensed = POSE+6*nodes+MARKERS
    return list(range(sensed))+list(range(sensed+CRITIC_ONLY, frame_size(nodes)))


def frame_row(estimate, *, origin, length, previous_action, progress, nodes=MARKERS+1, critic=None):
    """One frame from an Observer estimate. critic: the 8 critic-only values (simulation), zeros
    in flight (the actor never reads them)."""
    p, R = estimate['position'], estimate['rotation']
    cable = np.zeros((nodes, 3)); rates = np.zeros((nodes, 3))
    cable[1:] = estimate['cable_positions']-estimate['root']
    rates[1:] = estimate['cable_velocities']-estimate['root_velocity']
    return np.concatenate(((p-origin)/length, R.reshape(9), estimate['velocity']/4, estimate['angular_velocity']/4,
                           (cable/length).reshape(-1), (rates/4).reshape(-1), estimate['confidence'],
                           np.zeros(CRITIC_ONLY) if critic is None else critic, previous_action, [progress]))
