"""Attitude conversion for the drone telemetry panel."""

import numpy as np

def roll_pitch_yaw_deg(rotation):
    """ZYX attitude from body-to-world rotation; roll fixed to zero at gimbal lock."""
    horizontal = np.hypot(rotation[0, 0], rotation[1, 0])
    pitch = np.arctan2(-rotation[2, 0], horizontal)
    if horizontal > 1e-8:
        roll = np.arctan2(rotation[2, 1], rotation[2, 2])
        yaw = np.arctan2(rotation[1, 0], rotation[0, 0])
    else:
        roll = 0.0
        yaw = np.arctan2(-rotation[0, 1], rotation[1, 1])
    return np.rad2deg([roll, pitch, yaw])
