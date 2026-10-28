"""Cable measures shared by the flight stack's cable-over-drone watch, the envelope and the planner."""
import numpy as np


def cable_rise(cable_nodes, root):
    """Highest cable node above the attachment (m, per frame): cable_nodes (N, n, 3) with the root
    (attachment) first, root (N, 3). Positive means the cable is above the attachment: the flips of
    2026-10-28 reached +0.29 m, normal strokes stay at about -0.07 m (the first marker below)."""
    return (np.asarray(cable_nodes)[:, 1:, 2]-np.asarray(root)[:, None, 2]).max(1)


def first_link_angle_deg(cable_nodes, root):
    """Angle of the first link from hanging straight down (deg, per frame). Not a flip measure: the
    cable folds at the tip while the first link hangs."""
    link = np.asarray(cable_nodes)[:, 1]-np.asarray(root)
    return np.degrees(np.arccos(np.clip(-link[:, 2]/np.linalg.norm(link, axis=1), -1., 1.)))
