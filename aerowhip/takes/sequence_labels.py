"""Offline cable labelling of a whole recording at once.

Every frame of the marker cloud offers a few candidate cables: chains of points starting at
the attachment, each next point at the cable length of its marker interval from the previous
one (within a tolerance), the first point below the attachment so the drone's own markers are
excluded; one marker may be skipped where OptiTrack lost it. The labelling chosen is the
sequence of candidates, one per frame, with the lowest total cost over the recording: length
errors within the frame, marker motion between consecutive frames, and a credit per labelled
marker. A dynamic programme (Viterbi) finds it, so the steady hovers at both ends, where the
whole cable is visible and unambiguous, anchor the labels through fast or partly hidden
motion in between.

Where another sequence explains the recording almost as well, the markers on which the two
disagree are left unlabelled: a gap is data we do not use, a wrong label is data that
misleads the fit. Nothing here is causal; the live filter (`deployment.cable_filter`) is,
and is scored against these labels.
"""
from dataclasses import dataclass

import numpy as np

from aerowhip.takes.cable_labels import AUTO, Labels, NONE

MARKERS = 10


@dataclass
class Settings:
    length_tolerance_m: float = .015     # link length error allowed within a frame
    fold_minimum_m: float = .10          # c_k to c_k+2 when c_k+1 is hidden: at least this (60 deg fold)
    motion_sigma_m: float = .04          # per-frame marker displacement that costs one unit
    motion_cap: float = 30.              # displacement cost cap (a label jumping is rejected, not priced)
    marker_credit: float = 6.            # cost removed per labelled marker
    ambiguity_margin: float = 4.         # alternatives within this total cost make disagreeing markers gaps
    max_candidates: int = 160            # per frame, lowest local cost kept
    search_budget: int = 20000           # search nodes per frame


def candidates(points, root, arc, s):
    """Chains as index arrays (10,) with -1 for unlabelled markers, with their local costs
    (sum of squared link length errors in units of the tolerance)."""
    n = len(points)
    if n == 0:
        return [np.full(MARKERS, -1)], np.zeros(1)
    full_arc = np.r_[0., arc]
    distance_root = np.linalg.norm(points-root, axis=1)
    below = points[:, 2] < root[2]
    found = {}
    budget = [s.search_budget]

    def extend(position, marker, used, cost):
        # record the chain so far (markers after `marker` unlabelled)
        key = tuple(used)
        if key not in found or cost < found[key]:
            found[key] = cost
        if marker == MARKERS or budget[0] <= 0:
            return
        budget[0] -= 1
        d = np.linalg.norm(points-position, axis=1)
        for skip in (0, 1):
            target = marker+1+skip
            if target > MARKERS:
                break
            length = full_arc[target]-full_arc[marker]
            if skip == 0:
                error = np.abs(d-length)/s.length_tolerance_m
                ok = error < 1.
            else:
                ok = (d >= s.fold_minimum_m) & (d <= length+s.length_tolerance_m)
                error = np.where(ok, 1., np.inf)     # a skipped link costs one unit: known length not checked
            ok &= distance_root <= full_arc[target]+s.length_tolerance_m
            if marker == 0:
                ok &= below
            for j in np.flatnonzero(ok):
                if j in used:
                    continue
                chain = list(used)+[-1]*skip+[int(j)]
                extend(points[j], target, chain, cost+float(error[j])**2)
    extend(root, 0, [], 0.)
    chains, costs = [], []
    for key, cost in found.items():
        index = np.full(MARKERS, -1)
        index[:len(key)] = key
        chains.append(index)
        costs.append(cost)
    # keep the longest, best chains: shorter prefixes only matter when a tail label is wrong
    credited = np.array(costs)-s.marker_credit*np.array([(c >= 0).sum() for c in chains])
    order = np.argsort(credited)[:s.max_candidates]
    return [chains[i] for i in order], np.array(costs)[order]


def label(flight, settings=None):
    """Labels for every frame of `flight` by the lowest-cost sequence of candidates; markers on
    which a near-optimal alternative disagrees are unlabelled. Returns (labels, report)."""
    s = settings or Settings()
    F = flight.frames
    per_frame = [candidates(flight.points[f], flight.root[f], flight.arc, s) for f in range(F)]
    positions = []          # per frame: (C, 10, 3) candidate marker positions, NaN unlabelled
    local = []
    for f, (chains, costs) in enumerate(per_frame):
        P = np.full((len(chains), MARKERS, 3), np.nan)
        for c, index in enumerate(chains):
            seen = index >= 0
            P[c, seen] = flight.points[f][index[seen]]
        positions.append(P)
        local.append(costs-s.marker_credit*np.array([(index >= 0).sum() for index in chains]))

    def transition(f):
        """(C_prev, C_next) motion cost between frame f-1 and f."""
        A, B = positions[f-1], positions[f]
        dt = max(float(flight.stamps[f]-flight.stamps[f-1]), 5e-3)
        scale = s.motion_sigma_m*max(dt/.01, 1.)
        drift = flight.root[f]-flight.root[f-1]            # the cable moves with the drone
        delta = (B[None, :, :, :]-drift)-A[:, None, :, :]
        cost = np.nansum(np.minimum((np.linalg.norm(delta, axis=3)/scale)**2, s.motion_cap), axis=2)
        return cost

    # forward and backward passes for exact two-sided costs through every candidate
    forward = [local[0].copy()]
    back_pointer = [None]
    for f in range(1, F):
        total = forward[f-1][:, None]+transition(f)
        back_pointer.append(np.argmin(total, axis=0))
        forward.append(total.min(axis=0)+local[f])
    backward = [None]*F
    backward[F-1] = np.zeros(len(local[F-1]))
    for f in range(F-2, -1, -1):
        total = transition(f+1)+(local[f+1]+backward[f+1])[None, :]
        backward[f] = total.min(axis=1)
    best_total = float(forward[F-1].min())
    choice = np.empty(F, int)
    choice[F-1] = int(np.argmin(forward[F-1]))
    for f in range(F-1, 0, -1):
        choice[f-1] = back_pointer[f][choice[f]]
    index = np.full((F, MARKERS), -1)
    uncertain = np.zeros((F, MARKERS), bool)
    alternatives = 0
    for f in range(F):
        chains = per_frame[f][0]
        chosen = chains[choice[f]]
        index[f] = chosen
        through = forward[f]+backward[f]
        close = np.flatnonzero(through <= best_total+s.ambiguity_margin)
        for c in close:
            if c == choice[f]:
                continue
            alternatives += 1
            disagree = chains[c] != chosen
            uncertain[f] |= disagree
    index[uncertain] = -1
    source = np.where(index >= 0, AUTO, NONE).astype(np.uint8)
    confidence = np.where(index >= 0, 1., 0.)
    labels = Labels(index, source, confidence)
    report = dict(frames=F, labelled_per_frame=float((index >= 0).sum(1).mean()),
                  complete_frames=int((index >= 0).all(1).sum()),
                  uncertain_marker_frames=int(uncertain.sum()), frames_with_alternatives=int((uncertain.any(1)).sum()),
                  total_cost=best_total, candidates_per_frame=float(np.mean([len(c[0]) for c in per_frame])),
                  max_candidates=int(max(len(c[0]) for c in per_frame)))
    return labels, report
