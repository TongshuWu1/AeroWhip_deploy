"""Live cable-state filter: ten marker positions and velocities from an OptiTrack marker
cloud, every frame, with a confidence per marker and a quality flag for the whole cable.

One state (positions, velocities) is predicted forward each frame with the measured motion
of the attachment and a constant-velocity model per marker, projected onto the cable's
length constraints. Points are associated to markers by nearest-to-prediction within a
gate, each point to at most one marker, and the whole chain is checked against the cable
lengths; markers that were seen are corrected towards their measurement, hidden markers
coast on the prediction. When markers are missing and free points remain, a chain search
from the attachment (each next point at the cable length from the previous one, the first
below the attachment) re-labels the cable from the cloud: it recovers a cable that moved
faster than the prediction, e.g. folding in a whip snap. The same code runs offline over a recorded cloud (`cable_labels`)
and live, so what is validated on recordings is what flies.

No learned components. Costs per frame are O(markers x points) NumPy operations.
"""
import math

import numpy as np

MARKERS = 10
DRONE_RADIUS_M = .07        # points this close to the tracking origin are the drone's markers
DUPLICATE_RADIUS_M = .01    # points this close together are one marker


def cable_candidates(points, drone_position, *, drone_radius_m=DRONE_RADIUS_M, duplicate_radius_m=DUPLICATE_RADIUS_M):
    """The cloud points that may be cable markers: not the drone's own markers (within
    drone_radius_m of the tracking origin), duplicates merged (the first of points closer than
    duplicate_radius_m is kept). Same preprocessing offline (cable labels) and in simulation."""
    xyz = np.asarray(points, float).reshape(-1, 3)
    xyz = xyz[np.isfinite(xyz).all(1)]
    xyz = xyz[np.linalg.norm(xyz-np.asarray(drone_position, float), axis=1) > drone_radius_m]
    kept = []
    for q in xyz:
        if not kept or np.min(np.linalg.norm(np.array(kept)-q, axis=1)) > duplicate_radius_m:
            kept.append(q)
    return np.array(kept, float).reshape(-1, 3)


class CableFilter:
    def __init__(self, arc, *, dt=.01, gate_m=.06, gate_growth_m=.5, noise_m=.004, length_tolerance_m=.015,
                 start_tolerance_m=.02, start_minimum=6, max_hidden_s=.5, projection_iterations=3,
                 max_speed_mps=12., reacquire_m=.05, search_limit=4000, prediction_weight=.25):
        """arc: distance of each marker along the cable from the attachment (m).
        gate_m: association radius for a marker seen in the previous frame; the gate grows by
        gate_growth_m per second hidden. noise_m: OptiTrack position noise, sets how much a
        measurement moves the state. max_hidden_s: after this long hidden, a marker's
        confidence is 0 and the whole-cable quality is reduced."""
        self.arc = np.asarray(arc, float)
        self.segments = np.diff(np.r_[0., self.arc])      # cable length between consecutive markers
        self.dt, self.gate, self.gate_growth, self.noise = dt, gate_m, gate_growth_m, noise_m
        self.tolerance, self.start_tolerance, self.start_minimum = length_tolerance_m, start_tolerance_m, start_minimum
        self.max_hidden, self.iterations = max_hidden_s, projection_iterations
        self.max_speed, self.reacquire = max_speed_mps, reacquire_m
        self.search_limit, self.prediction_weight = search_limit, prediction_weight
        self.reset()

    def reset(self):
        self.positions = np.full((MARKERS, 3), np.nan)
        self.velocities = np.zeros((MARKERS, 3))
        self.hidden_s = np.full(MARKERS, np.inf)        # time since each marker was last seen
        self.confidence = np.zeros(MARKERS)
        self.initialized = False
        self.time = None
        self.clock = None                                # camera frame clock (see observe)
        self.root = None
        self.root_velocity = np.zeros(3)
        self.quality = 0.                                # 0 unreliable .. 1 all markers seen
        self.index = np.full(MARKERS, -1)                # point index per marker in the last frame

    # ------------------------------------------------------------------ steps
    def start(self, root, points):
        """Initialise from a frame where at least start_minimum markers are unambiguous at their
        arc distances from the attachment, below it. Returns True when initialised."""
        rel = points-root
        if not len(rel):
            return False
        distance = np.linalg.norm(rel, axis=1)
        index = np.full(MARKERS, -1)
        for k, arc in enumerate(self.arc):
            candidates = np.flatnonzero((np.abs(distance-arc) < self.start_tolerance) & (rel[:, 2] < 0))
            if len(candidates) > 1:
                return False
            if len(candidates):
                index[k] = candidates[0]
        found = np.flatnonzero(index >= 0)
        if len(found) < self.start_minimum or len(set(index[found].tolist())) != len(found):
            return False
        self.positions[:] = np.nan
        self.positions[found] = points[index[found]]
        self._fill_hidden(root)
        if not self._chain_ok(root, self.positions):
            self.positions[:] = np.nan
            return False
        self.velocities[:] = 0.
        self.hidden_s[:] = np.inf
        self.hidden_s[found] = 0.
        self.index = index
        self.initialized = True
        return True

    def observe(self, time_s, root, points, root_velocity=None):
        """Advance to time_s with the attachment at `root` and the cloud `points` (N, 3)."""
        points = np.asarray(points, float).reshape(-1, 3)
        root = np.asarray(root, float)
        # The stamps are host receipt times, jittered by a few ms around the cameras' exact 10 ms
        # frames. A frame clock tracks the camera: each stamp advances it by the whole number of
        # frames it is ahead (at least one; a dropped frame counts two), and it follows slow drift.
        # Prediction and velocity use the frame clock, so jitter never enters a velocity.
        if self.clock is None:
            dt, self.clock = self.dt, time_s
        else:
            frames = max(1, round((time_s-self.clock)/self.dt))
            dt = self.dt*frames
            self.clock += dt
            self.clock += .05*(time_s-self.clock)
        self.time, self.root = time_s, root
        if not self.initialized:
            if self.start(root, points):
                self._update_quality()
            return self.state()
        # 1. predict: each marker continues with its velocity; the attachment end follows the drone
        predicted = self.positions+self.velocities*dt
        self._project(root, predicted)
        # 2. associate: nearest point within the gate, one point per marker, best matches first
        gates = self.gate+self.gate_growth*np.minimum(self.hidden_s, self.max_hidden)
        d = np.linalg.norm(predicted[:, None, :]-points[None, :, :], axis=2) if len(points) else np.empty((MARKERS, 0))
        index = np.full(MARKERS, -1)
        if d.size:
            order = np.argsort(d, axis=None)
            taken = np.zeros(len(points), bool)
            done = np.zeros(MARKERS, bool)
            for flat in order:
                k, j = divmod(int(flat), len(points))
                if d[k, j] > gates[k]:
                    break
                if done[k] or taken[j]:
                    continue
                index[k], done[k], taken[j] = j, True, True
        # 3. chain check: drop assignments inconsistent with the cable lengths, worst first
        measured = predicted.copy()
        seen = index >= 0
        measured[seen] = points[index[seen]]
        while True:
            bad = self._worst_violation(root, measured, seen)
            if bad is None:
                break
            index[bad], seen[bad] = -1, False
            measured[bad] = predicted[bad]
        # 3b. re-acquire markers hidden for long: a free point at the right cable distance from
        # both labelled neighbours (or the attachment), near the chain, and consistent
        if (~seen).any() and seen.any():
            taken = np.zeros(len(points), bool)
            taken[index[seen]] = True
            chain = np.vstack([root, measured])
            arc = np.r_[0., self.arc]
            known = np.flatnonzero(np.r_[True, seen])
            for k in np.flatnonzero(~seen & (self.hidden_s >= self.max_hidden)):
                free = np.flatnonzero(~taken)
                if not len(free):
                    break
                before, after = known[known < k+1], known[known > k+1]
                a = before[-1]
                da = np.abs(np.linalg.norm(points[free]-chain[a], axis=1)-(arc[k+1]-arc[a]))
                ok = da < self.tolerance
                if len(after):
                    b = after[0]
                    ok &= np.abs(np.linalg.norm(points[free]-chain[b], axis=1)-(arc[b]-arc[k+1])) < self.tolerance
                candidates = free[ok]
                if len(candidates) != 1:
                    continue
                j = int(candidates[0])
                if np.linalg.norm(points[j]-predicted[k]) > self.reacquire+self.gate_growth*self.max_hidden:
                    continue
                index[k], seen[k], taken[j] = j, True, True
                measured[k] = points[j]
                chain[k+1] = points[j]
                self.velocities[k] = 0.
        # 3c. chain search: markers still missing and free points left -> re-label from the root
        jumped = np.zeros(MARKERS, bool)
        if (~seen).any() and len(points) > seen.sum():
            chain = self._search(root, points, predicted)
            if chain is not None and len(chain) > seen[:len(chain)].sum() and len(chain) >= np.flatnonzero(np.r_[seen, True])[0]:
                for k, j in enumerate(chain):
                    if index[k] != j:
                        jumped[k] = np.linalg.norm(points[j]-predicted[k]) > gates[k]
                    index[k], seen[k] = j, True
                    measured[k] = points[j]
                used = set(chain)
                for k in range(len(chain), MARKERS):     # keep later labels only if not reused
                    if seen[k] and index[k] in used:
                        index[k], seen[k] = -1, False
                        measured[k] = predicted[k]
                while True:
                    bad = self._worst_violation(root, measured, seen)
                    if bad is None:
                        break
                    index[bad], seen[bad] = -1, False
                    measured[bad] = predicted[bad]
        # 4. correct: seen markers move to their measurement (noise-weighted), hidden ones coast
        new = predicted.copy()
        new[seen] = measured[seen]
        self._project(root, new)
        velocity = (new-self.positions)/dt
        speed = np.linalg.norm(velocity, axis=1, keepdims=True)
        velocity = velocity*np.minimum(1., self.max_speed/np.maximum(speed, 1e-9))
        alpha = np.where(seen, .5, .1)[:, None]          # velocity smoothing; hidden markers keep their velocity
        self.velocities = self.velocities+alpha*(velocity-self.velocities)
        self.velocities[jumped] = 0.                      # re-labelled far from the prediction
        self.positions = new
        self.hidden_s = np.where(seen, 0., self.hidden_s+dt)
        self.index = index
        self._update_quality()
        return self.state()

    # ------------------------------------------------------------------ helpers
    def _search(self, root, points, predicted):
        """Best chain of point indices for markers 1..L from the attachment, L as large as the
        cloud allows. Each link within the length tolerance; the first point below the
        attachment (drone markers sit above it at similar distances). Score: squared length
        errors plus prediction_weight x squared distance to the prediction. Markers on which
        the best and an equally long, nearly as good chain disagree are cut off."""
        distance_root = np.linalg.norm(points-root, axis=1)
        reach = distance_root <= self.arc[-1]+self.tolerance
        best, runner = [None, np.inf, 0], [None, np.inf]
        budget = [self.search_limit]

        def go(previous, used, cost):
            depth = len(used)
            if depth > best[2] or (depth == best[2] and cost < best[1]):
                if depth == best[2] and best[0] is not None:
                    runner[0], runner[1] = best[0], best[1]
                else:
                    runner[0], runner[1] = None, np.inf
                best[0], best[1], best[2] = tuple(used), cost, depth
            elif depth == best[2] and cost < runner[1]:
                runner[0], runner[1] = tuple(used), cost
            if depth == MARKERS or budget[0] <= 0:
                return
            budget[0] -= 1
            error = np.abs(np.linalg.norm(points-previous, axis=1)-self.segments[depth])
            ok = (error < self.tolerance) & reach & (distance_root <= self.arc[depth]+self.tolerance)
            if depth == 0:
                ok &= points[:, 2] < root[2]
            for j in np.flatnonzero(ok):
                if j in used:
                    continue
                miss = np.linalg.norm(points[j]-predicted[depth])
                go(points[j], used+[int(j)], cost+error[j]**2+self.prediction_weight*min(miss, .2)**2/100)
        go(root, [], 0.)
        chain = best[0]
        if not chain:
            return None
        if runner[0] is not None and len(runner[0]) == len(chain) and runner[1] < 1.5*best[1]+1e-6:
            agree = 0
            while agree < len(chain) and chain[agree] == runner[0][agree]:
                agree += 1
            chain = chain[:agree]
        return list(chain) if chain else None

    def _fill_hidden(self, root):
        """Place markers without a position between their known neighbours along the cable."""
        chain = np.vstack([root, self.positions])
        arc = np.r_[0., self.arc]
        known = np.flatnonzero(np.isfinite(chain).all(1))
        for k in range(1, MARKERS+1):
            if np.isfinite(chain[k]).all():
                continue
            before, after = known[known < k], known[known > k]
            a = before[-1]
            if len(after):
                b = after[0]
                chain[k] = chain[a]+(chain[b]-chain[a])*(arc[k]-arc[a])/(arc[b]-arc[a])
            else:
                prev = before[-2] if len(before) > 1 else None
                direction = chain[a]-chain[prev] if prev is not None else np.array([0., 0., -1.])
                chain[k] = chain[a]+(arc[k]-arc[a])*direction/max(np.linalg.norm(direction), 1e-9)
        self.positions = chain[1:]

    def _project(self, root, positions):
        """Pull consecutive markers to at most their cable distance apart (the cable cannot
        stretch); the attachment is fixed. A few Gauss-Seidel sweeps (scalar arithmetic: this
        is the filter's hot spot, sixty tiny vector operations per frame otherwise)."""
        chain = [tuple(map(float, root))]+[tuple(map(float, q)) for q in positions]
        segments = self.segments.tolist()
        for _ in range(self.iterations):
            for k in range(1, MARKERS+1):
                (ax, ay, az), (bx, by, bz) = chain[k-1], chain[k]
                gx, gy, gz = bx-ax, by-ay, bz-az
                length = math.sqrt(gx*gx+gy*gy+gz*gz)
                limit = segments[k-1]
                if length > limit:
                    s = limit/length
                    chain[k] = (ax+gx*s, ay+gy*s, az+gz*s)
        positions[:] = chain[1:]

    def _chain_ok(self, root, positions):
        chain = np.vstack([root, positions])
        steps = np.linalg.norm(np.diff(chain, axis=0), axis=1)
        return bool(np.all(steps <= self.segments+self.tolerance))

    def _worst_violation(self, root, positions, seen):
        """The seen marker whose link to a neighbour is most over length, or None."""
        chain = np.vstack([root, positions])
        steps = np.linalg.norm(np.diff(chain, axis=0), axis=1)
        excess = steps-self.segments-self.tolerance
        worst, value = None, 0.
        for i in np.flatnonzero(excess > 0):      # link i joins chain[i] and chain[i+1]: markers i-1 and i
            for k in (i-1, i):
                if 0 <= k < MARKERS and seen[k] and excess[i] > value:
                    worst, value = k, excess[i]
        return worst

    def _update_quality(self):
        self.confidence = np.clip(1.-self.hidden_s/self.max_hidden, 0., 1.)
        self.quality = float(self.confidence.mean()) if self.initialized else 0.

    def state(self):
        return dict(time_s=self.time, initialized=self.initialized, positions=self.positions.copy(),
                    velocities=self.velocities.copy(), confidence=self.confidence.copy(),
                    quality=self.quality, index=self.index.copy())

    # ------------------------------------------------------------------ offline support
    def get_state(self):
        return dict(positions=self.positions.copy(), velocities=self.velocities.copy(), hidden_s=self.hidden_s.copy(),
                    initialized=self.initialized, time=self.time, clock=self.clock,
                    root=None if self.root is None else self.root.copy(), index=self.index.copy())

    def set_state(self, state):
        self.positions, self.velocities, self.hidden_s = (state[k].copy() for k in ('positions', 'velocities', 'hidden_s'))
        self.initialized, self.time, self.clock = state['initialized'], state['time'], state['clock']
        self.root = None if state['root'] is None else state['root'].copy()
        self.index = state['index'].copy()
        self._update_quality()

    def override(self, root, points, index):
        """Impose a labelling for the current frame (manual correction): seen markers take
        their point, hidden ones keep the prediction; velocities of changed markers reset."""
        index = np.asarray(index)
        changed = index != self.index
        seen = index >= 0
        self.positions[seen] = points[index[seen]]
        self._project(root, self.positions)
        self.velocities[changed] = 0.
        self.hidden_s = np.where(seen, 0., self.hidden_s)
        self.index = index.copy()
        self.initialized = True
        self._update_quality()

