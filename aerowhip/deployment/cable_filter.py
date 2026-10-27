"""Live cable-state filter: ten marker positions and velocities from an OptiTrack marker
cloud, every frame, with a confidence per marker and a quality flag for the whole cable.

One state (positions, velocities) is predicted forward each frame with the measured motion
of the attachment and a constant-velocity model per marker, projected onto the cable's
length constraints. Points are associated to markers by nearest-to-prediction within a
gate, each point to at most one marker, and the whole chain is checked against the cable
lengths; markers that were seen are corrected towards their measurement, hidden markers
coast on the prediction. The same code runs offline over a recorded cloud (`cable_labels`)
and live, so what is validated on recordings is what flies.

No learned components. Costs per frame are O(markers x points) NumPy operations.
"""
import numpy as np

MARKERS = 10


class CableFilter:
    def __init__(self, arc, *, dt=.01, gate_m=.06, gate_growth_m=.5, noise_m=.004, length_tolerance_m=.015,
                 start_tolerance_m=.02, start_minimum=6, max_hidden_s=.5, projection_iterations=3,
                 max_speed_mps=12., reacquire_m=.05):
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
        self.reset()

    def reset(self):
        self.positions = np.full((MARKERS, 3), np.nan)
        self.velocities = np.zeros((MARKERS, 3))
        self.hidden_s = np.full(MARKERS, np.inf)        # time since each marker was last seen
        self.confidence = np.zeros(MARKERS)
        self.initialized = False
        self.time = None
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
        # OptiTrack stamps arrive bunched (0.1 ms, then 10 ms): never let a tiny step inflate a
        # velocity. The step used for prediction and velocity is at least half a nominal frame.
        dt = self.dt if self.time is None else max(time_s-self.time, .5*self.dt)
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
        # 4. correct: seen markers move to their measurement (noise-weighted), hidden ones coast
        new = predicted.copy()
        new[seen] = measured[seen]
        self._project(root, new)
        velocity = (new-self.positions)/dt
        speed = np.linalg.norm(velocity, axis=1, keepdims=True)
        velocity = velocity*np.minimum(1., self.max_speed/np.maximum(speed, 1e-9))
        alpha = np.where(seen, .5, .1)[:, None]          # velocity smoothing; hidden markers keep their velocity
        self.velocities = self.velocities+alpha*(velocity-self.velocities)
        self.positions = new
        self.hidden_s = np.where(seen, 0., self.hidden_s+dt)
        self.index = index
        self._update_quality()
        return self.state()

    # ------------------------------------------------------------------ helpers
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
        stretch); the attachment is fixed. A few Gauss-Seidel sweeps."""
        chain = np.vstack([root, positions])
        for _ in range(self.iterations):
            for k in range(1, MARKERS+1):
                gap = chain[k]-chain[k-1]
                length = np.linalg.norm(gap)
                limit = self.segments[k-1]
                if length > limit:
                    chain[k] = chain[k-1]+gap*(limit/length)
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
                    initialized=self.initialized, time=self.time, root=None if self.root is None else self.root.copy(),
                    index=self.index.copy())

    def set_state(self, state):
        self.positions, self.velocities, self.hidden_s = (state[k].copy() for k in ('positions', 'velocities', 'hidden_s'))
        self.initialized, self.time = state['initialized'], state['time']
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

