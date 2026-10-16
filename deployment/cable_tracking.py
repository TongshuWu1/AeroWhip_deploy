"""Experimental geometric/temporal association of an unordered cable point cloud.

No physical integration, controller, or persistent vendor IDs. Initialization
requires a roughly vertical, hanging cable. Missing observations remain missing.
Scores are association heuristics, not calibrated probabilities.
"""
from dataclasses import asdict, dataclass
import json

import numpy as np


@dataclass(frozen=True)
class TrackingSettings:
    marker_distances_m: tuple
    initialization_radius_m: float = .035
    length_tolerance_m: float = .015
    position_noise_m: float = .012
    maximum_speed_mps: float = 12.
    maximum_gap_s: float = .25
    ambiguity_margin: float = 1.
    missing_cost: float = 4.
    beam_width: int = 128
    maximum_candidates_per_marker: int = 12
    maximum_cloud_points: int = 2048
    duplicate_radius_m: float = .001

    def __post_init__(self):
        distances = np.asarray(self.marker_distances_m, dtype=float)
        if (distances.ndim != 1 or not 1 <= len(distances) <= 64 or
                not np.isfinite(distances).all() or np.any(np.diff(np.r_[0., distances]) <= 0)):
            raise ValueError('marker_distances_m must be increasing positive arc distances from the attachment')
        object.__setattr__(self, 'marker_distances_m', tuple(distances))
        for name, value in asdict(self).items():
            if name == 'marker_distances_m':
                continue
            if not np.isfinite(value) or value <= 0:
                raise ValueError(f'{name} must be finite and positive')
        for name in ('beam_width', 'maximum_candidates_per_marker', 'maximum_cloud_points'):
            if type(getattr(self, name)) is not int:
                raise ValueError(f'{name} must be an integer')
        if self.maximum_cloud_points > 2048:
            raise ValueError('maximum_cloud_points cannot exceed the relay limit of 2048')

    @classmethod
    def load(cls, path, intervals):
        values = {} if path is None else json.loads(path.read_text())
        if not isinstance(values, dict):
            raise ValueError('Cable tracking configuration must be a JSON object')
        values.setdefault('marker_distances_m', np.cumsum(intervals).tolist())
        try:
            return cls(**values)
        except TypeError as exc:
            raise ValueError('Invalid cable tracking configuration: '+str(exc)) from exc


class CableTracker:
    def __init__(self, settings):
        self.settings = settings
        self.arc = np.asarray(settings.marker_distances_m)
        self.reset()

    def reset(self):
        self.positions = np.zeros((len(self.arc), 3))
        self.velocities = np.zeros_like(self.positions)
        self.observed_at = np.full(len(self.arc), -np.inf)
        self.initialized = False
        self.previous_time = None
        self.latest = None

    def observe(self, time_s, root, points):
        cfg = self.settings
        root = np.asarray(root, dtype=float)
        points = np.asarray(points, dtype=float).reshape(-1, 3)
        if root.shape != (3,) or not np.isfinite(root).all() or not np.isfinite(time_s):
            raise ValueError('Cable tracking requires a finite attachment and timestamp')
        if self.previous_time is not None and time_s <= self.previous_time:
            raise ValueError('Cable cloud timestamp must increase')
        self.previous_time = time_s
        valid = np.zeros(len(self.arc), dtype=bool)
        confidence = np.zeros(len(self.arc))
        indices = np.full(len(self.arc), -1, dtype=int)
        output = np.zeros_like(self.positions)
        status = 'tracking'
        stale = np.zeros(len(self.arc), dtype=bool)
        crowded = []

        def finish():
            self.latest = dict(time_s=float(time_s), status=status,
                initialized=self.initialized, root_world_m=root.tolist(),
                positions_world_m=[p.tolist() if ok else None for p, ok in zip(output, valid)],
                point_indices=indices.tolist(), confidence=confidence.tolist(),
                observed_count=int(valid.sum()), marker_count=len(self.arc),
                reacquired_markers=(np.flatnonzero(valid & stale)+1).tolist(),
                crowded_markers=crowded,
                confidence_kind='Heuristic separation of retained association hypotheses; not a probability')
            return self.latest

        if len(points) > cfg.maximum_cloud_points:
            status = 'cloud capacity exceeded; assignments unavailable'
            return finish()
        if self.initialized:
            stale = time_s-self.observed_at > cfg.maximum_gap_s
            # A lost material marker must not erase the other visible identities.
            # Full loss still requires the explicit hanging initialization prior.
            if stale.all():
                self.initialized = False
                stale[:] = False
        # The upstream receiver may include the same marker in two NatNet lists.
        # Preserve original indices/raw cloud in evidence, deduplicate only here.
        candidates = []
        for index, point in enumerate(points):
            if not np.isfinite(point).all() or np.linalg.norm(point-root) > self.arc[-1]+cfg.length_tolerance_m:
                continue
            if candidates and np.min(np.linalg.norm(points[candidates]-point, axis=1)) <= cfg.duplicate_radius_m:
                continue
            candidates.append(index)
        candidates = np.asarray(candidates, dtype=int)
        cloud = points[candidates]
        if not self.initialized:
            prediction = root + np.column_stack((np.zeros(len(self.arc)), np.zeros(len(self.arc)), -self.arc))
            gates = np.full(len(self.arc), cfg.initialization_radius_m)
            scale = gates
        else:
            age = time_s-self.observed_at
            prediction = self.positions+self.velocities*age[:, None]
            gates = cfg.position_noise_m+cfg.maximum_speed_mps*age
            scale = cfg.position_noise_m+.25*cfg.maximum_speed_mps*age
        distances = np.linalg.norm(prediction[:, None, :]-cloud[None, :, :], axis=2)
        choices = [np.flatnonzero(row <= gate) for row, gate in zip(distances, gates)]
        costs = (distances/scale[:, None])**2
        if stale.any():
            recent = np.flatnonzero(~stale)
            for i in np.flatnonzero(stale):
                # Long-occluded markers have no trustworthy velocity prediction.
                # Bound recovery by the root and recent neighbours' reachable
                # regions, then enforce exact pairwise geometry in the search.
                feasible = np.linalg.norm(cloud-root, axis=1) <= self.arc[i]+cfg.length_tolerance_m
                for j in recent:
                    reach = (abs(self.arc[i]-self.arc[j])+cfg.length_tolerance_m+
                             cfg.position_noise_m+cfg.maximum_speed_mps*age[j])
                    feasible &= np.linalg.norm(cloud-self.positions[j], axis=1) <= reach
                choices[i] = np.flatnonzero(feasible)
                costs[i] = 0.  # Geometry-only hypotheses; never favour an old extrapolation.
        # Apply the existing root bound before the capacity check: candidates
        # impossible for this material marker must not consume its search budget.
        root_distances = np.linalg.norm(cloud-root, axis=1)
        for i, choice in enumerate(choices):
            choice = choice[root_distances[choice] <= self.arc[i]+cfg.length_tolerance_m]
            choices[i] = choice
            if len(choice) > cfg.maximum_candidates_per_marker:
                choices[i] = np.empty(0, dtype=int)
                crowded.append(i+1)
        # Reuse candidate-pair distances across beam hypotheses. Limit the matrix
        # to retained choices, rather than allocating it for the full raw cloud.
        used = np.unique(np.concatenate(choices))
        local_index = np.full(len(cloud), -1, dtype=int)
        local_index[used] = np.arange(len(used))
        active_cloud = cloud[used]
        pair_distances = np.linalg.norm(active_cloud[:, None, :]-active_cloud[None, :, :], axis=2)
        # Ordered global beam search; each point can represent at most one node.
        # Missing nodes remain hypotheses, including during occlusion/reacquisition.
        beams = [(0., ())]
        truncated = False
        for i, choices_i in enumerate(choices):
            next_beams = []
            for score, assignment in beams:
                next_beams.append((score+cfg.missing_cost, assignment+(-1,)))
                for point_id in choices_i:
                    if point_id in assignment:
                        continue
                    if any(j >= 0 and pair_distances[local_index[point_id], local_index[j]] >
                           self.arc[i]-self.arc[k]+cfg.length_tolerance_m
                           for k, j in enumerate(assignment)):
                        continue
                    next_beams.append((score+costs[i, point_id],
                                       assignment+(int(point_id),)))
            next_beams.sort(key=lambda pair: pair[0])
            truncated |= len(next_beams) > cfg.beam_width
            beams = next_beams[:cfg.beam_width]
        best_score, best = beams[0]
        plausible = [assignment for score, assignment in beams
                     if score <= best_score+cfg.ambiguity_margin]
        for i, point_id in enumerate(best):
            if point_id < 0 or any(other[i] != point_id for other in plausible):
                continue
            alternative = min((score for score, assignment in beams if assignment[i] != point_id),
                              default=best_score+cfg.missing_cost)
            confidence[i] = min(1., (alternative-best_score)/cfg.missing_cost)
            valid[i] = True
            indices[i] = candidates[point_id]
            output[i] = cloud[point_id]
        if not self.initialized and not valid.all():
            valid[:] = False
            indices[:] = -1
            confidence[:] = 0
            status = 'awaiting complete unambiguous hanging cable'
        else:
            for i in np.flatnonzero(valid):
                if self.initialized and not stale[i]:
                    dt = time_s-self.observed_at[i]
                    # ROS receipt timestamps can bunch camera frames together.
                    # Noise/speed gives a time scale below which a raw finite
                    # difference is unreliable; blend with the prior velocity.
                    tau = cfg.position_noise_m/cfg.maximum_speed_mps
                    velocity = (output[i]-self.positions[i]+tau*self.velocities[i])/(dt+tau)
                    norm = np.linalg.norm(velocity)
                    self.velocities[i] = velocity*min(1., cfg.maximum_speed_mps/max(norm, 1e-12))
                else:
                    self.velocities[i] = 0.
                self.positions[i] = output[i]
                self.observed_at[i] = time_s
            self.initialized = True
            status = 'tracking' if valid.all() else 'partial / ambiguous or missing markers'
            if crowded:
                status = 'partial / crowded markers remain unavailable'
        self.latest = finish()
        self.latest['beam_truncated'] = truncated
        return self.latest
