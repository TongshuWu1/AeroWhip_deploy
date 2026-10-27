"""Strike task: the cable tip hits a target point beyond the drone's reach with a fast tip, then
the drone recovers (specification: docs/strike_task.md).

The reward is defined by outcome only: distance closed to the target (potential-based, so it sums
to the net progress and cannot be farmed), speed toward the target (bounded), the strike itself,
and the recovery; costs for leaving the workspace, for large actions and for a terminated episode.
The whip mechanism is not rewarded: the fold diagnostics (sharpest joint angle) and the whip
event (tip speed, amplification over the attachment, extension) are measured and reported.
"""
from collections import deque
import math

import numpy as np

from aerowhip.envs.configuration import OBSERVATION_PERIOD_S
from aerowhip.envs.task import Task

FEATURE_SIZE = 16
TRACE_FIELDS = ('tip_target_distance_m', 'tip_speed_mps', 'approach_speed_mps', 'extension_fraction',
                'attachment_speed_mps', 'amplification', 'max_turn_deg', 'reward_rate', 'accrued_credit',
                'workspace_cost', 'action_cost', 'net_return', 'struck')


def segment_distance(a, b, target):
    """Closest distance between the segment a→b and the target point."""
    d = b-a
    length2 = float(d@d)
    t = 0. if length2 <= 0. else float(np.clip((target-a)@d/length2, 0., 1.))
    return float(np.linalg.norm(a+t*d-target))


def max_turn_deg(positions):
    """Sharpest joint turning angle of the chain (degrees); a cheap fold diagnostic."""
    edge = np.diff(positions, axis=0)
    norm = np.linalg.norm(edge, axis=1)
    if np.any(norm <= 0):
        return math.nan
    u = edge/norm[:, None]
    cosine = np.clip(np.sum(u[:-1]*u[1:], axis=1), -1., 1.)
    return float(np.degrees(np.arccos(cosine)).max())


class StrikeEpisode:
    """One lane. The workspace (stage radius, environment heights) and the stage's target and
    threshold are given at construction, so the stage is fixed within an episode."""

    def __init__(self, config, environment, workspace, stage, *, length, workspace_origin):
        self.config, self.environment, self.workspace, self.stage = config, environment, workspace, stage
        self.dt = OBSERVATION_PERIOD_S
        self.horizon_s = environment.duration_s
        self.steps = round(self.horizon_s/self.dt)
        self.L = float(length)
        self.origin = np.asarray(workspace_origin, float)
        if self.origin.shape != (3,) or not np.isfinite(self.origin).all():
            raise ValueError('Workspace origin must be finite world XYZ')
        self.target = self.origin+stage.target_distance_m*np.asarray(config.target_direction_world)
        self.target[2] -= config.target_below_origin_m
        self.v_star = stage.speed_threshold_mps
        self.window = round(config.amplification_window_s/self.dt)
        self.attachment_speeds = deque(maxlen=self.window+1)
        self.time = 0.
        self.accrued = self.cost = self.action_cost = self.failure_cost = 0.
        self.total = 0.
        self.struck = False
        self.hit_time = self.hit_speed = math.nan
        self.closest = math.inf
        self.peak = dict(speed=0., time=math.nan, amplification=math.nan, extension=math.nan)
        self.max_turn = 0.
        self.recovered = False
        self.terminal = None
        self.previous_tip = None
        self.previous_distance = None
        self.last = {k: 0. for k in TRACE_FIELDS}
        self.trace = []
        self.log = []
        self.initialized = False

    # ------------------------------------------------------------------ measurements
    def _measure(self, p, v):
        tip, attachment = p[-1], p[0]
        tip_v, attachment_v = v[-1], v[0]
        to_target = self.target-tip
        distance = float(np.linalg.norm(to_target))
        speed = float(np.linalg.norm(tip_v))
        approach = float(tip_v@to_target/distance) if distance > 1e-9 else 0.
        extension = float(np.linalg.norm(tip-attachment)/self.L)
        attachment_speed = float(np.linalg.norm(attachment_v))
        self.attachment_speeds.append(attachment_speed)
        base = max(self.attachment_speeds)
        amplification = speed/base if base > 1e-6 else math.inf if speed > 0 else 0.
        turn = max_turn_deg(p)
        return dict(distance=distance, speed=speed, approach=approach, extension=extension,
                    attachment_speed=attachment_speed, amplification=amplification, turn=turn)

    def _trace(self, m, rate):
        self.last = dict(tip_target_distance_m=m['distance'], tip_speed_mps=m['speed'], approach_speed_mps=m['approach'],
                         extension_fraction=m['extension'], attachment_speed_mps=m['attachment_speed'],
                         amplification=min(m['amplification'], 99.), max_turn_deg=m['turn'], reward_rate=rate,
                         accrued_credit=self.accrued, workspace_cost=self.cost, action_cost=self.action_cost,
                         net_return=self.total, struck=float(self.struck))
        self.trace.append([self.last[k] for k in TRACE_FIELDS])

    def initialize(self, p, v):
        if self.initialized:
            raise RuntimeError('Initialize exactly once')
        p, v = self._arrays(p, v)
        m = self._measure(p, v)
        self.previous_tip, self.previous_distance = p[-1].copy(), m['distance']
        self.max_turn = m['turn'] if math.isfinite(m['turn']) else 0.
        self._trace(m, 0.)
        self.initialized = True

    @staticmethod
    def _arrays(p, v):
        p, v = np.asarray(p, float), np.asarray(v, float)
        if p.shape[1:] != (3,) or p.shape != v.shape or len(p) < 2 or not np.isfinite(p).all() or not np.isfinite(v).all():
            raise ValueError('The strike task needs finite node positions and velocities')
        return p, v

    def _inside(self, drone):
        w = self.workspace
        return (float(np.linalg.norm(drone[:2]-self.origin[:2])) <= w.radius_m
                and w.height_lower_world_m <= drone[2] <= w.height_upper_world_m)

    # ------------------------------------------------------------------ steps
    def observe(self, time, p, v, *, drone_position, action=None):
        if not self.initialized or self.terminal is not None:
            raise RuntimeError('Episode is not active')
        if not math.isclose(time-self.time, self.dt, abs_tol=1e-8) or time > self.horizon_s+1e-9:
            raise ValueError('Observe causal 100 Hz samples through the configured horizon')
        p, v = self._arrays(p, v)
        drone = np.asarray(drone_position, float)
        if drone.shape != (3,) or not np.isfinite(drone).all():
            raise ValueError('The workspace cost needs a finite drone position')
        c, w = self.config, self.workspace
        m = self._measure(p, v)
        previous_total = self.total
        # costs
        radial = max(0., float(np.linalg.norm(drone[:2]-self.origin[:2]))-w.radius_m)
        vertical = max(0., w.height_lower_world_m-drone[2])+max(0., drone[2]-w.height_upper_world_m)
        self.cost += c.workspace_cost_weight*(radial*radial+vertical*vertical)/self.steps
        if action is not None:
            self.action_cost += c.action_cost_weight*float(np.sum(np.square(action)))/self.steps
        # credit
        rate = 0.
        if not self.struck:
            progress = (self.previous_distance-m['distance'])/self.L
            approach = min(max(m['approach'], 0.), self.v_star)/self.v_star
            rate = c.progress_weight*progress/self.dt+c.approach_weight*approach
            self.accrued += rate*self.dt
            closest = segment_distance(self.previous_tip, p[-1], self.target)
            self.closest = min(self.closest, closest)
            if closest <= c.hit_radius_m and m['speed'] >= self.v_star:
                self.struck, self.hit_time, self.hit_speed = True, float(time), m['speed']
                bonus = c.strike_bonus+c.speed_bonus_per_mps*(m['speed']-self.v_star)
                self.accrued += bonus
                rate += bonus/self.dt
                self.log.append(dict(kind='strike', time_s=time, tip_speed_mps=m['speed'], closest_approach_m=closest,
                                     extension_fraction=m['extension'], amplification=m['amplification']))
        if m['speed'] > self.peak['speed']:
            self.peak = dict(speed=m['speed'], time=float(time), amplification=m['amplification'], extension=m['extension'])
        if math.isfinite(m['turn']):
            self.max_turn = max(self.max_turn, m['turn'])
        self.previous_tip, self.previous_distance = p[-1].copy(), m['distance']
        end = self.struck and time >= min(self.hit_time+c.recovery_s, self.horizon_s)-1e-9
        if end:
            self.recovered = self._inside(drone) and m['attachment_speed'] <= c.recovery_speed_mps
            if self.recovered:
                self.accrued += c.recovery_bonus
                rate += c.recovery_bonus/self.dt
        self.total = self.prefix_return()
        self.time = float(time)
        self._trace(m, rate)
        if end:
            self.terminal = 'recovered' if self.recovered else 'struck'
            self.log.append(dict(kind=self.terminal, time_s=time, return_value=self.total))
        elif time >= self.horizon_s-1e-10:
            self.terminal = 'horizon'
            self.log.append(dict(kind='horizon', time_s=time, return_value=self.total))
        return self.total-previous_total

    def prefix_return(self):
        return self.accrued-self.cost-self.action_cost

    def fail(self, time, reason='numerical_failure'):
        """Termination before the horizon (numerical failure or envelope violation)."""
        if self.terminal is not None or not self.initialized:
            raise RuntimeError('Episode is not active')
        if not self.time <= time <= self.horizon_s+1e-9:
            raise ValueError('Failure must follow the valid prefix')
        self.failure_cost = self.config.failure_cost_per_remaining_s*max(0., self.horizon_s-time)
        previous_total = self.total
        self.total = self.prefix_return()-self.failure_cost
        self.time, self.terminal = float(time), reason
        self.log.append(dict(kind=reason, time_s=time, return_value=self.total, failure_cost=self.failure_cost))
        return self.total-previous_total

    # ------------------------------------------------------------------ reports
    def whip_event(self):
        c = self.config
        return bool(self.peak['speed'] >= self.v_star and self.peak['amplification'] >= c.amplification_ratio
                    and self.peak['extension'] >= c.extension_fraction)

    def features(self):
        m = self.last
        relative = (self.target-self.origin)/self.L
        since_hit = 0. if not self.struck else min(1., (self.time-self.hit_time)/self.config.recovery_s)
        values = [self.time/self.horizon_s, m['tip_target_distance_m']/self.L, m['approach_speed_mps']/self.v_star,
                  m['extension_fraction'], m['tip_speed_mps']/self.v_star, float(self.struck), since_hit,
                  self.accrued/10., self.cost/10., self.action_cost/10., self.total/10., self.peak['speed']/self.v_star,
                  min(m['amplification'], 10.)/10., *relative]
        return np.asarray(values, np.float32)

    def components(self):
        return dict(strike=float(self.struck), whip_event=float(self.whip_event()), recovered=float(self.recovered),
                    peak_tip_speed_mps=self.peak['speed'], amplification_at_peak=min(self.peak['amplification'], 99.),
                    extension_at_peak=self.peak['extension'], hit_time_s=self.hit_time, hit_speed_mps=self.hit_speed,
                    closest_approach_m=min(self.closest, 9.), max_joint_turn_deg=self.max_turn,
                    shaping_credit=self.accrued, workspace_cost=self.cost, action_cost=self.action_cost,
                    failure_cost=self.failure_cost, target_distance_m=self.stage.target_distance_m,
                    speed_threshold_mps=self.v_star)

    def record(self):
        return dict(terminal=self.terminal, terminal_time_s=self.time, events=list(self.log), episode_return=self.total,
                    strike=self.struck, whip_event=self.whip_event(), recovered=self.recovered,
                    target_world_m=self.target.tolist(), stage=dict(target_distance_m=self.stage.target_distance_m,
                    speed_threshold_mps=self.v_star, workspace_radius_m=self.workspace.radius_m),
                    peak=dict(self.peak), closest_approach_m=min(self.closest, 9.), max_joint_turn_deg=self.max_turn,
                    shaping_credit=self.accrued, workspace_cost=self.cost, action_cost=self.action_cost,
                    failure_cost=self.failure_cost, contract=self.config.name,
                    qualification_note='strike: tip within hit_radius_m of the target at or above the stage speed '
                                       'threshold; whip event: peak tip speed, amplification and extension conditions')


class StrikeTask(Task):
    """Batch of StrikeEpisode lanes behind the environment's Task interface. The environment
    sets the curriculum stage (set_stage) before reset; the stage is fixed within a cohort."""
    critic_feature_size = FEATURE_SIZE
    trace_fields = TRACE_FIELDS
    score_component = 'strike'

    def __init__(self, config, environment, rest):
        self.config, self.environment = config, environment
        self.name = config.name
        self.rest = np.asarray(rest, float)
        self.length = float(self.rest.sum())
        self.stage = config.curriculum.final
        self.workspace = environment.workspace
        self.episodes = None

    def set_stage(self, stage, workspace):
        """Curriculum stage for the next cohorts: StrikeStage and the matching CylinderWorkspace."""
        self.stage, self.workspace = stage, workspace

    def reset(self, observation):
        self.episodes = [StrikeEpisode(self.config, self.environment, self.workspace, self.stage, length=self.length,
                                       workspace_origin=o) for o in observation['workspace_origin']]
        for lane, episode in enumerate(self.episodes):
            episode.initialize(observation['cable_positions'][lane], observation['cable_velocities'][lane])

    def observe(self, lane, time_s, observation):
        return self.episodes[lane].observe(time_s, observation['cable_positions'], observation['cable_velocities'],
                                           drone_position=observation['drone_positions'], action=observation.get('action'))

    def fail(self, lane, time_s, reason='numerical_failure'):
        return self.episodes[lane].fail(time_s, reason)

    def done(self, lane):
        return self.episodes[lane].terminal is not None

    def end_time(self, lane):
        return self.episodes[lane].time

    def episode_return(self, lane):
        return self.episodes[lane].total

    def critic_features(self):
        return np.stack([e.features() for e in self.episodes])

    def components(self):
        keys = self.episodes[0].components().keys()
        rows = [e.components() for e in self.episodes]
        return {k: [r[k] for r in rows] for k in keys}

    def record(self, lane):
        return self.episodes[lane].record()

    def trace(self, lane):
        return np.asarray(self.episodes[lane].trace, float)

    def targets(self):
        """(B, 3) world target of every lane (for the actor's frame)."""
        return np.stack([e.target for e in self.episodes])
