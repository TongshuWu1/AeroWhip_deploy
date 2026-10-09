"""Matching-width curriculum; no physics, optimizer or whip-classifier changes."""
from dataclasses import dataclass, replace
import math


def validate_target_curriculum(settings, task=None):
    """Fixed-center target-size curriculum for PPO, distinct from old SAC widths."""
    c = settings.get('target_curriculum')
    if c is None:
        return
    if settings.get('algorithm') != 'ppo' or 'curriculum' in settings:
        raise ValueError('Target curriculum requires PPO without the legacy SAC curriculum')
    keys = {'widths_m', 'heights_m', 'consecutive_collections', 'minimum_qualified_fraction'}
    gates = {'maximum_mean_workspace_cost', 'maximum_workspace_rms_excess_m'}
    precision = settings.get('stage3_role') == 'precision_v1'
    allowed = keys | {'qualification'} if precision else keys
    if (not isinstance(c, dict) or set(c)-gates != allowed
            or len(set(c)&gates) != (0 if precision else 1)):
        raise ValueError('Provide explicit target-size curriculum and progression gates')
    for name in ('widths_m', 'heights_m'):
        x = c[name]
        if (not isinstance(x, list) or len(x) < 2 or
                any(isinstance(v, bool) or not isinstance(v, (int, float))
                    or not math.isfinite(v) or v <= 0 for v in x) or
                any(a <= b for a, b in zip(x, x[1:]))):
            raise ValueError('Target dimensions must be strictly decreasing positive finite lists')
    if len(c['widths_m']) != len(c['heights_m']):
        raise ValueError('Each target level needs matching width and height')
    if precision and (c['qualification'] != 'geometric_forward_v1'
            or c['widths_m'][-1] != .1 or c['heights_m'][-1] != .1):
        raise ValueError('Precision Stage3 requires geometric forward hits and final +/-5cm per axis')
    if type(c['consecutive_collections']) is not int or c['consecutive_collections'] < 1:
        raise ValueError('Require a positive consecutive-collection count')
    gate = next(iter(set(c)&gates), None)
    for name, upper in [('minimum_qualified_fraction', 1.)] + ([(gate, float('inf'))] if gate else []):
        v = c[name]
        if isinstance(v, bool) or not isinstance(v, (int, float)) or not math.isfinite(v) or not 0 <= v <= upper:
            raise ValueError('Invalid target curriculum gate: '+name)
    if gate == 'maximum_workspace_rms_excess_m':
        from tasks.whip.workspace import WorkspacePenalty
        penalty = WorkspacePenalty(**settings['workspace_penalty'])
        if penalty.weight <= 0:
            raise ValueError('Physical RMS gate requires positive boundary weight to recover mean-square excess')
    if task is not None and (task.strike_surface is None or
            task.strike_surface.width_m != c['widths_m'][-1] or
            task.strike_surface.height_m != c['heights_m'][-1]):
        raise ValueError('Frozen base task must describe the final target, not the current training level')


def target_workspace_gate(settings, workspace_cost):
    """Separate physical eligibility from reward weight; legacy settings unchanged.

    Weighted cost = w * mean((radial_excess/d)^2+(vertical_excess/d)^2).
    Historical input is the ALL-lane mean full-horizon reward cost. Staged-v1
    callers instead supply w/d² times the ALL-lane mean of individual executed
    prefix mean-squares. Do NOT pass their fixed-reference reward cost here.
    No numerical tolerance or gate relaxation is introduced.
    """
    c = settings['target_curriculum']
    if settings.get('stage3_role') == 'precision_v1':
        # Excursions remain measured/penalized, not a curriculum failure gate.
        return dict(workspace_gate_metric='diagnostic_only_full_horizon_v1',
            workspace_gate_value=workspace_cost if math.isfinite(workspace_cost) else None,
            workspace_gate_limit=None, workspace_gate_pass=math.isfinite(workspace_cost))
    if 'maximum_workspace_rms_excess_m' not in c:
        threshold = c['maximum_mean_workspace_cost']
        return dict(workspace_gate_metric='weighted_workspace_cost_v1',
            workspace_gate_value=workspace_cost if math.isfinite(workspace_cost) else None,
            workspace_gate_limit=threshold,
            workspace_gate_pass=math.isfinite(workspace_cost) and workspace_cost <= threshold)
    penalty = settings['workspace_penalty']
    weight, distance = penalty['weight'], penalty['distance_scale_m']
    threshold = c['maximum_workspace_rms_excess_m']
    finite = math.isfinite(workspace_cost) and workspace_cost >= 0
    rms = distance*math.sqrt(workspace_cost/weight) if finite else None
    # Compare in the supplied mean-square-equivalent units, not sqrt at equality.
    equivalent_cost_limit = weight*(threshold/distance)**2
    return dict(workspace_gate_metric=('executed_prefix_equal_episode_rms_excess_m_v2'
        if 'training_stage' in settings else 'rms_workspace_excess_m_v1'),
        workspace_gate_value=rms, workspace_gate_limit=threshold,
        workspace_gate_equivalent_weighted_cost_limit=equivalent_cost_limit,
        workspace_gate_pass=finite and workspace_cost <= equivalent_cost_limit)


@dataclass
class TargetCurriculum:
    level: int = 0
    streak: int = 0

    def validate(self, settings):
        c = settings['target_curriculum']
        if (type(self.level) is not int or not 0 <= self.level < len(c['widths_m']) or
                type(self.streak) is not int or self.streak < 0):
            raise ValueError('Invalid target curriculum checkpoint state')

    def configuration(self, strict, settings):
        self.validate(settings)
        c = settings['target_curriculum']
        width,height=c['widths_m'][self.level],c['heights_m'][self.level]
        surface=replace(strict.strike_surface,width_m=width,height_m=height)
        matching=strict.matching
        if settings.get('stage3_role') == 'precision_v1':
            matching=replace(matching,center_sigma_m=max(.05,max(width,height)/4))
        if matching.strike_mechanics_scale is not None:
            # Keep plane-distance guidance broad while progressively tightening
            # lateral/vertical overflow and centering. Legacy ladders unchanged.
            extent=max(width,height)
            surface=replace(surface,overflow_width_m=max(.05,min(.35,extent/2)))
            matching=replace(matching,center_sigma_m=max(.05,extent/4))
        return replace(strict,strike_surface=surface,matching=matching)

    def observe(self, settings, qualified_fraction, workspace_cost, *, valid_fraction=1.):
        self.validate(settings)
        c = settings['target_curriculum']
        passed = (math.isfinite(qualified_fraction) and math.isfinite(workspace_cost)
            and math.isfinite(valid_fraction) and valid_fraction == 1.
            and qualified_fraction >= c['minimum_qualified_fraction']
            and target_workspace_gate(settings,workspace_cost)['workspace_gate_pass'])
        self.streak = self.streak+1 if passed else 0

    def ready(self, settings, *, valid, qualified, workspace_cost):
        self.validate(settings)
        c = settings['target_curriculum']
        return (self.level < len(c['widths_m'])-1 and
            self.streak >= c['consecutive_collections'] and valid and qualified
            and target_workspace_gate(settings,workspace_cost)['workspace_gate_pass'])

    def advance(self, settings, *, valid, qualified, workspace_cost):
        if not self.ready(settings, valid=valid, qualified=qualified, workspace_cost=workspace_cost):
            raise ValueError('Target advancement requires sustained collection and updated-policy evaluation gates')
        self.level += 1
        self.streak = 0


def validate_curriculum(settings, task=None):
    raw = settings.get('curriculum')
    if raw is None:
        return
    expected = {'normal_widths_m', 'angle_widths_deg', 'consecutive_batches',
                'minimum_strike_fraction', 'maximum_mean_workspace_cost'}
    if set(raw) != expected:
        raise ValueError('Unexpected curriculum settings')
    for key in ('normal_widths_m', 'angle_widths_deg'):
        values = raw[key]
        if len(values) != 3 or any(isinstance(v, bool) or not math.isfinite(v) or v <= 0 for v in values):
            raise ValueError('Curriculum requires three positive finite widths')
        if not values[0] > values[1] > values[2]:
            raise ValueError('Curriculum widths must strictly decrease')
    if raw['angle_widths_deg'][0] >= 90:
        raise ValueError('Direction matching widths must be below 90 degrees')
    if type(raw['consecutive_batches']) is not int or raw['consecutive_batches'] < 1:
        raise ValueError('Provide a positive consecutive-batch count')
    for key, upper in (('minimum_strike_fraction', 1), ('maximum_mean_workspace_cost', float('inf'))):
        v = raw[key]
        if isinstance(v, bool) or not math.isfinite(v) or not 0 <= v <= upper:
            raise ValueError('Invalid curriculum threshold: '+key)
    if settings.get('evaluation_environments',settings['parallel_environments']) != 64 or 'workspace_penalty' not in settings:
        raise ValueError('This curriculum uses 64-episode evaluation batches and soft workspace penalties')
    if task is not None:
        if (task.matching.mechanics_credit_weight != 0 or task.strike_surface is None
                or task.strike_surface.overflow_width_m is None
                or task.strike_surface.approach_width_m != raw['normal_widths_m'][-1]
                or task.matching.direction_width_deg != raw['angle_widths_deg'][-1]):
            raise ValueError('Base task must be the strict final curriculum reward, with zero independent mechanics credit')


@dataclass
class Curriculum:
    stage: int = 0
    streak: int = 0
    review_round: int | None = None
    reviewed: bool = False

    def __post_init__(self):
        if type(self.stage) is not int or not 0 <= self.stage <= 2:
            raise ValueError('Invalid curriculum stage')
        if type(self.streak) is not int or self.streak < 0 or type(self.reviewed) is not bool:
            raise ValueError('Invalid curriculum checkpoint state')
        if self.stage > 0 and not self.reviewed:
            raise ValueError('Advanced curriculum requires recorded motion review')
        if self.review_round is not None and (type(self.review_round) is not int or self.review_round < 1
                                             or self.stage != 0 or self.reviewed):
            raise ValueError('Invalid pending review')

    def configuration(self, strict, settings):
        c = settings['curriculum']
        return replace(strict,
            matching=replace(strict.matching, direction_width_deg=c['angle_widths_deg'][self.stage]),
            strike_surface=replace(strict.strike_surface, approach_width_m=c['normal_widths_m'][self.stage]))

    def observe(self, settings, strike_fraction, mean_workspace_cost, round_number):
        c = settings['curriculum']
        passed = (math.isfinite(strike_fraction) and math.isfinite(mean_workspace_cost)
                  and strike_fraction >= c['minimum_strike_fraction']
                  and mean_workspace_cost <= c['maximum_mean_workspace_cost'])
        self.streak = self.streak+1 if passed else 0
        ready = self.stage < 2 and self.streak >= c['consecutive_batches']
        if ready and not self.reviewed:
            self.review_round = round_number
        return ready

    def advance(self, settings):
        if (self.stage >= 2 or not self.reviewed
                or self.streak < settings['curriculum']['consecutive_batches']):
            raise ValueError('Advancement requires consecutive passing batches and motion review')
        self.stage += 1
        self.streak = 0
        self.review_round = None
