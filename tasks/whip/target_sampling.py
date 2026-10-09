"""Manual saved-policy target placement, using original scalar task scorers.

The training grid sampler is retired. Its metadata validator remains only to
read historical snapshots; training.configuration.require_launchable blocks it.
Manual simulation tests do not establish learned target adaptation.
"""
from dataclasses import fields, replace
import numpy as np
import torch

from .learned_recovery import make_scorer
from .stage1 import Stage1Score

SAMPLER = 'lateral_vertical_grid_v1'


def validate_target_sampler(settings):
    """Validate archived metadata without authorizing training."""
    if 'target_sampler' not in settings:
        return
    if (settings['target_sampler'] != SAMPLER or settings.get('algorithm') != 'ppo'
            or settings.get('training_stage') not in (3, 4)
            or settings.get('target_observation') != 'initial_relative_position_v1'):
        raise ValueError('Target sampler requires goal-conditioned PPO Stage3/4')


class TargetScorers:
    """Group identical goals, evaluate the unmodified single-goal scorer per group."""
    def __init__(self, env, centers):
        centers = np.asarray(centers, dtype=np.float64)
        if centers.shape != (env.batch, 3) or not np.isfinite(centers).all():
            raise ValueError('One finite world XYZ target per physical lane required')
        self.center = env.model.tensor(centers)[:, None]
        self.batch, self.groups = env.batch, []
        unique, assignment = np.unique(centers, axis=0, return_inverse=True)
        for i, center in enumerate(unique):
            env.configuration.workspace.validate_placement(env.origin.tolist(), center)
            cfg = replace(env.configuration, request=(*env.configuration.request[:2],
                          float(center[2]), env.configuration.request[3]))
            scorer = make_scorer(env.model.rest, env.time, cfg, initial_position=env.origin)
            (scorer.base if cfg.recovery is not None else scorer).center = env.model.tensor(center)
            lanes = torch.as_tensor(np.flatnonzero(assignment == i), device=env.device)
            self.groups.append((lanes, scorer))
        self.direction, self.speed = self.groups[0][1].direction, self.groups[0][1].speed

    @torch.no_grad()
    def __call__(self, positions, velocities, valid, **kwargs):
        merged = None
        for lanes, scorer in self.groups:
            subset = {k: v.index_select(0, lanes) if isinstance(v, torch.Tensor) else v
                      for k, v in kwargs.items()}
            score = scorer(positions.index_select(0, lanes), velocities.index_select(0, lanes),
                           valid.index_select(0, lanes), **subset)
            if merged is None:
                def empty(v):
                    return torch.empty((self.batch, *v.shape[1:]), dtype=v.dtype, device=v.device)
                merged = Stage1Score(**{f.name: ({k: empty(v) for k, v in score.components.items()}
                    if f.name == 'components' else empty(getattr(score, f.name))) for f in fields(score)})
            if merged.components.keys() != score.components.keys():
                raise ValueError('Inconsistent per-goal score fields')
            for f in fields(score):
                if f.name == 'components':
                    for k, v in score.components.items():
                        merged.components[k].index_copy_(0, lanes, v)
                else:
                    getattr(merged, f.name).index_copy_(0, lanes, getattr(score, f.name))
        return merged


@torch.no_grad()
def configure_targets(env, centers):
    if 0 < env.decision < env.decisions and bool(env.active.any()):
        raise RuntimeError('Do not replace targets during an incomplete physical cohort')
    scorer = TargetScorers(env, centers)
    env.scorer = scorer
    if env.relative_target_input:
        env.goal_relative = ((scorer.center[:, 0]-env.origin)/env.model.rest.sum()).float()
    # Caller resets once, after configuration, recording the new initial goal.


def target_surface(env, lane):
    surface = env.configuration.strike_surface.description(env.origin.tolist(), env.configuration.request)
    center = env.scorer.center
    surface['center_world_m'] = (center[lane, 0] if center.ndim == 3 else center).tolist()
    surface['normal_world'] = env.scorer.direction.tolist()
    return surface
