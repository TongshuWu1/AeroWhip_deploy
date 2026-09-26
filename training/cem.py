"""Smoothed diagonal latent CEM with a fixed bounded action transform."""

from dataclasses import asdict, dataclass
import json
import math
from pathlib import Path

import torch


@dataclass(frozen=True)
class CEMConfiguration:
    elites: int
    alpha: float
    initial_std: float
    std_floor: float
    valid_elites_only: bool = False  # Preserve historical search behavior.

    def __post_init__(self):
        if type(self.valid_elites_only) is not bool:
            raise ValueError('valid_elites_only must be boolean')
        if type(self.elites) is not int or self.elites < 2:
            raise ValueError('CEM requires at least two elites')
        if not math.isfinite(self.alpha) or not 0 < self.alpha <= 1:
            raise ValueError('CEM alpha must be in (0,1]')
        if not 0 < self.std_floor <= self.initial_std or not math.isfinite(self.initial_std):
            raise ValueError('Require finite initial_std >= std_floor > 0')


class DiagonalCEM:
    """One fixed-context search distribution; not a contextual policy or iCEM."""

    def __init__(self, configuration, initial_action, action_matrix, *, seed, device, metadata):
        self.configuration = configuration
        self.seed = seed
        self.metadata = json.loads(json.dumps(metadata, allow_nan=False))
        initial = torch.as_tensor(initial_action, dtype=torch.float64, device=device).detach().clone()
        matrix = torch.as_tensor(action_matrix, dtype=torch.float64, device=device).detach().clone()
        if (initial.ndim != 1 or initial.numel() < 1 or matrix.shape != (len(initial), len(initial))
                or not bool(torch.isfinite(initial).all() & (initial.abs() < 1).all())
                or not bool(torch.isfinite(matrix).all()) or int(torch.linalg.matrix_rank(matrix)) != len(initial)):
            raise ValueError('Require an interior initial action and a finite full-rank square transform')
        if type(seed) is not int or not 0 <= seed < 2**63:
            raise ValueError('Provide a nonnegative integer seed')
        self.matrix, self.offset = matrix, initial.atanh()
        self.mean = torch.zeros_like(initial)
        self.variance = torch.full_like(initial, configuration.initial_std**2)
        self.generator = torch.Generator(device=device).manual_seed(seed)
        self.updates = 0

    def transform(self, latent):
        return (latent @ self.matrix.T+self.offset).tanh()

    @torch.no_grad()
    def act(self, batch_size=1):
        return self.transform(self.mean[None].expand(batch_size, -1))

    @torch.no_grad()
    def sample(self, count):
        if type(count) is not int or count < self.configuration.elites:
            raise ValueError('Sample count must cover the elite set')
        latent = self.mean+self.variance.sqrt()*torch.randn((count, len(self.mean)),
            device=self.mean.device, dtype=self.mean.dtype, generator=self.generator)
        return latent, self.transform(latent)

    @torch.no_grad()
    def update(self, latent, returns, *, valid=None):
        if (latent.ndim != 2 or latent.shape[1] != len(self.mean) or returns.shape != latent.shape[:1]
                or latent.device != self.mean.device or returns.device != self.mean.device
                or latent.dtype != self.mean.dtype or returns.dtype != self.mean.dtype
                or len(latent) < self.configuration.elites
                or not bool(torch.isfinite(latent).all() & torch.isfinite(returns).all())):
            raise ValueError('Provide a finite evaluated latent batch and aligned returns')
        valid_count = len(returns)
        ranked = returns
        if self.configuration.valid_elites_only:
            if (not isinstance(valid, torch.Tensor) or valid.shape != returns.shape
                    or valid.dtype != torch.bool or valid.device != returns.device):
                raise ValueError('Valid-only CEM requires an aligned boolean validity mask')
            valid_count = int(valid.sum())
            if valid_count < self.configuration.elites:
                # Count the attempted batch, not a distribution update. Do not
                # fit invalid/tied failure rows or silently resample a new batch.
                self.updates += 1
                return dict(latent_std_mean=float(self.variance.sqrt().mean()), elite_return_mean=None,
                            valid_candidates=valid_count, distribution_updated=False)
            ranked = returns.masked_fill(~valid, -float('inf'))
        elif valid is not None:
            raise ValueError('A validity mask requires valid_elites_only=True')
        selected = ranked.topk(self.configuration.elites)
        elite = latent[selected.indices]
        alpha = self.configuration.alpha
        self.mean = (1-alpha)*self.mean+alpha*elite.mean(0)
        self.variance = ((1-alpha)*self.variance+alpha*elite.var(0, unbiased=False)).clamp_min(self.configuration.std_floor**2)
        self.updates += 1
        return dict(latent_std_mean=float(self.variance.sqrt().mean()), elite_return_mean=float(selected.values.mean()),
                    valid_candidates=valid_count if self.configuration.valid_elites_only else None,
                    distribution_updated=True)

    def save(self, path):
        with Path(path).open('xb') as stream:
            torch.save(dict(schema='diagonal_cem_v1', configuration=asdict(self.configuration), seed=self.seed,
                mean=self.mean, variance=self.variance, matrix=self.matrix, offset=self.offset,
                generator=self.generator.get_state(), updates=self.updates, metadata=self.metadata), stream)

    @classmethod
    def load(cls, path, *, device, expected_metadata):
        state = torch.load(path, map_location=device, weights_only=True)
        if state.get('schema') != 'diagonal_cem_v1' or state['metadata'] != expected_metadata:
            raise ValueError('CEM checkpoint schema or provenance does not match')
        obj = cls(CEMConfiguration(**state['configuration']), state['offset'].tanh(), state['matrix'],
                  seed=state['seed'], device=device, metadata=state['metadata'])
        for key in ('mean', 'variance', 'offset'):
            value = state[key]
            if value.shape != obj.mean.shape or value.dtype != torch.float64 or not bool(torch.isfinite(value).all()):
                raise ValueError(f'Invalid CEM {key}')
            setattr(obj, key, value.clone())
        if (not bool((obj.variance >= obj.configuration.std_floor**2).all())
                or type(state['updates']) is not int or state['updates'] < 0):
            raise ValueError('Invalid CEM variance or update count')
        obj.updates = state['updates']
        obj.generator.set_state(state['generator'].cpu())
        return obj
