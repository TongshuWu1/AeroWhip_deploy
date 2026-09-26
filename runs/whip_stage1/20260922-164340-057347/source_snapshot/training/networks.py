"""A bounded diagonal-Gaussian policy and an independent context-value baseline."""

import math

import torch
from torch import nn


def mlp(inputs, widths, outputs):
    layers = []
    for width in widths:
        layers.extend((nn.Linear(inputs, width), nn.Tanh()))
        inputs = width
    layers.append(nn.Linear(inputs, outputs))
    return nn.Sequential(*layers)


class EpisodePolicy(nn.Module):
    """Diagonal latent Gaussian, optionally followed by a fixed affine tanh map.

    Context normalization is explicit and frozen. The caller defines context
    meanings/order and supplies an initial action if zero is inappropriate.
    Bounded actions do not guarantee feasible trajectories or physical safety.
    """

    def __init__(self, config, center, scale, initial_action, *, action_matrix=None):
        super().__init__()
        matrix = None
        if action_matrix is not None:
            matrix = torch.as_tensor(action_matrix, device='cpu', dtype=torch.float64).detach().clone()
            if (matrix.shape != (config.action_size, config.action_size)
                    or not bool(torch.isfinite(matrix).all())
                    or int(torch.linalg.matrix_rank(matrix)) != config.action_size):
                raise ValueError('Action matrix must be finite, square and numerically full rank')
        self.register_buffer("context_center", center.clone())
        self.register_buffer("context_scale", scale.clone())
        self.mean = mlp(config.context_size, config.hidden_sizes, config.action_size)
        self.log_std = nn.Parameter(torch.full((config.action_size,), math.log(config.initial_std)))
        self.min_log_std, self.max_log_std = config.min_log_std, config.max_log_std
        with torch.no_grad():
            self.mean[-1].weight.zero_()
            self.mean[-1].bias.copy_(torch.atanh(initial_action) if matrix is None else torch.zeros_like(initial_action))
        # Match the existing network/normalization precision, while keeping the
        # fixed action transform in float64 independently of network dtype.
        self.to(dtype=getattr(torch, config.dtype))
        self.register_buffer('action_matrix', matrix)
        self.register_buffer('action_offset', None if matrix is None else torch.atanh(initial_action).double().clone())

    def normalize(self, context):
        return (context-self.context_center)/self.context_scale

    def forward(self, context):
        mean = self.mean(self.normalize(context))
        log_std = self.log_std.clamp(self.min_log_std, self.max_log_std).expand_as(mean)
        return mean, log_std

    def transform(self, latent):
        """Same fixed map for sampling/inference; densities stay in latent space."""
        if self.action_matrix is None:
            return latent.tanh()
        return (latent.double() @ self.action_matrix.T+self.action_offset).tanh()

    def action(self, context):
        """Transformed latent mean, not the expected bounded action."""
        return self.transform(self(context)[0])


class EpisodeValue(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.network = mlp(config.context_size, config.hidden_sizes, 1)

    def forward(self, normalized_context):
        return self.network(normalized_context).squeeze(-1)
