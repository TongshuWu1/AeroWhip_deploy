"""Reward result container used by the current Stage 1 adapter."""
from dataclasses import dataclass
import torch

@dataclass(frozen=True)
class Stage1Score:
    score: torch.Tensor
    valid: torch.Tensor
    has_event: torch.Tensor
    event_indices: torch.Tensor
    components: dict[str, torch.Tensor]
    # Best complete event available at each 100-Hz observation. Invalid future
    # samples never erase or contribute to an already observed valid prefix.
    causal_best: torch.Tensor
