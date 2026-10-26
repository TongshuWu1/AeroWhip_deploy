"""Reward result container used by the current Stage 1 environment."""
from dataclasses import dataclass
import torch

@dataclass(frozen=True)
class Stage1Score:
    score: torch.Tensor
    valid: torch.Tensor
    components: dict[str, torch.Tensor]
