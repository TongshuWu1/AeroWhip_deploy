"""Learner settings, not a whip experiment or physical operating limits."""

from dataclasses import dataclass
import math
import operator


@dataclass(frozen=True)
class LearnerConfiguration:
    context_size: int
    action_size: int
    hidden_sizes: tuple[int, ...] = (64, 64)
    policy_learning_rate: float = 3e-4
    value_learning_rate: float = 1e-3
    update_epochs: int = 10
    mean_kl_limit: float = .01
    covariance_kl_limit: float = .01
    projection_penalty: float = 1.
    max_gradient_norm: float = 1.
    initial_std: float = .2
    min_log_std: float = -5.
    max_log_std: float = 1.
    backtrack_steps: int = 16
    seed: int = 0
    dtype: str = "float32"

    def __post_init__(self):
        for name in ("context_size", "action_size", "update_epochs", "backtrack_steps"):
            if isinstance(getattr(self, name), bool) or operator.index(getattr(self, name)) < 1:
                raise ValueError(f"{name} must be a positive integer")
            object.__setattr__(self, name, int(getattr(self, name)))
        widths = tuple(self.hidden_sizes)
        if not widths or any(isinstance(n, bool) or operator.index(n) < 1 for n in widths):
            raise ValueError("Provide positive hidden-layer sizes")
        object.__setattr__(self, "hidden_sizes", tuple(int(n) for n in widths))
        for name in ("policy_learning_rate", "value_learning_rate", "mean_kl_limit",
                     "covariance_kl_limit", "projection_penalty", "max_gradient_norm", "initial_std"):
            if not math.isfinite(getattr(self, name)) or getattr(self, name) <= 0:
                raise ValueError(f"{name} must be positive and finite")
            object.__setattr__(self, name, float(getattr(self, name)))
        if (not math.isfinite(self.min_log_std) or not math.isfinite(self.max_log_std)
                or not -20 <= self.min_log_std < self.max_log_std <= 5
                or not self.min_log_std < math.log(self.initial_std) < self.max_log_std):
            raise ValueError("Require -20 <= min_log_std < log(initial_std) < max_log_std <= 5")
        if self.dtype not in ("float32", "float64"):
            raise ValueError("Use float32 or float64 for the networks")
        if isinstance(self.seed, bool) or not 0 <= operator.index(self.seed) < 2**63:
            raise ValueError("seed must be an integer in [0, 2**63)")
        object.__setattr__(self, "seed", int(self.seed))
        object.__setattr__(self, "min_log_std", float(self.min_log_std))
        object.__setattr__(self, "max_log_std", float(self.max_log_std))
