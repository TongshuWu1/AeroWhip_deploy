"""Typed settings for the experiment file's "run" and "ppo" sections."""
from dataclasses import dataclass, fields

from aerowhip.envs.configuration import integer, number, require_keys


@dataclass(frozen=True)
class RunConfig:
    """Compute and budget. evaluation_episodes deterministic episodes (independent lanes, the
    final curriculum stage, randomised model draws) are scored at every evaluation; with one,
    evaluation_actor_batch sets the actor's batch layout for that single lane (one physical lane
    repeated, not extra trials)."""
    device: str
    seed: int
    parallel_environments: int
    evaluation_actor_batch: int
    evaluation_episodes: int
    training_episodes: int
    evaluation_interval_episodes: int

    @classmethod
    def from_dict(cls, raw):
        require_keys(cls, raw, 'run')
        return cls(**raw)

    def __post_init__(self):
        if self.device not in ('cpu', 'cuda'):
            raise ValueError('device must be cpu or cuda')
        integer('seed', self.seed, 0)
        for name in ('parallel_environments', 'evaluation_actor_batch', 'evaluation_episodes', 'training_episodes',
                     'evaluation_interval_episodes'):
            integer(name, getattr(self, name))

    @property
    def collections(self):
        """Number of collections needed for the episode budget (last one may be narrower)."""
        return (self.training_episodes+self.parallel_environments-1)//self.parallel_environments

    def evaluation_due(self, collection, episodes_before, episodes_after):
        """Evaluate after the first collection, at each interval crossing and at the end."""
        if collection == 1 or episodes_after == self.training_episodes:
            return True
        interval = self.evaluation_interval_episodes
        return episodes_before//interval != episodes_after//interval


@dataclass(frozen=True)
class EntropySchedule:
    """Entropy coefficient = final + (initial − final)·exp(−decay_exponent · n/N),
    n completed training episodes, N the original episode budget."""
    initial_weight: float
    final_weight: float
    decay_exponent: float

    def __post_init__(self):
        for f in fields(self):
            object.__setattr__(self, f.name, number(f.name, getattr(self, f.name)))
        if self.final_weight > self.initial_weight:
            raise ValueError('Entropy floor must not exceed its initial weight')


@dataclass(frozen=True)
class PPOConfig:
    """Clipped PPO with an asymmetric critic. post_step_kl_trials enables the cohort-KL
    acceptance check (up to that many halved learning-rate trials); None disables it.
    action_noise_correlation_s sets AR(1) exploration noise; None uses independent noise."""
    learning_rate: float
    batch_size: int
    hidden_units: int
    initial_log_std: float
    discount: float
    reward_scale: float
    epochs: int
    clip_ratio: float
    gae_lambda: float
    value_loss_weight: float
    max_grad_norm: float
    target_kl: float
    post_step_kl_trials: int | None
    entropy: EntropySchedule
    action_noise_correlation_s: float | None

    @classmethod
    def from_dict(cls, raw):
        require_keys(cls, raw, 'ppo')
        require_keys(EntropySchedule, raw['entropy'], 'ppo.entropy')
        return cls(**dict(raw, entropy=EntropySchedule(**raw['entropy'])))

    def __post_init__(self):
        for name in ('batch_size', 'hidden_units', 'epochs'):
            integer(name, getattr(self, name))
        for name in ('learning_rate', 'reward_scale', 'clip_ratio', 'gae_lambda', 'value_loss_weight',
                     'max_grad_norm', 'target_kl', 'discount'):
            object.__setattr__(self, name, number(name, getattr(self, name)))
        object.__setattr__(self, 'initial_log_std', number('initial_log_std', self.initial_log_std, positive=False))
        if not -5 <= self.initial_log_std <= 2:
            raise ValueError('initial_log_std must lie in [-5, 2]')
        if self.discount > 1 or self.gae_lambda > 1 or self.clip_ratio >= 1:
            raise ValueError('Require discount and GAE lambda in (0, 1], clip ratio in (0, 1)')
        if self.post_step_kl_trials is not None and not 1 <= integer('post_step_kl_trials', self.post_step_kl_trials) <= 12:
            raise ValueError('post_step_kl_trials must lie in 1..12')
        if self.action_noise_correlation_s is not None:
            object.__setattr__(self, 'action_noise_correlation_s',
                               number('action_noise_correlation_s', self.action_noise_correlation_s))
