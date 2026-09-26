"""On-policy episodic learning with diagonal trust-region projections.

One sampled latent/action and one explicit scalar return per episode. No PPO
clipping, replay buffer, step critic, simulator gradient, or built-in task score.
The final parameter backtracking guard is an AeroWhip implementation addition,
not a claim to reproduce all BBRL-TRPL experiments or guarantee true improvement.
"""

from copy import deepcopy
from dataclasses import asdict, dataclass
import json
from pathlib import Path

import torch

from .configuration import LearnerConfiguration
from .networks import EpisodePolicy, EpisodeValue
from .trust_region import diagonal_kl, gaussian_log_prob, project


@dataclass(frozen=True)
class EpisodeBatch:
    """Read-only by contract: preserve row order and score every sampled action."""

    contexts: torch.Tensor
    actions: torch.Tensor
    latents: torch.Tensor
    old_mean: torch.Tensor
    old_log_std: torch.Tensor
    old_log_prob: torch.Tensor
    old_value: torch.Tensor
    action_matrix: torch.Tensor | None = None
    action_offset: torch.Tensor | None = None


class EpisodicLearner:
    def __init__(self, configuration: LearnerConfiguration, *, device="cpu",
                 context_center=None, context_scale=None, initial_action=None, metadata=None, action_matrix=None):
        self.configuration = configuration
        self.device, self.dtype = torch.device(device), getattr(torch, configuration.dtype)
        if self.device.type not in ("cpu", "cuda"):
            raise ValueError("Use a CPU or CUDA device")
        if self.device.type == "cuda" and self.device.index is None:
            self.device = torch.device("cuda", torch.cuda.current_device())
        self.metadata = json.loads(json.dumps(metadata or {}, allow_nan=False))
        if not isinstance(self.metadata, dict):
            raise ValueError("Checkpoint metadata must be a JSON-compatible dictionary")
        c, a = configuration.context_size, configuration.action_size
        center = self._vector(context_center, c, 0.)
        scale = self._vector(context_scale, c, 1.)
        initial = self._vector(initial_action, a, 0.)
        if not bool((scale > 0).all()) or not bool((initial.abs() < 1).all()):
            raise ValueError("Context scales must be positive; initial actions strictly inside (-1, 1)")
        # Initialize on CPU without changing any process-global random stream.
        with torch.random.fork_rng(devices=[]):
            torch.random.default_generator.manual_seed(configuration.seed)
            self.policy = EpisodePolicy(configuration, center, scale, initial, action_matrix=action_matrix)
            self.value = EpisodeValue(configuration)
        self.policy.to(device=self.device)
        self.value.to(device=self.device, dtype=self.dtype)
        if (not bool(torch.isfinite(self.policy.context_center).all())
                or not bool(torch.isfinite(self.policy.context_scale).all())
                or not bool((self.policy.context_scale > 0).all())):
            raise ValueError("Context normalization must remain finite and positive in the network dtype")
        self.policy_optimizer = torch.optim.Adam(self.policy.parameters(), lr=configuration.policy_learning_rate)
        self.value_optimizer = torch.optim.Adam(self.value.parameters(), lr=configuration.value_learning_rate)
        self.generator = torch.Generator(device=self.device).manual_seed(configuration.seed)
        self.updates = 0
        self._pending = None

    @staticmethod
    def _vector(value, size, default):
        result = torch.full((size,), default, dtype=torch.float64) if value is None else torch.as_tensor(
            value, dtype=torch.float64, device="cpu").detach().clone()
        if result.shape != (size,) or not bool(torch.isfinite(result).all()):
            raise ValueError(f"Expected a finite vector of length {size}")
        return result

    def _contexts(self, contexts):
        if (not isinstance(contexts, torch.Tensor) or contexts.ndim != 2
                or contexts.shape[0] < 1 or contexts.shape[1] != self.configuration.context_size
                or not contexts.is_floating_point()):
            raise ValueError("Contexts must be a nonempty floating (batch, context_size) tensor")
        result = contexts.detach().to(device=self.device, dtype=self.dtype).clone()
        if not bool(torch.isfinite(result).all()):
            raise ValueError("Contexts must remain finite in the network dtype")
        return result

    @torch.no_grad()
    def act(self, contexts):
        """Deterministic inference; does not open a batch or consume sampling RNG."""
        result = self.policy.action(self._contexts(contexts))
        if not bool(torch.isfinite(result).all()):
            raise FloatingPointError("Policy produced nonfinite actions")
        return result

    @torch.no_grad()
    def sample(self, contexts):
        """Open exactly one on-policy batch, to be scored then updated/discarded.

        Send actions to the task's evaluator, converting dtype there if needed.
        Do not silently drop failures, resample feasible actions, or edit actions:
        those operations would change the sampling distribution used by learning.
        """
        if self._pending is not None:
            raise RuntimeError("Update or discard the outstanding episode batch first")
        contexts = self._contexts(contexts)
        mean, log_std = self.policy(contexts)
        mean, log_std = mean.double(), log_std.double()
        noise = torch.randn(mean.shape, device=self.device, dtype=torch.float64, generator=self.generator)
        latents = mean+log_std.exp()*noise
        values = self.value(self.policy.normalize(contexts)).double()
        log_prob = gaussian_log_prob(latents, mean, log_std)
        if not all(bool(torch.isfinite(x).all()) for x in (latents, values, log_prob)):
            raise FloatingPointError("Nonfinite policy distribution or value")
        matrix, offset = self.policy.action_matrix, self.policy.action_offset
        actions = self.policy.transform(latents)
        if not bool(torch.isfinite(actions).all()):
            raise FloatingPointError('Nonfinite transformed actions')
        batch = EpisodeBatch(contexts, actions, latents, mean.clone(), log_std.clone(), log_prob, values,
                             None if matrix is None else matrix.clone(), None if offset is None else offset.clone())
        self._pending = batch
        return batch

    def discard_batch(self):
        """Abandon an entire batch (e.g. evaluator exception), not selected rows."""
        self._pending = None

    def _surrogate(self, batch, returns, *, projected):
        mean, log_std = self.policy(batch.contexts)
        if projected:
            mean, log_std = project(mean, log_std, batch.old_mean, batch.old_log_std,
                mean_limit=self.configuration.mean_kl_limit,
                covariance_limit=self.configuration.covariance_kl_limit)
        ratio = (gaussian_log_prob(batch.latents, mean, log_std)-batch.old_log_prob).exp()
        advantage = returns-batch.old_value
        # No batch-centering: preserve the baseline estimator and one-row case.
        # Positive scaling affects step conditioning, not which samples are good.
        advantage = advantage/advantage.square().mean().sqrt().clamp_min(1e-8)
        return (ratio*advantage).mean(), mean, log_std

    @torch.no_grad()
    def _accept_policy(self, batch, returns, previous):
        """Guard the actual inference network, not just projected training outputs.

        Bounds apply at this batch's contexts only. No global/context-unseen KL
        or real-return guarantee is made. Rejected/shortened updates reset Adam
        to its pre-update state, avoiding moments from an unaccepted candidate.
        """
        candidate = {k: v.clone() for k, v in self.policy.state_dict().items()}
        baseline = (returns-batch.old_value)
        baseline = (baseline/baseline.square().mean().sqrt().clamp_min(1e-8)).mean()
        cfg = self.configuration
        tolerance = 1e-7 if self.dtype == torch.float32 else 1e-12
        for attempt in range(cfg.backtrack_steps):
            fraction = .5**attempt
            self.policy.load_state_dict({k: previous[k]+fraction*(v-previous[k]) for k, v in candidate.items()})
            objective, mean, log_std = self._surrogate(batch, returns, projected=False)
            mean_kl, cov_kl = diagonal_kl(mean, log_std, batch.old_mean, batch.old_log_std)
            acceptable = (torch.isfinite(objective) & torch.isfinite(mean_kl).all() & torch.isfinite(cov_kl).all()
                & (mean_kl.max() <= cfg.mean_kl_limit+tolerance)
                & (cov_kl.max() <= cfg.covariance_kl_limit+tolerance)
                & (objective >= baseline-tolerance))
            if bool(acceptable):
                return fraction
        self.policy.load_state_dict(previous)
        return 0.

    def update(self, batch: EpisodeBatch, returns):
        """Consume one complete batch with explicit finite episode returns.

        Failure scores are the task's responsibility; there is no hidden penalty
        or success filter. Nonfinite scores fail before either optimizer changes.
        Learner/optimizer states roll back if a numerical update fails.
        """
        if batch is not self._pending or batch is None:
            raise ValueError("Use this learner's outstanding, unconsumed batch")
        if not isinstance(returns, torch.Tensor) or returns.shape != (len(batch.contexts),) or not returns.is_floating_point():
            raise ValueError("Provide one floating scalar return per sampled episode")
        returns = returns.detach().to(device=self.device, dtype=torch.float64).clone()
        if not bool(torch.isfinite(returns).all()):
            raise ValueError("All episodes need finite explicit returns, including task-defined failures")
        with torch.no_grad():
            mean, log_std = self.policy(batch.contexts)
            for current, saved in ((self.policy.action_matrix, batch.action_matrix),
                                   (self.policy.action_offset, batch.action_offset)):
                if (current is None) != (saved is None) or (current is not None and not torch.equal(current, saved)):
                    raise ValueError('Action transform changed after sampling')
            if (not torch.equal(mean.double(), batch.old_mean) or not torch.equal(log_std.double(), batch.old_log_std)
                    or not torch.equal(batch.actions, self.policy.transform(batch.latents))
                    or not torch.equal(gaussian_log_prob(batch.latents, mean, log_std), batch.old_log_prob)):
                raise ValueError("Policy or sampled batch changed after sampling")
        previous = deepcopy(self.policy.state_dict())
        previous_value = deepcopy(self.value.state_dict())
        previous_optimizer = deepcopy(self.policy_optimizer.state_dict())
        previous_value_optimizer = deepcopy(self.value_optimizer.state_dict())
        cfg = self.configuration
        try:
            for _ in range(cfg.update_epochs):
                objective, projected_mean, projected_log_std = self._surrogate(batch, returns, projected=True)
                mean, log_std = self.policy(batch.contexts)
                penalty = sum(diagonal_kl(mean, log_std, projected_mean.detach(), projected_log_std.detach())).mean()
                loss = -objective+cfg.projection_penalty*penalty
                if not bool(torch.isfinite(loss)):
                    raise FloatingPointError("Nonfinite policy objective")
                self.policy_optimizer.zero_grad(set_to_none=True)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.policy.parameters(), cfg.max_gradient_norm, error_if_nonfinite=True)
                self.policy_optimizer.step()
                with torch.no_grad():
                    self.policy.log_std.clamp_(cfg.min_log_std, cfg.max_log_std)
            fraction = self._accept_policy(batch, returns, previous)
            if fraction < 1:
                self.policy_optimizer.load_state_dict(previous_optimizer)
            for _ in range(cfg.update_epochs):
                predicted = self.value(self.policy.normalize(batch.contexts)).double()
                value_loss = (predicted-returns).square().mean()
                if not bool(torch.isfinite(value_loss)):
                    raise FloatingPointError("Nonfinite value loss")
                self.value_optimizer.zero_grad(set_to_none=True)
                value_loss.backward()
                torch.nn.utils.clip_grad_norm_(self.value.parameters(), cfg.max_gradient_norm, error_if_nonfinite=True)
                self.value_optimizer.step()
            with torch.no_grad():
                objective, mean, log_std = self._surrogate(batch, returns, projected=False)
                mean_kl, cov_kl = diagonal_kl(mean, log_std, batch.old_mean, batch.old_log_std)
                value_loss = (self.value(self.policy.normalize(batch.contexts)).double()-returns).square().mean()
                diagnostics = torch.stack((returns.mean(), objective, value_loss, mean_kl.max(), cov_kl.max()))
                if not bool(torch.isfinite(diagnostics).all()):
                    raise FloatingPointError("Nonfinite final learner state")
        except Exception:
            self.policy.load_state_dict(previous)
            self.value.load_state_dict(previous_value)
            self.policy_optimizer.load_state_dict(previous_optimizer)
            self.value_optimizer.load_state_dict(previous_value_optimizer)
            raise
        self.updates += 1
        self._pending = None
        names = ("return_mean", "surrogate", "value_loss", "mean_kl_max", "covariance_kl_max")
        return dict(zip(names, diagnostics.cpu().tolist()), update=self.updates,
                    episodes=len(returns), accepted_fraction=fraction)

    def save(self, path):
        """Save at an update boundary. Never overwrite an existing checkpoint.

        Metadata should identify context/action conventions, model and reward
        versions. Caller-owned simulator/context-sampler states are not included.
        """
        if self._pending is not None:
            raise RuntimeError("Finish or discard the outstanding batch before saving")
        schema = 'episodic_diagonal_trpl_v1' if self.policy.action_matrix is None else 'episodic_affine_tanh_trpl_v2'
        payload = dict(schema=schema, configuration=asdict(self.configuration),
            metadata=self.metadata, policy=self.policy.state_dict(), value=self.value.state_dict(),
            policy_optimizer=self.policy_optimizer.state_dict(), value_optimizer=self.value_optimizer.state_dict(),
            updates=self.updates, generator=self.generator.get_state(), generator_device=str(self.device))
        # Parent directories are intentionally caller-owned; no automatic run tree.
        with Path(path).open("xb") as stream:
            torch.save(payload, stream)

    @classmethod
    def load(cls, path, *, device="cpu", expected_metadata=None, reset_sampling_rng=False):
        """Restore networks, optimizers and sampling RNG on the same device type.

        Across CPU/CUDA types, require reset_sampling_rng=True to start a new
        stream from the configured seed. Cross-device bitwise resume is not claimed.
        """
        payload = torch.load(path, map_location="cpu", weights_only=True)
        schema = payload.get('schema')
        if schema not in ('episodic_diagonal_trpl_v1', 'episodic_affine_tanh_trpl_v2'):
            raise ValueError("Unsupported learner checkpoint")
        has_transform = 'action_matrix' in payload['policy'] and 'action_offset' in payload['policy']
        if (schema == 'episodic_affine_tanh_trpl_v2') != has_transform:
            raise ValueError('Checkpoint schema does not match its action transform')
        if expected_metadata is not None and payload["metadata"] != expected_metadata:
            raise ValueError("Checkpoint metadata does not match the requested task/model/reward")
        same_device_type = torch.device(payload["generator_device"]).type == torch.device(device).type
        if not same_device_type and not reset_sampling_rng:
            raise ValueError("CPU/CUDA RNG states differ; pass reset_sampling_rng=True for a new sampling stream")
        config = LearnerConfiguration(**payload["configuration"])
        result = cls(config, device=device, metadata=payload["metadata"],
                     action_matrix=payload['policy'].get('action_matrix'))
        result.policy.load_state_dict(payload["policy"])
        result.value.load_state_dict(payload["value"])
        result.policy_optimizer.load_state_dict(payload["policy_optimizer"])
        result.value_optimizer.load_state_dict(payload["value_optimizer"])
        result.updates = payload["updates"]
        if same_device_type and not reset_sampling_rng:
            result.generator.set_state(payload["generator"])
        for module in (result.policy, result.value):
            if not all(bool(torch.isfinite(x).all()) for x in module.state_dict().values()):
                raise ValueError("Checkpoint contains nonfinite network state")
        if not bool((result.policy.context_scale > 0).all()):
            raise ValueError("Checkpoint contains invalid context scales")
        return result
