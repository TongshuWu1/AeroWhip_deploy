"""Shared synthetic fixtures. No saved campaign data, trained policy or flight recording."""
from dataclasses import replace
import tempfile

import torch

from aerowhip.experiment import DEFAULT, load
from aerowhip.paths import ROOT, project_path
from aerowhip.learning.ppo import PPO

# The default experiment file, parsed once.
EXPERIMENT, SNAPSHOT, SCENE = load(DEFAULT)
TASK, ENVIRONMENT = EXPERIMENT.task, EXPERIMENT.environment


def project_temporary():
    """Temporary folder inside the project (file tools only accept project paths)."""
    directory = project_path(ROOT/'tmp')
    directory.mkdir(exist_ok=True)
    return tempfile.TemporaryDirectory(prefix='test-', dir=directory)


def exact_state(a, b):
    if isinstance(a, torch.Tensor):
        return (isinstance(b, torch.Tensor) and a.dtype == b.dtype
                and a.shape == b.shape and torch.equal(a.detach().cpu(), b.detach().cpu()))
    if isinstance(a, dict):
        return isinstance(b, dict) and set(a) == set(b) and all(exact_state(a[k], b[k]) for k in a)
    if isinstance(a, (tuple, list)):
        return isinstance(b, (tuple, list)) and len(a) == len(b) and all(exact_state(x, y) for x, y in zip(a, b))
    return a == b


def small_agent(device='cpu', **changes):
    """PPO on 3 synthetic actor columns and 5 critic inputs: exercises the optimizer, not the plant."""
    return PPO(replace(EXPERIMENT.ppo, **changes), columns=[0, 1, 2], frame_size=6, privileged_size=5,
               device=device, policy_rate_hz=ENVIRONMENT.policy_rate_hz,
               entropy_episode_budget=EXPERIMENT.run.training_episodes)


class PPOFixture:
    """Independent-noise optimizer fixture; AR(1) noise has its own tests."""
    def agent(self, device='cpu'):
        return small_agent(device, batch_size=8, action_noise_correlation_s=None)

    def rollout(self, agent):
        device = agent.device
        frames = torch.randn(4, 11, 6, device=device)
        states = torch.randn(4, 4, 5, device=device)
        raw = torch.zeros(4, 3, 3, device=device)
        logp = torch.zeros(4, 3, device=device)
        mean, log_std, values = raw.clone(), raw.clone(), logp.clone()
        for t in range(3):
            index = torch.full((4,), t*3, device=device, dtype=torch.long)
            action, r, l, m, s, v = agent.collect_action(frames, index, states[:, t])
            raw[:, t], logp[:, t], mean[:, t], log_std[:, t], values[:, t] = r, l, m, s, v
            torch.testing.assert_close(action, r.tanh(), rtol=0, atol=0)
        return dict(collection=agent.collections+1, stride=3, frames=frames, privileged=states,
                    raw=raw, logp=logp, mean=mean, log_std=log_std, values=values,
                    rewards=torch.tensor([[.1,.2,.3],[.4,-1,0],[0,0,1],[.3,0,0]], device=device),
                    lengths=torch.tensor([3,2,3,1], device=device))
