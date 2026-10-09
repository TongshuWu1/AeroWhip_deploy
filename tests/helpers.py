"""Small synthetic fixtures shared by the current pipeline checks.

No saved campaign data, trained policy, or flight recording is required.
"""
import torch
import tempfile
from project_paths import ROOT, project_path
from training.configuration import FROZEN_PRESETS, load
from training.ppo import PPO


def project_temporary():
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


class PPOFixture:
    """Reduced observation dimensions exercise the optimizer, not the plant."""
    def agent(self, device='cpu', stage=1):
        settings = load(FROZEN_PRESETS[stage])[0]
        return PPO(dict(settings, device=device, batch_size=8), [0, 1, 2], 6,
                   device, privileged_size=5, encoder_options=dict(goal_columns=[3, 4, 5]))

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


def wave(rest, time):
    """Traveling cable bend: a scoring fixture, not a physical trajectory."""
    material = rest.cumsum(0)-rest/2
    angle = -2.5*torch.exp(-.5*((material[None]-(-.4+2*time[:, None]))/.11)**2)
    edges = rest[None, :, None]*torch.stack((angle.sin(), torch.zeros_like(angle), -angle.cos()), -1)
    p = torch.cat((torch.zeros((len(time), 1, 3), dtype=rest.dtype), edges.cumsum(1)), 1)
    p[..., 2] += 2
    return p[None], torch.gradient(p, spacing=(.01,), dim=(0,))[0][None]
