"""Observed-prefix neural networks shared by the PPO actor and value model."""
import math
import torch
from torch import nn
from torch.nn import functional as F

def scalar_metrics(metrics):
    """Transfer detached diagnostics together, only when a consumer needs them."""
    if not metrics:
        return {}
    values = torch.stack(tuple(metrics.values())).detach().cpu().tolist()
    return dict(zip(metrics, values))


class HistoryEncoder(nn.Module):
    def __init__(self, columns, hidden, *, history_samples=None):
        super().__init__()
        self.register_buffer('columns', torch.as_tensor(columns, dtype=torch.long))
        self.temporal = nn.Sequential(nn.Conv1d(len(columns)+1,32,5,stride=2,padding=2),nn.ReLU(),
            nn.Conv1d(32,32,5,stride=2,padding=2),nn.ReLU(),nn.AdaptiveAvgPool1d(16))
        self.output = nn.Sequential(nn.Linear(512+len(columns),hidden),nn.ReLU())
        self.history_samples = history_samples

    def prepare(self, history, index):
        """Immutable observable prefix only; NEVER cache learned embeddings."""
        if self.history_samples is not None and history.shape[1] != self.history_samples:
            # Causal window of the latest history_samples frames.
            # Shared by acting, value fitting and PPO prepared-input/KL paths.
            start = (index-(self.history_samples-1)).clamp_min(0)
            sample = start[:,None]+torch.arange(self.history_samples,device=index.device)[None]
            history = history.gather(1,sample.clamp_max(history.shape[1]-1)[:,:,None].expand(-1,-1,history.shape[2]))
            index = index-start
        values = history.index_select(-1, self.columns)
        mask = torch.arange(history.shape[1], device=history.device)[None] <= index[:,None]
        # Mask BEFORE temporal operations: no later frames can leak into a sample.
        prefix = torch.cat((values.masked_fill(~mask[:,:,None],0),mask[:,:,None].to(values.dtype)), -1)
        current = values[torch.arange(len(index),device=index.device),index]
        return prefix,current

    def forward(self, history, index):
        # PPO's fixed-cohort KL passes can reuse these weight-independent inputs.
        # Shapes/strides and ALL learned operations remain the same as ordinary input.
        prepared = history if isinstance(history,tuple) else self.prepare(history,index)
        prefix,current = prepared
        return self.output(torch.cat((self.temporal(prefix.transpose(1,2)).flatten(1),current),-1))


class Actor(nn.Module):
    def __init__(self, columns, hidden, log_std, action_size=3, **encoder_options):
        super().__init__()
        self.encoder = HistoryEncoder(columns,hidden,**encoder_options)
        self.hidden = nn.Sequential(nn.Linear(hidden,hidden),nn.ReLU())
        self.mean, self.scale = nn.Linear(hidden,action_size), nn.Linear(hidden,action_size)
        nn.init.zeros_(self.mean.weight); nn.init.zeros_(self.mean.bias)
        nn.init.zeros_(self.scale.weight); nn.init.constant_(self.scale.bias,log_std)

    def forward(self, history, index, deterministic=False):
        feature = self.hidden(self.encoder(history,index))
        mean = self.mean(feature)
        log_std = self.scale(feature).clamp(-5,2)
        raw = mean if deterministic else mean + log_std.exp()*torch.randn_like(mean)
        action = raw.tanh()
        log_p = (-.5*((raw-mean)/log_std.exp()).square()-log_std-.5*math.log(2*math.pi)
                 -2*(math.log(2)-raw-F.softplus(-2*raw))).sum(-1)
        return action, log_p
