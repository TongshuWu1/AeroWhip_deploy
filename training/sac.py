"""SAC with squashed Gaussian actions, twin critics and learned temperature.

The feed-forward temporal encoder consumes only the observed prefix. Critics
can use simulator state; actor columns are restricted to observable quantities.
No recurrent hidden state, simulator gradient or trajectory parameterization.
Reference: Haarnoja et al., Soft Actor-Critic Algorithms and Applications (2018).
"""

from copy import deepcopy
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
    def __init__(self, columns, hidden, *, history_samples=None, phase_columns=(), goal_columns=()):
        super().__init__()
        self.register_buffer('columns', torch.as_tensor(columns, dtype=torch.long))
        self.temporal = nn.Sequential(nn.Conv1d(len(columns)+1,32,5,stride=2,padding=2),nn.ReLU(),
            nn.Conv1d(32,32,5,stride=2,padding=2),nn.ReLU(),nn.AdaptiveAvgPool1d(16))
        self.output = nn.Sequential(nn.Linear(512+len(columns),hidden),nn.ReLU())
        self.history_samples = history_samples
        if phase_columns:
            self.register_buffer('phase_columns',torch.tensor(phase_columns,dtype=torch.long))
            self.phase_projection = nn.Linear(len(phase_columns),hidden,bias=False)
            nn.init.zeros_(self.phase_projection.weight)
        if goal_columns:
            self.register_buffer('goal_columns',torch.tensor(goal_columns,dtype=torch.long))
            self.goal_projection = nn.Linear(len(goal_columns),hidden,bias=False)
            nn.init.zeros_(self.goal_projection.weight)

    def prepare(self, history, index):
        """Immutable observable prefix only; NEVER cache learned embeddings."""
        phase = (history[torch.arange(len(index),device=index.device),index].index_select(-1,self.phase_columns)
                 if hasattr(self,'phase_columns') else None)
        goal = (history[torch.arange(len(index),device=index.device),index].index_select(-1,self.goal_columns)
                if hasattr(self,'goal_columns') else None)
        if self.history_samples is not None and history.shape[1] != self.history_samples:
            # Fixed source layout until sample450; causal rolling451 thereafter.
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
        prepared = (prefix,current,phase) if phase is not None else (prefix,current)
        return prepared+(goal,) if goal is not None else prepared

    def forward(self, history, index):
        # PPO's fixed-cohort KL passes can reuse these weight-independent inputs.
        # Shapes/strides and ALL learned operations remain the same as ordinary input.
        prepared = history if isinstance(history,tuple) else self.prepare(history,index)
        prefix,current = prepared[:2]
        feature = self.output(torch.cat((self.temporal(prefix.transpose(1,2)).flatten(1),current),-1))
        if hasattr(self,'phase_columns'):
            feature = feature+self.phase_projection(prepared[2])
        if hasattr(self,'goal_columns'):
            feature = feature+self.goal_projection(prepared[3 if hasattr(self,'phase_columns') else 2])
        return feature


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


class Critic(nn.Module):
    def __init__(self, columns, hidden, privileged_size, action_size=3):
        super().__init__()
        self.encoder = HistoryEncoder(columns,hidden)
        self.history_norm = nn.LayerNorm(hidden)
        # Preserve physically scaled state magnitudes. Only the learned history
        # embedding is normalized, per sample, never across parallel lanes.
        self.head = nn.Sequential(nn.Linear(hidden+privileged_size+action_size,hidden),nn.ReLU(),
                                  nn.Linear(hidden,hidden),nn.ReLU(),nn.Linear(hidden,1))

    def forward(self, history, index, action, privileged):
        feature = self.history_norm(self.encoder(history,index))
        return self.head(torch.cat((feature,privileged,action),-1)).squeeze(-1)


class SAC:
    def __init__(self, settings, columns, frame_size, device, action_size=3, *, privileged_size):
        self.settings, self.device = settings, device
        if privileged_size <= 0:
            raise ValueError('History-state SAC requires explicit privileged simulator inputs')
        self.interface = dict(schema='sac_history_state_v1', frame_size=frame_size,
                              actor_columns=list(columns), privileged_size=privileged_size,
                              action_size=action_size, hidden_units=settings['hidden_units'],
                              optimizer_convolution='deterministic_cudnn_v1')
        h = settings['hidden_units']
        self.actor = Actor(columns,h,settings['initial_log_std'],action_size).to(device)
        self.critics = nn.ModuleList([Critic(columns,h,privileged_size,action_size) for _ in range(2)]).to(device)
        self.targets = deepcopy(self.critics).requires_grad_(False)
        self.log_alpha = nn.Parameter(torch.tensor(math.log(settings['initial_temperature']),device=device))
        lr = settings['learning_rate']
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(),lr=lr)
        self.critic_optimizer = torch.optim.Adam(self.critics.parameters(),lr=lr)
        self.alpha_optimizer = torch.optim.Adam([self.log_alpha],lr=lr)
        self.updates = 0
        self.environment_steps = 0

    def set_environment_steps(self, steps):
        if type(steps) is not int or steps < self.environment_steps:
            raise ValueError('Exploration clock requires nondecreasing training transitions')
        self.environment_steps = steps

    @property
    def target_entropy(self):
        schedule = self.settings.get('exploration')
        final = self.settings['target_entropy']
        if schedule is None:  # Historical evaluation/saved settings.
            return final
        fraction = min(1., max(0., (self.environment_steps-schedule['hold_transitions']) /
            (schedule['anneal_end_transitions']-schedule['hold_transitions'])))
        return schedule['initial_target_entropy']+(final-schedule['initial_target_entropy'])*fraction

    @torch.no_grad()
    def action(self, history, index, deterministic=False):
        return self.actor(history,index,deterministic)[0]

    def update(self, batch, *, report_metrics=True):
        # A fixed RNG alone does not make cuDNN convolution gradients repeatable.
        # Scope this to neural updates, preserving enabled/TF32 preferences and
        # restoring global flags even on failure; physics capture is unaffected.
        with torch.backends.cudnn.flags(enabled=torch.backends.cudnn.enabled,
                benchmark=False,deterministic=True,allow_tf32=torch.backends.cudnn.allow_tf32):
            return self._update(batch,report_metrics=report_metrics)

    def _update(self, batch, *, report_metrics=True):
        history,index,next_index,privileged,next_privileged,action,reward,terminal = batch
        with torch.no_grad():
            # Replay/task accounting stays in original units. Scale the complete
            # reward (including costs and failure correction) exactly once here.
            reward = reward*self.settings.get('reward_scale', 1.0)
            next_action,log_p = self.actor(history,next_index)
            target_q = torch.stack([q(history,next_index,next_action,next_privileged) for q in self.targets]).amin(0)
            target = reward + (~terminal).float()*self.settings['discount']*(target_q-self.log_alpha.exp()*log_p)
        self.critic_optimizer.zero_grad(set_to_none=True)
        critic_loss = sum(F.mse_loss(q(history,index,action,privileged),target) for q in self.critics)
        if not torch.isfinite(critic_loss):
            raise FloatingPointError('Nonfinite SAC critic loss')
        critic_loss.backward()
        nn.utils.clip_grad_norm_(self.critics.parameters(),float('inf'),error_if_nonfinite=True)
        self.critic_optimizer.step()
        self.critics.requires_grad_(False)
        self.actor_optimizer.zero_grad(set_to_none=True)
        sampled,log_p = self.actor(history,index)
        q = torch.stack([c(history,index,sampled,privileged) for c in self.critics]).amin(0)
        actor_loss = (self.log_alpha.exp().detach()*log_p-q).mean()
        if not torch.isfinite(actor_loss):
            raise FloatingPointError('Nonfinite SAC actor loss')
        actor_loss.backward()
        nn.utils.clip_grad_norm_(self.actor.parameters(),float('inf'),error_if_nonfinite=True)
        self.actor_optimizer.step()
        self.critics.requires_grad_(True)
        self.alpha_optimizer.zero_grad(set_to_none=True)
        alpha_loss = -(self.log_alpha*(log_p.detach()+self.target_entropy)).mean()
        alpha_loss.backward()
        nn.utils.clip_grad_norm_([self.log_alpha],float('inf'),error_if_nonfinite=True)
        self.alpha_optimizer.step()
        with torch.no_grad():
            for target,current in zip(self.targets.parameters(),self.critics.parameters()):
                target.lerp_(current,self.settings['target_update_tau'])
        self.updates += 1
        metrics = dict(critic_loss=critic_loss.detach(), actor_loss=actor_loss.detach(),
                       temperature=self.log_alpha.exp().detach(), entropy=-log_p.mean().detach(),
                       target_entropy=log_p.new_tensor(self.target_entropy))
        return scalar_metrics(metrics) if report_metrics else metrics

    def state_dict(self):
        return dict(interface=self.interface.copy(), actor=self.actor.state_dict(),critics=self.critics.state_dict(),targets=self.targets.state_dict(),
            log_alpha=self.log_alpha.detach(),actor_optimizer=self.actor_optimizer.state_dict(),
            critic_optimizer=self.critic_optimizer.state_dict(),alpha_optimizer=self.alpha_optimizer.state_dict(),
            updates=self.updates,environment_steps=self.environment_steps)

    def load_state_dict(self, saved):
        if saved.get('interface') != self.interface:
            raise ValueError('SAC actor/critic interface differs; start a fresh run')
        if 'exploration' in self.settings and 'environment_steps' not in saved:
            raise ValueError('Scheduled exploration requires its saved transition clock')
        steps = saved.get('environment_steps', 0)
        if type(steps) is not int or steps < 0:
            raise ValueError('Invalid saved exploration transition clock')
        for name in ('actor','critics','targets','actor_optimizer','critic_optimizer','alpha_optimizer'):
            getattr(self,name).load_state_dict(saved[name])
        with torch.no_grad(): self.log_alpha.copy_(saved['log_alpha'])
        self.updates = saved['updates']
        self.environment_steps = steps


class EpisodeReplay:
    """Compact whole histories; sample transitions uniformly, never across resets."""
    def __init__(self, capacity, samples, frame_size, policy_stride, device, action_size=3, *, privileged_size):
        if (samples-1) % policy_stride or privileged_size <= 0:
            raise ValueError('Replay requires complete policy intervals and privileged state')
        self.capacity,self.stride,self.device = capacity,policy_stride,device
        self.frames = torch.zeros((capacity,samples,frame_size),device=device)
        self.actions = torch.zeros((capacity,(samples-1)//policy_stride,action_size),device=device)
        self.privileged = torch.zeros((capacity,self.actions.shape[1]+1,privileged_size),device=device)
        self.rewards = torch.zeros((capacity,(samples-1)//policy_stride),device=device)
        self.lengths = torch.zeros(capacity,device=device,dtype=torch.long)
        self.cursor = 0

    @property
    def transitions(self):
        return int(self.lengths.sum())

    def add(self, frames, actions, rewards, lengths, privileged):
        if (frames.shape[1:] != self.frames.shape[1:] or actions.shape[1:] != self.actions.shape[1:]
                or privileged.shape[1:] != self.privileged.shape[1:]
                or not (len(frames) == len(actions) == len(rewards) == len(lengths) == len(privileged))):
            raise ValueError('Replay frame/action shapes differ from the current interface')
        if not bool(torch.isfinite(frames).all() & torch.isfinite(actions).all()
                    & torch.isfinite(privileged).all()):
            raise ValueError('Replay observations/actions/privileged states must be finite')
        if rewards.shape != actions.shape[:2] or not bool(torch.isfinite(rewards).all()):
            raise ValueError('Replay requires finite per-transition rewards')
        if bool(((lengths <= 0) | (lengths > self.rewards.shape[1])).any()):
            raise ValueError('Replay episode lengths must cover valid policy transitions')
        for frame,action,reward,length,state in zip(frames,actions,rewards,lengths,privileged):
            i = self.cursor
            self.frames[i].copy_(frame); self.actions[i].copy_(action)
            self.rewards[i].copy_(reward); self.lengths[i]=length
            self.privileged[i].copy_(state)
            self.cursor=(i+1)%self.capacity

    def sample(self, count):
        episode = torch.multinomial(self.lengths.float(),count,replacement=True)
        step = (torch.rand(count,device=self.device)*self.lengths[episode]).long()
        terminal = step+1 == self.lengths[episode]
        reward = self.rewards[episode,step]
        return (self.frames[episode],step*self.stride,(step+1)*self.stride,
                self.privileged[episode,step],self.privileged[episode,step+1],
                self.actions[episode,step],reward,terminal)

    def state_dict(self):
        used = torch.nonzero(self.lengths>0).flatten()
        return dict(schema='sac_history_state_replay_v1',stride=self.stride,capacity=self.capacity,
                    indices=used,frames=self.frames[used],actions=self.actions[used],privileged=self.privileged[used],
                    rewards=self.rewards[used],lengths=self.lengths[used],cursor=self.cursor)

    def load_state_dict(self, saved):
        if (saved.get('schema') != 'sac_history_state_replay_v1'
                or saved.get('stride') != self.stride or saved.get('capacity') != self.capacity
                or saved['actions'].shape[1:] != self.actions.shape[1:]
                or saved['privileged'].shape[1:] != self.privileged.shape[1:]
                or saved['frames'].shape[1:] != self.frames.shape[1:]):
            raise ValueError('Replay control/observation interface differs; start a new run')
        self.lengths.zero_()
        for name in ('frames','actions','rewards','lengths','privileged'):
            getattr(self,name)[saved['indices']]=saved[name]
        self.cursor=saved['cursor']
