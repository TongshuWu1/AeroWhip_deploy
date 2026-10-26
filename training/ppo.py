"""Asymmetric on-policy PPO; finite episodes, squashed actions and explicit entropy.

Clipped surrogate: Schulman et al. (2017), arXiv:1707.06347. This module does
not change the simulator/task. Hyperparameters are AeroWhip settings, not an
asserted reproduction of Learning to Throw's unavailable configuration.
"""
from copy import deepcopy
import math
import torch
from torch import nn
from torch.nn import functional as F

from training.sac import Actor, HistoryEncoder, scalar_metrics


def validate_stage3_actor_training(settings):
    """Archived combined mask, retained for strict saved-learner reconstruction.

    Current fixed-target Stage3 trains the full actor. Launch validation rejects
    this retired tag for both new runs and continuation of historical training.
    """
    if 'stage3_actor_training' not in settings:
        return False
    if (settings['stage3_actor_training'] != 'combined_goal_head_v1'
            or settings.get('algorithm') != 'ppo' or settings.get('training_stage') != 3
            or settings.get('stage3_role') != 'precision_v1'
            or settings.get('target_observation') != 'initial_relative_position_v1'):
        raise ValueError('Combined goal/head training requires relative-target precision PPO Stage3')
    return True


def validate_ppo(c):
    keys = {'epochs', 'clip_ratio', 'gae_lambda', 'value_loss_weight',
            'max_grad_norm', 'target_kl', 'entropy'}
    if set(c)-{'post_step_kl', 'transfer_entropy_clock', 'action_noise'} != keys or type(c['epochs']) is not int or c['epochs'] < 1:
        raise ValueError('PPO requires explicit epochs, clipping, GAE, value/gradient/KL and entropy settings')
    if c.get('transfer_entropy_clock', 'inherit') not in ('inherit', 'stage_local'):
        raise ValueError('Transfer entropy clock must be inherit or stage_local')
    noise = c.get('action_noise')
    if 'action_noise' in c and (not isinstance(noise, dict)
            or set(noise) != {'mode','correlation_time_s'} or noise['mode'] != 'ar1_v1'
            or isinstance(noise['correlation_time_s'],bool)
            or not isinstance(noise['correlation_time_s'],(int,float))
            or not math.isfinite(noise['correlation_time_s']) or noise['correlation_time_s'] <= 0):
        raise ValueError('AR1 noise requires a version and positive correlation time in seconds')
    guard = c.get('post_step_kl')
    if guard is not None:
        if (not isinstance(guard, dict) or set(guard) != {'maximum_trials'}
                or type(guard['maximum_trials']) is not int or not 1 <= guard['maximum_trials'] <= 12):
            raise ValueError('Post-step KL requires 1 to 12 bounded Adam trials')
    elif 'post_step_kl' in c:
        raise ValueError('Omit post_step_kl to retain the historical update rule')
    for key in keys-{'epochs', 'entropy'}:
        value = c[key]
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or value <= 0:
            raise ValueError('Invalid PPO '+key)
    if c['clip_ratio'] >= 1 or c['gae_lambda'] > 1:
        raise ValueError('Require clip ratio in (0,1), GAE lambda in (0,1]')
    e = c['entropy']
    episode_clock = set(e) == {'initial_weight', 'final_weight', 'hold_fraction', 'anneal_end_fraction'}
    exponential = (set(e) == {'mode', 'initial_weight', 'final_weight', 'decay_episodes'}
                   and e['mode'] == 'exponential_episodes_v1')
    budget_exponential = (set(e) == {'mode','initial_weight','final_weight','decay_exponent'} and e['mode']=='exponential_budget_v1')
    if not episode_clock and not exponential and not budget_exponential and set(e) != {'initial_weight', 'final_weight', 'hold_transitions', 'anneal_end_transitions'}:
        raise ValueError('PPO entropy requires weights and one explicit episode-budget or historical transition clock')
    for key in ('initial_weight', 'final_weight'):
        value = e[key]
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or value < 0:
            raise ValueError('Invalid PPO entropy weight')
    if e['initial_weight'] <= e['final_weight']:
        raise ValueError('PPO entropy schedule must decrease')
    if exponential:
        if type(e['decay_episodes']) is not int or e['decay_episodes'] <= 0:
            raise ValueError('Exponential entropy requires a positive integer episode timescale')
    elif budget_exponential:
        if isinstance(e['decay_exponent'],bool) or not isinstance(e['decay_exponent'],(int,float)) or not math.isfinite(e['decay_exponent']) or e['decay_exponent']<=0:
            raise ValueError('Budget exponential entropy needs a finite positive exponent')
    elif episode_clock:
        for key in ('hold_fraction', 'anneal_end_fraction'):
            if isinstance(e[key], bool) or not isinstance(e[key], (int, float)) or not math.isfinite(e[key]):
                raise ValueError('Invalid PPO entropy episode fraction')
        if not 0 <= e['hold_fraction'] < e['anneal_end_fraction'] <= 1:
            raise ValueError('Require 0 <= entropy hold < decay end <= 1 of training episode budget')
    elif (type(e['hold_transitions']) is not int or type(e['anneal_end_transitions']) is not int
          or not 0 <= e['hold_transitions'] < e['anneal_end_transitions']):
        raise ValueError('Invalid PPO entropy transition clock')


def squashed_log_prob(raw, mean, log_std):
    # Store raw samples; never recover them with atanh(saturated float actions).
    return (-.5*((raw-mean)/log_std.exp()).square()-log_std-.5*math.log(2*math.pi)
            -2*(math.log(2)-raw-F.softplus(-2*raw))).sum(-1)


def ar1_distribution(mean, log_std, previous_mean, previous_log_std, previous_raw, first, rho):
    """Density conditional on fixed logged action history, with live policy fits.

    Gradients through BOTH observation-conditioned marginals are intentional.
    The first decision uses the stationary marginal, not innovation variance.
    """
    conditional_mean = mean+rho*(log_std-previous_log_std).exp()*(previous_raw-previous_mean)
    conditional_std = log_std+.5*math.log1p(-rho*rho)
    return (torch.where(first[:,None],mean,conditional_mean),
            torch.where(first[:,None],log_std,conditional_std))


def gaussian_kl(old_mean, old_std_log, mean, std_log):
    # Tanh is a fixed bijection in exact arithmetic: transformed-policy KL equals
    # Gaussian KL. Unlike a one-action estimate this is nonnegative per state.
    return (std_log-old_std_log +
        (old_std_log.exp().square()+(old_mean-mean).square())/(2*std_log.exp().square())-.5).sum(-1)


def bounded_actor_step(parameters, optimizer, criterion, threshold, maximum_trials):
    """Empirical cohort-KL acceptance, not TRPO or guaranteed improvement.

    Retry the same gradient with lr factors 1, 1/2, ... from identical actor
    parameters AND Adam moments/clocks. Rejection restores both exactly; an
    accepted trial advances Adam once and retains the nominal learning rate.
    No RNG draw is performed here. Caller supplies all frozen cohort states.
    """
    parameters = list(parameters)
    before = [p.detach().clone() for p in parameters]
    saved = deepcopy(optimizer.state_dict())
    rates = [group['lr'] for group in optimizer.param_groups]
    trials = []

    def restore():
        with torch.no_grad():
            for p, old in zip(parameters, before):
                p.copy_(old)
        optimizer.load_state_dict(deepcopy(saved))

    try:
        for attempt in range(maximum_trials):
            restore()
            factor = .5**attempt
            for group, rate in zip(optimizer.param_groups, rates):
                group['lr'] = rate*factor
            optimizer.step()
            tensors = parameters + [v for state in optimizer.state.values()
                for v in state.values() if isinstance(v, torch.Tensor)]
            # Adam step counters may be CPU tensors even for a CUDA actor.
            finite = all(bool(torch.stack([torch.isfinite(v).all() for v in tensors
                if v.device == device]).all()) for device in {v.device for v in tensors})
            kl = float(criterion()) if finite else float('nan')
            finite = finite and math.isfinite(kl)
            accepted = finite and 0 <= kl <= threshold
            trials.append(dict(factor=factor, kl=kl if finite else None, accepted=accepted))
            if accepted:
                for group, rate in zip(optimizer.param_groups, rates):
                    group['lr'] = rate
                return dict(accepted=True, factor=factor, trials=trials)
    except BaseException:
        restore()
        raise
    restore()
    return dict(accepted=False, factor=None, trials=trials)


def explained_variance(prediction, target):
    """Population explained variance; undefined for constant targets.

    Negative values are retained, not clipped into an apparent fitting success.
    """
    variance = target.var(unbiased=False)
    if float(variance) == 0.: return None
    return float(1-(target-prediction).var(unbiased=False)/variance)


def clipped_loss(log_prob, old_log_prob, advantage, clip):
    ratio = (log_prob-old_log_prob).exp()
    return -torch.minimum(ratio*advantage, ratio.clamp(1-clip, 1+clip)*advantage).mean(), ratio


@torch.no_grad()
def advantages(rewards, values, lengths, discount, lam):
    """Complete task episodes: horizon and failure both terminal, no bootstrap.

    Unexecuted suffixes and resets never enter GAE. Rewards include the scorer's
    terminal clawback and are scaled once by the caller before this operation.
    """
    if (rewards.shape != values.shape or rewards.ndim != 2 or lengths.shape != rewards.shape[:1]
            or lengths.dtype not in (torch.int32, torch.int64)
            or bool(((lengths <= 0) | (lengths > rewards.shape[1])).any())):
        raise ValueError('GAE requires complete finite episodes and valid lengths')
    if not bool(torch.isfinite(rewards).all() & torch.isfinite(values).all()):
        raise ValueError('Nonfinite PPO rollout')
    out = torch.zeros_like(rewards)
    carry = torch.zeros_like(rewards[:, 0])
    for t in reversed(range(rewards.shape[1])):
        active = t < lengths
        continuing = t+1 < lengths
        next_value = values[:, t+1] if t+1 < values.shape[1] else torch.zeros_like(carry)
        delta = rewards[:, t]-values[:, t]+discount*continuing*next_value
        carry = torch.where(active, delta+discount*lam*continuing*carry, 0.)
        out[:, t] = carry
    valid = torch.arange(rewards.shape[1], device=rewards.device)[None] < lengths[:, None]
    return out, torch.where(valid, out+values, 0.), valid


class PPOActor(Actor):
    # Same observable-history architecture/distribution as the retained actor.
    # Deterministic Actor.forward remains compatible with saved evaluation.
    def distribution(self, history, index):
        feature = self.hidden(self.encoder(history, index))
        return self.mean(feature), self.scale(feature).clamp(-5, 2)

    def sample(self, history, index):
        mean, log_std = self.distribution(history, index)
        raw = mean+log_std.exp()*torch.randn_like(mean)
        return raw.tanh(), raw, squashed_log_prob(raw, mean, log_std), mean, log_std


class FrozenWhipRecoveryActor(PPOActor):
    """Observable PD return backbone and learned correction; source whip frozen."""
    def __init__(self, columns, hidden, log_std, action_size, *, recovery_policy,
                 velocity_limits, **encoder_options):
        super().__init__(columns,hidden,log_std,action_size,**encoder_options)
        self.requires_grad_(False)
        self.recovery_policy = dict(recovery_policy)
        self.register_buffer('recovery_limits',torch.tensor(velocity_limits,dtype=torch.float32))
        self.recovery_head = nn.Sequential(nn.Linear(len(columns)+2,hidden),nn.ReLU(),
                                           nn.Linear(hidden,action_size))
        nn.init.zeros_(self.recovery_head[-1].weight)
        nn.init.zeros_(self.recovery_head[-1].bias)
        self.recovery_log_std = nn.Parameter(torch.full((action_size,),recovery_policy['initial_log_std']))

    def distribution(self, history, index):
        return self._distribution(history,index)

    def _distribution(self, history, index, frozen=None):
        # `frozen` is private fixed-chunk reuse for RNG-free cohort KL only.
        # Acting and shuffled optimizer minibatches retain the ordinary path.
        if isinstance(history,tuple):
            prepared = history
            current,phase = prepared[1:3]
            active = phase[:,:1]>.5
            recovery_only = bool(active.all())
        else:
            observed = history[torch.arange(len(index),device=index.device),index]
            phase = observed.index_select(-1,self.encoder.phase_columns)
            active = phase[:,:1]>.5
            # Negative indices keep the historical rolling-prefix behavior.
            recovery_only = bool((active[:,0] & (index>=0)).all())
            if recovery_only:
                current = observed.index_select(-1,self.encoder.columns)
            else:
                prepared = self.encoder.prepare(history,index)
                current,phase = prepared[1:3]
        # The frozen CNN has no stateful layers or RNG. Keep its original full
        # batch arithmetic for mixed/pre-hit inputs; skip only an unused branch.
        if not recovery_only:
            old_mean,old_std = (super().distribution(prepared,None)
                               if frozen is None else frozen)
        c = self.recovery_policy
        position = current[:,:3]*c['cable_length_m']
        velocity = current[:,12:15]*4
        command = -c['position_gain_s_inv']*position-c['velocity_damping']*velocity
        normalized = (command/self.recovery_limits).clamp(-.999,.999)
        mean = torch.atanh(normalized)+self.recovery_head(torch.cat((current,phase),-1))
        std = self.recovery_log_std.clamp(-5,2).expand_as(mean)
        if recovery_only:
            # Preserve the original materialized log-std layout for PPO callers.
            return mean,std.contiguous()
        return torch.where(active,mean,old_mean),torch.where(active,std,old_std)

    def forward(self, history, index, deterministic=False):
        mean,log_std = self.distribution(history,index)
        raw = mean if deterministic else mean+log_std.exp()*torch.randn_like(mean)
        return raw.tanh(),squashed_log_prob(raw,mean,log_std)


def make_ppo_actor(settings, columns, action_size=3, **options):
    combined = validate_stage3_actor_training(settings)
    if (settings.get('target_observation') == 'initial_relative_position_v1') != ('goal_columns' in options):
        raise ValueError('Actor factory requires the selected relative-target observation layout')
    if 'goal_columns' in options:
        goal_columns = options['goal_columns']
        if (not isinstance(goal_columns,(list,tuple)) or len(goal_columns) != 3
                or any(type(v) is not int or v < 0 for v in goal_columns)
                or len(set(goal_columns)) != 3 or set(columns)&set(goal_columns)
                or set(options.get('phase_columns',()))&set(goal_columns)):
            raise ValueError('Goal observations require three distinct additional XYZ columns')
    common = (columns,settings['hidden_units'],settings['initial_log_std'],action_size)
    if 'stage4_recovery_policy' in settings:
        return FrozenWhipRecoveryActor(*common,recovery_policy=settings['stage4_recovery_policy'],
            velocity_limits=settings['velocity_limit_mps'],**options)
    actor = PPOActor(*common,**options)
    if combined:
        # Reconstruct the archived mask/optimizer layout only. It is not a
        # current training option (require_launchable rejects this contract).
        actor.encoder.requires_grad_(False)
        actor.encoder.goal_projection.requires_grad_(True)
    return actor


class Value(nn.Module):
    def __init__(self, columns, hidden, privileged_size, **encoder_options):
        super().__init__()
        self.encoder = HistoryEncoder(columns, hidden,**encoder_options)
        self.history_norm = nn.LayerNorm(hidden)
        self.head = nn.Sequential(nn.Linear(hidden+privileged_size, hidden), nn.ReLU(),
                                  nn.Linear(hidden, hidden), nn.ReLU(), nn.Linear(hidden, 1))

    def forward(self, history, index, privileged):
        return self.head(torch.cat((self.history_norm(self.encoder(history, index)), privileged), -1)).squeeze(-1)


class PPO:
    def __init__(self, settings, columns, frame_size, device, action_size=3, *, privileged_size, encoder_options=None):
        validate_ppo(settings['ppo'])
        if privileged_size <= 0:
            raise ValueError('Asymmetric PPO needs explicit privileged value inputs')
        self.settings, self.device = settings, device
        self.interface = dict(schema='ppo_history_state_v1', frame_size=frame_size,
            actor_columns=list(columns), privileged_size=privileged_size, action_size=action_size,
            hidden_units=settings['hidden_units'], optimizer_convolution='deterministic_cudnn_v1')
        if settings.get('action_interface') == 'acceleration_direct_v1':
            self.interface.update(action_interface='acceleration_direct_v1',
                acceleration_limit_mps2=list(settings['acceleration_limit_mps2']),
                policy_rate_hz=settings['policy_rate_hz'])
        noise = settings['ppo'].get('action_noise')
        self.noise_rho = None if noise is None else math.exp(-1/(settings['policy_rate_hz']*noise['correlation_time_s']))
        if noise is not None:
            self.interface['action_noise'] = dict(noise, rho=self.noise_rho,
                likelihood='history_conditional_squashed_gaussian_v1')
        h = settings['hidden_units']
        options = dict(encoder_options or {})
        phase_options = {k:v for k,v in options.items() if k != 'goal_columns'}
        goal = settings.get('target_observation')
        if goal is not None:
            goal_columns = list(range(frame_size-3,frame_size))
            if (goal != 'initial_relative_position_v1' or settings.get('training_stage') not in (1,2,3,4)
                    or options.get('goal_columns') != goal_columns
                    or set(columns)&set(goal_columns)):
                raise ValueError('Relative target input requires its explicit appended XYZ columns')
            self.interface.update(target_observation=goal,goal_columns=goal_columns)
        elif 'goal_columns' in options:
            raise ValueError('Goal columns require explicit target_observation settings')
        if validate_stage3_actor_training(settings):
            self.interface['stage3_actor_training'] = settings['stage3_actor_training']
        if phase_options:
            if settings.get('continuous_whip_contract')=='continuous_directional_whip_four_second_v6' and phase_options==dict(history_samples=401):
                self.interface.update(history_layout='continuous_directional_whip_401_v1',time_reference_samples=400)
            elif settings.get('continuous_whip_contract') in ('continuous_horizontal_whip_v1', 'continuous_horizontal_whip_soft_workspace_v2','continuous_horizontal_whip_distance_v3','continuous_horizontal_whip_full_horizon_v4', 'continuous_directional_whip_full_horizon_v5') and phase_options == dict(history_samples=701):
                self.interface.update(history_layout='continuous_horizontal_whip_701_v1',time_reference_samples=700)
            elif settings.get('retained_fold_delivery_contract') == 'retained_fold_delivery_v1' and phase_options == dict(history_samples=701):
                self.interface.update(history_layout='retained_fold_delivery_701_v1',time_reference_samples=700)
            elif settings.get('horizontal_fold_contract') == 'horizontal_fold_outreach_v2' and phase_options == dict(history_samples=701):
                self.interface.update(history_layout='horizontal_fold_701_v2',time_reference_samples=700)
            elif 'stage2_first_hit' in settings and phase_options == dict(history_samples=451):
                self.interface.update(history_layout='stage2_first_hit_451_rolling_v1',time_reference_samples=450)
            elif settings.get('stage4_role') != 'learned_recovery_v1' or phase_options != dict(history_samples=451,phase_columns=[96,97]):
                raise ValueError('Only explicit Stage4 supports versioned phase/history transfer')
            else:
                self.interface.update(history_layout='stage4_451_rolling_v1',phase_columns=[96,97])
        if 'stage4_recovery_policy' in settings:
            if not phase_options:
                raise ValueError('Frozen whip recovery requires explicit phase/history inputs')
            self.interface['recovery_architecture'] = 'frozen_whip_pd_residual_v1'
        self.actor = make_ppo_actor(settings,columns,action_size,**options).to(device)
        self.value = Value(columns, h, privileged_size,**options).to(device)
        self.actor_optimizer = torch.optim.Adam((p for p in self.actor.parameters() if p.requires_grad), lr=settings['learning_rate'])
        self.value_optimizer = torch.optim.Adam(self.value.parameters(), lr=settings['learning_rate'])
        self.updates = self.value_updates = self.environment_steps = self.collections = 0
        self.entropy_step_offset = 0
        self.completed_episodes = self.entropy_episode_offset = 0
        self.entropy_episode_budget = settings['training_episodes']
        self._kl_inputs = None  # Ephemeral, scoped to ONE optimize call, never serialized.
        self._kl_frozen = None  # Only the architecturally frozen Stage4 branch.
        self._kl_input_cache_limit = 512*1024*1024  # Bounded additional memory, not a learning setting.

    def initialize_actor(self, saved):
        """Explicit policy transfer, NOT resume: keep fresh value/Adam/RNG/clocks."""
        if self.updates or self.value_updates or self.collections or self.environment_steps:
            raise ValueError('Actor initialization requires an unused PPO learner')
        source = dict(saved.get('interface', {}))
        expected = dict(self.interface)
        # Actor-only transfer can change a training mask; exact resume cannot.
        # Stage4 freezes the complete Stage3 actor, regardless of its old mask.
        has_source_training = 'stage3_actor_training' in source
        source_training = source.pop('stage3_actor_training', None)
        if has_source_training and (source_training != 'combined_goal_head_v1'
                or source.get('target_observation') != 'initial_relative_position_v1'):
            raise ValueError('Unknown saved Stage3 actor training contract')
        expected.pop('stage3_actor_training', None)
        expected.pop('recovery_architecture',None)
        layout = expected.pop('history_layout',None)
        if layout in ('continuous_horizontal_whip_701_v1','continuous_directional_whip_401_v1'):expected['history_layout']=layout
        if layout == 'stage2_first_hit_451_rolling_v1':
            expected.pop('time_reference_samples')
            if source.get('history_layout') == layout:
                source.pop('history_layout')
                if source.pop('time_reference_samples',None) != 450:
                    raise ValueError('First-hit source time coordinate changed')
            elif 'history_layout' in source:
                raise ValueError('First-hit transfer requires the original451sample history layout')
        migration = layout == 'stage4_451_rolling_v1'
        if migration:
            expected.pop('phase_columns')
            expected['frame_size'] -= 2
            if 'goal_columns' in expected:
                expected['goal_columns'] = [v-2 for v in expected['goal_columns']]
        goal_migration = 'target_observation' in expected and 'target_observation' not in source
        if goal_migration:
            expected.pop('target_observation')
            expected.pop('goal_columns')
            expected['frame_size'] -= 3
        keys = set(expected)-{'privileged_size'}
        if set(source) != set(expected) or any(source[k] != expected[k] for k in keys):
            raise ValueError('Actor observation/distribution interface differs from the selected checkpoint')
        weights = saved.get('actor', {})
        if not weights or any(not bool(torch.isfinite(v).all()) for v in weights.values()):
            raise ValueError('Actor initialization requires finite saved weights')
        if 'target_observation' in source and (not isinstance(weights.get('encoder.goal_columns'),torch.Tensor)
                or not torch.equal(weights['encoder.goal_columns'].cpu(),torch.tensor(source['goal_columns']))):
            raise ValueError('Saved actor goal columns disagree with its interface')
        offset = saved.get('environment_steps', -1)
        inherited = saved.get('entropy_step_offset', 0)
        if type(offset) is not int or offset < 0 or type(inherited) is not int or inherited < 0:
            raise ValueError('Invalid source exploration clock')
        if self.settings['ppo'].get('transfer_entropy_clock', 'inherit') == 'inherit':
            if saved.get('entropy_clock', 'transitions_v1') != self.entropy_clock:
                raise ValueError('Cannot inherit entropy clocks across units; choose stage_local actor transfer')
            self.entropy_step_offset = offset+inherited
            if self.entropy_clock == 'completed_episodes_v1':
                count, previous, budget = (saved.get(k) for k in
                    ('completed_episodes', 'entropy_episode_offset', 'entropy_episode_budget'))
                if any(type(v) is not int or v < 0 for v in (count, previous)) or type(budget) is not int or budget <= 0:
                    raise ValueError('Invalid source episode exploration clock')
                self.entropy_episode_offset = count+previous
                self.entropy_episode_budget = budget
        if migration or goal_migration:
            additions = ({'encoder.phase_columns','encoder.phase_projection.weight'} if migration else set())
            if goal_migration:
                additions |= {'encoder.goal_columns','encoder.goal_projection.weight'}
            if isinstance(self.actor,FrozenWhipRecoveryActor):
                additions |= {'recovery_limits','recovery_log_std'} | {
                    k for k in self.actor.state_dict() if k.startswith('recovery_head.')}
            if set(self.actor.state_dict())-set(weights) != additions or set(weights)-set(self.actor.state_dict()):
                raise ValueError('Unexpected versioned actor input migration fields')
            copied = self.actor.state_dict()
            goal_columns = copied.get('encoder.goal_columns')
            copied.update(weights)
            if goal_columns is not None:
                # Stage4 phase insertion moves appended goal columns, not their weights.
                copied['encoder.goal_columns'] = goal_columns
            if migration and bool(copied['encoder.phase_projection.weight'].count_nonzero()):
                raise ValueError('Stage4 phase adapter must initialize to zero')
            if goal_migration and bool(copied['encoder.goal_projection.weight'].count_nonzero()):
                raise ValueError('Relative target adapter must initialize to zero')
            self.actor.load_state_dict(copied,strict=True)
        else:
            self.actor.load_state_dict(weights, strict=True)

    @property
    def entropy_clock(self):
        return ('completed_episodes_v1' if ('hold_fraction' in self.settings['ppo']['entropy']
                or self.settings['ppo']['entropy'].get('mode') in ('exponential_episodes_v1','exponential_budget_v1'))
                else 'transitions_v1')

    def set_completed_episodes(self, episodes):
        # Completed cohorts only; failures count, evaluation/discarded prefixes do not.
        if type(episodes) is not int or episodes < self.completed_episodes:
            raise ValueError('PPO entropy clock requires nondecreasing completed training episodes')
        self.completed_episodes = episodes

    def set_environment_steps(self, steps):
        if type(steps) is not int or steps < self.environment_steps:
            raise ValueError('PPO entropy clock requires nondecreasing training transitions')
        self.environment_steps = steps

    @property
    def entropy_weight(self):
        e = self.settings['ppo']['entropy']
        if e.get('mode') == 'exponential_budget_v1':
            progress=(self.completed_episodes+self.entropy_episode_offset)/self.entropy_episode_budget
            return e['final_weight']+(e['initial_weight']-e['final_weight'])*math.exp(-e['decay_exponent']*progress)
        if e.get('mode') == 'exponential_episodes_v1':
            # Absolute completed-episode clock: lane count and budget extensions
            # must not reset or stretch the approved decay timescale.
            elapsed = self.completed_episodes+self.entropy_episode_offset
            return e['final_weight']+(e['initial_weight']-e['final_weight'])*math.exp(-elapsed/e['decay_episodes'])
        if self.entropy_clock == 'completed_episodes_v1':
            progress = (self.completed_episodes+self.entropy_episode_offset)/self.entropy_episode_budget
            f = min(1., max(0., (progress-e['hold_fraction']) /
                               (e['anneal_end_fraction']-e['hold_fraction'])))
        else:
            f = min(1., max(0., (self.environment_steps+self.entropy_step_offset-e['hold_transitions']) /
                               (e['anneal_end_transitions']-e['hold_transitions'])))
        return e['initial_weight']+(e['final_weight']-e['initial_weight'])*f

    def conditional_distribution(self, history, index, previous_raw=None, previous_index=None):
        mean, log_std = self.actor.distribution(history,index)
        if self.noise_rho is None:
            return mean, log_std
        first = index == 0
        if previous_raw is None or previous_index is None:
            if not bool(first.all()):
                raise ValueError('AR1 decisions after reset require the preceding logged raw action and index')
            return mean, log_std
        previous_mean, previous_std = self.actor.distribution(history,previous_index)
        return ar1_distribution(mean,log_std,previous_mean,previous_std,previous_raw,first,self.noise_rho)

    def rollout_distribution(self, r, b, t):
        previous = (t-1).clamp_min(0)
        return self.conditional_distribution(r['frames'][b],t*r['stride'],
            r['raw'][b,previous],previous*r['stride'])

    @torch.no_grad()
    def action(self, history, index, deterministic=False, *, previous_raw=None, previous_index=None):
        if deterministic or self.noise_rho is None:
            return self.actor(history,index,deterministic)[0]
        mean, log_std = self.conditional_distribution(history,index,previous_raw,previous_index)
        return (mean+log_std.exp()*torch.randn_like(mean)).tanh()

    @torch.no_grad()
    def collect_action(self, history, index, privileged, *, previous_raw=None, previous_index=None):
        # Explicit history arguments mean there is no episode state to leak into
        # evaluation/checkpoints. The worker resets these at every env.reset().
        mean, log_std = self.conditional_distribution(history,index,previous_raw,previous_index)
        raw = mean+log_std.exp()*torch.randn_like(mean)
        return raw.tanh(), raw, squashed_log_prob(raw,mean,log_std), mean, log_std, self.value(history,index,privileged)

    def optimize(self, rollout, *, progress=None):
        # Only one fresh completed collection; no replay, target Q or SAC alpha.
        if rollout['collection'] != self.collections+1:
            raise ValueError('PPO collection was already used or is out of order')
        try:
            with torch.backends.cudnn.flags(enabled=torch.backends.cudnn.enabled,
                    benchmark=False, deterministic=True, allow_tf32=torch.backends.cudnn.allow_tf32):
                result = self._optimize(rollout, progress)
        finally:
            self._kl_inputs = None  # Also release on a rejected/nonfinite/exception path.
            self._kl_frozen = None
        self.collections += 1
        return result

    @torch.no_grad()
    def cohort_diagnostics(self, r, episode, step, targets):
        """All VALID cohort states, bounded batches, no RNG draw or optimizer step.

        This is a diagnostic, not a replacement/relaxation of the actor KL gate.
        Value metrics use the same frozen, scaled GAE targets throughout a pass.
        """
        divergence, predictions = [], []
        for begin in range(0, len(episode), self.settings['batch_size']):
            b = episode[begin:begin+self.settings['batch_size']]
            t = step[begin:begin+self.settings['batch_size']]
            history, index = r['frames'][b], t*r['stride']
            mean, log_std = self.rollout_distribution(r,b,t)
            divergence.append(gaussian_kl(r['mean'][b,t], r['log_std'][b,t], mean, log_std))
            predictions.append(self.value(history, index, r['privileged'][b,t]))
        kl, prediction = torch.cat(divergence), torch.cat(predictions)
        if not bool(torch.isfinite(kl).all() & torch.isfinite(prediction).all()):
            raise FloatingPointError('Nonfinite PPO cohort diagnostics')
        return dict(policy_kl_mean=float(kl.mean()),
            value_explained_variance=explained_variance(prediction, targets),
            value_mse=float((prediction-targets).square().mean()))

    @torch.no_grad()
    def prepare_kl_inputs(self,r,episode,step):
        """Bounded fixed prefixes and ONLY Stage4's immutable whip distributions.

        Overflow chunks use the original path. No change of batch size/order,
        convolution shape/strides, float64 KL reduction, threshold, RNG or trials.
        """
        if self.noise_rho is not None:
            # The old cache contains only one prefix. Both adjacent current-policy
            # marginals must be recomputed for this conditional distribution.
            return None
        batches=[];used=0
        frozen_actor=isinstance(self.actor,FrozenWhipRecoveryActor)
        self._kl_frozen=[] if frozen_actor else None
        for begin in range(0,len(episode),self.settings['batch_size']):
            b=episode[begin:begin+self.settings['batch_size']]
            t=step[begin:begin+self.settings['batch_size']]
            frame_shape=r['frames'].shape
            needed=len(b)*(frame_shape[1]*(len(self.interface['actor_columns'])+1)+len(self.interface['actor_columns']))*r['frames'].element_size()
            if 'goal_columns' in self.interface:
                needed += len(b)*(len(self.interface['goal_columns'])+len(self.interface.get('phase_columns',()))) * r['frames'].element_size()
            frozen_bytes=(len(b)*2*self.interface['action_size']*r['frames'].element_size()
                          if frozen_actor else 0)
            keep_prefix=(len(batches)==begin//self.settings['batch_size']
                         and used+needed+frozen_bytes<=self._kl_input_cache_limit)
            # A frozen distribution is tiny compared with a451-sample prefix.
            # Overflow prefixes still use the ORIGINAL gather/prepare path;
            # their frozen outputs may independently fit the same memory cap.
            if not keep_prefix and (not frozen_actor or used+frozen_bytes>self._kl_input_cache_limit):
                break
            prepared=self.actor.encoder.prepare(r['frames'][b],t*r['stride'])
            if keep_prefix:
                batches.append(prepared);used+=sum(x.numel()*x.element_size() for x in prepared)
            if frozen_actor:
                # Same chunk/order/convolution shape and optimizer cuDNN flags.
                # No row compaction; recovery parameters still recompute each
                # trial. The optimizer cannot change these frozen whip weights.
                frozen=(None if bool((prepared[2][:,:1]>.5).all()) else
                        PPOActor.distribution(self.actor,prepared,None))
                self._kl_frozen.append(frozen)
                if frozen is not None:
                    used+=sum(x.numel()*x.element_size() for x in frozen)
        return batches

    @torch.no_grad()
    def cohort_policy_kl(self, r, episode, step):
        """All valid states; analytic old-to-new KL evaluated in float64.

        Network arithmetic remains float32. Double KL arithmetic/reduction
        avoids near-equality cancellation; no sampling, value fit or bootstrap.
        This sampled-cohort MEAN does not bound every state or future policies.
        """
        divergence = []
        for begin in range(0, len(episode), self.settings['batch_size']):
            b = episode[begin:begin+self.settings['batch_size']]
            t = step[begin:begin+self.settings['batch_size']]
            chunk=begin//self.settings['batch_size']
            history=(self._kl_inputs[chunk] if self._kl_inputs is not None and chunk<len(self._kl_inputs) else r['frames'][b])
            frozen=(self._kl_frozen[chunk] if self._kl_inputs is not None and self._kl_frozen is not None
                    and chunk<len(self._kl_frozen) else None)
            mean, log_std = (self.rollout_distribution(r,b,t) if self.noise_rho is not None else
                            self.actor._distribution(history,t*r['stride'],frozen)
                            if frozen is not None else
                            self.actor.distribution(history,t*r['stride']))
            divergence.append(gaussian_kl(r['mean'][b,t].double(), r['log_std'][b,t].double(),
                mean.double(), log_std.double()))
        return float(torch.cat(divergence).mean())

    def _optimize(self, r, progress):
        c = self.settings['ppo']
        raw, logp, values = r['raw'], r['logp'], r['values']
        if (raw.shape[:2] != values.shape or logp.shape != values.shape
                or r['mean'].shape != raw.shape or r['log_std'].shape != raw.shape
                or r['privileged'].shape[:2] != (len(values), values.shape[1]+1)):
            raise ValueError('PPO rollout interfaces differ')
        if any(not bool(torch.isfinite(r[k]).all()) for k in
               ('raw', 'logp', 'values', 'mean', 'log_std', 'frames', 'privileged')):
            raise ValueError('Nonfinite PPO rollout data')
        adv, target, valid = advantages(r['rewards']*self.settings.get('reward_scale', 1.),
                                        values, r['lengths'], self.settings['discount'], c['gae_lambda'])
        episode, step = valid.nonzero(as_tuple=True)
        flat_adv = adv[episode, step]
        normalized = (flat_adv-flat_adv.mean())/flat_adv.std(unbiased=False).clamp_min(1e-8)
        flat_target = target[episode, step]
        before = self.cohort_diagnostics(r, episode, step, flat_target)
        metrics = {}
        applied = value_applied = actor_epochs_started = 0
        stopped_kl = False
        stop_kl = None
        guard = c.get('post_step_kl')
        if guard is not None:
            self._kl_inputs = self.prepare_kl_inputs(r,episode,step)
        trial_count = backtracked = rejected = 0
        minimum_factor = 1.
        maximum_accepted_kl = 0.
        trial_records = []
        # Update whole current collection before responding to cooperative stop.
        # A checkpoint never contains a half-consumed PPO rollout/epoch cursor.
        for epoch in range(c['epochs']):
            order = torch.randperm(len(episode), device=episode.device)
            for begin in range(0, len(order), self.settings['batch_size']):
                j = order[begin:begin+self.settings['batch_size']]
                b, t = episode[j], step[j]
                history, index = r['frames'][b], t*r['stride']
                if not stopped_kl:
                    actor_epochs_started = epoch+1
                    mean, std_log = self.rollout_distribution(r,b,t)
                    kl = gaussian_kl(r['mean'][b, t], r['log_std'][b, t], mean, std_log).mean()
                    if not bool(torch.isfinite(kl)):
                        raise FloatingPointError('Nonfinite PPO KL')
                    if float(kl.detach()) > c['target_kl']:
                        stopped_kl, stop_kl = True, float(kl.detach())
                    else:
                        current_logp = squashed_log_prob(raw[b, t], mean, std_log)
                        policy_loss, ratio = clipped_loss(current_logp, logp[b, t], normalized[j], c['clip_ratio'])
                        # Fresh sample estimates BOUNDED distribution entropy.
                        entropy_raw = mean+std_log.exp()*torch.randn_like(mean)
                        entropy = -squashed_log_prob(entropy_raw, mean, std_log).mean()
                        actor_loss = policy_loss-self.entropy_weight*entropy
                        if not bool(torch.isfinite(actor_loss) & torch.isfinite(ratio).all()):
                            raise FloatingPointError('Nonfinite PPO actor loss/ratio')
                        self.actor_optimizer.zero_grad(set_to_none=True)
                        actor_loss.backward()
                        actor_norm = nn.utils.clip_grad_norm_(self.actor.parameters(), c['max_grad_norm'], error_if_nonfinite=True)
                        accepted = True
                        if guard is None:
                            self.actor_optimizer.step()
                        else:
                            outcome = bounded_actor_step(self.actor.parameters(), self.actor_optimizer,
                                lambda: self.cohort_policy_kl(r, episode, step),
                                c['target_kl'], guard['maximum_trials'])
                            trial_records.append(dict(accepted_update_index=self.updates+1, **outcome))
                            trial_count += len(outcome['trials'])
                            accepted = outcome['accepted']
                            if accepted:
                                backtracked += int(outcome['factor'] < 1.)
                                minimum_factor = min(minimum_factor, outcome['factor'])
                                maximum_accepted_kl = max(maximum_accepted_kl, outcome['trials'][-1]['kl'])
                            else:
                                rejected += 1
                                stopped_kl = True
                                stop_kl = outcome['trials'][-1]['kl']
                        if accepted:
                            self.updates += 1
                            applied += 1
                        metrics.update(actor_loss=actor_loss.detach(), entropy=entropy.detach(),
                            approx_kl=kl.detach(), clip_fraction=((ratio-1).abs() > c['clip_ratio']).float().mean().detach(),
                            actor_grad_norm=actor_norm.detach())
                # Actor/value parameters and optimizers are DISJOINT. An actor
                # KL stop must not truncate fitting the independent value model.
                prediction = self.value(history, index, r['privileged'][b, t])
                value_loss = .5*(prediction-target[b, t]).square().mean()
                if not bool(torch.isfinite(value_loss)):
                    raise FloatingPointError('Nonfinite PPO value loss')
                self.value_optimizer.zero_grad(set_to_none=True)
                (c['value_loss_weight']*value_loss).backward()
                value_norm = nn.utils.clip_grad_norm_(self.value.parameters(), c['max_grad_norm'], error_if_nonfinite=True)
                self.value_optimizer.step()
                self.value_updates += 1
                value_applied += 1
                metrics.update(critic_loss=value_loss.detach(),
                    entropy_weight=value_loss.new_tensor(self.entropy_weight), value_grad_norm=value_norm.detach())
                if progress and value_applied % 20 == 0:
                    progress(value_applied, dict(scalar_metrics(metrics),
                        optimizer_updates=applied, value_optimizer_updates=value_applied,
                        kl_early_stop=stopped_kl))
        result = scalar_metrics(metrics)
        after = self.cohort_diagnostics(r, episode, step, flat_target)
        # Aggregate roll-out-level facts; last-minibatch loss/entropy labeled as such.
        result.update(optimizer_updates=applied, kl_early_stop=stopped_kl,
            value_optimizer_updates=value_applied, actor_epochs_started=actor_epochs_started,
            epochs_started=epoch+1, rollout_transitions=len(episode), entropy_weight=self.entropy_weight,
            advantage_mean=float(flat_adv.mean()), advantage_std=float(flat_adv.std(unbiased=False)),
            return_target_mean=float(flat_target.mean()), policy_kl_mean_before=before['policy_kl_mean'],
            policy_kl_mean=after['policy_kl_mean'],
            value_explained_variance_before=before['value_explained_variance'],
            value_explained_variance=after['value_explained_variance'],
            value_mse_before=before['value_mse'], value_mse=after['value_mse'])
        if stopped_kl: result['stop_kl'] = stop_kl
        if guard is not None:
            result.update(post_step_kl_trials=trial_count, post_step_kl_backtracked_updates=backtracked,
                post_step_kl_rejected_updates=rejected, post_step_kl_minimum_lr_factor=minimum_factor,
                post_step_kl_maximum_accepted=maximum_accepted_kl,
                post_step_kl_final=self.cohort_policy_kl(r, episode, step),
                post_step_kl_trial_records=trial_records)
        return result

    def state_dict(self):
        return dict(interface=self.interface.copy(), actor=self.actor.state_dict(), value=self.value.state_dict(),
            actor_optimizer=self.actor_optimizer.state_dict(), value_optimizer=self.value_optimizer.state_dict(),
            updates=self.updates, value_updates=self.value_updates,
            environment_steps=self.environment_steps, collections=self.collections,
            entropy_step_offset=self.entropy_step_offset, entropy_clock=self.entropy_clock,
            completed_episodes=self.completed_episodes, entropy_episode_offset=self.entropy_episode_offset,
            entropy_episode_budget=self.entropy_episode_budget)

    def load_state_dict(self, saved):
        if saved.get('interface') != self.interface:
            raise ValueError('PPO checkpoint interface differs; SAC weights/replay are not a PPO continuation')
        def finite(value):
            if isinstance(value, torch.Tensor):
                return bool(torch.isfinite(value).all())
            if isinstance(value, dict):
                return all(finite(v) for v in value.values())
            if isinstance(value, (list, tuple)):
                return all(finite(v) for v in value)
            return not isinstance(value, float) or math.isfinite(value)
        for name in ('actor', 'value', 'actor_optimizer', 'value_optimizer'):
            if name not in saved:
                raise ValueError('Missing PPO saved '+name)
            if not finite(saved.get(name)):
                raise ValueError('Nonfinite PPO saved '+name)
        if 'goal_columns' in self.interface:
            expected = torch.tensor(self.interface['goal_columns'])
            for network in ('actor','value'):
                columns = saved.get(network,{}).get('encoder.goal_columns')
                if not isinstance(columns,torch.Tensor) or not torch.equal(columns.cpu(),expected):
                    raise ValueError('Saved '+network+' goal columns disagree with its interface')
        for k in ('updates', 'environment_steps', 'collections'):
            if type(saved.get(k)) is not int or saved[k] < 0:
                raise ValueError('Invalid PPO saved '+k)
        value_updates = saved.get('value_updates', saved['updates'])  # Legacy paired optimizer clock.
        offset = saved.get('entropy_step_offset', 0)
        if type(offset) is not int or offset < 0:
            raise ValueError('Invalid PPO saved entropy step offset')
        if type(value_updates) is not int or value_updates < saved['updates']:
            raise ValueError('Invalid PPO saved value_updates')
        if saved.get('entropy_clock', 'transitions_v1') != self.entropy_clock:
            raise ValueError('PPO checkpoint entropy clock units differ from settings')
        completed = saved.get('completed_episodes', 0)
        episode_offset = saved.get('entropy_episode_offset', 0)
        budget = saved.get('entropy_episode_budget', self.entropy_episode_budget)
        if any(type(v) is not int or v < 0 for v in (completed, episode_offset)) or type(budget) is not int or budget <= 0:
            raise ValueError('Invalid PPO saved episode entropy clock')
        if self.entropy_clock == 'completed_episodes_v1' and not all(k in saved for k in
                ('completed_episodes', 'entropy_episode_offset', 'entropy_episode_budget')):
            raise ValueError('Episode entropy checkpoint requires its original budget and clock')
        for k in ('actor', 'value', 'actor_optimizer', 'value_optimizer'):
            getattr(self, k).load_state_dict(saved[k])
        self.updates, self.environment_steps, self.collections = (saved[k] for k in
                                                               ('updates', 'environment_steps', 'collections'))
        self.value_updates = value_updates
        self.entropy_step_offset = offset
        self.completed_episodes, self.entropy_episode_offset, self.entropy_episode_budget = completed, episode_offset, budget
