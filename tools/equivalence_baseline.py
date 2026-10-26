"""Record or check the Stage 1 refactoring equivalence baseline.

Runs the current Stage 1 environment with fixed open-loop actions, then one
seeded PPO collection and update, and compares every produced tensor exactly
with a saved reference. This checks that restructuring code did not change
numbers; it is not a physical, learning or flight result.

    python tools/equivalence_baseline.py --record   # once, before refactoring
    python tools/equivalence_baseline.py            # after each change
"""
import argparse
import math
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import torch

from training.configuration import load
from tasks.whip.sac_environment import make_environment
from training.ppo import PPO

REFERENCE = ROOT/'tests/data/stage1_equivalence_reference.pt'
LANES = 4


def open_loop_actions(decisions):
    """Deterministic normalized actions: lane-specific sinusoids, one saturating lane."""
    t = torch.arange(decisions, dtype=torch.float32)[:, None]*.02
    lanes = []
    for lane in range(LANES):
        frequency = .8+.4*lane
        x = (.9 if lane == 3 else .6)*torch.sin(2*math.pi*frequency*t)
        y = .3*torch.sin(2*math.pi*.5*frequency*t+lane)
        z = .2*torch.sin(2*math.pi*1.3*t+.5*lane)
        lanes.append(torch.cat((x, y, z), 1))
    return torch.stack(lanes, 0)


def environment_outputs(device):
    settings, snapshot, configuration, scene = load()
    settings = dict(settings, device=device, parallel_environments=LANES)
    env = make_environment(settings, snapshot, configuration, scene, capture=device == 'cuda')
    env.reset()
    actions = open_loop_actions(env.decisions).to(env.device)
    done = False
    while not done:
        _, _, done = env.step(actions[:, env.decision])
    returns, valid, score, cost = env.finish()
    out = {f'history/{k}': v for k, v in env.history.items()}
    out.update({f'component/{k}': v for k, v in score.components.items()})
    out.update(frames=env.frames, critic_states=env.critic_states, ledger=env.ledger,
               transition_rewards=env.transition_rewards, returns=returns, valid=valid,
               workspace_cost=cost, numerical_valid=env.numerical_valid, lengths=env.lengths)
    return out


def ppo_outputs():
    settings, snapshot, configuration, scene = load()
    settings = dict(settings, device='cpu', parallel_environments=LANES)
    torch.manual_seed(settings['seed'])
    env = make_environment(settings, snapshot, configuration, scene, capture=False)
    agent = PPO(settings, env.actor_columns, env.frame_size, env.device, env.action_size,
                privileged_size=env.critic_state_size, encoder_options=env.policy_encoder_options)
    env.reset()
    shape = (env.batch, env.decisions)
    rollout = dict(collection=1, stride=env.stride, frames=env.frames, privileged=env.critic_states,
                   raw=torch.zeros((*shape, 3)), mean=torch.zeros((*shape, 3)),
                   log_std=torch.zeros((*shape, 3)), logp=torch.zeros(shape), values=torch.zeros(shape))
    previous_raw = previous_index = None
    done = False
    while not done:
        action, *policy = agent.collect_action(env.frames, env.indices(), env.critic_states[:, env.decision],
                                               previous_raw=previous_raw, previous_index=previous_index)
        previous_raw, previous_index = policy[0], env.indices().clone()
        for key, value in zip(('raw', 'logp', 'mean', 'log_std', 'values'), policy):
            rollout[key][:, env.decision].copy_(value)
        _, _, done = env.step(action)
    returns, *_ = env.finish()
    rollout.update(rewards=env.transition_rewards, lengths=env.lengths)
    metrics = agent.optimize(rollout)
    out = {f'ppo/rollout/{k}': v for k, v in rollout.items() if isinstance(v, torch.Tensor)}
    out.update({f'ppo/actor/{k}': v for k, v in agent.actor.state_dict().items()})
    out.update({f'ppo/value/{k}': v for k, v in agent.value.state_dict().items()})
    out['ppo/returns'] = returns
    for k, v in metrics.items():
        if isinstance(v, (int, float)) and not isinstance(v, bool):
            out[f'ppo/metric/{k}'] = torch.tensor(v, dtype=torch.float64)
    return out


def collect(devices):
    result = {}
    for device in devices:
        result.update({f'env/{device}/{k}': v.detach().cpu() for k, v in environment_outputs(device).items()})
    result.update({k: v.detach().cpu() for k, v in ppo_outputs().items()})
    return result


def compare(reference, current):
    problems = []
    for key in sorted(set(reference) | set(current)):
        if key not in current:
            problems.append(f'missing {key}')
        elif key not in reference:
            problems.append(f'unexpected {key}')
        else:
            a, b = reference[key], current[key]
            if a.shape != b.shape or a.dtype != b.dtype:
                problems.append(f'{key}: {a.dtype}{tuple(a.shape)} vs {b.dtype}{tuple(b.shape)}')
            elif not torch.equal(a, b):
                difference = (a.double()-b.double()).abs().nan_to_num(float('inf')).max()
                problems.append(f'{key}: max abs difference {float(difference):.3e}')
    return problems


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--record', action='store_true', help='overwrite the saved reference')
    parser.add_argument('--cpu-only', action='store_true', help='skip the CUDA environment rollout')
    args = parser.parse_args()
    devices = ['cpu'] + ([] if args.cpu_only or not torch.cuda.is_available() else ['cuda'])
    current = collect(devices)
    if args.record:
        REFERENCE.parent.mkdir(parents=True, exist_ok=True)
        torch.save(current, REFERENCE)
        print(f'Recorded {len(current)} tensors ({", ".join(devices)}) to {REFERENCE.relative_to(ROOT)}')
        return
    reference = torch.load(REFERENCE, weights_only=True)
    if args.cpu_only:
        reference = {k: v for k, v in reference.items() if not k.startswith('env/cuda/')}
    problems = compare(reference, current)
    if problems:
        print('NOT EQUIVALENT:')
        for line in problems:
            print('  '+line)
        sys.exit(1)
    print(f'Equivalent: {len(current)} tensors match exactly ({", ".join(devices)})')


if __name__ == '__main__':
    main()
