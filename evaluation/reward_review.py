"""Offline candidate reward review on saved 100-Hz trajectories; never run PPO.

python -m evaluation.reward_review --manifest <inputs.json> --output <new-folder>

The input manifest pins a common configuration snapshot and each replay by SHA256.
Existing mechanics are measured, not declared a physical whip classifier. Optional
human whip labels identify individual crossing intervals in those pinned replays.
Unreviewed qualifying crossings produce a return interval, never invented success.
"""
from dataclasses import replace
import argparse
import hashlib
import json
import math
from pathlib import Path
import tomllib

import numpy as np
import torch

from tasks.whip.stage1 import Stage1Scorer
from tasks.whip.stage1_configuration import ReleaseMatchingConfiguration
from tasks.whip.strike import surface_events
from tasks.whip.workspace import WorkspacePenalty
from training.configuration import task_configuration


SCHEMA = 'offline_linked_whip_reward_review_v1'
LEVELS = ((1., .6), (.6, .4), (.3, .2), (.1, .1))
PROXY_THRESHOLDS = (.01, .05, .1, .2, .3)


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def candidate_return(quality, cost, *, failed, success):
    """Approved offline envelope; success=None means classification unresolved."""
    if not (math.isfinite(quality) and 0 <= quality <= 1
            and math.isfinite(cost) and 0 <= cost <= 1):
        raise ValueError('Candidate quality and cost must be finite and bounded')
    if type(failed) is not bool or success is not None and type(success) is not bool:
        raise ValueError('Failure and reviewed success must be boolean or unresolved')
    if failed:
        return [-1., -1.]
    base = quality-.25*cost
    return ([base, base+5.] if success is None else
            [base+5*success, base+5*success])


def candidate_configuration(snapshot):
    """Use one common frozen task/model for comparison, independent of old rewards."""
    cfg = task_configuration(snapshot['task'], snapshot['settings'], snapshot['model'], {})
    if cfg.recovery is not None or cfg.termination != 'full_horizon_v1':
        raise ValueError('Select a full-horizon Stage1–3 reference, not recovery/first-hit')
    if cfg.request[1] != 0:
        raise ValueError('This review uses the existing vertical target plane')
    old = cfg.matching
    matching = ReleaseMatchingConfiguration(
        height_width_m=old.height_width_m, speed_width_mps=old.speed_width_mps,
        direction_width_deg=old.direction_width_deg, time_preference=0.,
        speed_mode='minimum', mechanics_credit_weight=1., shaping_weight=1.,
        reward_mode='directed_whip_v1', alignment_mode=old.alignment_mode,
        alignment_cone_deg=old.alignment_cone_deg)
    return replace(cfg, matching=matching,
                   mechanics=replace(cfg.mechanics, event_end_s=cfg.episode.duration_s
                                     +cfg.episode.post_hold_s-cfg.mechanics.tip_lag_s),
                   strike_surface=replace(cfg.strike_surface, width_m=.1, height_m=.1,
                                          strike_bonus_weight=0.))


def validate_arrays(arrays, nodes, horizon):
    n = len(arrays['time_s'])
    expected = dict(time_s=(n,), drone_positions=(n, 3),
                    drone_rotations=(n, 3, 3), cable_positions=(n, nodes, 3),
                    cable_velocities=(n, nodes, 3), state_valid=(n,))
    if not 2 <= n <= round(horizon/.01)+1:
        raise ValueError('Replay must be an executed prefix of the fixed horizon')
    for name, shape in expected.items():
        if name not in arrays or arrays[name].shape != shape:
            raise ValueError('Missing/wrong-shaped replay field: '+name)
        if not np.isfinite(arrays[name]).all():
            raise ValueError('Replay contains nonfinite measurements: '+name)
    if arrays['state_valid'].dtype != np.bool_ or not arrays['state_valid'].all():
        raise ValueError('Supply only the observed numerical prefix, not padded states')
    if not np.allclose(arrays['time_s'], np.arange(n)*.01, rtol=0, atol=1e-10):
        raise ValueError('Replay must start at zero and use the 100-Hz observation grid')
    rotations = arrays['drone_rotations']
    if (not np.allclose(rotations.transpose(0, 2, 1)@rotations, np.eye(3), atol=1e-7, rtol=0)
            or not np.allclose(np.linalg.det(rotations), 1., atol=1e-7, rtol=0)):
        raise ValueError('Replay rotations must be body-to-world proper rotations')


def _components(score):
    return {k: (float(v[0]) if bool(torch.isfinite(v[0])) else None)
            for k, v in score.components.items()}


def _at_crossing(scorer, all_windows, index, p, v, observed, terminal):
    # Restrict the existing scorer to ONE actual crossing and its preceding
    # windows. Never use an earlier episode maximum to qualify this crossing.
    selected = []
    for h, begins, ends, tips in all_windows:
        keep = (tips == index).any(-1)
        if bool(keep.any()):
            selected.append((h, begins[keep], ends[keep],
                             torch.full((int(keep.sum()), 1), index, dtype=torch.long)))
    if not selected:
        return 0., {}, None
    scorer.windows = selected
    score = scorer(p, v, observed, terminal_samples=terminal)
    components = _components(score)
    return components.get('mechanics_credit') or 0., components, score.event_indices[0].tolist()


@torch.no_grad()
def evaluate(arrays, snapshot, *, recorded_failure=None, whip_labels=()):
    """Rescore measurements only. Synthetic input must be labeled by the caller."""
    cfg = candidate_configuration(snapshot)
    horizon = cfg.episode.duration_s+cfg.episode.post_hold_s
    rest = torch.tensor(snapshot['model']['cable']['marker_interval_lengths_m'], dtype=torch.float64)
    validate_arrays(arrays, len(rest)+1, horizon)
    n, total = len(arrays['time_s']), round(horizon/.01)+1
    tensor = lambda x: torch.as_tensor(x, dtype=torch.float64)
    pos, rot, cable = (tensor(arrays[k]) for k in
                       ('drone_positions', 'drone_rotations', 'cable_positions'))
    initial = tomllib.loads(snapshot['scene_text'])['initial_state']['position_world_m']
    if not torch.allclose(pos[0], tensor(initial), rtol=0, atol=1e-10):
        raise ValueError('Replay start differs from the common frozen scene')
    if bool((cable.diff(dim=1).norm(dim=-1) <= 0).any()):
        raise ValueError('Replay contains zero-length cable edges')
    safe = ((rot[:, 2, 2] >= math.cos(math.radians(cfg.returns.maximum_tilt_deg)))
            & (cable[:, :, 2].amin(-1) >= cfg.returns.minimum_cable_height_m))
    failed = not bool(safe.all()) or recorded_failure is not None
    status = 'failed' if failed else 'completed' if n == total else 'incomplete'
    if recorded_failure is not None and (not isinstance(recorded_failure, str) or not recorded_failure.strip()):
        raise ValueError('Recorded failure requires a nonempty source explanation')
    inside = cfg.workspace.contains(pos[None], pos[:1])[0]
    contained = bool(inside.all())
    penalty = WorkspacePenalty(1., .1)  # Explicit offline candidate distance scale.
    radial, vertical = penalty.distances(pos[None], pos[:1], cfg.workspace)
    density = ((radial[0]/.1).square()+(vertical[0]/.1).square()).clamp_max(1)
    cost = float(density[1:].sum()/(total-1))
    # Padding exists only to satisfy the inherited scorer's fixed mesh. It is
    # masked, never simulated and never counted as an observed continuation.
    def pad(a):
        return torch.cat((a, a[-1:].expand(total-n, *a.shape[1:])), 0)[None]
    p, v = pad(cable), pad(tensor(arrays['cable_velocities']))
    time = torch.arange(total, dtype=torch.float64)*.01
    observed = torch.arange(total)[None] < n
    terminal = torch.tensor([n])
    scorer = Stage1Scorer(rest, time, cfg, initial_position=pos[0])
    best = scorer(p, v, observed, terminal_samples=terminal)
    if not bool(best.valid[0]):
        raise ValueError('Mechanics computation is nonfinite; do not rank this replay')
    best_whip = float(best.score[0])
    precision = replace(cfg, matching=replace(cfg.matching, reward_mode='directed_precision_v1',
                                              center_sigma_m=.05))
    crossing_scorer = Stage1Scorer(rest, time, precision, initial_position=pos[0])
    windows = crossing_scorer.windows
    # Only the observed prefix enters crossing detection (no padded crossings).
    surface = surface_events(p[:, :n], v[:, :n], time[:n], scorer.center, scorer.direction,
                             cfg.strike_surface, cfg.request[3])
    indices = torch.where(surface['crossing'][0] & (surface['forward'][0] > 0))[0].tolist()
    labels = {}
    for label in whip_labels:
        index = label['interval_index']
        if (type(index) is not int or index not in indices or index in labels
                or type(label.get('is_whip')) is not bool
                or not isinstance(label.get('reviewer'), str) or not label['reviewer'].strip()
                or not isinstance(label.get('reason'), str) or not label['reason'].strip()):
            raise ValueError('Whip review needs a unique actual crossing, boolean, reviewer and reason')
        labels[index] = label
    crossings = []
    for index in indices:
        w, components, event_indices = _at_crossing(crossing_scorer, windows, index,
                                                   p, v, observed, terminal)
        value = lambda k: float(surface[k][0, index])
        crossings.append(dict(interval_index=index, time_s=value('elapsed'),
            available_at_s=float(time[index+1]), lateral_error_m=value('lateral'),
            vertical_error_m=value('vertical'), forward_mps=value('forward'),
            entry_angle_deg=math.degrees(value('angle')), whip_proxy=w,
            preceding_window_indices=event_indices, mechanics=components,
            whip_review=labels.get(index)))
    # Same candidate throughout the existing spatial ladder. These are offline
    # eligibility tables, not a modified curriculum or advancement decision.
    levels = []
    for width, height in LEVELS:
        sigma = max(.05, max(width, height)/4)
        level_surface = surface_events(p[:, :n], v[:, :n], time[:n], scorer.center, scorer.direction,
            replace(cfg.strike_surface, width_m=width, height_m=height), cfg.request[3])
        events = []
        for c in crossings:
            index = c['interval_index']
            # Reuse existing inclusive bounds and floating-point roundoff rules.
            geometric = bool(level_surface['geometric_hit'][0, index])
            strict = bool(level_surface['qualified'][0, index])
            center = math.exp(-.5*(c['lateral_error_m']**2+c['vertical_error_m']**2)/sigma**2)
            events.append(dict(interval_index=c['interval_index'], geometric=geometric,
                strict=strict, center_quality=center, whip_proxy=c['whip_proxy'],
                quality=.5*c['whip_proxy']*(1+center), whip_review=c['whip_review']))
        eligible = [e for e in events if e['strict'] and contained and status == 'completed']
        success = (True if any(e['whip_review'] and e['whip_review']['is_whip'] for e in eligible)
                   else None if any(e['whip_review'] is None for e in eligible) else False)
        quality = max([.5*best_whip]+[e['quality'] for e in events])
        bounds = None if status == 'incomplete' else candidate_return(quality, cost,
                                                         failed=failed, success=success)
        levels.append(dict(width_m=width, height_m=height, center_sigma_m=sigma,
            geometric_crossings=sum(e['geometric'] for e in events),
            strict_crossings=sum(e['strict'] for e in events),
            reviewed_success=success, candidate_quality=quality, candidate_return_interval=bounds,
            events=events,
            proxy_sensitivity_only=[dict(threshold=t,
                strict_crossings_with_proxy=sum(e['strict'] and e['whip_proxy'] >= t for e in events),
                eligible_if_proxy_were_classifier=any(e['whip_proxy'] >= t for e in eligible))
                for t in PROXY_THRESHOLDS]))
    result = dict(status=status, recorded_failure=recorded_failure, contained=contained,
        executed_duration_s=float(time[n-1]), horizon_s=horizon,
        minimum_cable_height_m=float(cable[:, :, 2].min()),
        maximum_tilt_deg=float(torch.acos(rot[:, 2, 2].clamp(-1, 1)).max()*180/math.pi),
        maximum_radial_excess_m=float(radial.max()), maximum_vertical_excess_m=float(vertical.max()),
        bounded_workspace_cost=cost, best_whip_proxy=best_whip,
        best_whip_indices=best.event_indices[0].tolist(), best_whip_components=_components(best),
        crossings=crossings, levels=levels, final=levels[-1])
    trace = dict(time_s=arrays['time_s'], drone_positions=arrays['drone_positions'],
        tip_positions=arrays['cable_positions'][:, -1],
        tip_velocities=arrays['cable_velocities'][:, -1], workspace_density=density.numpy(),
        causal_best_whip_proxy=best.causal_best[0, :n].numpy())
    return result, trace


def run_manifest(manifest_path, output):
    manifest_path, output = Path(manifest_path).resolve(), Path(output).resolve()
    raw = json.loads(manifest_path.read_text())
    if raw.get('schema') != 'offline_whip_review_inputs_v1' or not raw.get('cases'):
        raise ValueError('Expected an offline_whip_review_inputs_v1 manifest with cases')
    def pinned(item):
        path = (manifest_path.parent/item['path']).resolve()
        if sha256(path) != item['sha256']:
            raise ValueError('Input hash mismatch: '+str(path))
        return path
    reference = pinned(raw['reference_snapshot'])
    snapshot = json.loads(reference.read_text())
    output.mkdir(parents=True, exist_ok=False)
    report = dict(schema=SCHEMA, manifest=str(manifest_path), manifest_sha256=sha256(manifest_path),
        reference_snapshot=str(reference), reference_sha256=sha256(reference),
        scope='Offline scoring only; no physics replay, policy inference, optimization or model promotion',
        definition=dict(return_formula='failed: -1; completed: 5*S + Q - 0.25*C',
            Q='max(0.5*best sampled W, max_crossing 0.5*W_crossing*(1+center_quality))',
            W='Inherited directed_whip_v1 M*U*D proxy, without preparation-path discount',
            event_search='Preceding windows use the full horizon, with the inherited 0.15-s tip lag',
            center_quality='exp(-0.5*(lateral^2+vertical^2)/sigma^2) at the SAME crossing',
            C='sum min(1,(radial_excess/0.1m)^2+(vertical_excess/0.1m)^2)*0.01/full_horizon',
            S='completed, contained, strict crossing AND human-reviewed whip at that crossing',
            classification='Unreviewed whip qualification is unknown; positive proxy is not a classifier',
            uncertainty='Return intervals reflect missing classification, not statistical confidence',
            feedback='No per-step redistribution or GAE change is implemented or validated'),
        source_hashes={str(p): sha256(p) for p in (Path(__file__).resolve(),
            Path(__file__).resolve().parents[1]/'tasks/whip/stage1.py',
            Path(__file__).resolve().parents[1]/'tasks/whip/strike.py',
            Path(__file__).resolve().parents[1]/'training/configuration.py')}, cases=[])
    names = set()
    for i, case in enumerate(raw['cases']):
        if not isinstance(case.get('name'), str) or case['name'] in names:
            raise ValueError('Each case needs a unique name')
        names.add(case['name'])
        if case['evidence_kind'] not in ('saved_simulation', 'synthetic_kinematic_control'):
            raise ValueError('Label saved simulation separately from synthetic scoring controls')
        path = pinned(case)
        source = pinned(case['source_snapshot'])
        source_snapshot = json.loads(source.read_text())
        if any(source_snapshot[k] != snapshot[k] for k in ('model', 'scene_text')):
            raise ValueError('A different plant/start requires a separate review: '+case['name'])
        with np.load(path, allow_pickle=False) as saved:
            arrays = {k: saved[k] for k in saved.files}
        result, trace = evaluate(arrays, snapshot, recorded_failure=case.get('recorded_failure'),
                                 whip_labels=case.get('whip_labels', ()))
        result.update(name=case['name'], evidence_kind=case['evidence_kind'],
            source_replay=str(path), replay_sha256=sha256(path), source_snapshot=str(source),
            source_snapshot_sha256=sha256(source), notes=case.get('notes', ''),
            source_action_interface=source_snapshot['settings'].get('action_interface', 'velocity_v1'))
        trace_name = f'trace-{i:03d}.npz'
        np.savez_compressed(output/trace_name, **trace)
        result['trace'] = trace_name
        report['cases'].append(result)
        print(case['name'], result['status'], result['final']['candidate_return_interval'], flush=True)
    # Do not silently establish a total ordering when return intervals overlap.
    report['definite_ordering'] = [dict(higher=a['name'], lower=b['name'])
        for a in report['cases'] for b in report['cases']
        if a['evidence_kind'] == b['evidence_kind']
        and a['final']['candidate_return_interval'] is not None
        and b['final']['candidate_return_interval'] is not None
        and a['final']['candidate_return_interval'][0] > b['final']['candidate_return_interval'][1]]
    (output/'report.json').write_text(json.dumps(report, indent=2, allow_nan=False)+'\n')
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--manifest', required=True, type=Path)
    parser.add_argument('--output', required=True, type=Path)
    args = parser.parse_args()
    torch.set_num_threads(2)
    run_manifest(args.manifest, args.output)


if __name__ == '__main__':
    main()
