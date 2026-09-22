"""PyCharm: score retained motions and a kinematic swing; no simulation or RL."""

from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import shutil
import sys

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[3]
if __package__ in (None, ''):
    sys.path.insert(0, str(ROOT))

from tasks.whip.experiment import load_experiment, write_json
from tasks.whip.metrics import WhipMetrics
from tasks.whip.peak_scoring import PeakWhipScorer
from tasks.whip.reach import ReachConfiguration, score_reach
from tasks.whip.workspace import CylinderWorkspace
from tasks.whip.analysis import ARCHIVE_ROOT

CONFIGURATION = ARCHIVE_ROOT/'configs/experiments/whip_reach_training.toml'
HISTORICAL = ARCHIVE_ROOT/'runs/whip_comparison/20260921-062943-373533'


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def verified_arrays(path, manifest_root):
    manifest = json.loads((manifest_root/'artifacts.json').read_text(encoding='utf-8'))
    manifest = {Path(k).as_posix(): v for k, v in manifest.items()}
    if manifest.get(path.relative_to(manifest_root).as_posix()) != digest(path):
        raise ValueError(f'Retained motion hash mismatch: {path}')
    with np.load(path, allow_pickle=False) as data:
        return {k: data[k] for k in data.files}


def main():
    torch.set_num_threads(2)
    raw, trajectory, scoring, returns = load_experiment(CONFIGURATION)
    reach = ReachConfiguration(**raw['reach'])
    workspace = CylinderWorkspace(**raw['workspace'])
    model_path = ROOT/raw['experiment']['model']
    model = json.loads(model_path.read_text(encoding='utf-8'))
    rest = np.concatenate([np.full(n, length/n) for length, n in zip(
        model['cable']['marker_interval_lengths_m'], model['simulation']['segments_per_marker_interval'])])
    time = np.arange(round((trajectory.duration_s+trajectory.post_hold_s)/.01)+1)*.01
    start = np.array([0., 0., 2.])
    target = np.array(raw['experiment']['target_position_world_m'])
    tensor = lambda x: torch.as_tensor(x, dtype=torch.float64)
    window = tensor([[model['controller']['command_delay_s'],
                      model['controller']['command_delay_s']+trajectory.duration_s]])
    measure = WhipMetrics(tensor(rest), tensor(time),
                         region_boundaries_m=tensor(scoring.region_fractions)*rest.sum())
    scorer = PeakWhipScorer(tensor(time), scoring)
    sources, cases = [], []
    for name in ('M3', 'M4', 'M5', 'M6', 'Gentle'):
        path = HISTORICAL/name/'measurements.npz'
        arrays = verified_arrays(path, HISTORICAL)
        np.testing.assert_allclose(arrays['cable_rest_lengths'], rest, rtol=0, atol=1e-15)
        cases.append((name, arrays, 'Historical simulated command; not the current spline family'))
        sources.append(dict(name=name, path=str(path), sha256=digest(path)))
    # Latest retained full-motion replay per training run, not a fresh rollout.
    replays = {}
    for folder in (*(ARCHIVE_ROOT/'runs/whip_evaluation').glob('*'),
                   *(ROOT/'runs/whip_evaluation').glob('*')):
        if not (folder/'summary.json').exists() or not (folder/'artifacts.json').exists():
            continue
        summary = json.loads((folder/'summary.json').read_text(encoding='utf-8'))
        if not summary.get('playable'):
            continue
        key = summary['source_run']
        if key not in replays or summary['update'] > replays[key][0]['update']:
            replays[key] = (summary, folder)
    for summary, folder in replays.values():
        path = folder/'rollout.npz'
        arrays = verified_arrays(path, folder)
        name = f"Policy_{Path(summary['source_run']).name}_{summary['update']}"
        cases.append((name, arrays, 'Retained deterministic simulated policy replay'))
        sources.append(dict(name=name, path=str(path), sha256=digest(path)))

    # Length-preserving straight-cable swing: an adversarial kinematic history,
    # NOT a dynamically simulated or command-feasible trajectory.
    u = np.minimum(time, 3.)
    amplitude = np.sin(np.pi*u/3.)**2
    rate = np.pi/3.*np.sin(2*np.pi*u/3.)
    s = np.r_[0., rest.cumsum()]
    for name, sign in (('Straight_swing_forward', 1.), ('Straight_swing_backward', -1.)):
        theta, omega = sign*1.4*amplitude, sign*1.4*rate
        drone = np.tile(start, (len(time), 1)); drone[:, 0] += sign*.25*amplitude
        root = drone+np.array(model['attachment']['position_tracking_m'])
        directions = np.column_stack((np.sin(theta), np.zeros_like(theta), -np.cos(theta)))
        cable = root[:, None]+s[None, :, None]*directions[:, None]
        root_velocity = np.column_stack((sign*.25*rate, np.zeros((len(time), 2))))
        derivative = omega[:, None]*np.column_stack((np.cos(theta), np.zeros_like(theta), np.sin(theta)))
        velocities = root_velocity[:, None]+s[None, :, None]*derivative[:, None]
        cases.append((name, dict(time_s=time, drone_positions=drone,
            drone_rotations=np.tile(np.eye(3), (len(time), 1, 1)), cable_positions=cable,
            cable_velocities=velocities, state_valid=np.ones(len(time), dtype=bool)),
            'Synthetic kinematic counterexample only; no dynamics or command-feasibility claim'))

    output = ROOT/'runs/reach_comparison'/datetime.now(timezone.utc).strftime('%Y%m%d-%H%M%S-%f')
    output.mkdir(parents=True, exist_ok=False)
    rows = []
    for name, arrays, kind in cases:
        count = len(time)
        np.testing.assert_allclose(arrays['time_s'][:count], time, rtol=0, atol=1e-12)
        shift = start-arrays['drone_positions'][0]
        positions = tensor(arrays['cable_positions'][:count]+shift)[None]
        velocities = tensor(arrays['cable_velocities'][:count])[None]
        drone = tensor(arrays['drone_positions'][:count]+shift)[None]
        valid = torch.as_tensor(arrays['state_valid'][:count], dtype=torch.bool)[None]
        measured = measure(positions, velocities, valid)
        bending = measured.region_rms_curvature_per_m*rest.sum()
        args = (bending, positions[:, :, -1], velocities[:, :, -1], measured.sample_valid, window)
        quality = scorer(*args)
        legacy = scorer(*args, tensor(target)[None])
        result = score_reach(tensor(time), positions[:, :, -1], velocities[:, :, -1],
            measured.sample_valid, quality.score_history, tensor(start)[None], tensor(target)[None],
            window, scoring.tip_window_s, workspace.radius_m, scoring.speed_reference_mps, reach)
        # Verify the production wrapper's algebra against a direct unweighted search.
        accuracy = torch.exp(-.5*((positions[:, :, -1]-tensor(target)).norm(dim=-1)/scoring.target_width_m)**2)
        recovered = legacy.score_history/(scoring.whip_weight+scoring.target_weight*accuracy)
        torch.testing.assert_close(recovered, quality.score_history, rtol=1e-12, atol=1e-12)
        if not bool(result.valid.all()):
            raise ValueError(f'{name}: invalid scoring history')
        inside = bool(workspace.contains(drone, tensor(start)[None]).all())
        tilt = bool((tensor(arrays['drone_rotations'][:count, 2, 2])
                     >= math.cos(math.radians(returns.maximum_tilt_deg))).all())
        above_floor = bool((positions[..., 2] >= returns.minimum_cable_height_m).all())
        row = dict(name=name, evidence=kind, translation_applied_m=shift.tolist(),
            legacy_motion_score=float(legacy.score[0]), reach_motion_score=float(result.score[0]),
            progress=float(result.progress[0]), quality=float(result.quality[0]), bonus=float(result.bonus[0]),
            tip_speed_mps=float(result.tip_speed_mps[0]),
            closest_window_distance_m=float(result.closest_window_distance_m[0]),
            selected_window_distance_m=float(result.selected_window_distance_m[0]),
            selected_extension_m=float(result.selected_extension_m[0]),
            selected_time_s=float(result.selected_time_s[0]),
            workspace_satisfied=inside, tilt_satisfied=tilt, cable_floor_satisfied=above_floor)
        rows.append(row)
        print(json.dumps(row), flush=True)
    source_hashes = {}
    for path in (Path(__file__), ROOT/'tasks/whip/reach.py', ROOT/'tasks/whip/scoring.py',
                 ROOT/'tasks/whip/peak_scoring.py', ROOT/'tasks/whip/metrics.py', CONFIGURATION):
        relative = (Path('configs/experiments')/path.name if path == CONFIGURATION
                    else path.relative_to(ROOT))
        destination = output/'source_snapshot'/relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(path, destination)
        source_hashes[str(relative)] = digest(path)
    write_json(output/'comparison.json', dict(configuration=raw, inputs=sources, source_sha256=source_hashes,
        device='cpu', dtype='float64', versions=dict(torch=torch.__version__, numpy=np.__version__),
        protocol='First 3.5 s; translation only to initial drone (0,0,2); one shared receiving region. '
                 'Motion scores exclude acceleration penalty and failure override. Sampled envelope flags are '
                 'reported separately; historical commands are not certified as admissible spline actions.',
        limitation='Reach rewards task accomplishment, not a physical whip classifier. The synthetic straight '
                   'swing deliberately tests this limitation; do not tune the reward to relabel it.',
        physics_rollouts=0, optimizer_updates=0, cases=rows))
    write_json(output/'artifacts.json', {str(p.relative_to(output)): digest(p)
        for p in output.rglob('*') if p.is_file()})
    print(f'Comparison saved: {output}', flush=True)
    return output


if __name__ == '__main__':
    main()
