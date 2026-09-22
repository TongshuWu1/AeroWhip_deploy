"""Run a full-state command CSV without the viewer and save one numerical run."""

import argparse
from dataclasses import asdict
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import platform
import sys

from time import perf_counter

import numpy as np
import torch


ROOT = Path(__file__).resolve().parents[2]
if __package__ in (None, ""):
    sys.path.insert(0, str(ROOT))

from trajectories.full_state import FullStateTrajectory
from simulation.configuration import DEFAULT_MODEL, load_configuration
from simulation.rollout import rollout
from simulation.runtime import Simulation


def _digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--trajectory', type=Path, required=True, help='Existing 100 Hz full-state CSV')
    parser.add_argument('--config', type=Path, default=ROOT/'configs/scenes/default.toml')
    parser.add_argument('--model', type=Path, default=DEFAULT_MODEL)
    parser.add_argument('--steps', type=int,
                        help='Positive number of 0.01 s intervals; default: CSV duration. '
                             'Extra intervals hold the final command; short runs truncate the command sequence.')
    parser.add_argument('--output', type=Path, help='New run directory; existing directories are never overwritten')
    parser.add_argument('--device', choices=('cpu', 'cuda'))
    parser.add_argument('--no-compile', action='store_true')
    args = parser.parse_args()
    if args.steps is not None and args.steps < 1:
        parser.error('--steps must be positive')

    try:
        paths = {name: getattr(args, name).resolve() for name in ('trajectory', 'config', 'model')}
        sources = {name: {'path': str(path), 'sha256': _digest(path)} for name, path in paths.items()}
        config = load_configuration(paths['config'])
        model = json.loads(paths['model'].read_text(encoding='utf-8'))
        trajectory = FullStateTrajectory.load(paths['trajectory'])
        trajectory.validate_initial(config)
        steps = trajectory.sample_count-1 if args.steps is None else args.steps
        for name, path in paths.items():
            if _digest(path) != sources[name]['sha256']:
                raise RuntimeError(f'{name} changed while loading')
        output = (args.output if args.output is not None else
                  ROOT/'runs/simulation'/datetime.now().strftime('%Y%m%d-%H%M%S-%f')).resolve()
        output.mkdir(parents=True, exist_ok=False)

        print('Initializing simulation...', flush=True)
        started_at = datetime.now(timezone.utc).isoformat()
        started = perf_counter()
        simulation = Simulation(config, model_config=model, device=args.device,
                                compile_physics=not args.no_compile, command_source=trajectory.command)
        setup_s = perf_counter()-started
        print(f'Running {steps} steps on {simulation.dynamics.device}...', flush=True)
        started = perf_counter()
        arrays = rollout(simulation, steps)
        execution_s = perf_counter()-started
        for name, path in paths.items():
            if _digest(path) != sources[name]['sha256']:
                raise RuntimeError(f'{name} changed during the run; output not saved')

        metadata = {
            'schema_version': 1, 'status': 'complete', 'started_at_utc': started_at,
            'sources': sources, 'model_config': model, 'scene_config': asdict(config),
            'steps': steps, 'state_samples': steps+1, 'timestep_s': simulation.timestep_s,
            'simulation_duration_s': steps*simulation.timestep_s,
            'trajectory_duration_s': trajectory.duration_s,
            'extra_hold_steps': max(0, steps-(trajectory.sample_count-1)),
            'command_delay_steps': simulation.dynamics.delay_steps,
            'command_columns': arrays['command_columns'].tolist(),
            'command_convention': 'World P/V/A at tracking origin (m, m/s, m/s^2); '
                                  'acceleration excludes gravity; yaw/yaw_rate in rad and rad/s.',
            'sample_convention': 'State row k is at time_s[k]. Command row k acts on '
                                 '[time_s[k], time_s[k+1]); applied_commands include configured delay.',
            'acceleration_convention': 'World tracking-origin backward velocity difference; '
                                      'initial row is a zero placeholder, not measured acceleration.',
            'reset_convention': 'Configured initial physical state; command delay queue filled '
                                'with initial position/yaw hold. Not a resumed checkpoint.',
            'device': simulation.dynamics.device, 'compiled': simulation.dynamics.compiled,
            'cuda_graph': simulation.dynamics.graph is not None,
            'setup_s': setup_s, 'execution_s': execution_s,
            'timing_convention': 'Execution includes reset, stepping and state readback; '
                                 'excludes setup/compilation and output-file writing.',
            'software': {'python': platform.python_version(), 'numpy': np.__version__,
                         'torch': str(torch.__version__), 'cuda': torch.version.cuda},
        }
        # Write metadata last: a trajectory without complete metadata is not a completed run.
        with (output/'trajectory.npz').open('xb') as file:
            np.savez_compressed(file, **arrays)
        with (output/'run.json').open('x', encoding='utf-8') as file:
            json.dump(metadata, file, indent=2, allow_nan=False, default=lambda value: value.tolist())
        print(f'Completed {steps*simulation.timestep_s:.2f} simulated seconds in {execution_s:.2f} s.')
        print(f'Output: {output}', flush=True)
    except (OSError, ValueError, RuntimeError, FloatingPointError) as exc:
        parser.exit(1, f'Run failed: {exc}\n')


if __name__ == '__main__':
    main()
