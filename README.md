# AeroWhip

Aerial cable whipping: a quadrotor with a free-hanging cable learns to whip the
cable tip, so the cable extends the drone's reach. This repository contains

- a coupled drone–cable simulator (C++ discrete elastic rod, two-way coupled with
  the drone, float64 implicit integration, batched CUDA execution for training);
- a batched training environment, task definitions and PPO;
- measurements for telling a whip from a swing on saved motion;
- tools for recorded OptiTrack flights and for fitting the model to them;
- ROS 2 / Crazyswarm adapters, a custom simulation server and a live viewer.

**Status.** The simulator and training pipeline run; the current task (strike a
target beyond the drone's reach with the cable tip, [docs](docs/strike_task.md)) has
not yet produced a trained policy. Cable parameters are working assumptions awaiting refits from new
flight data. Running a learned policy on hardware is not implemented yet.

## Layout

```text
aerowhip/      Python package (see docs/architecture.md)
native/        C++ physics engine
scripts/       entry points
configs/       experiment, model, controller, scene and ROS settings
data/          reference trajectories; recordings go in data/recordings/ (not tracked)
tests/         pytest suite
docs/          documentation
```

## Setup

Python 3.12, a C++17 compiler and, for training, a CUDA build of PyTorch 2.11.
Install the PyTorch build that matches your CUDA environment first.

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -e ".[build,dev]"
python scripts/build_engine.py
python -m pytest
```

On Windows, use `py -3.12 -m venv .venv` and `.venv\Scripts\Activate.ps1`, with
MSVC and a Windows SDK installed. The viewer needs a desktop OpenGL session.

## Running

| Command | Purpose |
|---|---|
| `python scripts/train.py [--start]` | Training window for the default experiment (`--settings` selects another file, `--resume` a checkpoint). |
| `python scripts/replay.py` | Browse and replay saved runs. |
| `python scripts/test_policy.py` | Run a trained checkpoint from an editable initial state. |
| `python scripts/simulate.py` | Native simulation with the flight viewer. |
| `python scripts/controller.py` | Full-state trajectory editor and controller in simulation. |
| `python scripts/custom_simulation.py` | Simulation server for Crazyswarm scripts ([docs](docs/simulation_and_flight.md)). |
| `python scripts/real_flight.py` | Live OptiTrack viewer with cable tracking and recording. |
| `python scripts/play_recording.py` | Replay a saved live recording. |
| `python scripts/take_processing.py` | Browse recorded takes, run calibrations and cable fits. |
| `python scripts/equivalence_check.py` | Bitwise check that a refactoring changed no numbers. |
| `python -m aerowhip.learning.worker --run <folder>` | Headless training of a prepared run folder. |
| `python -m aerowhip.evaluation.whip_event_review --manifest <json> --output <folder>` | Offline whip-event measurements on saved motion. |

## Documentation

- [Architecture](docs/architecture.md): packages, experiment files, data flow, units.
- [Simulation model](docs/simulation_model.md): physics, parameters, solver, controller.
- [Strike task](docs/strike_task.md): reward, hit and whip-event definitions, curriculum, randomisation, evaluation.
- [Flight data and fitting](docs/data_and_fitting.md): takes, checks, clock alignment, fitting stages.
- [Simulation and flight](docs/simulation_and_flight.md): Crazyswarm simulation server, real-flight viewer, cable tracking and recording.

## Data

Recordings, training runs and retained evidence are published as GitHub release
archives with per-file manifests and SHA-256 checksums; they are not part of the
Git checkout. Extract them at the repository root.
