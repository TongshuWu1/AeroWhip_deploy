# AeroWhip

Drone-cable simulation, recorded-data fitting, and four-stage fixed-target PPO
training. C++ implements the motor-driven physical plant; Python implements
controllers, tasks, learning, data processing, and the desktop UI.

This is the 2026-10-09 simulation milestone. The retained completed Stage1-4
checkpoint chain is in [models/milestone-20261009](models/milestone-20261009),
with its original settings and observations. Linux operation and Crazyflie
flight integration remain to be verified.

## Setup

Create a new virtual environment on each computer. The checked Windows setup
uses Python 3.12, Torch 2.11.0, and pybind11 3.0.1. Install a Torch build matching
the intended CUDA environment before building the tensor extension.

```powershell
py -3.12 -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install -e ".[build]"
python build_engine.py
python validate_stages.py
```

Windows builds require MSVC and a Windows SDK. On Linux, create a fresh
environment with `python3 -m venv .venv`, activate it with
`source .venv/bin/activate`, and run the same install/build/validation commands
with a C++17 compiler available. Rebuild both native extensions for the new OS
and Python ABI. CUDA training also requires a compatible Torch/CUDA/Triton
installation. The supplied Linux build path has not yet been exercised on the
user's new system. See [current implementation](doc/current_implementation.md)
for backend and physical-model limits.

## Current pipeline

| Stage | Learning objective |
|---|---|
| 1 | Directed whip mechanics and cable unfolding |
| 2 | Delivery toward one fixed 1.0 x 0.6 m target |
| 3 | Full-actor precision learning at the same target; size curriculum ends at 0.1 x 0.1 m |
| 4 | Frozen Stage3 whip, internal return controller, learned recovery residual and cable settling |

Current presets are `configs/learning/whip_stage{1..4}_ppo_cone25_v4.json`.
Each has 128 training lanes and a 10,240-episode cap: 80 full collections.
Plateau stopping is a configured heuristic; a cap does not establish convergence
or success. Stage2-4 require the preceding stage's final actor and start fresh
critics, optimizers, RNG, and local clocks.

All current stages use
`goal = (target_world - drone_initial_world) / cable_length` in world axes.
Training uses one fixed nominal target and fixed initial conditions. The
retired multi-target sampler and partial Stage3 training mask are not active.
A correction network learned between attempts is not implemented.

## Entry points

```text
python run_simulation.py
python run_training.py --stage 1
python run_training.py --stage 1 --start
python run_training.py --stage 2 --initialize-actor runs/ppo_stage1/NEW_STAGE1_RUN/latest.pt --start
python run_replay.py
python take_processing/run_take_processing.py
python validate_stages.py --output validation.json
```

Replace `NEW_STAGE1_RUN` with a completed compatible run. Use the same explicit
handoff for Stages3 and4. Opening Training without `--start` stays idle.
Saved replay renders recorded states without running physics. The Training UI
also provides a manual saved-policy simulation tester. A changed target is a
diagnostic, not demonstrated adaptation.

Shared physical settings are in `configs/simulation.json`; initial conditions
are in `configs/scenes/default.toml`. Physical refitting starts with
`python fitting/run_refit.py --base-model configs/simulation.json`.
It saves candidates and comparisons without installing a new default model.
Source recordings are external and selected through Take Processing/fitting.

## Milestone evidence and recovery

The selected completed chain used 80/80/120/60 collections. It predates the
current relative-goal observation addition, so its saved interfaces remain as
recorded. Its final actors load for inference on the current physical model;
source differences prevent a claim of exact historical resume or new current-
source training. The later October 9 campaign stopped in Stage1 at 68 collections
and did not produce a replacement Stage1-4 chain.

[The model manifest](models/milestone-20261009/manifest.json) records hashes,
initialization links, settings, summaries, and essential saved evaluations.
Nominal precision/recovery results do not establish workspace containment,
varied-target robustness, or flight readiness. The small current-pipeline
regression suite is self-contained and has no archived run-fixture dependency:
65 checks in seven modules passed on the current Windows checkout.

Development runs, old tests, and original documentation are recoverable in the
external archive `AeroWhip-development-archive-20261009`; its location and
preservation evidence are in [milestone verification](doc/milestone.json).
Git history is preserved.
Training and its scheduled continuation remain stopped.
