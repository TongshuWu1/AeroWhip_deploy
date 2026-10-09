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
also provides a manual saved-policy simulation tester. Replay and policy-source
browsers discover the four packaged milestone stages without archived run folders.
The tester displays source-compatibility failures and disables fresh simulation
for incompatible checkpoints; source loading is not a successful simulation.
A changed target is a diagnostic, not demonstrated adaptation.

Shared physical settings are in `configs/simulation.json`; initial conditions
are in `configs/scenes/default.toml`. Physical refitting starts with
`python fitting/run_refit.py --base-model configs/simulation.json`.
It saves candidates and comparisons without installing a new default model.
To use recorded data, copy the collection into `data/recordings/` first, preserving
its `data/flight_batches` and linked rehearsal layout. Original external recordings
are not moved or deleted. The personal take catalog and local folder preference
are generated locally and ignored by Git; a clone starts with an empty collection.

## Project-local files

Settings, models, trajectories, recordings, results, and UI open/save selections
must resolve inside this checkout. Relative paths use the project root, not the
terminal's working directory. Outside paths, home shortcuts, and symlink/junction
escapes are rejected. Copy inputs into the project before selecting them. Newly
written source/checkpoint references are project-relative; historical evidence is
not rewritten. These are application checks, not an operating-system sandbox.

## Milestone evidence and recovery

The selected completed chain used 80/80/120/60 collections. It predates the
current relative-goal observation addition, so its saved interfaces remain as
recorded. Its final actors load for CPU synthetic inference and saved replay.
The latest containment/UI changes affect files covered by the physical-source
guard, although no physics equations or numerical settings changed. That strict
guard intentionally blocks fresh plant evaluation/actor transfer of these old
checkpoints; rebuilding on another OS also changes registered binary hashes.
Linux checkpoint reevaluation needs a separately validated compatibility path,
not a silent hash bypass. Exact historical resume is not supported by this cleanup.
The later October 9 campaign stopped in Stage1 at 68 collections
and did not produce a replacement Stage1-4 chain.

[The model manifest](models/milestone-20261009/manifest.json) records hashes,
initialization links, settings, summaries, and essential saved evaluations.
Nominal precision/recovery results do not establish workspace containment,
varied-target robustness, or flight readiness. The small current-pipeline
regression suite is self-contained and has no archived run-fixture dependency:
74 checks in eight modules ran on Windows: 73 passed, one symlink-creation check
was skipped for missing Windows privileges. A separate junction-escape check
passed. Presets, packaged evidence, CPU actor inference, and idle UI selection/
save rejection also passed in a relocated source copy from another working
directory. This was not a fresh native build or Linux/interactive-3D validation.

Development runs, old tests, and original documentation are recoverable in the
external archive `AeroWhip-development-archive-20261009`; its location and
preservation evidence are in [milestone verification](doc/milestone.json).
Git history is preserved.
Training and its scheduled continuation remain stopped.
