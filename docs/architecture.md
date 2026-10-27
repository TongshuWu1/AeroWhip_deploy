# Architecture

AeroWhip has one Python package, `aerowhip/`, a C++ physics engine in `native/`,
and command-line entry points in `scripts/`. Packages depend only on the packages
listed above them:

| Package | Responsibility |
|---|---|
| `aerowhip.physics` | Coupled drone–cable plant: C++ engine bindings (`_native`, `_tensor_native`), model files, states, single-flight runtimes and the batched tensor model used for training and fitting. |
| `aerowhip.control` | Geometric flight controller (NumPy and Torch versions) and gamepad input. |
| `aerowhip.trajectories` | Full-state reference trajectories, B-splines and presets. |
| `aerowhip.envs` | `BatchEnvironment`: batched episodes over the native plant and its flight controller (lanes stepped in parallel on the CPU), command delay, 100 Hz history, validity limits, and the `Task` interface. |
| `aerowhip.tasks` | Reward definitions behind the `Task` interface, registered by name (`tasks/__init__.py`). |
| `aerowhip.experiment` | Experiment files: parsing, validation, run snapshots and run identity. |
| `aerowhip.learning` | PPO, run folders/checkpoints and the training worker. Independent of the task. |
| `aerowhip.evaluation` | Whip measurements on saved motion, offline event review, and frozen-policy tests. |
| `aerowhip.takes` | Recorded flight takes: loading, checks, clock alignment, catalog. |
| `aerowhip.fitting` | Cable, vehicle-response and aerodynamic-residual fitting from takes. |
| `aerowhip.deployment` | ROS 2 / Crazyswarm adapters, OptiTrack relay, cable-marker tracking, and the flight estimator (`observer`: frame clock, drone velocity, live cable filter) shared with the training environment. |
| `aerowhip.viewer` | 3D rendering of simulated, replayed and live states. |
| `aerowhip.ui` | Desktop applications: training, policy tester, take processing, controller editor. |

`aerowhip.paths.ROOT` is the repository root; `configs/`, `data/`, `runs/` and
`output/` are resolved from it, and file arguments must stay inside it.

## Experiments

An experiment is one JSON file in `configs/experiments/`, parsed by
`aerowhip.experiment.Experiment` into typed sections:

| Section | Type | Contents |
|---|---|---|
| `model`, `scene` | paths | Physical model (`configs/simulation.json`) and initial scene (`configs/scenes/*.toml`), shared with simulation, fitting and deployment. |
| `run` | `learning.configuration.RunConfig` | Device, seed, parallel episodes, episode budget, evaluation interval. |
| `environment` | `envs.configuration.EnvironmentConfig` | Horizon, policy rate, acceleration limits, workspace, tilt and cable-height validity limits, and the OptiTrack sensing model (`sensing`: latency range, position noise, marker dropout hazard by speed, reappearance probability, the drone's own markers and static room points in the cloud). |
| `task` | registered by `task.name` | Reward settings (fold transport: `tasks.whip.configuration.FoldTransportConfig`). |
| `ppo` | `learning.configuration.PPOConfig` | Learner, exploration noise and entropy schedule. |

Validation checks required keys, types and ranges, and rejects choices that are
not implemented (for example an output direction other than +X). Every number
the reward and environment use comes from this file.

## Training data flow

```text
experiment file ──load──▶ Experiment + snapshot (experiment, resolved model, scene text)
                              │
                 create_run ──▶ runs/ppo/<time>/  configuration.json, scene.toml, request.json
                              │
python -m aerowhip.learning.worker --run <folder>
      │
      ├─ BatchEnvironment (lanes) ── steps native plant + firmware flight law (OpenMP), 500 Hz controller
      │        │ every 100 Hz sample: OptiTrack model (dropout, noise, latency) → live cable filter → actor frame
      │        │                      Task.observe → reward;  on numerical failure: Task.fail
      │        ▼
      │   frames (actor: sensed pose, filtered markers, confidences), critic state (true state), reward ledger
      ├─ PPO.collect_action / optimize (one update per collection)
      └─ deterministic evaluation → round-NNNNN.npz, best.pt, latest.pt, events.jsonl
```

Each collection keeps every episode's outcome (`collection-NNNNN.npz`) and
replays of representative lanes. Rewards are differences of the task's running
return, so the per-decision rewards of an episode sum exactly to its return.

## Frames and units

World axes are Z up; rotations map body to world; quaternions are XYZW. Drone
position and velocity refer to the tracked top-plate origin. Cable nodes,
including attachment node 0, are world positions in metres. Actions are
normalized world accelerations (excluding gravity) scaled by
`environment.acceleration_limit_mps2`, with yaw fixed at its initial value. In the training
environment a zero action means hover: a vertical feed-forward acceleration, solved per lane from the
thrust map at that lane's battery voltage, is added so that the delivered thrust equals the weight of
drone and cable (in flight the host position controller supplies this trim).

## Equivalence check

`scripts/equivalence_check.py --record` saves the tensors of a fixed open-loop
rollout (CPU and CUDA) and one seeded PPO update; running it again compares every
tensor bitwise. It is the check that a refactoring changed no numbers, not a
physical or learning result.
