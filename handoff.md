# AeroWhip handoff

Updated: 2026-09-26. This is a checkpoint of existing work and the user's latest
direction, not approval to implement a replacement architecture.

## Start here

- Work in `C:\Users\wts28\Documents\PHD\AeroWhip_deploy`. The task's default
  working directory may instead be the old `particle_filter_cable_project`.
- Read `AGENTS.md`, this handoff, and relevant sections of
  `doc/current_implementation.md` before acting. Inspect source/configuration
  when documentation and current settings differ.
- The user has paused learning, reward development, and performance experiments
  to reconsider the simulation and learning formulation. Do not resume them or
  create automatic follow-ups without a new request.
- The 2026-09-26 process inspection found no Python workers. This is a point-in-time
  observation, not permission to start anything; inspect actual process commands
  again before future experiments. Project follow-up automations were already
  paused at the preceding audit.
- The latest request only authorizes preserving current changes, writing this
  handoff, committing, and pushing to the existing private repository.

## What the user wants now

The research objective remains a drone with a free-ended cable performing a
directed whip toward a target with a requested entry direction. The simulator
should also be reusable for other tasks and represent the actual custom lab
drone (Crazyflie Bolt), not merely predict a PVA-command response.

The user explicitly rejected continuing with CEM and questioned the need for
B-splines. They want to reconsider the project from its control/dynamics
foundation. Do not present CEM continuation, a CEM-to-policy pipeline, or spline
outputs as agreed requirements. Preserve the existing implementation/results
as a historical baseline; no deletion or replacement has been approved.

The key objection is not just the learning algorithm: the user does not want a
workflow that learns a distorted PVA command so that its predicted motion follows
a different desired trajectory. They want achievable control actions, credible
dynamics, and a controller/policy that can produce the intended motion. Keep
desired reference, applied action, predicted motion, and measured motion distinct.

For a whipping task, a complete desired drone trajectory may not be necessary;
the maneuver could emerge from learned actions conditioned on the task. That is
a candidate formulation, not an implemented or approved design.

Earlier command-correction and two-stage-learning plans are historical context,
not current authorization. The two-stage mechanics/accuracy idea must be
reconsidered after the action interface is decided. Do not quietly retain the
previous command-correction loop as the central architecture.

## The unresolved decision to discuss next

The last question asked of the user was whether the learned whip should remain:

1. An open-loop action sequence generated from initial state/task, potentially
   with a stabilizing controller underneath; or
2. A feedback policy that observes estimated state and chooses actions throughout
   the maneuver.

The user has NOT answered that question. Historically they preferred an
open-loop whip, but the new discussion questions the overall formulation.
Do not assume either choice is now approved.

Also unresolved:

- The executable action interface: motor commands, motor thrusts, collective
  thrust/body torque, thrust/body rates, or another hardware-supported interface.
- What observations are available during execution, including cable state and
  latency, and how the real drone estimates them.
- Exact flown firmware/controller version, gains, motor/propeller/ESC setup,
  thrust mapping, actuator limits, and available onboard telemetry.
- Replacement learning algorithm, policy inputs/outputs, reward and curriculum.
  Neither MBPO nor another RL algorithm has been selected for the redesign.

An action policy trained on torque cannot simply be exported as a PVA CSV and
be called the same controller. Settle the real execution interface first.

## What the code actually implements

- Current Stage-1 is fixed-context diagonal CEM trajectory search, not a reusable
  state/target-conditioned control policy. It searches 81 bounded action values
  defining 26 interior XYZ spline controls and one endpoint. There are 32 total
  quintic control points; initial P/V/A and terminal zero V/A impose the other
  controls. Yaw and duration are not learned by this search.
- Older direct episodic policy-learning code also exists. Do not claim that no
  learning code or learned model exists; distinguish it from the currently used
  CEM experiment and the fitted residual networks.
- `simulation/dynamics/coupled.py` is already a force/torque-driven rigid-body
  and cable engine. `substep(state, thrust_n, torque_body_nm, h)` accepts collective
  thrust and body torque. Coupled reaction forces/torques, rigid-body inertia,
  gyroscopic terms and the offset attachment are present.
- The cable is a reduced, naturally straight isotropic DDER/DER centerline model:
  bending elasticity, fixed segment lengths, external drag, no independent twist
  or segment-spin state. The user rejected adding spin. Do not revive the removed
  internal bending-damping or spin experiments without discussion.
- Current mesh: 20 nodes / 19 edges, total length 0.9525 m. Drone mass is 0.145 kg;
  cable plus configured moving markers is 0.017 kg. Tracking origin is top-plate
  center, COM offset `[0, 0, -0.018]` m, attachment `[0, 0, -0.051]` m relative to
  tracking origin. The attachment is a free pivot without direct moment, but its
  force acts at an offset from COM and therefore affects rotation.
- Current numerical core: float64, projected Lie RK4, maximum step 1/4800 s,
  four position projections plus a velocity projection. Feedback/output records
  are 100 Hz; generated PVA packets normally 30 Hz, with 0.03 s command delay.
- The effective controller reads ideal evolving state at every internal RK
  stage. This is not a sampled 100-Hz onboard controller or sensor estimator.
- `simulation/dynamics/coupled_control.py` contains an effective fitted PVA
  feedback/feedforward controller, not Crazyflie firmware. It has no integral
  state, motor allocation/dynamics, or upper motor/thrust/torque saturation.
  Thrust is lower-clipped to zero. Yaw-rate feedforward is unsupported.
- The drone residual modifies requested acceleration INSIDE that controller.
  The cable residual adds effective nodal forces inside the physical dynamics.
  The drone residual can mix unmodeled controller and physical effects; do not
  move it to a new plant model or reuse it alongside new actuator dynamics
  without checking meaning and possible double counting.
- Fitting currently scales position/velocity gains, attitude frequency, cable
  bending stiffness, and drag. It does not identify motor saturation, motor lag,
  inertia, or all aerodynamic effects from the existing data.

## Selected model and calibration caveat

The selected model is `models/residual_20260921-000505.json` referencing
`models/residual_20260921-000505.pt`.

- JSON SHA-256: `6239cb9f4b829e80f872aa16fe530762d68e80728f425a7bccd9eb4710d2499c`
- PT SHA-256: `3318953a60bba141b9b321024641122f2d82d5a77ffbb92da036089797efaf91`

The selected fit predates bending/integration corrections and mesh changes. It
is retained initialization, NOT a newly validated fit of today's physics. Do not
claim the old calibration remains established, or infer physical actuator limits
from task command limits. A more accurate simulator need not track perfectly if
the real controller does not; infeasible reference trajectories remain infeasible.

## Latest read-only tracking audit

The preceding audit inspected source and recomputed metrics from retained
current-model simulations; it did NOT run new controller-isolation experiments,
change parameters, retrain, or identify the cause of tracking error.

Using `runs/residual_features/20260924/diagnostics_4_reference_0.npz` and
`milestone_1_reference_0.npz`, current physics/model hashes matched the retained
protocol. Over 350 left-endpoint samples (3.5 s), comparing simulated tracking
position with the delayed/held commanded position at matching times:

| Case | Position RMS error | Maximum error |
|---|---:|---:|
| Stationary hover | Numerically negligible | Numerically negligible |
| Gentle horizontal motion, 0.5 m/s peak command speed | 0.061999 m | 0.111128 m |
| Gentle vertical motion, 0.3 m/s peak command speed | 0.011859 m | 0.020315 m |
| Milestone maneuver | 0.112513 m | 0.188035 m |

These are command-tracking errors, not prediction error against flight data and
not evidence that a particular controller fix is correct. Prior numerical
agreement/performance tests do not establish physical accuracy. Separate saved
half-second recorded comparisons had tracking prediction RMS errors of about
0.0321, 0.0786, and 0.2522 m; they are a limited sample, not aggregate validation.

The current preparation pipeline uses source-verified OptiTrack pose/cable
markers and aligned PVA command logs. It does not consume measured motor thrust,
motor speed, or torque. Historical scripts in the old project's `deployment/`
mention Mellinger and full-state commands, but their existence does not prove the
exact firmware/configuration flown in every take. Do not run hardware scripts.

## Recent performance work: preserve the distinctions

- Accepted production changes include corrected integration, independent command
  timing, 20-node configuration, structured constraint algebra and compiled
  rollout paths. See `doc/current_implementation.md` for detailed evidence.
- `runs/residual_features/20260924/`: grouped residual-feature transforms passed
  the executed bounded comparisons; steady 256-lane interval screen used about
  10.5-10.7% less time. Full 3.5-s medians were 50.751 -> 44.338 s (batch 256) and
  26.547 -> 25.460 s (single), with material timing variation. This candidate is
  NOT integrated into production; captured fitting backward, API compatibility,
  and reward/event/ranking qualification remain separate.
- `runs/stage_reuse/20260924/`: less than 1% measured benefit, not adopted.
- `runs/s2_comparison/20260924/`: dense reduced solve slower at batch 256 and
  failed some full-motion agreement gates, not adopted.
- Earlier custom fused constraint kernels were rejected. Do not reactivate them
  or silently relax old criteria. Numerical disagreement alone does not prove
  worse physical accuracy.
- The new priority is simulation/control formulation, not another optimization
  campaign. No C++ rewrite, CrazySim migration, or performance adoption approved.

## Repository navigation and local-only evidence

- `simulation/`: physics, runtime, sampled command timing, rollouts.
- `controllers/`: separate full-state CSV and gamepad interfaces.
- `viewer/`: presentation, not physics or task reward.
- `fitting/`, `take_processing/`: calibration and recordings.
- `tasks/whip/`, `training/`, `trajectories/`: existing task/search/policy code.
- `doc/current_implementation.md`: maintained implementation history, including
  obsolete experimental sections explicitly marked historical.

The directed preset CURRENTLY has duration 3.0 s, post-hold 0.5 s, cylinder radius
0.55 m, height 1.50-3.00 m, wall distance 1.1 m centered at Z=2.0 m, and 100 CEM
updates of 256 candidates. Some September 24 prose in the implementation record
still describes an earlier 4 s / upper height 2.60 m edit. Preserve the actual
user-edited configuration; do not restore it to match stale prose. These are
historical search settings, not the redesign requirements.

`runs/`, `tmp/`, `.venv/`, IDE files and `data/take_processing/` are Git-ignored.
This source-code push does not transfer training runs, audit arrays, or the raw
flight library. Some older evidence is in sibling
`AeroWhip_archive/20260922-cleanup/`. A fresh clone lacks that local evidence;
request it if needed instead of inventing results or silently rerunning jobs.
Tracked model checkpoints remain in `models/`.

Local run `runs/whip_stage1/20260924-153459-218233` has UI name
**Training milestone 2**. Renaming only changed local `display.json`, not the
run identity. Preserve saved protocols/checkpoints and their compatibility checks.

## Recommended continuation, pending discussion

First resolve the open-loop versus feedback question and actual hardware action
interface with the user. Then agree on a small controller/actuator design and
diagnostic scope before coding. Retain the existing coupled physics baseline.
Controller/residual-isolation tests are a possible next diagnostic, not a result
already obtained or an automatic instruction to launch a campaign.

User working preferences: English; Astra High only; no agents unless newly
authorized; concise, evidence-based discussion; modular reusable code; PyCharm
entry points and clear UI rather than normal workflows requiring command flags.
Ask before changes to core model, interfaces, rewards, learning, or expensive
experiments. No silent background training, random documentation/tests, cleanup,
hardware flights, or model promotion. Keep `doc/current_implementation.md` current
when approved implementation changes actually occur.

## Checkpoint verification scope

This handoff commit is a preservation checkpoint, not a new physical validation
or a claim that the redesign is implemented. Prior test outcomes above refer to
retained historical evidence. The commit preparation performs source/configuration
syntax, whitespace, selected-checkpoint identity and staged-content checks only;
it does not rerun training, GPU benchmarks, native UI sessions or flight tests.
