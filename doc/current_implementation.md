# Current implementation

Last reviewed: 2026-10-09.

This record describes the retained simulation pipeline and selected milestone
models. Development experiments and the previous detailed narrative are in the
recoverable external archive `AeroWhip-development-archive-20261009`.
Training and scheduled continuation are stopped.

## Pipeline and entry points

[run_simulation.py](../run_simulation.py) runs the native physical plant.
[run_training.py](../run_training.py) opens Training; `--start` explicitly starts
a run, `--stage 1..4` selects the current fixed-target PPO stage, `--settings`
selects an explicit settings file, and `--initialize-actor` performs actor-only
transfer. Exact resume is a separate, source-checked path.
[run_replay.py](../run_replay.py) renders saved arrays without inference/physics.
The manual policy tester remains available from Training.
[Take Processing](../take_processing/run_take_processing.py) manages external
recordings and fitting. [validate_stages.py](../validate_stages.py) runs the small
self-contained current-pipeline regression suite; `--output` optionally writes
its result. It does not launch a training campaign.

[simulation](../simulation) and [native](../native) own reusable physics.
[controllers](../controllers), [tasks/whip](../tasks/whip), [training](../training),
and [fitting](../fitting) own their respective behavior. Settings resolve from
the checkout through [project_paths.py](../project_paths.py), not the caller's
working directory. [configs/simulation.json](../configs/simulation.json) selects
physics, geometric controller, motors, and the unfitted aerodynamic schema.
[configs/scenes/default.toml](../configs/scenes/default.toml) sets the initial
state and viewer.

## Physical model, units, and timing

World XYZ has Z up. Rotation maps body to world; angular velocity, torque, and
inertia use the body frame/about COM. Public drone position/velocity refer to
the tracked top-plate origin. Quaternions use XYZW. Cable nodes, including
attachment node 0, use world coordinates.

| Quantity | Default |
|---|---|
| Drone mass excluding cable | 0.145 kg |
| Cable and moving-marker mass | 0.017 kg |
| Cable length / mesh | 0.9525 m / 11 nodes, 10 edges |
| Tracking-to-COM / attachment offset | (0,0,-0.018) / (0,0,-0.051) m |
| Maximum physics step | 1/600 s |
| Observation / controller / policy rate | 100 / 200 / 10 Hz |
| Command delay | 0.03 s |
| EI / Cb | 1.178939873e-8 N m² / 8.174015594e-5 N m² s |
| GJ / Ct | 1.178939873e-8 N m² / 8.174015594e-6 N m² s |

These are working model assumptions, not newly identified hardware parameters.
The ten measured marker intervals form the ten edges. The inextensible cable
is naturally straight and isotropic. A point clamp follows drone position and
body-down tangent; the first finite edge may bend. The free tip has no imposed
pose/end couple. Attachment force and direct material moment react on the drone
through one two-way coupled solve.

[native/material_frame.inc](../native/material_frame.inc) and the scalar/tensor
engine implement objective material bending/twist with Kelvin-Voigt damping.
Cb is internal material damping; external cable drag is a separate coefficient
in inverse seconds. Rayleigh dissipated power is twice its damping potential.
Material-angle history remains despite assumed negligible angular inertia.
There is no cable self-collision, ground/wall contact, or measured wind.
Degenerate edges and antiparallel tangent singularities are rejected.

The solvers use float64 projected Lie-BDF2. Uniform steps use BDF2; startup and
excessive step growth restart at backward-Euler order. Endpoint geometry,
velocities, material rates, loads, and constraints iterate together. The cap
is 20 Picard slots, relative-change tolerance 2e-11, length tolerance 1e-10 m,
velocity-constraint tolerance 1e-9 m/s, and four position projections per slot.
Controller, command, and output event times remain exact. Solver convergence
does not establish physical accuracy.

The geometric controller converts tracked-origin feedback and velocity commands
to thrust/body torque with fixed initial yaw. It has no hidden integrated
position command, integral loop, or exact dynamic cable-force cancellation.
Motor inverse mixing is clipped per motor; first-order motor response supplies
the delivered wrench. Motor parameters are marked uncalibrated. Continuation
requires previous BDF2 state/step, motors, held wrench, command queue, and clocks.
Drone telemetry alone cannot exactly resume a simulated trajectory.

Captured CUDA tasks default to the GPU-controlled Picard loop on 64-bit Windows;
other platforms default to host-controlled replay.
`AEROWHIP_CONDITIONAL_SOLVER=0` selects the host path; `1` selects the conditional
path and requires the supported Windows/NVRTC/CUDA-driver environment. Backend
failures propagate rather than silently changing the solver. CPU/differentiable
fitting paths are separate. Linux operation has not yet been verified.

## Policy observations and PPO

All current stages append the static world-axis input
`goal = (target_world - drone_initial_world) / cable_length`.
The origin is the episode's initial tracking point. It does not rotate with yaw
or become current drone-to-target error. The target plane is vertical: its normal
is the horizontal requested direction and its vertical axis is world+Z.
Independent tilted target planes are not implemented.

Actor history contains scaled tracked displacement, body-to-world rotation,
linear/angular velocity, attachment-relative cable position/velocity, previous
normalized action, and elapsed time. Physical frames record motor thrust and
held wrench, but these columns are excluded from the deployable actor encoder.
Cable observations are available in simulation. Stage1-3 use 99 frame columns;
Stage4 adds recovery phase/time for 101 and preserves 451-sample rolling whip
history. The goal has a separate projection. The Gaussian actor uses tanh-bounded
XYZ world velocity commands, limited to (2,2,0.8) m/s. Command limits do not
enforce workspace containment.

The separate value network has causal history plus privileged numerical, motor,
queue, and integration state; it is unnecessary for actor inference. Networks
and optimizers use float32; physics and analytic KL reduction use float64.
PPO uses five current-cohort epochs, minibatches 256, learning rate 0.0003,
clipped ratio 0.2, gamma 1, GAE lambda 0.95, value-loss weight 0.5, gradient norm
cap 0.5, and Gaussian mean-KL limit 0.02. Reward scale 100 is applied once before
GAE. Complete episodes are collected with frozen actor/value weights and their
data are discarded after optimization.

Actor/value Adam optimizers are independent. Up to eight post-step actor trials
restore identical pre-step weights/moments before trying smaller learning rates.
Accepted steps must be finite and satisfy the sampled-cohort KL gate. Rejection
stops actor steps while value fitting completes its passes. This empirical
mean-KL safeguard is not a per-state trust-region guarantee. Entropy counts
completed training episodes, including invalid lanes; evaluation does not advance
it. Actor transfer resets critic, Adam, RNG, and local clocks. Exact resume
retains them and rejects changed execution sources/settings except its explicit
supported budget override.

## Four stages and scientific criteria

Current presets are `configs/learning/whip_stage{1..4}_ppo_cone25_v4.json`,
linked to `configs/tasks/whip_stage{1..4}_cone25_v2.json`. Each uses 128 lanes,
seed 0/1/1/1, and a 10,240-episode cap: 80 complete collections. Automatic evaluation
is one nominal physical case, with actor batch shape 64, after the first
collection, every 512 training episodes, and at the end.

| Stage | Implemented learning |
|---|---|
| 1 | Directed mechanics/unfolding; fresh actor and value network |
| 2 | Fixed broad 1.0 x 0.6 m delivery; preceding Stage1 actor |
| 3 | One fixed target; full actor/value learning and target-size curriculum |
| 4 | Entire preceding Stage3 whip frozen; internal PD return and learned recovery head/logstd |

Nominal start is (0,0,2) m; target center is (1.1,0,2) m. Stage1-3 use the full 4.5 s
horizon. Mechanics proxy M combines bend transport/coherence, non-rigid release,
and bounded forward motion over 0.2/0.4/0.6/0.8 s windows, 0.05 s stride, and
0.15 s tip lag. It is a task proxy, not an experimentally established physical
whip classifier. Unfolding U measures recovered chord/arc slack. Alignment D
is squared positive forward alignment, flat within 25 degrees. Stage1 selects
maximum same-event W=M*U*D. Stage2 uses rectangle proximity W*(1+P); Stage3 uses
interpolated same-event forward crossings and center quality W*(1+I_cross*C).
Their positive bounds are 1/2/2. The virtual rectangle has no collision response.

Geometric hits and strict diagnostics are separate. A strict final hit requires
forward crossing inside the 0.1 x 0.1 m rectangle (inclusive +/-0.05 m per lateral/
vertical axis), forward speed >= 3 m/s, entry angle <= 30 degrees, and validity.
The 25-degree reward plateau does not redefine qualification. Invalid episodes
and missing crossings stay explicit failures/missing measurements.

Stage3 narrows target size 1.0 x 0.6 -> 0.6 x 0.4 -> 0.3 x 0.2 -> 0.1 x 0.1 m.
Progression requires all lanes valid and at least 80% geometric qualification
for three consecutive collections, plus an updated-policy nominal evaluation
gate. Workspace costs are measured but do not gate this precision curriculum.
Retired target-center sampler and combined goal/head-only Stage3 tags are rejected
at new launch. Historical metadata remains readable for saved inference. There
is no current multi-target or between-attempt correction learner.

Workspace radius is 0.55 m about episode start; world height is 1.5-2.6 m.
The outside squared-excess penalty has weight 0.1/distance scale 0.1 m. Inside
motion and acceleration have no current cost. This boundary is soft. Tilt
above 60 degrees, cable below 0.05 m, nonfinite state, or numerical failure
invalidates a lane; workspace excursion alone does not.

Stage4 switches at the causally observed first geometric hit, or 4.5 s miss
timeout, then runs 4 s recovery toward episode-start drone position. Its internal
return velocity uses position gain 1.5 inverse seconds and damping 0.4; only the
existing residual recovery head/logstd are trainable. A miss returns but earns
no recovery credit. Recovery weight 0.5 combines return/cable quality over the
window and final 1 s hold. Hold diagnostics require drone error <= 0.1 m,
speed <= 0.15 m/s, attachment-relative cable RMS speed <= 0.2 m/s, and horizontal
RMS extent <= 0.1 m. Whip, hover, cable settling, strict hit, and workspace remain
separate metrics.

Standard plateau detection may stop after its configured minimum 40 collections
using same-level all-lane windows, entropy floor, and patience. This is heuristic
termination. The last requested full 80-per-stage campaign used a registered
run-local warning mode; the human stopped Stage1 at 68 collections/8,704 episodes.
Stages2-4 were never launched. It is not a new completed chain.

## Retained models and evidence

[The milestone manifest](../models/milestone-20261009/manifest.json) records
one coherent completed chain: Stage1 `20261006-195224-532710` (80/10,240),
Stage2 `20261006-214007-568609` (80/10,240), Stage3 `20261007-001712-887951`
(120/15,360), and Stage4 `20261007-034135-720530` (60/7,680), with counts given
as collections/episodes. Each later initial actor is byte-identical to the
preceding final actor. Stage1 continued from a 36-collection run; its earlier
initial RNG byte-comparison gap remains unresolved.

The checkpoints preserve original 96-column Stage1-3 and 98-column Stage4
interfaces without the later relative goal. Current CPU loading, exact actor-state
restoration, finite synthetic inference, and physical-model/scene compatibility
were checked in milestone selection. Task/training execution sources have
changed, and unused study modules were archived; the final compatibility record
lists those differences. Exact historical resume/reproduction is not claimed.
No checkpoint is relabeled as trained with current observations. Evidence
contains saved settings/model/task/scene, summaries, initialization links,
historical audits, and final nominal evaluation/replay for each stage.

Saved final nominal results: Stage1 valid directed mechanics with unfolding 0.9911
and no geometric/strict hit; Stage2 valid geometric/strict broad hit; Stage3
reached the final target level and hit with 9.24 mm center error, 7.50 m/s forward
speed, and 21.22-degree angle. Stage4 retained the frozen whip hit and passed saved
recovery/cable diagnostics: final drone error 35.9 mm, speed 0.0158 m/s, held cable
relative RMS speed 0.0255 m/s, and horizontal RMS extent 6.25 mm. All four stages
breached the soft workspace; Stage3/4 maximum radial/vertical excess was 0.0883/
0.2085 m. These are saved single-case results, not fresh milestone training,
convergence, containment, varied-target robustness, or flight evidence.

Linux/Crazyflie command and feedback integration, cable-state estimation from
drone telemetry, hardware parameter identification, arbitrary starts, moving
targets, and real-flight safety remain unimplemented or unverified. No model
default or physical fit was promoted during cleanup.

## Recordings and fitting

Source recordings remain external. Take Processing resolves an explicit local
selection or opens a folder picker. Fitting checks catalog/alignment/source
hashes and complete 100 Hz windows. Missing observations are masked, not invented.
Causal history initializes velocities; twist and initial motors remain assumptions
where telemetry is absent. Command-driven and measured-boundary replay answer
different model-validation questions.

[fitting/run_refit.py](../fitting/run_refit.py) requires an explicit base model.
It fits cable EI/Cb/external drag with measured attachment motion, then vehicle
controller response scales with cable/motors frozen in the coupled plant. Default
selection is four training and thirteen historically exposed comparison takes.
`--data`, `--split`, and `--prepare-only` select/record inputs. Candidates and
comparisons are saved under `runs/refit` without automatic default installation.

[fitting/run_residual_fit.py](../fitting/run_residual_fit.py) separately fits
structured nonnegative aerodynamic coefficients: cable first, then drone with
cable weights frozen. It requires a calibrated physical base and gradient gates.
The active residual remains unfitted/disabled. Low fitting loss does not establish
identifiability or reliable whole-maneuver cable prediction.

## Build, validation, and dated change

[build_engine.py](../build_engine.py) invokes the C++17 extensions in
[setup.py](../setup.py); installed Torch enables the tensor build. Rebuild for
each OS/Python ABI. There is no Python solver fallback. Windows requires MSVC/
SDK; the supplied non-Windows branch uses strict floating-point compiler flags.
The compact suite covers current settings, scientific criteria, observation/
action and handoff contracts, checkpoint restoration, recovery, and bounded
numerical behavior. All 65 checks across seven modules passed on this Windows
checkout (OS exit 0, including the CUDA optimizer restart check); 128 retained
Python files parsed and `pip check` reported no broken requirements. The suite
includes short native numerical steps, not a new trained-policy evaluation.
Experiment-specific fixtures are archived. Passing this
suite does not establish learned-policy or flight success.

2026-10-09 change note: the human authorized a simulation milestone cleanup,
including removal/simplification of accumulated regression history. Originals
were copied byte-for-byte to the external archive before this rewrite. The
active record now describes current fixed-target behavior and the coherent
selected models, with their original budgets/interfaces and limitations.
This documentation work did not rerun training, identify parameters, or validate
Linux/hardware. A narrow settings-dialog fix preserves boolean types on accept;
all four current presets now round-trip unchanged. No physical model, reward,
success criterion or stage default changed. [Milestone verification](milestone.json)
records the external archive, SHA-256 preservation evidence and actual checks.
