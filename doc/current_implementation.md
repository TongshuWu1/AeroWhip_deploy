# Current implementation

Last reviewed: 2026-10-16.

This record describes the current simulation, learning pipeline and retained
milestone models. The external development archive
`AeroWhip-development-archive-20261009` remains separate from this checkout.
On2026-10-10 the user stopped the restored-radius Stage2 campaign and explicitly
requested deleting all project `runs/` contents and restarting Stage1.
All2,169 old run files were deleted after the worker stopped; the previous
training, motion-search, cloning and reward-review artifacts are no longer
available in `runs/`. Existing `models/`, `data/` and configurations were preserved.
Historical experiment narratives that relied on those deleted artifacts have
been removed from this active record; their prior text is retained only in
`tmp/stage1-clean-restart-20261010/current_implementation.before.md`.

The random-policy restart stopped after44 collections/5,632 episodes. Its
demonstration-seeded replacement stopped after121 collections/15,488 episodes
when nominal reach and height declined despite improved containment. Both runs
and the reconstructed demonstration are retained. The user approved a new
Stage1 reward with linear height credit and separate outreach/height/joint
contributions, transferring the stopped actor into a fresh critic and optimizers
with the existing exploration schedule restarted. The new experiment uses
150 collections (19,200 episodes),128 lanes and radius0.55 m. The first launch
was paused for simulation work before any completed collection or PPO update.
On2026-10-11 the user requested relaunching training. The
[Stage1 run](../runs/ppo_stage1/20261011-052639-788494/) completed all150 collections
with the paused actor, fresh critic/Adam/clocks and the same approved settings.
GPU scheduling source changed during the pause, so that launch was actor-only
initialization under current execution hashes, not exact resume. Its final native
evaluation improved reach and drone-relative height, but was not contained and
missed the strict crossing angle. The user requested continuing. A one-off
handoff-script error was corrected in a separate
[Stage2/3 continuation driver](../tmp/stage23-continuation-150-20261011/campaign.py),
preserving the failed driver and completed Stage1 artifacts. The
[Stage2 run](../runs/ppo_stage2/20261011-063417-108510/) and
[Stage3 run](../runs/ppo_stage3/20261011-073423-438409/) also completed their150
collections. Stage3 finished at target level1 (0.6×0.4 m), with no native final
0.1×0.1-m hits, and failed the existing Stage4 handoff. Completed budgets do
not imply that the precision curriculum was learned;
their existing Gaussian-height reward definitions remain unchanged. Each later
stage takes the preceding final actor with fresh critic, optimizers and
stage-local exploration clock. Stage4 training is not included. Preparation,
launch and completion records are separated below; deleted-run outcomes are not
presented as retained scientific evidence.

On2026-10-13 the user approved restarting Stage2 with actual-crossing priority:
`0.2W + max_crossing(C + 2G + S) - workspace_cost`, defined below. This separate
150-collection/128-lane experiment transfers the completed Stage2 final actor
with fresh critic, optimizers and exploration clock. The
[preflight](../tmp/stage2-crossing-restart-20261013/preflight.json) reproduced the
source's nominal native motion exactly and verified the new reward. Training
launch/status is recorded in the dated note below. It does not automatically
launch Stage3; a revised Stage3 target ladder remains proposed.

On2026-10-15 the user explicitly skipped the proposed control-learning pilot
and requested training the whip directly from scratch. The first acceleration
[Stage1 run](../runs/ppo_stage1/20261015-111659-848428/) completed all80 collections
(10,240 episodes) from a new random actor and critic, with fresh optimizers and
clocks. It used the unchanged compact Stage1 reward,128 lanes,50-Hz acceleration
actions and its existing plateau rule. The policy deteriorated: every episode
in collections29–80 was invalid, and the final native replay ended on the cable
height envelope. The [retained-results review](../tmp/stage1-acceleration-review-20261015/analysis.json)
records the evidence and reward-accounting diagnosis; completion is not learning
success. No training worker remains active at review time.
No demonstration, saved actor, resume, pilot or automatic later-stage launch is
part of this run. Launch, initial verification and completed-run diagnosis are
recorded separately below.

## Pipeline and entry points

[run_simulation.py](../run_simulation.py) runs the native physical plant.
[run_custom_simulation.py](../run_custom_simulation.py) runs that plant and the
flight viewer with the streaming Crazyswarm ROS adapter in the existing Docker
container. The Crazyswarm checkout's `./docker/start.sh --custom-sim` invokes this
host entry point; `--real` and stock `--sim` retain their existing launch paths.
[run_custom_simulation_test.py](../run_custom_simulation_test.py) sends the separate
scripted PVA connection test through the upstream streaming serializer and saves
its ROS feedback in the existing Crazyswarm `logs/` mount. These simulation entry points do
not initialize a learned actor or run a real-flight server.
[run_custom_real.py](../run_custom_real.py), invoked by Crazyswarm's
`./docker/start.sh --custom-real`, launches the stock real Crazyswarm/OptiTrack
stack on domain42 with RViz disabled and opens the shared read-only GPU viewer.
`--attach` opens the viewer for an existing real stack; Ctrl+C also stops its
matching local ROS launch. Display poses come
from the actual OptiTrack capture, not the onboard estimate or a simulated plant.
Its native cable-only simulation is driven by measured attachment motion and
displayed separately as simulated; it is not a measured/corrected cable state.
[run_training.py](../run_training.py) opens Training; `--start` explicitly starts
a run. The default and `--stage 1` select the new direct-acceleration Stage1
preset. `--stage 2/3` select acceleration variants of the existing reward recipes;
`--stage 4` retains velocity recovery and cannot accept an acceleration actor.
No revised curriculum is included; the current scratch Stage1 run is described above.
`--settings`
selects an explicit settings file, and `--initialize-actor` performs actor-only
transfer. Exact resume is a separate, source-checked path. Its UI entry point
now uses the read-only [resume-history display](../training_history.py),
reusing the existing Training window and charts. Reward-component charts preserve
missing keys and explicit JSON null values as gaps before applying the selected
run's reward scale; recorded numeric zeros remain zeros. This applies to both
the general component chart and the crossing-reward chart.
[run_replay.py](../run_replay.py) renders saved arrays without inference/physics.
Its existing saved-outcome browser remains the default; `--flight-view` opens
the selected saved arrays in the same docked flight viewer as local simulation.
`--run`/`--file` and `--replay` select the existing run/file and NPZ basename.
The manual policy tester remains available from Training.
[Take Processing](../take_processing/run_take_processing.py) manages project-local
recordings and fitting. [validate_stages.py](../validate_stages.py) runs the small
self-contained current-pipeline regression suite; `--output` optionally writes
its result. It does not launch a training campaign.

[evaluation/motion_search.py](../evaluation/motion_search.py), launched with
`python -m evaluation.motion_search`, is a separate nominal Stage1 diagnostic.
It searches scripted velocity strokes through the existing plant and scorer;
it does not train an actor or change default stage presets.
[evaluation/motion_refinement.py](../evaluation/motion_refinement.py), launched
with `python -m evaluation.motion_refinement --output <new-run-folder>`, refines
those programs using separate geometry/speed diagnostics. Its `--source` selects
a source search. The prior search datasets, including its default source, were
deleted in the requested cleanup; a new search is needed before refinement.

[simulation](../simulation) and [native](../native) own reusable physics.
[controllers](../controllers), [tasks/whip](../tasks/whip), [training](../training),
and [fitting](../fitting) own their respective behavior. Settings resolve from
the checkout through [project_paths.py](../project_paths.py), not the caller's
working directory. [configs/simulation.json](../configs/simulation.json) selects
physics, geometric controller, motors, and the unfitted aerodynamic schema.
[configs/scenes/default.toml](../configs/scenes/default.toml) sets the initial
state and viewer.
[deployment](../deployment) owns the external ROS command/feedback adapter and
connection test, separate from reusable physics, controllers and learning.

## Filesystem and UI portability

Shared path validation resolves file-dialog selections and relative inputs inside
the checkout, rejecting outside/home/foreign-drive paths and resolved symlink or
junction escapes. Run-owned replay references must additionally stay inside their
owning folder. Shared artifact writers and settings/trajectory exports use these
checks; new model-source, actor-source, resume, and evaluation request references
are project-relative. Existing historical evidence is immutable. This is
application-level containment, not an OS sandbox or protection against hostile
concurrent filesystem replacement.

[model_catalog.py](../model_catalog.py) discovers manifest-packaged stages in the
policy-source/replay browsers. It verifies registered artifact bytes and maps the
original final-evaluation replay filename to `final-replay.npz` in memory only;
recorded metric checks remain unchanged. Default replay selects packaged Stage4.
The empty Take Replay window initializes its folder before opening the picker.
Invalid file selections are reported without outside writes. The low-level
Training default now matches the current Stage1 preset selected by the launcher.
The evaluation worker uses the same frozen actor factory as the policy tester,
including all current goal/phase columns; this does not change task scoring.

Training charts now follow `request.json` exact-resume links, assembling parent
and current-session records in memory. Each ancestor is cut off at the saved
`resumed` collection; before that event exists, the selected checkpoint supplies
the boundary. Later parent collections are excluded when resuming an older
checkpoint. Parent/child identities must agree (the existing identity excludes
only the episode-budget override), references remain project-contained, and
inherited copies of the same evaluation are deduplicated. Training returns,
reward components, losses, validity and exploration curves retain their original
collection numbers and selected-run reward scale. New stages and actor-only
transfers begin separate histories. Missing/incompatible ancestry produces a
visible warning while retaining available current-session records. JSONL logs,
checkpoints, replay files and accounting arrays are never rewritten by this
reader. Earlier evaluations retain their original run ownership for the latest
preview and selected/best/latest replay actions. An earlier evaluation is
identified as inherited; a new failed evaluation remains the latest result.
The display lives at the UI entry point outside pinned learner execution sources,
so no execution-hash comparison, trainer code, identity or scientific criterion
was weakened. Already-open windows keep their loaded code; during the previous
campaign an updated viewer was opened separately from its worker-owning window.
That campaign is now stopped. Later Training launches use the updated history
display.

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
| Observation / controller / policy rate | 100 / 200 / 50 Hz for acceleration; retained velocity presets 10 Hz |
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

The geometric controller retains its velocity-feedback path for `velocity_v1`.
The new `acceleration_direct_v1` path converts world XYZ acceleration requests
(excluding gravity) directly to nominal force with the existing static cable-weight
trim, then desired attitude, collective thrust and body torque. Initial yaw stays
fixed. This path has no position/velocity feedback or integrated references;
zero acceleration supplies weight trim, not braking or position hold. Both paths
use the existing attitude loop. There is no integral loop or exact dynamic
cable-force cancellation.
Motor inverse mixing is clipped per motor; first-order motor response supplies
the delivered wrench. Motor parameters are marked uncalibrated. Continuation
requires previous BDF2 state/step, motors, held wrench, command queue, and clocks.
Drone telemetry alone cannot exactly resume a simulated trajectory.

Captured CUDA tasks default to the GPU-controlled Picard loop on 64-bit Windows
and on64-bit Linux when Torch was built with CUDA12.8 or newer. Other platforms
and older Linux Torch CUDA builds default to host-controlled replay.
`AEROWHIP_CONDITIONAL_SOLVER=0` selects the host path; `1` selects the conditional
path and requires a CUDA driver API12.8+ and working NVRTC. An explicit task
constructor choice takes precedence over this environment variable. The
[conditional driver](../tasks/whip/conditional_driver.py) loads the Windows driver
or Linux `libcuda.so.1`, and locates NVRTC in Torch/NVIDIA package libraries or
the Linux system library search. Backend failures propagate rather than silently
changing the solver. The same compiled Picard slot, first-converged endpoint
mask,20-slot cap, and numerical acceptance gates remain in use. CPU/differentiable
fitting paths are separate.

The retained [Linux scheduler comparison](../tmp/simulation-performance-20261011/scheduler-parity.json)
checks50 synthetic100-Hz steps in both1- and128-lane batches. Host and conditional
paths match every physical, BDF2-history, motor and held-wrench field bitwise;
the one-lane scalar comparison's maximum absolute difference is3.17524e-14,
within the existing1e-9 numerical comparison tolerance. Both schedulers reject
collapsed first edges and retain the20-slot cap. This is numerical evidence,
not a model fit, policy result or real-flight result. A full current Stage1–4
Linux campaign and a fresh Windows/Linux execution comparison remain unverified.

The [local performance record](../tmp/simulation-performance-20261011/summary.json)
uses RTX4080, Torch2.11/CUDA13.0 and the existing11-node float64 plant. Across
three warm trials, the median of mean128-lane100-Hz step times is41.897 ms for
the host scheduler and37.056 ms for the conditional scheduler:11.55% less step
time, or13.06% more plant throughput. This measures the plant adapter without
policy inference, task scoring or PPO updates; it does not establish the same
gain for complete training. Native single-flight PVA steps average2.143 ms and
the prescribed cable adapter2.009 ms in separate mild synthetic motions; these
are not timing guarantees for whipping maneuvers or a causal real-flight observer.

The [viewer](../viewer/pyvista_viewer.py) renders through NVIDIA OpenGL. The prior
1440×900 off-screen display-motion baseline averaged2.159 ms per geometry update
and render with vertical-sync waiting disabled; default synchronization averaged
17.084 ms. Those timings precede the viewer rebuild and are not measurements of
the new complete UI.

### Local simulation execution and display

[viewer/sources.py](../viewer/sources.py) now gives the local native simulation
one background owner thread, independent of Qt render callbacks. Native stepping
releases the Python GIL, reacquiring it for the existing Python controller.
The worker polls the existing controller-input callback on its5-ms wake cycle
and calls the unchanged `Simulation.advance` with monotonic elapsed time.
The100-Hz output,200-Hz physical controller, maximum1/600-s physics substep,
timestamped reference events and configured transport delay remain unchanged.
Desktop scheduling is not a hardware real-time guarantee. Existing wall-time
debt/burst limits remain: overload slows simulated time without enlarging or
skipping physical steps. Input polling previously ran in the viewer timer.

A bounded64-command FIFO applies play/pause, single step, reset, source changes,
optional force diagnostics and pacing changes between output steps. User actions
wait for acknowledgement, with a5-s response bound; render/telemetry reads never
wait for integration. Pause and pacing changes discard paused wall time. Closing
the window stops and joins the worker. Failure pauses execution and publishes a
visible error with the last published physical state. Changing command sources
while running is rejected. The existing trajectory window uses the facade's
source/reset controls and the controller's pure `prepare_reference` for CSV
export; it cannot directly access native state through that facade.

The latest-frame mailbox owns immutable array bytes, keeping producer/consumer
state separate and bounding backlog to one display snapshot. Its source revision,
monotonic publication time, running/error state and simulation timeline are
distinct. Rendering may skip intermediate display snapshots; numerical states,
command queues and controller updates are not skipped. Recorded replay uses a
separate adapter over `StateReplay`, with no plant, actor inference or new scoring.

[viewer/display_state.py](../viewer/display_state.py) defines a read-only world
description and named-robot display frames. Positions use metres in the declared
world frame with Z up; XYZW quaternions rotate body to world. Velocity/acceleration
use world coordinates, angular velocity uses body coordinates. Each robot and
optional cable has its own timeline timestamp and simulated/measured/estimated
provenance. Missing telemetry remains absent rather than zero-filled. World/frame
names must match; source adapters are responsible for transforms. The local
adapter labels its robot `drone` and its cable simulated; recorded playback is
explicitly labeled recorded simulation. Synthetic measured/estimated display
fixtures demonstrate presentation only; the implemented OptiTrack live adapter
is described below and its synthetic checks do not establish real-flight evidence.
These frames contain display data, not BDF2 continuation or policy observations.
The thread-safe `DisplayFeed` accepts these immutable frames directly without
constructing a plant. `PyVistaViewer.show_source(feed)` uses the same window with
playback/trajectory actions absent. An external producer must supply frame
transforms, timestamps and connection errors; no middleware or estimator is
included in the feed. Optional force telemetry remains labeled model forces.

[viewer/robot_scene.py](../viewer/robot_scene.py) retains CAD, rotor, cable-tube,
marker and overlay actors per named robot, updating transforms/coordinates rather
than recreating meshes per frame. Robot addition/removal or cable-topology changes
rebuild only that robot's actors. Multiple robots and absent cables are supported.
CAD name plates now show display identity instead of a hardcoded drone number;
this does not identify or calibrate real hardware. Floor/grid contrast changed
only in the renderer; there is still no physical floor/contact model.

The [docked flight window](../viewer/simulation_window.py) has a central3D view,
Displays and Inspector docks, camera presets (orbit/top/front/side), scene/drone
framing, translation-only drone following, layer/path/force toggles and visual
force scaling. Playback controls retain0.25/0.5/1× pacing, step/reset and trajectory
editing; saved replay adds sample seeking. Force arrows start hidden. Qt checks
for new snapshots on a16-ms timer, limits source-driven rendering to60 Hz, and
refreshes telemetry at10 Hz; camera interactions can render separately. Running
snapshots older than0.25 s are labeled stale, paused snapshots are labeled paused,
and robot/cable timeline lags and unavailable fields remain explicit. Pose timeline
lags over0.25 s also mark a running source stale; older cables carry their own
stale label. A missing robot frame shows no data and removes old robot actors.
The corner orientation axes follow the camera and the world-axes display toggle.
This age threshold is a display indicator, not a scientific acceptance/safety
criterion.
Existing Training/replay outcome windows retain their scoring and artifact UI
while reusing the updated renderer. The streaming ROS source described below
shares this window, with local trajectory editing absent because ROS owns its
command stream. A real-flight cable observer, firmware-controller equivalence
and learned-policy deployment remain unimplemented or unverified.

### OptiTrack connection to real flight display

The user approved a separate real-flight viewer on2026-10-13 and explicitly
required real OptiTrack capture for the drone. The
[launcher](../run_custom_real.py), [host feed](../deployment/real_flight.py),
[ROS reader](../deployment/ros_real_bridge.py) and
[stock-stack supervisor](../deployment/real_stack.py) implement that scope.
The adjacent Crazyswarm `docker/start.sh --custom-real` launches the existing
`crazyflie launch.py` with `backend:=cflib`, `mocap:=True`, `teleop:=False` and
`rviz:=False` on domain42. Exact snapshots of both original YAML texts are passed
to temporary container files; hardware configuration is not rewritten. An
exclusive host launch lease and ROS graph check reject an already running real
stack; `--attach` starts only the display reader and can coexist with RViz.

All enabled Crazyflies with vendor or `librigidbodytracker` motion capture are
displayed by YAML name by default; repeated `--robot` restricts that list.
The YAML initial position is retained as configuration metadata, never shown
as a measured pose. The display begins empty. This adapter constructs a cable-only
simulation, with no drone dynamics, controller or policy. It uses the existing host `.venv` GPU window and container
ROS runtime, sending exact reader/supervisor source through Docker stdin.

The sole drone-pose source is the configured `/poses` `NamedPoseArray` publisher
`/motion_capture_tracking`. Read-only parameter queries verify its backend is
`optitrack` or `optitrack_closed_source` and its configured frame matches the
loaded YAML. The reader requires one discovered publisher endpoint belonging to
that node and rejects an ambiguous/replaced source until verification succeeds.
ROS Jazzy does not expose publisher GID in Python callback metadata; endpoint
checks use ROS graph discovery. No `/NAME/pose` estimator or drone TF
fallback supplies robot poses. The motion-capture node's TF path may replace
unavailable orientation with identity; `/poses` is read before that substitution
so invalid orientation is explicitly rejected here.

Positions are metres in the named capture/display frame; XYZW orientation
rotates the captured rigid-body frame into that frame. Pose age uses the capture
header timestamp on the real ROS clock (`use_sim_time=False`). The default
display frame is `topics.frame_id` in motion-capture YAML, currently `mocap`.
`--frame world` or another frame requires an existing TF transform at that
capture timestamp, including its rotation and translation. No identity
`mocap`-to-`world` assumption or alignment calibration is introduced. CAD uses
the existing shared model/visual attachment offset; this does not calibrate the
actual OptiTrack rigid-body pivot to the physical drone/cable attachment.

Duplicate-name frames and individual invalid/nonfinite poses or missing
orientations are rejected. Quaternion norm error above0.001 is rejected; accepted orientations
are normalized for the renderer. This is display validation, not a fitting or
scientific acceptance criterion. The reader preserves the last valid pose while
an independent60-Hz heartbeat advances the display timeline; tracking gaps become
stale under the existing0.25-s display-age threshold. Invalid source/transform
diagnostics are visible errors, and ROS clock regression clears poses and starts
a new timeline generation. Unknown capture names are ignored. Vendor tracking
is labeled measured; `librigidbodytracker` reconstruction is labeled estimated.
Measured-drone velocity, acceleration, body rates and forces remain absent.
The [live cable boundary driver](../simulation/live_cable.py) reuses the existing
native `Model.prescribed_advance` material-cable solver, preserving free-node
positions/velocities, material angles and BDF2 history separately for each robot.
The real drone is not integrated or overwritten in a coupled plant. Cable
provenance is **Simulated**; drone provenance retains the verified capture source.
The shared viewer already exposes both provenance labels and independent ages.

The relay batches every accepted selected-robot pose between its60-Hz display
heartbeats in `pose_samples`; latest poses still populate `robots` for display.
Accepted buffered samples are processed even if heartbeat source verification
changes after their capture callbacks. Source timestamps use integer ROS-header
nanoseconds relative to the relay epoch before conversion to seconds. These
remain capture-node receipt timestamps, not synchronized Motive exposure times.
Duplicate timestamps do not advance physics. A512-sample relay buffer bounds
memory; overflow is explicit and freezes the cable. The transport's owner thread
advances native physics independently of the Qt renderer; rendering cannot reset
or advance cable state. These connection traces are separate from the retained
source logger and do not constitute calibrated flight/fit evidence.

For the configured attachment offset, root position is `p + R*r_attachment` in
the selected Z-up display/capture frame. Arrival of the next pose permits causal
advancement over the completed interval: root endpoint velocity is a backward
secant, with cubic Hermite P/V/A retaining the previous endpoint velocity.
Attitude uses shortest SO(3) geodesic interpolation with piecewise derived body
rate. These are boundary-interpolant derivatives, not measured drone telemetry;
the method uses no future samples beyond the latest received endpoint and does
not extrapolate between arrivals. Existing native maximum substep and constraint
checks are unchanged. `--model` defaults to `configs/simulation.json`; model/source
hashes and initial-condition/timing assumptions are saved in launch metadata.
Scene attachment offset and marker-interval lengths must match the selected model.
An implicit endpoint-convergence, position-projection or velocity-constraint
rejection retries that completed interval from the same accepted state/BDF2
history. Each retry multiplies the baseline substep count by2,4, then8, so even
an interval shorter than the native maximum step is actually subdivided. Native
iteration limits and convergence/constraint tolerances remain unchanged; successful
output alone commits state. Invalid inputs, other failures and exhausted refinement
still freeze the cable. Launch metadata records `numerical_refinement_factors`
and the retry/subdivision rule. This is bounded numerical refinement of the
boundary driver, not a
parameter fit, smoothing filter or modification of the coupled/training integrator.

At the first accepted pose the model's hanging, stationary, untwisted cable is
an explicit prior; YAML positions do not supply real state. Frame/body-origin
calibration still requires physical verification. There is no ground contact,
so a ground-level attachment can yield cable geometry below the visual floor.
Gaps above0.25 s, timestamp regression, overflow or integration failure freeze
the last valid cable and show an error; viewer restart explicitly reinitializes
it. A source endpoint change or ROS clock generation change initializes a new
timeline. New flight-script identities clear trails without resetting cable
physics/history. No measured cable correction, force publication, artificial
propeller rotation or learned-policy deployment is introduced.

#### Experimental marker-cloud cable reconstruction

`--track-cable` adds a separate read-only observation path for one explicitly
selected robot, using [deployment/cable_tracking.py](../deployment/cable_tracking.py).
It does not feed the native cable integrator, controller, fitting or policy.
The selected model's cumulative `marker_interval_lengths_m` define material marker
arc distances; the user confirmed the real cable retains this spacing. Current
distances are0.063,0.150,0.250,0.350,0.450,0.550,0.650,0.7525,0.8525,0.9525 m
from the attachment. Root is the matching measured rigid-body pose plus the same
body attachment offset used by the display. Length overrides and heuristic
settings can be provided with `--cable-tracking-config`; the explicit example is
[configs/ros/cable_tracking.json](../configs/ros/cable_tracking.json). Actual settings
and source hashes are saved in launch metadata.

The optional relay subscribes to `/pointCloud` (`sensor_msgs/PointCloud2`), verifies
its sole publisher is the configured capture node and also checks the currently
verified pose endpoint. XYZ FLOAT32 decoding respects offsets, endian convention,
organized rows and padding. Current pinned `libmotioncapture` appends both NatNet
OtherMarkers and LabeledMarkers into this cloud, discarding marker IDs, asset and
visibility flags. This is therefore a **mixed candidate cloud**, not an
unlabeled-only or visibility-certified cable measurement. Point indices identify
only a row in one retained cloud; they are never persistent physical IDs. The
raw capture-frame XYZ array and separately transformed display-frame array are
recorded, with nonfinite coordinates represented by JSON null. Backend/frame
verification is the same as the pose path. Requested TF is evaluated at the
sample timestamp for both cloud and pose; there is no assumed frame alignment.

Clouds and selected poses pair by exact integer ROS header timestamp, which the
existing capture node sets once for each received frame. They are receipt stamps,
not Motive exposure timestamps. Either callback arrival order works. Pending
frames are bounded to32 and unmatched clouds expire after100 ms when subsequent
data arrives, retaining an explicit missing-pose record with no reconstruction.
Source/clock generation changes clear pairing and association history. Each paired
cloud is sent separately from the60-Hz display heartbeat. The relay caps frames
at2048 points, reports oversized/malformed clouds instead of truncating them, and
uses a1-MiB packet limit. `connection.jsonl` stores `capture_display` marker packets
before association, followed by `cable_reconstruction` results; all accepted cloud
callbacks are retained, while upstream/DDS drops are not detectable as camera
frame loss. This remains connection/reconstruction evidence, not lossless source
capture or a fitting dataset with verified identities.

Initialization requires a complete unambiguous approximately vertical hanging
cable, using world -Z and a3.5-cm spatial gate around the arc-distance prior.
It does not infer arbitrary coiled initial geometry. Tracking predicts each
material marker from its last accepted position and bounded constant velocity;
temporal gates widen with elapsed time. Velocity updates blend the new finite
difference with the previous estimate using `tau = position_noise_m /
maximum_speed_mps` (1 ms with current settings):
`v_new = (delta_position + tau*v_old)/(dt + tau)`, followed by the existing speed
cap. This damps spikes from bunched ROS receipt timestamps; it does not recover
camera exposure timing or constitute calibrated velocity estimation. An ordered beam search considers missing
nodes and one-to-one candidate assignments, with geometric upper bounds between
the attachment and markers and between all assigned marker pairs. Chord distance
may be shorter than arc spacing; no straight-segment equality is imposed. The
root-distance bound filters distant clutter, and1-mm spatial deduplication handles
duplicate list entries while preserving raw arrays. Rigid-body candidates are
not semantically excluded: their IDs/flags are unavailable upstream. Candidate-pair
distances are computed once per cloud over the retained candidate union and reused
across beam hypotheses; the matrix is not allocated over the full raw cloud.

Default beam width128, at most12 candidates per marker,15-mm chord tolerance,
12-mm noise scale,12-m/s motion bound, missing cost4 and ambiguity margin1 are
engineering settings, not scientifically calibrated probabilities or acceptance
thresholds. Results record whether search pruning occurred. Near-best retained
hypotheses that disagree on a marker leave that marker null; reported confidence
is a normalized score margin among retained hypotheses and cannot certify identity
outside the bounded search. Cloud capacity overflow produces no assignments;
more than12 candidates for one marker suppresses that marker's choices while
allowing other identities to remain visible. The existing root-distance bound
is applied before this limit so geometrically impossible points do not consume
the candidate budget. Partial occlusion preserves accepted
per-marker history. After0.25 s without an accepted observation, that marker
uses geometric recovery instead of stale constant-velocity extrapolation:
root/arc bounds and recent neighbours' reachable regions select candidates,
then the same one-to-one search and exact pairwise chord bounds test assignments.
Recovery candidates have equal geometric cost; disagreement among retained
near-best hypotheses still leaves the marker null. Accepted recovery resets its
velocity to zero. Results include one-based `reacquired_markers` and
`crowded_markers` lists for inspection. When all marker histories are older than
0.25 s, complete hanging reinitialization is required. These gates do not prove
material identity in clutter or after a fold. A flight-script trajectory epoch
does not reset associations. Arbitrary-shape initialization,
multiple interacting cables, model-based measurement correction and identity
ground truth are unimplemented.

[viewer/cable_tracking_scene.py](../viewer/cable_tracking_scene.py) renders separate
persistent gray candidate points and blue estimated marker identities/straight
links, alongside the existing green simulated cable. Links are only between
adjacent observed material markers (including the measured root); no gap filling
or smooth centerline inference is displayed. Labels remain Estimated because
identity is inferred and upstream visibility flags are unavailable. The inspector
shows count, age, status and minimum heuristic confidence. Separate visibility
controls select the estimate and raw candidates; overlays older than0.25 s are
hidden. Numerical integration, physical parameters and training stay unchanged.

#### Hand-test recording and offline playback

`run_custom_real.py --capture-only --robot NAME` launches only the existing
OptiTrack node plus the shared viewer and enables marker-cable tracking. It
does not start `crazyflie_server`, teleoperation or a radio connection. The
default domain is43; an explicit domain in0..232 is supported for isolated
capture. `--capture-only` and `--attach` are mutually exclusive. Existing nodes
and domain leases are checked before launching; no existing process is stopped.
Normal real-flight launch and attach still default to42. To attach to a default
capture-only stack, pass `--attach --track-cable --domain 43` explicitly.
[deployment/real_stack.py](../deployment/real_stack.py) builds capture parameters
from the same YAML bytes and stock Crazyswarm rigid-body selection rules, omitting
an empty `rigid_bodies` map because the direct ROS YAML parser rejects it.
It launches `ros2 run motion_capture_tracking motion_capture_tracking_node` with
that temporary configuration. Original YAML files and Motive settings are not
edited. The existing session/process-group ownership checks supervise this node;
viewer close, Ctrl+C and worker crash clean up an owned capture-only session.
The normal real-flight viewer's close/attach behavior remains as described below.

The shared window now offers **Record**, **Stop & save**, **Replay last**, and
**Open recording…**. Record controls are enabled for configured live real/capture
feeds; Open recording is also available in the other shared-viewer modes.
`--record` starts a clip at launch, including its initial waiting-for-capture
frame, and works with bounded headless operation. Each new clip gets a distinct
`output/custom_real/<session>/recordings/<UTC-time>-<id>/` directory. It contains
`recording.json` (schema `aerowhip_display_recording_v1`), `frames.jsonl`, and
`events.jsonl`. Scene/display configuration and the session's model, tracker,
frame, YAML/source identities and settings are copied into the manifest. This
is a display recording, not a native checkpoint or calibrated fitting dataset.

[viewer/recording.py](../viewer/recording.py) serializes the complete immutable
display contract: measured drone poses and absent telemetry, simulated cable
geometry, estimated cable marker identities, observed masks/confidence, finite
raw display points, original source timestamps, provenance, trajectory epoch and
error. Received pose/cloud packets, including raw capture-frame coordinates and
nonfinite/null markers, and association results are retained separately as events.
All accepted display publications are recorded at the relay heartbeat rate
(nominally60 Hz), independent of which frames Qt renders; every received packet
during the clip is an event. Recording does not reset live simulation/association
history and does not recompute prior motion. There is no claim of retaining
camera frames that were dropped before the relay. Starting a clip snapshots the
current display; physical initialization still uses the existing hanging prior.

The feed's owner lock makes Start/Stop and sample/frame writes atomic relative to
live processing. Writes use line-buffered append-only streams; Stop and orderly
window shutdown flush/fsync streams before atomically marking the manifest complete.
Write errors remain visible in recording status and prevent a complete mark. A
native crash can leave an incomplete recording whose complete lines are replayable
with a warning; only an interrupted final partial line is skipped. Malformed
interior lines and inconsistent completed frame counts are rejected. No recording
is overwritten or silently promoted to complete after interruption. Synchronous
disk writes can add capture latency; this is a desktop recorder, not a hard
real-time or lossless camera recorder.

[run_recording.py](../run_recording.py) opens a folder/manifest or a file picker
and uses `RecordingSource` with the same GPU viewer. It loads no model/native
plant, starts no ROS nodes, and reads saved results without rerunning physics or
association. Frame offsets are indexed once; only the requested frame is decoded
on seek. Play/pause, step, reset,0.25/0.5/1× pacing and a timeline use elapsed
host-monotonic recording time, while original ROS/pose/cable stamps and ages remain
unchanged inside displayed frames. This permits explicit replay of clock resets,
gaps and errors. Recorded errors/incomplete warnings remain visible but do not
disable playback. Seeking clears display trails; camera/trails are presentation,
not saved physical results. Offline replay requires neither Docker nor hardware.
`--inspect` validates/indexes frames and prints a summary without opening the GPU
window. Live and replayed blue estimated markers remain separate from green
simulated cable geometry; comparing them does not itself establish accuracy.

The existing [Docker receiver patch](../../Crazyflie/crazyswarm2_jazzy_whipping/docker/motion_capture_tracking-optitrack-interface.patch)
now also repairs NatNet unicast while preserving multicast. The server advertises
the mode at connection. Unicast retains the endpoint that sent `NAT_CONNECT`,
sends `NAT_KEEPALIVE` every1 s, and ignores interleaved frame packets while waiting
for server/model replies. Frame reception ignores control replies and retains
the latest complete frame. Command replies and missing frames time out after5 s
with an explicit error. Multicast still binds the advertised data port and joins
the advertised group on `interface_ip`. The running container was rebuilt and
future Docker builds apply the same patch to the pinned source. Motive mode changes
require ROS reconnect/restart; this work does not modify Motive settings.
The patched capture node also finishes normally when SIGINT invalidates its ROS
context during frame reception/publication or executor creation; exceptions during
active operation are preserved. This addresses an observed capture-only shutdown
abort and does not suppress source/integration failures during operation.

The adapter publishes no flight commands, invokes no command services and
changes no firmware parameters. Existing flight scripts still use the normal
Crazyswarm `./docker/shell.sh --real` domain42 shell. Closing the GPU window stops
only its reader; an owned real stack remains active until Ctrl+C in the launch
terminal, matching RViz lifecycle. The supervisor owns a separate process group;
cleanup verifies session identity, process environment and group ownership
before signalling it. With `--attach`, startup snapshots the matching local
`ros2 launch crazyflie launch.py backend:=cflib mocap:=True` process and its
children on the selected domain. Ctrl+C/SIGTERM also stops that captured stack.
[Stack lifecycle handling](../deployment/stack_lifecycle.py) checks PID start
time (`/proc/<pid>/stat` clock ticks), exact argv and ROS domain before each signal; it remembers child identities
if the launch parent exits first. SIGINT lets ROS launch shut down normally;
surviving captured descendants receive SIGTERM/SIGKILL only after timeouts.
An attached window close or bounded headless duration still leaves ROS running.
Absent matching local launches (including remote or manually assembled stacks)
are reported as viewer-only shutdown; ambiguous matches are refused. Docker and
independently started flight/logger processes are outside this captured tree.
The graphical real launcher now runs its viewer/reader in a separate worker
process with a host supervisor. Ctrl+C sets a shutdown event; a Qt timer closes
the window outside the Python signal handler. The shared window completes VTK
module initialization before importing PyVistaQt; this avoided the observed
intermittent native import/finalization crashes in the retained checks. Normal
Python finalization is retained. If a native worker nevertheless crashes before
a stop request, an owned launch keeps its supervisor terminal and ROS stack
running and prints attach/recovery instructions. Ctrl+C in that terminal then
performs session-checked cleanup. If the crash happens during requested shutdown,
the surviving supervisor performs that cleanup directly. The attached supervisor
also retains its startup snapshot for cleanup after a worker failure during a
stop request. No container-wide kill or automatic real-stack restart is performed.
Both custom modes isolate their Docker clients from terminal signals and defer
GUI shutdown through the shared Qt timer. Read-only feeds omit playback controls,
shortcuts and playback status hints in the shared window.

Each launch writes unique `output/custom_real/` connection evidence with session,
domain, robot selection, YAML/source hashes, frame and timestamped sampled
display packets. This is not a full-rate OptiTrack recording or fitting input.

Both ROS-connected viewers monitor endpoint identities on each selected robot's
`cmd_full_state` topic and `/all/cmd_full_state` at10 Hz. A newly discovered
publisher advances a per-robot display `trajectory_epoch`; disappearance alone
does not clear trails. The installed Crazyswarm Python client creates these
publishers when a script connects, including scripts using high-level commands.
Real capture packets carry the epochs to the host; the custom simulation uses a
separate display-session packet. On an epoch change the renderer clears the
drone trail, cable-tip trail and displayed reference, preserving pose, cable
geometry and provenance. **Clear trails** provides the same display-only cleanup
manually. No flight command, physical reset, clock change, plan cancellation or
recording deletion is caused by this display event. Discovery can lag a new
client by DDS discovery plus the polling interval; a long-running client reused
for another flight does not create a new endpoint and should use the button.
Publisher rediscovery may also clear display history; this is a display heuristic,
not a scientific flight-segmentation record. The prescribed-motion cable simulation
supplies explicitly simulated state; optional marker association supplies separate
estimated identities as described above, without correcting simulated state.
The retained [synthetic reconstruction checks](../tmp/custom-real-20261013/capture-checks.json)
cover direct capture, rotated/translated TF, multiple names, capture freezes,
invalid orientation, duplicate names and recovery with an unrelated estimator
still publishing. [GPU display checks](../tmp/custom-real-20261013/gui-checks.json)
and [capture](../tmp/custom-real-20261013/real-viewer-synthetic.png) show the shared
NVIDIA-rendered read-only window using synthetic poses.
[Source/frame rejection checks](../tmp/custom-real-20261013/source-frame-rejection-checks.json)
confirm that a non-OptiTrack source and an unavailable TF frame show diagnostics
without accepting a pose.
[Qt signal checks](../tmp/custom-real-20261013/gui-signal-checks.json) confirm that
Ctrl+C closes the actual GPU window/local event loop and joins its display reader.
[Mock lifecycle checks](../tmp/custom-real-20261013/stack-lifecycle-checks.json)
verify exact YAML/launch flags, keeping the stack after viewer close, Ctrl+C
cleanup and rejection of an unowned process. These checks used no real radio,
OptiTrack hardware or flight; actual capture alignment and hardware behavior
remain unverified. Physics, fitting, learned controllers and training are unchanged.

### Crazyswarm command connection to simulation flight

[deployment/custom_simulation.py](../deployment/custom_simulation.py) adapts the
existing native owner thread without modifying its plant. PVA remains the default;
an explicit ROS-only acceleration selector reuses the sampled nominal controller
with zero translation gains while armed.
The host `.venv` owns native physics and GPU visualization; the existing Docker
ROS Python runs [deployment/ros_bridge.py](../deployment/ros_bridge.py). The
launcher sends the exact checked adapter source to Docker Python stdin, sourcing
the installed ROS Jazzy/Crazyswarm setup. No image rebuild, host ROS installation,
container Torch installation or additional volume mount is required. The
simulation client patch is applied to the existing container's source symlink;
the adjacent Dockerfile includes it for future builds. This is a Linux/Docker
command adapter. Firmware high-level planning is reused; firmware motor-control
and estimator equivalence are not implemented.

`./docker/start.sh --custom-sim` requires the container already started with
`./docker/compose`. The default container is `crazyswarm2_jazzy_whipping_dev`;
`--container` can select another existing container. The launcher finds the
adjacent AeroWhip checkout, with `AEROWHIP_ROOT` as an explicit location override.
Alternatively, run `run_custom_simulation.py` directly in the host project/PyCharm.
The simulation starts playing after the ROS bridge connects. `--headless` omits
the viewer and `--duration` bounds headless wall time. Closing the viewer stops
the transport, ROS adapter and native owner. Ctrl+C/SIGTERM sets the same deferred
shutdown event in graphical and headless modes. The bridge is captured by its
unique session argument and ROS domain after connection; identity-checked
cleanup also handles a failed transport shutdown. Domain/port leases are released
without stopping Docker. The standalone `run_simulation.py` uses the same Qt
stop event and closes its controller receiver on Ctrl+C. The renderer remains on the host
GPU. Only one drone/cable plant is exposed. [Configuration selection](../deployment/configuration.py)
reads the hardware `crazyflies.yaml`: enabled Crazyflies, type/connection, URI and
reference-frame precedence follow the stock server. One enabled drone is inferred;
multiple enabled drones require an explicit `--robot` selection and only that
plant is exposed. Both PVA and direct acceleration require the `world` frame; other
frames fail explicitly. RViz itself consumes TF and a transient-local robot
description; initial states are established by the simulation server.

`--crazyflies-yaml` selects hardware metadata and the selected robot's exact
`initial_position` in world-frame metres. It is the default position for startup
and Reset; there is no implicit height offset or simulation YAML override. Reset
restores the configuration loaded at startup rather than rereading edited YAML.
`--simulation-yaml` is optional and overrides position only when explicitly
supplied. The retained [airborne example](../configs/ros/custom_simulation.yaml)
sets `cf_2` to `(0,0,1.1)` m and can reproduce the earlier airborne connection
setup by passing `--simulation-yaml configs/ros/custom_simulation.yaml` (relative
to the AeroWhip checkout). The hardware YAML remains `(0,0,0)` m. Original and
effective positions, paths/hashes of the files actually loaded and robot identity
remain in metadata. The effective position initializes the native plant and is
exposed through `robots.<name>.initial_position`. Orientation, velocities and
material-frame settings still come from the selected AeroWhip scene; explicitly
supplied cable geometry translates by the change in its drone origin.

Initialization does not infer a physical floor from the world origin or from
`viewer.ground_z_m`. The floor/grid are display geometry and no longer reject
otherwise valid native initial states. The unchanged native model is free-space:
omitted cable geometry hangs vertically, including below the displayed plane
when the configured drone position is near that plane. Native root/length,
velocity and material geometry checks remain active. A `z=0` initial position
does not implement resting contact, cable support, touchdown or unpowered flight.
General physical world/contact behavior remains unimplemented. Neither hardware
firmware parameters nor the YAML robot type override the explicit native model
or geometric controller; the adapter still owns one selected drone/cable plant.

ROS uses domain142 by default, separate from the existing real/stock simulation
domain42. The new launcher and test reject domain42. Both sides identify the
same robot, domain and fresh session over localhost TCP port47654 (configurable;
port0 chooses an unused port). This transport is distinct from the existing
manual-control TCP port47653. Physics never runs in a ROS or Qt callback.
An OS-released host domain lease prevents two custom launches from sharing the
global Crazyswarm service names, including launches using different TCP ports.
A ROS graph probe also rejects an existing `crazyflie_server` before advertising
the custom server. Neither check terminates another server. These checks do not
replace domain42 separation or form a distributed ROS locking protocol.
The listener enables `SO_REUSEADDR` for immediate restarts after TCP `TIME_WAIT`,
while preserving live-listener exclusivity. A busy port produces a short launcher
error; it never terminates the existing listener. If an older connection still
blocks binding and no simulation is running, `--port 0` selects a free port and
passes that actual port to the Docker adapter automatically.

The supported command is `crazyflie_interfaces/FullState` on
`/<robot>/cmd_full_state`. World positions are metres at the tracked top-plate
origin; velocity and acceleration are world-frame m/s and m/s², yaw is radians,
and angular-rate feedforward is the existing nominal desired-body XYZ rad/s
reference. Command orientation must be a normalized yaw-only XYZW quaternion.
The adapter accepts `world` and upstream `/world` headers, rejecting other
frames, nonfinite references and unsupported roll/pitch attitude commands.
Streaming clients must use `use_sim_time:=true`: `/clock` is the native plant
timeline. Headers must lie within `[latest_time-0.5, latest_time+0.02]` seconds,
with nondecreasing stamps and increasing transport sequence numbers. These are
transport validity limits, not fitted latency or flight acceptance thresholds.

Accepted commands enter a128-packet FIFO. At most one command becomes a native
request event at each next available100-Hz output boundary; bursts stay ordered
and queue overflow is explicitly rejected. Header time, published receipt time,
actual request boundary and applied-event time remain distinct in retained logs;
headers never backdate a causal live command. The existing native30-ms modeled
delay and200-Hz controller are preserved, so transport/receipt quantization adds
its measured latency to that modeled delay. Without a new command the plant
retains its last reference. Pause/reset/step and pacing controls remain available;
reset clears input/native history and changes the reset generation. Connection
loss or a2-s heartbeat stall pauses the host with a visible error and requires
restarting the custom simulation. This is desktop pacing, not a hard real-time
radio or firmware-watchdog implementation.

[Flight command arbitration](../deployment/flight_commands.py) supplies functional
arm, takeoff, land, absolute/relative go-to, polynomial upload/start, group masks,
stream release and emergency services under both `/<robot>/` and `/all/`. The ROS
node is named `crazyflie_server`, with the parameter services and robot metadata
needed by stock `CrazyflieServer()`/`Crazyswarm()` discovery. Takeoff/land/go-to
coefficients come from the pinned `cffirmware` high-level planner in Docker.
The host evaluates these XYZ/yaw polynomials at native100-Hz request boundaries,
including analytic P/V/A and nominal desired-body rate feedforward from firmware
flatness (`GRAV=9.81` m/s²). Uploaded trajectories support1..64 eight-coefficient
pieces, piece offsets without gaps, positive timescale, reverse playback and
relative XYZ shifts. This planner layer does not change native physics, the
200-Hz geometric controller,30-ms modeled delay or the motor model.

Streaming has priority until `notifySetpointsStop`; its validity interval begins
at the actual native notification boundary. Separate ROS service/topic clients
can reorder delivery: a high-level request arriving before notification remains
pending without actuation until notification releases it. Pre-notification
stream packets cannot cancel that release. The closing stream's cutoff includes
the20-ms permissible header-clock allowance at notification receipt. Packets
from it arriving after high-level activation are explicitly counted/logged as
discarded; newer streams can regain priority. API receipt/scheduling counts,
pending-release state, generated high-level samples and stream discard counts
remain distinct. Upstream services have empty responses; service failures appear
in `/aerowhip/simulation/status.command_error` and ROS logs. Writable parameter
emulation is limited to functional `commander.enHighLevel` and
`hlCommander.groupmask`, mirrored between robot and `all`; arbitrary firmware
gains/controllers are not emulated. Each ROS parameter transaction is fully
validated before sending one host batch; conflicting robot/`all` aliases are
rejected. The host validates again and commits the batch under one command lock
before acknowledging it, independently of the flight-command queue. Disarm
therefore cannot discard an accepted parameter update. Disabling high-level
control also cancels pending/active plans. A lost API acknowledgement or send
failure ends the bridge session rather than continuing with uncertain state;
reconnect requires restarting the session. These parameters are control metadata,
not changes to physical model parameters.

The separate boolean parameter `aerowhip.acceleration_only` selects the ROS-only
[controller wrapper](../deployment/acceleration_control.py), default false. It
must be set alone. Its owner-thread commit requires disarmed hold, an empty
command queue and matching native requested/applied references (mode transitions
must drain the modeled delay). In this mode an armed plant uses FullState world
acceleration and yaw, ignoring P/V placeholders; body-rate feedforward must be
zero and high-level trajectory requests are rejected. It uses the unchanged
nominal force equation `m*a - (m + sum(cable_node_mass))*gravity`, attitude loop,
motor dynamics, 200-Hz controller and 30-ms modeled command delay. No firmware
gain emulation or identification is implied. Disarmed hold uses PVA control.
Reset clears acceleration mode, and successful scripted completion explicitly
restores PVA. Status/evidence records `acceleration_only` and the selected
`controller_interface`; `reference_position_world_m` is unavailable in direct
mode because the transmitted P is a placeholder. Direct-mode checkpoint exports
use `aerowhip_ros_acceleration_runtime_v1` so generic PVA restore rejects them;
a dedicated continuation importer is not implemented.

The free-space plant holds its configured pose with control active and
API arming enabled. `arm(False)` gates new commands, cancels queued plans and
requests a controlled hold at the latest feedback pose while `/clock` continues.
It does not model motor power removal, falling or touchdown. Emergency immediately
suspends the native owner, clears incoming commands and latches an error; play
and step are blocked until explicit simulation reset/restart. Reset changes the
generation and restores the configured initial state. Other scripts issuing a
low landing height receive a free-space descent, not a ground/contact solution.

In the Crazyswarm checkout, plain `./docker/shell.sh` defaults to custom simulation
on domain142 with `AEROWHIP_SIMULATION=1`. The explicit alias
`./docker/shell.sh --custom-sim [domain]` allows a different simulation domain.
`./docker/shell.sh --real` and `--sim` explicitly select domain42 and
`AEROWHIP_SIMULATION=0` for real flight and stock simulation respectively.
This selection affects newly opened shells and their child scripts; it does not
retarget an existing terminal or change running ROS nodes. The simulation-only
[Crazyflie client patch](../../Crazyflie/crazyswarm2_jazzy_whipping/docker/crazyflie-py-custom-simulation.patch)
enables `use_sim_time`, verifies the custom server/session metadata and exposes
simulation status and configured return height. The existing parameter name
`aerowhip.airborne_landing_height_m` is retained for client compatibility; its
value is the effective initial world Z, including zero or negative coordinates.
Ordinary `TimeHelper` scripts
then follow `/clock`; the real/stock shell does not enable this behavior. The
[CSV repeat player](../../Crazyflie/crazyswarm2_jazzy_whipping/scripts/full_state_pva_csv_repeat.py)
additionally processes ROS clock/status callbacks during simulated playback and
resets each new hover's rate deadline in both real and simulation modes, avoiding
catch-up bursts after the CSV's separate scheduler has advanced time.
Simulation playback checks session/generation, command rejection and landing
reference completion. Its landing height explicitly returns to the configured
initial world Z; this is printed before arming and still represents a free-space
reference rather than touchdown. Emergency waits use wall
time because the plant clock stops. The real-flight wall-time playback scheduler,
landing-height default and hot-loop behavior are preserved.

Feedback uses the same native timestamps: `/<robot>/pose` is `PoseStamped`,
`/<robot>/simulation_state` is `FullState` with world linear velocity/net
acceleration and body angular velocity, `/tf` is world-to-robot, and
`/<robot>/odom` expresses its twist in the robot child/body frame (linear velocity
is `Rᵀ v_world`). `/<robot>/simulation_cable` is a `PoseArray` of simulated node
positions, with identity placeholder orientations; it does not publish measured
cable geometry or material frames. `/aerowhip/simulation/status` carries session,
generation, robot/frame, clock periods, error and command counts, plus API arm/mode
and pending-release state. Drone and cable are labeled simulated in the viewer.
`/<robot>/robot_description` is transient-local URDF, and `/<robot>/status` exposes
simulation API supervisor flags. Battery/radio fields in this required fixed
message remain zero placeholders, not simulated measurements. Hover, legacy
velocity and position streaming, arbitrary firmware parameter changes and
multiple simultaneous plants are not implemented by this adapter.

The separate [PVA test](../deployment/ros_test.py) uses the upstream
`Crazyflie.cmdFullState` serializer through a streaming-only client, without
initializing high-level services or a joystick. It waits for matching simulation
status, clock and feedback, then sends at nominal30 Hz:1-s hold, a6-s `sin⁴`
outward/return excursion of `(0.20,0.10,0.05)` m, and2-s hold. P/V/A are analytic;
yaw and rate feedforward are zero. These are connection-test settings, not a whip
definition/reward, calibrated flight maneuver or policy evaluation. The script
records actual stamps/missed slots and aborts on clock/session reset, stalled
feedback, command rejection or incomplete acceptance/scheduling.

Each host launch creates a new evidence directory under `output/custom_simulation`
(or project-contained `--output`), preserving existing files. `metadata.json`
retains scene/model/session/adapter identities, `connection.jsonl` records
receipts, request boundaries and latest output snapshots, `ros-bridge.log` retains
adapter diagnostics, and `final-checkpoint.npz` stores exact native BDF2/motor/
queue continuation at shutdown with the model in metadata. Output snapshots and
their last-step diagnostics may coalesce when a wall-pacing burst advances more
than one output; command receipt/request records remain separate. They are
connection/numerical evidence, not recordings of real flight or fitted models.
The native checkpoint is not an implemented ROS session/input-FIFO resume path.

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
history. The goal has a separate projection. The Gaussian actor uses three tanh-bounded outputs. In retained `velocity_v1`
recipes these scale to world XYZ velocity, limited to (2,2,0.8) m/s. The default
`acceleration_direct_v1` Stage1 preset scales them to world XYZ acceleration,
limited to (8,8,4) m/s² at50 Hz. Acceleration means the nominal translational
request excluding gravity, not measured acceleration or IMU specific force.
Neither interface enforces workspace containment. The same99-column Stage1–3
layout is retained; the previous-action and privileged delay-queue columns take
the selected interface's units and scales. Acceleration checkpoints carry the
new run schema `ppo_stage1_acceleration_direct_v1` and explicit actor control
metadata; cross-interface actor transfer/resume is rejected. Existing artifacts
retain velocity semantics. Stage4's velocity residual and scripted return, and
velocity demonstration cloning, are explicitly unavailable for acceleration.

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
it. Actor transfer starts with fresh critic, Adam, RNG, and local clocks;
`transfer_entropy_clock` selects inherited exploration progress or a stage-local
restart. Exact resume
retains them and rejects changed execution sources/settings except its explicit
supported budget override.

## Shared policy interface (proposed)

The shared live policy runner and real-flight integration remain proposed.
On2026-10-14 the user approved direct acceleration outputs. That controller/action
profile is now implemented for PPO training and deterministic evaluation, with
separate Stage1–3 acceleration presets, starting with
[Stage1](../configs/learning/whip_stage1_ppo_acceleration_v1.json).
The earlier integrated-PVA proposal is superseded; no reference generator or new
reference-error observations were added. The ROS bridge defaults to PVA and now
has an explicit direct-acceleration mode for the standalone circle test; it has
not been connected to this policy. Existing velocity presets/checkpoints
retain their original behavior and units. No new reward/curriculum, fitted model,
onboard configuration or flight result is established by this change.

### Confirmed interface gap and recommended command path

[Training](../tasks/whip/sac_environment.py) dispatches explicitly by action interface.
Retained velocity recipes scale three normalized actions into world XYZ velocity
and call [the velocity controller](../controllers/full_state_controller/tensor_controller.py).
Its translational acceleration is `Kv * (v_requested - v_tracking)`. The new direct
acceleration path uses the scaled action itself in this force conversion, with
no translation feedback. Both hold initial yaw. The
[interactive native controller](../controllers/full_state_controller/geometric_controller.py)
instead uses `a_ref + Kp * (p_ref - p_tracking) + Kv * (v_ref - v_tracking)`.
The [ROS bridge](../deployment/ros_bridge.py) defaults to this PVA interface;
`aerowhip.acceleration_only=true` explicitly selects the nominal direct path.
Integrating policy velocity into a position reference therefore changes the
closed-loop system; setting position to the latest measured pose also fails to
remove position feedback after transport/controller delay.

The [installed-source audit](../tmp/policy-contract-20261011/source-and-codec-audit.json)
records Crazyswarm commit `fcf51e27fe675bbdb564a40f4ca1ee4336c062ea`, firmware-build
commit `16c5630707e92830dcd16455a57b5560e5838735`, cflib 0.1.28 and inspected file
hashes. The hardware YAML requests `stabilizer.controller=2` (Mellinger), while
the flashed firmware and active onboard parameters remain unverified. The
inspected firmware FullState decoder selects absolute XYZ control. Its Mellinger
translation branch uses P/V/A plus position integrals in that mode; the
non-absolute branch does not implement the training velocity-only servo.
Crazyswarm's real servers expose `cmd_velocity_world`, but the installed Python
client method is commented out and that topic is unsupported by our custom
bridge. A velocity subscription alone would not close the controller gap.

The user approved `acceleration_direct_v1` on2026-10-14: three bounded
world-acceleration requests feed force construction and the existing attitude
controller directly. This supersedes the unimplemented `acceleration_ref_pva_v1`
generator proposal. The direct path computes
`F_des = m_drone * a_requested - (m_drone+m_cable) * g_world`.
It does not add a COM/tracking-offset acceleration correction or cancel dynamic
cable loads. Requested and achieved acceleration are therefore distinct. This is
a nominal simulation controller, not a claim of Crazyflie Mellinger equivalence.

The separate nominal PVA controller computes
`F_des = m_drone * (a_ref + Kp*e_p + Kv*e_v) - (m_drone+m_cable)*g_world`,
then desired attitude, collective thrust and body torque. Static cable-weight
trim is not dynamic cable-load cancellation; motor response and cable loads
mean `a_actual` need not equal `a_ref`. The inspected firmware instead uses its
configured mass multiplying acceleration and gravity, plus force-domain P/V/I
terms. Its gains and mass are not interchangeable with nominal simulation
parameters. Acceleration feedforward is supported by both paths, but this does
not establish controller or force equivalence.

The deeper [Mellinger source/configuration review](../tmp/policy-acceleration-20261011/mellinger-review.json)
retains the force, attitude and command-output source excerpts and YAML hash.
For `FullState` absolute-XYZ mode the firmware computes
`F_des = m_fw*(a_ref-g_world) + Kp_force*e_p + Kd_force*e_v + Ki_force*I_p`,
where the integral is accumulated and clamped. It aligns desired body +Z with
`F_des`, constructs the other axes from desired yaw, and projects desired force
onto the actual body +Z for collective thrust. It uses the commanded quaternion
for yaw in this path; force determines desired roll/pitch. The subsequent
attitude/rate loop outputs legacy commands with thrust scaled by `massThrust`;
these are not the nominal simulator's SI body-torque commands.

The adjacent Crazyswarm `config/crazyflies.yaml` requests `ctrlMel.mass=0.144 kg`,
`massThrust=17700`, `kp_xy=0.4`, `kd_xy=0.3`, `kp_z=2.5`, `kd_z=0.4`,
`ki_z=1.5` and `i_range_z=15`. These are requested controller parameters, not
measured physical mass or verified active onboard values. Unlisted gains may
retain firmware defaults or persistent settings. The inspected source selects
`ATTITUDE_RATE=500 Hz` for Mellinger, whereas the nominal simulation uses 200 Hz
and has no integral loop. Matching command units alone does not close these
controller, timing, load-trim or actuator differences; the audit did not change
firmware/controller defaults.

The implemented training path is **direct acceleration-to-force control**, with
no position/velocity reference feedback. Learned-policy deployment remains
unimplemented; the standalone stock-Mellinger interface experiment below uses a
host PID instead of a learned policy. In
the inspected source, setting all XY/Z
translational P/D/I gains to zero would algebraically retain the acceleration
and gravity force terms in the same `FullState` mode. These gains are exposed as
persistent `ctrlMel` parameters, so new firmware is not necessarily required.
The user authorized a standalone takeoff/circle/landing script on2026-10-15;
its distinct controller configuration and restoration are described below.
This does not establish matching simulation/training or authorize policy flights. Simply filling the
`acc` field of `FullState` does not disable existing feedback, and gain changes
cannot be inferred from a packet. Other routes include legacy attitude/thrust
commands, which are not an SI force-vector interface and are unsupported by
this simulation bridge. No onboard gains or firmware were altered. The user
selected direct
acceleration for training; implementing its real-flight command path still
requires a separate controller/transport change and validation.

#### Standalone cf_2 acceleration-interface circle experiment

The user-requested
[`scripts/acceleration_circle_cf2.py`](../../Crazyflie/crazyswarm2_jazzy_whipping/scripts/acceleration_circle_cf2.py)
is a simulation/real-flight script in the adjacent Crazyswarm checkout.
`AEROWHIP_SIMULATION=1` (the default Docker client shell) selects simulation;
`--simulation` or `--real` explicitly overrides that selection. Normal invocation
attempts flight in the selected mode; `--check` performs read-only ROS preflight and `--dry-run` samples
the reference without ROS. It selects only `cf_2` and uses the shell's ROS domain.
Real mode requires one real Crazyswarm server, one OptiTrack `/poses` publisher, fresh
`cf_2` status, and stock Mellinger (`stabilizer.controller=2`) in the server mirror.
It starts from a stationary, level, disarmed pose assumed to be on the floor.

Default reference: rise1.3 m relative to that tracking-origin height over6 s,
hover3 s, execute one0.2-m-radius horizontal circle over16 s, hover2 s, descend
in6 s to1.5 cm below the starting height, then allow up to3 s for settling.
The circle center is0.2 m along world +X from the initial pose, so the circle
extends0.4 m in +X; a quintic angle schedule starts/stops at rest. Initial yaw
is held. Landing acceptance requires proximity to the starting position and low
measured speed for0.5 s; this is a pose-based cutoff heuristic, not a contact sensor.

The host runs `a = a_ref + Kp*(p_ref-p) + Kd*(v_ref-v) + Ki*integral` targeting50 Hz
using OptiTrack positions and a causal filtered velocity in real mode, or the
native tracking-origin position/velocity in simulation mode. Gains are4 s^-2,
3 s^-1 and0.6 s^-3 on each axis, with integral contribution limited to1.5 m/s^2;
integration freezes on acceleration saturation. Horizontal acceleration magnitude
is limited to2 m/s^2 and vertical acceleration to[-3,3] m/s^2. These are uncalibrated
experiment settings, not fitted dynamics or validated performance limits.

In real mode, before arming, the script saves the server-mirrored gains, zeros all XY/Z
translation P/D/I gains and both translation integral ranges, and leaves mass,
thrust calibration and attitude gains unchanged. Every motion phase uses
`/cf_2/cmd_full_state` with only acceleration changing, initial yaw, fixed initial
position placeholders, zero velocity and zero body rates. Acceleration is world
m/s^2 excluding gravity; onboard Mellinger adds gravity and constructs attitude
and thrust. There are no high-level takeoff/goTo/land requests. This is a host
trajectory controller testing an acceleration interface, not an acceleration-only
learned policy or an open-loop trajectory guarantee.

Simulation mode verifies the custom server identity, session, reset generation,
world frame, simulation clock and feedback before sending commands. It starts
from idle PVA hold and waits up to 10 wall seconds for the same stationary/level
limits; this does not establish that the cable has settled. It disarms, drains
the modeled delay, selects `aerowhip.acceleration_only=true` and arms. Motion uses
simulation time targeting 50 Hz, with no catch-up bursts; actual request timing
is retained in evidence. ROS service/clock-progress waits use bounded wall time.
The final stream must be consumed before disarm so topic/service reordering
cannot deliver a late packet after disarm. Successful completion restores PVA,
allowing an ordinary waypoint script in the same session. Pause, stale feedback,
clock stalls, reset/session changes or flight-bound failures stop the simulation;
Reset/Play or a restart is then required. This is a free-space return to the
starting height with the unchanged settling limits, not physical touchdown.
Residual cable motion can fail the settling check even when the drone appeared
stationary at entry. Nominal cable-weight trim and the SI wrench/motor model
remain distinct from stock Mellinger firmware; this test does not validate real
flight or deploy a learned policy.

In real mode, fresh onboard status confirms arm/disarm. Pose loss over0.15 s, status loss over
2.5 s, command-loop delays over0.12 s, graph changes, or configured motion bounds
cause a selected-robot emergency-stop/disarm request, not a recovery landing.
Ctrl+C/SIGTERM does the same. Changed parameters are restored after confirmed
disarm, or after a setup failure before any arm attempt. If disarm is unconfirmed,
restoration is withheld to avoid re-enabling feedback with placeholder setpoints.
SIGKILL, power loss and radio loss cannot guarantee cleanup. Per-run original
parameters, requested/measured motion, sent acceleration and cleanup outcome are
saved under the Crazyswarm `logs/acceleration-circle-cf2-*` folder.

The installed cflib ROS server acknowledges parameter writes when queued and
serves cached parameter reads. The script verifies these server mirrors and waits
before arming; this does **not** independently verify firmware application of the
zero gains or restoration. The actual flashed firmware, gain application, mass/
thrust calibration, contact behavior and flight remain unverified. No changes
were made to the coworker's force controller, radio server, YAML defaults,
training rewards or physical model.

Current `velocity_v1` checkpoints must not be relabeled as acceleration policies.
The implemented profile changes action units and controller semantics, requiring
compatible new training. Cross-interface actor initialization and exact resume
are rejected even though both networks have three outputs. Same-interface
transfers retain the existing source/compatibility checks. The selected bounds
are nominal experiment settings, not identified hardware limits; firmware
equivalence remains unverified.

### State and observation contract

The proposed source-neutral state envelope identifies robot, world frame,
tracked body origin, body-axis calibration, model/topology, session/reset
generation, sample sequence, source timestamp, receipt timestamp and aligned
control time. Fields carry validity and provenance independently; a measured
pose does not make a derived velocity or simulated cable measured.
The real viewer's prescribed-motion cable simulation is now implemented, but
policy-state/history assembly, measurement-corrected cable observation and
real-flight policy execution described here remain proposed and unvalidated.

| State field | Meaning and units | Simulation flight | Proposed real-flight source |
|---|---|---|---|
| Tracking position `p` | World XYZ, m, calibrated tracked top-plate origin | Native simulated state | Calibrated motion capture or an explicitly selected estimator |
| Orientation `q`, rotation `R` | Unit XYZW quaternion; body-to-world rotation | Native simulated state | Calibrated rigid-body orientation or onboard estimate |
| Tracking velocity `v` | Velocity of that same tracked origin, world m/s | Native simulated state | Timestamped estimator or causal derivative, labeled estimated |
| Angular velocity `omega` | Physical body XYZ, rad/s | Native simulated state | Calibrated gyro/estimator, with explicit unit/frame conversion |
| Cable `q_i`, `v_i` | Node/marker position and velocity, world m and m/s; ordered topology | Native simulated cable | Causal cable observer, labeled estimated |
| Net acceleration | World `dv/dt`, m/s²; optional diagnostic | Native endpoint diagnostic | Explicitly estimated; not raw IMU specific force |
| Issued reference and action | Requested acceleration, fixed yaw, normalized actor request and timing | Known command history | Proposed known command history, not claimed onboard applied state |

The topology and attachment transform come from the selected model/calibration;
the current 11-node cable is one supported model, not a global interface constant.
For offset `r_attachment`, attachment kinematics are
`p_attachment = p + R*r_attachment` and
`v_attachment = v + R*(omega cross r_attachment)`. COM, tracking origin and
motion-capture rigid-body origin must not be interchanged implicitly. Cable
observer dynamics are driven by measured/estimated attachment motion while
preserving cable history; repeatedly resetting the cable or overwriting a
two-way coupled simulated drone with measurements is not this observer design.

The actor adapter must reproduce its registered column order and scales:
tracked displacement `(p-p_episode_start)/L`, row-major `R`, tracking velocity
`v/(4 m/s)`, body rate `omega/(4 rad/s)`, attachment-relative cable position
`(q_i-q_0)/L` and velocity `(v_i-v_0)/(4 m/s)`, previous normalized actor action,
and checkpoint-defined normalized episode time. In retained velocity training the previous
action means normalized velocity; in direct-acceleration training it means
normalized acceleration.
Its static goal remains
`(target_world-p_episode_start)/L` in world axes. This is not a yaw-rotated goal
or current drone-to-target error. Motor/wrench, BDF2, delay-queue and material
state remain privileged/model diagnostics; they are not required actor inputs.
Absent required cable/velocity/history data cannot be replaced with zero fields.

Existing Stage1-3 frames are 99 columns and Stage4 frames 101 for the current
mesh, with motor/wrench columns excluded by the registered actor selection.
Direct acceleration keeps the existing99-column Stage1–3 actor layout, with no
integrated reference or extra reference-error fields. Its action units, limits
and policy rate are explicit in the checkpoint interface. Stage4's velocity
recovery architecture has not been adapted. Topology, selected columns, scales,
task clock, goal/phase fields, device and actor batch shape remain registered.

[History encoding](../training/sac.py) is causal and shape-dependent. Preserve
the checkpoint's sample spacing, history length, prefix mask and episode index;
future-frame masking does not authorize filling missing past samples. Stage4's
451-sample rolling history and phase logic remain checkpoint/task-specific. A
simulation-feedback source must deliver coherent 100-Hz samples before this
encoder is used; the current latest-frame mailbox can coalesce output steps and
is not sufficient evidence of an uninterrupted actor history.

### Implemented training action and timing contract

The default direct-acceleration preset uses50-Hz actions and100-Hz observations.
A finite three-vector `u` in `[-1,1]` gives
`a_requested = (8,8,4) * u` in world axes, in m/s² excluding gravity. These initial
bounds equal the retained velocity controller's zero-speed full-command requests:
`(4,4,5) s^-1 * (2,2,0.8) m/s`. This does not make the two controllers equivalent.
The downward bound must keep desired vertical force positive, preventing a
zero-force/heading singularity anywhere in the action box. No jerk, speed or
position clamp was added. Zero action does not brake an already moving drone.

Fixed initial yaw accompanies each acceleration command. Commands are held for
20 ms; the existing30-ms delay applies once on the100-Hz observation grid. The
attitude controller runs at200 Hz with zero desired body-rate feedforward.
Per-motor allocation/clipping and first-order response remain unchanged. Both
native evaluation and CUDA collection call the same selected tensor controller.
No translation-reference integrator, position feedback or velocity feedback is
present in the direct path. These clocks and limits are nominal simulation
settings, not measured firmware timing or identified hardware capabilities.

The default Stage1 preset copies the previous compact v5 task, reward, budget
and learner settings; only action interface, limits and policy rate change.
No training was started by this interface implementation. Stage2/3 launch presets
likewise copy their previous cone25 v4 tasks/rewards, changing only the control
interface, limits and rate; they require compatible acceleration actors. Stage4
and velocity demonstrations are blocked for acceleration
rather than reinterpreted. This change does not approve a new curriculum.

Saved acceleration replays identify action interface, command units, limits and
policy rate. `command` and `applied_command` contain requested/delayed acceleration
plus yaw; motor thrust and requested wrench retain their existing units. Physical
acceleration remains a separately derived diagnostic. Actor observations keep
100-Hz causal history; the action/critic delay queue is normalized by acceleration
limits. Checkpoint schemas and actor manifests prevent three velocity outputs from
being silently relabeled as acceleration.

Live policy ownership, inference deadlines, ROS/radio timing and a direct-force
firmware command path remain unimplemented. The bridge's default PVA behavior
is unchanged; its explicit acceleration selector applies only to custom simulation.
Sending acceleration in real `FullState` does not disable onboard P/V/I feedback. Transport and firmware equivalence require separate verification.

The audit's captured cflib FullState packet confirms signed-int16 millimetre,
mm/s, mm/s² and milliradian/s fields, plus compressed quaternion. The rate
packer and firmware decoder use rad/s despite the installed cflib docstring
listing degrees/s; no degrees conversion belongs in this FullState path.
Nonfinite or unrepresentable encoded fields must fail rather than wrap or be
silently clamped. The current ROS/native bridge does not emulate radio
quantization or quaternion compression. A future transport profile must retain
backend/codec identity and compare commands after the actual encode/decode;
nominal unquantized PVA and effective firmware reference are distinct evidence.
The no-radio probe checked scalar packing/range rejection, not quaternion
decoding, onboard timing or controller behavior.

### Lifecycle and staged validation

YAML startup/Reset still initializes the selected simulation robot exactly.
Policy activation is a separate explicit episode boundary after checking the
checkpoint's required initial domain, cable state and history availability.
An arbitrary YAML pose, including `z=0`, does not make a fixed-start actor
compatible. No flight start, takeoff, floor contact or trained-start tolerance
is inferred from this contract. Task/reward/phase logic stays outside physics
and retains the existing research definitions.

Session/reset changes must clear actor history and command state together. Invalid
frames, missing required data, stale input beyond configured limits, inference
overruns and transport rejection must be recorded and stop policy ownership;
continuation uses an explicitly chosen mode-specific supervisor action. The
simulation's pause/hold/emergency behavior is not a physical landing or real
emergency-stop model. Freshness/timeout parameters require explicit selection;
none is presented here as a calibrated flight limit.

Remaining live integration should proceed through separate evidence gates:
common encoder reconstruction; direct-acceleration controller/codec checks; one
compatible policy through the ROS simulation compared with direct execution at
recorded command boundaries; new-data drone/controller/actuator identification
and held-out flight comparison; then a causal cable observer compared against
simulated truth and newly recorded cable measurements. Live policy input must
use the observer's estimate in observer evaluation. Numerical equality,
transport success, model fitting, policy performance and real-flight results
remain separate. No scientific acceptance threshold or model promotion is
established by this design document.

## Four stages and scientific criteria

The completed linear outreach/height campaign used these retained experimental
settings (the new crossing-priority Stage2 restart is described separately below):

| Stage | Learning preset | Budget and initialization |
|---|---|---|
| 1 | [outreach/height radius055 v12](../configs/learning/whip_stage1_ppo_outreach_height_radius055_150_v12.json) | 150 new collections; paused collection0 actor (identical to original collection121 actor), fresh critic/Adam/clocks, seed0 |
| 2 | [delivery radius055 v9](../configs/learning/whip_stage2_ppo_outreach_speed_radius055_150_v9.json) | 150 collections; final new Stage1 actor, fresh critic, seed1 |
| 3 | [precision radius055 v8](../configs/learning/whip_stage3_ppo_outreach_speed_radius055_150_v8.json) | 150 collections; final new Stage2 actor, fresh critic, seed1 |
| 4 | [recovery radius055 v6](../configs/learning/whip_stage4_ppo_outreach_speed_radius055_80_v6.json) | Prepared80-collection recipe; not launched by this restart |

Stages1–3 each completed19,200 new episodes at128 lanes in the retained campaign;
Stage3 did not reach its final target level. Their task presets are
[Stage1 v8](../configs/tasks/whip_stage1_outreach_height_radius055_v8.json),
[Stage2 v4](../configs/tasks/whip_stage2_outreach_speed_radius055_v4.json), and
[Stage3 v4](../configs/tasks/whip_stage3_outreach_speed_radius055_v4.json).
[Stage4 v4](../configs/tasks/whip_stage4_outreach_speed_radius055_v4.json) retains
learned recovery. Stage1 v8 is the newly approved experiment; other stage recipes
retain their existing rewards. These do not replace launcher defaults. Bare `run_training.py`/`--stage 1`
selects direct-acceleration Stage1 v1 with the same compact Stage1 v5 task/reward;
`--stage 2/3` select acceleration variants with their previous cone25 v4 tasks
and budgets. Stage4 remains velocity-only. The scratch acceleration Stage1 run
completed its budget but failed to learn; acceleration Stages2–4 have not run.
Select the active experiment with `--settings` or use its opened Training window.
Older compact, level-outreach,40/80/160-collection and0.40 m presets remain
readable configuration variants, but their deleted runs are not active models.

The current nominal start is world(0,0,2) m and target center(1.1,0,2) m.
Stages1–3 execute the full4.5 s horizon. The requested horizontal direction is
world+X. The virtual target rectangle has no collision response.
Automatic evaluation uses one native physical lane, preserving CUDA actor batch
shape64: after the first collection, every512 training episodes, and at the end.
These evaluations are one nominal case, not independent robustness trials.

Mechanics proxy M combines bend transport/coherence, non-rigid release and
bounded forward motion over0.2/0.4/0.6/0.8 s windows,0.05 s stride and0.15 s tip
lag. U measures recovered chord/arc slack; D is squared positive forward
alignment with a plateau within25 degrees. Raw W=M*U*D remains a diagnostic.
These are approved task proxies, not an established physical whip classifier.
For rest cable length L=0.9525 m, the retained Gaussian-height event credit B
(earlier Stage1 speed presets and the unchanged Stage2–4 presets) is

`B(e) = [MU/(MU+0.05)] * D * r * H * [v_positive/(v_positive+3 m/s)]^2 * exp(-0.15*l/L)`.

Here `r=clip(dot(tip_world-attachment_world,requested_direction)/L,0,1)`,
`H=exp[-0.5*((tip_z-drone_z)/(0.35*L))^2]`, and `v_positive` is positive forward
tip velocity in m/s. The height width is0.333375 m; drone_z refers to the tracked
top plate at the same release, not the cable attachment or initial height.
The path l is actual tracked-drone travel in meters from episode start through
that observed release, calculated from100 Hz tracked-drone positions. All terms
share one complete observed event; future crossings or recovery states do not
rewrite earlier credit. Reach/height/speed alone cannot earn credit with MU=0.
The speed scale and height preference are research weights, not success thresholds.
`outreach_credit-preparation_cost` reconstructs B; raw `mechanics_credit` is W.

The completed linear outreach/height Stage1 experiment
(`directed_outreach_height_whip_v1`) instead selects
`max_e R(e)`, where

`K(e) = [MU/(MU+0.05)] * D * [v_positive/(v_positive+3 m/s)]^2 * exp(-0.15*l/L)`

`h(e) = clip(1 - abs(tip_z-drone_z)/L, 0, 1)`

`R(e) = K(e) * [0.5*r(e) + 0.5*h(e) + 2*r(e)*h(e)]`.

This linear height preference uses tracked-drone Z at the same observed release;
it is not a new physical success tolerance. The existing mechanics/unfolding,
alignment, forward-speed and actual-travel factors remain unchanged. All three
contributions share one complete event; separate maxima cannot be combined.
The new explicit weight fields are `outreach_reach_weight=0.5`,
`outreach_height_weight=0.5` and `outreach_joint_weight=2`. No Gaussian height
width is accepted in this mode. Diagnostic `level_height_factor` now means h,
and `level_outreach_quality` means r*h for this mode only. The three diagnostic
`outreach_*_credit` components are measured before path reduction and sum to
`outreach_credit`; subtract `preparation_cost` once to reconstruct R. They are
shown in Training alongside their total. Raw W remains a mechanics diagnostic.
Zero mechanics/unfolding or zero positive forward speed still earns zero credit.
The older Stage1 (`directed_outreach_speed_whip_v1`) selects `max_e B(e)`.
The retained Stage2 (`directed_outreach_speed_delivery_v1`) selects `max_e B(e)*(1+P(e))`.
P is the Gaussian of Euclidean distance from that same observed tip release to
the current target rectangle, with width0.35 m. Stage2's target is fixed1.0×0.6 m.
`approach_reward=B*P`, including the same preparation discount.
Stage3 (`directed_outreach_speed_precision_v1`) selects
`max_e B(e)*(1+P(e)+C(e))`, with
`C=exp[-||tip(e)-target_center||^2/(2*sigma^2)]` at the same world-coordinate
release. `center_reward=B*C`. Stage3 narrows its rectangle
1.0×0.6 →0.6×0.4 →0.3×0.2 →0.1×0.1 m with sigma0.25/0.15/0.075/0.05 m.
Current-target stochastic scores and final-target native scores are distinct.
`crossing_center_quality` is separately measured on the forward-plane interval;
it is zero without a crossing in that interval. Full-episode actual crossings
are measured independently of event selection.

The approved Stage2 restart uses
[crossing-priority learning v10](../configs/learning/whip_stage2_ppo_crossing_priority_radius055_150_v10.json)
and [task v5](../configs/tasks/whip_stage2_crossing_priority_radius055_v5.json).
Mode `crossing_priority_delivery_v1` defines positive credit

`0.2W + max_actual_forward_crossing [C + 2G + S]`,

where W is the approved linear Stage1 best-event R above divided by3, and
`C=1/[1+(lateral_error/0.25 m)^2+(vertical_error/0.15 m)^2]`. Errors are relative
to the target center at the interpolated actual crossing. G is the existing
geometric-hit indicator; S is its existing strict-qualified counterpart
(forward speed≥3 m/s, entry angle≤30 degrees). All three crossing contributions
come from one crossing; separate component maxima cannot be combined. No
crossing earns zero crossing credit. W and the crossing record are deliberately
independent and can occur at different times; crossing reward does not require
an eligible mechanics window and is not evidence of physical whip behavior.
The radius0.55 m, full4.5-s horizon, fixed1.0×0.6-m target centered at
(1.1,0,2) m, invalid-episode replacement and outside penalty are retained.
Before workspace costs, positive credit is bounded by4.2; a geometric miss
is bounded by1.2, a geometric hit earns at least2 and a strict hit at least3.
Workspace penalties can reverse net-return ordering.

The normalized whip support and the crossing each maintain a causal record.
Crossing credit becomes available at its observed interval endpoint, never
before that observation. PPO differences telescope to the same episode return;
a later invalid episode cancels positive credit with the existing failure rule.
`whip_support_reward`, `crossing_center_reward`, `crossing_geometric_bonus` and
`crossing_strict_bonus` reconstruct the positive reward. `crossing_reward` is
the total crossing credit; `crossing_reward_*` timing/errors describe its
selected crossing. Existing outreach/path components and event indices still
describe the unscaled linear Stage1 record. `target_proximity` means C at the
reward-selected crossing in this mode; whole-episode crossing diagnostics are
retained separately. Training plots show the actual four contributions plus
workspace cost. The new source is the completed Stage2
[latest actor](../runs/ppo_stage2/20261011-063417-108510/latest.pt), not a random
policy or an exact learner resume. Budget is150 new collections/19,200 episodes
at128 lanes; critic, Adam and stage-local exploration restart. Saved-array
review follows completion before any Stage3 decision. Existing presets/runs
and the Stage3/4 criteria remain preserved; no new Stage3 ladder is implemented.

For the older event-only modes, causal credit is the best complete event
available so far; repeated events do not accumulate credit. The new Stage1 event bound is3; its older Gaussian
variant has bound1. The unchanged Stage2/3 event bounds are2/3. Raw returns
across these different reward definitions are not directly comparable.
A separate outside-only squared-excess penalty uses weight0.1 and distance
scale0.1 m. The tracked drone's cylinder has radius0.55 m about initial XY and
world-Z bounds1.5–2.6 m. The cable is not constrained to the cylinder. This
boundary is soft: no command projection/filter or excursion termination is
introduced. Inside travel affects the existing event path factor, not an added
workspace charge. Acceleration cost remains zero.
Tilt above60 degrees, cable below0.05 m, nonfinite state or numerical failure
invalidates an episode; invalid event credit is replaced by failure return−1
while accrued workspace cost remains charged.

Geometric forward-plane crossings inside the current rectangle and strict
crossing diagnostics are separate. Strict crossing requires valid execution,
forward speed≥3 m/s and entry angle≤30 degrees. Final Stage3/4 geometry is
0.1×0.1 m, inclusive±0.05 m on both target-plane axes. The25-degree reward
plateau does not redefine qualification. Stage3 advancement requires all lanes
valid, at least80% geometric hits for three consecutive collections and an
updated-policy nominal geometric hit. Workspace is measured but does not gate
this curriculum. No multi-target, random-start or robustness learner is active.

Stage4 freezes the entire preceding Stage3 whip and trains only the internal PD
residual recovery head/logstd. It switches at the first causally observed
geometric hit or4.5 s miss timeout and then recovers toward episode-start drone
position for4 s. Recovery weight0.5 uses existing return/cable-settling terms;
a miss earns no recovery credit. Internal position gain is1.5 inverse seconds
and velocity damping0.4. The final1 s hold diagnostics require drone error≤0.1 m,
speed≤0.15 m/s, attachment-relative cable RMS speed≤0.2 m/s and horizontal RMS
extent≤0.1 m. Its positive bound is3.5. The existing campaign handoff requires
Stage3's final target level plus a valid, contained final native strict hit.
Failure is reported without bypassing a gate or relabeling a baseline as trained.
This restart evaluates that handoff after Stage3 without training Stage4.

Stage1's fresh entropy schedule is0.02→0.001; Stage2/3 use0.005→0.001. Hold
fraction0.1 and anneal-end fraction0.8 apply to each new19,200-episode budget.
Plateau detection keeps its thresholds and warns; it does not stop the approved
full budget. These are fresh runs, not exact resumes. Actor-only Stage2
initialization permits Stage1 or Stage2 sources; Stage3 permits Stage2 or
precision Stage3 sources. Role, identity, finite-weight, model/scene/control and
physical-source compatibility checks remain intact. Exact resume may change
only the total episode budget and retains the saved learner/exploration state.

[motion_search.py](../evaluation/motion_search.py),
[motion_refinement.py](../evaluation/motion_refinement.py), and
[demonstration_search.py](../evaluation/demonstration_search.py) remain available
for bounded simulated-motion diagnostics; [demonstration.py](../training/demonstration.py)
can clone simulated causal state/action sequences into an actor. These tools do
not alter plant physics or automatically install a policy. Their old datasets
and original searched clone were deleted with `runs/`. A replay of that clone's
closed-loop motion survived separately in
[the retained baseline](../tmp/stages1to4-80-20261010/stage1-seed-interface-baseline.npz).
The approved [reconstruction and cloning script](../tmp/stage1-replay-demo-150-20261010/build_seed.py)
recovered45 requested velocity actions at10 Hz from recorded samples1+10*k,
normalized by world-XYZ limits(2,2,0.8) m/s. Reintegration through the current
native plant reproduced tracked-drone position/velocity/rotation and cable
position/velocity exactly, remained valid for4.5 s, and was contained in the
current0.55 m workspace. The [new teacher dataset](../runs/motion_search/20261010-replay-derived-demo-radius055-v1/demonstration.npz)
contains causal observation frames, action targets and encoder/interface
metadata; it represents the earlier closed-loop clone, not the original search
winner or a handheld recording. Its [reconstruction record](../runs/motion_search/20261010-replay-derived-demo-radius055-v1/assessment.json)
retains replay, source-provenance and deleted-original-checkpoint hashes.

The fresh seed0 actor was fitted with the existing supervised bounded-command
MSE procedure for1,200 Adam steps at0.0003 learning rate, freezing scale-head
weights and selecting the lowest teacher-state error. There is no imitation
reward bonus. The [new seed checkpoint](../runs/motion_search/20261010-replay-derived-demo-radius055-v1/cloning/actor.pt)
has zero PPO counters and fresh critic/optimizers/clocks. Its
[native closed-loop replay](../runs/motion_search/20261010-replay-derived-demo-radius055-v1/cloning/cloned-actor.npz)
is separately evaluated from the supervised fit. The
[cloning assessment](../runs/motion_search/20261010-replay-derived-demo-radius055-v1/cloning/assessment.json)
reports teacher-state MSE9.36546e-5, nominal raw return0.0151354, selected-event
reach0.639620 m, forward tip speed5.01312 m/s and tip-to-drone height error
-0.737543 m, with full-horizon validity and no workspace breach. This is a useful
initial stroke, not attainment of near-level outreach or target/strict crossing
success. One nominal evaluation does not establish robustness or flight validity.
Handheld retargeting remains a proposal described under recordings below.

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
lists the earlier milestone differences, not approval of later source changes.
The subsequent containment/UI edits changed physically guarded Python files
(`simulation/model_configuration.py` and full-state controller UI/launcher paths),
without changing equations, parameters, rewards, thresholds, or timesteps.
The existing physical-source and binary-hash guards therefore reject fresh plant
reevaluation/actor transfer from the packaged historical checkpoints. Policy
Tester still loads them for inspection but disables incompatible fresh tests.
Saved replay and CPU synthetic inference work. Rebuilding native binaries on
Linux requires separate compatibility validation; these guards were not loosened.
Exact historical resume/reproduction is not claimed.
No checkpoint is relabeled as trained with current observations or the new
Stage1 compact reward. These retained models predate that reward change, and no
new trained model has been promoted. Evidence
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

Streaming Crazyflie command/feedback integration with the native simulation is
implemented on Linux as described above. Real-flight integration, cable-state
estimation from drone telemetry, firmware/controller equivalence, hardware
parameter identification, arbitrary starts, moving targets, and real-flight
safety remain unimplemented or unverified. No model default or physical fit was
promoted during cleanup or this connection work.

## Recordings and fitting

The adjacent
[motor/PVA logger](../../Crazyflie/crazyswarm2_jazzy_whipping/scripts/experiment_logger_optitrack_pva_motor_v3.py)
now subscribes to raw `motion_capture_tracking_interfaces/NamedPoseArray` on
`/poses` (`--poses-topic` overrides it), selecting the exact `--drone` name.
The existing script filename is retained, but new recordings use schema
`optitrack_poses_pva_motor_v4`: `pose_stamp_ns`, `pose_time_s` and `pose_dt_s`
replace the old `tf_*` columns. Metadata records source topic, pose name, parent
frame and source-clock limitations. No coordinate conversion is applied.
`--parent-frame` can require a particular frame; otherwise the first matching
parent is locked. Former `--tf-topic`/`--mocap-frame` options fail with migration
instructions because TF visualization may replace missing orientation with an
identity quaternion. The logger retains invalid raw quaternion samples with
`pose_valid=0` and invalidates finite-difference velocity across them and tracking
gaps; a legitimate measured identity quaternion remains valid. Missing named
poses do not produce invented samples. Command and motor sidecars retain their
existing fields, clocks and validity rules. Existing recordings are untouched;
this change does not migrate old schemas or establish clock synchronization.

The adjacent `force_controller.py`, `force_controller_cal.py` and
`force_controller_v1.py` mark their saved gain state dirty before the first
parameter write, so partial write failures enter the existing restoration path.
The v1 exception handler also attempts emergency even if restoration fails.
This is best-effort cleanup: asynchronous parameter writes and physical gain
restoration still require hardware verification. No force definition, gain
default, calibration formula or learned-policy interface changed.

Original source recordings remain external and untouched. For active use, copy
the collection inside `data/recordings/` with its original `data/flight_batches`
and linked rehearsal layout. Take Processing selects only project-contained
collections; it no longer infers an external root from historical catalog paths.
The prior personal catalog was removed from tracked files during the earlier
containment cleanup. Its former project-run backup is no longer available.
New catalogs and local preferences are ignored by Git. A clone
has an empty recording folder and needs user-supplied data for fitting.
Fitting checks catalog/alignment/source
hashes and complete 100 Hz windows. Missing observations are masked, not invented.
Causal history initializes velocities; twist and initial motors remain assumptions
where telemetry is absent. Command-driven and measured-boundary replay answer
different model-validation questions.

Handheld OptiTrack demonstrations are a proposed next source of motion evidence;
no such recording is currently available in the project. The recommended pilot
uses the same marked cable with a rigid tracked handle and a mounting boundary
matching the drone clamp. Capture handle position/orientation and all ten cable
markers, retain original TAK/CSV files at native capture rate, and record the
attachment offset and mounting direction in the handle's local frame. A brief
stationary lead-in/out and several complete preparation/drive/pullback attempts
would support tracking checks and motion comparison. Use meters, world
coordinates, quaternion rotations, explicit world-Z-up convention and CSV
headers; export physical marker observations rather than substituting solved
constraint positions for occluded cable markers. These controls are described
in [OptiTrack's CSV export documentation](https://docs.optitrack.com/motive/data-export/data-export-csv).

This is a capture/retargeting proposal, not an implemented imitation pipeline.
[load_take](../take_processing/take_data.py) currently expects a drone identity
and ten named cable markers; existing fitting preparation additionally requires
verified flight-command clock alignment and contiguous native 100 Hz data, with
no resampling. A hand take should not be relabeled as drone flight or supplied
invented command logs. It provides observed motion, not drone action labels.
The proposed sequence is measured-boundary cable replay, then a search for
feasible bounded drone commands and timing, then imitation from those simulated
state/action trajectories. The existing prescribed-cable engine can represent
attachment P/V/A and mounting rotation/angular rate, but hand-specific import,
fixture calibration, derived-data resampling if needed, retargeting objectives
and imitation remain unimplemented. Wrist motion and drone actuation may differ;
physical parameters, rewards and curriculum are not changed automatically to
match a demonstration.

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
SDK; non-Windows builds use the supplied strict floating-point compiler flags.
Existing Linux native/tensor binaries and local `.venv` remain unchanged.
The current suite checks configuration, rewards/causality, numerical behavior,
actor/control contracts, checkpoint restoration, recovery and UI provenance.
It does not establish learned-policy or real-flight success. Previous validation
logs under `tmp/` remain local historical checks; this cleanup does not imply
a full suite, model fit, simulation or training experiment was rerun.

2026-10-10 approved clean Stage1 restart: the previous restored-radius Stage2
run was safely stopped after five complete collections/640 episodes; Stage3
was not launched. The old campaign terminated on `user_stop`. The user explicitly
approved deleting all old project runs,150 Stage1 collections and a new random
policy. The [cleanup manifest](../tmp/stage1-clean-restart-20261010/cleanup-manifest.json)
records2,169 deleted files/1,095,648,064 bytes, including old PPO runs, motion
searches, cloned demonstration and reward-review evidence. The old Training
windows were closed after their workers stopped. `runs/` was verified empty
before the new experiment. All108 checked `models/`, `data/` and existing
configuration files remained unchanged. The maintained record was shortened to
replace deleted-run references and describe the active recipe; a prior text
snapshot is retained in the cleanup folder. Original external recordings were
not part of the deletion. The restart changes only the Stage1 budget from its
prepared80-collection counterpart to150, and uses fresh actor/value initialization;
rewards, physics, workspace, scientific criteria and source code are unchanged.
No old run outcome is promoted or presented as retained flight evidence.
Preparation, launch and new experiment outcomes are recorded below.

2026-10-10 clean-restart preparation:
the [preflight](../tmp/stage1-clean-restart-20261010/preflight.json) loaded all
three150-collection/128-lane/0.55 m configurations and exercised the actual
Stage1 create-run path with `resume=null` and `initialize_actor=null`. No initial
actor payload is present. CUDA is available; no physical rollout was performed
by this check. The Stage1 settings differ from the prepared80-collection recipe
only in total episode budget. All108 pre-cleanup protected files stayed exact;
the new campaign pins109 input files and71 execution hashes.
The one-off [campaign driver](../tmp/stage1-clean-restart-20261010/campaign.py)
starts fresh Stage1 in the Training UI and retains an accounting/native review
after each completed stage. It transfers each final actor into the next stage
with a fresh critic/Adam/exploration clock, records native transfer checks and
stops the sequence if the user stops a run. After Stage3 it records the existing
Stage4 handoff assessment without starting Stage4 training. The retained-array
[reviewer](../tmp/stage1-clean-restart-20261010/review.py) distinguishes fresh
Stage1 initialization from later actor transfers and preserves new logs and
source/ownership references. This is preparation, not a completed policy result.


2026-10-10 Clean Stage1–3 150-collection restart: Stage1 launched in [20261010-215414-837444](../runs/ppo_stage1/20261010-215414-837444); its subsequent status and review are recorded below.


2026-10-10 fresh Stage1 launch verification:
the [live initial checkpoint](../tmp/stage1-clean-restart-20261010/initial-live-check.json)
confirmed counters0, no actor payload/provenance, empty actor/value Adam states,
zero exploration offsets, seed0 and19,200-episode entropy budget. The active
[Stage1 run](../runs/ppo_stage1/20261010-215414-837444/) completed its first new
collections and one nominal native evaluation. Its
[interim saved-array review](../runs/ppo_stage1/20261010-215414-837444/interim-review.json)
checked256 stochastic episodes for return/workspace/transition accounting and
valid native event scoring/crossing diagnostics. Maximum float32 return
reconstruction error was0 and transition-sum error4.77e-7. This is a launch
check, not evidence of learned success or a completed150-collection result.
The [cleanup/ownership preservation check](../tmp/stage1-clean-restart-20261010/launch-checks.json)
verified all2,169 old run files absent, exactly one new Stage1 run, an active
campaign/UI/worker and all109 pinned input files/71 execution hashes unchanged.
This describes the initial random-policy launch, now stopped. The subsequent
approved demonstration-seeded replacement is recorded below.

2026-10-10 approved replay-demonstration restart preparation:
the [stopped random run](../runs/ppo_stage1/20261010-215414-837444/summary.json)
ended at44 complete collections/5,632 episodes, with its logs and checkpoints
preserved; its driver terminated without launching Stage2. Collection40's
native evaluation was valid with no numerical failure, selected reach0.042924 m
and forward tip speed0.052959 m/s; its base event credit was8.38948e-10, before
workspace cost. The saved evaluations suggest an exploration/optimization
problem; they do not prove an eventual random-policy run could never learn.
The [approved request](../tmp/stage1-replay-demo-150-20261010/restart-request.json)
covers reconstruction, fresh actor cloning, native verification and the new
150-collection Stage1/2/3 chain, with unchanged reward, physics and workspace.
The [seed assessment](../tmp/stage1-replay-demo-150-20261010/seed-assessment.json)
separates exact teacher reconstruction from the new cloned actor's nominal
execution and confirms unchanged critic weights, empty PPO optimizers and zero
learner/exploration counters after supervised initialization. Original protected
inputs and stopped-run files are pinned by hash. No PPO result or model promotion
is claimed by these seed checks. Training launch verification follows below.

2026-10-10 replay-demonstration restart launch verification:
the [actor-transfer preflight](../tmp/stage1-replay-demo-150-20261010/preflight.json)
exercised the actual create-run path with an immutable actor payload and
`resume=null`. Loading the seed into a fresh PPO learner left critic, RNG,
optimizers and counters unchanged and reproduced its five retained physical
state arrays exactly in a native rollout. All three settings loaded as
150 collections/19,200 episodes at128 lanes and radius0.55 m. The new
[campaign driver](../tmp/stage1-replay-demo-150-20261010/campaign.py)
opened [Stage1](../runs/ppo_stage1/20261010-221814-887854/) in the Training UI,
with subsequent Stage2/3 final-actor transfers and reviews retained in this
campaign's folder. The copied one-off [reviewer](../tmp/stage1-replay-demo-150-20261010/review.py)
now accepts the approved Stage1 actor initializer and checks the150-collection
range; the copied driver's episode label is19,200. These fixes affect the
campaign's review/reporting, not the learner or success criteria.
The [live initial checkpoint check](../tmp/stage1-replay-demo-150-20261010/initial-live-check.json)
confirmed exact seed actor weights, exact fresh seed0 critic weights, empty
actor/value Adam states, zero learner/exploration offsets and19,200-episode
entropy budget at collection0. Stage1 training is running; no completed PPO
result is claimed. The campaign retains the original per-stage reward,
curriculum, workspace, physical execution and Stage4 handoff assessment.
Stage2 and Stage3 are pending this new chain's preceding final actor.

2026-10-10 replay-demonstration initial PPO check:
the [retained-array review](../runs/ppo_stage1/20261010-221814-887854/interim-review.json)
checked the first128 stochastic episodes (117 valid,1 valid and contained),
including all failed lanes. Float32 return reconstruction error was0;
transition-sum error was9.53674e-7. The first updated actor's native evaluation
was valid with selected-event reach0.621090 m, forward tip speed5.42418 m/s and
tip-to-drone height error-0.751389 m. Its whole-episode maximum radial excess
was0.143841 m, so it was not contained; raw return0.00277292 includes workspace
cost0.0124904. Saved native reward reconstruction was exact. This first update
retains a useful stroke but has not solved height, containment or target success.
The [initial review summary](../tmp/stage1-replay-demo-150-20261010/initial-review-summary.json)
and [launch preservation check](../tmp/stage1-replay-demo-150-20261010/launch-checks.json)
retain those limits and confirm300 pinned files/71 execution hashes unchanged,
including the stopped random run. Training continues with the approved reward
and schedule; no scientific criterion was relaxed and no full regression suite
or real-flight experiment was rerun for this one-off seed reconstruction.


2026-10-10 Clean Stage1–3 150-collection restart: Stage1 stopped by the user at collection44; no later stage was launched. Existing results and checkpoints were retained; no completion is claimed.


2026-10-10 Replay-demonstration Stage1–3 150-collection restart: Stage1 launched in [20261010-221814-887854](../runs/ppo_stage1/20261010-221814-887854); its subsequent status and review are recorded below.

2026-10-10 Stage1 reward/progress review, without implementation changes:
the updated [saved-array review](../runs/ppo_stage1/20261010-221814-887854/interim-review.json)
checks113 collections/14,464 episodes and native evaluations through collection112.
Maximum float32 cohort return and native return reconstruction errors were0.
The collection112 native rollout was valid and contained, but selected-event
reach was0.370392 m and tip-to-drone height error-0.921438 m, compared with
the seed's0.639620 m and-0.737543 m. Forward tip speed remained5.85219 m/s.
The existing Gaussian height factor was0.0219331 and event credit0.00263726.
This confirms declining nominal reach/height while containment improved; weak
reward signal is a diagnosis hypothesis, not proof of a single causal failure.
The [review and proposal record](../tmp/stage1-replay-demo-150-20261010/reward-review-proposal.json)
separates these retained measurements from illustrative same-event calculations
for an unapproved alternative. Those altered coordinates were not simulated;
the alternative was not implemented, all other events were not rescored under
it, and no new policy was trained. The current reward and automatic campaign
remained unchanged until the subsequent approval recorded below.

2026-10-10 approved linear outreach/height Stage1 implementation and preparation:
the [approved request](../tmp/stage1-linear-outreach-height-150-20261010/approved-request.json)
specifies `K*(0.5r+0.5h+2rh)`, linear h, unchanged workspace/validity rules and
150 new collections at128 lanes. The preceding campaign
[stopped after121 collections](../tmp/stage1-linear-outreach-height-150-20261010/stopped-summary.json)
without launching Stage2. Its final actor was copied immutably to
[source-actor.pt](../tmp/stage1-linear-outreach-height-150-20261010/source-actor.pt);
all logs/checkpoints remain in the original run. The implementation adds a
versioned Stage1 reward and explicit weight guards to task/scorer, loader,
environment and worker dispatch, with matching existing-UI labels/components.
Earlier reward arithmetic, physical execution, source recordings and model
selection are preserved; this is actor-only transfer, not exact resume.

[Validation](../tmp/stage1-linear-outreach-height-150-20261010/validation.json)
passed129/129 tests, including8 new synthetic scoring/delivery probes for the
formula and selected-event accounting, height symmetry/clipping, rigid-motion
and speed gates, future/invalid-suffix causality, failure correction and CUDA
parity. Synthetic coordinate probes do not establish feasible drone motions.
The [native transfer preflight](../tmp/stage1-linear-outreach-height-150-20261010/preflight.json)
exercised the actual immutable initializer/create-run path, confirmed exact
source actor with fresh critic/Adam/RNG/clocks, and re-integrated one source
actor under old and new task rewards. Five physical state arrays were exactly
identical, and execution was valid and contained. The new reward selected
reach0.358478 m and height error-0.921940 m, with event credit0.0760935; larger
credit here is a scoring change, not motion improvement. Independent same-event
accounting and an old saved native return reconstruction passed. All766 checked
retained input/run files stayed exact, with no physical source changes.
The [campaign driver](../tmp/stage1-linear-outreach-height-150-20261010/campaign.py)
will review Stage1's native motion and all-lane accounting before the existing
Stage2/3 continuation, retaining their original reward and handoff definitions.
It stops the sequence on a user stop or execution failure. Preparation alone
does not establish PPO improvement, robustness, target success or real flight.

2026-10-10 linear outreach/height launch verification:
the new [Stage1 run](../runs/ppo_stage1/20261010-232154-080392/)
was launched in the Training UI with the new v12 preset, then stopped at the
user's explicit pause before any completed collection or PPO update. The
[live initial checkpoint check](../tmp/stage1-linear-outreach-height-150-20261010/initial-live-check.json)
confirmed exact actor transfer from collection121, exact fresh seed0 critic
weights (different from the trained source critic), empty actor/value Adam
states, zero learner/exploration counters and a19,200-episode entropy budget.
The immutable actor payload matches the retained source hash. This is a fresh
reward experiment; its history is separate from the preserved old-reward logs.
The new driver retains a full review before each later-stage launch. Stage2/3
remain on their already approved presets and have not yet launched in this
new campaign. No completion or improvement is claimed by initialization checks.


2026-10-10 Replay-demonstration Stage1–3 150-collection restart: Stage1 stopped by the user at collection121; no later stage was launched. Existing results and checkpoints were retained; no completion is claimed.


2026-10-10 Linear outreach/height Stage1 experiment and existing Stage2/3 continuation: Stage1 launched in [20261010-232154-080392](../runs/ppo_stage1/20261010-232154-080392); its subsequent status and review are recorded below.


2026-10-10 Linear outreach/height Stage1 experiment and existing Stage2/3 continuation: Stage1 stopped by the user at collection0; no later stage was launched. Existing results and checkpoints were retained; no completion is claimed.

2026-10-10 explicit training pause for simulation work:
the [pause check](../tmp/stage1-linear-outreach-height-150-20261010/pause-check.json)
and current run summary confirm a user stop with0 completed collections,
0 completed episodes and0 gradient updates. The interrupted first cohort was
discarded. The latest checkpoint and all logs remain available. No training
workers are active, the campaign driver has exited, and Stage2/3 have not
launched. Training will remain paused until the user requests a restart or
resume. This state check did not rerun simulations or scientific experiments.


2026-10-10 measured simulation execution and Linux CUDA conditional support:
training was already stopped before these checks; it was not restarted.
The task-owned driver now supports Linux NVRTC/driver loading, and supported
Linux Torch CUDA builds select the existing device-controlled loop by default.
The [retained scripts, source hashes, timings and numerical checks](../tmp/simulation-performance-20261011/summary.json)
separate CPU plant/prescribed-cable timings, GPU plant throughput and display
rendering. Native/simulation sources and the checked physical/actuator
configurations match their pre-change hashes. Host/conditional1- and128-lane
state comparisons are bitwise equal; the collapsed-edge rejection and driver
WHILE caps1/4/20 were checked. No physical equation, solver tolerance, controller
clock, model artifact, reward, or policy was changed by this simulation work.
No model fitting, new policy training, real-flight test, viewer redesign or ROS
integration was run. The Windows loader branch was not rerun on Windows.
Exact-resume source checks remain enforced: task execution-source changes do
not authorize bypassing old checkpoint provenance.
The [existing regression suite](../tmp/simulation-performance-20261011/validation.json)
passed129/129 tests with CUDA available and no skips. These checks do not
establish policy improvement or real-flight validity.
The [default-backend check](../tmp/simulation-performance-20261011/default-backend-check.json)
constructed the128-lane task without an explicit scheduler choice and advanced
one finite step using `cuda_conditional_v1`. The
[paused-run source comparison](../tmp/simulation-performance-20261011/paused-run-source-check.json)
finds changed hashes for the two task scheduling files in the retained zero-update
run; exact resume under its old execution provenance remains incompatible.
This work did not restart that run or weaken its source checks.

2026-10-11 independent simulation execution and flight viewer: the user approved
the worker/display-interface/viewer scope. The local simulation now runs in its
own thread, while Qt consumes immutable latest frames. The shared renderer/window
accept named robots and optional cables with explicit source/timestamp metadata;
simulation, saved replay and a read-only frame feed are implemented. The docked
UI and camera/layer/playback controls replace the former simulation sidebar.
The existing saved-outcome browser remains available; `run_replay.py --flight-view`
opens its selected arrays in the new window.

The [regression report](../tmp/viewer-rebuild-20261011/validation-final-ui.json)
passes139/139 checks with CUDA available and no skips. New checks cover30 native
trajectory steps (0.3 simulated seconds) with every runtime checkpoint bitwise
equal to direct stepping, diagnostic toggles, pause/reset/source switching,
immutable ownership, invalid-publication prefix retention, freshness indicators,
read-only feed/frame guards and saved-array replay isolation. These are numerical
and interface checks, not policy-training or real-flight results.
The [X11/OpenGL UI checks](../tmp/viewer-rebuild-20261011/ui-checks.json) cover
trajectory application, manual restoration, force/layer/camera controls,
default/small window layouts, retained actor identities, multiple robots with an
absent cable, worker shutdown and the existing packaged Stage4 replay window.
The final checkbox check exposed unsafe inline tree rebuilding. Structural tree
updates now wait until checkbox callbacks finish, and force-diagnostic changes
retain simulation checkbox items. The [50-cycle layer check](../tmp/viewer-rebuild-20261011/layer-stress.json)
passes repeated force/robot hiding and restoration with stable checkbox identity
and a joined worker; two subsequent complete UI checks passed. This is a local
GUI check, not proof of cross-platform graphics stability.
A deliberately blocked Qt consumer allows14 native output steps while it is
idle; a short24-render camera rotation loop allows2 steps. Both resulting native
checkpoints equal direct execution at the same output count. This demonstrates
execution independence in those short cases; it does not establish a sustained
interactive-frame-rate or worst-case timing guarantee. The
[entry-point check](../tmp/viewer-rebuild-20261011/entry-point-check.json) opens
the simulation with an ephemeral local controller port, single-steps and closes
with its worker joined. The read-only feed preview also displays measured pose
and an explicitly stale estimated cable without playback/trajectory controls.
[Simulation preview](../tmp/viewer-rebuild-20261011/simulation-flight.png),
[replay preview](../tmp/viewer-rebuild-20261011/replay-flight.png) and
[frame-feed preview](../tmp/viewer-rebuild-20261011/display-feed.png) are retained
display artifacts, not flight measurements. The checked126 native/simulation,
controller, task, training and configuration files match their
[pre-change hashes](../tmp/viewer-rebuild-20261011/source-hashes-before.json).
No physics, physical timing, reward, learner source, active model or recorded
data was changed, and training remains paused. Exact-resume source guards remain
intact; the earlier task-scheduler changes still require their existing provenance
checks. This viewer change did not include a ROS bridge; the subsequent streaming
adapter is recorded below. A causal real-flight cable observer and learned-policy
deployment remain proposed work. Windows-specific interactive behavior was not
exercised by these Linux checks.

2026-10-11 approved custom-simulation ROS connection: added the host launcher,
streaming-only Docker ROS adapter, Crazyswarm `--custom-sim` entry and scripted PVA
test. Real/stock simulation launch branches remain unchanged. The adapter owns
transport/frame/clock checks and recording; native physics/controller, physical
parameters, delays, rewards, learning and active artifacts remain unchanged.

The [regression report](../tmp/ros-connection-20261011/validation-final.json) passes145/145
checks with CUDA available and no skips, including6 new transport/frame rejection,
FIFO/state-preservation, socket lifecycle and native command-clock parity checks.
The direct-step check compares every complete native checkpoint over10 commanded
output steps and confirms the configured controller/delay behavior.
The [retained ROS connection check](../tmp/ros-connection-20261011/connection-checks.json)
records270 scripted commands sent/received/scheduled with zero rejection and no
missed command slots. ROS pose and full-state feedback equal the retained native
snapshots bitwise. Replaying the actual recorded request schedule through the
same native model reproduces1,502 output states and the final exact checkpoint
bitwise; all270 retained applied events match their request plus0.03 s. Header
intervals in this run range0.02–0.05 s despite nominal30-Hz scheduling; these are
observed desktop/ROS timing results, not a hard-rate guarantee. The receipt-sample
to request-boundary lag ranges0–0.01 s in this case. The
[raw PVA result](../tmp/ros-connection-20261011/pva-result.json) and
[host run](../tmp/ros-connection-20261011/ros-pva-final/metadata.json) are retained.
The checked121 physical/controller/task/training/configuration source files match
their [pre-change hashes](../tmp/ros-connection-20261011/source-hashes-before.json).
These are transport/reconstruction checks, not identification, policy-training,
firmware equivalence or real-flight evidence. Training remains paused and no
policy is promoted.
The [final X11/GPU viewer check](../tmp/ros-connection-20261011/ui-checks-final.json)
runs a shorter2-s motion/5-s stream with150 accepted commands while the viewer
is open. The named `cf_2` source, absence of local trajectory editing, force
diagnostics, pause/step/reset and native-owner shutdown passed. During a0.22-s
blocked Qt consumer the owner advances22 output steps. The
[received-feedback check](../tmp/ros-connection-20261011/feedback-checks.json)
compares535 pose/full-state/odometry/cable/clock samples and536 TF samples to the
native trace: pose, quaternion, full-state fields, TF and cable positions agree
bitwise, timestamps match, and independent `Rᵀ v_world` reconstruction differs
from published body-frame velocity by at most1.39e-17 m/s. The
[viewer capture](../tmp/ros-connection-20261011/custom-simulation-flight-final.png)
is a simulated display artifact. The checked ROS adapters terminate after normal
window/headless shutdown; no test ROS server remains running. Windows behavior,
high-level Crazyswarm services, firmware controller agreement, calibrated latency,
causal cable observation and actual policy/flight use were not tested or added
in those original streaming checks; the later API step is recorded below.

2026-10-11 custom-simulation TCP restart fix: the reported port47654 failure had
no active listener when inspected; its original cause cannot be established
retrospectively. A separate closed-connection reproduction confirmed `Errno98`
with the old listener lacking address reuse. The listener now sets
`SO_REUSEADDR`, and occupied-port errors are reported without a traceback while
normal source/evidence cleanup still runs. Default port/domain and command,
feedback, physics, controller, reward and learning behavior remain unchanged.
The [restart checks](../tmp/ros-port-restart-20261011/restart-checks.json) pass two
back-to-back Docker-backed headless0.5-s launches on47654. An occupied-port CLI
check preserves the existing listener and returns the intended concise error.
The [regression report](../tmp/ros-port-restart-20261011/validation.json) passes146/146
checks without skips, including a real accepted-TCP-connection restart and
live-listener exclusivity test. These are transport lifecycle and short native
hold checks; the PVA trajectory, GPU viewer, fitting, policy training and real
flight were not rerun for this fix.

2026-10-11 Crazyswarm airborne command compatibility: user-approved command/YAML
integration now loads the hardware robot metadata with the explicit simulation
position override, exposes stock client discovery/parameters and functional
high-level services, and keeps command planning outside the unchanged native
plant/controller. The simulation shell/client clock patch and simulated CSV
scheduler handle ordinary Python execution, hover pacing and stream-to-land
handoff. Initial API/CSV checks found a stale hover deadline and independent ROS
topic/service delivery races. Their failed traces remain under
`tmp/ros-api-20261011`; they are not counted as successful results.

The [completed CSV check](../tmp/ros-api-20261011/csv-repeat-final-verified/checks.json)
runs the user's actual M5 CSV with3 repeats,15-s pre-hover,15-s between-hover and
3-s post-hover on isolated domain144. It records2,160 received/scheduled stream
commands and5 received/scheduled API operations, with zero stream/API rejections,
8,199 native steps over81.99 s and a completed airborne landing reference before
disarming. Final physical feedback remains distinct from the landing reference;
these results do not establish a real-flight tracking acceptance threshold.
The [reconstruction](../tmp/ros-api-20261011/csv-repeat-final-verified/replay-checks.json)
replays the retained request boundaries: all8,198 retained drone/cable snapshots
and the final BDF2 checkpoint agree bitwise. This is transport/numerical evidence,
not parameter identification or calibrated controller agreement.

The [API checks](../tmp/ros-api-20261011/api-check-complete/checks.json)
exercise upstream `Crazyswarm()` discovery without a joystick, metadata,
arm/disarm, relative/absolute go-to, group filtering, upload/reverse/timescale,
stream release/landing and emergency/reset at0.5× simulation pacing. Emergency
leaves the native checkpoint unchanged while suspended and blocks resume until
reset. The [firmware planner comparison](../tmp/ros-api-20261011/firmware-planner-checks.json)
checks63 takeoff/land/go-to samples against pinned float32 firmware polynomial
evaluation; the maximum absolute component difference is1.04e-5 (mixed SI fields),
within the declared2e-4 numerical comparison tolerance. It does not establish
firmware motor-control equivalence.

The [GPU viewer check](../tmp/ros-api-20261011/ui-checks-final.json) runs ROS PVA
commands with the overridden1.1-m initial state, named `cf_2`, simulated cable,
force controls and pause/step/reset. During a0.22-s blocked Qt consumer the
native owner advances21 output steps, and normal window closure joins the owner.
The [capture](../tmp/ros-api-20261011/custom-simulation-flight-final.png) is a
simulation display artifact. The [domain checks](../tmp/ros-api-20261011/domain-checks.json)
reject a duplicate custom domain and an existing ROS server without stopping it.
The [regressions](../tmp/ros-api-20261011/validation-complete.json) pass153/153
checks with no skips. The [source check](../tmp/ros-api-20261011/protected-source-checks.json)
finds all121 protected native/simulation/controller/task/learning/config source
hashes unchanged from the retained pre-connection baseline. Ground/contact,
physical power cutoff, full firmware parameter emulation, simultaneous multiple
plants, real-flight cable observation, policy deployment and new fitting/flight
experiments remain outside this implementation. Training remains paused and no
model or policy is promoted.

2026-10-11 consistent YAML initialization: the selected robot's hardware YAML
position now initializes startup and Reset by default. The1.1-m simulation
example is used only with an explicit `--simulation-yaml`; metadata hashes only
the YAML files actually loaded. The launcher no longer treats the displayed
floor as a physical boundary. This removes a renderer-dependent restriction,
without changing native cable validation, contact assumptions or controller
behavior. `z=0` remains a free-space initial state, not a grounded/contact model.

The [initialization checks](../tmp/yaml-initialization-20261011/verified/checks.json)
run three4-s Docker-backed headless launches on isolated domains172–174:
the actual YAML at `(0,0,0)`, an explicit `(0,0,1.1)` override, and an independently
named `generic_robot` at `(2.3,-1.7,-0.4)` with explicit cable geometry and a
display plane at100 m. Each completes400 native output steps; native initial
positions, ROS position parameters and status agree with configuration. The
52/53/51 captured ROS pose samples respectively agree bitwise with retained
native states. Explicit cable nodes translate with the configured drone origin;
the attachment agrees within the declared1e-14-m floating-point reconstruction
tolerance. Owner-thread Reset restores the exact initial BDF2 checkpoint and
position, zeros time/step counters and increments generation in all three cases.
The hardware YAML bytes remain unchanged.

The [focused regressions](../tmp/yaml-initialization-20261011/verified/regressions.json)
pass14/14 without skips, including default selection, arbitrary robot identity
and explicit overrides. The [source hashes](../tmp/yaml-initialization-20261011/verified/protected-source-checks.json)
find all121 previously protected physical/controller/task/learning/configuration
files unchanged. The first checker used a different floating-point association
for its expected cable translation; its traces remain outside `verified` and
are not the reported completed check. The expected arithmetic was corrected
without relaxing native criteria. Full CSV playback, GPU viewing, physical
ground/contact, fitting, policy training and real flight were not rerun for this
initialization change.

2026-10-11 README refresh: [README](../README.md) now documents Linux setup,
standalone versus Crazyswarm-connected simulation, the Docker compose/start/shell
sequence, CSV execution, exact YAML startup/Reset, explicit overrides, and the
free-space/contact limits. Outdated Linux/connection status and launcher-preset
descriptions were replaced using the inspected entry points and this record.
Checked11 local README links/anchors, seven Bash examples with syntax-only
`bash -n`, balanced code fences and diff whitespace. This is documentation-only
work: no launch example, simulation, regression suite, fitting, training or
real-flight experiment was executed or rerun.

2026-10-11 shared policy contract: added the explicitly
[proposed state/action interface](#shared-policy-interface-proposed) and a README
pointer. Inspected the current actor encoding, velocity/PVA controller paths,
bridge/mailbox timing, installed Crazyswarm servers, FullState serialization and
firmware source. The [installed-source and no-radio packet audit](../tmp/policy-contract-20261011/source-and-codec-audit.json)
retains commits, file hashes and a mocked cflib send. Scalar packing confirms
millimetre/milliradian scaling, truncation and overflow rejection; quaternion
decode, flashed firmware, active onboard parameters and flight behavior remain
unverified. This initial draft recommended a separately versioned shared
velocity-intent to PVA generator with reference-state actor inputs and compatible
new training; that recommendation was superseded by the acceleration proposal
below. No profile, policy runner or cable observer was implemented. Existing
profiles, checkpoints, controller defaults and research definitions were not
changed. [Static checks](../tmp/policy-contract-20261011/static-checks.json)
verified 18 local README/contract links and anchors, seven Bash blocks with
syntax-only `bash -n`, balanced fences, absence of nonbreaking spaces and all 121
protected source hashes unchanged. Diff whitespace also passed. No simulation,
model fit, policy training, regression suite or real-flight experiment ran.

2026-10-11 acceleration-action proposal and Mellinger review: at the user's
request, replaced the unimplemented velocity-intent proposal with acceleration
outputs, distinguishing acceleration feedforward through integrated PVA from
direct acceleration-to-force control. The
[read-only source/configuration review](../tmp/policy-acceleration-20261011/mellinger-review.json)
retains force/attitude/output equations, exposed gains, source control rate and
requested YAML parameters. `FullState` retains absolute-XYZ feedback; zeroing
all translational P/D/I gains would remove those feedback terms in the inspected
source without necessarily changing firmware, but that configuration and its
mode transitions are not approved or implemented. The proposed PVA generator
uses exact held-acceleration reference updates; acceleration bounds, braking,
jerk handling and the deployment-route choice remain unselected. Existing
physics, controllers, presets, checkpoints, firmware and onboard parameters
were not changed. [Static checks](../tmp/policy-acceleration-20261011/static-checks.json)
verified 19 README/current-contract local links and anchors, seven Bash blocks
with syntax-only `bash -n`, balanced fences, no nonbreaking spaces and all 121
protected source hashes unchanged; diff whitespace passed. This was source and
documentation review only: no new packet probe, simulation, fitting, policy
training, regression suite or real flight ran. Flashed firmware and active
onboard parameters remain unverified.

2026-10-11 requested simulation relaunch: reopened the existing custom-simulation
flight viewer through the Crazyswarm `./docker/start.sh --custom-sim` entry point
for `cf_2` on simulation ROS domain142, with the current hardware YAML initial
position `(0,0,0)` and no position override. The new
[session](../output/custom_simulation/20261011-092036-b69e92d3/metadata.json) and
[launch check](../tmp/simulation-relaunch-20261011/checks.json) confirm a live
simulation process, connected ROS bridge and ongoing state feedback with no
reported simulation error. No training workers are running; training remains
paused. No settings, reward, model or policy changed, and no scripted flight,
regression suite or real-flight experiment was run for this relaunch.


2026-10-11 Linear outreach/height Stage1 relaunch and existing Stage2/3 continuation: Stage1 launched in [20261011-052639-788494](../runs/ppo_stage1/20261011-052639-788494); its subsequent status and review are recorded below.

2026-10-11 training relaunch verification: the user clarified that the relaunch
request meant training. The simulation opened by this task was
[closed](../tmp/simulation-relaunch-20261011/closed-for-training.json), preserving
its evidence. The [compatibility check](../tmp/stage1-linear-outreach-height-150-relaunch-20261011/compatibility.json)
confirmed0 completed collections/episodes/updates in the paused source and exact
actor equality with its original collection121 initializer. Only the conditional
CUDA driver and scheduler-selection source differ from the paused execution
snapshot; exact-resume checks remain enforced. The already approved reward,
150-collection budget,128 lanes,0.55-m workspace and later-stage definitions
remain unchanged.

The [native preflight](../tmp/stage1-linear-outreach-height-150-relaunch-20261011/preflight.json)
checked immutable actor initialization, fresh critic/Adam/RNG/clocks, same-event
reward accounting, old saved-return reconstruction and804 retained input/run
files. The [baseline comparison](../tmp/stage1-linear-outreach-height-150-relaunch-20261011/native-baseline-comparison.json)
finds five native drone/cable state arrays identical to the retained pre-pause
baseline. The [initial live checkpoint](../tmp/stage1-linear-outreach-height-150-relaunch-20261011/initial-live-check.json)
confirms exact actor transfer, fresh seed0 value weights, empty optimizers and
zero learner/exploration offsets with a19,200-episode entropy budget. At that
initial check a live worker was collecting the first128-lane cohort in the
Training UI. This launch
check does not establish learned improvement or full CUDA campaign equivalence;
no regression suite, fitting or real-flight experiment was rerun. The replacement
driver retains evaluation before each later-stage handoff and stops on user stop
or failure, without adding research thresholds or model promotion.

2026-10-11 training relaunch early accounting check:
the [saved-array review](../runs/ppo_stage1/20261011-052639-788494/interim-review.json)
checks the first2 completed collections/256 episodes and the collection1 native
evaluation. Cohort float32 return reconstruction error is0; maximum transition
sum error is5.96046e-8. The first native evaluation is valid, with raw return
0.0860904 under the unchanged approved linear Stage1 reward; native return
reconstruction error is0. The [live launch check](../tmp/stage1-linear-outreach-height-150-relaunch-20261011/launch-checks.json)
confirms the UI, campaign driver and sole training worker are running, with815
pinned files/current execution hashes unchanged. Stage2/3 have not launched.
These early checks confirm execution and accounting, not convergence or learned
motion improvement.



2026-10-11 Linear outreach/height Stage1 relaunch and existing Stage2/3 continuation: Stage1 completed to collection150 /19,200 total stage episodes. Its [retained review](../runs/ppo_stage1/20261011-052639-788494/review.json) checks 19200 newly collected episodes, saved native scoring/crossing accounting, finite checkpoints and input preservation. Current-run valid/contained/geometric/strict counts are19200/9061/2583/976. Cohort counts use the current target; Stage3 native evaluation uses final target geometry. One nominal case and adaptive cohorts do not establish robustness or real flight.


2026-10-11 Linear outreach/height Stage1 relaunch and existing Stage2/3 continuation: Stage1 native evaluation reviewed before Stage2 continuation: {'forward_reach_m': 0.606187296161614, 'tip_drone_height_error_m': -0.41458605080642297, 'forward_mps': 6.19299179610647}. Existing Stage2/3 reward definitions remain unchanged; no new handoff threshold or model promotion.


2026-10-11 Linear outreach/height Stage1 relaunch and existing Stage2/3 continuation: Execution needs attention: cannot unpack non-iterable Stage1Score object. No successful completion is claimed.


2026-10-11 Approved Stage2/3 continuation after Stage1 handoff script repair: Stage2 launched in [20261011-063417-108510](../runs/ppo_stage2/20261011-063417-108510); its subsequent status and review are recorded below.

2026-10-11 completed Stage1 review and Stage2/3 continuation repair:
the [final Stage1 review](../runs/ppo_stage1/20261011-052639-788494/review.json)
checks150 collections/19,200 episodes,39 native evaluations and finite
checkpoints. All19,200 training episodes were valid;9,061 were contained,
2,583 had broad geometric crossings and976 met the strict crossing diagnostic.
These adaptive cohort totals are not independent robustness trials. Float32
cohort return reconstruction error is0; maximum transition-sum error is5.96046e-8.
At the same new-reward nominal start, final selected-event reach increased from
0.358478 to0.606187 m and tip-to-drone height error changed from-0.921940 to
-0.414586 m. Final forward tip speed is6.19299 m/s. The final native rollout
exceeded the0.55-m radius by0.0177724 m and its broad crossing entry angle was
31.2389 degrees, so it does not pass containment or the existing30-degree strict
diagnostic. No model was promoted and no research threshold changed.

The original campaign stopped before Stage2 because its handoff helper attempted
to unpack the single `Stage1Score` returned by `Stage1Scorer`. The
[continuation request/fix record](../tmp/stage23-continuation-150-20261011/continuation-request.json)
and separate driver correct only that return-value use and begin after the
completed Stage1. Original scripts, failure traces, logs and checkpoints remain
unchanged; Stage1 is not retrained. The
[Stage2 native handoff](../tmp/stage23-continuation-150-20261011/stage2-handoff.json)
retains valid execution and five drone/cable state arrays identical to the final
Stage1 replay, with exact actor transfer and fresh critic/Adam/RNG/clocks checks.
The [initial live check](../tmp/stage23-continuation-150-20261011/stage2-initial-live-check.json)
confirms a live Stage2 UI/worker, exact actor weights, a critic distinct from the
trained source critic, empty optimizers, reset counters/exploration offsets and
all1,452 pinned files/current execution hashes unchanged. Stage2/3 keep their
approved Gaussian-height rewards,150 collections each,128 lanes,0.55-m workspace
and unchanged handoff/curriculum criteria. This repair was verified through the
actual handoff and launch; no regression suite, fitting or real-flight experiment
was rerun.

2026-10-11 Stage2 early continuation check:
the [interim saved-array review](../runs/ppo_stage2/20261011-063417-108510/interim-review.json)
checks3 completed collections/384 episodes: all valid,133 contained,281 geometric
crossings and71 strict crossing diagnostics. Float32 return reconstruction error
is0, with maximum transition-sum error1.49012e-8. The first updated native actor
is valid and its saved return reconstruction is exact; this is early execution
evidence, not convergence or target success. The
[launch check](../tmp/stage23-continuation-150-20261011/stage2-launch-checks.json)
confirms the Stage2 UI, sole training worker and continuation driver remain live,
with1,452 pinned files/current execution hashes unchanged. Stage3 is pending the
completed Stage2 review and unchanged native actor-transfer checks.


2026-10-11 Approved Stage2/3 continuation after Stage1 handoff script repair: Stage2 completed to collection150 /19,200 total stage episodes. Its [retained review](../runs/ppo_stage2/20261011-063417-108510/review.json) checks 19200 newly collected episodes, saved native scoring/crossing accounting, finite checkpoints and input preservation. Current-run valid/contained/geometric/strict counts are19200/5771/18741/9955. Cohort counts use the current target; Stage3 native evaluation uses final target geometry. One nominal case and adaptive cohorts do not establish robustness or real flight.


2026-10-11 Approved Stage2/3 continuation after Stage1 handoff script repair: Stage3 launched in [20261011-073423-438409](../runs/ppo_stage3/20261011-073423-438409); its subsequent status and review are recorded below.


2026-10-11 Approved Stage2/3 continuation after Stage1 handoff script repair: Stage3 completed to collection150 /19,200 total stage episodes. Its [retained review](../runs/ppo_stage3/20261011-073423-438409/review.json) checks 19200 newly collected episodes, saved native scoring/crossing accounting, finite checkpoints and input preservation. Current-run valid/contained/geometric/strict counts are19112/2533/5918/3626. Cohort counts use the current target; Stage3 native evaluation uses final target geometry. One nominal case and adaptive cohorts do not establish robustness or real flight.


2026-10-11 Approved Stage2/3 continuation after Stage1 handoff script repair: Existing Stage4 handoff assessment: {'final_target_level_reached': False, 'final_native_valid': True, 'final_native_contained': False, 'final_native_strict_hit': False}. This Stage1–3 restart did not train Stage4 or relax any criterion.

2026-10-13 completed-stage motion and curriculum review:
the [saved-array analysis](../tmp/stage-review-20261013/comparison.json),
[analysis script](../tmp/stage-review-20261013/analyze.py) and visually inspected
[comparison figure](../tmp/stage-review-20261013/motion-comparison.png) compare
three final native replays and Stage3's best-by-saved-reward replay. All450 retained
128-lane collection files independently reconstruct float32 returns exactly;
the maximum transition-sum error is1.19209e-7. Four native return reconstructions
are exact. All2,612 retained run files and current execution hashes remain
unchanged; no worker is active. This work rescored saved arrays only: no physics,
policy inference, training, fitting, regression suite or real-flight experiment
was rerun, and no model was promoted.

Final task-selected reach for Stages1/2/3 is0.606187/0.668553/0.638201 m;
tip-to-drone height error is-0.414586/-0.342179/-0.326884 m and selected forward
tip speed is6.19299/5.11391/4.80189 m/s. Their maximum drone sideways offsets
are0.430323/0.481688/0.545378 m, all occurring after the rewarded releases.
Full-motion peak forward tip speeds are6.36394/5.50993/6.09444 m/s, so selected
release speeds must not be described as full-motion peaks. Rescoring all three
final motions with the same approved linear Stage1 reward gives raw returns
0.403537/0.417631/0.386438; this task proxy does not define physical whip success.
Stages2/3 retain the older Gaussian-height base, so their original raw returns
cannot be compared directly with Stage1's new reward. Stage2 improved broad
delivery: all1,024 episodes in its last8 collections geometrically hit the broad
target, with64% strict diagnostics when rounded to whole percent (exact63.8672%).
Its final native broad crossing was strict-qualified, although radial excess
remained0.00361050 m. Stage3 improved aim but traded some reach/release speed and
containment; this is a mixed outcome rather than loss of all learned motion.

Stage3 advanced once, at collection4, from1.0×0.6 to0.6×0.4 m. Its next8
collections averaged1.07422% geometric hits, recovering to71.4844% in the last8,
without satisfying80% hits/all lanes valid for3 consecutive collections plus
the nominal hit. The longest post-shrink streak was2; collection141 had81.25%
hits but only127/128 valid lanes and reset the streak. None of39 saved native
evaluations hit the final0.1×0.1-m target. The final forward-plane crossing
missed its center by-0.284614 m laterally and-0.0569181 m vertically; both are
outside the approved±0.05-m per-axis tolerance. The best-by-reward collection132
also missed, with lateral error-0.294977 m. Only1.75781% of the last1,024 Stage3
episodes were contained, despite generally small weighted boundary costs.

Inspection confirms that event credit rewards a best release with broad
proximity/centering, and the travel discount stops at that release. The final
precision native center bonus was1.09634e-9 while positive base/proximity credit
remained; current-target center guidance uses the wider0.15-m sigma. These facts
support hypotheses of weak precision guidance, an abrupt first size transition
and insufficient incentive for settling after release, but do not isolate their
causal effects. Consistent outreach/height bases, stronger guidance at actual
crossings, intermediate target sizes and explicit settling would be proposed
research changes requiring approval; none was implemented in this review.
Stage4 was not trained because this campaign did not request it and its existing
handoff assessment failed. The completed Stage2 actor remains a comparison
reference, without promotion or replacement of registered models.

2026-10-13 approved custom real-flight display: added `--custom-real` and
viewer-only `--attach`, using the existing real ROS/OptiTrack stack and shared
GPU window. Drone poses use `/poses` capture in its declared `mocap` frame;
there is no onboard-estimate fallback, YAML-position stand-in or simulated cable.
The [focused checks](../tmp/custom-real-20261013/focused-tests.log) passed20/20.
Retained [synthetic ROS reconstruction](../tmp/custom-real-20261013/capture-checks.json)
and [source/frame rejection](../tmp/custom-real-20261013/source-frame-rejection-checks.json)
checks cover direct/transformed capture, multiple names, stale tracking, invalid
orientation, duplicate names, recovery and unavailable/wrong sources.
The actual NVIDIA GPU window, Qt Ctrl+C cleanup and mock owned-stack lifecycle
were checked separately. The [final check record](../tmp/custom-real-20261013/final-checks.json)
records syntax, whitespace, reader cleanup and unchanged hashes of121 protected
physics/controller/task/training/configuration files. Failed initial Jazzy
metadata attempts and display-test harness failures remain retained alongside
the successful checks; no scientific threshold was changed. No real flight,
OptiTrack hardware capture, fitting, policy training or physical-model experiment
was run for this change. Hardware capture alignment and flight behavior remain
unverified; force-controller and learned-policy deployment work is unchanged.


2026-10-13 approved audit fixes: reset real and simulated CSV hover deadlines;
record raw OptiTrack poses before TF orientation fallback; mark force-controller
gains dirty before partial writes and retain emergency cleanup on v1 restoration
failure; validate and commit simulator parameter batches atomically, terminating
the bridge on uncertain API acknowledgement. New logger recordings use schema
`optitrack_poses_pva_motor_v4` and `pose_*` source timestamp columns; the existing
v3 script filename and command/motor sidecars remain.
The [full regression record](../tmp/audit-fixes-20261013/regressions.json) passed
163/163 with CUDA available, and the adjacent
[flight-script checks](../tmp/audit-fixes-20261013/script-regressions.log) passed
3/3, covering real/simulation pacing, raw quaternion validity and injected
partial-write/restoration failures across all three force scripts.
An [isolated ROS parameter check](../tmp/audit-fixes-20261013/atomic-probe.log)
on domain221 verified rejected-batch preservation, successful batch/alias
agreement and conflicting-alias rejection. The
[synthetic ROS logger check](../tmp/audit-fixes-20261013/logger-ros-probe-verified.json)
on domain222 retained12 poses,12 commands and12 motor packets, including three
invalid raw orientations marked invalid. Its first attempt received11 poses
before DDS matching settled; that failed test-harness assertion is retained
separately. These are software, synthetic recording and connection checks.
No hardware flight, onboard gain-restoration verification, physical calibration,
new fitting or policy training was performed. Physical models, controller defaults,
rewards, checkpoints and source recordings were not changed by these fixes.
The final comparison with the prior audit also found unrelated changes in six
training/task files; these were left untouched. The
[final check record](../tmp/audit-fixes-20261013/final-checks.json) distinguishes
those changes from this task's three deployment-source edits. Regression results
refer to the recorded test execution, not an assertion that concurrently edited
training files stayed unchanged afterward.


2026-10-13 Approved Stage2 crossing-priority restart: implemented the explicitly
approved `0.2W + max_crossing(C+2G+S)` formula in a separate task/preset, with
causal endpoint delivery, complete-crossing selection, retained strict criteria
and fresh actor-only initialization. No physical plant/controller changed.
The [experiment approval](../tmp/stage2-crossing-restart-20261013/approval.json)
and [native transfer preflight](../tmp/stage2-crossing-restart-20261013/preflight.json)
retain provenance. Native tracked-drone position/velocity/rotation and cable
position/velocity arrays exactly match the preceding Stage2 final replay.
New raw return is3.256495237350464; independent NumPy crossing arithmetic and
transition-return sums have zero discrepancy. Critic/Adam/RNG/clocks remain
fresh after exact actor transfer; failure replacement/clawback was checked
on that rollout. All3,321 protected existing run/model files were unchanged.
Scoring57/57 (including7 new crossing tests and CPU/CUDA comparison), pipeline8/8,
handoff10/10 and records/UI14/14 tests passed; the new preset's UI summary and
experiment details initialized correctly offscreen. These checks establish
numerical/accounting behavior and one native nominal baseline, not trained
improvement, robustness or real-flight performance. The subsequent launch is recorded below.


2026-10-13 Stage2 crossing-priority experiment: Launched [20261013-144515-220029](../runs/ppo_stage2/20261013-144515-220029/) in the visible Training UI for150 new collections at128 lanes. The user stopped this run after52 collections/6,656 episodes; the subsequent stop record is below. No Stage3 continuation was launched.


2026-10-13 Stage2 crossing-priority launch verification: the active worker's
[snapshot/initialization check](../tmp/stage2-crossing-restart-20261013/launch-checks.json)
confirms the intended actor-only source,150-collection/128-lane budget, new
reward,0.55-m workspace and stage-local exploration. Its
[first-collection review](../runs/ppo_stage2/20261013-144515-220029/interim-review.json)
reconstructs all128 raw float32 returns exactly and verifies transition sums
within the retained1e-5 bound and additive components within1e-10. Counts were
128 valid,51 contained,128 broad geometric hits,88 strict broad hits and zero
final10×10-cm hits. One saved nominal evaluation also rescored exactly. This is
a startup/accounting check, not evidence of convergence or improved precision.
The [single-run driver status](../tmp/stage2-crossing-restart-20261013/status.json)
and run progress are the live status sources; Stage3 is not automatically
launched. Existing run/model artifacts remain preserved.


2026-10-13 User-requested Stage2 stop: the crossing-priority run
[20261013-144515-220029](../runs/ppo_stage2/20261013-144515-220029/)
stopped safely after52 completed collections/6,656 episodes using its existing
`stop.request` protocol. The worker exited and released its lock; `latest.pt`
was checked to match the stopped summary's round and episode count. Existing
checkpoints, logs and replays are retained. No Stage3 continuation was launched.
The [stop verification](../tmp/stage2-crossing-restart-20261013/user-stop.json)
records this action; stopping does not establish convergence or a new policy
performance result. No reward, physics or target definition changed.


2026-10-13 approved viewer shutdown and consecutive-flight fixes: moved GUI
shutdown out of Python signal handlers into the Qt loop; isolated the graphical
real viewer under a surviving host supervisor with ownership-checked cleanup;
completed VTK initialization before PyVistaQt import. Added per-robot display
trajectory epochs from newly discovered flight-client publishers in both ROS
adapters, plus a manual Clear trails button. Epoch changes clear displayed paths
without resetting physical state or sending flight commands. The real cable
feed remains unavailable; the custom simulation's existing cable rendering and
physical model are unchanged.
[Final regressions](../tmp/viewer-lifecycle-20261013/regressions-final.json)
passed172/172. [Eight GPU checks](../tmp/viewer-lifecycle-20261013/gpu-normal-exit.json)
passed with SIGINT delivered during VTK rendering, normal interpreter exit,
automatic/manual trail clearing and visible simulated cable. Earlier attempts
exposed native import/finalization faults and a test assertion made before the
60-Hz display refresh; those logs remain separate from the final results.
[Isolated ROS checks](../tmp/viewer-lifecycle-20261013/session-results.json)
on domains218/219 observed two consecutive client identities without restarting
either viewer source. [Mock stack ownership checks](../tmp/viewer-lifecycle-20261013/stack-lifecycle-checks.json)
verified window-close preservation, Ctrl+C cleanup and rejection of an unowned
process; [injected native worker crashes](../tmp/viewer-lifecycle-20261013/supervisor-crash.json)
verified the surviving owner's cleanup and attached-view isolation.
A [final read-only production attachment](../tmp/viewer-lifecycle-20261013/production-attach-final.json)
opened against the already running real stack and exited normally on Ctrl+C,
leaving the existing ROS processes running. No new real server, flight command,
physical calibration, fitting, training or real-flight experiment was performed
by these checks. No physical/controller parameters, research criteria or model
artifacts were changed by this work.

2026-10-13 approved Ctrl+C stack shutdown: an attached real viewer now captures
its matching local Crazyswarm/OptiTrack ROS launch and stops that tree on
Ctrl+C/SIGTERM, including supervisor fallback after a worker fails during the
stop request. This supersedes the previous attached-viewer Ctrl+C preservation
behavior above; closing only its window still leaves ROS running. The custom
simulation and standalone simulation now defer signal-driven window shutdown
through the shared Qt timer. Custom simulation isolates its Docker client and
retains session/domain/PID identities for ROS bridge cleanup; Docker remains up.
[Focused regression checks](../tmp/ctrl-c-stack-20261013/focused-tests.log) passed
41 tests, including stale-identity rejection, ambiguous-launch rejection,
preservation of another domain and an independent same-domain process, and
cleanup of remembered children after their launch parent exits.
[Docker/Qt integration checks](../tmp/ctrl-c-stack-20261013/integration-results.json)
closed the custom simulation GUI and its bridge, immediately restarted a
headless simulation on the same isolated domain224, stopped a synthetic attached
real ROS launch and its capture/server children on domain226, and closed the
standalone GPU viewer on Ctrl+C. All four launchers exited0. No test-domain
processes remained in the container. Syntax, CLI help and whitespace checks
passed. Real hardware shutdown and physical flight were not exercised; no radio
or flight-command services were used. These checks did not fit a model, train a
policy, change physical assumptions, or establish real-flight evidence.

2026-10-13 approved real-motion cable simulation and NatNet unicast support:
`--custom-real` now drives the existing cable-only native material solver from
accepted OptiTrack pose samples while preserving per-robot cable/BDF2 history.
The display labels the drone Measured and cable Simulated; its hanging, stationary,
untwisted initial condition is an explicit prior, with no ground contact or cable
measurement correction. The relay retains accepted samples between display
heartbeats and uses integer source timestamps. Policy observations/actions,
controllers, physical model files, fitting, training and research criteria were
not changed by this work.
[Focused checks](../tmp/real-cable-20261013/focused-tests-final.log) passed47 tests.
[GPU checks](../tmp/real-cable-20261013/gpu-results.json) and the
[synthetic measured-drone/simulated-cable capture](../tmp/real-cable-20261013/measured-drone-simulated-cable.png)
verified visible cable geometry, separate provenance, preserved history across
flight-display epochs, and deferred Ctrl+C shutdown. These are display checks.
[Live read-only Motive connection](../tmp/real-cable-20261013/live-refined-results.json)
on isolated domain228 accepted2805 `cf_2` samples, advanced6347 native substeps,
and ended without cable errors; no flight server or command was started.
This is capture/connection evidence, not physical cable accuracy or real flight.

Initial live attempts retained explicit failures. Accepted-prefix handling was
corrected, and a later retained jitter-driven implicit rejection motivated bounded
substep refinement. [Reconstruction of that exact rejection](../tmp/real-cable-20261013/refinement-reconstruction.json)
required one2× refinement and retained the existing native criteria; the final
live run also used one2× refinement. Maximum observed segment-length residual
was1.11e-16 m in both checks. This is a numerical constraint check, not a model-fit
or policy result. Exhausted refinement still reports failure rather than relaxing
criteria or replacing state.

The Docker patch retains the registered unicast endpoint, sends keepalives,
filters interleaved replies/frames, and preserves multicast mode selection.
The existing capture-node Ctrl+C context race was also corrected. The current
container and persistent `crazyswarm2:jazzy-pinned` image were rebuilt with the
same Crazyswarm/firmware pins; retained
[image build record](../tmp/real-cable-20261013/docker-image-results.json) and
[fresh-image source hashes](../tmp/real-cable-20261013/fresh-image-source-hashes.json)
identify them. A [fresh-image isolated unicast fixture](../tmp/real-cable-20261013/fresh-image-unicast.log)
published503 poses, retained its endpoint across10 keepalives, handled interleaved
command/frame packets, and preserved the explicit5-s missing-frame error. The
final real multicast capture node stopped with exit0 on SIGINT. Motive settings
were unchanged; that checkpoint did not include repaired live Motive unicast.
The later unicast connection check is retained below. Syntax, CLI-help and
whitespace checks passed; no test ROS processes remained at that checkpoint.

2026-10-13 real-motion cable freeze repair: the user's retained
[unicast display trace](../output/custom_real/20261013-210524-243e9f5e/connection.jsonl)
reproduced an implicit endpoint rejection at15.1855 s, after which the displayed
drone continued moving while the cable remained frozen. The old retry rule
could repeat the same single substep for very short receipt-time intervals.
Retries now multiply the actual baseline subdivision by2,4,8 and also cover
numerical position-projection and velocity-constraint rejections. Every retry
starts from the same accepted cable state/BDF2 history; native equations,
iterations, accuracy thresholds and physical configuration files are unchanged.
Invalid inputs and exhausted retries still report an error and freeze.

[Retained-capture reconstruction](../tmp/real-cable-motion-20261013/replay-results.json)
processed all9532 accepted pose samples through33.2697 s without cable errors.
The simulated attachment matched every displayed measured attachment exactly;
observed root motion ranges were0.506/0.931/0.351 m in XYZ, and the tip also moved.
There were13 refinement retries, with maximum8× subdivision. Maximum observed
segment-length residual was1.11e-16 m. These are reconstruction/numerical checks
under receipt timestamps, not physical cable accuracy, model-fit or policy results.
[GPU checks](../tmp/real-cable-motion-20261013/gpu-results.json) verified60
reconstructed frames: cable-line points, marker translations and attachment
labels followed their accepted states and the window closed normally.
The floor was hidden for the retained
[before](../tmp/real-cable-motion-20261013/moving-cable-before.png) and
[after](../tmp/real-cable-motion-20261013/moving-cable-after.png) display captures;
no ground-contact behavior was added.

A separate [live read-only Motive unicast connection](../tmp/real-cable-motion-20261013/live-results.json)
on isolated domain231 accepted2052 poses and advanced4634 native cable substeps
without cable errors; the capture node stopped with exit0 on SIGINT.
No flight server or flight commands were started, and Motive settings were
unchanged. The source/drone was near the floor during this connection check;
moving-drone behavior was verified using the retained unicast reconstruction.
The [final check record](../tmp/real-cable-motion-20261013/final-checks.json)
records49 focused tests plus syntax, CLI-help, whitespace and process-cleanup
checks. Physical cable accuracy and new live moving-drone flight remain unverified.

2026-10-13 approved experimental cable reconstruction: added optional
`--track-cable` using the user's confirmed model marker spacing, exact ROS
cloud/pose timestamp pairing, retained mixed raw candidate clouds, bounded
geometric/temporal association, explicit null/ambiguous nodes and separate
Estimated display geometry. This does not correct the existing simulated cable.
The current upstream cloud discards NatNet IDs/visibility flags; it is not an
unlabeled-only source. [Focused tests](../tmp/cable-tracking-20261013/focused-tests.log)
passed57 checks including permutation/duplicate/clutter handling, short occlusion
and reacquisition, ambiguous points, bending, initialization/gap/capacity rejection,
PointCloud2 layout decoding and generation/timestamp handling. The isolated
[synthetic ROS check](../tmp/cable-tracking-20261013/ros-results.json) retained644
clouds and reconstruction results, including632 full and12 partial frames, exact
pose/cloud timestamp pairing and test-node cleanup. No real capture node, radio
server or flight command was started. [GPU verification](../tmp/cable-tracking-20261013/gpu-results.json)
checked70 synthetic frames, separate raw/estimated/simulated geometry, missing-node
gaps, stale hiding, independent visibility and normal window closure; the
[missing-marker screenshot](../tmp/cable-tracking-20261013/tracking-gap.png) was
visually inspected. CLI help, syntax compilation and diff whitespace checks
passed. These are reconstruction/transport/rendering checks; real Motive cable
identity accuracy, occlusion robustness during whipping and physical accuracy
remain unverified. Native physics, controller, firmware, Motive settings and
training behavior were not modified by this change.

2026-10-13 approved hand-test recording and replay: added `--capture-only`,
Record/Stop/Replay/Open controls, separate clip folders, saved mixed-provenance
display states and received events, and offline `run_recording.py` playback.
[Focused checks](../tmp/record-mode-20261013/focused-tests.log) passed64 tests,
including exact saved-frame round trips, absent telemetry and missing markers,
source-clock regression on the monotonic clip timeline, multiple clips, shutdown
finalization, incomplete-prefix recovery, corruption rejection, disk-write error
reporting and capture-only command/supervisor isolation.
[Synthetic ROS recording](../tmp/record-mode-20261013/ros-results.json) retained422
display frames and1718 received/association events; all saved frames compared
exactly on offline replay, including7 partial-marker frames. Its test nodes were
stopped. A separate [network-isolated NatNet fixture](../tmp/record-mode-20261013/capture-only.log)
ran the actual capture-only launcher/receiver, received408 poses, observed no
Crazyflie flight server, and exited0 after SIGINT. The first attempt exposed an
empty-map ROS YAML parser rejection; the launcher now omits that empty mapping,
and the successful rerun is retained. No actual Motive source, radio or flight
commands were used for these tests.
[GPU verification](../tmp/record-mode-20261013/gpu-results.json) exercised60 live
synthetic frames, button-driven clip creation, saving a second clip on close,
and36 exact replay frames, with seeking, step, speed and play/pause controls.
The [recording](../tmp/record-mode-20261013/recording.png) and
[playback](../tmp/record-mode-20261013/replay.png) windows were visually inspected.
CLI help, offline inspection, syntax compilation and diff whitespace checks
passed. Real hand-movement tracking accuracy and long-duration/disk-latency behavior
remain unverified. This change adds recording/presentation and capture lifecycle;
it does not change native physics, fitting, rewards, policies or firmware.

2026-10-13 cable tracking recovery correction: partial occlusion no longer resets
all material-marker identities after0.25 s. Long-missing markers use bounded
geometric recovery, with ambiguous assignments left null; full loss still needs
the hanging prior. Crowding is handled per marker after the existing root-distance
filter. Receipt-time velocity updates use the noise/speed time scale described
above to damp bunched-frame spikes. Candidate-pair distances are reused across
beam hypotheses without changing chord bounds or assignment costs.
[Focused checks](../tmp/cable-reacquisition-20261013/focused-tests.log) passed68 tests,
including long partial occlusion while bent, ambiguous returns, impossible-clutter
filtering, closely bunched timestamps and full-loss hanging reinitialization.
[Offline reassociation](../tmp/cable-reacquisition-20261013/optimized/replay-results.json)
uses all9,228 reconstruction frames from the retained real capture
[20261013-225429-55c05031](../output/custom_real/20261013-225429-55c05031/connection.jsonl),
with its saved settings and same-frame attachment poses. The
[distance-cache comparison](../tmp/cable-reacquisition-20261013/final-checks.json)
verified identical results, including confidence, on all9,228 frames before and
after caching. Measured offline per-call median/p95 were5.77/10.89 ms for the
cached version; these do not establish sustained live100-Hz throughput. Originally
the tracker lost all assignments at36.526 s after one marker became stale, and never recovered.
The revised tracker recovers that marker and retains at least one assignment on
every subsequent frame; full10-marker frames increase from1,234 to3,036, with
partial results still present. These are assignment counts, not independently
verified cable identities: in particular the later crowded/folded configuration
still leaves most markers unavailable. The saved source is unchanged. No new
Motive capture, live-throughput test, physical cable accuracy validation or flight
was performed. Native physics, controllers, training and Motive settings remain
unchanged by this correction. Syntax compilation and diff whitespace checks passed.


### 2026-10-14 — Direct acceleration policy interface

The user approved changing policy outputs to world XYZ acceleration after
reviewing the paper's acceleration-to-thrust/attitude control structure.
Implemented `acceleration_direct_v1` in the shared tensor controller, collection
adapter, native evaluation, checkpoint/run schemas, compatible actor transfer,
replay metadata and Training UI. The new Stage1–3 launch presets retain their
previous default tasks/rewards, budgets and curriculum rules, and select50-Hz
acceleration actions bounded by(8,8,4) m/s². Existing physical equations, attitude
gains,200-Hz servo,30-ms delay, motor clipping/response and original presets are
preserved. The unimplemented PVA-reference-generator design above was replaced
with the selected direct path. No firmware parameters were changed.

[Regression evidence](../tmp/acceleration-interface-20261014/regression-checks.json)
records116 passing relevant tests, including9 direct-acceleration checks for
force/units, absence of translational feedback, action scaling/delay, native/tensor
agreement, Stage1→2→3 actor transfer, checkpoint rejection/roundtrip, and UI labels.
These are software/numerical checks, not trained-policy results.
[Nominal motion checks](../tmp/acceleration-interface-20261014/motion-checks.json)
retain four4.5-s native replays: zero command holds the initial equilibrium with
zero recorded position drift; ±1 m/s² X/Y and ±0.5 m/s² Z pulses remain numerically
valid. The horizontal cases end about0.0381 m from the start and0.0202 m lower,
illustrating that zero acceleration does not restore position. A128-lane CUDA
conditional-graph check over15 policy steps/0.3 s matches all native physical,
BDF2-history, motor and held-wrench state fields within4.11e-15 maximum absolute
difference (comparison tolerance1e-8). This is one nominal initial condition,
not hardware validation or robust tracking evidence.

[Final verification](../tmp/acceleration-interface-20261014/final-verification.json)
confirms3,583 existing run/model files are byte-identical and no training/evaluation
worker is active. No training campaign or model promotion occurred. Historical
velocity checkpoints retain their semantics and saved replays; exact resume and
fresh reevaluation still enforce the existing source-hash checks, so changed
controller sources are not silently accepted. Stage4 velocity recovery and old
velocity demonstrations cannot initialize an acceleration run. The curriculum
review, compatible recovery design, live policy runner, firmware matching and
real-flight validation remain outstanding.


### 2026-10-15 — Direct whip training from a random acceleration policy

The user requested skipping the control-learning pilot and clarified that
“zero shot” meant training the whip directly from scratch. Started the visible
Training UI with the unchanged
[acceleration Stage1 preset](../configs/learning/whip_stage1_ppo_acceleration_v1.json),
creating [this run](../runs/ppo_stage1/20261015-111358-998023/).
The immutable request has `resume=null` and `initialize_actor=null`; the run has
no source actor or demonstration. Seed0 initializes a new actor/critic and
optimizers. The existing recipe uses128 collection lanes,50-Hz acceleration
requests bounded by(8,8,4) m/s², radius0.55 m,4.5-s episodes, and an80-collection
(10,240-episode) cap. Existing reward terms, failure criteria, entropy schedule
and plateau-stopping rules were not changed. No automatic Stage2/3/4 handoff is
scheduled. This is policy training in nominal simulation, not real flight.

The [launch record](../tmp/stage1-acceleration-scratch-20261015/launch.json)
confirms no active training/evaluation worker before launch and exact source-hash
agreement with the verified acceleration implementation. Original run/model
files were inventoried before launch. The worker successfully initialized CUDA
and began collection1; no improvement or learned whipping is claimed from
startup. Subsequent initial verification is retained with this launch record.


The first launch completed collection1 but failed while embedding its evaluated
replay into the best checkpoint: the acceleration interface's scalar text labels
were passed to `torch.as_tensor`. Its logs, numerical replay and initial checkpoint
are retained in the first run; it is no longer active. Fixed only checkpoint
serialization: numeric arrays remain tensors, scalar text becomes plain strings,
and resume restores both representations to NPZ arrays. This preserves safe
`weights_only=True` loading and historical tensor-only replays. The existing
source-hash guards remain strict; the replacement run starts from scratch.
[Fix regression checks](../tmp/stage1-acceleration-scratch-20261015/checkpoint-fix-checks.json)
passed47 relevant tests, including checkpoint/replay metadata and inherited-replay
roundtrips. A [roundtrip of the actual failed evaluation replay](../tmp/stage1-acceleration-scratch-20261015/serialization-check.json)
preserved every numeric and text array exactly. No physics, reward, optimizer,
action limits, curriculum or budget was changed by this fix.

The corrected [scratch run](../runs/ppo_stage1/20261015-111659-848428/) was launched through the
Training UI with the same preset and seed; its
[restart record](../tmp/stage1-acceleration-scratch-20261015/restart-launch.json)
confirms no actor initialization/resume and matching frozen settings/sources.
Initial collection, evaluation and checkpoint verification are recorded separately
from completed-training results. No later-stage launch is scheduled.


[Initial live-run verification](../tmp/stage1-acceleration-scratch-20261015/initial-verification.json)
confirms the corrected run completed collection1 (128 episodes), logged its native
evaluation, saved loadable `latest.pt` and `best.pt` with intact acceleration
metadata, and entered collection2. All3,583 pre-existing run/model files and all12
files from the failed launch remain byte-identical. These startup checks establish
that learning/evaluation/checkpointing execute; they do not establish whip quality,
convergence or real-flight readiness. The run subsequently completed; its
performance review follows.

### 2026-10-15 — Scratch acceleration Stage1 failure review

Read all80 retained collection arrays and21 native evaluation replays from
[the completed run](../runs/ppo_stage1/20261015-111659-848428/), plus its frozen
configuration, summary and event log. The
[analysis record](../tmp/stage1-acceleration-review-20261015/analysis.json)
contains per-collection/evaluation measurements and SHA256 hashes of the input
artifacts. No simulation, evaluation or training was rerun; no reward, controller,
curriculum, model selection or physical parameter was changed by this review.

The run reached its80-collection budget with13,667 actor updates and22,825 value
updates. Only267 of10,240 episodes were valid. Collection8 first reached zero
valid episodes; collections14 and28 had isolated valid episodes, and all52
collections after28 had none. Validity means the task's numerical/envelope
criteria, not workspace containment or a successful whip. The best native
evaluation remained collection1, with raw return1.6508906810486224e-6. The other20
native evaluations ended below the0.05-m cable-height limit and reported no
numerical solver failure. The final replay lasted2.01 s: drone Z was0.994344 m,
minimum cable Z was0.042056 m and maximum tilt was30.45 degrees. This was an
envelope cutoff, not evidence of a simulated contact event.

In the first four collections, mean raw return was−15.4936 and mean workspace
cost14.9796; valid episodes' mean event credit was only0.03235. The last eight
collections had zero valid episodes, mean return−1.03348 and mean workspace
cost0.03348. [Episode scoring](../tasks/whip/sac_environment.py) gives failed
episodes `−1 − workspace_cost`, cancelling positive event credit.
[Workspace cost](../tasks/whip/workspace.py) is an uncapped squared-distance
penalty accumulated only over the observed prefix, divided by the full planned
horizon. Short failures can therefore score above longer exploratory episodes
that leave the soft workspace. All10,240 saved returns reconstruct from this
formula within6.54e-6; stored transition-reward sums agree within7.63e-6
(float32 outputs).

The descent, shrinking episode durations and lower workspace charges are
consistent with learning this cheaper failure behavior. This is a supported
diagnosis, not a causal ablation or proof that it is the sole learning problem.
The plateau detector continued because containment improved while validity and
whip quality stayed zero. The recommended next step is to correct the ordering
of failure and workspace costs before another direct-from-scratch whip run.
Any new formula/weights remain proposed and require approval; no pilot,
demonstration, restart or model promotion was performed.

2026-10-15 capture-only ROS domain default: at the user's request,
`--custom-real --capture-only` now selects43 when `--domain` is omitted. Explicit
domain overrides remain supported; normal real-flight and attach defaults remain42.
The domain is resolved before viewer supervision and independently by its worker.
Checked six CLI parsing/default/override cases with supervision mocked, Python
syntax compilation and diff whitespace. No ROS stack or hardware was launched
for these checks. README launch table and hand-capture instructions were updated.

2026-10-15 Training startup null-component fix: both reward-component plotting
paths now check for a non-null value before scaling. The saved Stage1 run
`runs/ppo_stage1/20261015-111659-848428/events.jsonl` has null plotted components
at round8; these remain gaps instead of raising `int * NoneType` during startup.
Verified offscreen Training window construction and chart processing with that
actual run, its round8 gap, and an in-memory crossing-chart case containing null,
missing and numeric-zero values. The source log hash remained unchanged and no
training worker was launched. All13 resume-history tests, Python syntax compilation
and diff whitespace checks passed. The offscreen platform does not support the
OpenGL preview; GPU preview rendering and training were not rerun. Reward
calculations, saved records, checkpoints and physical models were not changed.

2026-10-15 user-requested stock-Mellinger cf_2 acceleration circle: added the
standalone Crazyswarm script and usage instructions described above. All motion
phases use host-feedback acceleration through FullState with translation gains/
integral ranges zeroed and original values restored after confirmed disarm.
[Nine focused checks](../tmp/acceleration-circle-cf2-20261015/unit-tests.log) passed:
reference derivatives/continuity, saturation/anti-windup, normal lifecycle,
read-only preflight, partial parameter-write rollback, unknown arm outcome,
streaming failure, withheld restoration after unconfirmed disarm, and loop stalls.
The [network-isolated ROS fixture](../tmp/acceleration-circle-cf2-20261015/ros-results.json)
used actual ROS message/service types with synthetic OptiTrack/status, a fake
parameter server and an ideal acceleration actuator; it received1,665 FullState
commands, verified fixed P/zero V/zero body-rate fields, observed16 parameter
writes, and confirmed disarm and restoration. The read-only check sent no commands
or parameter writes. The disposable container had no external network or radio;
its synthetic plant is not the native cable/drone physics. CLI help in the existing
container, trajectory dry-run, syntax compilation and diff whitespace checks
passed. No real-flight execution, parameter changes on hardware, calibration,
firmware verification, native simulation validation or policy training occurred.

2026-10-15 default Docker client shell: at the user's request, plain
`./docker/shell.sh` now opens custom-simulation domain142 with
`AEROWHIP_SIMULATION=1`. Explicit `--real` and `--sim` select domain42 and disable
the custom client flag; `--custom-sim [domain]` remains supported. Existing shells
must be reopened; no running process or container-wide environment was changed.
Eight shell argument-routing/rejection checks passed with Docker mocked, along
with Bash syntax and diff whitespace checks. README real/stock examples now use
explicit shell modes. No ROS commands, simulations or real flights were executed
for this change.


2026-10-15 custom-simulation acceleration circle: added the explicit disarmed-only
ROS acceleration selector and adapted the adjacent cf_2 circle script to use
native feedback and simulation time. Default PVA behavior, the physical model,
training/reward settings and real firmware parameters were not changed. The
[22 native/connection checks](../tmp/acceleration-simulation-20261015/native-tests.log)
and [15 script checks](../tmp/acceleration-simulation-20261015/script-tests.log)
passed, including identical default-PVA native evolution, identical direct-mode
motion despite different P/V placeholders, transition-delay rejection, reset,
checkpoint incompatibility, clock/session failures and mode cleanup sequencing.
The [retained integration verification](../tmp/acceleration-simulation-20261015/verification.json)
records a complete fresh YAML-origin circle on isolated ROS domain225 with1,496
FullState commands, fixed P/zero V/zero body rates/fixed yaw, no command/API
rejections, and restored PVA. The unchanged
[waypoint script then completed](../tmp/acceleration-simulation-20261015/waypoint-after-circle.log)
in the same session. Native bridge/command records are retained in
[final-native-ros](../tmp/acceleration-simulation-20261015/final-native-ros/).
The script targets50 Hz but asynchronous ROS/native clock pacing yielded about
44 Hz average in this run; this is not a hard-real-time timing claim.
An earlier test exposed a final topic packet arriving after service disarm; the
final script drains scheduled stream receipts before disarming. An additional
circle after a waypoint maneuver failed the unchanged settling limits with
residual drone/cable oscillation and suspended the simulation; that failed run
is preserved in adjacent Crazyswarm `logs/acceleration-simulation-20261015-repeat`.
Stationary drone preflight does not certify settled cable state. No acceptance
limits were relaxed. Syntax and whitespace checks passed. These are nominal
numerical/transport/script checks, not model fitting, policy-training results,
firmware equivalence, contact validation or real-flight evidence. Only owned
isolated test sessions were closed; the user's existing simulator was untouched.


2026-10-16 GitHub source and data snapshot: preserved the two-project layout
and existing AeroWhip branch history. Source, configuration, trajectory CSVs and
packaged model artifacts are tracked in the two private repositories. At the
user's explicit request, recordings, training runs and retained test evidence
are also included as release archives under `backup-2026-10-16`; they remain
ignored by normal Git commits and require a separate download/restore. Archive
manifests identify individual files and SHA-256 hashes, and source stability and
full decompression checks verify the snapshot before upload. Successful, failed
and exploratory runs are retained without model promotion. Crazyswarm's `.env`
contains only public upstream commit pins and remains part of its source.
Virtual environments, build/cache files, publication tooling and credentials
are excluded. Both READMEs describe the checkout layout and data restoration.
This is packaging/documentation work; no simulations, fitting, training or
hardware experiments were rerun. Upload verification is retained locally under
`tmp/github-upload-20261016/` after publication.
