# Current implementation

Last reviewed: 2026-10-27.

## Active training and handoff

**Training is stopped.** Only the four-second +X Stage 1 PPO recipe remains supported.
Read [the complete Stage 1 specification](../docs/fold_transport_task.md) for the reward, fold tracking,
termination, control, exploration, evidence and limitations. The default is
[the v6 experiment file](../configs/experiments/stage1_fold_x_v6.json):
64 lanes, 200 collections, 12,800 episodes, ten physical links, fresh actor/critic.
The latest production run `20261025-215750-176045` completed zero collections and
zero optimizer updates. Its stopped checkpoint is not a learned policy. Its saved
task snapshot uses the pre-2026-10-26 task layout, so current code no longer opens
it; use commit `375b44e` to inspect it.

There is no target-hit reward, early completion, automatic Stage 2 or hardware deployment.
Current reward and physics are unchanged by the cleanup. Old reward modes, Stage 2–4
training, SAC and historical actor-transfer paths have been removed with explicit user
approval; old checkpoint execution/replay compatibility is not promised.

[Other-computer setup and restoration](handoff-20261025/README.md).
The full previous implementation record and historical model artifacts remain in
[Git commit 3c9a49e](https://github.com/TongshuWu1/AeroWhip_deploy/tree/3c9a49e).
Historical local run links in the retained physical/fitting notes below require the data
release and describe their original experiments, not a new verification.

## Pipeline boundaries

All Python code is in the `aerowhip/` package; [docs/architecture.md](../docs/architecture.md)
lists its subpackages and their dependencies. The C++ engine stays in `native/` and builds into
`aerowhip/physics/`. Entry points are in `scripts/`; `aerowhip.paths.ROOT` is the repository
root. The simulation, fitting, recorded-flight and Crazyswarm interfaces were retained; the
separate standalone simulation is not the ten-link training model.

## Physical model, units, and timing

World XYZ has Z up. Rotation maps body to world; angular velocity, torque, and
inertia use the body frame/about COM. Public drone position/velocity refer to
the tracked top-plate origin. Quaternions use XYZW. Cable nodes, including
attachment node 0, use world coordinates.

| Quantity | Default |
|---|---|
| Drone mass excluding cable | 0.145 kg |
| Cable and moving-marker mass (total) | 0.017 kg; markers 10 × 0.96 g (unchanged), bare cable 7.4 g |
| Cable length / mesh | 0.995 m / 11 nodes, 10 edges (0.095 m, then 9 × 0.1 m) |
| Tracking-to-COM / attachment offset | (0,0,-0.018) / (0,0,-0.051) m |
| Maximum physics step | 1/600 s |
| Observation / controller / policy rate | 100 / 200 / 50 Hz for current training |
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
This checkout's deployment/training solvers have no cable self-collision,
ground/wall contact, or measured wind. The separate standalone coupled simulator
now has experimental cable-only self-contact, described below.
Degenerate edges and antiparallel tangent singularities are rejected.
A pre-contact standalone 20-edge diagnostic with 0.5 m X-amplitude Oscillation,
1.5 s cycle and 4 s startup produces self-overlap from about 4.77 s, then fails
implicit endpoint convergence after the last accepted 20.85 s at a 1/600 s
maximum step. An in-memory 1/1200 s variant reaches 30 s but still self-overlaps;
no production default was changed by that diagnostic. The subsequently approved
standalone contact/retry change completes this 30 s case, but its folded-motion
prediction remains sensitive to timestep/barrier choices and physically unvalidated.
Earlier non-contact trajectory checks do not validate contact. See
[fold reproduction](../tmp/cable-fold-review-20261025/oscillation-1p5-reproduction.json)
and [step-size diagnostic](../tmp/cable-fold-review-20261025/timestep-1200-summary.json).

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
[conditional driver](../aerowhip/physics/conditional_driver.py) loads the Windows driver
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

The [viewer](../aerowhip/viewer/pyvista_viewer.py) renders through NVIDIA OpenGL. The prior
1440×900 off-screen display-motion baseline averaged2.159 ms per geometry update
and render with vertical-sync waiting disabled; default synchronization averaged
17.084 ms. Those timings precede the viewer rebuild and are not measurements of
the new complete UI.

### Local simulation execution and display

[viewer/sources.py](../aerowhip/viewer/sources.py) now gives the local native simulation
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

[viewer/display_state.py](../aerowhip/viewer/display_state.py) defines a read-only world
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

[viewer/robot_scene.py](../aerowhip/viewer/robot_scene.py) retains CAD, rotor, cable-tube,
marker and overlay actors per named robot, updating transforms/coordinates rather
than recreating meshes per frame. Robot addition/removal or cable-topology changes
rebuild only that robot's actors. Multiple robots and absent cables are supported.
CAD name plates now show display identity instead of a hardcoded drone number;
this does not identify or calibrate real hardware. Floor/grid contrast changed
only in the renderer; there is still no physical floor/contact model.

The [docked flight window](../aerowhip/viewer/simulation_window.py) has a central3D view,
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
[launcher](../scripts/real_flight.py), [host feed](../aerowhip/deployment/real_flight.py),
[ROS reader](../aerowhip/deployment/ros_real_bridge.py) and
[stock-stack supervisor](../aerowhip/deployment/real_stack.py) implement that scope.
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
The [live cable boundary driver](../aerowhip/physics/live_cable.py) reuses the existing
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
selected robot, using [deployment/cable_tracking.py](../aerowhip/deployment/cable_tracking.py).
It does not feed the native cable integrator, controller, fitting or policy.
The selected model's cumulative `marker_interval_lengths_m` define material marker
arc distances; the user confirmed the real cable has this spacing. Current
distances are 0.095, 0.195, 0.295, 0.395, 0.495, 0.595, 0.695, 0.795, 0.895, 0.995 m
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

[viewer/cable_tracking_scene.py](../aerowhip/viewer/cable_tracking_scene.py) renders separate
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
[deployment/real_stack.py](../aerowhip/deployment/real_stack.py) builds capture parameters
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

[viewer/recording.py](../aerowhip/viewer/recording.py) serializes the complete immutable
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

[run_recording.py](../scripts/play_recording.py) opens a folder/manifest or a file picker
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
[Stack lifecycle handling](../aerowhip/deployment/stack_lifecycle.py) checks PID start
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

[deployment/custom_simulation.py](../aerowhip/deployment/custom_simulation.py) adapts the
existing native owner thread without modifying its plant. PVA remains the default;
an explicit ROS-only acceleration selector reuses the sampled nominal controller
with zero translation gains while armed.
The host `.venv` owns native physics and GPU visualization; the existing Docker
ROS Python runs [deployment/ros_bridge.py](../aerowhip/deployment/ros_bridge.py). The
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
GPU. Only one drone/cable plant is exposed. [Configuration selection](../aerowhip/deployment/configuration.py)
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

[Flight command arbitration](../aerowhip/deployment/flight_commands.py) supplies functional
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
[controller wrapper](../aerowhip/deployment/acceleration_control.py), default false. It
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

The separate [PVA test](../aerowhip/deployment/ros_test.py) uses the upstream
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

[estimate_clock_alignment.py](../aerowhip/takes/estimate_clock_alignment.py)
writes the `<batch>/time_alignment.json` entry that `load_take` requires
before commands are attached. The entry is keyed by take stem and records
`offset_s` such that logger `time_s` = Motive `Time (Seconds)` + `offset_s`, plus
the export and logger SHA-256s. It searches offsets on a 10 ms grid within
`--max-lag` (default ±30 s) of aligned starts and refines to 0.1 ms. The score is
the RMS distance between Motive drone positions and logger poses linearly
interpolated at the shifted times; no interpolation spans a logger gap over 50 ms,
and source data are not modified. Because the logger stamps receipt time, the
offset includes transport latency. Entries are saved with `clock_verified: false`
and the method, residuals and acceptance settings. Nothing is written if the RMS
exceeds `--max-rms` (0.01 m), if an offset more than 0.25 s away scores within
2× the best RMS (for example a take with no motion; start each take with a sharp
vertical bump), or if the best offset is on the search boundary. Existing entries
need `--replace`. Comparing positions directly assumes the export and the logger's
`/poses` stream share a frame (Z up); a mismatch fails the RMS check rather than
being corrected. These thresholds are initial engineering settings. On 2026-10-26
the script was checked only on synthetic files, which recovered a known offset
plus the injected 0–4 ms latency to within 1 ms. It has not been run on a
recorded take.

[check_take.py](../aerowhip/takes/check_take.py) is a read-only post-capture
check of one Motive export. It reports:

- export units, coordinate space and rotation type
- capture and export rates, required to be exactly 100 Hz
- frame-number jumps and off-nominal (non-10 ms) steps
- drone, cable, unlabeled and rigid-body-marker labels
- per-marker completeness and longest dropout
- the hanging-cable direction, required to be −Z
- complete, gap-free 100 Hz runs of at least `--min-window` (default 1.8 s =
  1.7 s fit window + 0.1 s history)
- whether `load_take` accepts the file and an alignment is applied
- which logger sidecars are present

It exits 1 on a blocking problem; a missing alignment is a warning. It does not
assess command logs, motion content or fit quality. On 2026-10-26 it was also
checked only on synthetic files.

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
[load_take](../aerowhip/takes/take_data.py) currently expects a drone identity
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

[fitting/run_refit.py](../aerowhip/fitting/run_refit.py) requires an explicit base model.
It fits cable EI/Cb/external drag with measured attachment motion, then vehicle
controller response scales with cable/motors frozen in the coupled plant. Default
selection is four training and thirteen historically exposed comparison takes.
`--data`, `--split`, and `--prepare-only` select/record inputs. Candidates and
comparisons are saved under `runs/refit` without automatic default installation.

[fitting/run_residual_fit.py](../aerowhip/fitting/run_residual_fit.py) separately fits
structured nonnegative aerodynamic coefficients: cable first, then drone with
cable weights frozen. It requires a calibrated physical base and gradient gates.
The active residual remains unfitted/disabled. Low fitting loss does not establish
identifiability or reliable whole-maneuver cable prediction.


## Dated change note — 2026-10-25 cleanup

Explicitly retired old training and historical checkpoint compatibility. Kept the current
Stage 1 contract, shared simulation and data tools. Replaced contradictory historical
training descriptions with `stage1.md`; original notes remain in Git. Detailed deletion,
verification and archive accounting is in [cleanup report](handoff-20261025/cleanup-report.md).
Software and short numerical checks are not learned-policy or real-flight validation.

## Dated change note — 2026-10-26 branch consolidation

Merged commit 546188e (rigid-body marker exclusion, tag capture-update-2026-10-16) into main; the develop branch was removed. Live Motive validation, partial initialization and simulated-cable failure behavior remain pending. The detailed 2026-10-16 note is in commit 546188e.

## Dated change note — 2026-10-26 take checks

Added `take_processing/estimate_clock_alignment.py` (writes estimated, unverified
`time_alignment.json` entries) and `take_processing/check_take.py` (read-only
export check), with synthetic tests in `tests/test_take_checks.py`. `load_take`,
fitting, splits and physical models are unchanged. Neither script has been run on
a recorded take: the `backup-2026-10-16` recordings archive contains no flight
takes. Thresholds remain to be confirmed on the first refit-collection takes.

## Dated change note — 2026-10-26 reorganization phase 1 (retired code removed)

Removed code and configuration that no current path used, on branch `reorganize`:
- Task file sections `mechanics`, `matching`, `strike_surface`, `request` and the
  unused `return` fields (`acceleration_cost_*`, `failure_return`), `episode.settling_s`
  and `episode.termination`; learning keys `algorithm`, `training_stage` and
  `ppo.transfer_entropy_clock`. The task file now has `episode`, `workspace`, `return`
  and `continuous_whip` only.
- The three-level environment chain (`sac_environment.py`, `horizontal_fold_environment.py`,
  `continuous_whip_environment.py`) is one class in `tasks/whip/environment.py`; branches
  for physical/workspace termination, velocity actions and radial output were never
  taken by the current task and are gone.
- PPO actor-transfer entropy offsets and unused phase/goal encoder inputs; the
  `model_catalog.py` packaged-model support (no `models/` folder exists),
  `evaluation/worker.py`, `simulation/batched_rollout.py`, SAC/historical branches in
  `evaluation/records.py`, the replay viewer's strike/target cards, and the policy
  tester's display-only target field (protocol now `initial_state_policy_test_v1`).
- The offline event review's virtual board (1.1 m along +X, 1 × 0.6 m at z = 2 m) now
  lives in `evaluation/whip_event_review.review_surface`, with unchanged geometry.
- Whipping diagnostics now use only the fixed world +X output, as the reward does.

Checked: `tools/equivalence_baseline.py` (fixed open-loop actions through four CPU
lanes and four CUDA lanes, plus one seeded PPO collection and update) reproduces all
154 recorded tensors exactly against the reference recorded at `375b44e`; 231 unit
tests pass (five tests of removed features were deleted). Physics, reward, controller,
timing and PPO arithmetic are unchanged. Runs saved before this change no longer load.
Not checked: the GUIs beyond offscreen construction; no training or flight was run.

## Dated change note — 2026-10-26 reorganization phase 2 (one experiment file)

- One experiment file, [configs/experiments/stage1_fold_x_v6.json](../configs/experiments/stage1_fold_x_v6.json),
  replaces the separate learning preset and task file. It holds the run, control and PPO
  settings and an inline `task` section. The physical model (`configs/simulation.json`) and
  scene stay separate and are referenced by path, because simulation, fitting and deployment
  share them. Run snapshots keep `settings` and `task` as before.
- [tasks/whip/configuration.py](../aerowhip/tasks/whip/configuration.py) defines the typed
  `Stage1TaskConfig`. The reward (`continuous_whip.py`) and environment read the horizon,
  workspace, workspace-cost weight, envelope limits, failure cost, fold-support windows and
  thresholds, rate weight and powers from it. The descriptive `continuous_whip` contract
  block and its repeated literals are gone.
- Validation checks required keys, types and ranges instead of exact approved values.
  Unimplemented choices fail explicitly: only the `continuous_directional_whip_four_second_v6`
  reward, world +X output, the acceleration interface, a 100 Hz observation clock, policy
  rates dividing 100 Hz and the ten-link mesh are supported. Changing a value in the file
  now defines a new experiment; the approved values are the file's contents.
- The duplicate workspace-penalty implementation used only for logged metrics was removed;
  the logged `workspace_cost` is now the reward's own C. Its values can differ from earlier
  logs in the last floating-point digits.
- PPO no longer checks task contract names; its history length comes from the environment
  (`history_samples` in the saved interface).
- The Training window's description is generated from the experiment values.
- Fixed a pre-existing settings-dialog defect: pressing OK clamped `seed` to 10,000,000 and
  rounded `initial_log_std` to nine decimals even when untouched. Untouched fields now keep
  their exact stored values (checked offscreen).

Checked: the equivalence check reproduces all 154 reference tensors exactly (CPU and CUDA
rollouts, one seeded PPO update); 231 unit tests pass, including new rejection tests for
invalid task and run values. No training, simulation campaign or flight was run.

## Dated change note — 2026-10-26 reorganization phase 3 (layering)

- Experiment files now have sections `model`, `scene`, `run`, `environment`, `task`, `ppo`
  ([experiment.py](../aerowhip/experiment.py)); each section is a typed dataclass next to its consumer.
  Horizon, policy rate, acceleration limits, workspace and the tilt/cable-height limits moved
  from the task to `environment`; the task section holds reward settings only and is selected
  by `task.name` through the registry in [tasks/__init__.py](../aerowhip/tasks/__init__.py).
- [envs/environment.py](../aerowhip/envs/environment.py) (`PlantEnvironment`) is task-independent:
  it calls a `Task` ([envs/task.py](../aerowhip/envs/task.py)) at every 100 Hz sample and on numerical
  failure. The fold-transport reward is `FoldTransportTask` in
  [tasks/whip/fold_transport.py](../aerowhip/tasks/whip/fold_transport.py). Fold geometry is one module.
- PPO, the worker, checkpoints and run folders are task-independent (`runs/ppo/`, checkpoint
  schema `aerowhip_ppo_checkpoint_v1`, run snapshots `aerowhip_run_v1` with an `experiment`
  entry). Replay files store the task trace as `task_trace`/`task_trace_fields`/`task_record_json`.
- Result components `time_cost` and `return_transform_adjustment` (always zero) and
  `failure_adjustment` (equal to −`failure_cost`) were removed.
- CUDA solver-graph helpers moved from `tasks/whip/` to `simulation/` with one entry point,
  `simulation.iteration_graph.accelerate`.
- Tests were reduced to a core science suite run with pytest (74 test functions, 90 cases):
  new exact physics checks (free fall, zero-command hover equilibrium, momentum conservation of
  a bending cable), reward counterexamples and accounting, environment validity and failure
  handling, experiment validation, PPO mathematics and restart, whip-measurement invariances and
  take checks. UI-widget, ROS-mock and file-plumbing tests were removed.

Checked: the equivalence check reproduces all 148 remaining reference tensors exactly (CPU and
CUDA rollouts, one seeded PPO update); the pytest suite passes; the GUIs construct offscreen and
the settings dialog round-trips the experiment unchanged; a two-lane CPU training run completed
one collection, one PPO update and one evaluation. Runs saved before this change do not load.

## Dated change note — 2026-10-26 reorganization phase 4 (package and documentation)

- Code moved into one package, `aerowhip/`: `simulation` → `physics`, controllers → `control`
  (windows → `ui/controller`), `training` → `learning` (windows → `ui/training`), `take_processing`
  → `takes` (windows → `ui/takes`), fitting windows → `ui/fitting`, the policy tester →
  `ui/policy_tester`, plus `envs`, `tasks`, `evaluation`, `fitting`, `deployment`, `viewer`,
  `trajectories`, `experiment.py` and `paths.py`. Entry points moved to `scripts/` with plain names
  (`train.py`, `replay.py`, `test_policy.py`, `simulate.py`, `custom_simulation.py`, `real_flight.py`,
  `play_recording.py`, `take_processing.py`, `controller.py`, `controller_headless.py`, `gamepad.py`,
  `build_engine.py`, `equivalence_check.py`). Scripts put the repository root on `sys.path`, so they
  run from a checkout; `pip install -e .` installs the `aerowhip` distribution.
- Native extensions are now `aerowhip.physics._native` and `aerowhip.physics._tensor_native`;
  source hashes, subprocess module names (`aerowhip.learning.worker`,
  `aerowhip.evaluation.policy_test`, `aerowhip.fitting.run_fit`) and the Docker launcher paths in
  the companion repository's `docker/start.sh` (left uncommitted there) were updated.
  `run_residual_fit` now also hashes `.so` binaries, as `run_fit` already did.
- Public documentation is `README.md` and `docs/`; this record, the 2026-10-25 handoff notes and the
  publication procedure ([PUBLICATION.md](PUBLICATION.md)) live in `internal/`, which the publication
  branch omits. The labelled side-swing example moved to `data/examples/`.

Checked: the equivalence check reproduces all 148 reference tensors exactly on CPU and CUDA in the
new layout (with the binaries moved, not rebuilt); a clean build of both extensions under the new
module names succeeds in a scratch copy and imports; pytest passes (90); every script runs `--help`
from an unrelated directory (except `gamepad.py`, which needs tkinter, not installed here); the
training window, settings dialog, policy tester, replay browser and take manager construct offscreen;
a two-lane CPU run of `python -m aerowhip.learning.worker` completes. Not checked: hardware, the
Docker launchers end to end, and Windows.

## Dated change note — 2026-10-27 new cable

The old cable is retired; the model describes the new one (user measurements: 16 g with
markers, same material and 3.5 mm diameter, first marker 0.095 m from the attachment, then
every 0.1 m, 1 m long).

- `configs/physics/drone_cable.json`: `total_mass_kg` 0.017 → 0.016 (whole cable including
  markers; the engine subtracts `moving_marker_masses_kg`, kept at 10 × 0.96 g, so the bare
  cable is 6.4 g — the marker/bare split is carried over, not re-measured);
  `marker_interval_lengths_m` → 0.095 then 9 × 0.1 (0.995 m).
- `configs/scenes/default.toml`, `m3.toml`: `cable_segment_lengths_m` to match.
- `configs/actuators/drone.json`: `initial_thrust_n` is the hover thrust of drone + cable,
  (0.145 + 0.016) kg × g / 4 = 0.39472 N per motor (was 0.39717 for 17 g). Without this the
  motors start 0.6 % high and a zero-acceleration hover drifts up (caught by
  `test_zero_acceleration_command_holds_hanging_cable_at_rest`). Maximum thrust unchanged.
- EI, Cb, GJ, Ct and drag unchanged (same material; still the earlier working assumptions).
- Correction to the 2026-10-26 discussion: the old model's cable was 17 g in total, not
  26.6 g; `total_mass_kg` already included the markers.
- The fold bandwidth formula `0.075 L / 0.9525` is unchanged and now gives 0.0783 m.

Checked: 90 tests pass; native and tensor engines hold a hanging cable at rest under a zero
command (drift 0); native edge lengths 0.995 m, node masses 0.016 kg. The equivalence
reference was re-recorded with the new model (148 tensors, CPU and CUDA, match); the old
reference is not kept in the repository. Not checked: any training run, fit or replay with
the new cable; the 2026-10-26 flights were flown with the old cable.

## Dated change note — 2026-10-27 new cable re-weighed

The user re-weighed the new cable: 17 g with markers, not 16 g. `total_mass_kg` 0.016 → 0.017
(bare cable 7.4 g with the unchanged 10 × 0.96 g markers) and `initial_thrust_n` back to
(0.145 + 0.017) kg × g / 4 = 0.39717 N per motor. Geometry unchanged. Checked: tests, the
hanging-cable rest check on both engines; equivalence reference re-recorded.

