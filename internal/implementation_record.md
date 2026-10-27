# Current implementation

Last reviewed: 2026-10-28.

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
| Tracking-to-COM / attachment offset | (0,0,-0.018) / (0,0,-0.0423) m |
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

## Dated change note — 2026-10-27 cable labels from the marker cloud

The companion flight script records every OptiTrack marker cloud during a cable flight
(`cloud.jsonl`). New `aerowhip/deployment/cable_filter.py` is the cable-state filter meant
to feed the policy live (positions, velocities, per-marker confidence, cable quality): a
predicted chain state, gated association, cable-length chain check, re-acquisition, and a
guard against OptiTrack's bunched stamps (steps of 0.1 ms followed by 10 ms would otherwise
produce 50 m/s velocities). `aerowhip/takes/cable_labels.py` runs it over a recorded cloud
and reconstructs hidden markers into a separate file with a method per position;
`aerowhip/ui/takes/cable_review.py` is a 3D review window; `scripts/label_cable.py` runs
either. Decisions with the user: partial frames are valid data and ten markers are never
forced; reconstruction is never saved as a measurement; the earlier beam-search tracker
(`deployment/cable_tracking.py`) stays for the live display but no longer labels data.

Checked on all ten 2026-10-27 cable flights, judged by the raw cable geometry (consecutive
labelled markers at the wrong distance by > 2 cm): filter 0.0–0.9 % of frames per flight,
the earlier tracker 0.0–4.5 %; 0.13 ms per frame against 2.2 ms. Reconstruction accuracy on
hidden known markers: time gaps 1–2 mm, interior 5–10 mm, free end 10–20 mm median. Report:
companion `calibration_data/LABELLING_REPORT.md`. Found: every marker sits 8–12 mm closer to
the modelled attachment than its arc distance (attachment offset or first-marker distance
off by ~9 mm; model unchanged). Not done: live ROS node (cloud/pose matching by content
with the Crazyflie's own position as fallback, as the user asked), the window on screen,
fitting tools reading these files.

## Dated change note — 2026-10-27 attachment offset from the hanging cable

In every one of the ten 2026-10-27 cable flights, in still hover frames, the first cable
marker hangs 137.3 ± 1.1 mm below the tracking origin along the drone's axis (sideways
within ±6 mm), where the model gave 146 mm (attachment −0.051 m plus 0.095 m of cable). The
user chose to correct the model: `attachment.position_tracking_m` −0.051 → −0.0423 m in
`configs/physics/drone_cable.json`, both scenes, and the replay window (which now reads
the scene value). The marker spacing along the cable (0.095, then 0.1 m) is unchanged,
measured on the cable. Checked: tests. The equivalence reference could NOT be re-recorded:
`scripts/equivalence_check.py --record` crashed (core dump) on this machine, one of several
unexplained crashes that night (container Python, loginctl, PyCharm twice); the saved
reference still corresponds to the model before this change and the check will report
differences until it is re-recorded on a healthy machine.

## Dated change note — 2026-10-27 first fits on the calibration flights

New `aerowhip/takes/calibration_flights.py` loads a calibration flight folder (50 Hz host
log, OptiTrack poses on their true stamps, cable labels with gaps) and cuts gap-free 2 s
windows per log phase, with the acceleration command in force at each frame (zero-order
hold, trims stripped) and a seen-mask per marker. OptiTrack ran at 99.5 Hz with bunched
stamps (median step 9.65 ms); windows carry true stamps. New
`aerowhip/fitting/calibration_cable.py`: measured attachment (true stamps, first-knot
velocity from the initial state) drives the native prescribed cable; loss = RMS over seen
markers; multipliers on EI, Cb, drag. New `aerowhip/fitting/calibration_drone.py`: native
coupled runtime in direct-acceleration mode (kp = kv = 0, negligible cable) driven by the
sent commands on drone-only flights; parameters: acceleration gain, attitude-frequency
scale, extra delay. The old Motive-CSV take path is untouched.

Indicative results (this machine was faulting; see below): on 33 windows from six cable
flights a 6×6 grid gives Cb ×4, drag ×4 at 37 mm RMS against 63 mm for the base; EI has no
effect; on a drone-only 8 m/s² line window the base drone model is 1.1 m RMS off over 2 s
(the delivered acceleration is ~0.85 of the command and the model has no such gain).

Machine fault: 33 kernel-logged crashes in 24 h (libusb ×15, python ×5, CUDA ×2, systemd,
the desktop app) at different addresses inside the interpreter binary, after runs
complete; no OOM, temperatures normal. Treated as a hardware (RAM) fault: the equivalence
reference was not re-recorded, and the fit numbers above must be reproduced on a healthy
machine before use. `scripts/equivalence_check.py` has an untested fresh-process PPO
variant in `git stash`.

## Dated change note — 2026-10-27 drone response: what the data says

`calibration_drone.window_loss` now feeds the simulated drone the full sent command
(trims included) minus the vertical mass trim, at the effective mass 0.157 kg, so the
simulated and real drones receive the same lateral command and hover at the same thrust.
On a drone-only 8 m/s² line window (`line 45 8`), with the initial state matching to
0.1 mm and 0.01 m/s: in the first 0.3 s the command averages 4.9 m/s², the simulated
drone delivers 5.1 (104 %), the real one 3.9 (80 %). Tilt: measured lag behind the
commanded tilt 110 ms, simulated 90 ms; measured tilt is 0.81 of the commanded, simulated
0.93. The missing effect is a tilt shortfall of ~15 %, not a delay; neither an
acceleration gain nor the attitude frequency alone reproduces it (best joint 655 mm over
2 s against 1058 base). A model decision is needed (tilt gain / attitude limit).

Both retries of the equivalence reference re-record segfaulted (one at interpreter start);
a cable grid rerun segfaulted in the garbage collector during JSON parsing. The machine
is faulting; a memory test is the next step before further computation. The cable grid's
first run (Cb ×4, drag ×4, 37 mm vs 63 mm) stands as indicative only.

## Dated change note — 2026-10-27 motor thrust-stand data

The user's thrust-stand ramp tests of one cf_2 motor+propeller (2026-10-26, four usable
battery voltages 7.2–8.1 V, one discarded run) are saved in the companion repository under
`calibration_data/bench/motor_ramp_2026-10-26/` with a README and `summary.json`. Findings:
thrust per RPM² 6.8–7.0e-10 N/rpm² at all voltages; thrust at a fixed ESC command falls ~20 %
per volt (motor speed, not the propeller); hover at the effective 0.157 kg needs 0.385 N per
motor, 1400 µs at 8.1 V and ~1570 µs at 7.5 V. The Mellinger firmware maps force to PWM
open-loop (`massThrust` = 17700, no voltage compensation), which explains the ~9 % thrust
shortfall and its drift with battery. Model decision pending: thrust scaled by logged battery
voltage in the drone block (replacing the constant effective mass). The motor time constant
is not resolvable from these tests. `configs/actuators/drone.json` is unchanged.


## Dated change note — 2026-10-27 user interfaces reworked

Every window now uses one shared style: `aerowhip/ui/style.py` (colour-blind-safe palette in
which each colour has one meaning: measured, reference, command, model, error, x/y/z) and
`aerowhip/ui/plotting.py` (matplotlib canvas with a playback cursor and click-to-seek).
Hand-painted charts are replaced by matplotlib figures with SI units:
- training monitor: per-collection return median with a 10–90 % band, outcomes, return
  decomposition, policy KL against its target, critic explained variance, entropy;
- saved-motion replays: commanded vs achieved acceleration and tilt, tip velocity, motor
  thrusts (`aerowhip/ui/motion_plots.py`);
- take manager: position histories, a tracking-error panel with interval RMS, and a
  per-marker visibility raster (partial frames are normal);
- cable review: per-marker labels and chain-length error over flight time;
- flight viewer: a rolling time-history dock (`aerowhip/viewer/time_histories.py`);
- cable-fit dialog: a convergence plot.

Behaviour is unchanged except that the policy tester's Run stays disabled until a
checkpoint is loaded. Checked visually on DISPLAY=:1 (screenshots and the data used are in
`internal/gui_review.md` and `internal/gui_screenshots/`); live real flight was not tested.

## Dated change note — 2026-10-28 engine v2 (branch `engine-v2`, not merged)

Decided with the user on 2026-10-27 night, implemented overnight. Status: the native engine,
calibration replays and fits are done and checked; training still uses the tensor engine on
`main`, which cannot load the v2 configuration, so `engine-v2` is not merged (see "Open").

**Cable (`native/engine.cpp`, `material_frame.inc`, `prescribed_material.inc`).** Twist is
quasi-static: an isotropic rod with a free end carries no twisting moment, so the material
frame is the frame parallel-transported from the clamp and the material angles are no longer
state (packed state 18+6m, `der_twist_free_kelvin_voigt_v2`; `GJ_n_m2`/`Ct_n_m2_s` are
rejected by the loader). Kelvin–Voigt bending damping acts on the material curvature in that
frame (objective). Checked: planar bending (horizontal release, quarter-circle release) is
identical to v1 to 0.00 mm over 3 s; a helical release differs by 46 mm RMS (v1's twist
coupling, which we have no data to support). Steps are 25–40 % faster than v1. Air: quadratic
normal drag ½ρC_d d ℓ|v_n|v_n per node (`diameter_m`, `normal_drag_coefficient`,
`simulation.air_density_kg_m3`) plus the linear term.

**Drone (`native/flight_controller.hpp`).** The real pipeline below the host: firmware thrust
vector F = m_fw(a_sent + g), m_fw = 0.144 kg; attitude PD towards F/|F|; firmware thrust
T_cmd = F·z_body mapped to delivered thrust T = κ(1+β(V−V₀))T_cmd^p. Identified directly on
the drone-only flights (`configs/controllers/crazyflie_flight.toml`): p = 2.2, κ = 0.553,
β = 0.177/V, V₀ = 7.2 V; residual 57 mN on fitting flights, 41–81 mN held out, against
120–190 mN for a linear map per flight. The attitude loop fitted on the measured body z-axis:
ω = 19–20.5 rad/s, ζ = 1.2–1.4, delay 3 ms (2.4–3.0° RMS against 3.3–8.6° for "achieved =
commanded"). The earlier "tilt factor 0.81" and "110 ms lag" are this overdamped attitude
response plus the thrust map, not separate effects. `maximum_thrust_n` raised to 0.85 N per
motor (peak measured collective 3.39 N). A constant world acceleration bias per lane is
available for randomization (the replay residual is a slow ~0.1 m/s² drift the real host PID
absorbs).

**Runtime.** `BatchRuntime`: independent lanes with their own `FlightController` (voltage,
bias), stepped with OpenMP, GIL released; lanes reproduce a single `Runtime` bit for bit
(`tests/test_native_physics.py`, with free fall, hanging hold and momentum checks: all pass).

**Calibration set and replays (`aerowhip/fitting/calibration_set.py`, `flight_cable.py`,
`flight_drone.py`).** Windows of 2 s every 1 s inside the usable spans of the take log;
`random_high` and `whip` held out of every fit. Drone-only `strokes` is cut at t = 104.1 s
(battery 6.78 V under load, link loss). Cable replay: measured clamp motion, cable from the
measured markers, hidden markers skipped. Drone replay: sent commands (sideways trims removed)
through the flight controller, open loop; loss raw and debiased (per-window constant
acceleration removed).

**Fits (`runs/calibration_2026-10-27/`, each number computed twice, reproduced).**
- Drone (`drone_fit.json`, Nelder–Mead on attitude ω, ζ, motor τ, delay): 43.2 mm debiased
  on the fitting flights (237 raw) but 57.6 mm held out, against 52–53 mm held out for both
  the component fits and the previous simulator gains. Not adopted: the 2 s open-loop
  position error is dominated by the drift, and the attitude/thrust identification above is
  the better-conditioned evidence.
- Cable v2 on the live-filter labels (`cable_fit_v2.json`): 31.1 mm fitting, 65.2 mm held
  out (whip 83 mm). Superseded by the relabelling below; to be rerun.

**Labels.** The user found wrong labels (c8–c10) in the whip snap: the causal filter lost the
folding cable. New offline labeller `aerowhip/takes/sequence_labels.py`: per-frame candidate
chains from the clamp (one skipped marker allowed), a Viterbi pass over the whole recording
(length errors, marker motion, credit per label), anchored by the hovers; markers on which a
near-optimal alternative disagrees stay unlabelled (user's rule: "if we are not sure just
don't use it"). On `whip_01`: 9.80 labelled markers per frame against 8.61, zero links
outside 20 mm; the final hover, where the filter had latched onto a drone marker, is correct.
The filter's labels are kept as `cable_labels_filter.npz` to score the live filter.

**Open.** Training environment on the batch runtime (replacing `tensor_model`); the live
filter's chain search is untested in flight; EI is unidentifiable (gravity-dominated cable).

**Results after relabelling (2026-10-28, `runs/calibration_2026-10-27/`; each computed twice, reproduced).**
Labels: sequence labeller on all ten cable flights (`cable_labels_report.json` per flight); `circle_01`
crashed once with an out-of-range point index and passed on an identical rerun (cause not found).
- Cable v2 refit on the new labels: 30.1 mm on the fitting flights, 56.6 mm held out (random_high
  35.0, whip 68.0). Values: EI 1.14e-7 (unidentifiable), Cb 3.02e-4, Cd 0.689, linear drag 0.0526 s⁻¹;
  adopted in `drone_cable.json` (drone mass 0.146 kg, weighed).
- Previous cable model (v1, twist) on the same windows: configured 47.1 / 52.8 mm (fit / held out);
  its earlier best (Cb ×4, drag ×4) 32.0 / 59.5 mm. So v2 fits the fitting flights about as well
  without twist, but held out it is NOT better than v1 at its configured values (56.6 vs 52.8 mm):
  the held-out gain is unproven; the whip flight is hardest for every cable model.
- Coupled model (drone pipeline + cable, nothing fitted on these), per flight, fitted vs previous
  simulator values: cable shape 25.7–90.5 mm vs 28.0–121.1 mm; held out random_high 60.0 vs 74.3,
  whip 73.7 vs 73.5; tip 40–153 mm. Drone debiased position error 30–79 mm; held out
  random_high 54.7 vs 49.2 (worse), whip 54.2 vs 62.9 (better). Drone attitude settings in
  `geometric.toml` are left unchanged.
- Batch runtime (24 threads, coupled plant, flight controller per lane): 4.3k lane-steps/s at 24
  lanes, 7.2k at 96 (72 simulated s per s), no lane failures. Not compared with the tensor engine on
  the GPU in this session.

## Dated change note — 2026-10-28 training environment on the batch runtime (branch `engine-v2`)

Scope approved by the user ("faithful port"; keep the tensor engine until the real-flight test).

- New [envs/batch_environment.py](../aerowhip/envs/batch_environment.py) (`BatchEnvironment`,
  `make_batch_environment`) replaces `PlantEnvironment` in the training worker and the frozen-policy
  test. Same attributes (`frames`, `critic_states`, `history`, `ledger`, `step`, `finish`, ...), so the
  learner, task and reward are unchanged. Physics: `BatchRuntime` lanes with a native `FlightController`
  per lane (firmware thrust vector, thrust map with battery voltage, attitude loop, 30 ms delay),
  stepped in parallel on the CPU; networks stay on `run.device`. The GPU options (CUDA graphs,
  `capture`, `conditional`) no longer exist for the environment.
- Zero action means hover: a per-lane vertical feed-forward from the thrust map (at that lane's
  voltage) makes delivered thrust equal the weight of drone and cable. Without it zero action sags
  (a probe fell 17 m in 4 s). Voltage is the firmware reference 7.2 V unless `battery_voltage` is given.
- Differences from the tensor environment: validity guards (workspace, tilt, cable height) are
  evaluated at the 100 Hz samples, not per physics substep; a numerically failed lane is charged from
  the end of the failing 10 ms step; the critic's motor/wrench slots hold the four motor thrusts and the
  applied (delayed) command, and its previous-state slot is the previous 100 Hz sample. The critic
  vector therefore changed (198 values, free cable nodes only), so old checkpoints' critics do not load.
  Actor inputs have the same layout.
- Tests: `tests/test_environment.py` rewritten (schedule, hover holds within 1 mm, runs repeat bit for
  bit, a lane alone equals the same lane in a batch, validity and failure paths with a synthetic
  plant, ledger telescopes). The tensor-model tests in `tests/test_physics.py` were removed because the
  tensor model cannot load v2 configurations; `tests/test_native_physics.py` covers the same checks.
  Whole suite: 93 passed after removal.
- Not done / unverified: no training run on the new environment; `scripts/equivalence_check.py` and its
  reference still target the tensor environment and fail; the reward geometry (`fold_geometry.regions`,
  Python) now costs about 2.4 ms per lane-sample and takes >95 % of the step time (24 lanes, 4 s: ~25 s),
  so training throughput is limited by the task, not the physics. Domain randomization of voltage,
  mass or delay is not added.

## Dated change note — 2026-10-28 firmware flight law (Mellinger port) and stage-2 identification

Approved design: [simulation_package_design.md](simulation_package_design.md) (committed 5f50dcb). Steps 1–2 of
its order of work are done on `engine-v2`.

**Why.** Open-loop replay of the cable flights showed the simulated drone leaning away from the cable's
pull (−5.1° at large swings) and sinking (−0.19 m/s² mean vertical bias), while the drone-only flights
had no bias. The identified attitude PD was several times too soft; the real firmware has no roll/pitch
integral (checked in `controller_mellinger.c`: `ki_m_xy = 0`), so the fix was the firmware itself.

**What.**
- [native/firmware.hpp](../native/firmware.hpp): line-by-line port of `controllerMellinger()`
  (position branch, `e_R` expression, `e_R.y` flip, rate error, `kd_omega_rp` derivative, yaw integral,
  ±32000 clamps, reset on non-positive thrust), `powerDistributionLegacy()` and `powerDistributionCap()`
  of bitcraze/crazyflie-firmware b48aba18, in float. The sources are vendored in
  [native/third_party/crazyflie-firmware](../native/third_party/crazyflie-firmware) (MIT) and compiled by
  [tests/test_firmware_port.py](../tests/test_firmware_port.py) into a harness: on 1600 random steps
  (800 near hover, 800 saturating) the int16 roll/pitch/yaw, the four PWM values and the capped flag
  are identical and the float thrust agrees to 2e-6 relative. Battery compensation is assumed off.
- `FlightController` ([native/flight_controller.hpp](../native/flight_controller.hpp)) has two laws:
  `firmware` (the port; per-motor thrust T_i = κ'/4 (pwm_i/massThrust)^p from the identified collective
  map, mixed to a wrench with the actuator geometry) and `identified` (the previous PD, fallback).
  `step` advances the firmware memory, `peek` does not; `BatchRuntime.reinitialize` resets it.
- Configs: [crazyflie_firmware.toml](../configs/controllers/crazyflie_firmware.toml) carries `ctrlMel.*`
  as flown (crazyswarm2 `crazyflies.yaml` cf_2: mass 0.144, massThrust 17700; translation gains zeroed
  by the host script; attitude gains firmware defaults). Controller clock 0.002 s (500 Hz). Actuator
  motor layout follows the legacy mixer (M1 front-right, M2 rear-right, M3 rear-left, M4 front-left;
  M2/M4 positive yaw). `flight_drone.firmware_config()` defaults to the firmware law; the training
  environment, benchmark and coupled validation therefore use it; `fit_drone_flights.py` keeps the
  identified law.
- Stage 2 ([scripts/fit_firmware_drone.py](../scripts/fit_firmware_drone.py), drone-only flights,
  debiased 2 s open-loop loss, held out random_high and whip; cable flights scored coupled, never fitted):
  roll/pitch inertia ×1.375 (1.877e-4 kg m²), motor arm ×1.225 (0.0613 m), motor response 22.5 ms,
  command delay 0 — adopted in `drone_cable.json`, `drone.json`, `geometric.toml`. Numbers reproduced
  twice (`runs/calibration_2026-10-27/firmware_drone_fit.json`; a refinement from this point,
  `firmware_drone_fit_refined.json`, moved to inertia ×1.79 / arm ×1.43 / 4 ms with the same held-out
  scores — inertia and arm are correlated and the search surface is flat; the arm must be measured).

**Results (vector RMS over 2 s windows, mm; fit / held out).**

| law | drone-only debiased | drone-only raw | cable debiased | cable raw |
|---|---|---|---|---|
| identified (previous) | 57 / 53 | 256 / 248 | 74 / 79 | 383 / 362 |
| firmware, unfitted (inertia and arm as guessed) | 76 / 64 | 313 / 301 | 72 / 58 | 332 / 271 |
| firmware, stage 2 (adopted) | 65 / 58 | 278 / 265 | 61 / 54 | 296 / 250 |

Per axis on the cable windows, the lean away from the cable at large swings fell from −5.1° to −1.9°
and the mean vertical bias from −0.19 to −0.10 m/s² (the hover residual, see "Open"). The firmware law
is worse than the identified PD on the drone-only flights it was not tuned on by hand (+4 mm held out)
and much better on the cable flights (−25 mm held out, −31 % raw): one model for both, with the
controller gains fixed by the firmware.

**Checks.** 96 tests pass (firmware harness, exact physics, environment hover within 1 mm, batch =
single bit for bit). Batch throughput with the 500 Hz firmware law: 2.8k lane-steps/s at 24 lanes,
4.9k at 96 (was 4.3k / 7.2k with the 200 Hz identified law).

**Open.** Hover residual −0.10 m/s² with the cable (mass bookkeeping or downwash; lab tests listed in
the design §5). Motor arm and inertia unmeasured (stage 0). Per-motor PWM→thrust map and PWM
saturation level unidentified (stage 1): the 0.85 N per-motor limit and the power-law extrapolation
above hover are the previous assumptions. The firmware's state estimate is the true state (no
attitude bias yet). Drone-only identification could be improved by a search that does not stall at
the delay bound. Coupled validation table with the adopted configuration:
`runs/calibration_2026-10-27/coupled_validation_adopted.json` (`validate_coupled.py --adopted`, every row reproduced twice): cable shape 25.0–66.6 mm on the fitting flights, held out random_high 45.1 mm and whip 70.6 mm (previous identified-law coupled validation: 60.0 and 73.7); tip 39–131 mm; drone debiased 32–103 mm (line_star worst), held out random_high 46.6 and whip 54.3 (previous 54.7 and 54.2).

## Dated change note — 2026-10-28 OptiTrack sensing model and the live cable filter in the environment (branch `engine-v2`)

Design step 3 of [simulation_package_design.md](simulation_package_design.md) (approved 2026-10-28).
No reward, task or physics change.

- New [envs/sensing.py](../aerowhip/envs/sensing.py): `SensingConfig` (the `environment.sensing`
  section of the experiment file, exact keys) and `OptiTrackSensor`, one per lane. Each 100 Hz frame
  the sensor turns the true state into the cloud the cameras would report — the ten markers minus
  those hidden by a per-marker visible/hidden Markov chain (hazard of hiding piecewise constant in
  the marker's speed, constant reappearance probability), white position noise, the drone's own
  rigid-body markers (body frame) and static room points, in random order — queues it for the lane's
  latency (drawn per episode from `latency_frames`, uniform), preprocesses it as the offline labeller
  does (`cable_filter.cable_candidates`: 7 cm drone exclusion, 1 cm duplicate merge, now shared) and
  runs the live `CableFilter` on it. The first frame of an episode is complete (the policy would not
  be started before the filter has labelled the hanging cable); until the latency queue has filled the
  policy sees that first frame. Draws are a function of (`sensing_seed`, lane).
- [envs/batch_environment.py](../aerowhip/envs/batch_environment.py): the actor frame now holds the
  sensed pose (delayed), the filter's node positions and velocities relative to the sensed attachment,
  and ten marker confidences (new slots after the cable velocities); motors/applied command stay critic
  only. Frame 106 columns, actor 98, critic 186 (unchanged content: true state). History gains
  `sensed_cable_positions` and `sensed_confidence`; replay arrays gain `sensing_latency_frames`;
  provenance records the sensing settings.
- Identification, [fitting/sensing_statistics.py](../aerowhip/fitting/sensing_statistics.py) and
  [scripts/identify_sensing.py](../scripts/identify_sensing.py) on the nine cf_2 flights
  (`runs/calibration_2026-10-27/sensing_statistics.json`). Camera level (points within the cable's
  reach against ten markers): frames with a missing marker 0.1–4.4 % (vertical worst); per-marker
  per-frame hazard 7.8·10⁻⁴ at rest, 1.3–2.5·10⁻⁴ for 0.25–4 m/s, 6.5·10⁻⁴ for 4–8 m/s, 0.035 above
  8 m/s (22 frames); reappearance 0.3155 per frame (label level); resting-marker white noise 0.097 mm
  per axis (drone 0.02 mm); receipt jitter < 0.5 ms; stamps bunched 5–16 ms. Drone body markers at
  (9,1,0), (−8,−2,0), (0,−4,0), (0,5,0) cm; static room points at about (3.3,−0.3,0.05),
  (−0.55,3.9,0.7), (2.0,3.2,0.9) m. Adopted as `environment.sensing` in the v6 experiment file.
  Latency range 0–2 frames is **assumed** (not measurable from these files).
- Validation, [scripts/validate_sensing.py](../scripts/validate_sensing.py)
  (`runs/calibration_2026-10-27/sensing_validation.json`, reproduced twice): per flight the longest
  completely reconstructed cable span becomes the 'true' cable, synthetic clouds go through the
  filter, and its label statistics are compared with the same filter's labels on the real clouds
  (`cable_labels_filter.npz`). Frames with a label gap, synthetic / real: circle 1.5 / 2.6 %, decay
  1.5 / 4.3, figure8 1.6 / 2.6, line_star 2.3 / 9.9, random 1.4 / 0.5, random_high 1.0 / 1.9, strokes
  1.6 / 1.4, vertical 6.1 / 6.8, whip 12.9 / 19.4; labelled fraction whip 0.903 / 0.862; mean confidence
  whip 0.941 / 0.862. The long gaps on vertical and whip come from the filter losing the chain in both
  worlds. The synthetic clouds are optimistic on the slow flights with real clutter near the cable
  (line_star, decay) and on the whip, because ambiguous real points (partial tape markers, reflections)
  are not modelled; the camera-level hazards were kept rather than inflated to match.
- Checks: 103 tests pass (`tests/test_sensing.py`: config validation, cloud contents and candidate
  preprocessing, Markov statistics against the hazard table within 1 %, latency shifts what the policy
  sees, hidden markers coast with reduced confidence, draws depend on seed and lane;
  `tests/test_environment.py`: frame layout and confidences, lane 0 alone equals lane 0 of three).
  End-to-end environment throughput with zero actions: 1.9k lane-steps/s at 24 lanes, 2.1k at 96
  (plant alone 2.9k / 4.9k); the Python filter and cloud synthesis take about 60 % of the step.
- Not done / open: the live display still runs the older `deployment/cable_tracking.CableTracker`;
  the flight node that will run `CableFilter` and the policy does not exist yet. The firmware's state
  estimate is still the true state (attitude bias, design §2.5, not added). No training run on the new
  observations. `scripts/equivalence_check.py` still targets the tensor environment. A C++ port of the
  filter would restore the batch throughput.

## Dated change note — 2026-10-28 drone air drag (branch `engine-v2`)

Approved by the user after the residual analysis ("Yes lets do it").

- **Evidence.** Per 2 s open-loop window (adopted firmware model, 323 drone-only and 287 cable
  windows) the constant acceleration that explains the error (sim minus real) correlates with the
  window's velocity along the same axis (corr 0.62 x / 0.48 y drone-only, 0.58 / 0.42 cable): the
  simulated drone outran the real one in its direction of travel. A linear term fitted on the
  fitting flights lowered the held-out residual 0.288 → 0.229 m/s² (drone-only), 0.272 → 0.206
  (cable); a quadratic term added nothing held out; per-flight constants (attitude-estimate bias)
  only 0.229 → 0.217. One-off analysis, not retained.
- **Model.** [native/engine.cpp](../native/engine.cpp): `drone_body_drag_acceleration`,
  F = −m R diag(k) Rᵀ v on the COM velocity (still air), added to the drone load and to
  `drone_air_force`. New required key `drone.body_drag_s_inv` (three non-negative rates; bindings
  reject a missing key). The residual-network rebuild path and `Model.info()` carry it.
- **Fit.** [scripts/fit_drone_drag.py](../scripts/fit_drone_drag.py), Nelder–Mead on the raw 2 s
  error of the drone-only fitting flights, horizontal rates shared, everything else fixed:
  k_xy 0.518, k_z 0.846 s⁻¹ (`runs/calibration_2026-10-27/drone_drag_fit.json`, reproduced twice).
  Raw / debiased mm, before → after: drone-only fit 278/64.9 → 139/37.6, held out 265/57.7 →
  134/42.2; cable fit 296/60.8 → 179/38.9, cable held out 250/54.2 → 164/45.6. Adopted in
  `drone_cable.json`.
- **Coupled validation** with the adopted configuration
  (`runs/calibration_2026-10-27/coupled_validation_adopted.json`; the previous one is kept as
  `coupled_validation_adopted_nodrag.json`), every row reproduced twice: drone debiased 27–44 mm
  (was 32–103; line_star 103 → 44), raw 119–213 mm (was 167–392); held out random_high 41.6 / 133
  raw (46.6 / 221), whip 40.8 / 174 raw (54.3 / 250). Cable shape improved on seven flights
  (line_star 66.6 → 50.4, random_high 45.1 → 42.7) and is slightly worse on whip 70.6 → 73.8 (tip
  130.7 → 135.0) and vertical 25.0 → 25.8.
- **Checks.** 107 tests pass, including `test_body_drag_is_linear_in_body_frame_velocity` (total
  momentum change equals −m R diag(k) Rᵀ v to 1e−12, through the clamp).
- **Open.** k_z is above the residual regression (0.2–0.33) and may absorb vertical thrust
  dynamics (motor map, stage 1); stage 2 (inertia, arm, motor response) was fitted without drag and
  has not been refitted jointly; drag proportional to thrust (rotor drag) versus constant was not
  tested; whip cable shape is not improved.

## Dated change note — 2026-10-28 faster drone-only replays; joint stage-2 + drag refit (not adopted)

- [fitting/flight_drone.py](../aerowhip/fitting/flight_drone.py) `drone_only`: the stand-in cable for
  drone-only flights is now one 2 cm link of 1 mg (was ten links of 0.1 g). On 65 drone-only windows
  it moves the drone by at most 0.12 mm over 2 s relative to a ten-link 1 mg cable, whereas the 0.1 g
  cable moved it by up to 12.6 mm (pooled RMS change 0.14 mm); one replay costs about half. Earlier
  drone-only results (stage 2, drag fit) were computed with the 0.1 g cable; recomputed with the new
  one, the adopted model scores 139 raw / 37.6 debiased (fit) and 135 / 42.1 (held out), within 1 mm of
  the reported values.
- Physics step: the configured 1/600 s step runs as 1 ms steps under the 2 ms controller period (each
  period is split into two). A 2 ms step was compared with 1 ms on whip, line_star, strokes and
  random_high cable windows: maximum drone difference 0.8–2.4 mm, marker 1.0–2.4 mm (median 0.3–0.7)
  over 2 s, no solver failures, 1.9x faster. Not adopted (pending the user's decision); one-off check.
- [scripts/fit_drone_joint.py](../scripts/fit_drone_joint.py): stage 2 (inertia scale, arm scale,
  motor response, delay) refitted together with the drag rates on the raw loss of the drone-only
  flights (`runs/calibration_2026-10-27/drone_joint_fit.json`, reproduced twice; 87 evaluations,
  3 minutes). Best: inertia ×0.934, arm ×0.678, motor 19.1 ms, delay 41.6 ms, k_xy 0.70, k_z 0.93.
  Raw / debiased mm, adopted → joint: fit 139/37.6 → 131/35.4; held out 135/42.1 → 138/38.7 (whip raw
  132 → 139); cable fit 179/38.9 → 177/38.2; cable held out 164/45.6 → 152/40.6. **Not adopted**: the
  gain is small and mixed, and the parameters moved far along known trade-offs (arm against inertia,
  delay against drag and motor response) — these flights do not identify them separately. Needs the
  measured arm and the acceleration-step flights (design §5) first.

## Dated change note — 2026-10-28 physics step 2 ms (convergence study)

Approved by the user (switch to 2 ms if peak tip speed holds).

- [scripts/step_convergence.py](../scripts/step_convergence.py) (`runs/calibration_2026-10-27/step_convergence.json`,
  every replay run twice and identical): all 2 s windows of whip (27, held out) and strokes (24), open
  loop with the sent commands ×1 and ×1.5 (the policy part; trims kept), physics steps 2, 1, 0.5 ms
  (reference). 2 ms vs 0.5 ms: drone max 1.0–2.1 mm, marker max 1.3–2.8 mm (median 0.23–0.26), peak tip
  (c10) speed within 0.09 %, peak timing shift ≤ 1 frame, fastest reference peaks 6.1 m/s (×1) and
  7.5 m/s (×1.5); no failures. 1 ms vs 0.5 ms: about one fifth (second order). `DroneReplay.run`
  gained `scale` and returns marker velocities with markers.
- Adopted `simulation.physics_timestep_s` 0.002 (was 1/600 s, which ran as two 1 ms steps per 2 ms
  controller period). Coupled validation with the adopted configuration re-run: every number within
  0.1 mm of the 1 ms run (`coupled_validation_adopted_1ms.json` kept). Plant benchmark 2.9k/4.9k →
  5.2k/9.0k lane-steps/s (24/96 lanes); environment end to end 1.9k/2.1k → 2.4k/2.7k (the Python
  sensing and task now dominate). `tests/test_live_cable.py` derives its step count from the model.
  107 tests pass.

## Dated change note — 2026-10-28 whip error split; live filter frame clock and velocity check

Approved by the user ("Yes": cable vs drone split, then filter velocities).

- **Error split**, [scripts/cable_error_split.py](../scripts/cable_error_split.py)
  (`runs/calibration_2026-10-27/cable_error_split.json`, reproduced twice): every cable-flight window
  replayed with the measured drone pose driving the clamp (cable alone) and open loop from the sent
  commands (coupled), scored as cable shape. Cable alone / coupled, mm: whip 67.9 / 73.8 (tip 127 / 135),
  strokes 48.5 / 52.3, decay 43.5 / 48.5, random_high 35.0 / 42.7, line_star 29.1 / 50.4, others 19–32 /
  25–37. The cable model dominates on the fast flights; its error grows with marker speed (whip 39 mm below
  1 m/s, 97 at 1–2, 128–187 above 2) and toward the tip, and through the window (whip 21 → 93 mm from the
  first to the last half second). Diagnostics (one-off, not retained): 2 and 3 segments per marker interval
  improve whip by 1.5–1.7 mm (`flight_cable.CableReplay` now starts finer meshes from the measured markers);
  a central-difference initial velocity changes nothing; one-at-a-time sensitivity on whip / strokes /
  random_high: Cb 1e-4 → 58.8 / 49.8 / 37.1 (adopted 3.0e-4: 67.9 / 48.5 / 35.0), Cb 1e-3 → 87.9 / 64.0 /
  53.9, Cd 0.3–1.2 and the linear drag ±2 mm, EI 1e-7…1e-4 ≤ 1 mm. Bending damping is the term to revisit;
  not refitted (whip is held out).
- **Filter frame clock**, [deployment/cable_filter.py](../aerowhip/deployment/cable_filter.py): the stamps
  in the recordings are host receipt times (stamp − receipt 0.3 ms; mean step 10.00 ms, steps 5–16 ms with
  stalls to 37 ms). `observe` now advances a frame clock by whole frames (≥ 1) and pulls it 5 % toward
  each stamp; prediction and velocity use it. `get_state`/`set_state` carry it. Effect on the recorded
  clouds: whip labels agree with the sequence labels on 99.96 % of labelled marker-frames and label 99.38 %
  (stored previous-filter labels: 84.2 % / 86.2 %); line_star 99.50 / 98.26 (98.74 / 97.72). The stored
  `cable_labels_filter.npz` files were not regenerated.
- **Velocity check**, [scripts/filter_velocity_check.py](../scripts/filter_velocity_check.py)
  (`filter_velocity_check.json` on the recorded stamps, `_uniform`/`_stamps` variants kept, each reproduced
  twice): reference = central differences of `cable_labels.npz` on the 10 ms clock; snaps = tip peaks
  > 3 m/s fully labelled ±5 frames (83 on nine flights; peaks inside a label gap are not scored — an
  earlier version without this rule reported spurious 1.4× overshoots). Frame-clock filter on recorded
  stamps: velocity RMS 0.08–0.26 m/s, tip 0.11–0.32; snap peak ratio median 0.99–1.00 per flight, delay
  median 1 frame; outliers 0.52–0.75 at six snaps right after delivery stalls (uniform clock: min 0.90).
  On simulated whips with exact truth (one-off): peak within 2–4 %, 1–2 frames late. Recommendation for
  flight: publish the NatNet frame number with the cloud.
- **Sensing validation** ([scripts/validate_sensing.py](../scripts/validate_sensing.py)) now runs the
  current filter over the real clouds instead of reading `cable_labels_filter.npz`, and uses as synthetic
  truth only physically possible reconstructed frames (every link ≤ 15 mm over its interval; extrapolated
  markers had violated it by up to 1.8 m and caused most of the earlier synthetic whip gaps). Labelled
  fraction simulated / real within 1 % on every flight (whip 0.996 / 0.993); frames with a gap 0.8–1.6 %
  simulated vs 0.5–6.0 % real, real gaps shorter (1.3–2.1 frames vs 2.9–6.6). Supersedes the earlier
  validation numbers.
- 109 tests pass (frame clock: jitter and a dropped frame).
