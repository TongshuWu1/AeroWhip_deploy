# AeroWhip

Drone-cable simulation, recorded-data fitting, and four-stage fixed-target PPO
training. C++ implements the motor-driven physical plant; Python implements
controllers, tasks, learning, data processing, and the desktop UI.

The native simulation, flight viewer, and Crazyswarm command/feedback connection
have been exercised on Linux. The single-flight native engine runs on the CPU;
the viewer uses the host GPU, and training uses the separate CUDA batch path.
Real-flight cable estimation, learned-policy deployment, and firmware controller
equivalence remain unimplemented or unverified.

The historical completed Stage1-4 checkpoint chain remains in
[models/milestone-20261009](models/milestone-20261009), with its original settings
and observations. See [current implementation](doc/current_implementation.md) for
active configurations, retained evidence, and implemented versus proposed work.

## Setup

The two source projects use this checkout layout for automatic discovery:

```text
<workspace>/
├── AeroWhip_deploy/
└── Crazyflie/
    └── crazyswarm2_jazzy_whipping/
```

Keep these directory names when cloning. `AEROWHIP_ROOT` can override the
AeroWhip path for the Docker launchers. The repositories include source,
configuration, trajectory CSVs and packaged model artifacts. Generated sessions,
training runs, temporary evidence, virtual environments and build caches remain
local; some historical evidence links in the implementation record therefore
require the original local workspace. Recreate dependencies using the setup
instructions rather than copying a virtual environment.

Run setup commands from the AeroWhip checkout. Create a new virtual environment
on each computer; rebuild native extensions for its OS and Python ABI. The
checked setup uses Python 3.12, Torch 2.11.0, and pybind11 3.0.1. Install a Torch
build matching the intended CUDA environment before building the tensor extension.

On Linux, with a C++17 compiler available:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -e ".[build]"
python build_engine.py
python validate_stages.py
```

On Windows, with MSVC and a Windows SDK available:

```powershell
py -3.12 -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install -e ".[build]"
python build_engine.py
python validate_stages.py
```

CUDA training also requires a compatible Torch/CUDA/Triton installation. The
interactive viewer needs a working desktop/OpenGL session. Setup and numerical
checks do not establish policy performance or real-flight readiness.

## Simulation flight

For standalone simulation, run from the AeroWhip checkout:

```bash
.venv/bin/python run_simulation.py
```

This uses [configs/scenes/default.toml](configs/scenes/default.toml) for initial
state and display settings, and [configs/simulation.json](configs/simulation.json)
for the physical model, controller, actuator, and residual configuration. It
does not start the ROS bridge. In PyCharm, select this checkout's `.venv`
interpreter and run `run_simulation.py`.

### Connect Crazyswarm scripts to the simulation

ROS runs in the existing Crazyswarm Docker container; native physics and the
viewer run in the host AeroWhip environment. In a terminal opened at the
Crazyswarm checkout (`crazyswarm2_jazzy_whipping`):

```bash
./docker/compose
./docker/start.sh --custom-sim
```

Keep that terminal/viewer open. In a second terminal at the same Crazyswarm
checkout, open a shell configured for the custom simulation:

```bash
./docker/shell.sh --custom-sim
```

Then run a flight script inside that shell. For example, for `cf_2`:

```bash
python3 /workspace/scripts/full_state_pva_csv_repeat.py \
  /workspace/scripts/trajectory_csv/fullstate_30hz_M5_replan.csv \
  --drone cf_2 \
  --repeats 3 \
  --between-hover 15 \
  --pre-hover 15 \
  --post-hover 3
```

Use ordinary spaces in copied multiline commands. The custom shell selects ROS
domain `142` and enables simulation time; client timing follows the native
`/clock`. Closing the viewer shuts down the bridge and native owner.

The default `./docker/shell.sh` opens a custom-simulation client shell on domain
142. Use `./docker/shell.sh --real` for real flights or `./docker/shell.sh --sim`
for stock simulation (both domain 42). Existing terminals must be reopened to
apply this selection.

The launch modes remain separate:

| Crazyswarm command | Runs | Default ROS domain |
|---|---|---|
| `./docker/start.sh --real` | Real-flight ROS server with motion capture and RViz | 42 |
| `./docker/start.sh --sim` | Stock Crazyswarm simulation with RViz | 42 |
| `./docker/start.sh --custom-sim` | AeroWhip native simulation and flight viewer | 142 |
| `./docker/start.sh --custom-real` | Real Crazyswarm/OptiTrack stack and the shared GPU viewer | 42 |
| `./docker/start.sh --custom-real --capture-only` | OptiTrack hand capture and the shared GPU viewer | 43 |

The launcher finds AeroWhip in the current adjacent-project layout. For another
layout, set `AEROWHIP_ROOT` to the AeroWhip checkout's absolute path. You can also
run `.venv/bin/python run_custom_simulation.py` from AeroWhip or run that file in
PyCharm with the host `.venv`; the ROS container must already be running.
`--container` selects another container and `--crazyflies-yaml` selects another
robot YAML file.

### Robot initialization and limits

The custom simulation reads the Crazyswarm `config/crazyflies.yaml`. It selects
the single enabled Crazyflie, or requires `--robot NAME` when several are
enabled. Only one selected drone/cable plant is simulated. Robot identity and
the exact `initial_position` in world-frame metres come from that YAML; the
current bridge supports the `world` frame.

Startup and Reset use the same loaded initial position. Reset does not reread an
edited YAML file; restart the simulation to load configuration changes. Other
initial state fields come from the AeroWhip scene, and explicit cable geometry
translates with the drone's initial position. The YAML robot type and firmware
parameters do not select or calibrate the native physical model/controller.

Position overrides are optional and require `--simulation-yaml`. From the
AeroWhip checkout, the retained airborne connection example can be selected with:

```bash
.venv/bin/python run_custom_simulation.py \
  --simulation-yaml configs/ros/custom_simulation.yaml
```

That example overrides `cf_2` to `[0, 0, 1.1]` m. It is not loaded by default and
does not modify the hardware YAML. The standalone simulation continues to use
its scene's initial position.

The current physical model is free space. Floor/grid geometry is for display;
it does not provide collision, cable support, or touchdown. At a YAML height of
zero, the hanging cable can extend below the displayed floor. Disarm cancels
commands and requests a controlled hold; emergency suspends physics until Reset
or restart. Neither behavior implements physical motor cutoff or ground contact.
The CSV player's simulation landing reference returns to the configured initial
height. These command checks do not establish firmware controller equivalence.

FullState streaming, arm, takeoff/land, go-to, polynomial trajectories, group
masks, stream release, and emergency are supported. See the
[connection details](doc/current_implementation.md#crazyswarm-command-connection-to-simulation-flight)
for command units, feedback topics, timing, and unsupported interfaces.

For the host-feedback acceleration circle, restart the custom simulation after
updating its adapter, then run in the default Docker client shell:

```bash
python3 /workspace/scripts/acceleration_circle_cf2.py --simulation --check
python3 /workspace/scripts/acceleration_circle_cf2.py --simulation
```

It rises 1.3 m, flies a 0.2 m radius circle, returns to its starting height and
restores PVA mode for subsequent waypoint scripts. Only acceleration changes in
FullState; P/V are placeholders. The script selects the nominal direct controller
and uses native feedback at 50 Hz simulation time. The model includes static
cable-weight trim and has no floor contact. If a bound or settling check fails,
use viewer Reset/Play or restart. This is a scripted controller test; real-flight
Mellinger equivalence and learned-policy deployment are not established.

Learned-policy ROS execution is not implemented yet. The
[proposed shared policy interface](doc/current_implementation.md#shared-policy-interface-proposed)
records the state/action contract and the proposed shared acceleration-reference
to PVA path. Existing velocity-only checkpoints are not automatically compatible
with that proposed controller/observation profile.

For a short scripted PVA connection check, leave the custom simulation running
and use a separate host terminal in AeroWhip:

```bash
.venv/bin/python run_custom_simulation_test.py --robot cf_2
```

Its result is saved in the Crazyswarm checkout's `logs/` directory. Host
connection evidence is saved under AeroWhip `output/custom_simulation/`.
`--headless --duration 10` runs a bounded launch without the viewer. The default
transport port is `47654`; `--port 0` selects an unused port. If another server
owns the simulation domain, close it or select a different domain with
`--domain N`, then use `./docker/shell.sh --custom-sim N` and pass `--domain N`
to the connection test.

Ctrl+C in the `--custom-sim` launch terminal closes the viewer, simulation and
its ROS adapter. Closing the simulation window does the same. Docker stays
running, so you can run `./docker/start.sh --custom-sim` again. The standalone
`run_simulation.py` window also closes cleanly on Ctrl+C.

## Real flight

From the Crazyswarm checkout, start the existing real stack with the shared
viewer in place of RViz:

```bash
./docker/compose
./docker/start.sh --custom-real
```

This launches the existing `cflib` server and OptiTrack node on ROS domain `42`,
using the same hardware and motion-capture YAML files as `--real`, with RViz
disabled. In a second Crazyswarm terminal, use `./docker/shell.sh --real` and run your
existing flight scripts. The viewer is read-only; it sends no flight commands.
Closing its window leaves the real stack running, as with RViz. Ctrl+C in its
launch terminal stops the stack owned by that launch.
The viewer runs in a separate supervised process. If it crashes unexpectedly,
the owning terminal stays open and keeps ROS running; use `--attach` to reopen
the viewer. Ctrl+C in either the owning terminal or an attached viewer terminal
stops that local ROS stack.

If the real stack is already running, attach only the viewer:

```bash
./docker/start.sh --custom-real --attach
```

Ctrl+C in an attached viewer terminal closes the viewer and stops the existing
local Crazyswarm/OptiTrack ROS launch in its selected domain. Closing just the
attached window leaves ROS running. Shutdown checks process identities and
does not stop the Docker container or flight scripts. A remote or manually
assembled stack without a matching local `ros2 launch crazyflie launch.py`
process cannot be stopped this way; the launcher reports that limitation.
A direct
AeroWhip/PyCharm entry point is `run_custom_real.py` with the host `.venv`;
`--attach` has the same meaning there. `--container`, `--crazyflies-yaml` and
`--motion-capture-yaml` can select another installation.

Keep ROS and the viewer open between completed flight scripts. Both custom modes
clear the drone/cable-tip trails and displayed reference when a new Crazyswarm
flight-script publisher is discovered. This changes display history only, not
the physical/simulated state, commands or saved recordings. **Clear trails** in
the viewer toolbar provides the same display cleanup manually, including when
one long-running client executes several flights.

Drone position and attitude come from the actual OptiTrack `/poses` capture,
matched to all enabled Crazyflie names in the hardware YAML. Use repeated
`--robot NAME` arguments to display a subset. The current capture frame is
`mocap`; the default viewer uses the frame declared in the motion-capture YAML.
`--frame world` requires an existing TF transform from the capture frame at
the capture timestamp. An absent transform is shown as an error; the viewer
does not assume those frames coincide.

Real display starts empty and waits for capture rather than placing a drone at
the YAML initial position. It does not fall back to onboard estimated pose.
The existing native cable model now follows the measured drone attachment motion.
The drone is labeled **Measured** and the cable **Simulated**. Cable state and
integration history persist between flight scripts; trail clearing changes only
the display. `--model PATH` selects the cable model, defaulting to
`configs/simulation.json`, as in custom simulation. The viewer scene's attachment
offset and cable lengths must match that model.

Short capture intervals are subdivided on numerical rejection, including
position/velocity constraint checks, with bounded2×/4×/8× retries from the same
accepted state. Solver tolerances and the physical cable model remain unchanged.

At the first pose, the cable is assumed hanging, stationary and untwisted.
That is an initial condition, not a measured cable shape. The current model has
no ground contact, so a cable attached to a grounded drone may extend below the
displayed floor. Tracking gaps over0.25 s or simulation failures freeze the cable
and show an error; restarting the viewer explicitly initializes a new cable
simulation. Stopped capture becomes stale, and invalid/missing orientation is
rejected. Drone velocity, acceleration, body rates and forces remain unavailable;
propellers have no synthetic animation. Vendor-tracked poses are
labeled measured, while poses reconstructed by `librigidbodytracker` are
labeled estimated. Viewer/world geometry remains display-only.
To inspect a simulated cable below the floor, uncheck **World → Floor** in
the Displays panel; this changes visibility only.

To test experimental cable reconstruction from the marker cloud, select one robot:

```bash
./docker/start.sh --custom-real --robot cf_2 --track-cable
# Add --attach if the real ROS stack is already running.
```

Hold the drone/cable with the cable fully visible and approximately vertical,
clear of the ground, until the display reports **10/10 · tracking**. Initialization
uses a hanging shape within3.5 cm of the configured marker positions; it cannot
initialize an arbitrary coil. Marker arc distances come from the selected model
(currently0.063,0.150,0.250,0.350,0.450,0.550,0.650,0.7525,0.8525,0.9525 m
from the attachment). Once initialized, move it slowly by hand first. The tracker
uses temporal prediction and maximum chord distances from the known arc lengths.
Velocity updates damp spikes caused by closely bunched receipt timestamps.
Occlusions leave gaps without discarding the remaining visible marker identities.
After0.25 s, a missing marker uses geometric recovery from the attachment and
recent neighbours instead of its old velocity prediction. It returns only when
the retained assignment hypotheses agree; ambiguous or crowded markers remain
missing. If all marker histories become older than0.25 s, hold the complete cable
hanging again to reinitialize. These are initial engineering settings to test,
not fitted accuracy limits.

**Gray** points are raw candidates, **blue** points/links have estimated cable
identities, and **green** is the existing simulated cable. The inspector shows
observed count, status, age and heuristic association confidence. Hide each cable
and the raw cloud independently in Displays. Missing/ambiguous nodes have no
rendered replacement or bridging link; stale reconstruction is hidden. The
tracking overlay does not correct the cable simulation or supply a controller.

The existing `/pointCloud` topic mixes loose and rigid-body markers and discards
NatNet IDs and visibility flags. Accordingly this first implementation records
the mixed XYZ cloud, not a certified unlabeled-only stream or persistent Motive
IDs. It requires a verified capture publisher and exact matching cloud/pose
header timestamps. Source and transformed XYZ arrays, original point indices,
assignment/null decisions and confidence are retained in `connection.jsonl`
under the printed output directory. These are ROS receipt timestamps; upstream
and DDS losses cannot be reconstructed. No synchronized camera-exposure record
or physical tracking accuracy is claimed.

`--point-cloud-topic` selects a different cloud topic. Use
`--cable-tracking-config configs/ros/cable_tracking.json` to edit the explicit
association settings; an optional `marker_distances_m` array overrides the
model-derived arc distances for tracking only. Settings and source hashes are
saved in metadata. This mode supports one selected cable; multi-cable association
and arbitrary-shape initialization are not implemented.

### Record a hand-movement test and replay it

From the Crazyswarm checkout, start a capture-only session:

```bash
./docker/compose
./docker/start.sh --custom-real --capture-only --robot cf_2
```

This uses ROS domain **43** by default (`--domain N` overrides it), starts
OptiTrack and the viewer, enables cable tracking, and does not start
the Crazyflie flight server. Hold the cable hanging and fully visible until
**10/10 · tracking**, click **Record**, then move the drone by hand. Click
**Stop & save**, followed by **Replay last** to open the saved test in a separate
window. You can make multiple clips without restarting capture or resetting the
simulation. Closing a capture-only viewer saves an active recording and stops
its owned OptiTrack node; Ctrl+C performs the same cleanup. An existing capture
or flight stack is not stopped to make room for this mode. To use an existing
stack, use `--attach --track-cable --robot cf_2 --domain 43` for a default
capture-only stack (or its selected domain), with the previously
documented attach shutdown behavior.

The recording contains measured drone poses, the mixed raw candidate cloud,
estimated marker identities/confidence, and the simulated cable as displayed.
Missing markers, stale samples and errors are retained. Record/Stop does not
reinitialize either the tracker or the cable simulation. **Blue** remains the
estimated cable and **green** remains simulated; neither is a validated ground
truth reference. Initialize with a hanging cable before a test as described above.

Recordings are saved under the printed session directory:
`output/custom_real/<session>/recordings/<clip>/recording.json`.
The manifest stores configuration/provenance; `frames.jsonl` stores display
results and `events.jsonl` stores received samples and association decisions.
Each clip has a new directory. Stop flushes both files before marking it complete.
An interrupted clip can replay its retained complete frames with an explicit
incomplete-recording warning.

Use **Open recording…** in the viewer to return to a saved clip. You can also
open `run_recording.py` in PyCharm, or run from the AeroWhip checkout:

```bash
.venv/bin/python run_recording.py
# Or supply a recording folder or its recording.json path:
.venv/bin/python run_recording.py /path/to/recording.json
```

Playback has play/pause, step, a seek slider and0.25×/0.5×/1× speeds. It reads
saved states without ROS, the drone, model loading or physics integration. The
playback clock is elapsed recording time; original pose/cable timestamps and
ages stay in the inspector. Recorded errors remain visible while playback can
continue through them. Replaying a clip does not rerun cable estimation or
change its saved results. `--record` on the live launcher starts recording
immediately, including while waiting for capture; it also works with
`--headless --duration SECONDS`.

Connection traces are saved under `output/custom_real/`. They are sampled
display evidence, not the full-rate source logger used for fitting. The adapter
and GPU viewer are checked separately from physical cable accuracy and flight.
The
[implementation record](doc/current_implementation.md#optitrack-connection-to-real-flight-display)
contains the data-source contract and retained verification artifacts.

The Docker OptiTrack receiver now supports both Motive **multicast** and
**unicast**, selected from the server's advertised mode. Unicast retains its
registered client socket and sends keepalives; multicast retains the shared
stream and configured interface. Changing mode in Motive requires restarting
the ROS launch to reconnect. The repair is included in the existing Docker build
patch and has also been rebuilt in the current development container.
Multicast sends one stream for multiple receivers; unicast sends a separate
stream per client and permits asset filtering. OptiTrack recommends unicast for
Wi-Fi. [OptiTrack streaming documentation](https://docs.optitrack.com/v3.2/motive/data-streaming)

## Current pipeline

| Stage | Learning objective |
|---|---|
| 1 | Directed whip mechanics and cable unfolding |
| 2 | Delivery toward one fixed 1.0 x 0.6 m target |
| 3 | Full-actor precision learning at the same target; size curriculum ends at 0.1 x 0.1 m |
| 4 | Frozen Stage3 whip, internal return controller, learned recovery residual and cable settling |

The Stage1 launcher default is
[compact v5](configs/learning/whip_stage1_ppo_compact_v5.json); Stage2-4 launcher
defaults remain `configs/learning/whip_stage{2..4}_ppo_cone25_v4.json`.
These defaults have 128 training lanes and a 10,240-episode cap: 80 full collections.
Plateau stopping is a configured heuristic; a cap does not establish convergence
or success. Stage2-4 require the preceding stage's final actor and start fresh
critics, optimizers, RNG, and local clocks.

The paused outreach/height experiment uses explicit settings rather than those
launcher defaults. Its Stage1
[v12 settings](configs/learning/whip_stage1_ppo_outreach_height_radius055_150_v12.json)
and planned Stage2/3 continuations each specify 150 collections. Select an
experimental recipe with `--settings`; do not assume `--stage` selects it. See
[the maintained stage record](doc/current_implementation.md#four-stages-and-scientific-criteria)
for initialization, reward definitions, and current experiment status.

All current stages use
`goal = (target_world - drone_initial_world) / cable_length` in world axes.
Training uses one fixed nominal target and fixed initial conditions. The
retired multi-target sampler and partial Stage3 training mask are not active.
A correction network learned between attempts is not implemented.

## Entry points

```text
python run_simulation.py
python run_custom_real.py
python run_training.py --stage 1
python run_training.py --stage 1 --start
python run_training.py --stage 2 --initialize-actor runs/ppo_stage1/NEW_STAGE1_RUN/latest.pt --start
python run_replay.py
python run_replay.py --flight-view
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
The ROS launcher explicitly accepts external Crazyswarm YAML paths as bridge
configuration; its scene, model, and output paths remain project-contained.

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
Later experimental recipes remain separate from the packaged historical chain;
no new Stage1-4 chain or flight-ready policy has been promoted.

[The model manifest](models/milestone-20261009/manifest.json) records hashes,
initialization links, settings, summaries, and essential saved evaluations.
Nominal precision/recovery results do not establish workspace containment,
varied-target robustness, or flight readiness. The small current-pipeline
regression suite is self-contained and has no archived run-fixture dependency.
Retained Linux connection checks cover the actual repeated CSV script, native
command reconstruction, the GPU viewer, and YAML startup/Reset. Their artifacts
and scope are linked from [current implementation](doc/current_implementation.md);
they are simulation evidence, not a physical fit, policy result, or real-flight
validation. Updating this README does not rerun those checks.

Development runs, old tests, and original documentation are recoverable in the
external archive `AeroWhip-development-archive-20261009`; its location and
preservation evidence are in [milestone verification](doc/milestone.json).
Git history is preserved.
Training and its continuation remain paused for simulation work; no README
command has been executed as part of this documentation update.
