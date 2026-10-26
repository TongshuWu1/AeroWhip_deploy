# AeroWhip

Drone-cable simulation, recorded-data fitting, and current Stage 1 PPO
training. C++ implements the motor-driven physical plant; Python implements
controllers, tasks, learning, data processing, and the desktop UI.

The native simulation, flight viewer, and Crazyswarm command/feedback connection
have been exercised on Linux. The single-flight native engine runs on the CPU;
the viewer uses the host GPU, and training uses the separate CUDA batch path.
Real-flight cable estimation, learned-policy deployment, and firmware controller
equivalence remain unimplemented or unverified.

Read [the current Stage 1 specification](doc/stage1.md) for its exact reward,
fold measurement, termination and exploration. Historical stages and models are
recoverable from Git commit `3c9a49e`; their training code is retired.

## Current handoff — 2026-10-25

Training is stopped. The active Stage1 preset is the **four-second, fixed +X experiment**,
with 64 environments and 200 collections. Its latest run completed zero collections;
there is no newly trained policy to resume. See [other-computer setup and data restoration](doc/handoff-20261025/README.md).

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
configuration and trajectory CSVs. Recordings, training
runs and retained test evidence are stored separately in GitHub releases:
`handoff-2026-10-25` for the current handoff and `backup-2026-10-16` for older
recordings, with per-file manifests and SHA-256 checksums.
They are not downloaded by `git clone`; restore them separately below to use
historical local evidence links. Virtual environments and build caches remain
local. Recreate dependencies using the setup instructions.

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

## Recordings and training backup

[Download the data release](https://github.com/TongshuWu1/AeroWhip_deploy/releases/tag/backup-2026-10-16).
It contains three archives: `aerowhip-training-runs.tar.gz` (`runs/`),
`aerowhip-recordings-and-output.tar.gz` (`output/` and `data/recordings/`), and
`aerowhip-retained-evidence.tar.gz` (`tmp/`, excluding build/cache files and
publication tooling). Each has a JSON file manifest; `SHA256SUMS` verifies the
release downloads. These retain successful, failed and exploratory work without
promoting any trained model or changing recorded scientific results.

Restore into a fresh checkout to avoid overwriting newer local recordings:

```bash
gh release download backup-2026-10-16 --repo TongshuWu1/AeroWhip_deploy --dir ../aerowhip-data-backup
(cd ../aerowhip-data-backup && sha256sum -c SHA256SUMS)
tar -xzf ../aerowhip-data-backup/aerowhip-training-runs.tar.gz
tar -xzf ../aerowhip-data-backup/aerowhip-recordings-and-output.tar.gz
tar -xzf ../aerowhip-data-backup/aerowhip-retained-evidence.tar.gz
```

The companion Crazyswarm release contains its flight logs. This AeroWhip repository and its releases are currently public. Access to the companion
repository depends on its own permissions. The newer handoff release is linked above.

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

The patched open OptiTrack backend preserves the mixed `/pointCloud` XYZ rows
and adds aligned `natnet_id`, `natnet_flags` and `natnet_source` fields. Cable
association accepts observed, unassociated passive labeled markers and excludes
asset markers, occluded/model-solved points and deprecated unclassified duplicate
rows. IDs filter membership; they do not define persistent cable identity.
All raw points remain recorded. Missing metadata shows an explicit unavailable/
rebuild diagnostic rather than associating the mixed cloud. Backends without
this metadata cannot supply this tracker. Rebuild the Crazyswarm Docker image
before recreating a container; the current development container was rebuilt.

Tracking requires a verified capture publisher and exact matching cloud/pose
header timestamps. Source and transformed XYZ arrays, marker metadata, original
and excluded point indices, assignment/null decisions and confidence are retained in `connection.jsonl`
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

Capture-only initially shows the measured drone and **blue accepted cable
estimates**. Raw gray candidates, the green simulated cable and its tip trail
start hidden. Use **Displays** to show them for comparison. Missing or ambiguous
markers remain gaps; accepted identities are estimates, not certified ground truth.

The recording contains measured drone poses, the mixed raw candidate cloud,
estimated marker identities/confidence, and simulated cable geometry, including
layers hidden in the live view.
New capture events retain aligned marker membership/visibility metadata. Older
recordings remain playable but cannot recover metadata discarded at capture.
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

One supported training task: **four-second +X fold transport and opening**,
64 parallel episodes, 200 collections. Return is continuous fold/output credit
minus uncapped squared-distance workspace cost. There is no target-hit reward,
physical-envelope early stop, effort/path cost, or automatic next stage.

The [Stage 1 specification](doc/stage1.md) describes the exact formulas and
limitations. The simulation and fitting pipeline remains separate and unchanged.

## Entry points

```text
python run_simulation.py
python run_custom_real.py
python run_training.py --stage 1
python run_training.py --stage 1 --start
python run_replay.py
python run_replay.py --flight-view
python take_processing/run_take_processing.py
```

Opening Training without `--start` stays idle. Only the current Stage 1 recipe
is supported. Saved replay renders recorded states without running physics; the
policy tester simulates compatible current checkpoints. Historical checkpoints
may require the preserved source at commit `3c9a49e`. Source loading alone is
not a successful simulation or policy evaluation.

Offline whipping measurements use pinned saved replays and keep deformation,
distal motion and delivery separate. Install plotting support with
`python -m pip install -e ".[review]"`, then run:

```bash
python -m evaluation.whip_event_review --manifest /absolute/path/inputs.json --output /absolute/path/new-review
python run_replay.py --measurement-report /absolute/path/new-review/report.json --case 0
```

The review writes synchronized plots, `review.md`, `events.csv`, full measurement
traces and `report.json`. `--case` and optional `--event` are zero-based. Candidate
intervals remain unreviewed; they do not change rewards, training outcomes or
curriculum. The manifest format and retained example bundle are described in
[the preserved measurement documentation](https://github.com/TongshuWu1/AeroWhip_deploy/blob/3c9a49e/doc/current_implementation.md#offline-whipping-event-measurements--implemented-2026-10-22).

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

## Historical code and evidence

Retired presets, Stage 1–4 models and old reward/replay implementations remain in
[commit 3c9a49e](https://github.com/TongshuWu1/AeroWhip_deploy/tree/3c9a49e).
Use a separate checkout to investigate them; they are not current training choices.
The [handoff guide](doc/handoff-20261025/README.md) links the experiment archives.
No old result or current software test establishes physical-flight readiness.
