# Simulation and flight

How to fly the native simulation, connect Crazyswarm scripts to it, and view,
track and record real OptiTrack flights. ROS runs in the companion
`crazyswarm2_jazzy_whipping` Docker container;
physics and the viewer run in this repository's Python environment. Keep both
checkouts side by side:

```text
<workspace>/
├── AeroWhip_deploy/
└── Crazyflie/
    └── crazyswarm2_jazzy_whipping/
```

`AEROWHIP_ROOT` overrides the AeroWhip location for the Docker launchers.

## Native simulation

For standalone simulation, run from the AeroWhip checkout:

```bash
python scripts/simulate.py
```

This uses [configs/scenes/default.toml](../configs/scenes/default.toml) for initial
state and display settings, and [configs/simulation.json](../configs/simulation.json)
for the physical model, controller, actuator, and residual configuration. It
does not start the ROS bridge. In PyCharm, select this checkout's `.venv`
interpreter and run `scripts/simulate.py`.

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
run `python scripts/custom_simulation.py` from AeroWhip or run that file in
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
python scripts/custom_simulation.py \
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
masks, stream release, and emergency are supported. 
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

Running a learned policy through ROS is not implemented yet.

For a short scripted PVA connection check, leave the custom simulation running
and use a separate host terminal in AeroWhip:

```bash
python scripts/custom_simulation_test.py --robot cf_2
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
`scripts/simulate.py` window also closes cleanly on Ctrl+C.

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
AeroWhip/PyCharm entry point is `scripts/real_flight.py` with the host `.venv`;
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
(currently 0.095, 0.195, 0.295, 0.395, 0.495, 0.595, 0.695, 0.795, 0.895, 0.995 m
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
open `scripts/play_recording.py` in PyCharm, or run from the AeroWhip checkout:

```bash
python scripts/play_recording.py
# Or supply a recording folder or its recording.json path:
python scripts/play_recording.py /path/to/recording.json
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

The Docker OptiTrack receiver now supports both Motive **multicast** and
**unicast**, selected from the server's advertised mode. Unicast retains its
registered client socket and sends keepalives; multicast retains the shared
stream and configured interface. Changing mode in Motive requires restarting
the ROS launch to reconnect. The repair is included in the existing Docker build
patch and has also been rebuilt in the current development container.
Multicast sends one stream for multiple receivers; unicast sends a separate
stream per client and permits asset filtering. OptiTrack recommends unicast for
Wi-Fi. [OptiTrack streaming documentation](https://docs.optitrack.com/v3.2/motive/data-streaming)
