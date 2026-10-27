# Flight data and model fitting

Two recording formats exist: calibration flights recorded by the companion
repository's flight script (below), and the earlier takes (Motive CSV plus logger
sidecars, further down). The fitting tools currently read the earlier takes.

## Calibration flights

The companion repository (`crazyswarm2_jazzy_whipping`, `DATA_COLLECTION.md`) flies a
fixed series through the acceleration interface and writes one folder per flight under
`calibration_data/<motion>/`: `parameters.json`, `flight.jsonl` (50 Hz host log),
`poses.jsonl` (every OptiTrack drone pose) and, with the cable, `cloud.jsonl` (every
OptiTrack marker cloud, unlabelled, on the same stamps as `poses.jsonl`).

### Cable labels from the marker cloud

The flight script records the whole OptiTrack marker cloud (`cloud.jsonl`). Labels are
assigned offline to the recording as a whole (`aerowhip.takes.sequence_labels`,
`python scripts/label_sequence.py <flight folders>`): every frame offers candidate cables,
chains of points from the attachment with each link at its marker interval length (one
marker may be skipped where it was not seen; the first point must lie below the
attachment, which excludes the drone's own markers), and the sequence of candidates with the
lowest total cost over the recording is chosen: link length errors, marker motion between
frames, and a credit per labelled marker. The steady hovers at both ends, where all ten
markers are visible, anchor the labels through fast or partly hidden motion. Where a second
sequence explains the recording almost as well, the markers on which the two disagree are
left unlabelled rather than guessed. Partial frames are normal data: hidden markers are
gaps, never filled in the labels.

The live filter (`aerowhip.deployment.cable_filter`), which must be causal, labels the same
clouds frame by frame and is scored against these labels; its output is kept beside them
(`cable_labels_filter.npz`). `python scripts/label_cable.py <folders>` opens the review
window to inspect either labelling and correct it by hand.

### Reconstruction of hidden markers

`reconstruct` fills the gaps separately, into `cable_reconstruction.npz`, with a method
per position: a marker hidden for at most 10 frames between two measurements is
interpolated in time (cubic, with the velocities at both ends); a marker with known
markers on both sides along the cable is placed at the cable distances from its two
nearest known neighbours, closest to its motion prediction; markers past the last known
one continue its direction. Measured against hidden known markers on four flights: time
gaps 1–2 mm, interior markers 5–10 mm, the free end 10–20 mm with outliers. The fit should
use measured markers as data and the reconstruction only where a complete cable is
needed (for example a simulation's initial cable state).

```bash
python scripts/label_cable.py --auto <flight folder> ...   # label and save, no window
python scripts/label_cable.py <flight folder>              # review window
```

The review window shows the frame's cloud in 3D with the labelled cable (markers coloured
from c1 at the drone to c10 at the tip), display-only estimates for missing markers, and,
over flight time, which markers are labelled in each frame, the flagged runs and the
length error between adjacent labelled markers against the flag tolerance. A
click selects a point; 1–9, 0 label it c1–c10; Delete marks the selected marker not
visible; Backspace returns it to automatic; A accepts a flagged run; Ctrl+S saves.

## Takes

A take is one Motive CSV export plus logger sidecars, loaded by
`aerowhip.takes.take_data.load_take`:

- **Motive export** in metres, Global coordinates, quaternion rotations, captured
  and exported at exactly 100 Hz. The drone rigid body is named `cf_N` (or as given
  in a `.tracking.json` file); the ten cable markers are labelled `cable:c1`…`cable:c10`.
- **Logger files** from the Crazyswarm flight logger
  (`scripts/experiment_logger_optitrack_pva_motor_v3.py` in the companion
  repository): `experiment_*.csv` with drone poses and commands on the host clock,
  plus `.commands.csv` and `.motors.csv`.
- **Clock alignment** in `<batch>/time_alignment.json`, mapping logger time to
  Motive time. Without it, commands are not attached to the take.

Put collections under `data/recordings/` (not tracked by Git).

### Checking a take

```bash
python -m aerowhip.takes.check_take path/to/take.csv
```

Reports export units, frame space and rotation type; capture and export rate
(must be 100 Hz); frame-number jumps and off-nominal steps; drone, cable and
unlabelled marker labels; per-marker completeness and longest dropout; whether
the hanging cable points along −Z; and gap-free 100 Hz runs of at least 1.8 s.
It exits with status 1 on a blocking problem.

### Estimating the clock offset

```bash
python -m aerowhip.takes.estimate_clock_alignment path/to/take.csv
```

Finds `offset_s` with logger time = Motive time + offset by matching drone
positions: a 10 ms grid search within ±30 s, refined to 0.1 ms, scored by RMS
position distance. It refuses to write an entry if the RMS exceeds 1 cm, the best
offset lies on the search boundary, or a different offset (> 0.25 s away) scores
within twice the best (for example a take with no motion — start each take with a
sharp vertical bump). The offset includes logger receipt latency; entries are
saved with `clock_verified: false`.

## Fitting

Fitting uses the calibration flights of one session through the native engine, on 2 s
windows every 1 s inside each flight's usable span (`aerowhip.fitting.calibration_set`);
two flights (`random_high`, `whip`) are held out of every fit and scored only with the final
parameters. Every reported number is computed twice.

1. **Drone pipeline** (`aerowhip.fitting.flight_drone`, `scripts/fit_drone_flights.py`,
   `scripts/fit_firmware_drone.py`): the sent acceleration commands drive the flight
   controller and plant open loop from the measured state. The thrust map and, for the
   identified law, the attitude response are identified directly from the measured thrust
   (mass × acceleration along the body axis) and body attitude. For the firmware law the
   controller gains are firmware constants and the vehicle around it is fitted instead: the
   roll/pitch inertia scale, the motor arm scale, the motor response time and the command
   delay (Nelder–Mead on the debiased open-loop position error of the drone-only flights);
   the cable flights are then scored with nothing refitted, raw and debiased, as the
   cross-check. The open-loop position error over 2 s, after removing a constant
   acceleration drift per window, checks the result.
2. **Cable** (`aerowhip.fitting.flight_cable`, `scripts/fit_cable_flights.py`): the measured
   drone pose drives the clamp; the cable starts from the measured markers and runs open
   loop. Loss: RMS distance between simulated and measured markers over the markers that were
   seen. Search: Nelder–Mead on log EI, log Cb, the normal drag coefficient and the linear
   drag.
3. **Coupled check** (`scripts/validate_coupled.py`): the full model, driven by the sent
   commands, scored on the cable flights for drone position and cable shape, with nothing
   fitted.

The identified values and their residuals are in `runs/calibration_<session>/`. EI is not
identifiable from flight data on this cable (its shape is set by gravity, inertia, damping
and air); the fits report it but it is not a measured property. The earlier take-based
fitting tools (`aerowhip.fitting.cable_fit`, `coupled_fit`, `run_refit`, the take-processing
application) remain for the older recordings.
