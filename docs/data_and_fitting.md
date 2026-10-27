# Flight data and model fitting

Two recording formats exist: calibration flights recorded by the companion
repository's flight script (below), and the earlier takes (Motive CSV plus logger
sidecars, further down). The fitting tools currently read the earlier takes.

## Calibration flights

The companion repository (`crazyswarm2_jazzy_whipping`, `DATA_COLLECTION.md`) flies a
fixed series through the acceleration interface and writes one folder per flight under
`calibration_data/<setup>/<take>/<NNN>/`: `parameters.json`, `flight.jsonl` (50 Hz host log),
`poses.jsonl` (every OptiTrack drone pose), `motors.jsonl` (motor PWM and battery, 30 Hz) and,
with the cable, `cloud.jsonl` (every OptiTrack marker cloud, unlabelled, on the same stamps as
`poses.jsonl`).

The pose stamps are host receipt times: they jitter by a few milliseconds (5th–95th percentile
8.6–12.1 ms between 100 Hz frames) while the cameras deliver exactly one frame per period, and
frames are occasionally dropped. `aerowhip.takes.calibration_flights.camera_frames` recovers each
pose's camera frame number: a late pose raises its offset from the nominal clock only until the
next pose catches up, a dropped frame raises it for every later pose, so the dropped count is the
suffix minimum of the whole-frame offset; the drift between the camera and host clocks (−79 to
+73 ppm across the flights) is fitted and removed first. On every flight the recovered drop count
equals the count implied by the recording length, and second-difference accelerations computed on
camera frame times have 0–16 spikes above 20 m/s² per flight where receipt times give 185–539.
Captures at 200 Hz are thinned to the even camera frames (the 100 Hz grid); they drop more frames
(up to 1.7 %, in bursts of up to 150 ms) than 100 Hz captures (0.1–0.8 %) but give more complete
cable labels. Fitting segments require consecutive camera frames (no dropped frame), and their
times are camera frame times.

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

The recorded stamps are host receipt times: the cameras deliver exactly one frame per 10 ms
(the mean step over a flight is 10.00 ms), but the stamps jitter between about 5 and 16 ms
and occasionally stall for 20–40 ms before a burst. The filter therefore runs on a frame
clock that advances by whole frames and follows slow drift; using the raw stamp differences
put the jitter into every velocity and made the filter lose the cable during the whip
strokes. With the frame clock it agrees with the whole-recording labels on 99.96 % of the
labelled marker-frames of the whip flight and labels 99.4 % of them (84 % and 86 % with raw
stamps). `scripts/filter_velocity_check.py` scores its marker velocities against central
differences of the whole-recording labels: 0.08–0.26 m/s RMS per flight, the tip within
0.1–0.2 m/s, and at every snap above 3 m/s that the labels cover (83 on the nine flights)
the filter's peak tip speed is within 1 % of the reference in the median, about one frame
(10 ms) later. A few snaps right after a delivery stall are under-read (down to 52 % of the
peak); publishing the camera frame number with the cloud would remove them.

### OptiTrack sensing statistics

The training environment lets the policy observe through a model of the cameras
(`aerowhip.envs.sensing`), identified from the same recordings by
`scripts/identify_sensing.py`. Two levels are kept apart. At the camera level, the points
within the cable's reach of the attachment are counted against the ten markers: a shortfall
is a marker the cameras did not report. On the nine cable flights of 2026-10-27 the cameras
missed at least one marker in 0.1–4.4 % of the frames (the vertical flight worst), and the
per-marker, per-frame hazard of disappearing is 8·10⁻⁴ at rest, 1–3·10⁻⁴ between 0.25 and
4 m/s, 7·10⁻⁴ at 4–8 m/s and 0.035 above 8 m/s (22 frames); a hidden marker reappears with
probability 0.32 per frame (from the filter's labels; the camera-level run length is not
observable). White position noise of a resting marker is 0.1 mm per axis (drone rigid body
0.02 mm and 0.5 mrad in rotation); receipt stamps jitter by about ±3 ms around the frame
times (stalls are not modelled). The cloud also holds the drone's own markers (two of them 8–9 cm from the tracking
origin, beyond the 7 cm exclusion) and about three static points elsewhere in the room.
`scripts/validate_sensing.py` scores the model at the level the policy sees: the longest
physically possible span of each flight's reconstructed cable is turned into synthetic clouds,
the filter labels them, and its label statistics are compared with the filter run over the
real clouds of the same span. The fraction of marker-frames labelled agrees within 1 % on every
flight (whip 99.6 % simulated, 99.3 % real). The real filter leaves more frequent but shorter
gaps on the flights with clutter near the cable (frames with a gap: decay 1.4 % simulated versus
4.3 % real, line_star 1.4 versus 6.0, whip 1.1 versus 4.6; mean gap 1.4–2.1 frames real versus
3–7 simulated), because ambiguous real points are not modelled. The absolute
capture-to-host latency cannot be measured from these files (receipt jitter is below 0.5 ms;
the clock offset needs the onboard estimate against OptiTrack), so the latency is an assumed
range of 0–2 frames, drawn per lane and episode.

### The flight estimator on the recordings

`scripts/replay_observer.py` runs the flight estimator over each cable flight as the flight
node will receive it: receipt stamps, rigid-body pose, unfiltered cloud. Against central
differences of the recorded poses, its drone velocity is within 16–55 mm/s RMS (whip 55,
above 1 m/s up to 0.18 m/s) without measurable lag, its body rate within 0.24–0.36 rad/s (pose
noise alone accounts for about 0.05 rad/s; the rest is attitude motion faster than the 40 ms
fit), and all ten markers are labelled in 92–99.5 % of the frames; 0.4–0.5 ms per frame. In
the training environment the same estimator on simulated frames is within 10 mm/s and
0.39 rad/s of the true state.

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

Fitting uses the calibration flights through the native engine
(`aerowhip.fitting.calibration_set`), inside each flight's usable span. Flights are stored by
category (`<setup>/<take>/<NNN>/`) and every flight of a category is pooled, whatever the day.
Which flights exist, their roles (`fit`, `held_out`, `hover`, `excluded`), usable spans and
logs, the vehicle measurements and the bench tests all come from one manifest beside the
data, `model_calibration.yaml` (`aerowhip.fitting.manifest`), written during the sessions
and checked by the data repository's `check_calibration_data.py`; nothing about the data set
is hard-coded in the fitting code. At present every flight has the role `fit`; evaluation
motions not used by any fit are to be recorded separately. A cable flight is used only once its
markers are labelled (`scripts/label_sequence.py` writes `cable_labels.npz`); loading an
unlabelled cable flight is an error. Every reported number is computed twice.

The stages, their scripts and how a result is adopted into the configuration files are listed
in `aerowhip.fitting.stages`. The Model Fit window (`python -m aerowhip.ui.fitting`) runs a
stage as its own process, follows its progress (each fit writes `<result>.progress.jsonl`
beside its result, `aerowhip.fitting.progress`; fits started from a terminal are followed
too), compares a result with the configuration and adopts it only on an explicit click; each
adoption is appended to `runs/fits/adoptions.jsonl`. New results are written to
`runs/fits/<stage>/`.

0. **Thrust map** (`scripts/fit_thrust_map.py`): from the motor PWM logs of the drone-only
   flights, the map T = κ(1+β(V−V0))/4 Σ(pwm/massThrust)^p is fitted so that its mean over
   2500-count PWM bins matches the thrust the measured motion requires (mass × acceleration
   along the body axis, with the drone drag). The four motors swing several thousand counts
   about their mean in flight, which the simulated firmware does not reproduce; because the map
   is convex this raises the real mean thrust, and the measured factor (about 1.07) is folded
   into the collective κ the simulation uses.

**Least squares on short segments** (`aerowhip.fitting.least_squares`; current stages 2 and 3).
Each flight is cut into consecutive segments (1 s by default, `--horizon`), and every segment is
replayed from the measured state at its start (multiple shooting), so errors cannot build up over
long open-loop horizons. The residuals are per-frame position errors; scipy's trust-region least
squares fits the parameters within bounds with a finite-difference Jacobian. Segments that fail at
the start point are dropped once, with their reasons recorded; a failure during the search is
counted and gives a large fixed residual, so the step is rejected; only numerical failures are
caught. The 95 % intervals use the covariance clustered by flight (sandwich estimate), because
residuals within a flight are strongly correlated; on synthetic data with flight-level errors it
matches the true scatter of the estimates (0.106 against 0.109), whereas the independent-residual
formula is 5–9 times too small.

- **Drone** (`scripts/fit_drone_segments.py`, drone-only flights): roll/pitch inertia, motor
  response and body drag (the measured motor arm kept). One constant acceleration bias per flight
  and axis is removed analytically (attitude-estimate offsets and trims the host loop absorbs);
  three numbers per flight cannot absorb dynamics as a per-window bias can. The command delay
  acts only at the 2 ms controller ticks, so the loss is a staircase in it: the other parameters
  are fitted at each delay of a grid and the profile is reported. Commands still in the delay line
  at a segment start are applied (`DroneReplay.run(pipeline=True)`).
- **Cable** (`scripts/fit_cable_segments.py`, cable flights): log10 EI, log10 Cb, Cd and log10
  linear drag; the measured drone pose drives the clamp and each segment starts from the measured
  markers; residuals are the marker position errors of every seen marker.

The earlier Nelder–Mead fits below are kept for their retained results.

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
   acceleration drift per window, checks the result. The drone's linear air drag
   (`scripts/fit_drone_drag.py`) is fitted afterwards on the same flights with everything
   else fixed, on the raw error (it is a slowly varying force): horizontal 0.52 s⁻¹,
   vertical 0.85 s⁻¹. It roughly halves the raw open-loop error (drone-only held out 265 →
   134 mm, cable flights held out 250 → 164 mm) and lowers the debiased error (57.7 → 42.2
   and 54.2 → 45.6 mm). The vertical rate is larger than a regression of the residuals on
   velocity suggests (0.2–0.33 s⁻¹) and may also absorb vertical thrust dynamics; it should be
   re-checked once the motor map is identified on the bench.
   `scripts/cable_error_split.py` separates the cable model's own error from the drone's: the
   same windows are replayed with the measured drone pose driving the clamp. On the whip flight
   the cable alone accounts for 68 of the 74 mm coupled cable-shape error (tip 127 of 135 mm),
   and on every flight for most of it; the error grows with marker speed (whip: 39 mm below
   1 m/s, 130–190 mm above 2 m/s) and along the cable. A finer mesh (two or three segments per
   marker interval) changes it by 1.5–2.3 mm; the bending damping Cb is the sensitive term
   (lower Cb helps the whip and hurts the slower flights), so the fitted value is a compromise
   across speeds.
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
