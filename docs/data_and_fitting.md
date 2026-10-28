# Flight data and model fitting

The model is fitted on calibration flights recorded by the companion repository's flight
script.

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
(`cable_labels_filter.npz`). The filter and the observer around it (drone velocity and body rate
from the last poses) run as native code (`native/estimator.cpp`), the same in flight, in the
training environment and offline; they follow the Python reference implementations kept beside them
(`ReferenceCableFilter`, `ReferenceObserver`) statement by statement, and on a recorded whip flight
(8,701 frames) give identical labels and bit-identical marker states, about 16 times faster. `python scripts/label_cable.py <folders>` opens the review
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

## Fitting

Fitting uses the calibration flights through the native engine
(`aerowhip.fitting.calibration_set`), inside each flight's usable span. Flights are stored by
category (`<setup>/<take>/<NNN>/`) and every flight of a category is pooled, whatever the day.
Which flights exist, their roles (`fit`, `held_out`, `hover`, `excluded`), usable spans and
logs, the vehicle measurements and the bench tests all come from one manifest beside the
data, `model_calibration.yaml` (`aerowhip.fitting.manifest`), written during the sessions
and checked by the data repository's `check_calibration_data.py`; nothing about the data set
is hard-coded in the fitting code. Flights with the role `held_out` are never used by a fit and
are scored separately. A cable flight is used only once its
markers are labelled (`scripts/label_sequence.py` writes `cable_labels.npz`); loading an
unlabelled cable flight is an error. Every reported number is computed twice.

The stages, their scripts and how a result is adopted into the configuration files are listed
in `aerowhip.fitting.stages`. The Model Fit window (`python -m aerowhip.ui.fitting`) runs a
stage as its own process, follows its progress (each fit writes `<result>.progress.jsonl`
beside its result, `aerowhip.fitting.progress`; fits started from a terminal are followed
too), compares a result with the configuration and adopts it only on an explicit click; each
adoption is appended to `runs/fits/adoptions.jsonl`. New results are written to
`runs/fits/<stage>/`.

1. **Thrust map** (`scripts/fit_thrust_map.py`): from the motor PWM logs of the drone-only
   flights, the map T = κ(1+β(V−V0))/4 Σ(pwm/massThrust)^p is fitted so that its mean over
   2500-count PWM bins matches the thrust the measured motion requires (mass × acceleration
   along the body axis, with the drone drag). The four motors swing several thousand counts
   about their mean in flight, which the simulated firmware does not reproduce; because the map
   is convex this raises the real mean thrust, and the measured factor (about 1.07) is folded
   into the collective κ the simulation uses.

**Least squares on short segments** (`aerowhip.fitting.least_squares`; stages 2–5).
Each flight is cut into consecutive segments (1 s by default, `--horizon`), and every segment is
replayed from the measured state at its start (multiple shooting), so errors cannot build up over
long open-loop horizons. Commands still in the delay line at a segment start are applied
(`DroneReplay.run(pipeline=True)`). Scipy's trust-region least squares fits the parameters within
bounds with a finite-difference Jacobian. Segments that fail at the start point are dropped once,
with their reasons recorded; a failure during the search is counted and gives a large fixed
residual, so the step is rejected; only numerical failures are caught. The 95 % intervals use the
covariance clustered by flight (sandwich estimate), because residuals within a flight are strongly
correlated; on synthetic data with flight-level errors it matches the true scatter of the
estimates (0.106 against 0.109), whereas the independent-residual formula is 5–9 times too small.

2. **Drone** (`scripts/fit_drone_segments.py`, drone-only flights): roll/pitch inertia, motor
   response and body drag (the measured motor arm kept), with the command delay profiled on a grid
   (it acts only at the 2 ms controller ticks, so the loss is a staircase in it). Two residuals
   per frame, each normalised to 1 at the configuration now: the position error after removing
   one constant acceleration bias per flight and axis (attitude-estimate offsets and trims the
   host loop absorbs; three numbers per flight cannot absorb dynamics as a per-window bias can),
   and the roll/pitch error after removing one constant offset per flight. `--estimator` adds the
   onboard attitude-estimate model of the firmware configuration.
3. **Cable** (`scripts/fit_cable_segments.py`, cable flights): log₁₀ EI, log₁₀ Cb, Cd and log₁₀
   linear drag, jointly with the cable residual (`--terms`: the steady force, searched in a
   Legendre basis along the cable, `--vertical constant` for its vertical part; the drag
   correction; `--frame body|world`; `--fix`/`--set` hold parameters); the measured drone pose
   drives the clamp and each segment starts from the measured markers; residuals are the marker
   position errors of every seen marker. On 820 fit and 249 held-out segments (held-out RMS):
   physics only 33.5 mm; the steady force with everything free 22.9 mm, but EI then falls to its
   bound and the vertical shape terms cancel one another; the adopted fit (steady force, vertical
   part constant, EI free) 23.3 mm with every parameter identified; body and world frame are
   indistinguishable on these flights (22.9 / 23.0 mm, the drone stays nearly level); the drag
   correction was not identified (intervals of ±18 in Cd, ±33 in the log multiplier); a clamp angle
   in place of the sideways force needs 15° or more and gives 24.6 mm.
4. **Coupled sensitivity** (`scripts/coupled_sensitivity.py`) and **refinement**
   (`scripts/fit_coupled_segments.py`): the full model (drone driven by the sent commands,
   carrying the cable) from the measured drone state and markers, with three residuals each
   normalised to 1 at the configuration now (`aerowhip.fitting.coupled`): drone position (one
   acceleration bias per flight removed), drone tilt (one offset per flight removed) and cable
   shape (each marker relative to the drone). The sensitivity run differentiates this loss with
   respect to the centre-of-mass height, the attachment height and the four cable parameters
   and gives the intervals a joint fit would have. On the cable fit flights
   (`runs/fits/coupled_sensitivity/20261027-210207.json`) every one-step change moves the loss by
   less than 0.4 %, and the joint 95 % half-widths are ±35 mm for the centre-of-mass height and
   ±14 mm for the attachment height (measured to ±3 mm): the coupling geometry is not
   identifiable from these flights, so it is measured, not fitted.
5. **Coupled validation** (`scripts/validate_coupled.py`): the same coupled residuals with
   nothing fitted, reported per flight and pooled, separately for the fit and the held-out
   flights: drone position, tilt, cable shape and tip (last marker) errors; then again with each
   flight's drone bias cancelled in the flight controller (the bias drags the clamp); then the
   **stroke test**: 0.6 s from each stroke onset of the decay, strokes and whip flights, the tip's
   peak speed against the measured one, the tip error over the stroke, drone position and tilt.
   Held-out results with the adopted cable (`runs/fits/coupled/20261028-015741.json`), bias
   cancelled: shape 28.8 mm and tip 46.1 mm per 1 s segment (38.2 and 60.9 mm before the
   residual), drone position 48.7 mm and tilt 1.0° (unchanged); strokes: tip error 31.7 mm (36.7
   before), tip peak speed 0.90 of the measured (0.94 before), one frame late.

## Planned identification

Each identification flight is planned: `scripts/optimize_trajectory.py` searches for a free
trajectory (a sum of sines per axis, at rest at both ends, 0.1–1.2 Hz) whose peak horizontal
acceleration is one level above what the cable has flown cleanly (the measured envelope), that
every member of an ensemble of fitted models predicts safe (the cable stays below the
attachment, motors above a floor), and that teaches the fit the most about the parameter it knows
least: the Fisher information of the simulated marker and attitude measurements, scored per
parameter after the others are allowed to vary (so coupled parameters count as poorly
determined), with the correlations of the estimates reported. The Identification Loop window
(`python -m aerowhip.ui.planning`) runs the loop: optimise, fly, measure the envelope, label and
fit, check. It plays each planned flight back in 3D from one simulation of the adopted model
(`scripts/preview_trajectory.py`), with the predicted cable height above the clamp along the
timeline. Only the recorded span from the settling hover to the final hover is used; takeoff,
landing and safety landings are excluded, since the cable may then touch the ground.

## Online identification

`scripts/online_fit.py` (package `aerowhip.online`) estimates the parameters a flight can teach
while it flies, every second, from the newest 2 s of the flight. It reads the flight folder as the
flight script writes it (the logs are written line by line), so it needs no connection to the
flight stack; a recorded flight can be replayed through it as if live (`--replay`).

- **Labels:** the cable markers come from the same cable filter that runs in flight (the native
  observer), applied to the recorded marker cloud frame by frame; a marker counts where the filter
  associated a point to it with confidence.
- **Parameters:** the cable's bending stiffness, bending damping and drag coefficient (log10 EI,
  log10 Cb, Cd), and the drone's motor response, sideways body drag and command delay. The prior is
  the offline fits (values, standard deviations, correlations). Parameters that a single window
  cannot separate (the cable's linear drag against Cd, the static residual's coefficients against
  each other) stay at the offline fit.
- **Update:** for each family, the window is replayed at the current estimate and with each
  parameter perturbed (the cable driven by the measured drone pose; the drone by its sent commands,
  from the measured state and cable), and one Gauss-Newton step is taken in information form with
  the current covariance as prior. A window counts as a few independent samples (its samples are
  strongly correlated), the measurement noise starts at the offline fit's typical residual, and a
  step is applied only if replaying it lowers the window's error.
- **Robust filter:** windows outside the recorded motion, with the cable over the drone, with the
  drone tilted beyond 60° or with too few markers are not used. A window whose step is improbable
  under the current uncertainty, or whose error the model cannot explain, is held; it counts only
  if most of the following windows agree with it (then the pooled change is applied with the
  uncertainty widened), and expires otherwise. So a single flip or gust does not move the estimate,
  while a real change (a different battery, a new cable) does.

The output (`runs/online/*.jsonl`) holds the estimate and its standard deviation after every
window and what happened to each window; the Identification Loop window plots it live.
`scripts/online_validation.py` replays recorded flights from the prior and from a deliberately
wrong start and scores the whole flight at each result.

**The running model.** The online estimate is carried from flight to flight in
`runs/online/model_state.json`: each real flight followed live starts from it and updates it
every window, and old evidence loses half its weight every 5 minutes of flight (the uncertainty
never grows beyond the offline fit's). The planner (`optimize_trajectory.py`, `plan_flight.py`),
its model ensemble and the 3D preview use it in place of the offline fits for the online
parameters, with two holds: the command delay stays at the offline fit (the replay that estimates
it and the closed-loop simulation that plans with it differ by one 10 ms frame), and a value more
than 4 offline standard deviations from the offline fit is held at that limit and reported. The
measured envelope remains the safety authority. Replays and synthetic flights do not change it
unless asked (`--state`); "Reset to the offline fit" keeps the old file under a dated name.
A change in the estimate counts only if it lowers the error of the window it is applied on; the
motor response and body drag are bounded to physically possible values. Flights captured at
rates other than 100 or 200 Hz are refused with a message (the fitting windows need one of them).

## Failed takes and cable contact

A take is a failure, and is used by nothing that learns from data (offline fits, validation, the
online fit and its running model), when it did not complete (interrupted, landed early) or when the
cable touched the drone. The cable rising above the clamp without touching the drone is allowed.
Contact (`aerowhip.takes.contact`): the drone is stacked cylinders in its body frame (the battery,
radius 4 cm, from the clamp; the arms and plates from the CAD meshes, 7.6 cm; the propeller guard
ring, 11.4 cm, to 2 cm above the marker plane); the cable is its labelled markers and the chords
between seen neighbours, without its first 4 cm; contact is any of it within 1 cm of the drone in
3 frames in a row. The verdict is computed from the recorded marker cloud with the in-flight
labeller and cached as `contact.json` in the flight folder. The planner requires every model's cable
to stay at least 5 cm from the drone (from the first marker on). The online fit skips windows with
contact and, after a failed take, restores the running model to its state before the flight.

The Online page shows each window's error over all ten cable markers before the window updates the
model (a prediction), against the offline model on the same window, per marker and as the running
improvement: the measure of whether the online fit is learning.
