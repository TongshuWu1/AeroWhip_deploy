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

`aerowhip.takes.cable_labels` labels the cable markers c1…c10 from `cloud.jsonl`:

- **Candidates:** points further than 7 cm from the drone's tracking origin; points
  closer than 1 cm to each other are merged.
- **Start:** the first frame where at least six markers each have exactly one candidate
  at their distance along the cable from the attachment (±2 cm), and no marker has two.
  Markers OptiTrack does not see stay unlabelled. This works for a swaying cable.
- **Tracking:** the geometric/temporal `CableTracker` (beam search over predicted
  positions with exact cable-length constraints) labels every later frame.
- **Flags:** a marker that is unlabelled while an unassigned point lies where it is
  expected, low tracker confidence, a neighbour further than the cable allows, and labels
  moving more than 8 cm between frames. OptiTrack often does not see every marker; a
  missing marker alone is normal and not flagged.

Labels go to `cable_labels.npz` (positions with NaN where a marker was not seen, point
indices, source, confidence, manual labels, checked frames) and a summary
`cable_labels.json`. Gaps are kept: the labels are measurements only.

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

The review window shows the frame's cloud in 3D with the labelled cable, a timeline of
labelled markers and flagged runs, and display-only estimates for missing markers. A
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

Fitting is staged; each stage writes a candidate model and never replaces the
active `configs/simulation.json` automatically.

1. **Cable material** (`aerowhip.fitting.cable_fit`): multipliers on EI, Cb and
   external drag, with the measured attachment motion driving the cable. Loss:
   weighted marker RMSE over an early and a late window per take. Search: a
   log-space grid followed by two local refinements.
2. **Vehicle response** (`aerowhip.fitting.coupled_fit`, `run_fit`): log-multipliers
   on the controller's position gain, velocity gain and attitude frequency, with the
   cable frozen in the coupled plant. Loss: tracked position (2 cm scale) and
   rotation error over 1.7 s windows. Optimizer: projected Adam on finite-difference
   or CUDA-graph gradients.
3. **Aerodynamic residual** (`aerowhip.fitting.residual_fit`, `run_residual_fit`):
   small non-negative drag-correction networks for cable and drone, fitted after a
   calibrated physical base. Training currently requires the unfitted residual.

`python -m aerowhip.fitting.run_refit --data <collection> [--base-model <model.json>] [--split <split.json>]`
runs stages 1–2 (the base model defaults to `configs/simulation.json`; the split file
lists training and comparison takes, decided before fitting) and writes everything
under `runs/refit/`. Comparison takes are evaluated only after a candidate is
chosen. A low fitting loss does not establish identifiability or reliable
prediction of whole manoeuvres.

The take-processing application (`python scripts/take_processing.py`) browses
collections, replays takes and runs calibration comparisons and cable fits.
