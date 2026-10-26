# Flight data and model fitting

> The recording pipeline is being redesigned: one live recorder (drone pose, raw
> markers with Motive frame numbers, commands and motor data on one clock), with
> cable-marker labels added offline from Motive and joined by frame number. This
> page describes the current take format and tools.

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
