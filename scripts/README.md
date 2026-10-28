# Scripts

Run from the repository root (`python scripts/<name>.py --help` for each). Long fits and training
go through their windows (`python -m aerowhip.ui.fitting`, `scripts/train.py`).

**Identification loop** (most of these are buttons in `python -m aerowhip.ui.planning`)
- `optimize_trajectory.py` — the next excitation flight (Fisher information, safe on every model)
- `plan_flight.py` — fallback planner from motion primitives
- `preview_trajectory.py` — one simulation of a plan for the 3D preview
- `online_fit.py` — online identification of a flight (live or replayed); the running model
- `label_sequence.py`, `label_cable.py` — cable marker labels (whole recording; review window)
- `real_flight.py` — custom real: the flight stack with the live 3D view (`./docker/start.sh --custom-real`)

**Model fitting** (stages of the Model Fit window)
- `fit_thrust_map.py`, `fit_drone_segments.py`, `fit_cable_segments.py`, `validate_coupled.py`
- `fit_coupled_segments.py`, `coupled_sensitivity.py` — stage 4 coupled refinement and its sensitivity check
- `randomisation_from_fits.py` — training randomisation ranges from the adopted fits

**Training, policy and tools**
- `train.py`, `test_policy.py`, `build_reset_bank.py`
- `simulate.py`, `controller.py`, `controller_headless.py`, `gamepad.py`, `replay.py`, `play_recording.py`
- `custom_simulation.py` — the native simulation behind the ROS interface (`./docker/start.sh --custom-sim`)
- `build_engine.py` — compile the C++ engine

**Checks and diagnostics** (each produced evidence cited in the implementation record)
- planning: `bracket_check.py`, `closed_loop_check.py`, `tuning_prescreen.py`
- online fit: `online_validation.py`, `online_sweep.py`, `synthetic_flight.py`
- model: `stroke_diagnostics.py`, `step_convergence.py`, `solver_tolerance.py`, `benchmark_batch.py`
- sensing: `identify_sensing.py`, `validate_sensing.py`, `replay_observer.py`, `filter_velocity_check.py`
- lab scan (parked): `clean_lab_scan.py`
