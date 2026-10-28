# GUI review and rework, 2026-10-27

Goal: research-grade windows — plots that carry physical meaning, real axes with SI units,
one consistent visual language, less decorative or redundant text. No physics, fitting,
reward, data format or file-writing behaviour was changed.

## Shared foundation

| Module | What it is |
|---|---|
| `aerowhip/ui/style.py` | Okabe–Ito palette with fixed meanings (measured near-black, reference grey dashed, command orange, model blue, error vermillion; x/y/z blue/green/vermillion); one Qt stylesheet (light, and dark for the flight viewer); matplotlib defaults (small fonts, no top/right spines, light grid, constrained layout). |
| `aerowhip/ui/plotting.py` | `PlotCanvas`: matplotlib canvas with the shared style, playback cursors and animated artists redrawn by blitting (static plot rendered once), click-to-seek signal; `gapped`, `rms`, `message` helpers. |
| `aerowhip/ui/motion_plots.py` | Time histories of one saved motion: applied (after-delay) vs achieved drone acceleration per axis, commanded vs achieved tilt, cable-tip speed and its component along the task output direction, motor thrusts. Used by every saved-motion replay. |
| `aerowhip/viewer/time_histories.py` | Rolling 10 s histories for the flight viewer. |

## Survey and decisions

| Window | Before | Noise removed | Added / changed |
|---|---|---|---|
| Training monitor (`ui/training/window.py`, `history.py`, `plots.py`) | 7 hand-painted QPainter charts over 3 tabs, mean-only returns, mixed units on one axis (peak rate, fold seconds), raw last-minibatch losses, a no-op "Approve motion" button, 24 px title | Separate Reward/Exploration tabs, mixed-unit "mechanics" chart, actor/critic raw loss chart, no-op button | One 3×2 learning-curve figure: return median and 10–90 % band per collection (from `collection-NNNNN.npz`) with evaluation markers and best-so-far; outcome fractions; return decomposition (task units, costs labelled as subtracted); mean KL after update with target and early-stop marks; critic explained variance before/after; entropy with the bonus coefficient on a labelled twin axis. Integer collection ticks; thousands separators on counters. |
| Whipping diagnostics (`whip_diagnostics.py`) | QPainter views with hand-drawn ticks, no colour bar, 3-line footnote | Footnote shortened to one line | Matplotlib: equal-aspect side/top views (m), bending map with colour bar (rad/m) and tracked bend centres, tip reach/height (m) and +x velocity (m/s) with cursors; blitted playback. |
| Saved-motion replay (`ui/training/replay_window.py`; Training "Latest evaluation", Replay browser, Policy tester) | Large heading, card with 36 px speed number, repeated "SAVED FINAL OUTCOME" card, two footnotes | Big-number card, vector readout card, duplicate footnote | `MotionPlots` beside the 3D view (commanded vs achieved acceleration and tilt, tip velocity, motor thrusts) with cursor and click-to-seek; one-line readout at the cursor. |
| Take manager (`ui/takes/manager_window.py`, `takes/position_plots.py`, `takes/quality.py`) | QPainter x/y/z strips with only min/max labels; pose/cable bars where "cable" meant all 10 markers seen | All-markers bar | Matplotlib x/y/z (m) with legend; tracking error panel |measured − command| with RMS over the selected interval (when commands exist); per-marker visibility raster (pose, c1–c10) with mean markers per frame — partial frames are normal; summary reports markers per frame first. |
| Take replay (`ui/takes/replay_window.py`) | Ad-hoc colours | — | Semantic colours shared with the plots. |
| Cable review (`ui/takes/cable_review.py`) | Count-of-labels line over frame index; rainbow marker colours | Frame-index axis | Per-marker label raster over flight time (s) with flagged runs and corrections; adjacent-marker chain-length error (mm) against the flag tolerance; viridis marker colours ordered c1 → c10; blitted cursor; click-to-seek in seconds. |
| Cable fit (`ui/fitting/cable_fit_window.py`) | Text log only | Log reduced to a short panel | Live convergence plot: each trial's fitting-set marker RMSE (cm), best so far, input model line. |
| Calibration comparison (`ui/fitting/calibration_window.py`) | Unstyled | — | Shared style; note de-emphasised; primary Run button. |
| Flight viewer (`viewer/simulation_window.py`; simulation, replay, real flight, recordings) | Own dark stylesheet; numbers only | — | Shared dark stylesheet; "Time histories" dock (View menu): drone acceleration per axis (states when the source does not provide it), tilt, cable-tip speed from successive tip positions, cable force and thrust when model forces exist. |
| Controller panel (`ui/controller/controller_window.py`) and dialogs | QPainter x/y/z with min/max only; red error colours inline | — | Matplotlib reference plot (x/y/z with units, title naming frame and origin) with blitted playback cursor; dark style matching the viewer; setup dialog styled. |
| Policy tester (`ui/policy_tester.py`) | Own stylesheet; Run enabled without a checkpoint (failed with "Is a directory") | — | Shared style; Run disabled until a checkpoint loads; results use `MotionPlots`. |

## Checks performed (all windows shown on DISPLAY=:1, never off-screen)

Pictures in `internal/archive/gui_screenshots/` are the windows' own renderings with the 3D views
pasted in from their render windows (Wayland blocks screen capture of the desktop).

| Window | Data | Picture |
|---|---|---|
| Training learning curves, latest evaluation, whipping diagnostics | A real 12-collection, 64-lane PPO run made for this test with the default experiment (`runs/ppo/20261027-020517-203204`, worktree only, 15 min on the GPU) | `training_learning_curves.png`, `training_latest_evaluation.png`, `training_whip_diagnostics.png` |
| Replay browser | Same run, evaluation of collection 12 | `replay_browser.png` |
| Policy tester | A real test of that run's `latest.pt` from the default initial case (native simulation, 4 s) | `policy_tester.png` |
| Take manager | `2026-10-27_cf_2_hover_start_01.csv` (Motive export, copied into the worktree) | `take_manager.png` (before: `before_take_manager.png`) |
| Cable review | `whip/2026-10-27_cf_2_whip_01` (read only; nothing edited or saved) | `cable_review.png` |
| Cable fit dialog, calibration comparison dialog, controller setup | Opened, not run | `cable_fit.png`, `calibration_comparison.png`, `controller_setup.png` |
| Flight viewer with controller | Native simulation, "Back and forth" preset, played 15 s | `flight_viewer_simulation.png`, `controller_panel.png` |
| Flight viewer, real-flight recording | `output/custom_real/20261016-221932-0bedc746/.../recording.json` (hand capture, played) | `flight_viewer_real_recording.png` |

`before_training.png` is the old training window with no run selected (no current-schema
run existed before the test run).

Not tested: live real flight (`--custom-real`, needs ROS and OptiTrack), the cable fit and
calibration comparison actually running (only opened), preset and B-spline dialogs (styled
through their parent only), and the old-schema runs in `runs/ppo_stage*` (their
`configuration.json` has no `experiment` key; the training window could not open them
before this change either).

## Record note

2026-10-27 · GUI rework (branch `gui-research`). Shared style (`aerowhip/ui/style.py`),
matplotlib canvas (`aerowhip/ui/plotting.py`) and saved-motion time histories
(`aerowhip/ui/motion_plots.py`); hand-painted charts in the training monitor, whipping
diagnostics, take manager, controller panel replaced by matplotlib figures with units;
training curves now show the per-collection return distribution, KL, critic explained
variance and entropy; replays show commanded vs achieved acceleration and tilt; the cable
review shows per-marker labels and chain-length error over time; the flight viewer has a
rolling time-history dock; the cable fit shows convergence. Behaviour unchanged except:
policy tester Run is disabled until a checkpoint is loaded. Checked visually on
DISPLAY=:1 with the data listed in `internal/archive/gui_review.md`; live real flight not tested.
