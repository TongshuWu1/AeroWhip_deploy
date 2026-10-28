# Handoff — 2026-10-28, night (read this first)

Written at the end of a long session, before the user switches to the other computer. It replaces
`handoff_2026-10-28.md` for the current state; that file stays the record of the cf_2 results and
the fitting design. Details and evidence for everything below are in
`internal/implementation_record.md` (dated notes of 2026-10-28) and the user's standing rules in
`internal/agent_notes/` (copy them into your memory folder on the new computer, see below).

## Where things stand

The drone is **cf_6** (cf_2 retired: failed motor/ESC). cf_6 is tuned and the tune is frozen
(firmware 2026.08, extQuatStdDev 0.01, kR_xy 157500, kd_omega_rp 300, host PID KP 4 / KD 3 / KI 0.6,
battery compensation). Instead of a fixed calibration campaign, the project now runs a **planned
identification loop**: optimise one excitation flight, fly it, measure, label, refit, check, repeat.
All the tools for that loop are built, tested on recordings and synthetic flights, and committed,
and they all live in one window, **the Identification Loop** (`python -m aerowhip.ui.planning`).
An **online fit** estimates six drone/cable parameters every second during a flight; its estimate
is the **running model** the planner uses. **Nothing of this has flown for real yet.** Round 1
(`runs/planning/excitation_002.json`) is planned, rehearsed in the simulator and ready.

Pushed (code and data) to GitHub at the end of the session: AeroWhip `engine-v2`, crazyswarm2 `main`.
Cleanup before the push (see the record): dead scripts and modules removed, `scripts/legacy/` removed
from the flight stack, superseded internal notes in `internal/archive/`, `scripts/README.md` indexes
the scripts. The 9 failing tests in crazyswarm2 `test_acceleration_circle_cf2.py` are stale (written
before the arena limits, the estimator tuning and cf_6: fake drone outside the area, missing
parameters, robot name); the flight script itself is fine.

## Switching to the other computer (the user is doing this now)

1. Code **and data are on GitHub** (pushed 2026-10-28 night): clone both into one folder, side by side:
   `<folder>/AeroWhip_deploy` (branch **`engine-v2`**: `git checkout engine-v2` after cloning; `main` is
   older) and `<folder>/Crazyflie/crazyswarm2_jazzy_whipping` (branch `main`). The AeroWhip code finds
   the flight data beside it (`../Crazyflie/crazyswarm2_jazzy_whipping/calibration_data`), whatever the
   user name; `AEROWHIP_CALIBRATION_DATA` overrides it. A clone lacks `CLAUDE.md` (excluded from git
   locally); `AGENTS.md` has the same rules and the pointer here.
2. In git now: the flight data (`calibration_data/`: cf_6, cf_6_nocable, the archived cf_2 flights,
   manifest, envelope, plans, firmware bundles) and AeroWhip `runs/` (the archived cf_2 fits that are
   the current prior, reset bank, diagnostics, round 1, online validation runs) and the lab scan
   sources. **Not** in git, left on the old computer: the old cf_2 PPO training checkpoints
   (`runs/archive_cf_2_2026-10-28/ppo*`, 1.1 GB, a retired model), AeroWhip `output/` (9 GB of viewer
   and simulation evidence) and `tmp/` (4 GB scratch), the flight stack's `logs/`.
3. AeroWhip setup: `README.md` → Setup (venv, `pip install -e ".[build,dev]"`, `scripts/build_engine.py`).
   Added this session to the venv: `trimesh`, `pymeshlab` (lab scan only).
4. Flight stack: the Docker image (`crazyswarm2:jazzy-pinned`; `docker save`, or rebuild from
   `docker/`), the Crazyradio, the lab network for OptiTrack. Hand installs inside the container are
   listed in `handoff_2026-10-28.md` ("Where things are").
5. Memory: copy `AeroWhip_deploy/internal/agent_notes/*.md` into the Claude memory folder of the new
   machine (`~/.claude/projects/<key of the folder you start in>/memory/`; on the old machine the key
   was `-home-tony-lab`). `MEMORY.md` is the index.
6. Check after the move: AeroWhip `pytest` → 106 pass, 11 fail/error (all the missing reset bank,
   known); crazyswarm2 `pytest scripts/tests` → 22 pass, 9 fail (old `test_acceleration_circle_cf2.py`
   failures, known, unrelated). Open the window once: `python -m aerowhip.ui.planning`.
7. The old machine (i7-13700K) was unstable all day: random worker crashes, segfaults in tests,
   whole-PC crashes, and a corrupted system library (`libLLVM.so.20.1` failed `dpkg --verify`; it
   crashed any GL window of the custom simulator). Results there that reproduced are fine; do not
   chase its crashes as code bugs.

## How to run the loop (what the user will do next)

All from the Identification Loop window; the commands the buttons copy are listed so a session can
explain them. One command per block for the user, never chained.

1. **Plan** (step 1): round 1 exists: `excitation_002` (60 s free trajectory, peak 3 m/s² horizontal,
   ≤ 2 m/s² vertical, 0.1–1.2 Hz), checked on 9 models: the cable stays ≥ 6.8 cm from the drone
   (limit 5 cm), smallest motor 0.12 of hover share. Next rounds: "Optimise next flight"
   (`scripts/optimize_trajectory.py`, ~6 min on all cores; one level above the clean envelope).
2. **Fly** (step 2), in order:
   - "Start live view": opens custom real in its own Terminator window
     (`./docker/start.sh --custom-real --track-cable --simulated-cable --model <AeroWhip>/runs/online/running_model.json`):
     the stack, the live cable labeller, and the running model's cable simulated next to it.
     The Loop window stops rendering its own 3D view meanwhile.
   - "Start online fit (waits for the next flight)" before take-off
     (`scripts/online_fit.py --watch <calibration_data>/cf_6 --cpus auto`).
   - In the flight shell (`./docker/shell.sh --real`): the preflight check
     `python3 /workspace/scripts/acceleration_circle_cf2.py --setup cf_6 --trajectory /workspace/calibration_data/plans/excitation_002.json --check`,
     then the flight through the series runner (it records the manifest entry):
     `python3 /workspace/scripts/collect_series.py --setup cf_6 --trajectory /workspace/calibration_data/plans/excitation_002.json`.
     ("Copy flight command" copies the plan into `calibration_data/plans/` first.)
3. **Measure** (step 3): "Update envelope" (`scripts/envelope.py --setup cf_6`). Also
   `python3 scripts/radio_report.py calibration_data/cf_6/trajectory/*` (radio health; the radio
   trims below take effect from this flight).
4. **Label and fit** (step 4): "Label newest flight" (`scripts/label_sequence.py`), then the drone and
   cable refits in Model Fit (`python -m aerowhip.ui.fitting`; full fits go through the window).
   cf_6 needs its own drone fit (delay, motor response, drag): the closed-loop simulator under-commands
   sharp strokes (~20 %) until then.
5. **Check** (step 5): held-out validation in Model Fit; the Online page (error of all 10 markers vs
   the offline model, the running model table); the 3D view overlays the measured flight on the plan.

## What was built this session (all committed; see the record for each)

Planning: `aerowhip/planning/` (ensemble, simulate, Fisher information with estimate correlations,
Fourier trajectories), `scripts/optimize_trajectory.py`, `plan_flight.py` (primitive fallback),
`preview_trajectory.py`, `bracket_check.py`, `closed_loop_check.py`.
Window: `aerowhip/ui/planning/` (`loop_window.py`, `views.py`, `scene.py`, `loop_data.py`): loop rail,
3D preview with playback, Information / Safety / Envelope / Online / Log pages, measured-flight
overlay, live-view launch and 3D pause.
Online fit: `aerowhip/online/` (`stream.py` reads the flight folder as written; `estimator.py`;
`state.py` the running model), `scripts/online_fit.py`, `online_validation.py`, `online_sweep.py`,
`synthetic_flight.py` (known-parameter flights written like real ones).
Contact rule: `aerowhip/takes/contact.py`; fits skip failed takes (`calibration_set.usable`).
Flight stack: `--trajectory` mode and `TrajectoryFile`; `collect_series.py --trajectory`; trajectory
phase labelled `excitation <level>`; near-contact landing with learned drone markers; envelope clean
rule; `radio_report.py`; radio trims; legacy scripts moved to `scripts/legacy/`.

## Decisions the user made this session (do not reopen without them)

- Plan free optimised trajectories, not random primitives; Fisher-information scoring with coupling
  shown; simple loop, no "spaghetti".
- Use only the recorded span (hover start to hover end); takeoff and landing are never used.
- Online fit: robust filter — a flip or sudden change counts only if it keeps happening.
- **Apply the online fit straight away** as a running model: old history halves every **5 min of
  flight**; the online **delay is held** out of planning (timing convention, below); values beyond
  4 offline sd are held at that limit for planning.
- **A take where the cable touches the drone, or that is interrupted, is a failure and not used at
  all** (fits, validation, online fit). The cable rising above the clamp is allowed. Contact = cable
  within 1 cm of the drone (battery 4 cm / arms 7.6 cm / guard 11.4 cm cylinders) in 3 frames in a
  row. In flight: land when ≥ 2 cable points are within 5 cm for 0.15 s. Planner: every model's cable
  ≥ 5 cm from the drone.
- One live view during flights (custom real, launched from the Loop window); the Loop window's 3D
  pauses. A full merge of the two windows was proposed (Loop window's 3D page as the live view, the
  stack still its own process) and **not yet decided**.
- Radio: onboard `pose` topic off, `attitude_estimate` 50 Hz (from 100).
- Lab scan: parked ("use it later, not now"); the 3D view shows the plain floor.

## Verified, and not

Verified on recordings / synthetic / simulator: the trajectory flight end to end in the custom
simulation (collect_series --simulation; the simulator's mode-switch guard was fixed for it); the
online fit replayed on 85 recorded flights (no failure, every update < 0.9 s) and live on a synthetic
flight written in real time; recovery of known parameters on synthetic flights; rollback of the
running model on the cf_6 flip flight; contact detection on all cable flights (contact exactly on the
failed takes, plus cf_2 whip/002); the near-contact landing rule fires on every recorded flip and on
no normal flight; the window offscreen.
**Not verified: anything on the real drone with these tools** — the trajectory flight, the live view
launch (custom real with `--simulated-cable` was only checked up to the stack start), the online fit
live during a real flight, the running model updated by a real flight, the radio trims.

## Open findings (evidence in the record), roughly by importance

1. **The drone replay runs one 10 ms frame behind the closed-loop simulator** (same parameters,
   every window). The replay is what the offline drone fit and the online fit use; the planner and
   previews simulate closed loop. A fitted delay is therefore ~8–10 ms short in planning terms —
   likely part of the closed-loop under-commanding. Cause not found (starting the replay motors at the
   commanded thrusts made it worse; reverted). Until fixed the online delay is held from planning.
2. **The closed-loop simulator under-commands sharp strokes** (real host loop 8.9 vs simulated
   7.2 m/s² on the cf_6 flip stroke): needs the cf_6 drone refit, maybe item 1.
3. **Cable parameters are not identifiable from every motion**: strokes, whip, random and sharp stops
   fit a wrong start as well as the prior; circles and figure-8s recover them. The cable's static
   residual coefficients are almost perfectly correlated (P0~P1 −0.99) — reparametrising them
   (orthogonal basis) would help; EI couples with static z (+0.89), Cd with linear drag (−0.81).
4. **2 s online windows prefer a stiffer cable** than the 1 s offline fit (EI up ~0.3 log on every
   cf_2 flight from the prior) — a window-length dependence of the model, unresolved.
5. **Cd reads ~10 % high** on synthetic flights (likely initial cable velocities from noisy markers).
6. **Drone steps near the optimum often do not lower the window error** ("no gain"; the delay acts in
   2 ms ticks). Remedy to try: profile the delay on a grid as the offline fit does.
7. **The cf_2 fits used as the prior include failed takes**: 8 takes used through usable spans, plus
   whip/002 (cable touched the drone at 73 s). cf_2 is retired; the cf_6 fits will be clean.
8. **OptiTrack at 150 Hz is refused** by the loaders (offline and online): fly at 100 or 200 Hz, or
   build resampling (a data-preparation decision for the user).
9. **The reset bank is missing** (11 AeroWhip tests fail): rebuild from cf_6 recordings before training.
10. The contact verdicts write `contact.json` caches into flight folders; the raw-cloud envelope check
    and the labelled check can differ by millimetres at the 1 cm boundary (mix_a/002's 3 m/s² part).

## Next steps, in order

1. On the new computer: set up and check (above).
2. Fly round 1 (excitation_002) with the live view and the online fit running; then envelope, radio
   report, label, cf_6 drone and cable refits in Model Fit, validation. Watch the Online page's
   10-marker error against the offline model: that is whether the online fit learns.
3. Investigate the one-frame replay lag (finding 1) with `synthetic_flight.py` (known delay) — it
   decides whether the online delay can be used for planning.
4. Next rounds: optimise at envelope + 1 m/s² from the running model; repeat until the held-out score
   stops improving for two rounds; then freeze the model, rebuild the reset bank, train.
5. Ask the user about: merging the windows; resampling 150 Hz captures; refitting without failed
   cf_2 takes (optional); the lab scan when they want it (its cleanup was rewritten after running out
   of memory and has not been run; `configs/lab_scan.yaml` has the marked centre, +x assumed).

## Working rules (from the user; full notes in `internal/agent_notes/`)

- Approve scope before coding; update `internal/implementation_record.md` and `docs/` in the same
  change; never relax a criterion silently; preserve recordings and run artifacts (archive, never
  delete).
- Research decisions (reward, task, randomisation, model adoption) need the user's approval.
- Full refits and training go through their windows; only short checks run silently.
- Commands for the user: one per block, never chained with `&&`.
- Never fly the drone yourself; do not touch the radio or the drone (or start the flight stack) unless
  the user asks. No `ros2` CLI queries in the flight container during flights.
- Nothing CPU-heavy on the flight computer while the user may be flying; the online fit is the
  exception the user chose (pinned with `--cpus auto`, lowest priority).
- A take with cable contact or an interruption is a failure; partial cable markers are not.
- Another session of the user's was given an **audit-only role** ("analyse, audit, check, propose —
  do not change code unless asked", `role-audit-not-implement.md`). When unsure which role you are in,
  ask the user before editing.
