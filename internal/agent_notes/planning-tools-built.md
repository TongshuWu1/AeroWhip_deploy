---
name: planning-tools-built
description: "2026-10-28 evening: the planned-identification loop is implemented (flight-script --plan/--trajectory modes, cable-over-drone descent, envelope.py, ensemble + bracket check, closed-loop check, plan_flight.py, optimize_trajectory.py); round 1 ready; the plan is in the handoff"
metadata:
  type: project
---

Built 2026-10-28 evening (AeroWhip `aerowhip/planning/`, `scripts/plan_flight.py`,
`scripts/optimize_trajectory.py`, `bracket_check.py`, `closed_loop_check.py`; crazyswarm2
`--plan`, `--trajectory`, cable-over-drone descent, `envelope.py`). The agreed loop (handoff "THE
PLAN"): optimise a free excitation trajectory one level above the measured envelope, fly it, run
envelope.py, label + fit drone and cable (stroke windows weighed), check held out and closed loop,
repeat. Nothing flown yet with these tools; round 1 files exist under runs/planning/.
Key facts: the ensemble needs bending damping ÷3 to reach the measured flip; the simulated host
loop under-commands sharp strokes ~20 % until the drone is refitted on cf_6 data; fast horizontal
terms (>1.2 Hz) or vertical >2 m/s² drive a simulated motor to zero (motor-floor constraint is
real). Parked: live estimator, streaming re-planning, differentiable engine, bench measurements.

**How to apply:** don't rebuild or redesign these; run the loop. The user wants a clear simple
plan, no "spaghetti". Related: [[online-identification-rethink]].
