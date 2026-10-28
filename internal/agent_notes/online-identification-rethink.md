---
name: online-identification-rethink
description: "2026-10-28 evening: calibration campaigns stopped; user wants bench measurements for what can be measured and flights planned over a horizon (information + coverage objective, chance constraints on cable angle, measured envelope, survivable flips) for what must be learned; design in internal/safe_planned_identification.md"
metadata:
  type: project
---

On 2026-10-28 evening the user stopped the mixed calibration series (two mix_a flights flipped
the cable over the drone at a 6 m/s² sharp stop; the simulator predicted 33° where the real cable
went to 170°) and asked for a rethink. Not online fitting: (1) a clear split between physical
properties measured on the bench (CoM by plumb lines, EI and damping of a short cantilever, clamp
stiffness, thrust map and motor time constant on the stand, optional bifilar inertia) and what is
learned from flight (delays, drag, rotor wash, cable damping under large motion, clamp under
dynamic load); (2) data collection planned over a horizon: command sequences scored by Fisher
information + state-space coverage, chance constraints evaluated on a model ensemble that must
bracket the observed flips, a measured envelope stepped outward one level at a time, and a
cable-over-drone descent so flips are survivable. Design: internal/safe_planned_identification.md
(after a literature pass: Hausman 2016, IDTD/ASID, Koller 2018, swing-angle MPC).

**Why:** campaigns are slow, brittle to small hardware changes, and found the unsafe level by
breaking cf_2.

**How to apply:** don't propose resuming the series; build in the order listed in that document
(planned-trajectory flight mode + cable-over-drone descent, envelope bookkeeping, ensemble with
the bracket check, planner). First flights stay inside the measured envelope (3 m/s² sharp stops
flew at 59°). Related: [[tune-drone-before-new-data]], [[new-airframe-redo-fits]].
