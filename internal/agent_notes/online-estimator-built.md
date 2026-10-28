---
name: online-estimator-built
description: "2026-10-28 late: online identification built (scripts/online_fit.py, aerowhip/online); robust filter per the user's rule; its running model now feeds the planner (user approved; 5-min half-life, delay held)"
metadata:
  type: project
---

Built on the user's request ("finish the complete online fitting"). Moving 2 s window every 1 s, six
parameters (EI, Cb, Cd, motor response, drag xy, delay), information-form Gauss-Newton with the
offline fits as prior. User's rule, implemented as the robust filter: a flip or sudden change is
ignored unless it keeps happening (gated / held outliers / change only when 4 of 6 agree).
Validation: drone timing recovers on all flights; cable parameters only on informative flights
(circle, figure8); strokes/whip/random/decay do not identify the cable (reported as not passing).
2026-10-28 later, user's decision: apply it straight away as a running model (runs/online/model_state.json,
half-life 5 min of flight, delay held from planning, >4 offline sd held and reported); the planner and
preview use it. The user is moving flights to the other (AMD) computer, where it can run live.

**How to apply:** don't redesign; next steps are a live run beside a real flight and the
adoption question. Related: [[planning-tools-built]], [[no-heavy-compute-during-flight]].
