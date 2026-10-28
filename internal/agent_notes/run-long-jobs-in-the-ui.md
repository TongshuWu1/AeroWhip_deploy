---
name: run-long-jobs-in-the-ui
description: "Full refits and training runs must be started through their windows (Model Fit, Training) so the user can monitor them; only small checks may run silently"
metadata:
  node_type: memory
  type: feedback
  originSessionId: ab84d596-c67b-413a-a8dc-e2a7a0e5d9ae
  modified: 2026-10-28T00:01:20.657Z
---

Never start a full model refit or a training run as a silent background process. Start it through
its window (`python -m aerowhip.ui.fitting` for fits, `python scripts/train.py` for training) so the
user can watch progress, or give the user the command to run. Small, short checks (a smoke test of
a few minutes, a unit test, a quick analysis) may run quietly.

**Why:** the user (2026-10-28): "I don't want you to run things silently unless it's just a small
task; things like a full refit and training, I would like you to always run with UI, so I can
monitor it myself." Earlier that day two identical fits had run in parallel (one silent, one from
the window) and halved each other's speed.

**How to apply:** before a refit or training run, open or point to the window; if a run must be
scripted (e.g. a sweep), say so and ask first. See [[one-command-at-a-time]] for how commands are
given to the user.
