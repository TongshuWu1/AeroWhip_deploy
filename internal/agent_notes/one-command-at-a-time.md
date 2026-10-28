---
name: one-command-at-a-time
description: "When giving the user shell commands to run, give one command per block, never chained with &&"
metadata:
  node_type: memory
  type: feedback
  originSessionId: ab84d596-c67b-413a-a8dc-e2a7a0e5d9ae
  modified: 2026-10-27T22:13:17.237Z
---

Give the user one command at a time to run; never chain commands with `&&` (or `;`) in commands meant for them.

**Why:** the user said "don't use &&, I want to do one command at a time" (2026-10-27, lab session) — they want to see each step succeed before the next, e.g. `./docker/compose` then `./docker/start.sh --custom-real`.

**How to apply:** every command the user runs gets its own fenced bash block, in order; a `cd` is its own step too.
For calibration flights: one take per command (`python3 scripts/collect_series.py --setup cf_2 --only whip`),
never several takes in one `--only`; and don't repeat the stack start-up commands — the user already knows them.
