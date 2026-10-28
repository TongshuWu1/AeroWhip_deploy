---
name: no-heavy-compute-during-flight
description: "Never run simulations, fits or other CPU-heavy jobs on the flight computer while the user may be flying; host stalls drop OptiTrack poses and trigger the motor-cut abort"
metadata:
  node_type: memory
  type: feedback
  originSessionId: d9ff1455-a877-4890-b4dc-ff0f79bcc835
  modified: 2026-10-28T15:01:21.598Z
---

Don't run anything CPU-heavy (simulator pre-screens, fits, validations, replays) on the flight
computer while the user is in the lab and may be flying. Light reads of finished logs and reports
(`nice -n 19`) are fine between flights.

**Why:** on 2026-10-28 at 10:59 I ran `tuning_prescreen.py` while the user flew a tuning flight;
the host stalled (OptiTrack pose gaps of 120–170 ms clustered in half a second, motor log gap
0.34 s at the same time), the flight script's 0.15 s pose-staleness check fired, and the drone
was dropped from 1.3 m by the emergency stop. Not proven, but the timing lines up.

**How to apply:** when the user is flying (lab session, stack up), do analysis only after they
say a flight has landed, and nothing heavy at all; queue simulator work for when they say they're
done flying. Related: [[no-ros-queries-during-flight]], [[run-long-jobs-in-the-ui]].
