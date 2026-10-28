---
name: tune-drone-before-new-data
description: "2026-10-28 decision: fully tune the drone (trim, extQuatStdDev, attitude gains) before recording more flights or refitting; this i7 machine stays the flight computer until then"
metadata:
  node_type: memory
  type: project
  originSessionId: d9ff1455-a877-4890-b4dc-ff0f79bcc835
  modified: 2026-10-28T14:02:17.969Z
---

On 2026-10-28 the user decided to tune the drone completely before collecting more calibration data
or refitting. Order: re-measure the hover trim; extQuatStdDev 0.05 vs 0.01 (hover tests); onboard
attitude gains (kR_xy/kw_xy were never tuned, still stock Crazyflie defaults on the 157 g Bolt; the
position gains the user tuned by eye are zeroed in flight). Tuning flights without cable, take names
`tune_*`. Then re-fly the drone-only series and the cable series at the final settings, and refit
stages 1–4.

**Why:** the drone's open-loop drift (~1 m in 5 s) comes partly from estimate error (1.25° RMS) and a
stale trim (~0.2 m/s² in y); the attitude loop lags 120–140 ms. All new data should share one setting.

**How to apply:** don't propose recording calibration flights or refits until tuning is done. The
i7 machine stays the flight computer until then; the move to the AMD 9950X comes after
(internal/handoff_2026-10-28.md). See [[aerowhip-bench-stand]], [[one-command-at-a-time]].
