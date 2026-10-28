---
name: tune-drone-before-new-data
description: "2026-10-28 decision: fully tune the drone (trim, extQuatStdDev, attitude gains) before recording more flights or refitting; i7 machine stays the flight computer until then; status at end of 2026-10-28"
metadata:
  node_type: memory
  type: project
  originSessionId: d9ff1455-a877-4890-b4dc-ff0f79bcc835
  modified: 2026-10-28T15:54:53.635Z
---

On 2026-10-28 the user decided to tune the drone completely before collecting more calibration data
or refitting. Order: estimator (extQuatStdDev), attitude stiffness (kR_xy/kw_xy), attitude integral
(ki_m_xy), then re-fly the drone-only and cable series at the final settings and refit stages 1–4.
Tuning flights: `acceleration_circle_cf2.py --tune PLAN` (config/tuning.yaml), cable off, take
names `tune_*`; report with `scripts/tuning_report.py`.

Status end of 2026-10-28: firmware 2026.08 flashed; extQuatStdDev 0.01 adopted; kR_xy 157500 chosen
(kw_xy default) but commented out in the YAML; integral not flown. cf_2 grounded: a motor failed
after an emergency drop and the replacement does not spin (ESC channel suspected); a colleague will
fix it or another Bolt is used (then: own robot name, physical.yaml entry, firmware flash, gain
re-check, data kept separate from cf_2). Abort rule changed on the user's decision (2026-10-28): a lost link lands the drone (crazyswarm2 e9d246d).

Fresh start on cf_6 (2026-10-28, user): all cf_2 data, logs and runs archived (dated archive
folders), manifest emptied to cf_6; tuning restarts with plans cf6_baseline, cf6_estimator,
cf6_stiffness, cf6_damping, cf6_integral (config/tuning.yaml).

**Why:** the drone's open-loop drift and slow attitude response were the main sim-to-real gaps;
all new data should share one drone setting.

**How to apply:** don't propose calibration flights or refits until tuning is done. Details in
AeroWhip_deploy/internal/implementation_record.md (note of 2026-10-28 lab). See
[[no-heavy-compute-during-flight]], [[one-command-at-a-time]].
