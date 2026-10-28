---
name: new-airframe-redo-fits
description: "2026-10-28: cf_2 retired (motor/ESC); user builds a new lighter airframe; all drone-dependent stages (tuning, stages 0-4) are redone on it; work moves to the AMD 9950X computer"
metadata:
  node_type: memory
  type: project
  originSessionId: d9ff1455-a877-4890-b4dc-ff0f79bcc835
  modified: 2026-10-28T15:57:29.522Z
---

On 2026-10-28 the user decided not to repair cf_2 (failed motor, suspected ESC channel after an
emergency drop) but to build a new, lighter airframe, possibly with design changes. It will take a
while. When it exists: own robot name (flight scripts still assume `cf_2`: make it a setting),
physical.yaml entry, firmware 2026.08 flash, motor check, tuning with `--tune` starting from
extQuatStdDev 0.01 and kR_xy ~157500 / kw_xy 20000, then the attitude integral, then stages 0–4 from
new flights. cf_2's data and fits stay as the reference for the method, never mixed in.

**Why:** a new frame changes mass, inertia, motors, rotor wake on the cable and the clamp geometry,
so none of cf_2's fitted drone parameters or the cable residual transfer.

**How to apply:** don't propose fits on cf_2 data as if they were the final model; plan work as
"method ready, new drone pending". Handoff: AeroWhip_deploy/internal/handoff_2026-10-28.md.
Related: [[tune-drone-before-new-data]], [[mesh-decision-after-fitting]].
