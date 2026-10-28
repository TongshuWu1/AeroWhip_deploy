---
name: new-airframe-redo-fits
description: "2026-10-28: cf_2 retired (motor/ESC); the user continues with cf_6 (same Bolt build, 156 g, props 1.5 cm / clamp 5.5 cm below markers); all fits redone on cf_6"
metadata:
  type: project
---

cf_2 was retired on 2026-10-28 (failed motor, suspected ESC channel after an emergency drop). The
user first planned a new lighter airframe, then chose cf_6: the same Bolt build, 0.156 kg,
propeller plane 1.5 cm and cable attachment 5.5 cm below the marker plane, same cable and clamp;
centre of mass taken 2.3 cm below the markers (moves with the frame). Config and flight stack are
switched (cf_6 enabled; scripts follow the enabled robot). All fits (thrust map, drone, cable,
residual) are still cf_2's and are to be redone from cf_6 flights; cf_2 data stays separate.

**Why:** cf_6's mass, marker mounting and wear differ; fitted drone parameters and the rotor-wake
residual do not transfer.

**How to apply:** treat current fitted values as placeholders for cf_6; before the first cf_6 fit
make the AeroWhip setup name a setting (code defaults to 'cf_2'). The reset bank must be rebuilt
(user's go-ahead). Handoff: AeroWhip_deploy/internal/handoff_2026-10-28.md.
Related: [[tune-drone-before-new-data]].
