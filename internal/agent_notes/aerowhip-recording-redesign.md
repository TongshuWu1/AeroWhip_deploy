---
name: aerowhip-recording-redesign
description: Decided design for the new AeroWhip flight recording pipeline (to build after reorganization phases 3–4)
metadata:
  node_type: memory
  type: project
  originSessionId: ab84d596-c67b-413a-a8dc-e2a7a0e5d9ae
  modified: 2026-10-26T20:16:02.436Z
---

Decided 2026-10-26 with the user: old flight recordings (Motive CSV export + separate computer logger, two clocks) were bad and the recording pipeline will be redesigned and re-implemented, AFTER reorganization phases 3–4 of AeroWhip_deploy.

Design:
- One live recorder on the computer, one file, one clock: drone rigid-body pose, all raw marker points (NatNet IDs/flags from the patched /pointCloud), Motive frame number + timestamp, commands, motor telemetry.
- Live NatNet streaming in the user's setup does NOT provide marker-set labels, so cable marker identities come offline: label the saved .tak in Motive with the cable marker set, export CSV, join to the live recording by Motive frame number (no clock alignment needed). Live cable tracker is only a cross-check.
- Recorded flights use the acceleration command interface (same as the policy: FullState acc, Mellinger translation gains zeroed, host hover loop).

**Why:** two-file/two-clock takes needed fragile alignment and manual relabeling; fitting must identify the interface training simulates.
**How to apply:** when building the recorder, follow this design; package it under aerowhip/takes (and the Crazyswarm side for command logging). See [[aerowhip-reorganization]].
