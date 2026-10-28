---
name: cable-markers-partial-frames
description: "OptiTrack often sees fewer than 10 cable markers; partial frames are valid data, never force all 10"
metadata:
  node_type: memory
  type: feedback
  originSessionId: ab84d596-c67b-413a-a8dc-e2a7a0e5d9ae
  modified: 2026-10-27T04:50:32.000Z
---

OptiTrack does not see every one of the 10 cable markers in every frame. A frame with
fewer markers is normal, valid data: the cable curve can still be constructed from the
visible ones. Never force a labelling to find all 10, never fill hidden markers with
estimates in saved data, and do not flag a frame just because a marker is missing.

**Why:** the user said so explicitly (2026-10-27) while the cable auto-labeller was being
built; forcing 10 would pick wrong points (reflections, other markers).
**How to apply:** in cable labelling (AeroWhip `aerowhip/takes/cable_labels.py`) keep gaps,
start from partial frames, flag only fixable gaps or geometry/jump errors. See
[[aerowhip-recording-redesign]].
