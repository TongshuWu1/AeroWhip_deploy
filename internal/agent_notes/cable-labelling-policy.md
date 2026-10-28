---
name: cable-labelling-policy
description: "Offline cable labels come from a whole-recording sequence pass anchored on the hovers; uncertain markers stay gaps, never guesses; more flights can be flown to fill holes"
metadata:
  node_type: memory
  type: feedback
  originSessionId: ab84d596-c67b-413a-a8dc-e2a7a0e5d9ae
  modified: 2026-10-27T06:46:34.145Z
---

For cable marker labels used in model fitting, label the recording as one sequence (Viterbi
over per-frame candidate chains, anchored by the start/end hovers where all 10 markers are
visible), not frame by frame. When two labellings explain the data almost equally, leave
the disagreeing markers unlabelled.

**Why:** The user (2026-10-28) saw wrong labels (c8–c10 folded wrongly in the whip snap) and
said: be selective, use as much information as possible, but "if we are not sure just dont
use it"; they can record more flights to replace dropped data. A wrong label misleads the
fit; a gap costs nothing.

**How to apply:** `aerowhip/takes/sequence_labels.py` (offline, ground truth) vs
`aerowhip/deployment/cable_filter.py` (causal, live; scored against the offline labels).
Keep the live filter's output as `cable_labels_filter.npz` for that comparison. See also
[[cable-markers-partial-frames]].
