---
name: online-identification-rethink
description: "2026-10-28 evening: user paused calibration data collection; wants an online identification loop (fly with planning, learn drone+cable dynamics per flight, no full recollection after small drone changes, don't avoid motions for flip-over risk); proposal in internal/online_identification_proposal.md"
metadata:
  node_type: memory
  type: project
  originSessionId: d9ff1455-a877-4890-b4dc-ff0f79bcc835
  modified: 2026-10-28T18:19:36.315Z
---

On 2026-10-28 evening the user stopped the mixed calibration series after two mix_a flights flipped
the cable over the drone, and asked to rethink: "something online", fly with planning and learn the
dynamics as we go, use the current model as the baseline, don't stop testing a motion just because
the cable may swing up, and don't redo everything whenever the drone changes a little.

Key fact for the rethink: the simulator (cf_2's cable fit) predicts 33° of first-link swing where
the real cable went over the top (162–175°) within 1 s of a 6 m/s² sharp stop; 21° vs 59° at
3 m/s². The cable model is badly wrong for sharp strokes.

**Why:** fixed campaigns (15 takes, hours of refits) are slow, brittle to small hardware changes,
and found the unsafe level by breaking a drone (cf_2's motor 4 ESC).

**How to apply:** don't propose resuming the calibration series or refits as before. Discuss and
agree the loop in internal/online_identification_proposal.md first (one flight per round, fast
warm-started fits, planner creeping levels, measured flip envelope, survivable flips). Related:
[[tune-drone-before-new-data]], [[new-airframe-redo-fits]].
