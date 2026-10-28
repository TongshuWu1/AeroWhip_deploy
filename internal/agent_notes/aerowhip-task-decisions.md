---
name: aerowhip-task-decisions
description: "User-approved training task decisions (2026-10-28): target strike objective, whip event defined by outcome, fold as diagnostic only, curriculum with reference-state init, one strike + recovery per episode"
metadata:
  node_type: memory
  type: project
  originSessionId: ab84d596-c67b-413a-a8dc-e2a7a0e5d9ae
  modified: 2026-10-27T23:31:33.016Z
---

Approved by the user on 2026-10-28 ("nothing, I like all you have"), replacing the fold-transport
reward as the training objective:

1. **Objective: target strike.** Sparse hit reward plus dense outcome shaping (tip speed outward,
   extension, tip-to-target distance). Drone costs: workspace box, hard termination with cost on
   tilt > ~60° or cable near the floor.
2. **Whip event (evaluation definition), by outcome:** tip speed ≥ threshold (~4 m/s; scripted whips
   give 4–5 m/s), amplification ≥ 2–3× the drone's peak speed in the preceding 0.5 s, extension
   ≥ 90 % of the cable length at the peak; the drone back in a steady hover within ~1 s. A strike
   is a whip event with the tip within a radius of the target. Report both.
3. **Fold is a diagnostic, not reward.** A faded fold-shaping hint is a fallback only if the outcome
   reward alone fails to find a whip.
4. **Curriculum, simple:** dense shaping from the start; stages on target distance / speed threshold /
   workspace size; reference-state initialisation from the recorded scripted whips.
5. **Episode:** hover → one strike → 1 s recovery → end. Repeated whipping is a later task.
   Domain randomisation from the fitted intervals from the start.

**Why:** the fold is how a whip happens, the fast far tip is what a whip is; the user's own scripted
whips never cross the old 90° fold threshold, and a strike gives a yes/no success countable in real
flights (sim-to-real-to-sim). See [[aerowhip-reorganization]] for the publication plan.

**How to apply:** exact numbers (speed threshold, amplification, hit radius, target placement,
workspace) need the user's explicit approval before coding; reward/success definitions are research
decisions (CLAUDE.md).
