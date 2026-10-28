---
name: cable-contact-take-is-failure
description: "A take where the cable touches the drone (rising above the clamp is fine) or the flight is interrupted is a failure, not used at all; implemented 2026-10-28 (aerowhip.takes.contact)"
metadata:
  node_type: memory
  type: feedback
  originSessionId: 3b3c5036-1c7a-4fef-9fc4-21587d9eae68
  modified: 2026-10-28T21:44:12.275Z
---

User rule (2026-10-28): if the cable touches the drone or the take gets interrupted, the whole take is a failure and is not used at all. Example: cf_6/mix_a/001 and /002 (cable flip).

**Why:** the user doesn't want contact/flip events in the data; such takes are outside the regime being modelled.

**How to apply:** when auditing, check that every consumer (offline fits, validation, envelope inputs aside, online estimator and running model, previews) drops these takes entirely, not just their bad windows. The envelope may still record them as failures (that's how levels get capped). Missing cable markers alone are not a failure ([[cable-markers-partial-frames]]).

Clarified the same day: contact with the drone is the failure, not the cable rising above the clamp.
User chose: contact = within 1 cm of the drone envelope, 3 frames in a row; land in flight within
5 cm of the drone; planner keeps every model's cable 5 cm from the drone. Implemented in both repos
(stacked-cylinder envelope from the CAD: battery 4 cm, arms 7.6 cm, guard 11.4 cm). The 3D-view
cable-tracking side (custom-real) does not use it yet.
