---
name: no-ros-queries-during-flight
description: Never run ros2 CLI commands (topic hz/echo/list) in the flight container while the user may be flying; it stalled the flight loop and caused an emergency stop
metadata:
  node_type: memory
  type: feedback
  originSessionId: ab84d596-c67b-413a-a8dc-e2a7a0e5d9ae
  modified: 2026-10-27T19:58:44.019Z
---

Never run `ros2 topic hz`, `ros2 topic echo`, `ros2 node list` or any other ROS CLI command inside the flight container (crazyswarm2_jazzy_whipping_dev, domain 42) unless the user has confirmed the drone is on the ground and no flight is starting.

**Why:** on 2026-10-27 a link check I ran (`ros2 topic hz` + `echo`) coincided with the user starting a flight; the new DDS participants stalled the flight script's 50 Hz command loop (>0.12 s) and it triggered an emergency motor stop right after takeoff.

**How to apply:** check flights from the files (stack log in AeroWhip_deploy/output/custom_real/<session>/, the flight folder's logs) — those are passive. For a live link check, ask the user first and run it only with the drone landed; tell them not to start a flight until it finishes.
