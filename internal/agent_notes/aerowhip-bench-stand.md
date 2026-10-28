---
name: aerowhip-bench-stand
description: User has an RC Benchmark thrust stand; plans dedicated motor tests for stage-1 (per-motor map); will share their stand code from the lab
metadata:
  node_type: memory
  type: project
  originSessionId: ab84d596-c67b-413a-a8dc-e2a7a0e5d9ae
  modified: 2026-10-27T15:50:28.185Z
---

The user has an RC Benchmark thrust stand (tested it 2026-10-26; ramps in companion repo `calibration_data/bench/motor_ramp_2026-10-26`). On 2026-10-27 they said they will bring the stand test code from the lab so we can design dedicated tests for stage 1 of the simulation identification ([[aerowhip-reorganization]]).

**Why:** existing ramps only reach 1600 µs (0.40 N), sample at ~8 Hz (no motor time constant), and their PWM→µs link is an unverified assumption (bench exponent ~1.85 vs flight-fitted 2.2).

**How to apply:** don't change the motor map in configs from the old ramps alone. Proposed tests: firmware-driven motor PWM override on the stand (removes protocol ambiguity), full range to saturation with short holds, step responses at max sample rate, fixed voltages, torque/thrust ratio, all four motors. Ask for the firmware build config (ESC protocol, battery compensation) and the stand's max sample rate.
