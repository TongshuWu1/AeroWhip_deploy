# AeroWhip simulation package — design

Status: proposal for approval, 2026-10-28. Nothing in this file is implemented unless the
implementation record says so. Written after the engine-v2 work and the open-loop replay study of the
2026-10-27 calibration flights (drone-only: no bias; with cable: the simulated drone leaned away from
the cable because the attitude loop was modelled too soft; a residual −0.10 m/s² at hover remains).

## 1. What the package must do

One simulator, one configuration, one identification pipeline, used unchanged for four jobs:

1. **Replay**: run a real flight's sent commands open loop and report the error against OptiTrack,
   per flight and held out. This is the model's only evidence.
2. **Train**: step thousands of independent lanes per second on the CPU (`BatchRuntime`) for the
   reinforcement-learning environment, with domain randomization drawn from identified spreads.
3. **Fly**: run interactively with a gamepad or a scripted reference through the same code path.
4. **Publish**: be the model of the paper, with every parameter traceable to a measurement, a fit
   with held-out flights, or a firmware constant.

Non-goals: ground contact, rotor aerodynamics beyond an induced-velocity field, wind, vehicle
flexibility, visual sensing.

## 2. Layers

Everything below the policy is native C++ (`native/`), one process-wide model, stepped at the
physics step (1/600 s) with the firmware and host loops on their own clocks.

```
policy (50 Hz)  ->  host (100 Hz, +delay)  ->  firmware (500 Hz)  ->  motors  ->  plant  ->  sensing
   acceleration       acceleration + trims     Mellinger port       PWM->thrust   drone+cable  OptiTrack model
```

### 2.1 Plant (`engine.cpp`)

- Drone: rigid body, mass, inertia about the centre of mass, tracking origin (top plate), centre of
  mass and cable attachment as body offsets. Thrust along body z, torques from the mixer.
- Cable: twist-free discrete elastic rod, 10 links at the marker spacing, point clamp at the attachment.
  Parameters EI, Kelvin–Voigt Cb, mass per link with lumped marker masses, diameter.
- Air on the cable: linear drag rate, quadratic normal drag (Cd, diameter) on the **relative** air
  velocity, where air velocity = induced-velocity field of the rotors (2.4) + 0 (still air).
- Air on the drone: identified drag rates (currently in the residual drone branch; keep as a
  physical linear/quadratic term with fitted coefficients).
- Integrator: projected Lie-BDF2 with Picard iterations, unchanged.

### 2.2 Motors and mixer (new: `firmware.hpp`)

Port of the firmware's power distribution, not a fitted abstraction:

- Mixer: `m1..m4 = thrust ∓ roll ∓ pitch ± yaw` in firmware units (int16 torque outputs, uint16
  thrust), clipped to [idle, 65535]. Saturation is where whip strokes live; it must be exact.
- PWM to thrust per motor: `T_i = f(pwm_i, V)`. Identified from the bench thrust stand (available)
  and the hover-versus-voltage data; replaces the current `thrust_scale/exponent/voltage_slope`
  collective map, which stays as the fallback until the per-motor map is validated.
- Motor lag: first order, 15 ms (keep; re-identify from the bench ramps if available).
- Battery: voltage is a state per lane: initial value sampled from the recorded range, slow sag
  with delivered power; the firmware does not see it (no compensation in stock Mellinger).

### 2.3 Firmware controller (new: `firmware.hpp`, exact Mellinger)

A line-by-line port of `controller_mellinger.c` (Bitcraze master, verified 2026-10-28) with the
parameters read from the flight's `crazyflies.yaml` (`ctrlMel.*`, `massThrust`), never retyped:

- Position branch with the gains the host script zeroes (kp/kd/ki_xy, kp/kd/ki_z): kept in the
  port and set to zero from the same YAML the script edits, so the simulator follows the script.
- `target_thrust = mass*(a_sp + g) + ...`, `thrust = massThrust * (target_thrust · z_body)`.
- Desired axes from `z_axis_desired` and the commanded yaw; `e_R` with the firmware's expression
  (no 1/2), `e_R.y` sign flip, `ew`, `err_d_roll/pitch` finite difference, yaw integral
  (`ki_m_z`, `i_range_m_z`), roll/pitch integral present but zero by default, output clamps ±32000.
- Runs at `ATTITUDE_RATE` (500 Hz) on the firmware's state estimate (2.5), with the measured
  command delay in front of it (30 ms host+radio, keep as the identified value).
- Test: the vendored C file compiled into a test harness and driven with the same random inputs as
  the port; outputs must agree to float precision. This is the "same firmware as the real drone"
  guarantee.

What is **not** ported from the firmware: the Kalman estimator (2.5 models its error instead) and
the sensor drivers.

### 2.4 Rotor induced-velocity field (new, physical, three parameters)

Air velocity under the drone in the body frame
`v_air = −k √(T/2ρA) · f(r/R_jet) · exp(−z/z_decay) · z_body`, applied to cable nodes through the
existing drag law. Status: hypothesis for the −0.10 m/s² hover residual; the footprint test on the
existing flights did not confirm it (R² 0.1, wrong sign for the swing-dependent part, which the
attitude model explains). It enters the code behind a switch and is fitted only after the lab tests
in §5 separate it from mass bookkeeping. If those tests show mass, the term stays off.

### 2.5 State estimate and sensing

- Firmware state estimate = true state + per-flight attitude bias (constant, sampled 0–5°; the real
  drone hovered 4.8° tilted while believing it was level) + Kalman lag (identify from the logged
  `stateEstimate` against OptiTrack; else 0).
- Policy observations = OptiTrack model: 100 Hz, latency (identified clock offset), position noise
  (measured from static markers), cable marker dropout with the same statistics as the recordings
  (per-marker visibility 94–100 %, bursts during fast motion), then the **same `CableFilter`** the live
  node runs. Training on filtered, partly missing cable observations is what makes the policy
  deployable; the simulator must therefore produce the raw marker cloud, not the true node positions.

### 2.6 Residual (last, bounded)

Structured residual network on top of the physical terms, fitted on the force the model fails to
produce (open-loop drone and cable errors), with body-frame inputs (node position relative to the
rotors, thrust, relative air speed) so it can represent what the physical terms cannot, a magnitude
penalty, and promotion only when the held-out flights improve. Off in every result until promoted.

## 3. Identification pipeline (`aerowhip.fitting`, one command per stage)

Each stage has its own data, its own held-out flights, writes `runs/calibration_<session>/<stage>.json`
with every evaluation, and recomputes its reported numbers twice. A stage never refits what an
earlier stage fixed.

| stage | fixes | data | loss |
|---|---|---|---|
| 0 geometry & mass | masses (drone, cable, hardware, markers), attachment offset, link lengths, inertia | scale, hanging-cable flights, CAD/bifilar pendulum | measured |
| 1 motors | PWM→thrust map per motor, voltage dependence, lag | bench stand, hover at 4 voltages | thrust-stand fit |
| 2 attitude | apparent roll/pitch inertia scale, command delay, estimate bias model | drone-only flights (steps 2–10 m/s², random) | debiased 2 s open-loop drone error |
| 3 cable | EI (static droop test), Cb, Cd, linear drag | cable flights, prescribed clamp | 2 s cable marker error |
| 4 coupled | induced field (if any), mass correction | cable flights open loop, +20 g and dead-weight hovers | raw 2 s drone error (bias included) |
| 5 residual | network weights | all flights | force residual, held out |

Reporting: one `scripts/validate_model.py` producing the standard table (per flight: raw and debiased
drone error, cable shape, tip; held out marked) and the real-versus-simulated 3D page. The paper's
model table is generated from these files, not typed.

## 4. Training environment and randomization

`BatchEnvironment` stays as built (same interface to PPO and the tools). Per-lane randomization,
every item tied to an identified spread, listed in the run snapshot:

battery voltage (recorded range), drone mass (±weighing error + payload), attitude estimate bias
(0–5°), command delay (20–40 ms), cable parameters within their fit confidence, residual on/off,
induced field on/off (if adopted), OptiTrack latency and dropout statistics.

Zero action hovers (feed-forward from the thrust map at the lane's voltage) — this stays, because the
host supplies that trim in flight.

## 5. Data to record (lab), in order

1. Weigh the complete flying assembly with cable and hook attached.
2. Hover + slow motions with +20 g on the body; hover + motions with a 17 g dead weight at the hook
   and no cable (separates induced field from mass).
3. Drone-only: acceleration steps 2–10 m/s² in x, y, z; two repeats (attitude stage).
4. Cable static droop (clamped horizontal, photo/mocap of the shape) for EI.
5. Each existing motion twice more (held-out sets per motion); circles at 3–5 speeds; five whip strokes.
6. Log `stateEstimate.*`, `pm.vbat`, `motor.m1..4` at ≥100 Hz in every flight (the YAML already lists
   the motors and vbat); add `ctrlMel.cmd_thrust`/`cmd_roll/pitch/yaw` if the radio allows.

## 6. Package layout and cleanup

- `native/engine.*` (plant), `native/firmware.hpp` (Mellinger port, mixer, motor map, battery),
  `native/runtime.hpp` (clocks, delay, events), `native/batch.hpp` (lanes). Python owns configs,
  fitting, environments, UI.
- One physics JSON (`configs/physics/drone_cable.json`), one firmware TOML generated from or read
  with the flight `crazyflies.yaml`, one actuator JSON. Every value carries its source stage.
- Retire: the tensor engine and its fits, `equivalence_check.py` (replaced by the exact physics
  tests, the firmware harness test and the replay tables). Merge `engine-v2` once the firmware port
  passes its harness test and the replay numbers are reproduced.
- Docs: `docs/simulation_model.md` (equations and parameters with their stage), `docs/data_and_fitting.md`
  (pipeline and tables), `internal/implementation_record.md` (what is implemented, dated).

## 7. Order of work and checks

1. Firmware port + harness test against the vendored C file (no data needed).
2. Mixer + per-motor map from the bench data; apparent inertia fit on the existing cable flights;
   re-run the replay table (expect held-out raw error ≤ 154 mm, lean ≤ 2°; drone-only unchanged).
3. Sensing model + raw marker cloud output + `CableFilter` in the environment; scored against the
   recordings' dropout statistics.
4. After the lab session: stages 0, 1, 2, 4 refit; induced field decided; residual last.
5. Merge, retire the tensor engine, regenerate docs and tables.

Each step ends with the replay table and the tests; no step changes a reward or task definition.
