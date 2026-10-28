# Fitting and adaptation design

Written 2026-10-28 from the diagnostics of that day. A design, agreed in discussion, not a record of
working code: what exists is marked; the rest is to be built. The implementation record holds what
is implemented.

## Purpose

The policy flies the coupled drone–cable system through one interface: acceleration commands in at
50 Hz, OptiTrack and the cable filter out. The simulation is therefore judged on one thing: how the
coupled system responds to commands during strokes, on flights the fits never saw. Everything below
serves that.

## What the diagnostics established (2026-10-28, existing data)

Evidence: `runs/diagnostics/strokes_*.json` (`scripts/stroke_diagnostics.py`), the cable-alone and
hanging-shape checks run in the session (numbers below), `runs/fits/coupled_sensitivity/`.

1. **The coupling is right.** On 0.6 s stroke windows replayed open loop from the measured state, the
   drone's position error is smaller with the cable than without it (debiased 41 vs 53 mm at 0.6 s;
   held out 46 vs 51 mm), and the tilt error is the same (0.6–1.6° with per-flight offsets removed).
   The coupled stroke loss moves under 0.6 % for a 30 % inertia change, a 10 % arm change or a 5 mm
   centre-of-mass change; the joint intervals a stroke fit would give are ±2.4 (inertia scale), ±1.1
   (arm scale), ±36 ms (motor response), ±11 mm (CoM). These are not identifiable from flight and the
   coupled behaviour is insensitive to them at this accuracy.
2. **The drone is a good plant plus a slow bias.** Raw position error 150–230 mm per second,
   11–37 mm once one constant acceleration per flight is removed. The bias (~0.15 m/s²) is the
   onboard attitude estimate's error (0.85° RMS) and trims; the IMU misalignment (roll −1.5°, pitch
   +0.6°, yaw −2.3°, measured from the gyro) is real but correcting it onboard changed nothing
   measurable (random_high/009 vs /010).
3. **The cable's error is a steady lean, not a speed effect.** Replayed from the measured clamp, the
   marker error at 0.7 s is 34 mm with the cable still, 35 mm swinging gently, 31 mm at 1–3 m/s and
   40 mm above 3 m/s. The model's hanging equilibrium differs from the measured hanging shape along a
   straight line: 6 mm at the first marker to 40 mm at the tip, about 2.3° off the model's vertical.
   The simulated cable spends its first second swinging into its own equilibrium; that swing is most
   of the error in every speed bin. Candidate causes, not yet separated: rotor downwash on the cable,
   OptiTrack's vertical against gravity, the clamp off-centre.
4. **The measured cable is never straight.** Every marker joint shows a 2–3° bend at rest, the first
   3–8°, the last 6–14°, on all 21 flights; edge lengths 83.2, 100.1, 99.3, 101.7, 100.3, 98.9, 101.2,
   98.9, 100.1, 96.9 mm (std ~1 mm, i.e. label noise of ~1° per joint) against the configured
   85 and 100 mm.
5. **In strokes the simulated tip is 5–8 % too slow** (peak-speed ratio 0.947 fit, 0.936 held out),
   one frame late.
6. **Per-frame acceleration targets are unusable** (`scripts/residual_targets.py`): evaluated at a
   measured shape, the model's bending stiffness reacts to the label-noise kinks with 6–20 m/s² of
   node acceleration that the real cable never feels (its measured accelerations are 2–7 m/s²). A
   residual regressed on such targets would learn to cancel forces the simulator never produces.

## The model and where each part comes from

| Part | Source | Decision |
|---|---|---|
| Masses, attachment height, diameter, air density | measured | fixed |
| Marker intervals | **labels** (mean edge lengths above), not the ruler | change (stage 0) |
| Firmware controller, mixer, massThrust | known code, ported and tested | fixed |
| Thrust map κ, p, β | stage 1 (motor logs) | fitted, done |
| Command delay, drone body drag | stage 2 (drone-only flights) | fitted, done |
| Roll/pitch inertia, arm, motor response, CoM | not identifiable | nominal, randomised over the flat range |
| Onboard estimate error | attitude logs | randomised slow disturbance; not modelled |
| Cable EI, Cb, Cd, linear drag | stage 3 (measured clamp) | fitted jointly with the residual, then frozen |
| Cable residual (below) | stage 3, same rollouts | fitted; the part that adapts |
| Coupling geometry | measured | fixed |

Principle: each parameter is identified from the data that can see it and only there. Nothing is
fitted on the coupled flights, where the drone's and the cable's errors trade against each other;
the coupled flights validate.

## The cable residual

Why: the three cable findings (lean at rest, tip-speed deficit, tip curl) are forces the model does
not have, not values of its parameters; and the current parameter values are already compensating
for them (lower Cb helped the whip and hurt the slow flights). So the residual is fitted together
with EI, Cb, Cd and the linear drag, or the physics stays biased.

Form: compact, linear in its coefficients, bounded, added as a per-node force in the native step so
training sees it. Terms, in order of confidence, each kept only if the held-out score improves and
its clustered interval excludes zero:

1. **Steady force per unit length in the drone's body frame, smooth along the cable** (a few
   coefficients per axis). Represents the rotor wake, any calibration lean and the cable's own set.
   Identified by the still and slow windows.
2. **Drag correction with speed**, along and across the local tangent, varying along the cable.
   Represents what the quadratic-cylinder drag misses in strokes. Identified by the fast windows.
3. **Natural curvature at the free end** (one or two parameters), for the persistent tip curl.
   Optional.

Not a network: per-frame targets are unusable (finding 6) and a network fitted through rollouts
would need the differentiable engine that was removed. If the compact residual leaves a clear
structured error on policy flights, that is the case for a learned term, on those flights.

## Stages

0. **Measurement update.** Marker intervals from the labels. Moves the first and last node 2–3 mm.
1. **Thrust map.** Done; unchanged (`scripts/fit_thrust_map.py`).
2. **Drone.** Done; unchanged (`scripts/fit_drone_segments.py`). Its per-flight biases are an input
   to stage 4.
3. **Cable + residual** on 1 s segments of every cable fit flight, all phases (still windows
   constrain term 1, strokes term 2), from the measured clamp. Three passes: physics + term 1,
   + term 2, + term 3. Clustered confidence intervals as now (`aerowhip.fitting.least_squares`).
4. **Coupled validation, bias-corrected, nothing fitted.** The coupled replay with each flight's
   stage-2 bias applied through the controller's acceleration-bias input (the bias alone adds ~10 mm
   of apparent cable error by dragging the clamp). Scored by the **stroke test**: on held-out whip,
   strokes and decay flights, windows from each stroke's onset (`scripts/stroke_diagnostics.py`),
   tip peak-speed ratio, peak timing, tip path error, drone tilt over the stroke; plus per-segment
   position, tilt, shape and tip RMS as diagnostics. This is the yardstick for every decision.
5. **Randomisation** from what the fits leave: battery, delay, motor response, inertia and CoM over
   their flat ranges; the drone bias as a slowly varying disturbance of the measured size; the
   residual coefficients over their fit-vs-held-out spread.
6. **Freeze, train, fly.**

Fits run through the Model Fit window.

## Adaptation: two loops

Every policy flight is recorded as a calibration flight (same logs, same loader, manifest role
`policy`) plus the observation vector the policy saw and the action it produced at each decision.
Then every tool above runs on policy flights unchanged.

### Big loop: make the simulator right

Physics frozen after stage 3. Per round: the residual is refitted on the new flights (warm start,
last flight of each kind held out), randomisation widens only along measured axes, the policy is
fine-tuned from its last checkpoint, evaluated in simulation across the randomisation set, flown.
Target cost: minutes for the fit (batched native replay of the new flights' windows; one 1 s cable
window costs 104 ms, nearly all in C++), under an hour for the fine-tune.

Its product is a better simulator for the *next* policy. It cannot help the policy flying now.

### Small loop: correct the action

At the policy's output, at 50 Hz, without changing the simulator or the policy. For every decision of
a recorded flight (s_t, a_t, measured s_{t+1}) the simulator run from s_t gives what a_t should have
produced, and the inverse, the action a′_t for which the simulated response equals the measured
s_{t+1}, is a small least squares per decision. δ = a′ − a is the correction that was needed; a small
regressor δ(s, a) fitted to these pairs (seconds) deploys as a + δ(s, a) at the next flight, with a
cap on |δ| and a bound on its change between flights. It is learned from the measured response only,
never from the reward.

It captures what is systematic in the drone's immediate response and not in the simulator: the
per-flight thrust bias, battery and motor drift, the estimate's tilt error, host timing. It cannot
fix the cable's slower dynamics; that is the big loop's.

### Diagnosing a difference, in order

1. **Deployment**: the native observer and filter run offline on the recorded cloud, the policy on
   the result, must reproduce the logged actions bit for bit.
2. **Sensing**: the flight's sensing statistics against the sensing model's ranges
   (`identify_sensing`, `validate_sensing`).
3. **Plant**: open-loop replay of the policy flight (segments, stroke test) against the calibration
   flights' numbers. Larger errors: distribution shift, refit the residual on these flights (big
   loop). Drone off: stage 2 on them.
4. **Brittleness**: the policy simulated from the flight's measured initial state across the
   randomisation set; the real trajectory inside the spread means robustness (retrain or widen),
   outside means a missing axis (the replay of step 3 says which).

The discriminator between "fix the model" and "fix the policy" is open-loop replay error against
closed-loop divergence.

### Alternation and switching rules

Big loop (fit, train, fly) → three flights of the same policy, same conditions; the between-flight
spread is the floor below which no change is meaningful → small loop passes while all hold:

- the stroke metrics improved by more than the between-flight spread;
- |δ| stays below its cap and its flight-to-flight change shrinks;
- the part of the response δ cannot explain is not growing.

Back to the big loop when any holds:

- the metrics stop improving or δ saturates: the mismatch is bigger than a nudge;
- δ has structure (depends on speed, tension, battery, position): a missing term, cheaper in the
  simulator than corrected every flight;
- the mismatch is in the cable (tip speed, shape, timing) while the drone's response is corrected.

The big loop then takes the new flights for the residual and δ's structure as the hint for the
missing term or randomisation axis. δ resets to zero with each new policy; the sequence of δ
magnitudes across rounds is the convergence curve. Stop when the real stroke metrics lie within the
simulated spread, δ is small and unstructured, and the success rate holds across flights.

Safety: first rounds at a reduced acceleration limit under the flight script's abort bounds, after
the policy passes the simulated evaluation; the stroke test gates raising the limit.

## To build, in order

1. Stage 0 (config), the residual terms in the engine and `fit_cable_segments.py`, the bias-corrected
   coupled validation with the stroke test. Then the first stage-3 fit.
2. The batched native replay call and the warm-started linear-residual fit (minutes per round).
3. The deployment node: observer + filter + policy at 50 Hz through the acceleration interface and
   the calibration script's abort bounds, logging in the calibration format plus observations and
   actions; the `policy` manifest role.
4. The closed-loop comparison tool (policy from a flight's measured state, with and without
   randomisation, against the recorded trajectory).
5. The small-loop inverse solve and δ regressor.

Open decisions: the real-world success definition for the strike (OptiTrack-measured), the first
policy's envelope, the brittleness threshold (set from the first round's between-flight spread).
