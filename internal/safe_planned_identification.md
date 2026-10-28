# Safe, planned identification: what to measure, what to learn, how to collect data

Written 2026-10-28 (evening) at the user's request, after a literature pass. A design for
discussion; nothing here is implemented. It replaces the "campaign" view of
`archive/fitting_and_adaptation_design.md` (stages 0–6) with: bench measurements for what can be measured,
flights planned over a horizon for what must be learned, and safety that does not depend on the
model being right. Facts about the drone and the cable are in the implementation record.

## 1. What is measured, what is learned

The rule (as in the fitting design): a quantity is **measured** on the bench when a bench
measurement is cheaper or more accurate than a flight; it is **learned** from flight when it
exists only in flight (delays, aerodynamic interactions) or when flight data pins it better. The
split below is what the evidence so far supports.

### Measured without flying (fix them, with their measurement uncertainty)

| Quantity | Method | Cost | Status |
|---|---|---|---|
| Drone mass, cable mass, marker masses | scale | minutes | done |
| Motor positions, propeller plane, attachment point, marker plane | ruler | minutes | done |
| Marker intervals along the cable | labels of a still hang (as stage 0) | minutes | done |
| **Centre of mass** (3-D) | hang the drone from a thread at two different points inside the OptiTrack volume; the thread line in the marker frame each time; the CoM is where the lines cross (a plumb-line method; no flight, uses the markers) | 20 min | not done (nominal 8 mm below the props) |
| **Roll/pitch/yaw inertia** | bifilar pendulum, period read from the onboard gyro (Crazyflie perching paper, MicroCART notes: J = m g D² / (4 h ω²)) | 30–45 min | not done; the coupled fit is insensitive to it (±30 % → 0.6 % loss), so low priority |
| **Thrust map (PWM → thrust), thrust vs voltage** | the RC Benchmark stand with the drone's own ESC/motor, driven through `motorPowerSet` (so the map is in firmware units) | 1 h | ramps exist for cf_2's motor type; redo with the stand driving through the firmware |
| **Motor time constant** | step changes on the stand, thrust or RPM response | with the ramps | not done; fitted from flight so far (18–22 ms, ±14) |
| **IMU alignment to the marker frame** | static: the gyro against OptiTrack in slow hand rotations (done 2026-10-28 on cf_2: −1.5°, +0.6°, −2.3°) | 10 min | measured, not applied (no effect on tracking) |
| **Cable linear density, diameter** | scale, caliper | minutes | done |
| **Cable bending stiffness EI** | static: cantilever a short free length (10–20 cm) from the clamp, measure the tip droop with the markers; EI from the elastica (or the simple beam formula for small droop). Dynamic: pluck the short cantilever, first-mode frequency gives EI (cantilever free-vibration method) | 30 min | not done; fitted (1.4e-4, ×/÷1.65) |
| **Cable internal damping** | log decrement of the plucked short cantilever in air (air damping is small for a stiff short segment); the standard method separates internal and air damping | with the above | not done; Cb fitted (4.15e-5) |
| **Clamp angular stiffness / point-clamp assumption** | hang the cable, push the first marker sideways with a known small force (a weight over a pulley, or a second drone-free gauge), read the first-link angle with the markers | 30 min | not done; the first link's angle is where the sharp-stroke misfit sits |
| **Cable natural curvature (set)** | the hanging shape with no load, markers (done implicitly: 2–3° per joint at rest) | done | |
| **Cable drag coefficient Cd** | literature: a circular cylinder at Re 200–900 (3.5 mm at 1–4 m/s) has Cd ≈ 1.0–1.2; measurable only in a wind tunnel | none | fitted 0.75 (±0.2); keep fitted, prior 1.0–1.2 |
| Battery law (thrust per volt) | hover (done), or the stand | done | |

### Learned from flight (the fits), with the data that sees each

| Quantity | Why flight | Data that identifies it |
|---|---|---|
| Command delay (radio, host, firmware, OptiTrack) | exists only in the loop | sharp tilt steps (strokes) |
| Effective thrust in flight (downwash, ground, battery under load) | differs from the stand | hovers, climbs |
| Drone body drag | aerodynamic | fast lines, random |
| Attitude-estimate behaviour (wander, correction time) | onboard filter | attitude logs on any flight |
| Cable drag in motion (Cd correction with speed, the tangential part) | aerodynamic, depends on speed and shape | strokes, whips, fast lines |
| Rotor-wash force on the cable (the steady residual, body frame) | aerodynamic interaction | hovers and slow motion at several headings |
| Cable damping under large motion (saturation, the sharp-stroke regime) | the structure is unknown; bench damping is small-amplitude | sharp stops at increasing levels, whips |
| Clamp behaviour under dynamic load | may differ from static | sharp stops, first-link angle |
| The coupled response (validation, not fitting) | — | everything, held out |

What this changes against today: EI, internal damping, the clamp stiffness, CoM and (optionally)
inertia and the motor time constant become **priors from the bench** with known uncertainty, so the
flight fits have fewer free parameters and the sharp-stroke misfit can be attributed: if the bench
EI and damping are right and the model still gives 33° where the cable flipped, the structure
(clamp, damping law, rotor wash in strokes) is wrong, not the values.

## 2. Collecting data safely with planning over a horizon

### Objective

A flight should (a) be **informative** about the quantities to learn, (b) **cover** the state
space the task uses (drone speed and tilt, cable angle and tip speed, headings, heights), (c)
stay **safe** against crashes, with the cable over the drone the main hazard, and (d) push the
envelope outward deliberately, so limits are found without breaking the drone.

### Literature in one paragraph

Informative inputs for identification are chosen by maximising a scalar of the Fisher
information matrix (A-: trace of the inverse, D-: log-determinant, E-: smallest eigenvalue), or
by minimising the condition number of the regressor of a Fourier-parameterised excitation
trajectory (the robot-identification tradition). For multirotors, a sampling-based planner over
belief states with an EKF information measure cut parameter convergence time and uncertainty by
~4× in flight (Hausman et al., 2016). Recent sim-to-real work (ASID, SPI-Active, IDTD) maximises
Fisher information of the trajectory, IDTD per parameter through the Schur complement, which
handles coupled parameters. Safety during learning is handled by **chance-constrained** planning
over a horizon with the model's uncertainty propagated (polynomial chaos, GP confidence
intervals; Koller et al. 2018 with a terminal safe set so a safe action always remains), and for
cable-suspended loads by **swing-angle constraints inside an MPC**. The closest project to ours
(Learning to Throw, 2026) pairs an identified analytical quadrotor with a rope solver and trains in
that; it does not plan its data collection.

### The design

**Model ensemble.** K parameter sets (K ≈ 16–32) drawn from the fits' clustered intervals and the
bench uncertainties, plus **structural variants** where the structure is in doubt (two damping
laws, two clamp models). The ensemble must **bracket the measured behaviour**: on the flights that
flipped the cable, at least some members must put the cable over the drone, or the ensemble is
too narrow to be trusted for safety. (Today's single model gives 33° where reality gave 170°: it
cannot bracket, so until the cable model is fixed the ensemble is widened by hand on the cable's
energy, e.g. damping ÷5 and EI ×3 as members.) The ensemble is the simulator's batch runtime.

**Plan over a horizon.** The command is the 50 Hz acceleration the policy will also use. A flight
plan is a sequence of acceleration segments over a horizon H (a whole 90 s flight planned offline,
or a receding 3–5 s horizon re-planned during the flight from the measured state). Parameterise
each segment as a motion primitive with free parameters (direction, peak acceleration, duration,
height change), or as splines/Fourier series for continuous motion. Evaluate a candidate plan by
simulating it with the whole ensemble:

- **Information**: the Fisher information of the parameters to learn, from finite-difference
  sensitivities of the predicted measurements (markers, pose) to each parameter (what the fits
  already compute as Jacobians); score by the A-criterion, or IDTD's per-parameter Schur score so
  the planner targets the weakest parameter.
- **Coverage**: a grid over (drone speed, tilt, cable first-link angle, tip speed, heading);
  reward cells not yet visited by the recorded flights.
- **Chance constraints**, every member of the ensemble, every step of the horizon: inside the flight
  area and height band, tilt < 50°, speed < limit, motor commands between idle and 90 %, **cable
  first-link angle < 110°** and **no cable node above the attachment**, distance of any cable node
  to the propeller disc > 10 cm. A plan is feasible only if a fraction ≥ 1−δ of members satisfy
  all constraints (δ ≈ 5 %); with structural doubt, require all members.
- **Envelope expansion**: the admissible region is the measured envelope (below) plus one step; a
  plan may exceed the highest level flown cleanly by one step per kind and heading.

Optimise with a sampling planner (random primitive sequences, keep the best, refine the best
few by coordinate search on their parameters); the batch runtime makes a candidate flight of 90 s
with K = 16 members cost ~15 s on 16 cores, so a few hundred candidates per plan are affordable
offline; the receding variant restricts itself to the next segment.

**Measured envelope.** After each flight, from the recorded cloud through the live filter: the
cable's maximum first-link angle and maximum rise per (kind, level, heading), and the drone's
maximum tilt. The envelope is the set of (kind, level, heading) flown with angle < 110°. The
planner may step one level beyond it; the ensemble decides whether that step is predicted safe.
Where the ensemble and the envelope disagree (the ensemble predicts safe but the neighbour flew
at 150°), the envelope wins and the ensemble is widened.

**Run-time safety, independent of the model.** The flight script keeps its aborts (area, height,
speed, 60° tilt, 0.8 m tracking) and the link-loss landing. New: a **cable-over-drone response**:
the live cable filter's first-link angle above 120° for 0.2 s, or any node above the attachment,
triggers a descent with the motors running (the link-landing profile) so the cable falls clear,
instead of waiting for the 60° tilt cut that drops the drone. Survivable flips are the condition
for testing near the limit at all.

**The loop.** Bench measurements once → ensemble → plan a flight → fly it → label → fit (offline,
warm start, minutes) → update the ensemble and the envelope → plan the next. The fit is per flight
as in the earlier proposal; the planner and the safety layer are what is new. Online fitting
during the flight is not needed for this; it can be added later for the receding-horizon variant.

### What the first rounds would look like

1. Bench: EI, damping and clamp stiffness of the cable (1 h), CoM by plumb lines (20 min); optional
   inertia and motor time constant.
2. Ensemble widened to bracket the flips (checked on `cf_6/mix_a/001`, `/002`).
3. Round 1 plan: sharp stops at 2, 3, 4 m/s² on x with 8 s holds plus moderate strokes at 45°,
   chosen by the planner inside the envelope (3 m/s² flew at 59°). Fit the cable with the stroke
   windows weighed; check whether the bench EI/damping with the clamp variants reproduce the
   first-link angle. This is the structural question answered with data that cannot flip the cable.
4. Rounds 2…: the planner steps outward (4.5, 5, …) only where the updated ensemble, now checked
   against rounds 1–k, predicts the cable below 110° for every member. The flip limit is approached
   from below with the model's prediction verified at each step; "test the limit" becomes
   measuring the angle at each level rather than discovering the flip.

### What to build, in order

1. Flight script: fly a planned acceleration trajectory from a file (a sequence of primitive
   specs; `--mix` already takes a list; add free durations/heights), and the cable-over-drone
   descent. 1 day.
2. Envelope bookkeeping from recorded clouds (per kind/level/heading: max angle, rise). Half a day.
3. Ensemble construction from the fits' intervals and bench priors; the bracket check on the
   flipped flights. 1–2 days.
4. Planner: candidate generation, batch evaluation with constraints and the information/coverage
   objective, offline for a whole flight. 3–4 days.
5. Receding-horizon variant (re-plan during the flight from the measured state) later, once the
   offline planner's plans fly as predicted.

## Sources

- Hausman et al., Sampling-based motion planning for active multirotor system identification
  (arXiv 1612.05143).
- IDTD: Informationally decoupled trajectory design for sim-to-real system identification
  (arXiv 2610.10905); ASID (arXiv 2404.12308); SPI-Active (arXiv 2505.14266).
- Koller, Berkenkamp, Turchetta, Krause, Learning-based MPC for safe exploration (CDC 2018,
  arXiv 1906.12189); chance-constrained trajectory optimisation for safe exploration (arXiv
  2005.04374); Pfrommer et al., safe RL with chance-constrained MPC (PMLR v168).
- Constrained MPC with swing-angle limits for a quadrotor with a suspended load (Mathematics 2021,
  9(15):1822); HPA-MPC (arXiv 2411.11982).
- Learning to Throw: cable-suspended payload delivery with a quadrotor (arXiv 2606.27603).
- Excitation-trajectory design by condition number / Fourier parameterisation (arXiv 2401.16566,
  2603.12099, 2609.16786); AIAA flight-vehicle identification review (10.2514/1.C037269).
- Bifilar pendulum inertia with the onboard gyro (arXiv 2403.00128; MicroCART notes); cantilever
  free-vibration EI and log-decrement damping (BioResources 8(1) 2013; Muroran IT on separating
  internal and air damping).
