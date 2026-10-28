# Online identification: a proposal for discussion

**Superseded in part (2026-10-28 evening) by `safe_planned_identification.md`:** the user asked
for a bench-vs-flight split and for planning the commands over a horizon with safety constraints
rather than online fitting; the per-flight loop and the during-flight analysis below remain as
background.

Written 2026-10-28 evening, after the user paused data collection. A proposal, not a design that
is agreed or implemented. Facts it rests on are in the implementation record's notes of that date.

## What the day showed

1. **A drone change costs the whole pipeline.** Replacing cf_2 by cf_6 (same build, 11 g heavier,
   markers 5 mm higher, one motor swapped) invalidated every drone-dependent stage: tuning, thrust
   map, drone fit, cable residual, coupled validation, the reset bank. The planned recovery was
   ~12 min of flying per setup, ~40 min of labelling, hours of refits. Small changes (a new
   battery, a prop, a different cable attachment) will keep happening.
2. **Fixed calibration series are slow and brittle.** 15 single-kind takes (then 6 mixed) are
   flown before any model is updated; a bad level (6 m/s² sharp stop) is discovered by breaking the
   drone, and the fix is to lower a number in a file and fly again.
3. **The simulator is wrong where the task lives.** The configured cable (cf_2's fit) predicts
   33° of first-link swing for a sharp stop that put the real cable over the drone (162–175°)
   within one second; 21° against 59° even at 3 m/s². The coupled stroke test already showed the
   tip 10 % too slow. The sharp-stroke regime is exactly the whip, and the model has barely seen it:
   the 1 s fitting windows are dominated by slow motion, and the sharp parts are a few seconds per
   flight.
4. **The tuning loop worked.** One flight, one question, a report a minute after landing, a
   decision, the next flight. That cadence (flight → fast analysis → decision) is what data
   collection should have too.

## What the user asked for

Fly, with some planning, and learn the dynamics of drone and cable from the flights as they happen;
don't avoid a motion only because the cable may swing up; don't redo everything when the drone
changes a little; use the current model as the baseline.

## Proposal: an identification loop instead of a calibration campaign

Each round is one flight of ~2 min and a few minutes of computation, repeated until the model stops
improving on held-out data. The drone is never flown "for data" without the model being updated
before the next flight.

### Round

1. **Plan** the next flight from the current model and its uncertainty (below): a sequence of
   motion primitives (the mixed-flight parts: sharp stop, strokes, whip, line, circle, figure-8,
   vertical, random) with levels and headings chosen where the model is least sure or most wrong,
   inside a **measured** safety envelope (not a simulated one, see 3).
2. **Fly** it (the existing flight script; `--mix` already takes a part list).
3. **Label** the cable right after landing (the sequence labeller, 1.5–4 min per flight today; to
   be made faster: label only the recorded window, reuse the live filter's labels as the start).
4. **Update** the model in minutes, not hours (below), warm-started from the previous round.
5. **Score** on a rolling held-out set (the previous round's flight, or a fixed validation flight
   flown once) and on the flight just flown: per part, drone position/tilt error, cable shape and
   tip error, stroke peak-speed ratio. Print it like the tuning report.
6. Decide: continue with the planner's next flight, or stop when the held-out score has not
   improved for two rounds.

### What gets updated, and how fast

| Part of the model | Changes when | Fit from | Cost |
|---|---|---|---|
| Hover trims, battery law | battery, mass, markers | the flight's hovers (as today) | seconds |
| Thrust scale, delay, drone drag, motor response | drone hardware | the flight's drone motion, 1 s windows, ProcessRunner | minutes (stage 2 on one flight: ~2–3 min; the warm start removes the grid search) |
| Cable EI, Cb, Cd, linear drag | the cable | the flight's cable windows from the measured clamp | minutes with a warm start |
| Cable residual (steady force, drag correction) | drone (rotor wash), attachment | the same windows | with the cable fit |
| Roll/pitch inertia, arm, CoM | not identifiable (shown on cf_2) | nominal, randomised | none |

The expensive part today is not the fit but the campaign: 15 flights before the first fit, the
full grid searches, and reproducing everything twice at 1e-7 tolerance. Per-round fits warm-start
from the last round, fit on one flight plus the last few, and are reproduced twice only at
adoption time.

### The planner

The model keeps an ensemble of parameter sets (the fit's clustered confidence intervals give them
for free; later, a sampled posterior). For each candidate primitive (kind, level, heading), the
ensemble is simulated and the spread of the predicted cable tip path and drone tilt is the
**uncertainty**; the measured error on the parts flown so far gives the **bias**. The next flight
takes the primitives with the largest uncertainty + bias, subject to:

- **Level creep, not jumps.** A kind is flown at most one step above the highest level already
  flown cleanly for that kind and heading (steps of 1 m/s²), so the envelope is explored from the
  inside. The 3 → 6 m/s² jump of mix_a is what broke the drone.
- **The measured envelope.** From the flown flights, the cable's maximum angle per (kind, level)
  as the raw cloud shows it (59° at 3 m/s², over the top at 6, for sharp stops on cf_6). A level
  whose measured neighbours exceed a bound (say 120°) is flown only when the user says so; the
  user's instruction is that flip-over is not a reason to stop testing, so this is a flag with a
  prompt, not a refusal.
- Battery: at most ~2 min per flight (the battery compensation keeps the thrust right; the flight
  time stays bounded by the abort limits).

Until the ensemble exists, the planner can be a fixed sweep that creeps levels (1 → 2 → 3 ...) and
rotates headings, which is already most of the benefit.

### Making the cable model right for sharp strokes

The flip finding says the cable's sharp-stroke response is the first thing the loop must fix.
Candidates, each a hypothesis to test on the sharp-stop windows, not a decision:

- the fitting windows: weigh stroke windows (0.6 s from stroke onset) together with the 1 s
  segments, as `fit_cable_segments.py --windows strokes` already allows (the previews moved EI/Cb
  towards a stiffer, less damped cable);
- the damping law: Cb saturating at high curvature rate (`saturation` term, fitted at its bound on
  slow data, undecided);
- the clamp: the attachment's bending moment and the point-clamp assumption, which dominate the
  first link's angle;
- the rotor wash on the first links during the stroke (the steady residual is body-fixed; a
  stroke tilts the drone 30°+ and the wash sweeps the cable).

A sharp-stop flight at 2, 3, 4 m/s² on one heading, labelled, fitted with each candidate, scored on
the tip path and the first-link angle, is one round of the loop.

### Surviving flip-overs

The user's instruction is to keep testing motions that may flip the cable. Then the flip must be
survivable: when the cable comes over the drone, the right action is a fast descent with the motors
running (the cable falls clear) rather than a motor cut (the drone falls 1.3 m: that killed cf_2's
motor). Today a flip triggers the 60° tilt cut. A **cable-over-drone detector** (the live cable
filter already tracks the cable; a first-link angle above ~110° for 0.2 s) that starts the link-loss
landing before the tilt bound is reached is a small change to the flight script. Hardware help (prop
guards, a cable stand-off ring) is the user's call.

### What stays from today

The recording format and manifest, the flight script and its safety behaviour, the tuning tools,
the fitting code and its scoring, the simulator. What changes is the schedule: no campaign, one
flight per round, the fits warm-started and per flight, the planner choosing the next flight, and
the stroke regime weighed properly.

## Questions for the user

1. Is the loop above the shape you have in mind, or do you mean learning online in the stronger
   sense (a model updated during the flight, at 50 Hz, from the live state)? The small loop of the
   adaptation design (action correction) is that; the big loop (physics) is per flight.
2. How much of each round may be automatic: fly → label → fit → score → next plan proposed, with
   you approving the next flight? Or should the plan be fully automatic?
3. Flip-over: accept the survivable-flip approach (descend with motors running), and is any
   hardware protection on the table?
4. First round to try, if agreed: sharp stops on x at 2, 3, 4 m/s² with 8 s holds (one flight,
   ~1.5 min), then the cable refit with the stroke windows weighed in, scored on the first-link
   angle and the tip path. This is the cheapest test of whether the loop can fix the model where it
   is most wrong.

## During the flight: analysis (2026-10-28, evening; the user's choice)

The user means learning during the flight, from the live state, and points out that the pieces
exist: a robust live cable labeller, the native observer, the fast simulator. What follows is
what can be learned on which time scale, what the pieces give, what is missing, and the risks.

### What exists and what it gives

| Piece | State | What it gives online |
|---|---|---|
| Live cable filter (`deployment/cable_filter.py`, native) | runs in flight at 100 Hz, chain-search recovery, identical offline/online, no learned parts | ten marker positions and velocities with per-marker confidence every frame: the measurement for the cable |
| Native observer | in flight and in training | drone velocity and body rate from the poses |
| Native batch runtime (`BatchRuntime`, OpenMP) with `checkpoint`/`restore` | in training | many parameter candidates replayed in parallel from one measured state; ~8× real time per core at 10 links (mesh note), so ~100× on 16 cores |
| `FlightController` (firmware law at the flown gains, thrust map, battery) | in replays and training | the drone part of every candidate |
| Fitting residuals and scores (`aerowhip/fitting`) | offline | cable shape/tip and drone position/tilt errors of a replayed window: the same score online |
| Flight script | logs poses, cloud, motors, attitude, the sent command (also echoed on `/<robot>/sent_acceleration`) | all the inputs a learner needs are already on ROS topics |
| Measured parameter uncertainty (clustered confidence intervals) | offline | the starting ensemble |

### What can be learned during a flight, by time scale

- **Seconds** (hover): trims, effective mass, battery law. Already learned online by the host
  PID's integral; an explicit estimate would just name it.
- **Tens of seconds** (a few strokes): command delay, motor response, drone drag; the cable's
  stiffness, damping and drag at the speeds flown so far; the residual's steady part.
- **The whole flight** (a range of speeds and headings): the drag correction with speed, the
  heading dependence of the residual.
- **Never from flight** (shown on cf_2): roll/pitch inertia, arm, CoM. Stay nominal.
- **Not online at all**: the model's structure. If the sharp-stroke misfit (33° simulated vs
  170° real) is the clamp model or the damping law, no parameter value fixes it; the online
  estimator will show a parameter that keeps moving with the motion type, which is the
  diagnostic. Structure changes stay offline, informed by that diagnostic.

### The method: a moving-horizon estimator on the batch runtime

Every T (~1 s) during the flight:
1. Take the last W seconds (~2 s) of measured drone state, cable markers (confident ones only) and
   sent commands from the topics.
2. Replay that window from its measured start state with N parameter candidates in the batch
   runtime (N lanes), each lane one candidate drawn around the current estimate.
3. Score each lane as the offline fit does (cable shape and tip, drone position and tilt).
4. Update the estimate and its spread from the scores (a weighted mean and covariance of the
   candidates: a particle/ensemble update, or one Gauss–Newton step on the best candidates; the
   existing fit's least squares applies). Warm-start the next window.

Cost: N candidates × W s / (100× real time) ≈ N × 20 ms; N = 32 → ~0.6 s per update on 16
cores, ~1.2 s on this i7. So the estimate updates every one to two seconds and converges over the
flight. Ruled out: an EKF over the 60-dimensional constrained cable state (heavy, fragile), and a
learned residual trained on per-frame targets (shown unusable: label noise through the bending
stiffness).

The estimator is the offline fit's residual function on a sliding window with warm starts; it
reuses `aerowhip/fitting` and the batch runtime, with a ROS subscriber in front. Its output is the
parameter trajectory over the flight plus the spread, logged with the flight.

### Planning during the flight

While part k flies, the planner chooses part k+1: candidate primitives are the neighbours of what
has been flown (one level step up per kind and heading, new headings), each simulated with the
ensemble (batch, ~3 s of motion × 16 candidates × 10 neighbours ≈ 5 s wall); the next part is the
one with the largest ensemble disagreement on tip path and drone tilt, subject to the measured
envelope (the live filter's first-link angle per level flown: it reports the angle it saw) and the
battery. The flight script gains a streaming mode: the next part arrives on a topic and is
appended (rest-to-rest move, hold, part) under the same aborts, link-loss landing and the
battery-run voltage limit; the battery-run mode is already most of this.

### Validation before any flight

Replay recorded flights through the estimator as if live (poses, cloud and commands fed at their
stamps), on cf_2's archived flights and on cf_6's tuning flights: the parameter trajectory must
settle near the offline stage-2/3 values on cf_2's data, and its spread must cover the clustered
intervals. This costs no flight and is the test of the whole chain (labels → window → batch replay
→ update).

### Risks

- **Host load during flight.** A simulator run on this computer stalled the host and dropped the
  drone today. The estimator must run with a bounded budget on cores the stack does not use
  (`taskset` + `nice`), or on the AMD machine over the lab network; the link-loss landing now makes
  a stall survivable, but it still ends the flight. Test on the ground with the replay-as-live harness.
- **Labels during a flip.** The live filter is validated on normal flights; what it reports with
  the cable over the drone is not. Score only confident markers; the estimator ignores low-quality
  frames.
- **Parameter drift versus structure.** A parameter that follows the motion type is a structure
  error, not a value: the estimator must show that (per-part estimates) rather than average it away.

### Effort, roughly

Estimator on the batch runtime with the ROS front end: a few days. Replay-as-live harness: a day.
Planner (level creep + ensemble disagreement): one to two days. Flight-script streaming mode and the
cable-over-drone descent: a day. The sharp-stroke structure question stays an offline study that
the online diagnostics will point at.
