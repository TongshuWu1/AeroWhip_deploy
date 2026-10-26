# Current Stage 1: continuous directional fold transport

Last reviewed: 2026-10-25. Implemented experiment, **not a validated whipping classifier or a successful learned policy**.

## What is running, and what it should learn

Training is stopped. The latest production run, `runs/ppo_stage1/20261025-215750-176045`,
stopped during its first collection: **zero completed collections, zero completed episodes,
zero optimizer updates**. Its checkpoint is an initialization, not a trained result.

The only supported training recipe is
[`whip_stage1_ppo_continuous_directional_four_second_v6.json`](../configs/learning/whip_stage1_ppo_continuous_directional_four_second_v6.json),
with its [task contract](../configs/tasks/whip_stage1_continuous_directional_four_second_v6.json).
It asks the drone to generate a coherently moving fold and associated outward cable opening
in **world +X**. Sideways/backward preparation is allowed. It does not ask the tip to hit a target.
Repeated useful output can earn reward throughout the episode; there is no numbered-attempt
state, per-whip bonus, or maximum number of whips.

The old excitation, shape-release, fold-formation, first-hit, and Stage 2–4 recipes are retired.
They must not be interpreted as alternative current presets or automatically resumed.

## Timing, policy and training budget

| Item | Current value |
|---|---|
| Cable | 10 links, 11 nodes; rest length 0.9525 m |
| Maximum episode duration | 4 s |
| Physics maximum step | 1/600 s, unchanged native solver |
| Controller / observation / policy | 200 / 100 / 50 Hz |
| Per episode | 400 reward samples, 401 states including reset, 200 actions |
| Actions | Normalized XYZ acceleration commands, scaled to (8, 8, 4) m/s²; fixed yaw |
| Collection width | 64 episodes |
| Budget | 200 collections = 12,800 episodes |
| Initialization | Fresh random actor and critic; seed 20261028; no actor transfer |
| PPO | 5 epochs, minibatch 256, learning rate 0.0003, hidden width 128 |
| PPO safeguards | Clip ratio 0.2, cohort-mean KL limit 0.02, up to 8 post-step trials |
| Return estimation | Discount 1, GAE lambda 0.95 |
| Policy history | 401 samples; time normalization by 400 observation intervals |
| Evaluation cadence | First collection, then each 512 completed episodes (collections 8, 16, …), and final |

The actor sees measured state histories, not reward-internal fold tracks or privileged solver
state. The value function also sees simulator state and task statistics. The actor input has
88 selected columns from a 96-column frame; the critic receives 219 privileged values.
Zero acceleration input does **not** brake or hold position. This is acceleration control,
not commanded drone position.

## Reward, frames and units

At each 100 Hz endpoint, let:

- `a`, `v_a`: cable attachment position and velocity in world coordinates;
- `p`, `v`: tip position and velocity in world coordinates;
- `r = p − a`, `w = v − v_a`, `L`: rest cable length;
- `e_x = (1,0,0)`; `clip01(x) = min(1,max(0,x))`.

The same-sample factors are

```text
H = clip01(max(0, r.x) / L)
A = clip01(min(v.x / ||v||, w.x / ||w||))
E = w · r / ||r||
U = max(0, min(v.x, w.x, E))
rate = 10 F H² A² U / L
Q = sum(0.01 × rate)
```

A zero speed has zero directional cosine. For separation below `1e-6 L`, extension speed
is zero. `H`, `A` and `F` are dimensionless; `U` is m/s. The reward rate is integrated in
seconds. All output factors refer to the same observation, not unrelated peak values.

Consequences:

- A straight, static cable earns zero output credit.
- Rigid translation cannot earn output credit: attachment-relative speed is zero.
- Rigid rotation at constant end-to-end distance has zero extension speed.
- Pure sideways/backward/downward output earns zero +X credit.
- Oblique forward output can earn partial credit; there is no hard horizontal cone.
- Forward tip motion while the cable shortens earns zero output credit.
- A high speed without recent coherent fold transport earns zero credit.

This is a kinematic proxy. Positive reward alone does not prove energy focusing, a physically
correct fold, useful target impact, or acceptable drone behavior.

## How fold support F is measured

The physics mesh stays at ten links. The dense curve below is a measurement of that mesh,
not a finer physical simulation.

1. Normalize each link tangent. At the nine internal joints, compute turning angle with
   `atan2(||u_i × u_(i+1)||, u_i · u_(i+1))` and its turning-axis direction.
2. Spread these angles over material arc length using normalized Gaussian kernels.
   Bandwidth is `0.075 L / 0.9525` m; the measurement grid has 1,001 points between the
   first and last internal joints.
3. Find bend regions and their half-height cores. Associate a region through time only
   when the previous/current core overlap is unique in both directions. Ambiguous
   split/merge identities, plateaus and unresolved antiparallel joints cannot supply support.
4. For the same track over 11 consecutive samples (100 ms), form:

```text
strength  = clip01((max turning angle in window − π/2) / (π/3))
coherence = clip01(min directional coherence in window / 0.8)
travel    = clip01((current material centre − old material centre) / (0.1 L))
transport = strength × coherence × travel
```

Directional coherence is the magnitude of the angle-weighted sum of turning-axis vectors,
divided by the sum of turning angles. It separates a consistently turning region from
opposing bends. Positive material travel means motion toward the cable tip.

Take the maximum transport across eligible regions. `F` is the maximum of these recent
transport values after a linear fade to zero over 100 ms. A gap in observations or an
ambiguous/reappearing identity cannot reuse the prior track history.

These windows, bandwidths and angle scales are the existing experimental choices. Cleanup
has not changed them. In particular, the current reward does not require a separately
completed 150° formation → propagation → 60° opening event.

Sources: [fold geometry](../tasks/whip/horizontal_fold.py),
[region extraction](../tasks/whip/fold_geometry.py),
[continuous reward](../tasks/whip/continuous_whip.py).

## Workspace cost

The drone's **tracked origin**, not the cable attachment or tip, is charged for leaving its
operating region. Horizontal centre is reset XY; height limits are absolute world Z.

```text
radial_excess = max(0, ||drone_xy − reset_xy|| − 0.55 m)
vertical_excess = max(0, 1.5 m − drone_z) + max(0, drone_z − 2.6 m)
C = sum(10 × (radial_excess² + vertical_excess²) / 400)
```

`C` is an uncapped fixed-horizon average squared-distance cost. A constant 0.1 m radial
excess for all four seconds costs 0.1; a constant 1 m excess costs 10. There is no saturated
negative-return transform, acceleration cost, travelled-path cost or elapsed-time cost.
The cable may extend outside the drone workspace. This is a soft cost, not enforced containment.

For a full numerical episode, **return = Q − C**. Transitions receive differences in the
accumulated return; the ledger telescopes to the episode return. Shortening the old seven-second
horizon to four reduced the time available to accumulate Q, while C retains an average-cost
normalization. The relative balance is therefore not identical to a seven-second experiment.

## Termination and validity

- Normal completion is the four-second horizon, regardless of how many outputs occur.
- Workspace exit does not end an episode.
- Tilt above 60° or any cable node below world Z = 0.05 m is recorded as a physical-envelope
  violation. The finite simulation continues, and the episode is reported task-invalid.
  There is no additional fixed failure penalty for these violations.
- Numerical corruption, an inadmissible solver state or invalid rotation ends the lane.
  The return is `Q_prefix − C_prefix − (4 − tau)`, where `tau` is the failing substep time.
  The corrupt endpoint is not passed to geometry or rewarded. There is no fixed −10 total.
- Interrupted collections are omitted from PPO; their partial episodes do not advance the
  completed-episode entropy clock.

The simulator has no ground-contact model. A finite below-ground trajectory can continue;
that does not make it physically valid. Neither low return for every physical violation nor
successful containment is mathematically guaranteed by this reward.

## Exploration and evaluation

For `n` completed training episodes and the original planned budget `N = 12800`,

```text
entropy coefficient = 0.001 + 0.019 exp(−8 n/N)
```

It starts at 0.02 and is about 0.0010471 at 75% of the budget (collection 150).
This coefficient weights PPO's entropy term; it is not an exploration probability.
The Gaussian policy also learns its scale. Action noise has a 100 ms AR(1) correlation,
with `rho = exp(−0.02/0.1) ≈ 0.81873`; new episodes reset the noise history.
Exact resume retains the saved original budget and episode clock.

Training preserves all-lane accounting, rewards, numerical validity, cable motion, and drone
motion; selected individual replays are available in the UI's whipping-diagnostics page.
Report Q, C, numerical abort cost, net return, physical validity, fold support and output
geometry separately. All attempts, including failures, belong in evaluation denominators.

Deterministic evaluation uses one physical initial state with the saved actor batch layout
(128 actor lanes), not 128 independent physical trials. `best.pt` is selected by deterministic
return, **not** by an independently validated whip-success criterion. A continuous positive
score is not labelled successful whipping. There is no automatic Stage 2 promotion.

## Starting on another computer

Follow the [handoff setup](handoff-20261025/README.md), then:

```bash
source .venv/bin/activate
python run_training.py --start
```

This opens the UI and starts a fresh current Stage 1. Without `--start`, it only opens the UI.
Do not restore old presets into the active configuration directory. To inspect old reward
implementations or replay their models, use a separate checkout of commit `3c9a49e` and the
archived data. Source-hash checks for exact resume remain enforced.
