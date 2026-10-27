# Strike task

The training objective: the cable tip hits a target point beyond the drone's reach with a fast
tip, then the drone recovers. It is the default experiment,
[`configs/experiments/strike_x_v1.json`](../configs/experiments/strike_x_v1.json), registered as task
`strike_x_v1` ([tasks/strike](../aerowhip/tasks/strike)). The whip mechanism (a bend travelling
down the cable) is not rewarded; it is measured and reported. Every number below is read from the
experiment file: horizon, action limits and validity limits are `environment.*`
([envs/configuration.py](../aerowhip/envs/configuration.py)), reward constants and thresholds are
`task.*` ([tasks/strike/configuration.py](../aerowhip/tasks/strike/configuration.py)). The values
shown are the file's current values, not constants in code.

## Why outcome, not mechanism

A whip is energy focusing: the drone puts a modest motion into the cable and the cable
concentrates it so the tip moves far faster than the drone. The travelling bend is how that
happens; the fast, far tip is what a whip is. The recorded scripted whips (calibration flights,
`whip`) reach tip speeds of 4–5 m/s while the drone moves 1–2 m/s, at 92–98 % of the cable length,
with the sharpest joint angle on the ten-link mesh at 96–116°. Rewarding the fold shape would grade
the policy against a template those recordings do not match; rewarding the outcome lets the
policy find the stroke, and gives a success you can count in real flights.

## Episode

| Item | Current value |
|---|---|
| Cable | 10 links, 11 nodes; rest length 0.985 m |
| Start | hover at the scene origin, cable hanging (70 %); a recorded whip state (30 %, see below) |
| Maximum duration | 5 s; a hit ends the episode `recovery_s` = 1 s later |
| Controller / observation / policy | 500 / 100 / 50 Hz |
| Actions | normalized XYZ acceleration commands, scaled to (8, 8, 4) m/s²; fixed yaw; zero means hover |
| Target | 1.3 m along world +X from the workspace origin (final stage), 0.35 m below it |
| Workspace | radius 0.55 m (final stage) around the reset XY, heights 1.5–2.6 m |
| Termination | tilt above 60° or any cable node below 0.05 m ends the episode (`terminate_on_violation`) |

The policy sees the same estimator output as in flight (`aerowhip.envs.observation`): the
delayed, noisy drone pose, the velocity and body rate fitted to the last five poses, the cable
filter's ten marker positions, velocities and confidences from the simulated marker cloud, the
target relative to the attachment (three columns, new in this task), the previous action and the
time. The critic also receives the true state and the task state (`StrikeEpisode.features`).

## Definitions

Let `p`, `v` be the tip's position and velocity, `a`, `v_a` the attachment's, `T` the target,
`L` the cable length, `v*` the stage speed threshold.

- **Hit:** the segment between the tip's positions at consecutive 100 Hz samples passes within
  `hit_radius_m` = 0.10 m of `T` while `||v|| ≥ v*`. Checked on the segment, so a tip moving
  4 cm per sample cannot step over the radius.
- **Whip event** (reported, never rewarded): at the sample of peak tip speed, `||v|| ≥ v*`,
  `||v|| ≥ amplification_ratio` (2) × the attachment's largest speed over the previous
  `amplification_window_s` (0.5 s), and `||p − a|| ≥ extension_fraction` (0.9) × `L`.
- **Recovered:** at the episode end after a hit, the drone is inside the workspace and the
  attachment moves slower than `recovery_speed_mps` = 0.3 m/s.
- **Strike** = hit. Report strike rate, whip-event rate and recovery rate separately; a strike
  with a whip event is the behaviour the paper claims. Inside the final 0.55 m workspace the drone
  cannot exceed about 3 m/s at 8 m/s², so a 4 m/s tip can only come from amplification: the
  workspace and the threshold together demand a whip without a fold term.

## Reward

The return is accumulated at every 100 Hz sample and transitions receive its differences, so the
per-decision rewards sum to the episode return.

Before the hit:

```text
progress = progress_weight × (d_prev − d) / L            d = ||T − p||   (potential-based)
approach = approach_weight × clip(v · (T − p)/d, 0, v*) / v* × 0.01 s   (bounded)
```

The progress term telescopes to the net distance closed, so moving the tip back and forth earns
nothing. The approach term pays at most `approach_weight` per second at full threshold speed
toward the target.

At the hit: `strike_bonus` (10) + `speed_bonus_per_mps` (1) × (`||v||` − `v*`). After the hit no
shaping is paid; at the episode end, `recovery_bonus` (2) if recovered.

Costs: `workspace_cost_weight` (10) × mean squared excess of the drone's tracked origin outside
the workspace over the horizon; `action_cost_weight` (0.01) × mean squared normalized action;
on termination (tilt, floor, numerical failure) `failure_cost_per_remaining_s` (1) per
unexecuted second.

## Curriculum

Stages advance when the strike rate over the last `window_episodes` (512) training episodes
reaches `advance_strike_rate` (0.7); the last stage is final and is the evaluation stage.

| stage | target distance | speed threshold | workspace radius |
|---|---|---|---|
| 1 | 0.8 m (within reach) | 2 m/s | 0.8 m |
| 2 | 1.0 m | 3 m/s | 0.7 m |
| 3 (final) | 1.3 m (beyond the cable) | 4 m/s | 0.55 m |

**Reference-state starts.** `reference_state_fraction` (0.3) of the training lanes start from a
state sampled along the recorded scripted whips instead of hover:
[`scripts/build_reset_bank.py`](../scripts/build_reset_bank.py) builds native initial states every
0.1 s inside the whip strokes of the `whip` calibration flights (drone pose, causal velocities,
the labelled cable projected onto the configured lengths), moved so the flight's hover origin is
at zero with zero yaw. The current bank, `runs/reset_bank/whip.npz`, holds 1,302 states from
whip/001 and whip/002 (tip speeds up to 4.2 m/s). A reference state the cable filter cannot
label from its first frame, or that the engine rejects, falls back to hover and is counted
(`reference_fallbacks` in the provenance). The workspace origin and the target stay those of the
hover start. Evaluation never uses reference starts.

## Domain randomisation

`environment.randomisation`, drawn at every reset of a training cohort as a function of the
randomisation seed and the cohort index. Per lane: battery voltage (uniform in
`battery_voltage_v`) and a constant world acceleration bias (normal, per-axis sd
`acceleration_bias_sd_mps2`), both native per-lane inputs of the flight controller. Per group of
lanes (`groups` = 4 native batch runtimes per cohort): body drag scale and bending damping scale
(log-uniform), motor response time and command delay (uniform). The current ranges are
provisional (voltage 6.9–7.6 V, bias sd 0.03/0.20/0.15 m/s², drag ×0.8–1.25, Cb ×0.5–2,
response 5–30 ms, delay 20–40 ms); they are to be replaced by the 95 % intervals of the adopted
least-squares fits. The critic's command-queue slots use the nominal delay.

## Training and evaluation

| Item | Current value |
|---|---|
| Collection width | 64 episodes |
| Budget | 25,600 episodes |
| PPO | as the fold-transport recipe: 5 epochs, minibatch 256, learning rate 0.0003, hidden 128, clip 0.2, KL limit 0.02, AR(1) action noise |
| Evaluation | `evaluation_episodes` = 200 independent deterministic episodes at the final stage with their own model draws, every 512 training episodes; the replay saved is the first lane's |

The evaluation logs the mean return, the strike, whip-event and recovery rates, the peak tip
speed, amplification and extension, the closest approach and the sharpest joint angle.
`best.pt` is selected by the mean evaluation return. For real flights, count strikes against a
physical target over N attempts; the tip marker is often lost at its fastest moment, so a contact
target is preferred over the cameras for the hit itself.

## Running it

```bash
python scripts/train.py --start
```

Runs are written to `runs/ppo/`. No training run on this task has been completed yet; the
numbers above are the task's definition, not results.
