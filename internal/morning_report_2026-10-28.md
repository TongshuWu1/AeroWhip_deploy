# Morning report, 2026-10-28

## Done
- **Engine v2** (branch `engine-v2`, NOT merged): twist-free DER, quadratic normal drag (+ small linear), native
  flight controller (firmware thrust vector, power-law thrust map with battery voltage, attitude PD, bias),
  OpenMP batch runtime. 4 exact checks pass (free fall, hanging hold, momentum, batch = single bit-for-bit).
  Planar bending identical to v1; steps 25–40 % faster; helix release differs 46 mm (v1 twist).
- **UI rework** merged to `main` (all windows tested on screen; live real flight untested).
- **Labelling**: whole-recording Viterbi labeller on all 10 cable flights; uncertain markers left unlabelled.
  The live filter got a chain-search recovery (untested in flight).
- **Fits** (held out: random_high, whip; all numbers computed twice): cable 30.1 mm fit / 56.6 mm held out;
  v1 at configured values 47.1 / 52.8. Drone fit 43 mm fit / 57.6 held out (not adopted; thrust map and attitude
  response from direct identification are in `configs/controllers/crazyflie_flight.toml`).

## Not done / unverified
- Training is not on the batch runtime yet; the tensor engine on `main` cannot load v2 configs.
- v2 cable is not shown better than v1 on held-out flights. EI is unidentifiable.
- Drone attitude/motor/delay settings in the simulator config are unchanged.
- `circle_01` labelling crashed once with an unexplained index error (rerun fine).
- Equivalence reference still stale.

## Plan tomorrow
1. Look at the flagged label runs (review window) in `whip_01`, `line_star_01`, `vertical_01`.
2. Decide: merge `engine-v2` after porting the training environment to `BatchRuntime` (largest task).
3. Record more data (your call): (a) drone-only tilt step commands 2–10 m/s² to pin the attitude response;
   (b) slow steady circles with cable at 3–5 speeds for the drag law; (c) one more whip-like flight to test
   the held-out cable error.
