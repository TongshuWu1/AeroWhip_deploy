# Simulation audit and unified-engine proposal — 2026-10-26

Read-only audit of the two physics engines, with measurements on this machine
(24 CPU threads, RTX 4080, Torch 2.11/CUDA 13). Nothing in the repository was
changed by this audit. "Whip-like motion" below is a fixed 1.5 s open-loop
acceleration command (pull +X for 0.5 s, reverse 0.5 s, then +0.6, with a
vertical sine), reaching an 8.06 m/s tip speed on both engines.

## 1. What exists

Two implementations of the same model:

| | Scalar engine (`native/engine.cpp`, `runtime.cpp`) | Tensor engine (`native/tensor_physics.cpp`) |
|---|---|---|
| Used by | flight simulation, Crazyswarm server, real-flight cable, fitting (prescribed/native) | training environment, residual fitting (autograd) |
| Execution | one drone, CPU, plain C++ | batched lanes, Torch/ATen ops, CUDA graphs |
| Equations | identical: rigid body with root node eliminated, inextensible edges via tridiagonal constraint solve, discrete bending with point clamp, twist, Kelvin–Voigt material damping, linear drag, residual nets, gravity, first-order motors, variable-step BDF2 with backward-Euler restart, 20 Picard iterations, tolerances 2e-11 / 1e-10 m / 1e-9 m/s, 4 projections | same |
| Agreement | with identical commands for 1 s: drone and cable positions differ by ≤ 4e-15 m | |

Confirmed differences (engine comparison, file:line in the agent report):
- **Command delay.** The flight engine applies the 30 ms delay at exact event
  times (`runtime.cpp:110`); the training environment rounds it to whole 10 ms
  observation steps (`envs/environment.py`). Identical today only because 30 ms is a
  multiple of 10 ms; nothing checks this.
- **Validity checks.** The scalar engine checks rotation orthogonality, SPD pivots,
  non-negative thrust, inertia positive-definiteness and residual-network shapes;
  the tensor engine checks only finiteness, edge lengths and antiparallel edges (a
  bad state becomes NaN and the lane is dropped). Dense LU replaces Cholesky, so a
  loss of positive-definiteness is not detected.
- **Failure semantics.** Scalar: exception, clocks rolled back. Tensor: lane becomes
  NaN and is frozen at its last finite state. Intended, but it means the two paths
  cannot be compared on failure cases.

## 2. Numerical accuracy

Scalar engine, whip-like motion, 1.5 s.

**Time step** (reference 1/2400 s):

| step | max tip position error | max tip speed error | ms per 10 ms step |
|---|---|---|---|
| 1/1200 s | 0.16 mm | 0.001 m/s | 4.2 |
| **1/600 s (current)** | **0.82 mm** | **0.007 m/s** | **2.4** |
| 1/300 s | 1.9 mm | 0.015 m/s | 1.9 |
| 1/150 s | 8.0 mm | 0.056 m/s | 1.1 |

The current step is converged. A larger step barely helps because more Picard
iterations are needed per substep. The step is not the lever.

**Mesh** (reference 40 links, same motion):

| links | max tip error | mean tip error | max drone error | ms per step |
|---|---|---|---|---|
| **10 (current)** | **36.8 mm** | 5.8 mm | **25.3 mm** | 2.4 |
| 20 | 10.9 mm | 1.7 mm | 8.1 mm | 10.4 |
| 30 | 3.1 mm | 0.5 mm | 2.3 mm | 27 |
| 40 | — | — | — | 57 |

The 10-link mesh is under-resolved for this motion by about 4 cm at the tip and
2.5 cm at the drone (the drone error means the cable load on the vehicle is also
wrong). Convergence is roughly second order. **At least 20 links are needed, 30 to
be safe.** Cost grows like links^2.3 because every Picard iteration solves a dense
(6+4·links)² system.

Picard iterations per 1/600 s substep under whip motion: 1 to 6, mean 4.4 (tensor
engine, one lane); at hover, exactly 1.

## 3. Speed

Per 10 ms of simulated time, 10 links:

| engine | condition | time | aggregate speed |
|---|---|---|---|
| scalar, 1 process | whip | 2.4 ms | 4× real time |
| scalar, 8 processes | whip | 3.1 ms each | 26× |
| scalar, 16 processes | whip | 4.3 ms each | 37× |
| scalar, 16 processes, **20 links** | whip | 15 ms each | 10.6× |
| tensor, CPU, 1 lane | whip | 117 ms | 0.09× |
| tensor, GPU, 1 lane | hover | 11 ms | 0.9× |
| tensor, GPU, 64 lanes | hover | 16 ms | 40× |
| tensor, GPU, 128 lanes | hover | 22 ms | 59× |
| tensor, GPU, 256 lanes | hover | Inductor compile error | — |
| **tensor, GPU, 64 lanes** | **whip** | **128 ms** | **5×** |
| tensor, GPU, 128 lanes | whip | 228 ms | 6× |

Under whip motion the GPU path delivers about 5× real time. One training run of
the default experiment simulates 12,800 × 4 s = 51,200 s, so its physics alone takes
about 2.8 h at 10 links, and (extrapolating the links^2.3 scaling) more than 15 h at
20 links. The 16-process scalar engine is 7× faster than the GPU today at 10 links.

Why the tensor engine is slow (confirmed by profiling and the code comparison):
- One implicit iteration dispatches ≈ 11,000 ATen operations (`as_strided`,
  `expand`, `view`, `mul`, …) on tensors of a few dozen elements. On CPU that is
  4 ms per iteration per lane; on GPU, CUDA graphs remove launch overhead but the
  work is thousands of tiny dependent kernels.
- All lanes iterate until the slowest lane converges (5–7 slots under whip motion),
  so the per-step cost is set by the worst lane and converged lanes recompute.
- The material-frame Jacobian is a sequential recurrence over edges; dense LU of the
  (6+4·links)² system; the BDF predictor, rotation log and observation are recomputed
  in every iteration; buffers are allocated per slot.
- The environment adds six device-to-host copies per 100 Hz step and a Python loop
  over lanes for the reward.

Why the scalar engine is not faster than it is: forward-mode derivatives carry a
fixed 256-double array per scalar (zeroed on every operation), the Jacobian and
Rayleigh operator are dense, several quantities are computed twice per iteration,
and matrices are reallocated per iteration. Hard limit: 83 edges.

## 4. Physical completeness

Absent from both engines:
- **Aerodynamics.** Cable drag is linear, `a = −0.081·v`. A quadratic estimate for a
  3.5 mm cable (Cd ≈ 1.2) gives 1.7× more drag at 1 m/s, 7× at 4 m/s, **14× at 8 m/s**,
  21× at 12 m/s. At whip speeds the simulated cable is nearly undamped by air. The
  learned residual networks exist but are unfitted and disabled.
- **Ground, self-contact, cable–drone contact:** none (the floor is display only).
- **Wind and rotor downwash:** none.
- **Motors:** per-motor clipping and a first-order lag; no rotor-speed state, no
  voltage effects; parameters are marked uncalibrated.
- **Sensing:** observations are exact state; no noise or latency model.
- Cable EI, Cb, GJ, Ct and drag are working assumptions from the earlier one-way
  model; GJ equals EI and Ct = Cb/10 by construction, not measurement.
- Hypothesis from the code reading (not traced to the end): twist stiffness reaches
  the body only through the damping coupling, not as a direct elastic moment.

## 5. Proposal: one engine, used everywhere

**Keep the physics, change the execution.** The equations are the same in both
engines and are converged in time step; what is wrong is the mesh (too coarse), the
drag model, and the execution cost. The design:

1. **One C++ core** (`native/engine.cpp`, as now) is the only implementation. The
   tensor/ATen engine is retired.
2. **Batched by lanes, not by tensors.** A `BatchRuntime` holds N independent
   runtimes and advances them with OpenMP, each lane iterating to its own
   convergence with exact event-time command delays. One `step(actions)` call per
   10 ms returns all observations as one array; no per-lane Python work, no
   device copies. The flight simulation, real-flight cable and fitting use the
   same runtime with N = 1.
3. **Banded linear algebra.** The bending, constraint and mass terms are local, so
   the Picard system is block-banded. Two options for the material-frame damping,
   which is what makes today's operator dense:
   - (a) keep the material-frame form and the dense solve but switch to Cholesky,
     reuse factorizations, and remove the 256-element derivative arrays; or
   - (b) express the Kelvin–Voigt damping on the local curvature rate (equivalent
     for an isotropic cable), which makes every operator banded and the solve O(links).
   (b) is the one that scales to 30 links; it is a modelling change that needs your
   approval and is naturally paired with the coming refit.
4. **Validity and failure handling** as in the scalar engine today (orthogonality,
   SPD, thrust, inertia), for every lane; a failed lane is reported, not silently NaN.
5. **Fitting without autograd.** Physical parameters (3–6 of them) use finite-difference
   gradients, which `coupled_fit` already does on the native engine. The learned
   residual networks are dropped in favour of a physical quadratic drag term with one
   or two fitted coefficients (research decision).

Expected speed (estimates, to be measured as the first milestone): with OpenMP
alone, 37× real time at 10 links on 16 cores (measured today); with the banded
solver and the derivative cleanup, 20–30 links at ≈ 1–2 ms per step per lane, i.e.
roughly 80–160× real time on 16 cores, putting the physics of a 12,800-episode run
at 5–10 minutes. The next bottleneck is then the Python fold reward (≈ 2 ms per lane
per 100 Hz step in the old measurement), which will need vectorizing.

## 6. Order of work

A. **Now, no model change:** make the engine parity comparison a permanent test;
   apply the command delay exactly in the environment (or require a 10 ms
   multiple); bring the tensor path's validity checks up to the scalar engine's
   until it is retired.
B. **Unified engine:** `BatchRuntime` with OpenMP over the existing scalar engine
   (expected 7× faster than the GPU today under whip motion, with no numerical
   change — parity to 1e-12 against the current engine); then the derivative
   cleanup and banded solver; then the training environment on top of it; retire
   the tensor engine; re-record the equivalence reference.
C. **Model completeness (research decisions):** default mesh 20 or 30 links;
   quadratic drag with fitted coefficient; damping form (3b); ground contact;
   motor calibration from the new flight data; then the refit.
