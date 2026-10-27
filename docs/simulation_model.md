# Simulation model

The plant is a quadrotor rigid body coupled to an inextensible elastic cable,
solved together so that cable forces and moments act on the drone. The engine is
C++ ([native/](../native)); Python wraps it in `aerowhip.physics`.

## Frames and units

World axes are Z up. Rotations map body to world; angular velocity, torque and
inertia use the body frame about the centre of mass. Public drone position and
velocity refer to the tracked top-plate origin. Quaternions are XYZW. Cable
nodes, including the attachment node 0, are world positions. All quantities are SI.

## Default parameters

Defined in [configs/physics/drone_cable.json](../configs/physics/drone_cable.json),
[configs/controllers/geometric.toml](../configs/controllers/geometric.toml) and
[configs/actuators/drone.json](../configs/actuators/drone.json), combined by
[configs/simulation.json](../configs/simulation.json).

| Quantity | Default |
|---|---|
| Drone mass (without cable) | 0.145 kg |
| Cable and marker mass (total) | 0.017 kg |
| Cable length / mesh | 0.995 m / 10 edges between 11 nodes (the marker intervals: 0.095 m, then 9 × 0.1 m) |
| Tracking origin → centre of mass / → attachment | (0, 0, −0.018) / (0, 0, −0.0423) m (attachment from the hanging cable in ten flights) |
| Bending stiffness EI / bending damping Cb | 1.179e−8 N m² / 8.174e−5 N m² s |
| Torsional stiffness GJ / torsional damping Ct | 1.179e−8 N m² / 8.174e−6 N m² s |
| External cable drag | 0.0814 s⁻¹ (linear in node velocity) |
| Maximum physics step | 1/600 s |
| Output / controller rate | 100 / 200 Hz |
| Command delay | 0.03 s |
| Motor response time | 0.015 s (first order, uncalibrated) |

The cable parameters are working assumptions from an earlier model and are to be
refitted from new flight data; they are not identified properties of the hardware.

## Cable

The cable is a discrete elastic rod: inextensible edges, naturally straight and
isotropic, with objective bending and twist and Kelvin–Voigt material damping
([native/material_frame.inc](../native/material_frame.inc)). External drag is a
separate linear term. A point clamp attaches node 0 to the drone, prescribing its
position and the body-down tangent; the first edge may bend away from that
tangent, and the clamp's bending moment acts on the drone. The tip is free.
There is no cable self-contact, ground or wall contact, or wind. Collapsed edges
and antiparallel adjacent edges are rejected as invalid states.

## Integration

The solver is a projected Lie-BDF2 method in float64 (backward Euler at startup
and after large step growth). Endpoint geometry, velocities, material rates,
loads and constraints are iterated together: at most 20 Picard iterations,
relative tolerance 2e−11, length tolerance 1e−10 m, velocity-constraint
tolerance 1e−9 m/s and four position projections per iteration. Failure to
converge raises an error rather than accepting an approximate step. Solver
convergence does not by itself establish physical accuracy.

## Controller and actuators

`aerowhip.control.geometric_controller` (NumPy) and `tensor_controller` (Torch)
implement a geometric controller with a static cable-weight trim
((drone + cable mass) × g). Two command paths exist:

- **Position/velocity/acceleration (PVA)** tracking for scripted flights.
- **Direct acceleration** (`acceleration_direct_v1`), used by training: a world
  acceleration request (excluding gravity) becomes the desired force, then attitude,
  collective thrust and body torque. It has no position or velocity feedback, so a
  zero command holds weight but does not brake or hold position.

Motor commands are mixed with per-motor clipping; a first-order motor response
produces the delivered wrench.

## Batched execution

Training and fitting use the batched tensor model (`aerowhip.physics.tensor_model`)
on CUDA, with the same equations. One Picard iteration is compiled and replayed
with CUDA graphs; on Linux with Torch built for CUDA ≥ 12.8 (and on Windows) the
iteration loop itself runs on the GPU. `AEROWHIP_CONDITIONAL_SOLVER=0` forces the
host-controlled loop. Both paths keep the same convergence limits.

## Exact checks

`tests/test_physics.py` verifies, without fitted parameters: free fall at −g·t
with no thrust or drag; a zero acceleration command holding a hanging cable
exactly at rest; and conservation of total momentum (to 1e−16) while a cable
bends internally during free fall, with edge lengths preserved to 1e−12 m.
