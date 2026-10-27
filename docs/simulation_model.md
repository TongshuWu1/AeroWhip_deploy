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
[configs/controllers/geometric.toml](../configs/controllers/geometric.toml),
[configs/controllers/crazyflie_flight.toml](../configs/controllers/crazyflie_flight.toml),
[configs/controllers/crazyflie_firmware.toml](../configs/controllers/crazyflie_firmware.toml) and
[configs/actuators/drone.json](../configs/actuators/drone.json), combined by
[configs/simulation.json](../configs/simulation.json).

| Quantity | Default |
|---|---|
| Drone mass (with battery and clamp, without cable) | 0.146 kg |
| Cable and marker mass (total) | 0.017 kg |
| Cable length / mesh | 0.995 m / 10 edges between 11 nodes (the marker intervals: 0.095 m, then 9 × 0.1 m) |
| Tracking origin → centre of mass / → attachment | (0, 0, −0.018) / (0, 0, −0.0423) m (attachment from the hanging cable in ten flights) |
| Bending stiffness EI / bending damping Cb | 1.14e−7 N m² (not identifiable) / 3.02e−4 N m² s |
| Cable diameter / normal drag coefficient Cd / air density | 3.5 mm / 0.69 / 1.2 kg m⁻³ |
| Linear drag | 0.0526 s⁻¹ (on node velocity, mass proportional) |
| Drone air drag (body x, y / z) | 0.52 / 0.85 s⁻¹ (linear, mass proportional, on the COM velocity in body axes) |
| Maximum physics step | 1/600 s |
| Output / controller rate | 100 / 500 Hz (the firmware attitude rate) |
| Attitude loop / command delay / motor response | firmware law: Mellinger gains of the flight (kR 70000, kw 20000, kd_omega 200, massThrust 17700) / 0 s / 0.0225 s; roll/pitch inertia 1.877e-4 kg m² and motor arm 0.0613 m fitted together on the drone-only flights (correlated until the arm is measured). Identified law (fallback): 20 rad/s, ζ = 1 / 0.03 s / 0.015 s |
| Thrust map | T = 0.553 (1 + 0.177 (V − 7.2 V)) T_cmd^2.2 N, T_cmd the firmware thrust in N |
| Motor thrust limit | 0.85 N per motor (peak measured collective thrust 3.39 N) |

The cable and drone values are identified on the 2026-10-27 calibration flights
(see [Flight data and model fitting](data_and_fitting.md)); the held-out flights and
the residual errors are reported there.

## Cable

The cable is a discrete elastic rod: inextensible edges, naturally straight and
isotropic. Twist is quasi-static: with an isotropic section and a free tip the twisting
moment vanishes, so the material frame is the frame parallel-transported from the clamp
and no material angle is a state (the discrete viscous threads assumption,
Bergou et al. 2010). Bending energy is (EI/2l)|κb|² per vertex; bending damping is
Kelvin–Voigt, with dissipation (Cb/2l)|dk/dt|² on the material curvature (k₁, k₂) in that frame,
which is objective: a rigid motion of drone and cable together dissipates nothing
([native/material_frame.inc](../native/material_frame.inc)). Air acts on each node through its
tributary length ℓ: a quadratic normal drag −½ρ C_d d ℓ |v_n| v_n on the cylinder (v_n is the
velocity normal to the local tangent; still air) plus a small linear term. A point clamp
attaches node 0 to the drone, prescribing its position and the body-down tangent; the first
edge may bend away from that tangent, and the clamp's bending moment acts on the drone. The
tip is free. There is no self-contact, ground or wall contact, or wind. Collapsed edges and
antiparallel adjacent edges are rejected as invalid states.

On the identified values the cable is gravity dominated (EI/(m g L²) ≈ 3·10⁻⁷) and its bending
is strongly damped (Cb/EI of order 10³–10⁴ s): its shape is set by gravity, inertia, bending
damping and air drag, and EI is not identifiable from flight data (any small value fits).

## Drone air drag

The drone carries a linear air drag per body axis on its centre-of-mass velocity relative to
still air, F = −m R diag(k_x, k_y, k_z) Rᵀ v (`drone.body_drag_s_inv`), the usual first-order
model of rotor and body drag on a small quadrotor (horizontal axes share one rate). It was
identified because the open-loop replay error of the earlier model grew with flight velocity
along the direction of travel on every flight, with and without cable; the fit is reported on
the data and fitting page.

## Integration

The solver is a projected Lie-BDF2 method in float64 (backward Euler at startup
and after large step growth). Endpoint geometry, velocities, material rates,
loads and constraints are iterated together: at most 20 Picard iterations,
relative tolerance 2e−11, length tolerance 1e−10 m, velocity-constraint
tolerance 1e−9 m/s and four position projections per iteration. Failure to
converge raises an error rather than accepting an approximate step. Solver
convergence does not by itself establish physical accuracy.

## Controller and actuators

Two controllers drive the plant:

- **Flight pipeline** (`FlightController`, [native/flight_controller.hpp](../native/flight_controller.hpp)):
  the Crazyflie's onboard controller as flown, in one of two laws selected by the controller
  file's `law`:
  - `firmware` ([crazyflie_firmware.toml](../configs/controllers/crazyflie_firmware.toml)): a
    line-by-line port of the firmware's Mellinger controller and legacy power distribution
    ([native/firmware.hpp](../native/firmware.hpp)), in single precision as the firmware computes
    it, running at the firmware's 500 Hz attitude rate with the gains of the flight
    configuration (`ctrlMel.*`; the translation gains are zero for the acceleration interface).
    The sent world acceleration a (trims included) is the firmware setpoint; the firmware's
    thrust `massThrust · (m_fw (a + g ẑ) · z_body)` and its int16 roll/pitch/yaw outputs go
    through the legacy mixer and cap to four PWM values; each motor's thrust is
    T_i = κ' / 4 · (pwm_i / massThrust)^p with the identified map below (κ' includes the battery
    voltage), which reproduces the collective map when the four motors are equal. The port is
    checked against the vendored firmware sources compiled into a test harness
    ([tests/test_firmware_port.py](../tests/test_firmware_port.py)). The motor layout of the
    actuator file follows the firmware's legacy mixer (M1 front-right, M2 rear-right, M3
    rear-left, M4 front-left; M2 and M4 produce positive yaw torque). Battery compensation of
    the firmware is assumed off.
  - `identified` ([crazyflie_flight.toml](../configs/controllers/crazyflie_flight.toml)): the
    reduced model identified on the drone-only flights, kept as the fallback. F = m_fw (a + g ẑ)
    with m_fw = 0.144 kg; the body z axis is driven towards F/|F| by an attitude PD loop; the
    firmware thrust T_cmd = F · z_body is mapped to the delivered thrust by the same map.

  In both laws an optional constant world acceleration bias models the slowly varying
  disturbances seen in flight (about 0.1 m/s²) for randomization.
- **Geometric controller** (`aerowhip.control.geometric_controller`): position/velocity/
  acceleration tracking with a static weight trim, used by the interactive simulator and the
  scripted-flight tools.

Motor commands are mixed with per-motor clipping (the firmware law supplies motor forces already
mixed and capped by the firmware; the allocation reproduces them); a first-order motor response
produces the delivered wrench.

## Batched execution

`BatchRuntime` steps many independent lanes of one model in parallel on the CPU (OpenMP),
each lane with its own flight controller (battery voltage, acceleration bias); a lane that
raises is marked failed and left unchanged. Lanes reproduce a single runtime bit for bit.
The calibration replays (cable with the measured clamp motion; drone and coupled model driven
by the sent commands) run windows in parallel threads on the same engine.

## Exact checks

`tests/test_native_physics.py` verifies, without fitted parameters: free fall at −g·t with no
thrust or drag; a weight-equal thrust holding a hanging cable exactly at rest; conservation of
total momentum (to 1e−12) while a cable bends internally during free fall, with edge lengths
preserved to 1e−12 m; the drone body drag changing the total momentum by exactly
−m R diag(k) Rᵀ v; and batch lanes reproducing a single runtime exactly.
