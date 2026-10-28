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
[configs/controllers/onboard.toml](../configs/controllers/onboard.toml),
[configs/controllers/crazyflie_flight.toml](../configs/controllers/crazyflie_flight.toml),
[configs/controllers/crazyflie_firmware.toml](../configs/controllers/crazyflie_firmware.toml) and
[configs/actuators/drone.json](../configs/actuators/drone.json), combined by
[configs/simulation.json](../configs/simulation.json).

| Quantity | Default |
|---|---|
| Drone mass (with battery and clamp, without cable) | 0.145 kg (weighed 2026-10-27) |
| Cable and marker mass (total) | 0.016 kg (weighed 2026-10-27; assembly 0.162 kg) |
| Cable length / mesh | 0.977 m / 10 edges between 11 nodes (the marker intervals from the labelled flights: 83.2, 100.1, 99.3, 101.7, 100.3, 98.9, 101.2, 98.9, 100.1, 96.9 mm, the means over all frames of the 21 cable flights, std about 1 mm) |
| Tracking origin → centre of mass / → attachment | (0, 0, −0.018) / (0, 0, −0.055) m (attachment measured 2026-10-27; the hanging-cable fit had placed the first marker within 3 mm of this) |
| Bending stiffness EI / bending damping Cb | 1.40e−4 N m² (×/÷ 1.65) / 4.15e−5 N m² s (×/÷ 1.21), 95 % |
| Cable diameter / normal drag coefficient Cd / air density | 3.5 mm / 0.75 ± 0.10 / 1.2 kg m⁻³ |
| Linear drag | 0.178 s⁻¹ (×/÷ 1.21; on node velocity, mass proportional) |
| Cable residual | steady force per unit length, body frame: sideways 20 / 28 mN/m (body x / y) at the clamp, fading to about 2 mN/m by mid-cable; −9.5 mN/m vertical, constant (see below) |
| Drone air drag (body x, y / z) | 0.59 / 0.48 s⁻¹ (linear, mass proportional, on the COM velocity in body axes) |
| Maximum physics step | 2 ms (one step per firmware controller period) |
| Solver stopping tolerance | 1e−5 relative (`simulation.solver_tolerance`: training, simulator, tools); 1e−7 in the model fits (`simulation.fit_solver_tolerance`) |
| Output / controller rate | 100 / 500 Hz (the firmware attitude rate) |
| Attitude loop / command delay / motor response | firmware law: Mellinger gains of the flight (kR 70000, kw 20000, kd_omega 200, massThrust 17700) / 5 ms / 0.0196 s; roll/pitch inertia 1.877e-4 kg m² (the configured value: the drone fit did not identify it). Identified law (fallback): 20 rad/s, ζ = 1 / 0.03 s / 0.015 s |
| Thrust map | firmware law: per motor T_i = κ/4 (pwm_i/massThrust)^p (1 + β (V − 7.2 V)), κ = 0.553 N^(1−p), p = 2.19, β = 0.249 V⁻¹ ([crazyflie_firmware.toml](../configs/controllers/crazyflie_firmware.toml)); identified law: T = 0.553 (1 + 0.177 (V − 7.2 V)) T_cmd^2.2 N |
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
([native/material_frame.inc](../native/material_frame.inc)). An optional saturating form
(`cable.Cb_saturation_rate_rad_s`, off when null, as in the default model) divides each joint's
coefficient by √(1 + (|dk/dt|/r_s)²), with the factors taken at the start of each step; fitted
on the calibration flights it improved the held-out whip flight by 3.5 mm (68.5 → 65.0 mm) for
one extra parameter and was not adopted. Air acts on each node through its
tributary length ℓ: a quadratic normal drag −½ρ C_d d ℓ |v_n| v_n on the cylinder (v_n is the
velocity normal to the local tangent; still air) plus a small linear term. A point clamp
attaches node 0 to the drone, prescribing its position and the body-down tangent; the first
edge may bend away from that tangent, and the clamp's bending moment acts on the drone. The
tip is free. There is no self-contact, ground or wall contact, or wind. Collapsed edges and
antiparallel adjacent edges are rejected as invalid states.

On the identified values the cable is gravity dominated (EI/(m g L²) ≈ 10⁻³) and its bending
relaxes in Cb/EI ≈ 0.3 s. The segment fit of the cable flights, jointly with the residual below,
identifies the four cable parameters (`runs/fits/cable/20261028-012011.json`). Without the residual
the same fit gives EI four times larger (5.7e−4): a stiffer cable was compensating for the lean the
residual now carries.

### Cable residual

The cable model has a compact, fitted correction (`cable.residual` in the physics file; absent or
all zero is the plain physics above): per node, a steady force per unit length that is a quadratic
in arc length per axis, in the drone's body frame or the world frame (`frame`); a multiplier
exp(b₀ + b₁ s̃) on the quadratic normal drag; and a quadratic drag along the tangent with
coefficient c₀ + c₁ s̃ (s̃ the arc fraction). It exists because the calibration flights show the
model's hanging equilibrium 2.3° off the measured hanging shape (40 mm at the tip), which is a force
the model lacks rather than a value of its parameters; it is fitted together with EI, Cb, Cd and the
linear drag on the cable flights (data and fitting page), and terms are kept only where the held-out
flights improve.

The adopted residual has only the steady force, in the drone's body frame: sideways near the clamp
(20 and 28 mN/m along body x and y), fading to about 2 mN/m by mid-cable, and a constant −9.5 ±
2.3 mN/m vertically, about 6 % of the cable's weight. A force concentrated just below the drone that
follows its attitude is what the rotor wake would produce; a clamp that holds the cable a few
degrees off vertical bends the hanging shape the same way, and the flights cannot separate the two
(fitting the clamp's angle instead needs 15° or more and does worse). The drag terms were not
identified on these flights and are zero. The net force of the residual on the cable (6 mN per
horizontal axis, 9 mN down) is cancelled by the training environment's hover trim, as the host's
trims cancel steady pulls in flight.

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
stopping when the largest relative velocity change of an iteration is below
`simulation.solver_tolerance`, length tolerance 1e−10 m, velocity-constraint
tolerance 1e−9 m/s and four position projections per iteration. Failure to
converge raises an error rather than accepting an approximate step. Solver
convergence does not by itself establish physical accuracy.

The stopping tolerance is 1e−5 for training, the simulator and the tools, and 1e−7 in the
model fits (`simulation.fit_solver_tolerance`, applied by the fitting replays), where the
least-squares finite differences should see the model rather than the stopping rule; a
configuration without the key uses 2e−11. It was chosen by
[`scripts/solver_tolerance.py`](../scripts/solver_tolerance.py)
(`runs/calibration_2026-10-27/solver_tolerance.json`): against 1e−13, on every 2 s window of
whip/002 and strokes/002 (recorded commands ×1 and ×1.5, open loop), 1e−5 differs by at most
0.0074 mm in drone position and 0.016 mm in marker position, 0.001 % in peak tip speed with no
shift of the peak, and on the 1,069 one-second cable segments of the cable fit by at most
0.0017 mm (RMS marker error against the recordings unchanged at 27.727 mm); 1e−7 by at most
0.005 mm. No replay failed. 1e−5 takes 2.4 iterations per step instead of 4.7 at 2e−11, about
half the physics time. These differences are far below those of the 2 ms step below.

The 2 ms step was chosen by a convergence study on the whip and strokes cable flights
(`scripts/step_convergence.py`, every 2 s window replayed open loop at 2, 1 and 0.5 ms,
with the recorded commands and with the commands scaled by 1.5 to drive the tip faster).
Against 0.5 ms, the 2 ms step differs by at most 2.1 mm in drone position and 2.8 mm in
marker position over 2 s (median 0.26 mm), and by at most 0.09 % in peak tip speed (peaks up
to 7.5 m/s) with the peak at the same or the adjacent 10 ms sample; 1 ms differs by about a
fifth of that, as expected of a second-order method. The remaining model error on these
flights is 40–74 mm.

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
- **Host tracking law** (`aerowhip.control.flight_pipeline`): in the interactive simulator,
  the trajectory editor, the headless runner and the Crazyswarm simulation server, the
  reference (position, velocity, acceleration, yaw) goes through the host PID of the cf_2
  flight scripts (k_p = 4 s⁻², k_d = 3 s⁻¹, k_i = 0.6 s⁻³, integral action limited to
  1.5 m/s² per axis), plus the same vertical hover feed-forward as the training environment;
  the resulting acceleration is the firmware law's setpoint. When a Crazyswarm script streams
  acceleration-only commands it runs its own host law, and its acceleration passes straight
  through.

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
