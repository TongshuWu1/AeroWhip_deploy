#pragma once

#include <array>
#include <atomic>
#include <functional>
#include <limits>
#include <vector>

namespace aerowhip {
using Vector = std::vector<double>;
using Vec3 = std::array<double, 3>;
using Mat3 = std::array<double, 9>; // row major, body to world
using Wrench = std::array<double, 4>; // collective N (body +Z), body XYZ Nm
using Motors = std::array<double, 4>; // individual thrust N, not PWM or RPM
class Actuator;

struct Layer {
    int inputs = 0, outputs = 0;
    Vector weight, bias; // output-major weights
};
struct AerodynamicResidual {
    std::vector<Layer> cable, drone;
    // Rates have units s^-1; air is stationary in the world frame.
    double cable_log_bound = 0., drone_max_rate = 0., velocity_scale = 1.;
    bool cable_enabled = false, drone_enabled = false;
};
struct Parameters {
    double mass = 0., stiffness = 0., drag = 0., maximum_step = 0.;
    double bending_damping = 0.;
    // Bend-angle rate (rad/s) at which the bending damping saturates; infinite = linear Kelvin-Voigt.
    double bending_damping_saturation = std::numeric_limits<double>::infinity();
    // Air: linear drag rate (s^-1, mass proportional) plus quadratic normal drag on the
    // cable cylinder, 0.5*rho*Cd*diameter per unit length, with still air.
    double air_density = 0., normal_drag_coefficient = 0., diameter = 0.;
    int projections = 0;
    // Drone air drag: linear rates (s^-1, mass proportional) per body axis on the COM velocity
    // relative to still air (rotor and body drag), F = -m R diag(k) R^T v.
    Vec3 body_drag{};
    Mat3 inertia{};
    Vec3 arm{}, gravity{}, root_tangent_body{};
    Vector rest, node_mass;
    AerodynamicResidual residual;
};

struct ModelInput {
    Parameters parameters;
    Vec3 tracking_to_com{}, tracking_to_attachment{};
    Vector intervals, subdivisions, marker_mass;
    double total_cable_mass=0., output_step=0., delay=0.;
};
struct Model {
    explicit Model(ModelInput input);
    Parameters parameters;
    Vec3 com{}, attachment{};
    Vector intervals;
    std::vector<int> marker_indices;
    double output_step=0., delay=0.;
    Vector initial_state(Vec3 tracking, Mat3 rotation, Vec3 tracking_velocity, Vec3 omega,
                         const Vector& cable_positions={},const Vector& cable_velocities={}) const;
    // Tracking P/V, rotation, body omega, all cable P/V, provided tracking A.
    Vector observation(const Vector& state, Vec3 acceleration={}) const;
};

// State v3: COM p(3), R(9), COM v(3), body omega(3), free q(3M), free v(3M). Twist-free:
// the material frame is parallel transported from the clamp (no material angles).
// Integrator checkpoints are physical state, previous physical state, previous step (0 at reset).
Vector prescribed_advance(const Model& model,Vector state,const Vector& samples,Vector& previous,double& previous_step);
Vector prescribed_acceleration(const Model& model,const Vector& state,const Vector& boundary);
// All positions/linear velocities are world-frame SI. Node 0 is eliminated.
// No controller, renderer, Python, file format, or firmware dependency here.
class Engine {
public:
    explicit Engine(Parameters parameters);
    void reset(const Vector& state);
    Vector state() const;
    Vector checkpoint() const;
    void restore(const Vector& checkpoint);
    Vector acceleration(const Wrench& wrench) const;
    Vec3 drone_air_force() const;
    void advance(double duration, const Wrench& wrench);
    void advance_controlled(double duration, const std::function<Wrench(const Vector&)>& controller);
    Motors advance_motors(double duration, const Wrench& requested, const Actuator& actuator, const Motors& motors);
    int node_count() const { return static_cast<int>(p_.node_mass.size()); }
private:
    Parameters p_;
    Mat3 root_inverse_{};
    double root_mass_ = 0.;
    Vector state_, previous_state_;
    double previous_step_=0.;
    mutable std::atomic_flag busy_ = ATOMIC_FLAG_INIT;
    void validate_state(const Vector& state, bool constraints) const;
    Vector acceleration_at(const Vector& state, const Wrench& wrench) const;
    Vector generalized_load(const Vector& state, const Wrench& wrench) const;
    Vector integrate(const Vector& state, const Vector& previous, double previous_step, double h,
                     const std::function<Wrench(const Vector&,double)>& controller, double start) const;
    void advance_impl(double duration, const std::function<Wrench(const Vector&,double)>& controller);
};
} // namespace aerowhip
