#pragma once
#include "engine.hpp"

namespace aerowhip {
// Four parallel body +Z rotors. Signed yaw coefficients describe torque ON
// THE BODY per unit thrust, not visual spin directions or firmware motor IDs.
class Actuator {
public:
    Actuator(const std::array<Vec3,4>& positions, Motors yaw_per_thrust,
             Motors minimum, Motors maximum, Motors response_s, Motors initial);
    Motors allocate(const Wrench& requested) const; // inverse mixing, before limits
    Motors target(const Wrench& requested) const;   // independent motor clipping
    Motors response(const Motors& initial, const Motors& target, double elapsed) const;
    Wrench wrench(const Motors& forces) const;
    void validate(const Motors& forces) const;
    const std::array<double,16>& mixing() const { return mixing_; }
    const std::array<double,16>& inverse() const { return inverse_; }
    Motors minimum, maximum, response_s, initial;
private:
    std::array<double,16> mixing_{}, inverse_{};
};
}
