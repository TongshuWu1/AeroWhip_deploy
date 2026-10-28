#pragma once
// The flight estimator in native code: the OptiTrack frame clock, the live cable filter and the
// observer (drone velocity and body rate from the last poses, the cable from the marker cloud).
// One implementation runs in flight and inside the training environment. It follows the Python
// reference (aerowhip/deployment/cable_filter.py, observer.py) statement by statement; every
// comparison that decides a label uses the same floating-point operations in the same order.
// No ROS, no simulator, no Python.

#include <array>
#include <deque>
#include <vector>

namespace aerowhip {
namespace estimator {

constexpr int MARKERS = 10;
using P3 = std::array<double, 3>;
using Points = std::vector<P3>;

// Cloud points that may be cable markers: finite, farther than drone_radius from the tracking
// origin, duplicates (closer than duplicate_radius to a kept point) merged into the first.
Points cable_candidates(const Points& points, const P3& drone, double drone_radius=.07, double duplicate_radius=.01);

struct FrameClock {
    double dt = .01, time = 0.;
    bool started = false;
    explicit FrameClock(double step=.01) : dt(step) {}
    void reset() { started = false; time = 0.; }
    double advance(double stamp);
};

struct FilterSettings {
    double dt=.01, gate=.06, gate_growth=.5, noise=.004, tolerance=.015, start_tolerance=.02;
    int start_minimum=6; double max_hidden=.5; int iterations=3;
    double max_speed=12., reacquire=.05; int search_limit=4000; double prediction_weight=.25;
};

class CableFilter {
public:
    CableFilter(const std::vector<double>& arc, FilterSettings settings);
    void reset();
    void observe(double time, const P3& root, const Points& points);
    void project(const P3& root, std::array<P3, MARKERS>& positions) const;
    void override_labels(const P3& root, const Points& points, const std::array<int, MARKERS>& index);
    void update_quality();

    FilterSettings s;
    std::array<double, MARKERS> arc{}, segments{};
    std::array<P3, MARKERS> positions{}, velocities{};
    std::array<double, MARKERS> hidden{}, confidence{};
    std::array<int, MARKERS> index{};
    bool initialized = false, has_time = false, has_root = false;
    double time = 0., quality = 0.;
    P3 root{};
    FrameClock clock;
private:
    bool start(const P3& root, const Points& points);
    void fill_hidden(const P3& root);
    bool chain_ok(const P3& root, const std::array<P3, MARKERS>& positions) const;
    int worst_violation(const P3& root, const std::array<P3, MARKERS>& positions, const std::array<bool, MARKERS>& seen) const;
    std::vector<int> search(const P3& root, const Points& points, const std::array<P3, MARKERS>& predicted) const;
};

struct Estimate {
    double time = 0., stamp = 0.;
    P3 position{}, velocity{}, omega{}, root{}, root_velocity{};
    std::array<double, 9> rotation{};
    std::array<P3, MARKERS> cable_positions{}, cable_velocities{};
    std::array<double, MARKERS> confidence{};
    std::array<bool, MARKERS> labelled{};
    bool initialized = false;
};

class Observer {
public:
    Observer(const std::vector<double>& arc, const P3& attachment, double dt=.01, int history=5,
             FilterSettings settings=FilterSettings());
    void reset();
    Estimate observe(double stamp, const P3& position, const std::array<double, 9>& rotation, const Points& cloud);
    CableFilter filter;
    P3 attachment{};
    double dt = .01;
    int history = 5;
private:
    FrameClock clock;
    std::deque<double> times;
    std::deque<P3> positions;
    std::deque<std::array<double, 9>> rotations;
};

}  // namespace estimator
}  // namespace aerowhip
