#include "estimator.hpp"

#include <algorithm>
#include <functional>
#include <cmath>
#include <limits>
#include <numeric>
#include <stdexcept>

namespace aerowhip {
namespace estimator {
namespace {
constexpr double INF = std::numeric_limits<double>::infinity();
constexpr double NaN = std::numeric_limits<double>::quiet_NaN();
// NumPy's norm over the last axis of a (N, 3) array: sqrt((x*x + y*y) + z*z).
double norm(const P3& a) { return std::sqrt((a[0]*a[0]+a[1]*a[1])+a[2]*a[2]); }
P3 sub(const P3& a, const P3& b) { return {a[0]-b[0], a[1]-b[1], a[2]-b[2]}; }
double distance(const P3& a, const P3& b) { return norm(sub(a, b)); }
bool finite(const P3& a) { return std::isfinite(a[0]) && std::isfinite(a[1]) && std::isfinite(a[2]); }
}  // namespace

Points cable_candidates(const Points& points, const P3& drone, double drone_radius, double duplicate_radius) {
    Points kept;
    for (const P3& q : points) {
        if (!finite(q) || !(distance(q, drone) > drone_radius)) continue;
        double nearest = INF;
        for (const P3& k : kept) nearest = std::min(nearest, distance(k, q));
        if (kept.empty() || nearest > duplicate_radius) kept.push_back(q);
    }
    return kept;
}

double FrameClock::advance(double stamp) {
    if (!started) { started = true; time = stamp; return dt; }
    // Python round(): ties to even, as std::nearbyint in the default rounding mode.
    double step = dt*std::max(1., std::nearbyint((stamp-time)/dt));
    time += step;
    time += .05*(stamp-time);
    return step;
}

CableFilter::CableFilter(const std::vector<double>& arc_values, FilterSettings settings) : s(settings), clock(settings.dt) {
    if (arc_values.size() != MARKERS) throw std::invalid_argument("The cable filter tracks ten markers");
    for (int k = 0; k < MARKERS; ++k) {
        arc[k] = arc_values[k];
        segments[k] = arc[k]-(k ? arc[k-1] : 0.);
    }
    reset();
}

void CableFilter::reset() {
    for (auto& p : positions) p = {NaN, NaN, NaN};
    for (auto& v : velocities) v = {0., 0., 0.};
    hidden.fill(INF);
    confidence.fill(0.);
    initialized = has_time = has_root = false;
    time = 0.;
    clock = FrameClock(s.dt);
    root = {0., 0., 0.};
    quality = 0.;
    index.fill(-1);
}

bool CableFilter::start(const P3& r, const Points& points) {
    if (points.empty()) return false;
    std::array<int, MARKERS> found_index; found_index.fill(-1);
    std::vector<double> dist(points.size());
    for (size_t j = 0; j < points.size(); ++j) dist[j] = distance(points[j], r);
    for (int k = 0; k < MARKERS; ++k) {
        int count = 0, candidate = -1;
        for (size_t j = 0; j < points.size(); ++j)
            if (std::abs(dist[j]-arc[k]) < s.start_tolerance && points[j][2]-r[2] < 0) { ++count; if (candidate < 0) candidate = int(j); }
        if (count > 1) return false;
        if (count) found_index[k] = candidate;
    }
    std::vector<int> found;
    for (int k = 0; k < MARKERS; ++k) if (found_index[k] >= 0) found.push_back(k);
    std::vector<int> used;
    for (int k : found) used.push_back(found_index[k]);
    std::sort(used.begin(), used.end());
    if (int(found.size()) < s.start_minimum || std::adjacent_find(used.begin(), used.end()) != used.end()) return false;
    for (auto& p : positions) p = {NaN, NaN, NaN};
    for (int k : found) positions[k] = points[found_index[k]];
    fill_hidden(r);
    if (!chain_ok(r, positions)) { for (auto& p : positions) p = {NaN, NaN, NaN}; return false; }
    for (auto& v : velocities) v = {0., 0., 0.};
    hidden.fill(INF);
    for (int k : found) hidden[k] = 0.;
    index = found_index;
    initialized = true;
    return true;
}

void CableFilter::observe(double t, const P3& r, const Points& points) {
    const double dt = clock.advance(t);
    time = t; has_time = true; root = r; has_root = true;
    if (!initialized) {
        if (start(r, points)) update_quality();
        return;
    }
    const size_t n = points.size();
    // 1. predict
    std::array<P3, MARKERS> predicted;
    for (int k = 0; k < MARKERS; ++k)
        for (int i = 0; i < 3; ++i) predicted[k][i] = positions[k][i]+velocities[k][i]*dt;
    project(r, predicted);
    // 2. associate: nearest point within the gate, best matches first
    std::array<double, MARKERS> gates;
    for (int k = 0; k < MARKERS; ++k) gates[k] = s.gate+s.gate_growth*std::min(hidden[k], s.max_hidden);
    std::array<int, MARKERS> idx; idx.fill(-1);
    std::vector<double> d(MARKERS*n);
    for (int k = 0; k < MARKERS; ++k) for (size_t j = 0; j < n; ++j) d[k*n+j] = distance(predicted[k], points[j]);
    if (n) {
        std::vector<int> order(MARKERS*n);
        std::iota(order.begin(), order.end(), 0);
        std::stable_sort(order.begin(), order.end(), [&](int a, int b) { return d[a] < d[b]; });
        std::vector<bool> taken(n, false);
        std::array<bool, MARKERS> done{};
        for (int flat : order) {
            int k = flat/int(n), j = flat%int(n);
            if (d[flat] > gates[k]) break;
            if (done[k] || taken[j]) continue;
            idx[k] = j; done[k] = true; taken[j] = true;
        }
    }
    // 3. chain check: drop assignments inconsistent with the cable lengths, worst first
    std::array<P3, MARKERS> measured = predicted;
    std::array<bool, MARKERS> seen{};
    for (int k = 0; k < MARKERS; ++k) { seen[k] = idx[k] >= 0; if (seen[k]) measured[k] = points[idx[k]]; }
    for (;;) {
        int bad = worst_violation(r, measured, seen);
        if (bad < 0) break;
        idx[bad] = -1; seen[bad] = false; measured[bad] = predicted[bad];
    }
    bool any_seen = false, any_unseen = false;
    for (bool v : seen) (v ? any_seen : any_unseen) = true;
    // 3b. re-acquire markers hidden for long
    if (any_unseen && any_seen) {
        std::vector<bool> taken(n, false);
        for (int k = 0; k < MARKERS; ++k) if (seen[k]) taken[idx[k]] = true;
        std::array<P3, MARKERS+1> chain; chain[0] = r;
        for (int k = 0; k < MARKERS; ++k) chain[k+1] = measured[k];
        std::array<double, MARKERS+1> arc0; arc0[0] = 0.;
        for (int k = 0; k < MARKERS; ++k) arc0[k+1] = arc[k];
        std::vector<int> known{0};
        for (int k = 0; k < MARKERS; ++k) if (seen[k]) known.push_back(k+1);
        std::vector<int> candidates_k;
        for (int k = 0; k < MARKERS; ++k) if (!seen[k] && hidden[k] >= s.max_hidden) candidates_k.push_back(k);
        for (int k : candidates_k) {
            std::vector<int> free;
            for (size_t j = 0; j < n; ++j) if (!taken[j]) free.push_back(int(j));
            if (free.empty()) break;
            int a = -1, b = -1;
            for (int x : known) { if (x < k+1) a = x; else if (x > k+1 && b < 0) b = x; }
            int count = 0, j = -1;
            for (int f : free) {
                bool ok = std::abs(distance(points[f], chain[a])-(arc0[k+1]-arc0[a])) < s.tolerance;
                if (b >= 0) ok = ok && std::abs(distance(points[f], chain[b])-(arc0[b]-arc0[k+1])) < s.tolerance;
                if (ok) { ++count; j = f; }
            }
            if (count != 1) continue;
            if (distance(points[j], predicted[k]) > s.reacquire+s.gate_growth*s.max_hidden) continue;
            idx[k] = j; seen[k] = true; taken[j] = true;
            measured[k] = points[j];
            chain[k+1] = points[j];
            velocities[k] = {0., 0., 0.};
        }
    }
    // 3c. chain search: markers still missing and free points left -> re-label from the root
    std::array<bool, MARKERS> jumped{};
    int seen_count = 0;
    any_unseen = false;
    for (bool v : seen) { seen_count += v; if (!v) any_unseen = true; }
    if (any_unseen && int(n) > seen_count) {
        std::vector<int> chain = search(r, points, predicted);
        if (!chain.empty()) {
            int prefix_seen = 0;
            for (size_t k = 0; k < chain.size(); ++k) prefix_seen += seen[k];
            // np.flatnonzero(np.r_[seen, True])[0]: the first seen marker (MARKERS if none)
            int first_seen = MARKERS;
            for (int k = 0; k < MARKERS; ++k) if (seen[k]) { first_seen = k; break; }
            if (int(chain.size()) > prefix_seen && int(chain.size()) >= first_seen) {
                for (size_t k = 0; k < chain.size(); ++k) {
                    int j = chain[k];
                    if (idx[k] != j) jumped[k] = distance(points[j], predicted[k]) > gates[k];
                    idx[k] = j; seen[k] = true;
                    measured[k] = points[j];
                }
                for (int k = int(chain.size()); k < MARKERS; ++k)
                    if (seen[k] && std::find(chain.begin(), chain.end(), idx[k]) != chain.end()) {
                        idx[k] = -1; seen[k] = false; measured[k] = predicted[k];
                    }
                for (;;) {
                    int bad = worst_violation(r, measured, seen);
                    if (bad < 0) break;
                    idx[bad] = -1; seen[bad] = false; measured[bad] = predicted[bad];
                }
            }
        }
    }
    // 4. correct
    std::array<P3, MARKERS> next = predicted;
    for (int k = 0; k < MARKERS; ++k) if (seen[k]) next[k] = measured[k];
    project(r, next);
    for (int k = 0; k < MARKERS; ++k) {
        P3 velocity;
        for (int i = 0; i < 3; ++i) velocity[i] = (next[k][i]-positions[k][i])/dt;
        double speed = norm(velocity);
        double factor = std::min(1., s.max_speed/std::max(speed, 1e-9));
        double alpha = seen[k] ? .5 : .1;
        for (int i = 0; i < 3; ++i) {
            velocity[i] = velocity[i]*factor;
            velocities[k][i] = velocities[k][i]+alpha*(velocity[i]-velocities[k][i]);
        }
        if (jumped[k]) velocities[k] = {0., 0., 0.};
    }
    positions = next;
    for (int k = 0; k < MARKERS; ++k) hidden[k] = seen[k] ? 0. : hidden[k]+dt;
    index = idx;
    update_quality();
}

std::vector<int> CableFilter::search(const P3& r, const Points& points, const std::array<P3, MARKERS>& predicted) const {
    const size_t n = points.size();
    std::vector<double> distance_root(n);
    std::vector<bool> reach(n);
    for (size_t j = 0; j < n; ++j) {
        distance_root[j] = distance(points[j], r);
        reach[j] = distance_root[j] <= arc[MARKERS-1]+s.tolerance;
    }
    struct Best { std::vector<int> chain; bool has = false; double cost = INF; int depth = 0; } best;
    struct Runner { std::vector<int> chain; bool has = false; double cost = INF; } runner;
    int budget = s.search_limit;
    std::vector<int> used;
    std::vector<double> error(n);
    std::function<void(const P3&, double)> go = [&](const P3& previous, double cost) {
        int depth = int(used.size());
        if (depth > best.depth || (depth == best.depth && cost < best.cost)) {
            if (depth == best.depth && best.has) { runner.chain = best.chain; runner.has = true; runner.cost = best.cost; }
            else { runner.has = false; runner.chain.clear(); runner.cost = INF; }
            best.chain = used; best.has = true; best.cost = cost; best.depth = depth;
        } else if (depth == best.depth && cost < runner.cost) {
            runner.chain = used; runner.has = true; runner.cost = cost;
        }
        if (depth == MARKERS || budget <= 0) return;
        --budget;
        std::vector<double> err(n);
        std::vector<int> ok;
        for (size_t j = 0; j < n; ++j) {
            err[j] = std::abs(distance(points[j], previous)-segments[depth]);
            bool good = err[j] < s.tolerance && reach[j] && distance_root[j] <= arc[depth]+s.tolerance;
            if (depth == 0) good = good && points[j][2] < r[2];
            if (good) ok.push_back(int(j));
        }
        for (int j : ok) {
            if (std::find(used.begin(), used.end(), j) != used.end()) continue;
            double miss = distance(points[j], predicted[depth]);
            double m = std::min(miss, .2);
            used.push_back(j);
            go(points[j], (cost+err[j]*err[j])+s.prediction_weight*(m*m)/100.);
            used.pop_back();
        }
    };
    go(r, 0.);
    if (!best.has || best.chain.empty()) return {};
    std::vector<int> chain = best.chain;
    if (runner.has && runner.chain.size() == chain.size() && runner.cost < 1.5*best.cost+1e-6) {
        size_t agree = 0;
        while (agree < chain.size() && chain[agree] == runner.chain[agree]) ++agree;
        chain.resize(agree);
    }
    return chain;
}

void CableFilter::fill_hidden(const P3& r) {
    std::array<P3, MARKERS+1> chain; chain[0] = r;
    for (int k = 0; k < MARKERS; ++k) chain[k+1] = positions[k];
    std::array<double, MARKERS+1> arc0; arc0[0] = 0.;
    for (int k = 0; k < MARKERS; ++k) arc0[k+1] = arc[k];
    std::vector<int> known;
    for (int k = 0; k <= MARKERS; ++k) if (finite(chain[k])) known.push_back(k);
    for (int k = 1; k <= MARKERS; ++k) {
        if (finite(chain[k])) continue;
        std::vector<int> before, after;
        for (int x : known) { if (x < k) before.push_back(x); else if (x > k) after.push_back(x); }
        int a = before.back();
        if (!after.empty()) {
            int b = after.front();
            for (int i = 0; i < 3; ++i) chain[k][i] = chain[a][i]+(chain[b][i]-chain[a][i])*(arc0[k]-arc0[a])/(arc0[b]-arc0[a]);
        } else {
            P3 direction{0., 0., -1.};
            if (before.size() > 1) direction = sub(chain[a], chain[before[before.size()-2]]);
            double length = std::max(norm(direction), 1e-9);
            for (int i = 0; i < 3; ++i) chain[k][i] = chain[a][i]+(arc0[k]-arc0[a])*direction[i]/length;
        }
    }
    for (int k = 0; k < MARKERS; ++k) positions[k] = chain[k+1];
}

void CableFilter::project(const P3& r, std::array<P3, MARKERS>& p) const {
    std::array<P3, MARKERS+1> chain; chain[0] = r;
    for (int k = 0; k < MARKERS; ++k) chain[k+1] = p[k];
    for (int iteration = 0; iteration < s.iterations; ++iteration)
        for (int k = 1; k <= MARKERS; ++k) {
            const P3 a = chain[k-1], b = chain[k];
            double gx = b[0]-a[0], gy = b[1]-a[1], gz = b[2]-a[2];
            double length = std::sqrt(gx*gx+gy*gy+gz*gz);
            double limit = segments[k-1];
            if (length > limit) {
                double scale = limit/length;
                chain[k] = {a[0]+gx*scale, a[1]+gy*scale, a[2]+gz*scale};
            }
        }
    for (int k = 0; k < MARKERS; ++k) p[k] = chain[k+1];
}

bool CableFilter::chain_ok(const P3& r, const std::array<P3, MARKERS>& p) const {
    P3 previous = r;
    for (int k = 0; k < MARKERS; ++k) {
        if (!(distance(p[k], previous) <= segments[k]+s.tolerance)) return false;
        previous = p[k];
    }
    return true;
}

int CableFilter::worst_violation(const P3& r, const std::array<P3, MARKERS>& p, const std::array<bool, MARKERS>& seen) const {
    int worst = -1; double value = 0.;
    P3 previous = r;
    for (int i = 0; i < MARKERS; ++i) {      // link i joins chain[i] and chain[i+1]: markers i-1 and i
        double excess = (distance(p[i], previous)-segments[i])-s.tolerance;
        previous = p[i];
        if (!(excess > 0)) continue;
        for (int k : {i-1, i})
            if (0 <= k && k < MARKERS && seen[k] && excess > value) { worst = k; value = excess; }
    }
    return worst;
}

void CableFilter::update_quality() {
    for (int k = 0; k < MARKERS; ++k) confidence[k] = std::min(1., std::max(0., 1.-hidden[k]/s.max_hidden));
    if (!initialized) { quality = 0.; return; }
    // NumPy's pairwise sum of ten values: eight in a block, then the rest in order.
    const auto& c = confidence;
    double sum = ((c[0]+c[1])+(c[2]+c[3]))+((c[4]+c[5])+(c[6]+c[7]));
    sum += c[8]; sum += c[9];
    quality = sum/MARKERS;
}

void CableFilter::override_labels(const P3& r, const Points& points, const std::array<int, MARKERS>& labels) {
    for (int k = 0; k < MARKERS; ++k) if (labels[k] >= 0) positions[k] = points[labels[k]];
    project(r, positions);
    for (int k = 0; k < MARKERS; ++k) {
        if (labels[k] != index[k]) velocities[k] = {0., 0., 0.};
        if (labels[k] >= 0) hidden[k] = 0.;
    }
    index = labels;
    initialized = true;
    update_quality();
}

namespace {
using M3 = std::array<double, 9>;
// R^T * Q for row-major 3x3 matrices.
M3 transposed_product(const M3& R, const M3& Q) {
    M3 out{};
    for (int i = 0; i < 3; ++i) for (int j = 0; j < 3; ++j)
        out[3*i+j] = (R[0*3+i]*Q[0*3+j]+R[1*3+i]*Q[1*3+j])+R[2*3+i]*Q[2*3+j];
    return out;
}
P3 rotation_log(const M3& R) {
    double angle = std::acos(std::min(1., std::max(-1., (((R[0]+R[4])+R[8])-1)/2)));
    P3 v{(R[7]-R[5])/2, (R[2]-R[6])/2, (R[3]-R[1])/2};
    if (angle < 1e-9) return v;
    double f = angle/std::sin(angle);
    return {v[0]*f, v[1]*f, v[2]*f};
}
// Derivative at the newest sample of a least-squares quadratic through the samples (a line for
// two; zero for one). Solved on time in frame units (well conditioned), then rescaled.
P3 endpoint_derivative(const std::deque<double>& time, const std::vector<P3>& values, double unit) {
    const size_t n = time.size();
    if (n < 2) return {0., 0., 0.};
    const double t0 = time.back();
    if (n == 2) {
        double du = time[0]-t0;
        P3 out;
        for (int i = 0; i < 3; ++i) out[i] = (values[0][i]-values[1][i])/du;
        return out;
    }
    // Normal equations of [1, x, x^2] with x = (t - t_last)/unit.
    double A[3][3] = {}, B[3][3] = {};
    for (size_t k = 0; k < n; ++k) {
        double x = (time[k]-t0)/unit, basis[3] = {1., x, x*x};
        for (int r = 0; r < 3; ++r) {
            for (int c = 0; c < 3; ++c) A[r][c] += basis[r]*basis[c];
            for (int i = 0; i < 3; ++i) B[r][i] += basis[r]*(values[k][i]-values[n-1][i]);
        }
    }
    // Gaussian elimination with partial pivoting.
    int perm[3] = {0, 1, 2};
    for (int c = 0; c < 3; ++c) {
        int p = c;
        for (int r = c+1; r < 3; ++r) if (std::abs(A[r][c]) > std::abs(A[p][c])) p = r;
        std::swap(A[c], A[p]); std::swap(B[c], B[p]); std::swap(perm[c], perm[p]);
        for (int r = c+1; r < 3; ++r) {
            double f = A[r][c]/A[c][c];
            for (int k = c; k < 3; ++k) A[r][k] -= f*A[c][k];
            for (int i = 0; i < 3; ++i) B[r][i] -= f*B[c][i];
        }
    }
    double X[3][3];
    for (int i = 0; i < 3; ++i)
        for (int r = 2; r >= 0; --r) {
            double v = B[r][i];
            for (int k = r+1; k < 3; ++k) v -= A[r][k]*X[k][i];
            X[r][i] = v/A[r][r];
        }
    return {X[1][0]/unit, X[1][1]/unit, X[1][2]/unit};
}
}  // namespace

Observer::Observer(const std::vector<double>& arc, const P3& attach, double step, int frames, FilterSettings settings)
    : filter(arc, [&] { settings.dt = step; return settings; }()), attachment(attach), dt(step), history(frames), clock(step) {
    if (history < 1) throw std::invalid_argument("The observer needs at least one frame of history");
}

void Observer::reset() {
    filter.reset();
    clock.reset();
    times.clear(); positions.clear(); rotations.clear();
}

Estimate Observer::observe(double stamp, const P3& p, const M3& R, const Points& cloud) {
    Estimate e;
    double step = clock.advance(stamp);
    double frame_time = times.empty() ? 0. : times.back()+step;
    times.push_back(frame_time); positions.push_back(p); rotations.push_back(R);
    while (int(times.size()) > history) { times.pop_front(); positions.pop_front(); rotations.pop_front(); }
    std::vector<P3> pos(positions.begin(), positions.end()), logs;
    for (const auto& Ri : rotations) logs.push_back(rotation_log(transposed_product(R, Ri)));
    e.velocity = endpoint_derivative(times, pos, dt);
    e.omega = endpoint_derivative(times, logs, dt);
    P3 arm{}, spin{};
    for (int i = 0; i < 3; ++i) arm[i] = (R[3*i]*attachment[0]+R[3*i+1]*attachment[1])+R[3*i+2]*attachment[2];
    P3 w = e.omega, a = attachment;
    P3 c{w[1]*a[2]-w[2]*a[1], w[2]*a[0]-w[0]*a[2], w[0]*a[1]-w[1]*a[0]};
    for (int i = 0; i < 3; ++i) spin[i] = (R[3*i]*c[0]+R[3*i+1]*c[1])+R[3*i+2]*c[2];
    for (int i = 0; i < 3; ++i) { e.root[i] = p[i]+arm[i]; e.root_velocity[i] = e.velocity[i]+spin[i]; }
    filter.observe(frame_time, e.root, cable_candidates(cloud, p));
    bool usable = filter.initialized;
    for (const auto& q : filter.positions) usable = usable && finite(q);
    for (int k = 0; k < MARKERS; ++k) {
        e.cable_positions[k] = usable ? filter.positions[k] : e.root;
        e.cable_velocities[k] = usable ? filter.velocities[k] : P3{0., 0., 0.};
        e.confidence[k] = usable ? filter.confidence[k] : 0.;
        e.labelled[k] = filter.index[k] >= 0;
    }
    e.time = frame_time; e.stamp = stamp; e.position = p; e.rotation = R;
    e.initialized = filter.initialized;
    return e;
}

}  // namespace estimator
}  // namespace aerowhip
