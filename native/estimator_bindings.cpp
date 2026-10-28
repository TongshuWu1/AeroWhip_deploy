// Python bindings of the native flight estimator (native/estimator.hpp): the same dictionaries
// and arrays as the Python reference in aerowhip/deployment (cable_filter.py, observer.py).
#include "estimator.hpp"

#include <cmath>
#include <limits>
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

namespace py = pybind11;
using namespace aerowhip::estimator;
using Array = py::array_t<double, py::array::c_style | py::array::forcecast>;
using IntArray = py::array_t<long, py::array::c_style | py::array::forcecast>;

namespace {
Points points_of(const Array& a) {
    if (a.size() % 3) throw py::value_error("Expected points of three coordinates");
    Points out(a.size()/3);
    const double* v = a.data();
    for (size_t i = 0; i < out.size(); ++i) out[i] = {v[3*i], v[3*i+1], v[3*i+2]};
    return out;
}
P3 p3(const Array& a) {
    if (a.size() != 3) throw py::value_error("Expected three coordinates");
    return {a.data()[0], a.data()[1], a.data()[2]};
}
std::array<double, 9> m3(const Array& a) {
    if (a.size() != 9) throw py::value_error("Expected a 3 x 3 rotation");
    std::array<double, 9> out; std::copy_n(a.data(), 9, out.begin()); return out;
}
Array points_array(const Points& p) {
    Array out({py::ssize_t(p.size()), py::ssize_t(3)});
    double* v = out.mutable_data();
    for (size_t i = 0; i < p.size(); ++i) for (int k = 0; k < 3; ++k) v[3*i+k] = p[i][k];
    return out;
}
Array markers_array(const std::array<P3, MARKERS>& p) {
    Array out({py::ssize_t(MARKERS), py::ssize_t(3)});
    double* v = out.mutable_data();
    for (int i = 0; i < MARKERS; ++i) for (int k = 0; k < 3; ++k) v[3*i+k] = p[i][k];
    return out;
}
std::array<P3, MARKERS> markers_of(const Array& a) {
    if (a.size() != 3*MARKERS) throw py::value_error("Expected ten markers");
    std::array<P3, MARKERS> out;
    for (int i = 0; i < MARKERS; ++i) out[i] = {a.data()[3*i], a.data()[3*i+1], a.data()[3*i+2]};
    return out;
}
template<class T> py::array_t<T> vector_array(const std::array<T, MARKERS>& v) {
    py::array_t<T> out(MARKERS); std::copy(v.begin(), v.end(), out.mutable_data()); return out;
}
Array p3_array(const P3& p) { Array out(3); std::copy(p.begin(), p.end(), out.mutable_data()); return out; }
py::dict filter_state(const CableFilter& f) {
    py::dict d;
    d["time_s"] = f.has_time ? py::object(py::float_(f.time)) : py::object(py::none());
    d["initialized"] = f.initialized;
    d["positions"] = markers_array(f.positions);
    d["velocities"] = markers_array(f.velocities);
    d["confidence"] = vector_array(f.confidence);
    d["quality"] = f.quality;
    d["index"] = vector_array<long>([&] { std::array<long, MARKERS> i; for (int k = 0; k < MARKERS; ++k) i[k] = f.index[k]; return i; }());
    return d;
}
}  // namespace

void bind_estimator(py::module_& m) {
    m.def("cable_candidates", [](Array points, Array drone, double drone_radius, double duplicate_radius) {
        return points_array(cable_candidates(points_of(points), p3(drone), drone_radius, duplicate_radius));
    }, py::arg("points"), py::arg("drone"), py::arg("drone_radius") = .07, py::arg("duplicate_radius") = .01);

    py::class_<FrameClock>(m, "FrameClock")
        .def(py::init<double>(), py::arg("dt") = .01)
        .def("reset", &FrameClock::reset)
        .def("advance", &FrameClock::advance)
        .def_readonly("dt", &FrameClock::dt)
        .def_property_readonly("time", [](const FrameClock& c) {
            return c.started ? py::object(py::float_(c.time)) : py::object(py::none()); });

    py::class_<CableFilter>(m, "CableFilter")
        .def(py::init([](std::vector<double> arc, double dt, double gate, double gate_growth, double noise, double tolerance,
                         double start_tolerance, int start_minimum, double max_hidden, int iterations, double max_speed,
                         double reacquire, int search_limit, double prediction_weight) {
            FilterSettings s{dt, gate, gate_growth, noise, tolerance, start_tolerance, start_minimum, max_hidden, iterations,
                             max_speed, reacquire, search_limit, prediction_weight};
            return new CableFilter(arc, s);
        }), py::arg("arc"), py::arg("dt") = .01, py::arg("gate_m") = .06, py::arg("gate_growth_m") = .5, py::arg("noise_m") = .004,
            py::arg("length_tolerance_m") = .015, py::arg("start_tolerance_m") = .02, py::arg("start_minimum") = 6,
            py::arg("max_hidden_s") = .5, py::arg("projection_iterations") = 3, py::arg("max_speed_mps") = 12.,
            py::arg("reacquire_m") = .05, py::arg("search_limit") = 4000, py::arg("prediction_weight") = .25)
        .def("reset", &CableFilter::reset)
        .def("observe", [](CableFilter& f, double time, Array root, Array points) {
            f.observe(time, p3(root), points_of(points)); return filter_state(f); })
        .def("state", &filter_state)
        .def("project", [](const CableFilter& f, Array root, Array positions) {
            auto p = markers_of(positions); f.project(p3(root), p); return markers_array(p); })
        .def("override", [](CableFilter& f, Array root, Array points, IntArray index) {
            if (index.size() != MARKERS) throw py::value_error("Expected ten labels");
            std::array<int, MARKERS> labels; for (int k = 0; k < MARKERS; ++k) labels[k] = int(index.data()[k]);
            f.override_labels(p3(root), points_of(points), labels); })
        .def("get_state", [](const CableFilter& f) {
            py::dict d;
            d["positions"] = markers_array(f.positions); d["velocities"] = markers_array(f.velocities);
            d["hidden_s"] = vector_array(f.hidden); d["initialized"] = f.initialized;
            d["time"] = f.has_time ? py::object(py::float_(f.time)) : py::object(py::none());
            d["clock"] = f.clock.started ? py::object(py::float_(f.clock.time)) : py::object(py::none());
            d["root"] = f.has_root ? py::object(p3_array(f.root)) : py::object(py::none());
            std::array<long, MARKERS> i; for (int k = 0; k < MARKERS; ++k) i[k] = f.index[k];
            d["index"] = vector_array<long>(i);
            return d; })
        .def("set_state", [](CableFilter& f, py::dict d) {
            f.positions = markers_of(d["positions"].cast<Array>()); f.velocities = markers_of(d["velocities"].cast<Array>());
            auto hidden = d["hidden_s"].cast<Array>(); std::copy_n(hidden.data(), MARKERS, f.hidden.begin());
            f.initialized = d["initialized"].cast<bool>();
            f.has_time = !d["time"].is_none(); f.time = f.has_time ? d["time"].cast<double>() : 0.;
            f.clock.started = !d["clock"].is_none(); f.clock.time = f.clock.started ? d["clock"].cast<double>() : 0.;
            f.has_root = !d["root"].is_none(); if (f.has_root) f.root = p3(d["root"].cast<Array>());
            auto index = d["index"].cast<IntArray>(); for (int k = 0; k < MARKERS; ++k) f.index[k] = int(index.data()[k]);
            f.update_quality(); })
        .def_property_readonly("segments", [](const CableFilter& f) { return vector_array(f.segments); })
        .def_property_readonly("arc", [](const CableFilter& f) { return vector_array(f.arc); })
        .def_property_readonly("iterations", [](const CableFilter& f) { return f.s.iterations; })
        .def_property_readonly("initialized", [](const CableFilter& f) { return f.initialized; })
        .def_property_readonly("quality", [](const CableFilter& f) { return f.quality; })
        .def_property_readonly("positions", [](const CableFilter& f) { return markers_array(f.positions); })
        .def_property_readonly("velocities", [](const CableFilter& f) { return markers_array(f.velocities); })
        .def_property_readonly("confidence", [](const CableFilter& f) { return vector_array(f.confidence); })
        .def_property_readonly("index", [](const CableFilter& f) {
            std::array<long, MARKERS> i; for (int k = 0; k < MARKERS; ++k) i[k] = f.index[k]; return vector_array<long>(i); });

    py::class_<Observer>(m, "Observer")
        .def(py::init([](std::vector<double> arc, Array attachment, double dt, int history) {
            return new Observer(arc, p3(attachment), dt, history); }),
            py::arg("arc"), py::arg("attachment_body"), py::arg("dt") = .01, py::arg("history") = 5)
        .def("reset", &Observer::reset)
        .def("observe", [](Observer& o, double stamp, Array position, Array rotation, Array cloud) {
            Estimate e = o.observe(stamp, p3(position), m3(rotation), points_of(cloud));
            py::dict d;
            d["time_s"] = e.time; d["stamp"] = e.stamp;
            d["position"] = p3_array(e.position);
            Array R({3, 3}); std::copy(e.rotation.begin(), e.rotation.end(), R.mutable_data()); d["rotation"] = R;
            d["velocity"] = p3_array(e.velocity); d["angular_velocity"] = p3_array(e.omega);
            d["root"] = p3_array(e.root); d["root_velocity"] = p3_array(e.root_velocity);
            d["cable_positions"] = markers_array(e.cable_positions); d["cable_velocities"] = markers_array(e.cable_velocities);
            d["confidence"] = vector_array(e.confidence);
            py::array_t<bool> labelled(MARKERS); std::copy(e.labelled.begin(), e.labelled.end(), labelled.mutable_data());
            d["labelled"] = labelled; d["initialized"] = e.initialized;
            return d; })
        .def_property_readonly("filter", [](Observer& o) -> CableFilter& { return o.filter; }, py::return_value_policy::reference_internal)
        .def_readonly("history", &Observer::history)
        .def_readonly("dt", &Observer::dt);
}
