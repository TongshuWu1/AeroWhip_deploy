#include "engine.hpp"
#include "runtime.hpp"
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <algorithm>
#include <cmath>

namespace py = pybind11;
using namespace aerowhip;
using Array = py::array_t<double, py::array::c_style | py::array::forcecast>;

Vector vector(const py::handle& value, int ndim=1) {
    auto a=Array::ensure(value);
    if(!a || a.ndim()!=ndim) throw py::value_error("Incorrect array rank");
    return Vector(a.data(),a.data()+a.size());
}
template<size_t N> std::array<double,N> fixed(const py::handle& value, int ndim=1) {
    auto v=vector(value,ndim);
    if(v.size()!=N) throw py::value_error("Incorrect array size");
    if(ndim==2) {
        auto a=Array::ensure(value);
        if(a.shape(0)!=3 || a.shape(1)!=3) throw py::value_error("Expected a 3x3 matrix");
    }
    std::array<double,N> out; std::copy(v.begin(),v.end(),out.begin()); return out;
}
py::array_t<double> array(const Vector& v) {
    py::array_t<double> result(v.size()); std::copy(v.begin(),v.end(),result.mutable_data()); return result;
}
Parameters parameters(const py::dict& d) {
    Parameters p;
    p.mass=d["mass"].cast<double>(); p.stiffness=d["stiffness"].cast<double>(); p.drag=d["drag"].cast<double>();
    p.bending_damping=d["bending_damping"].cast<double>();p.torsion_stiffness=d["torsion_stiffness"].cast<double>();
    p.torsion_damping=d["torsion_damping"].cast<double>();
    p.maximum_step=d["maximum_step"].cast<double>(); p.projections=d["projections"].cast<int>();
    p.inertia=fixed<9>(d["inertia"],2); p.arm=fixed<3>(d["arm"]); p.gravity=fixed<3>(d["gravity"]);
    p.root_tangent_body=fixed<3>(d["root_tangent_body"]);
    p.rest=vector(d["rest"]); p.node_mass=vector(d["node_mass"]);
    if(d.contains("residual") && !d["residual"].is_none()) {
        auto r=d["residual"].cast<py::dict>();
        p.residual.cable_log_bound=r["cable_log_bound"].cast<double>();
        p.residual.drone_max_rate=r["drone_max_rate"].cast<double>();
        p.residual.velocity_scale=r["velocity_scale"].cast<double>();
        p.residual.cable_enabled=r["cable_enabled"].cast<bool>();
        p.residual.drone_enabled=r["drone_enabled"].cast<bool>();
        auto read_layers=[&](const char* key,std::vector<Layer>& layers) {
        for(auto item:r[key].cast<py::list>()) {
            auto l=item.cast<py::tuple>();
            if(l.size()!=2) throw py::value_error("Layer requires weights and bias");
            auto w=Array::ensure(l[0]);
            if(!w || w.ndim()!=2) throw py::value_error("Layer weights must be a matrix");
            layers.push_back(Layer{int(w.shape(1)),int(w.shape(0)),vector(w,2),vector(l[1])});
        }
        };
        read_layers("cable",p.residual.cable);read_layers("drone",p.residual.drone);
    }
    return p;
}
Model model(const py::dict& c,const py::object& residual) {
    auto d=c["drone"].cast<py::dict>(),a=c["attachment"].cast<py::dict>();
    auto cable=c["cable"].cast<py::dict>(),n=c["simulation"].cast<py::dict>();
    if(a["type"].cast<std::string>()!="point_clamp_der_halfcell_v1" || !a["transmit_direct_moment"].cast<bool>())
        throw py::value_error("This engine requires point_clamp_der_halfcell_v1; historical pivot models are not converted implicitly");
    ModelInput input;auto& p=input.parameters;
    p.root_tangent_body=fixed<3>(a["tangent_body"]);
    p.mass=d["mass_kg"].cast<double>();p.inertia=fixed<9>(d["inertia_body_about_com_kg_m2"],2);
    input.tracking_to_com=fixed<3>(d["center_of_mass_tracking_m"]);
    input.tracking_to_attachment=fixed<3>(a["position_tracking_m"]);
    input.intervals=vector(cable["marker_interval_lengths_m"]);
    input.subdivisions=vector(n["segments_per_marker_interval"]);
    input.marker_mass=vector(cable["moving_marker_masses_kg"]);
    input.total_cable_mass=cable["total_mass_kg"].cast<double>();
    input.output_step=n["timestep_s"].cast<double>();
    input.delay=c["controller"].cast<py::dict>()["command_delay_s"].cast<double>();
    p.gravity=fixed<3>(n["gravity_world_m_s2"]);p.stiffness=cable["EI_n_m2"].cast<double>();p.drag=cable["external_drag_s_inv"].cast<double>();
    if(cable["constitutive_model"].cast<std::string>()!="der_material_kelvin_voigt_v1")
        throw py::value_error("Expected explicit material cable schema; historical fits are not converted implicitly");
    p.bending_damping=cable["Cb_n_m2_s"].cast<double>();p.torsion_stiffness=cable["GJ_n_m2"].cast<double>();
    p.torsion_damping=cable["Ct_n_m2_s"].cast<double>();
    double iterations=n["constraint_iterations"].cast<double>();
    if(!std::isfinite(iterations)||iterations<1||std::floor(iterations)!=iterations||iterations>1000000.)
        throw py::value_error("Invalid projection count");
    p.projections=static_cast<int>(iterations);
    if(n.contains("physics_timestep_s")) {
        if(n.contains("cable_substeps")) throw py::value_error("Use physics_timestep_s or cable_substeps, not both");
        p.maximum_step=n["physics_timestep_s"].cast<double>();
    } else {
        double steps=n["cable_substeps"].cast<double>();
        if(py::isinstance<py::bool_>(n["cable_substeps"])||!std::isfinite(steps)||steps<1||std::floor(steps)!=steps)
            throw py::value_error("Invalid legacy cable_substeps");
        p.maximum_step=input.output_step/steps;
    }
    Model result(std::move(input));
    if(!residual.is_none()) {
        py::dict raw;
        const auto& v=result.parameters;
        raw["mass"]=v.mass;raw["inertia"]=py::array_t<double>({3,3},v.inertia.data());
        raw["arm"]=v.arm;raw["gravity"]=v.gravity;raw["rest"]=v.rest;raw["node_mass"]=v.node_mass;
        raw["root_tangent_body"]=v.root_tangent_body;
        raw["bending_damping"]=v.bending_damping;raw["torsion_stiffness"]=v.torsion_stiffness;raw["torsion_damping"]=v.torsion_damping;
        raw["stiffness"]=v.stiffness;raw["drag"]=v.drag;raw["maximum_step"]=v.maximum_step;raw["projections"]=v.projections;
        raw["residual"]=residual;result.parameters=parameters(raw);Engine validate(result.parameters);
    }
    return result;
}
py::dict info(const Model& model) {
    auto& p=model.parameters;py::dict d;
    d["mass"]=p.mass;d["inertia"]=py::array_t<double>({3,3},p.inertia.data());
    d["com_tracking"]=array(Vector(model.com.begin(),model.com.end()));
    d["attachment_tracking"]=array(Vector(model.attachment.begin(),model.attachment.end()));
    d["arm"]=array(Vector(p.arm.begin(),p.arm.end()));d["gravity"]=array(Vector(p.gravity.begin(),p.gravity.end()));
    d["root_tangent_body"]=array(Vector(p.root_tangent_body.begin(),p.root_tangent_body.end()));
    d["rest"]=array(p.rest);d["node_mass"]=array(p.node_mass);d["ei"]=p.stiffness;d["drag"]=p.drag;
    d["cb"]=p.bending_damping;d["gj"]=p.torsion_stiffness;d["ct"]=p.torsion_damping;
    d["dt"]=model.output_step;d["delay_s"]=model.delay;d["physics_dt"]=p.maximum_step;d["iterations"]=p.projections;
    d["marker_indices"]=model.marker_indices;d["n"]=p.node_mass.size();return d;
}
Controller callback(py::object controller) {
    return [controller=std::move(controller),last=std::optional<Command>{}](const Vector& s,const Command& c) mutable {
        py::gil_scoped_acquire acquire;
        if(!last || *last!=c) {controller.attr("set_command")(array(Vector(c.begin(),c.end())));last=c;}
        return fixed<4>(controller(array(s)));
    };
}
Actuator actuator(const py::dict& c) {
    auto positions=Array::ensure(c["motor_positions_com_m"]);
    if(!positions||positions.ndim()!=2||positions.shape(0)!=4||positions.shape(1)!=3)
        throw py::value_error("motor_positions_com_m must have shape (4,3)");
    std::array<Vec3,4> p;
    for(int i=0;i<4;++i) std::copy(positions.data()+3*i,positions.data()+3*i+3,p[i].begin());
    return Actuator(p,fixed<4>(c["yaw_torque_per_thrust_m"]),fixed<4>(c["minimum_thrust_n"]),
                    fixed<4>(c["maximum_thrust_n"]),fixed<4>(c["response_time_s"]),fixed<4>(c["initial_thrust_n"]));
}
py::dict actuator_info(const Actuator& a) {
    py::dict out;
    // Copy matrices: never expose a dangling view of a temporary C++ object.
    auto matrix=[](const std::array<double,16>& v){py::array_t<double> m({4,4});std::copy(v.begin(),v.end(),m.mutable_data());return m;};
    out["mixing"]=matrix(a.mixing());out["inverse"]=matrix(a.inverse());
    out["minimum"]=a.minimum;out["maximum"]=a.maximum;out["response_s"]=a.response_s;out["initial"]=a.initial;
    return out;
}
py::object actuation_info(const Runtime& r) {
    if(!r.actuator()) return py::none();
    const auto& a=*r.actuator();py::dict out;
    out["motor_thrust_n"]=r.motors();out["delivered_wrench"]=r.delivered_effort();
    out["requested_wrench"]=r.has_effort()?py::cast(r.effort()):py::none();
    out["unclipped_motor_thrust_n"]=r.has_effort()?py::cast(a.allocate(r.effort())):py::none();
    out["target_motor_thrust_n"]=r.has_effort()?py::cast(a.target(r.effort())):py::none();
    return out;
}
PYBIND11_MODULE(_native, m) {
    m.doc()="Native float64 coupled drone/DDER plant and event runtime; Python owns control.";
    m.attr("api_version")=11;
    py::class_<Actuator>(m,"Actuator")
        .def(py::init(&actuator))
        .def("info",&actuator_info)
        .def("allocate",&Actuator::allocate)
        .def("target",&Actuator::target)
        .def("response",&Actuator::response)
        .def("wrench",&Actuator::wrench)
        .def("validate",&Actuator::validate);
    py::class_<Model>(m,"Model")
        .def(py::init(&model),py::arg("config"),py::arg("residual")=py::none())
        .def("info",&info)
        .def("initial_state",[](const Model& m,py::object p,py::object R,py::object v,py::object omega,py::object q,py::object qv){
            return array(m.initial_state(fixed<3>(p),fixed<9>(R,2),fixed<3>(v),fixed<3>(omega),
                                        q.is_none()?Vector{}:vector(q,2),qv.is_none()?Vector{}:vector(qv,2)));
        },py::arg("position"),py::arg("rotation"),py::arg("velocity"),py::arg("omega"),py::arg("cable_positions")=py::none(),py::arg("cable_velocities")=py::none())
        .def("observation",[](const Model& m,Array s){return array(m.observation(vector(s)));})
        .def("prescribed_advance",[](const Model&m,Array s,Array samples,Array history,double previous_step){
            auto state=vector(s),inputs=vector(samples,2),previous=vector(history);Vector out;
            if(samples.shape(1)!=85)throw py::value_error("Expected h and four boundary samples");
            {py::gil_scoped_release release;out=prescribed_advance(m,std::move(state),inputs,previous,previous_step);}
            return py::make_tuple(array(out),array(previous),previous_step);
        })
        .def("prescribed_acceleration",[](const Model&m,Array s,Array b){
            auto state=vector(s),boundary=vector(b);Vector out;
            if(boundary.size()!=21)throw py::value_error("Expected21-value boundary");
            {py::gil_scoped_release release;out=prescribed_acceleration(m,state,boundary);}return array(out);
        });
    py::class_<Runtime>(m,"Runtime")
        .def(py::init([](Model m,Array s,Array c,std::optional<double> period,std::optional<double> delay,std::optional<Actuator> a){
            return new Runtime(std::move(m),vector(s),vector(c),period,delay,std::move(a));
        }),py::arg("model"),py::arg("state"),py::arg("hold"),py::arg("controller_period")=py::none(),py::arg("delay")=py::none(),py::arg("actuator")=py::none())
        .def("reset",&Runtime::reset)
        .def("checkpoint",[](const Runtime& r){return array(r.checkpoint());})
        .def("restore",[](Runtime& r,Array checkpoint){r.restore(vector(checkpoint));})
        .def("step",[](Runtime& r,py::list events,py::object controller){
            std::vector<Event> checked;
            for(auto event:events) {auto item=event.cast<py::tuple>();if(item.size()!=2)throw py::value_error("Expected timestamp/command pairs");checked.push_back({item[0].cast<double>(),vector(item[1])});}
            auto evaluate=callback(std::move(controller));{py::gil_scoped_release release;r.step(checked,evaluate);}
        })
        .def("snapshot",[](const Runtime& r,bool forces,py::object controller){
            auto evaluate=callback(std::move(controller));Vector out;
            {py::gil_scoped_release release;out=r.snapshot(forces,evaluate);}return array(out);
        })
        .def("state",[](const Runtime& r){return array(r.state());})
        .def_property_readonly("requested",[](const Runtime& r){auto c=r.requested();return array(Vector(c.begin(),c.end()));})
        .def_property_readonly("applied",[](const Runtime& r){auto c=r.applied();return array(Vector(c.begin(),c.end()));})
        .def_property_readonly("command",[](const Runtime& r){auto c=r.command();return array(Vector(c.begin(),c.end()));})
        .def_property_readonly("effort",&Runtime::effort)
        .def_property_readonly("actuation",&actuation_info)
        .def_property_readonly("time",&Runtime::time)
        .def_property_readonly("requested_events",&Runtime::requested_events)
        .def_property_readonly("applied_events",&Runtime::applied_events)
        .def_property_readonly("controller_updates",&Runtime::controller_updates);
    py::class_<Engine>(m,"Engine")
        .def(py::init([](py::dict p){return new Engine(parameters(p));}))
        .def_property_readonly("node_count",&Engine::node_count)
        .def("reset",[](Engine& e,Array s){auto v=vector(s); py::gil_scoped_release release; e.reset(v);})
        .def("state",[](const Engine& e){return array(e.state());})
        .def("checkpoint",[](const Engine& e){return array(e.checkpoint());})
        .def("restore",[](Engine& e,Array s){auto v=vector(s);py::gil_scoped_release release;e.restore(v);})
        .def("acceleration",[](const Engine& e,Array w){
            auto input=fixed<4>(w); Vector result;
            {py::gil_scoped_release release; result=e.acceleration(input);} return array(result);
        })
        .def("advance",[](Engine& e,double duration,Array w){
            auto input=fixed<4>(w); py::gil_scoped_release release; e.advance(duration,input);
        })
        .def("advance_motors",[](Engine& e,double duration,Array w,const Actuator& a,Array motors){
            auto input=fixed<4>(w),f=fixed<4>(motors);Motors out;
            {py::gil_scoped_release release;out=e.advance_motors(duration,input,a,f);}return out;
        })
        .def("advance_controlled",[](Engine& e,double duration,py::function controller){
            auto callback=[&](const Vector& state){
                py::gil_scoped_acquire acquire;
                // Owned copy: a controller may retain this state safely.
                return fixed<4>(controller(array(state)));
            };
            py::gil_scoped_release release; e.advance_controlled(duration,callback);
        });
}
