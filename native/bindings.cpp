#include "engine.hpp"
#include "runtime.hpp"
#include "flight_controller.hpp"
#include <exception>
#include <memory>
#ifdef _OPENMP
#include <omp.h>
#endif
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <algorithm>
#include <string>
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
    p.bending_damping=d["bending_damping"].cast<double>();p.air_density=d["air_density"].cast<double>();
    if(d.contains("bending_damping_saturation")&&!d["bending_damping_saturation"].is_none()) p.bending_damping_saturation=d["bending_damping_saturation"].cast<double>();
    p.normal_drag_coefficient=d["normal_drag_coefficient"].cast<double>();p.diameter=d["diameter"].cast<double>();
    p.maximum_step=d["maximum_step"].cast<double>(); p.projections=d["projections"].cast<int>();
    p.inertia=fixed<9>(d["inertia"],2); p.arm=fixed<3>(d["arm"]); p.gravity=fixed<3>(d["gravity"]);
    p.root_tangent_body=fixed<3>(d["root_tangent_body"]);
    if(d.contains("body_drag")) p.body_drag=fixed<3>(d["body_drag"]);
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
    if(!d.contains("body_drag_s_inv")) throw py::value_error("drone.body_drag_s_inv (linear drag rates per body axis, s^-1) is required");
    p.body_drag=fixed<3>(d["body_drag_s_inv"]);
    for(double k:p.body_drag) if(!std::isfinite(k)||k<0.) throw py::value_error("drone.body_drag_s_inv must be finite and nonnegative");
    input.tracking_to_com=fixed<3>(d["center_of_mass_tracking_m"]);
    input.tracking_to_attachment=fixed<3>(a["position_tracking_m"]);
    input.intervals=vector(cable["marker_interval_lengths_m"]);
    input.subdivisions=vector(n["segments_per_marker_interval"]);
    input.marker_mass=vector(cable["moving_marker_masses_kg"]);
    input.total_cable_mass=cable["total_mass_kg"].cast<double>();
    input.output_step=n["timestep_s"].cast<double>();
    input.delay=c["controller"].cast<py::dict>()["command_delay_s"].cast<double>();
    p.gravity=fixed<3>(n["gravity_world_m_s2"]);p.stiffness=cable["EI_n_m2"].cast<double>();p.drag=cable["external_drag_s_inv"].cast<double>();
    if(cable["constitutive_model"].cast<std::string>()!="der_twist_free_kelvin_voigt_v2")
        throw py::value_error("Expected the twist-free cable schema der_twist_free_kelvin_voigt_v2; earlier fits are not converted implicitly");
    if(cable.contains("GJ_n_m2")||cable.contains("Ct_n_m2_s"))
        throw py::value_error("The twist-free cable has no torsion parameters; remove GJ_n_m2/Ct_n_m2_s");
    p.bending_damping=cable["Cb_n_m2_s"].cast<double>();p.diameter=cable["diameter_m"].cast<double>();
    if(!cable.contains("Cb_saturation_rate_rad_s")) throw py::value_error("cable.Cb_saturation_rate_rad_s is required (null: linear bending damping)");
    if(!cable["Cb_saturation_rate_rad_s"].is_none()) p.bending_damping_saturation=cable["Cb_saturation_rate_rad_s"].cast<double>();
    p.normal_drag_coefficient=cable["normal_drag_coefficient"].cast<double>();
    p.air_density=n["air_density_kg_m3"].cast<double>();
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
        raw["root_tangent_body"]=v.root_tangent_body;raw["body_drag"]=v.body_drag;
        raw["bending_damping"]=v.bending_damping;raw["air_density"]=v.air_density;raw["bending_damping_saturation"]=v.bending_damping_saturation;
        raw["normal_drag_coefficient"]=v.normal_drag_coefficient;raw["diameter"]=v.diameter;
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
    d["body_drag"]=array(Vector(p.body_drag.begin(),p.body_drag.end()));
    d["rest"]=array(p.rest);d["node_mass"]=array(p.node_mass);d["ei"]=p.stiffness;d["drag"]=p.drag;
    d["cb"]=p.bending_damping;d["cb_saturation"]=p.bending_damping_saturation;d["cd"]=p.normal_drag_coefficient;d["diameter"]=p.diameter;d["air_density"]=p.air_density;
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
mellinger::Parameters mellinger_parameters(const py::dict& d) {
    mellinger::Parameters p;
    auto get=[&](const char* key,float& field){if(d.contains(key))field=d[key].cast<float>();};
    get("mass",p.mass);get("massThrust",p.massThrust);get("kp_xy",p.kp_xy);get("kd_xy",p.kd_xy);get("ki_xy",p.ki_xy);get("i_range_xy",p.i_range_xy);
    get("kp_z",p.kp_z);get("kd_z",p.kd_z);get("ki_z",p.ki_z);get("i_range_z",p.i_range_z);get("kR_xy",p.kR_xy);get("kw_xy",p.kw_xy);
    get("ki_m_xy",p.ki_m_xy);get("i_range_m_xy",p.i_range_m_xy);get("kR_z",p.kR_z);get("kw_z",p.kw_z);get("ki_m_z",p.ki_m_z);get("i_range_m_z",p.i_range_m_z);
    get("kd_omega_rp",p.kd_omega_rp);
    for(auto item:d){std::string k=py::str(item.first);static const char* known[]={"mass","massThrust","kp_xy","kd_xy","ki_xy","i_range_xy","kp_z","kd_z","ki_z","i_range_z","kR_xy","kw_xy","ki_m_xy","i_range_m_xy","kR_z","kw_z","ki_m_z","i_range_m_z","kd_omega_rp"};
        bool ok=false;for(auto n:known)if(k==n)ok=true;if(!ok)throw py::value_error("Unknown ctrlMel parameter: "+k);}
    return p;
}
FlightController flight_controller(const py::dict& firmware,const py::dict& model_config,double voltage) {
    // Attitude gains from the model's controller section; firmware and thrust map from `firmware`.
    FlightControllerParameters p;
    auto controller=model_config["controller"].cast<py::dict>();auto drone=model_config["drone"].cast<py::dict>();
    p.firmware_mass=firmware["firmware_mass_kg"].cast<double>();
    p.thrust_scale=firmware["thrust_scale"].cast<double>();p.thrust_exponent=firmware["thrust_exponent"].cast<double>();
    p.voltage_slope=firmware["voltage_slope_v_inv"].cast<double>();p.reference_voltage=firmware["reference_voltage_v"].cast<double>();
    p.tilt_gain=firmware["tilt_gain"].cast<double>();
    auto w=fixed<3>(controller["attitude_frequency_rad_s"]),z=fixed<3>(controller["attitude_damping_ratio"]);
    for(int i=0;i<3;++i){p.attitude_stiffness[i]=w[i]*w[i];p.attitude_damping[i]=2.*z[i]*w[i];}
    p.inertia=fixed<9>(drone["inertia_body_about_com_kg_m2"],2);p.com=fixed<3>(drone["center_of_mass_tracking_m"]);
    p.gravity=fixed<3>(model_config["simulation"].cast<py::dict>()["gravity_world_m_s2"]);
    std::string law=firmware.contains("law")?firmware["law"].cast<std::string>():"identified";
    if(law=="firmware"){
        p.law=FlightLaw::firmware;
        p.firmware=mellinger_parameters(firmware["ctrlMel"].cast<py::dict>());
        auto a=actuator(model_config["actuator"].cast<py::dict>());
        p.mixing=a.mixing();p.motor_minimum=a.minimum;p.motor_maximum=a.maximum;
    } else if(law!="identified") throw py::value_error("law must be 'identified' or 'firmware'");
    return FlightController(p,voltage);
}
// Independent lanes of one model, each with its own flight controller (battery voltage),
// stepped in parallel without the GIL. A lane that raises is marked failed and left as it was.
struct BatchRuntime {
    std::vector<std::unique_ptr<Runtime>> lanes;std::vector<FlightController> controllers;std::vector<char> failed;
    std::vector<std::string> errors;
    void step(const std::vector<Command>& commands) {
        if(commands.size()!=lanes.size()) throw py::value_error("One command per lane");
        #pragma omp parallel for schedule(dynamic,1)
        for(long i=0;i<long(lanes.size());++i) {
            if(failed[i]) continue;
            try {
                auto& c=controllers[i];
                lanes[i]->step({{lanes[i]->time(),commands[i]}},[&c](const Vector& s,const Command& u){return c.step(s,u);});
            } catch(const std::exception& e) {failed[i]=1;errors[i]=e.what();}
        }
    }
};
void bind_estimator(py::module_& m);   // native/estimator_bindings.cpp

PYBIND11_MODULE(_native, m) {
    bind_estimator(m);
    m.doc()="Native float64 coupled drone/DDER plant and event runtime; Python owns control.";
    m.attr("api_version")=12;
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
        .def("step",[](Runtime& r,py::list events,FlightController& controller){
            std::vector<Event> checked;
            for(auto event:events) {auto item=event.cast<py::tuple>();if(item.size()!=2)throw py::value_error("Expected timestamp/command pairs");checked.push_back({item[0].cast<double>(),vector(item[1])});}
            py::gil_scoped_release release;r.step(checked,[&controller](const Vector& s,const Command& c){return controller.step(s,c);});
        })
        .def("reinitialize",[](Runtime& r,Array s){auto v=vector(s);r.reinitialize(v);})
        .def("step",[](Runtime& r,py::list events,py::object controller){
            std::vector<Event> checked;
            for(auto event:events) {auto item=event.cast<py::tuple>();if(item.size()!=2)throw py::value_error("Expected timestamp/command pairs");checked.push_back({item[0].cast<double>(),vector(item[1])});}
            auto evaluate=callback(std::move(controller));{py::gil_scoped_release release;r.step(checked,evaluate);}
        })
        .def("snapshot",[](const Runtime& r,bool forces,const FlightController& controller){
            Vector out;{py::gil_scoped_release release;out=r.snapshot(forces,[&controller](const Vector& s,const Command& c){return controller.peek(s,c);});}return array(out);
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
    py::class_<FlightController>(m,"FlightController")
        .def(py::init(&flight_controller),py::arg("firmware"),py::arg("model_config"),py::arg("battery_voltage"))
        .def_property("battery_voltage",&FlightController::voltage,&FlightController::set_voltage)
        .def_property("acceleration_bias",&FlightController::bias,[](FlightController& c,py::object b){c.set_bias(fixed<3>(b));})
        .def("delivered_thrust",&FlightController::delivered_thrust)
        .def("reset",&FlightController::reset)
        .def_property_readonly("law",[](const FlightController& c){return c.law()==FlightLaw::firmware?"firmware":"identified";})
        .def("motors",[](const FlightController& c,Array s,Array command){return c.motors(vector(s),vector(command));})
        .def("__call__",[](const FlightController& c,Array s,Array command){return c(vector(s),vector(command));})
        .def("step",[](FlightController& c,Array s,Array command){return c.step(vector(s),vector(command));});
    // The firmware port alone, with the harness layout of tests/test_firmware_port.py:
    // in (23): sp_pos sp_vel sp_acc sp_yaw_deg pos vel quat_xyzw gyro_deg_s; out (9): thrust roll pitch yaw pwm1..4 capped.
    py::class_<mellinger::Controller>(m,"Mellinger")
        .def(py::init([](py::dict p){return new mellinger::Controller(mellinger_parameters(p));}))
        .def("reset",&mellinger::Controller::reset)
        .def("step",[](mellinger::Controller& c,py::array_t<float,py::array::c_style|py::array::forcecast> in){
            if(in.size()!=23) throw py::value_error("Expected 23 inputs");const float* v=in.data();
            mellinger::Input i;i.setpoint_position={v[0],v[1],v[2]};i.setpoint_velocity={v[3],v[4],v[5]};i.setpoint_acceleration={v[6],v[7],v[8]};
            i.setpoint_yaw_deg=v[9];i.position={v[10],v[11],v[12]};i.velocity={v[13],v[14],v[15]};i.qx=v[16];i.qy=v[17];i.qz=v[18];i.qw=v[19];i.gyro={v[20],v[21],v[22]};
            auto o=c.step(i);py::array_t<float> out(9);float* d=out.mutable_data();
            d[0]=o.thrust;d[1]=o.roll;d[2]=o.pitch;d[3]=o.yaw;for(int k=0;k<4;++k)d[4+k]=o.pwm[k];d[8]=o.capped?1.f:0.f;return out;});
    py::class_<BatchRuntime>(m,"BatchRuntime")
        .def(py::init([](const Model& model,Array state,Array hold,double period,double delay,const Actuator& actuator,std::vector<FlightController> controllers){
            auto b=new BatchRuntime;auto s=vector(state),c=vector(hold);
            for(size_t i=0;i<controllers.size();++i) b->lanes.push_back(std::make_unique<Runtime>(model,s,c,period,delay,actuator));
            b->controllers=std::move(controllers);
            b->failed.assign(b->lanes.size(),0);b->errors.assign(b->lanes.size(),"");return b;}),
            py::arg("model"),py::arg("state"),py::arg("hold"),py::arg("controller_period"),py::arg("delay"),py::arg("actuator"),py::arg("controllers"))
        .def_property_readonly("size",[](const BatchRuntime& b){return b.lanes.size();})
        .def_static("threads",[](){
#ifdef _OPENMP
            return omp_get_max_threads();
#else
            return 1;
#endif
        })
        .def("reinitialize",[](BatchRuntime& b,py::array_t<double,py::array::c_style|py::array::forcecast> states){
            if(states.ndim()!=2||size_t(states.shape(0))!=b.lanes.size()) throw py::value_error("Expected one state row per lane");
            size_t w=states.shape(1);std::vector<Vector> rows(b.lanes.size());
            for(size_t i=0;i<rows.size();++i) rows[i]=Vector(states.data()+i*w,states.data()+(i+1)*w);
            py::gil_scoped_release release;
            #pragma omp parallel for
            for(long i=0;i<long(rows.size());++i){
                try{b.lanes[i]->reinitialize(rows[i]);b.controllers[i].reset();b.failed[i]=0;b.errors[i].clear();}
                catch(const std::exception& e){b.failed[i]=1;b.errors[i]=e.what();}}
        })
        .def("set_voltage",[](BatchRuntime& b,py::array_t<double,py::array::c_style|py::array::forcecast> v){
            if(size_t(v.size())!=b.lanes.size()) throw py::value_error("One voltage per lane");
            for(size_t i=0;i<b.lanes.size();++i) b.controllers[i].set_voltage(v.data()[i]);
        })
        .def("set_bias",[](BatchRuntime& b,py::array_t<double,py::array::c_style|py::array::forcecast> v){
            if(v.ndim()!=2||size_t(v.shape(0))!=b.lanes.size()||v.shape(1)!=3) throw py::value_error("One world bias row per lane");
            for(size_t i=0;i<b.lanes.size();++i) b.controllers[i].set_bias({v.data()[3*i],v.data()[3*i+1],v.data()[3*i+2]});
        })
        .def("step",[](BatchRuntime& b,py::array_t<double,py::array::c_style|py::array::forcecast> commands){
            if(commands.ndim()!=2||size_t(commands.shape(0))!=b.lanes.size()||(commands.shape(1)!=11&&commands.shape(1)!=13))
                throw py::value_error("Expected one 11- or 13-value command row per lane");
            size_t w=commands.shape(1);std::vector<Command> rows(b.lanes.size());
            for(size_t i=0;i<rows.size();++i) rows[i]=Command(commands.data()+i*w,commands.data()+(i+1)*w);
            py::gil_scoped_release release;b.step(rows);
        })
        .def("observations",[](const BatchRuntime& b){
            // Model observation per lane (tracking P/V, R, body rate, cable P/V, acceleration).
            size_t width=b.lanes[0]->model().observation(b.lanes[0]->state()).size();
            py::array_t<double> out({b.lanes.size(),width});double* d=out.mutable_data();
            {py::gil_scoped_release release;
            #pragma omp parallel for
            for(long i=0;i<long(b.lanes.size());++i){auto o=b.lanes[i]->model().observation(b.lanes[i]->state());std::copy(o.begin(),o.end(),d+i*width);}}
            return out;
        })
        .def("states",[](const BatchRuntime& b){
            size_t width=b.lanes[0]->state().size();py::array_t<double> out({b.lanes.size(),width});double* d=out.mutable_data();
            for(size_t i=0;i<b.lanes.size();++i){auto s=b.lanes[i]->state();std::copy(s.begin(),s.end(),d+i*width);}return out;
        })
        .def("motors",[](const BatchRuntime& b){
            py::array_t<double> out({b.lanes.size(),size_t(4)});double* d=out.mutable_data();
            for(size_t i=0;i<b.lanes.size();++i){auto f=b.lanes[i]->motors();std::copy(f.begin(),f.end(),d+4*i);}return out;
        })
        .def_property_readonly("failed",[](const BatchRuntime& b){return std::vector<bool>(b.failed.begin(),b.failed.end());})
        .def_property_readonly("errors",[](const BatchRuntime& b){return b.errors;});
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
