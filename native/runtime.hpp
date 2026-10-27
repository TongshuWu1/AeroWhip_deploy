#pragma once
#include "engine.hpp"
#include "actuator.hpp"
#include <deque>
#include <optional>
#include <utility>

namespace aerowhip {
// Legacy PVA/yaw/zero yaw-rate (11), or PVA/yaw/desired body rates (13).
// Scheduling is agnostic to the concrete Python controller law.
using Command=Vector;
using Event=std::pair<double,Command>;
using Controller=std::function<Wrench(const Vector&,const Command&)>;

class Runtime {
public:
    Runtime(Model model,Vector initial,Command hold,std::optional<double> controller_period,
            std::optional<double> command_delay,std::optional<Actuator> actuator=std::nullopt);
    void reset();
    void reinitialize(Vector initial) {engine_.reset(initial);initial_=std::move(initial);reset();}
    Vector checkpoint() const;
    void restore(const Vector& checkpoint);
    void step(const std::vector<Event>& events,const Controller& controller);
    Vector snapshot(bool forces,const Controller& controller) const;
    Vector state() const { return engine_.state(); }
    Command requested() const { return requested_left_; }
    Command applied() const { return applied_left_; }
    Command command() const { return held_; }
    Wrench effort() const { return effort_; }
    Wrench delivered_effort() const { return actuator_->wrench(motors_); }
    Motors motors() const { return motors_; }
    const std::optional<Actuator>& actuator() const { return actuator_; }
    bool has_effort() const { return controller_index_>0; }
    const std::vector<Event>& requested_events() const {return requested_events_;}
    const std::vector<Event>& applied_events() const {return applied_events_;}
    const Vector& controller_updates() const {return updates_;}
    double time() const {return output_index_*model_.output_step;}
    const Model& model() const {return model_;}
private:
    Model model_;
    Engine engine_;
    Vector initial_;
    Command hold_, last_,held_,requested_left_,applied_left_;
    std::optional<double> period_;
    double delay_=0.,physics_time_=0.;
    int64_t output_index_=0,controller_index_=0;
    std::deque<Event> pending_;
    std::vector<Event> requested_events_,applied_events_;
    Vector updates_;
    Wrench effort_{};
    std::optional<Actuator> actuator_;
    Motors motors_{};
    Vec3 acceleration_{};
    void integrate(double duration,const Controller& controller);
};
}
