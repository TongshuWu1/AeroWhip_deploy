#include "runtime.hpp"
#include <algorithm>
#include <cmath>
#include <limits>
#include <stdexcept>

namespace aerowhip {
namespace {
constexpr double eps=1e-12;
void check(bool valid,const char* message) {if(!valid) throw std::invalid_argument(message);}
void command_valid(const Command& c) {
    check(c.size()==11 || c.size()==13,"Expected 11 legacy or 13 full-state command values");
    for(double x:c) check(std::isfinite(x),"Command must contain finite P/V/A/yaw values");
    check(c.size()!=11 || c[10]==0.,"Legacy yaw-rate feedforward is not supported; supply body rates");
}
void effort_valid(const Wrench& w) {
    for(double x:w) check(std::isfinite(x),"Controller returned nonfinite effort");
    check(w[0]>=0.,"Collective thrust must be nonnegative");
}
}
Runtime::Runtime(Model model,Vector initial,Command hold,std::optional<double> period,std::optional<double> delay,std::optional<Actuator> actuator)
    : model_(std::move(model)),engine_(model_.parameters),initial_(std::move(initial)),hold_(hold),period_(period),delay_(delay.value_or(model_.delay)),actuator_(std::move(actuator)) {
    check(period_ && std::isfinite(*period_)&&*period_>eps,"A sampled controller period is required");
    check(actuator_.has_value(),"The runtime requires four-motor actuation; ideal-wrench mode was removed");
    check(std::isfinite(delay_)&&delay_>=0.,"Invalid command delay");command_valid(hold_);reset();
}
void Runtime::reset() {
    engine_.reset(initial_);last_=held_=requested_left_=applied_left_=hold_;
    pending_.clear();requested_events_.clear();applied_events_.clear();updates_.clear();
    output_index_=controller_index_=0;physics_time_=0.;effort_={};acceleration_={};
    motors_=actuator_->initial;
}
void Runtime::integrate(double duration,const Controller& controller) {
    const double end=physics_time_+duration;
    while(physics_time_<end-eps) {
        double tick=controller_index_*(*period_);
        if(tick<=physics_time_+eps) {
            check(tick>=physics_time_-eps,"Controller clock skipped an update");
            effort_=controller(engine_.state(),held_);effort_valid(effort_);
            updates_.push_back(tick);++controller_index_;tick=controller_index_*(*period_);
        }
        double stop=std::min(end,tick);
        motors_=engine_.advance_motors(stop-physics_time_,effort_,*actuator_,motors_);
        physics_time_=stop;
    }
    physics_time_=end;
}
Vector Runtime::checkpoint() const {
    // Versioned, serializable wire state. Restore only into the same configured
    // runtime/controller; no model coefficients or Python controller are saved.
    Vector out{1.,model_.output_step,*period_,delay_,physics_time_,double(output_index_),double(controller_index_)};
    auto append=[&](const auto& x){out.push_back(double(x.size()));out.insert(out.end(),x.begin(),x.end());};
    append(engine_.checkpoint());append(last_);append(held_);append(requested_left_);append(applied_left_);
    append(effort_);append(motors_);append(acceleration_);append(updates_);
    for(const auto& events:{std::vector<Event>(pending_.begin(),pending_.end()),requested_events_,applied_events_}) {
        out.push_back(double(events.size()));for(const auto&e:events){out.push_back(e.first);append(e.second);}
    }
    return out;
}
void Runtime::restore(const Vector& in) {
    check(in.size()>=7,"Invalid runtime checkpoint");
    for(double x:in)check(std::isfinite(x),"Nonfinite runtime checkpoint");
    check(in[0]==1.&&in[1]==model_.output_step&&in[2]==*period_&&in[3]==delay_,"Runtime checkpoint schema/clock mismatch");
    check(in[5]>=0&&in[6]>=0&&in[5]==std::floor(in[5])&&in[6]==std::floor(in[6])
          &&in[5]<double(std::numeric_limits<int64_t>::max())&&in[6]<double(std::numeric_limits<int64_t>::max()),"Invalid checkpoint indices");
    check(std::abs(in[4]-in[5]*model_.output_step)<eps,"Inconsistent checkpoint time");
    check(in[6]*(*period_)>=in[4]-eps&&in[6]*(*period_)<in[4]+(*period_)+eps,"Inconsistent controller clock");
    size_t cursor=7;
    auto scalar=[&](){check(cursor<in.size(),"Truncated runtime checkpoint");return in[cursor++];};
    auto count=[&](){double n=scalar();check(n>=0&&n==std::floor(n)&&n<=double(in.size()-cursor),"Invalid checkpoint field count");return size_t(n);};
    auto read=[&](){size_t n=count();Vector v(in.begin()+cursor,in.begin()+cursor+n);cursor+=n;return v;};
    auto history=read(),last=read(),held=read(),requested=read(),applied=read();
    for(const auto&c:{last,held,requested,applied})command_valid(c);
    auto effort=read(),motors=read(),acceleration=read(),updates=read();
    check(effort.size()==4&&motors.size()==4&&acceleration.size()==3,"Invalid checkpoint actuation");
    Wrench w;Motors f;std::copy(effort.begin(),effort.end(),w.begin());std::copy(motors.begin(),motors.end(),f.begin());
    effort_valid(w);actuator_->validate(f);
    std::array<std::vector<Event>,3> lists;
    for(auto&events:lists){size_t n=count();double previous=-std::numeric_limits<double>::infinity();
        for(size_t i=0;i<n;++i){double t=scalar();auto command=read();command_valid(command);
            check(t>=previous,"Unordered checkpoint events");previous=t;events.push_back({t,std::move(command)});}}
    check(cursor==in.size(),"Extra runtime checkpoint fields");
    if(!lists[0].empty())check(lists[0].front().first>=in[4]-eps,"Past pending checkpoint event");
    engine_.restore(history); // Validate before committing clocks, motors or queues.
    physics_time_=in[4];output_index_=int64_t(in[5]);controller_index_=int64_t(in[6]);
    last_=std::move(last);held_=std::move(held);requested_left_=std::move(requested);applied_left_=std::move(applied);
    effort_=w;motors_=f;std::copy(acceleration.begin(),acceleration.end(),acceleration_.begin());updates_=std::move(updates);
    pending_=std::deque<Event>(lists[0].begin(),lists[0].end());requested_events_=std::move(lists[1]);applied_events_=std::move(lists[2]);
}
void Runtime::step(const std::vector<Event>& events,const Controller& controller) {
    const double start=time(),end=(output_index_+1)*model_.output_step;
    double previous=-std::numeric_limits<double>::infinity();
    for(const auto& event:events) {
        check(std::isfinite(event.first)&&event.first>=start-eps&&event.first<end-eps&&event.first>previous,
              "Command events must increase within the current feedback interval");
        command_valid(event.second);previous=event.first;
    }
    // All scheduling/engine changes commit together. A failed Python callback
    // must not leave a partially advanced clock with the previous output index.
    auto saved_history=engine_.checkpoint();auto saved_state=engine_.state();auto saved_pending=pending_;
    auto saved_requested_events=requested_events_,saved_applied_events=applied_events_;auto saved_updates=updates_;
    auto saved_last=last_,saved_held=held_,saved_requested=requested_left_,saved_applied=applied_left_;
    auto saved_effort=effort_;auto saved_acceleration=acceleration_;
    auto saved_motors=motors_;
    const auto saved_time=physics_time_;const auto saved_controller=controller_index_;
    try {
        requested_events_=events;applied_events_.clear();updates_.clear();requested_left_=last_;
        for(const auto& event:events) {
            if(event.first<=start+eps) requested_left_=event.second;
            last_=event.second;pending_.push_back({event.first+delay_,event.second});
        }
        while(!pending_.empty()&&pending_.front().first<=start+eps) {
            auto event=pending_.front();pending_.pop_front();held_=event.second;applied_events_.push_back(event);
        }
        applied_left_=held_;auto observation=model_.observation(saved_state);double cursor=start;
        while(!pending_.empty()&&pending_.front().first<end-eps) {
            auto event=pending_.front();pending_.pop_front();
            if(event.first>cursor+eps) integrate(event.first-cursor,controller);
            cursor=std::max(cursor,event.first);held_=event.second;applied_events_.push_back(event);
        }
        integrate(end-cursor,controller);auto after=model_.observation(engine_.state());
        for(int i=0;i<3;++i) acceleration_[i]=(after[3+i]-observation[3+i])/model_.output_step;
        ++output_index_;
    } catch(...) {
        engine_.restore(saved_history);pending_=std::move(saved_pending);requested_events_=std::move(saved_requested_events);
        applied_events_=std::move(saved_applied_events);updates_=std::move(saved_updates);last_=saved_last;held_=saved_held;
        requested_left_=saved_requested;applied_left_=saved_applied;effort_=saved_effort;acceleration_=saved_acceleration;
        physics_time_=saved_time;controller_index_=saved_controller;motors_=saved_motors;throw;
    }
}
Vector Runtime::snapshot(bool forces,const Controller& controller) const {
    auto s=engine_.state(), out=model_.observation(s,acceleration_);
    if(forces) {
        auto w=delivered_effort();effort_valid(w);auto a=engine_.acceleration(w);auto air=engine_.drone_air_force();
        out.insert(out.end(),s.begin(),s.begin()+3);
        for(int i=0;i<3;++i) out.push_back(w[0]*s[3+3*i+2]);
        // Display cable-on-drone force, not cable + external vehicle drag.
        for(int i=0;i<3;++i) out.push_back(model_.parameters.mass*(a[i]-model_.parameters.gravity[i])-w[0]*s[3+3*i+2]-air[i]);
    }
    return out;
}
}
