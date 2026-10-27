#pragma once
// Line-by-line port of the Crazyflie firmware's Mellinger controller and legacy power
// distribution (bitcraze/crazyflie-firmware, master b48aba18, 2026-10-28, MIT), in float as the
// firmware computes it. The vendored C sources in third_party/crazyflie-firmware are compiled
// into a test harness and must agree with this port (tests/test_firmware_port.py).
//
// Scope: the position-control branch of controllerMellinger() (setpoint modes x/y/z/yaw Abs,
// attitude rates zero), powerDistributionLegacy() and powerDistributionCap(). Battery
// compensation (CONFIG_ENABLE_THRUST_BAT_COMPENSATED) is assumed off: the identified voltage
// dependence of the delivered thrust is consistent with no onboard compensation.
#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>

namespace aerowhip {
namespace mellinger {
constexpr float PI_F=3.14159265358979323846f;
constexpr float GRAVITY_MAGNITUDE=9.81f;      // physicalConstants.h
constexpr float ATTITUDE_RATE=500.f;          // stabilizer_types.h: RATE_500_HZ
inline float radians(float d) {return (PI_F/180.0f)*d;}
inline float fsqr(float x) {return x*x;}
inline float clamp(float v,float lo,float hi) {if(v<lo)return lo;if(v>hi)return hi;return v;}
struct vec {float x,y,z;};
inline vec mkvec(float x,float y,float z) {return {x,y,z};}
inline vec vsub(vec a,vec b) {return {a.x-b.x,a.y-b.y,a.z-b.z};}
inline float vdot(vec a,vec b) {return a.x*b.x+a.y*b.y+a.z*b.z;}
inline vec vscl(float s,vec v) {return {s*v.x,s*v.y,s*v.z};}
inline vec vdiv(vec v,float s) {return vscl(1.0f/s,v);}
inline float vmag(vec v) {return sqrtf(vdot(v,v));}
inline vec vnormalize(vec v) {return vdiv(v,vmag(v));}
inline vec vcross(vec a,vec b) {return {a.y*b.z-a.z*b.y,a.z*b.x-a.x*b.z,a.x*b.y-a.y*b.x};}

struct Parameters {          // ctrlMel.* (firmware defaults; the flight YAML overrides)
    float mass=0.029f, massThrust=132000.f;
    float kp_xy=0.4f, kd_xy=0.2f, ki_xy=0.05f, i_range_xy=2.0f;
    float kp_z=1.25f, kd_z=0.4f, ki_z=0.05f, i_range_z=0.4f;
    float kR_xy=70000.f, kw_xy=20000.f, ki_m_xy=0.f, i_range_m_xy=1.0f;
    float kR_z=60000.f, kw_z=12000.f, ki_m_z=500.f, i_range_m_z=1500.f;
    float kd_omega_rp=200.f;
};
struct Input {
    vec setpoint_position{}, setpoint_velocity{}, setpoint_acceleration{};
    float setpoint_yaw_deg=0.f;                 // mode.yaw Abs
    vec position{}, velocity{};                 // state estimate, m, m/s
    float qx=0.f,qy=0.f,qz=0.f,qw=1.f;          // attitudeQuaternion
    vec gyro{};                                 // deg/s, sensor frame as the firmware reads it
};
struct Output {
    float thrust=0.f;                           // massThrust units (PWM scale)
    int16_t roll=0,pitch=0,yaw=0;               // legacy int16 outputs
    std::array<uint16_t,4> pwm{};               // after legacy mixing and capping
    bool capped=false;
};

class Controller {
public:
    explicit Controller(Parameters p={}) : p_(p) {}
    const Parameters& parameters() const {return p_;}
    void reset() {i_error_x=i_error_y=i_error_z=0.f;i_error_m_x=i_error_m_y=i_error_m_z=0.f;}
    // controllerMellinger() position branch + powerDistributionLegacy() + powerDistributionCap().
    Output step(const Input& in) {
        const float dt=(float)(1.0f/ATTITUDE_RATE);
        vec setpointPos=in.setpoint_position, setpointVel=in.setpoint_velocity;
        vec statePos=in.position, stateVel=in.velocity;
        float x=in.qx,y=in.qy,z=in.qz,w=in.qw;
        // z_axis = mcolumn(quat2rotmat(q), 2)
        vec z_axis=mkvec(2*x*z+2*y*w, 2*y*z-2*x*w, 1-2*x*x-2*y*y);
        float desired_yaw=radians(in.setpoint_yaw_deg);
        vec r_error=vsub(setpointPos,statePos), v_error=vsub(setpointVel,stateVel);
        i_error_z+=r_error.z*dt; i_error_z=clamp(i_error_z,-p_.i_range_z,p_.i_range_z);
        i_error_x+=r_error.x*dt; i_error_x=clamp(i_error_x,-p_.i_range_xy,p_.i_range_xy);
        i_error_y+=r_error.y*dt; i_error_y=clamp(i_error_y,-p_.i_range_xy,p_.i_range_xy);
        vec target_thrust;
        target_thrust.x=p_.mass*in.setpoint_acceleration.x                   +p_.kp_xy*r_error.x+p_.kd_xy*v_error.x+p_.ki_xy*i_error_x;
        target_thrust.y=p_.mass*in.setpoint_acceleration.y                   +p_.kp_xy*r_error.y+p_.kd_xy*v_error.y+p_.ki_xy*i_error_y;
        target_thrust.z=p_.mass*(in.setpoint_acceleration.z+GRAVITY_MAGNITUDE)+p_.kp_z*r_error.z+p_.kd_z*v_error.z+p_.ki_z*i_error_z;
        float thrust=p_.massThrust*vdot(target_thrust,z_axis);
        z_axis_desired=vnormalize(target_thrust);
        vec x_c_des=mkvec(cosf(desired_yaw),sinf(desired_yaw),0);
        vec y_axis_desired=vnormalize(vcross(z_axis_desired,x_c_des));
        vec x_axis_desired=vcross(y_axis_desired,z_axis_desired);
        vec eR,ew,M;
        eR.x = (-1 + 2*fsqr(x) + 2*fsqr(y))*y_axis_desired.z + z_axis_desired.y - 2*(x*y_axis_desired.x*z + y*y_axis_desired.y*z - x*y*z_axis_desired.x + fsqr(x)*z_axis_desired.y + fsqr(z)*z_axis_desired.y - y*z*z_axis_desired.z) +    2*w*(-(y*y_axis_desired.x) - z*z_axis_desired.x + x*(y_axis_desired.y + z_axis_desired.z));
        eR.y = x_axis_desired.z - z_axis_desired.x - 2*(fsqr(x)*x_axis_desired.z + y*(x_axis_desired.z*y - x_axis_desired.y*z) - (fsqr(y) + fsqr(z))*z_axis_desired.x + x*(-(x_axis_desired.x*z) + y*z_axis_desired.y + z*z_axis_desired.z) + w*(x*x_axis_desired.y + z*z_axis_desired.y - y*(x_axis_desired.x + z_axis_desired.z)));
        eR.z = y_axis_desired.x - 2*(y*(x*x_axis_desired.x + y*y_axis_desired.x - x*y_axis_desired.y) + w*(x*x_axis_desired.z + y*y_axis_desired.z)) + 2*(-(x_axis_desired.z*y) + w*(x_axis_desired.x + y_axis_desired.y) + x*y_axis_desired.z)*z - 2*y_axis_desired.x*fsqr(z) + x_axis_desired.y*(-1 + 2*fsqr(x) + 2*fsqr(z));
        eR.y=-eR.y;                                   // Crazyflie coordinate system
        float err_d_roll=0, err_d_pitch=0;
        float stateAttitudeRateRoll=radians(in.gyro.x), stateAttitudeRatePitch=-radians(in.gyro.y), stateAttitudeRateYaw=radians(in.gyro.z);
        ew.x=radians(0.f)-stateAttitudeRateRoll;      // attitudeRate setpoints are zero here
        ew.y=-radians(0.f)-stateAttitudeRatePitch;
        ew.z=radians(0.f)-stateAttitudeRateYaw;
        if(prev_omega_roll==prev_omega_roll) {
            err_d_roll=((radians(0.f)-prev_setpoint_omega_roll)-(stateAttitudeRateRoll-prev_omega_roll))/dt;
            err_d_pitch=(-(radians(0.f)-prev_setpoint_omega_pitch)-(stateAttitudeRatePitch-prev_omega_pitch))/dt;
        }
        prev_omega_roll=stateAttitudeRateRoll; prev_omega_pitch=stateAttitudeRatePitch;
        prev_setpoint_omega_roll=radians(0.f); prev_setpoint_omega_pitch=radians(0.f);
        i_error_m_x+=(-eR.x)*dt; i_error_m_x=clamp(i_error_m_x,-p_.i_range_m_xy,p_.i_range_m_xy);
        i_error_m_y+=(-eR.y)*dt; i_error_m_y=clamp(i_error_m_y,-p_.i_range_m_xy,p_.i_range_m_xy);
        i_error_m_z+=(-eR.z)*dt; i_error_m_z=clamp(i_error_m_z,-p_.i_range_m_z,p_.i_range_m_z);
        M.x=-p_.kR_xy*eR.x+p_.kw_xy*ew.x+p_.ki_m_xy*i_error_m_x+p_.kd_omega_rp*err_d_roll;
        M.y=-p_.kR_xy*eR.y+p_.kw_xy*ew.y+p_.ki_m_xy*i_error_m_y+p_.kd_omega_rp*err_d_pitch;
        M.z=-p_.kR_z*eR.z+p_.kw_z*ew.z+p_.ki_m_z*i_error_m_z;
        Output out; out.thrust=thrust;
        if(thrust>0) {
            out.roll=(int16_t)clamp(M.x,-32000,32000);
            out.pitch=(int16_t)clamp(M.y,-32000,32000);
            out.yaw=(int16_t)clamp(-M.z,-32000,32000);
        } else {
            out.roll=out.pitch=out.yaw=0; reset();
        }
        // powerDistributionLegacy(): int16 half torques, float sums stored as int32
        int16_t r=(int16_t)(out.roll/2.0f), p=(int16_t)(out.pitch/2.0f);
        std::array<int32_t,4> uncapped{(int32_t)(out.thrust-r+p+out.yaw),(int32_t)(out.thrust-r-p-out.yaw),
                                       (int32_t)(out.thrust+r-p+out.yaw),(int32_t)(out.thrust+r+p-out.yaw)};
        // powerDistributionCap(): common reduction above UINT16_MAX, idle thrust 0 (autoarming build)
        const int32_t maxAllowed=65535; int32_t highest=0;
        for(int32_t v:uncapped) if(v>highest) highest=v;
        int32_t reduction=0; if(highest>maxAllowed){reduction=highest-maxAllowed;out.capped=true;}
        for(int i=0;i<4;++i){int32_t v=uncapped[i]-reduction; out.pwm[i]=(uint16_t)(v<0?0:v);}   // capMinThrust(float, 0)
        return out;
    }
    vec z_axis_desired{};
    float i_error_x=0,i_error_y=0,i_error_z=0;
    float prev_omega_roll=0,prev_omega_pitch=0,prev_setpoint_omega_roll=0,prev_setpoint_omega_pitch=0;
    float i_error_m_x=0,i_error_m_y=0,i_error_m_z=0;
private:
    Parameters p_;
};
}
}
