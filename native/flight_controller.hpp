#pragma once
// The real flight pipeline below the host: the Crazyflie Mellinger controller with its
// translation gains zeroed (acceleration-only), its open-loop thrust map, and the attitude
// loop. Identified on the 2026-10-27 drone-only calibration flights; not a firmware port.
//
//   firmware thrust vector   F = m_fw (a_sent + g z)          (a_sent: world, trims included)
//   desired body z           z_d = F/|F|, heading from the commanded yaw
//   firmware thrust          T_cmd = F . z_body               (Mellinger projection)
//   delivered thrust         T = kappa (1 + beta (V - V0)) T_cmd^p   (motor/propeller map)
//   attitude                 alpha = kR e_R - kW omega, torque = J alpha + omega x J omega
//   disturbance              b: a constant world acceleration added to a_sent (domain randomization;
//                            the replay residual is a slowly varying bias of about 0.1 m/s^2)
#include "engine.hpp"
#include <cmath>
#include <stdexcept>

namespace aerowhip {
struct FlightControllerParameters {
    double firmware_mass=0.144;          // ctrlMel.mass, kg
    double thrust_scale=0.;              // kappa, N^(1-p)
    double thrust_exponent=1.;           // p
    double voltage_slope=0.;             // beta, 1/V
    double reference_voltage=7.2;        // V0, V (battery under load)
    double tilt_gain=1.;                 // horizontal thrust-direction gain (1 = none)
    Vec3 attitude_stiffness{}, attitude_damping{};   // kR = w^2, kW = 2 zeta w (per body axis)
    Mat3 inertia{};
    Vec3 com{};                          // tracking origin to COM, body (unused by the law)
    Vec3 gravity{0.,0.,-9.80665};
};

class FlightController {
public:
    explicit FlightController(FlightControllerParameters p,double battery_voltage)
        : p_(p),voltage_(battery_voltage) {
        auto finite=[](double x){return std::isfinite(x);};
        if(!(finite(p_.firmware_mass)&&p_.firmware_mass>0.&&finite(p_.thrust_scale)&&p_.thrust_scale>0.
             &&finite(p_.thrust_exponent)&&p_.thrust_exponent>0.&&finite(p_.voltage_slope)&&finite(p_.reference_voltage)
             &&finite(p_.tilt_gain)&&p_.tilt_gain>0.))
            throw std::invalid_argument("Invalid flight controller parameters");
        set_voltage(battery_voltage);
    }
    void set_voltage(double v) {
        if(!std::isfinite(v)||v<=0.) throw std::invalid_argument("Battery voltage must be positive");
        voltage_=v;
    }
    double voltage() const {return voltage_;}
    void set_bias(Vec3 b) {for(double x:b) if(!std::isfinite(x)) throw std::invalid_argument("Bias must be finite"); bias_=b;}
    Vec3 bias() const {return bias_;}
    const FlightControllerParameters& parameters() const {return p_;}
    double delivered_thrust(double firmware_thrust) const {
        double efficiency=p_.thrust_scale*(1.+p_.voltage_slope*(voltage_-p_.reference_voltage));
        return efficiency*std::pow(std::max(firmware_thrust,0.),p_.thrust_exponent);
    }
    Wrench operator()(const Vector& s,const Command& c) const {
        if(s.size()<18||(c.size()!=11&&c.size()!=13)) throw std::invalid_argument("Flight controller state/command shape");
        Mat3 R{};std::copy(s.begin()+3,s.begin()+12,R.begin());
        Vec3 omega{s[15],s[16],s[17]};
        Vec3 F{};for(int i=0;i<3;++i)F[i]=p_.firmware_mass*(c[6+i]+bias_[i]-p_.gravity[i]);
        Vec3 direction{F[0]*p_.tilt_gain,F[1]*p_.tilt_gain,F[2]};
        double size=std::sqrt(direction[0]*direction[0]+direction[1]*direction[1]+direction[2]*direction[2]);
        if(!(size>1e-9)) throw std::runtime_error("Firmware thrust direction undefined at zero force");
        Vec3 z{direction[0]/size,direction[1]/size,direction[2]/size};
        Vec3 heading{std::cos(c[9]),std::sin(c[9]),0.};
        Vec3 y{z[1]*heading[2]-z[2]*heading[1],z[2]*heading[0]-z[0]*heading[2],z[0]*heading[1]-z[1]*heading[0]};
        double length=std::sqrt(y[0]*y[0]+y[1]*y[1]+y[2]*y[2]);
        if(!(length>1e-8)) throw std::runtime_error("Thrust direction parallel to the yaw heading");
        for(double& v:y) v/=length;
        Vec3 x{y[1]*z[2]-y[2]*z[1],y[2]*z[0]-y[0]*z[2],y[0]*z[1]-y[1]*z[0]};
        Mat3 D{x[0],y[0],z[0],x[1],y[1],z[1],x[2],y[2],z[2]};   // columns x, y, z
        // e_R = vee(R^T D - D^T R)/2, sign convention of the geometric controller.
        Mat3 A{},B{};
        for(int i=0;i<3;++i)for(int j=0;j<3;++j){double a=0.,b=0.;for(int k=0;k<3;++k){a+=R[3*k+i]*D[3*k+j];b+=D[3*k+i]*R[3*k+j];}A[3*i+j]=a;B[3*i+j]=b;}
        Vec3 e{.5*(A[7]-B[7]),.5*(A[2]-B[2]),.5*(A[3]-B[3])};
        Vec3 alpha{};for(int i=0;i<3;++i)alpha[i]=p_.attitude_stiffness[i]*e[i]-p_.attitude_damping[i]*omega[i];
        Vec3 Jw{},Ja{};
        for(int i=0;i<3;++i)for(int j=0;j<3;++j){Jw[i]+=p_.inertia[3*i+j]*omega[j];Ja[i]+=p_.inertia[3*i+j]*alpha[j];}
        Vec3 gyro{omega[1]*Jw[2]-omega[2]*Jw[1],omega[2]*Jw[0]-omega[0]*Jw[2],omega[0]*Jw[1]-omega[1]*Jw[0]};
        double firmware_thrust=F[0]*R[2]+F[1]*R[5]+F[2]*R[8];
        return {delivered_thrust(firmware_thrust),Ja[0]+gyro[0],Ja[1]+gyro[1],Ja[2]+gyro[2]};
    }
private:
    FlightControllerParameters p_;
    double voltage_;
    Vec3 bias_{};
};
}
