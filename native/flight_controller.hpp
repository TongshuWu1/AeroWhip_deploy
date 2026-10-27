#pragma once
// The real flight pipeline below the host. Two laws:
//
//   firmware    the ported Crazyflie Mellinger controller (firmware.hpp): the host's acceleration
//               command enters as the firmware setpoint, the firmware computes thrust and legacy
//               roll/pitch/yaw, the legacy mixer and cap give four PWM values, and the motor map
//               turns PWM into newtons per motor. Runs on the state the firmware would estimate.
//   identified  the reduced model identified on the 2026-10-27 drone-only flights (kept as the
//               fallback): firmware thrust vector F = m_fw (a_sent + g z), desired body z = F/|F|,
//               attitude alpha = kR e_R - kW omega, torque = J alpha + omega x J omega.
//
// Both share the delivered-thrust map T = kappa (1 + beta (V - V0)) T_cmd^p identified on the same
// flights (per motor in firmware mode: T_i = kappa'/4 (pwm_i / massThrust)^p, which reproduces the
// collective map when the four motors are equal), and a constant world acceleration bias b added to
// a_sent (domain randomization; the replay residual is a slowly varying bias of about 0.1 m/s^2).
#include "engine.hpp"
#include "firmware.hpp"
#include <cmath>
#include <stdexcept>

namespace aerowhip {
enum class FlightLaw {identified, firmware};
struct FlightControllerParameters {
    FlightLaw law=FlightLaw::identified;
    double firmware_mass=0.144;          // ctrlMel.mass, kg
    double thrust_scale=0.;              // kappa, N^(1-p)
    double thrust_exponent=1.;           // p
    double voltage_slope=0.;             // beta, 1/V
    double reference_voltage=7.2;        // V0, V (battery under load)
    double tilt_gain=1.;                 // horizontal thrust-direction gain (1 = none; identified law)
    Vec3 attitude_stiffness{}, attitude_damping{};   // identified law: kR = w^2, kW = 2 zeta w (per body axis)
    Mat3 inertia{};
    Vec3 com{};                          // tracking origin to COM, body (unused by the laws)
    Vec3 gravity{0.,0.,-9.80665};
    mellinger::Parameters firmware;      // ctrlMel.* of the flight
    std::array<double,16> mixing{};      // motor forces -> wrench (Actuator::mixing), firmware law
    Motors motor_minimum{}, motor_maximum{};
};

class FlightController {
public:
    explicit FlightController(FlightControllerParameters p,double battery_voltage)
        : p_(p),voltage_(battery_voltage),firmware_(p.firmware) {
        auto finite=[](double x){return std::isfinite(x);};
        if(!(finite(p_.firmware_mass)&&p_.firmware_mass>0.&&finite(p_.thrust_scale)&&p_.thrust_scale>0.
             &&finite(p_.thrust_exponent)&&p_.thrust_exponent>0.&&finite(p_.voltage_slope)&&finite(p_.reference_voltage)
             &&finite(p_.tilt_gain)&&p_.tilt_gain>0.))
            throw std::invalid_argument("Invalid flight controller parameters");
        if(p_.law==FlightLaw::firmware&&!(p_.firmware.massThrust>0.f)) throw std::invalid_argument("massThrust must be positive");
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
    FlightLaw law() const {return p_.law;}
    void reset() {firmware_=mellinger::Controller(p_.firmware);}
    const mellinger::Controller& firmware() const {return firmware_;}
    double efficiency() const {return p_.thrust_scale*(1.+p_.voltage_slope*(voltage_-p_.reference_voltage));}
    double delivered_thrust(double firmware_thrust) const {
        return efficiency()*std::pow(std::max(firmware_thrust,0.),p_.thrust_exponent);
    }
    // One controller update (advances the firmware's integrators and derivative memory).
    Wrench step(const Vector& s,const Command& c) {return evaluate(s,c,firmware_);}
    // The wrench this state and command would produce, without advancing any memory.
    Wrench peek(const Vector& s,const Command& c) const {auto copy=firmware_;return evaluate(s,c,copy);}
    Wrench operator()(const Vector& s,const Command& c) const {return peek(s,c);}
    // Motor forces (N) the firmware law commands for a state/command, without advancing memory.
    Motors motors(const Vector& s,const Command& c) const {auto copy=firmware_;return firmware_motors(s,c,copy);}
private:
    static void check_shape(const Vector& s,const Command& c) {
        if(s.size()<18||(c.size()!=11&&c.size()!=13)) throw std::invalid_argument("Flight controller state/command shape");
    }
    Wrench evaluate(const Vector& s,const Command& c,mellinger::Controller& fw) const {
        check_shape(s,c);
        if(p_.law==FlightLaw::firmware) {
            Motors f=firmware_motors(s,c,fw);
            Wrench w{};for(int r=0;r<4;++r)for(int k=0;k<4;++k)w[r]+=p_.mixing[4*r+k]*f[k];
            return w;
        }
        return identified(s,c);
    }
    Motors firmware_motors(const Vector& s,const Command& c,mellinger::Controller& fw) const {
        check_shape(s,c);
        Mat3 R{};std::copy(s.begin()+3,s.begin()+12,R.begin());
        mellinger::Input in;
        // The firmware's estimate: the simulated state (position/velocity setpoints equal the state,
        // the translation gains are zero in flight anyway).
        in.position=in.setpoint_position={float(s[0]),float(s[1]),float(s[2])};
        in.velocity=in.setpoint_velocity={float(s[12]),float(s[13]),float(s[14])};
        in.setpoint_acceleration={float(c[6]+bias_[0]),float(c[7]+bias_[1]),float(c[8]+bias_[2])};
        in.setpoint_yaw_deg=float(c[9]*180./M_PI);
        // Body-to-world rotation (row-major) to the firmware's xyzw quaternion.
        double tr=R[0]+R[4]+R[8],qx,qy,qz,qw;
        if(tr>0.){double S=std::sqrt(tr+1.)*2.;qw=.25*S;qx=(R[7]-R[5])/S;qy=(R[2]-R[6])/S;qz=(R[3]-R[1])/S;}
        else if(R[0]>R[4]&&R[0]>R[8]){double S=std::sqrt(1.+R[0]-R[4]-R[8])*2.;qw=(R[7]-R[5])/S;qx=.25*S;qy=(R[1]+R[3])/S;qz=(R[2]+R[6])/S;}
        else if(R[4]>R[8]){double S=std::sqrt(1.+R[4]-R[0]-R[8])*2.;qw=(R[2]-R[6])/S;qx=(R[1]+R[3])/S;qy=.25*S;qz=(R[5]+R[7])/S;}
        else{double S=std::sqrt(1.+R[8]-R[0]-R[4])*2.;qw=(R[3]-R[1])/S;qx=(R[2]+R[6])/S;qy=(R[5]+R[7])/S;qz=.25*S;}
        in.qx=float(qx);in.qy=float(qy);in.qz=float(qz);in.qw=float(qw);
        const double deg=180./M_PI;
        in.gyro={float(s[15]*deg),float(s[16]*deg),float(s[17]*deg)};
        auto out=fw.step(in);
        Motors f{};
        double k=efficiency()/4.;
        for(int i=0;i<4;++i){
            // control->thrust is the per-motor PWM level (the legacy mixer gives it to all four), so
            // pwm/massThrust is the collective-equivalent command in newtons.
            double command=double(out.pwm[i])/double(p_.firmware.massThrust);
            f[i]=std::clamp(k*std::pow(command,p_.thrust_exponent),p_.motor_minimum[i],p_.motor_maximum[i]);
        }
        return f;
    }
    Wrench identified(const Vector& s,const Command& c) const {
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
    FlightControllerParameters p_;
    double voltage_;
    Vec3 bias_{};
    mellinger::Controller firmware_;
};
}
