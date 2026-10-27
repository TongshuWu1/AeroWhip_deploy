/* Test harness: drives the vendored firmware controller and power distribution with flat arrays. */
#include <string.h>
#include "controller_mellinger.h"
#include "power_distribution.h"
static controllerMellinger_t self;
void harness_init(const float* p) {
    controllerMellingerInit(&self);
    self.mass=p[0]; self.massThrust=p[1]; self.kp_xy=p[2]; self.kd_xy=p[3]; self.ki_xy=p[4]; self.i_range_xy=p[5];
    self.kp_z=p[6]; self.kd_z=p[7]; self.ki_z=p[8]; self.i_range_z=p[9]; self.kR_xy=p[10]; self.kw_xy=p[11];
    self.ki_m_xy=p[12]; self.i_range_m_xy=p[13]; self.kR_z=p[14]; self.kw_z=p[15]; self.ki_m_z=p[16]; self.i_range_m_z=p[17];
    self.kd_omega_rp=p[18];
}
/* in: sp_pos(3) sp_vel(3) sp_acc(3) sp_yaw_deg pos(3) vel(3) quat xyzw(4) gyro(3)  -> out: thrust roll pitch yaw pwm(4) capped */
void harness_step(const float* in, float* out) {
    setpoint_t sp; state_t st; sensorData_t se; control_t c; motors_thrust_uncapped_t u; motors_thrust_pwm_t pwm;
    memset(&sp,0,sizeof sp); memset(&st,0,sizeof st); memset(&se,0,sizeof se); memset(&c,0,sizeof c);
    sp.position.x=in[0]; sp.position.y=in[1]; sp.position.z=in[2];
    sp.velocity.x=in[3]; sp.velocity.y=in[4]; sp.velocity.z=in[5];
    sp.acceleration.x=in[6]; sp.acceleration.y=in[7]; sp.acceleration.z=in[8];
    sp.attitude.yaw=in[9]; sp.mode.x=sp.mode.y=sp.mode.z=modeAbs; sp.mode.yaw=modeAbs;
    st.position.x=in[10]; st.position.y=in[11]; st.position.z=in[12];
    st.velocity.x=in[13]; st.velocity.y=in[14]; st.velocity.z=in[15];
    st.attitudeQuaternion.x=in[16]; st.attitudeQuaternion.y=in[17]; st.attitudeQuaternion.z=in[18]; st.attitudeQuaternion.w=in[19];
    se.gyro.x=in[20]; se.gyro.y=in[21]; se.gyro.z=in[22];
    controllerMellinger(&self,&c,&sp,&se,&st,0);    /* step 0: RATE_DO_EXECUTE true */
    powerDistribution(&c,&u);
    int capped=powerDistributionCap(&u,&pwm);
    out[0]=c.thrust; out[1]=c.roll; out[2]=c.pitch; out[3]=c.yaw;
    out[4]=pwm.motors.m1; out[5]=pwm.motors.m2; out[6]=pwm.motors.m3; out[7]=pwm.motors.m4; out[8]=capped;
}
