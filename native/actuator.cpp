#include "actuator.hpp"
#include <algorithm>
#include <cmath>
#include <stdexcept>

namespace aerowhip {
namespace {
void check(bool ok,const char* message) { if(!ok) throw std::invalid_argument(message); }
Motors multiply(const std::array<double,16>& matrix,const Motors& value) {
    Motors out{};
    for(int row=0;row<4;++row) for(int col=0;col<4;++col) out[row]+=matrix[4*row+col]*value[col];
    return out;
}
}
Actuator::Actuator(const std::array<Vec3,4>& positions, Motors yaw, Motors low,
                   Motors high, Motors lag, Motors start)
    : minimum(low), maximum(high), response_s(lag), initial(start) {
    for(int i=0;i<4;++i) {
        for(double x:positions[i]) check(std::isfinite(x),"Motor positions must be finite");
        check(std::isfinite(yaw[i]),"Yaw torque-to-thrust coefficients must be finite");
        check(std::isfinite(low[i])&&std::isfinite(high[i])&&low[i]>=0&&high[i]>low[i],"Invalid motor thrust bounds");
        check(std::isfinite(lag[i])&&lag[i]>=0,"Motor response times must be finite and nonnegative");
        mixing_[i]=1.;mixing_[4+i]=positions[i][1];mixing_[8+i]=-positions[i][0];mixing_[12+i]=yaw[i];
    }
    validate(initial);
    // Row scaling makes the rank check independent of the N versus Nm units.
    double augmented[4][8]{}, scale[4]{};
    for(int r=0;r<4;++r) {
        for(int c=0;c<4;++c) scale[r]=std::max(scale[r],std::abs(mixing_[4*r+c]));
        check(scale[r]>0,"Motor geometry cannot generate all four wrench components");
        for(int c=0;c<4;++c) augmented[r][c]=mixing_[4*r+c]/scale[r];
        augmented[r][4+r]=1.;
    }
    for(int col=0;col<4;++col) {
        int pivot=col;
        for(int r=col+1;r<4;++r) if(std::abs(augmented[r][col])>std::abs(augmented[pivot][col])) pivot=r;
        check(std::abs(augmented[pivot][col])>1e-10,"Motor mixing matrix is singular or ill-conditioned");
        for(int c=0;c<8;++c) std::swap(augmented[col][c],augmented[pivot][c]);
        double divisor=augmented[col][col];
        for(int c=0;c<8;++c) augmented[col][c]/=divisor;
        for(int r=0;r<4;++r) if(r!=col) {
            double factor=augmented[r][col];
            for(int c=0;c<8;++c) augmented[r][c]-=factor*augmented[col][c];
        }
    }
    for(int r=0;r<4;++r) for(int c=0;c<4;++c) inverse_[4*r+c]=augmented[r][4+c]/scale[c];
}
void Actuator::validate(const Motors& forces) const {
    for(int i=0;i<4;++i) check(std::isfinite(forces[i])&&forces[i]>=minimum[i]&&forces[i]<=maximum[i],
                               "Motor forces must be finite and within configured bounds");
}
Motors Actuator::allocate(const Wrench& requested) const {
    for(double x:requested) check(std::isfinite(x),"Requested wrench must be finite");
    check(requested[0]>=0,"Requested collective thrust must be nonnegative");
    auto raw=multiply(inverse_,requested);
    for(double x:raw) check(std::isfinite(x),"Motor allocation overflow");
    return raw;
}
Motors Actuator::target(const Wrench& requested) const {
    auto out=allocate(requested);
    for(int i=0;i<4;++i) out[i]=std::clamp(out[i],minimum[i],maximum[i]);
    return out;
}
Motors Actuator::response(const Motors& start,const Motors& goal,double t) const {
    validate(start);validate(goal);check(std::isfinite(t)&&t>=0,"Invalid actuator response interval");
    Motors out;
    for(int i=0;i<4;++i) out[i]=response_s[i]==0 ? goal[i] :
        std::clamp(start[i]+(goal[i]-start[i])*(-std::expm1(-t/response_s[i])),minimum[i],maximum[i]);
    return out;
}
Wrench Actuator::wrench(const Motors& forces) const { validate(forces);return multiply(mixing_,forces); }
}
