#include "engine.hpp"
#include "actuator.hpp"

#include <algorithm>
#include <cmath>
#include <limits>
#include <stdexcept>
#include <utility>
#include <numeric>

namespace aerowhip {
namespace {
struct Guard {
    std::atomic_flag& flag;
    explicit Guard(std::atomic_flag& value) : flag(value) {
        if (flag.test_and_set()) throw std::runtime_error("Engine is busy; do not access one engine concurrently");
    }
    ~Guard() { flag.clear(); }
};
void require(bool condition, const char* message) {
    if (!condition) throw std::invalid_argument(message);
}
template<class Values> bool finite(const Values& values) {
    for (double v : values) if (!std::isfinite(v)) return false;
    return true;
}
Vec3 at(const Vector& x, size_t k) { return {x[k], x[k+1], x[k+2]}; }
void put(Vector& x, size_t k, Vec3 v) { std::copy(v.begin(), v.end(), x.begin()+k); }
Vec3 operator+(Vec3 a, Vec3 b) { for (int i=0;i<3;++i) a[i]+=b[i]; return a; }
Vec3 operator-(Vec3 a, Vec3 b) { for (int i=0;i<3;++i) a[i]-=b[i]; return a; }
Vec3 operator*(Vec3 a, double s) { for (double& v:a) v*=s; return a; }
Vec3 operator/(Vec3 a, double s) { for (double& v:a) v/=s; return a; }
double dot(Vec3 a, Vec3 b) { return (a[0]*b[0]+a[1]*b[1])+a[2]*b[2]; }
double norm(Vec3 a) { return std::sqrt(dot(a,a)); }
Vec3 cross(Vec3 a, Vec3 b) {
    return {a[1]*b[2]-a[2]*b[1], a[2]*b[0]-a[0]*b[2], a[0]*b[1]-a[1]*b[0]};
}
Mat3 rotation(const Vector& s) { Mat3 R; std::copy_n(s.begin()+3,9,R.begin()); return R; }
Mat3 identity() { return {1.,0.,0.,0.,1.,0.,0.,0.,1.}; }
Mat3 skew(Vec3 v) { return {0.,-v[2],v[1],v[2],0.,-v[0],-v[1],v[0],0.}; }
Vec3 mv(const Mat3& A, Vec3 v) {
    Vec3 y{};
    for(int i=0;i<3;++i) for(int j=0;j<3;++j) y[i]+=A[3*i+j]*v[j];
    return y;
}
Mat3 transpose(const Mat3& A) {
    Mat3 B{}; for(int i=0;i<3;++i) for(int j=0;j<3;++j) B[3*i+j]=A[3*j+i]; return B;
}
Mat3 mm(const Mat3& A, const Mat3& B) {
    Mat3 C{};
    for(int i=0;i<3;++i) for(int j=0;j<3;++j)
        for(int k=0;k<3;++k) C[3*i+j]+=A[3*i+k]*B[3*k+j];
    return C;
}
Mat3 inverse(Mat3 A) {
    Mat3 B=identity();
    for(int j=0;j<3;++j) {
        int pivot=j;
        for(int i=j+1;i<3;++i) if(std::abs(A[3*i+j])>std::abs(A[3*pivot+j])) pivot=i;
        require(std::isfinite(A[3*pivot+j]) && A[3*pivot+j]!=0., "Singular body inertia");
        for(int k=0;k<3;++k) { std::swap(A[3*j+k],A[3*pivot+k]); std::swap(B[3*j+k],B[3*pivot+k]); }
        double d=A[3*j+j];
        for(int k=0;k<3;++k) { A[3*j+k]/=d; B[3*j+k]/=d; }
        for(int i=0;i<3;++i) if(i!=j) {
            double f=A[3*i+j];
            for(int k=0;k<3;++k) { A[3*i+k]-=f*A[3*j+k]; B[3*i+k]-=f*B[3*j+k]; }
        }
    }
    return B;
}
double determinant(const Mat3& A) {
    return A[0]*(A[4]*A[8]-A[5]*A[7])-A[1]*(A[3]*A[8]-A[5]*A[6])+A[2]*(A[3]*A[7]-A[4]*A[6]);
}
Mat3 exp_rotation(Vec3 theta) {
    double angle=norm(theta), half=.5*angle;
    double a=angle==0.?1.:std::sin(angle)/angle;
    double sh=half==0.?1.:std::sin(half)/half;
    double b=.5*sh*sh;
    Mat3 K=skew(theta), K2=mm(K,K), R=identity();
    for(int i=0;i<9;++i) R[i]+=a*K[i]+b*K2[i];
    return R;
}
// Principal SO(3) logarithm; a near-pi step is invalid, not clipped.
Vec3 log_rotation(const Mat3& R) {
    Vec3 z{.5*(R[7]-R[5]),.5*(R[2]-R[6]),.5*(R[3]-R[1])};
    double sine=norm(z),cosine=.5*(R[0]+R[4]+R[8]-1.),angle=std::atan2(sine,cosine);
    require(angle<3.14159265358979323846-1e-6,"BDF2 rotation history crosses the principal-log boundary");
    return z*(sine<1e-14?1.:angle/sine);
}
struct BdfCoefficients { double memory, beta; };
BdfCoefficients bdf_coefficients(double h,double previous) {
    // Explicit order restart for startup or growth >2 after an event fragment.
    // No retry, tolerance relaxation or adaptive physical substepping.
    if(previous<=0. || h>2.*previous*(1.+1e-12))return {0.,h};
    double r=h/previous;return {r*r/(1.+2.*r),h*(1.+r)/(1.+2.*r)};
}
Vector velocity(const Vector& s, size_t m) {
    Vector u(6+3*m); std::copy_n(s.begin()+12,6,u.begin());
    std::copy_n(s.begin()+18+3*m,3*m,u.begin()+6); return u;
}
void with_velocity(Vector& s, const Vector& u, size_t m) {
    std::copy_n(u.begin(),6,s.begin()+12); std::copy_n(u.begin()+6,3*m,s.begin()+18+3*m);
}
Vector displaced(const Vector& s, const Vector& d, size_t m) {
    Vector out=s; put(out,0,at(s,0)+at(d,0));
    Mat3 R=mm(rotation(s),exp_rotation(at(d,3))); std::copy(R.begin(),R.end(),out.begin()+3);
    for(size_t j=0;j<3*m;++j) out[18+j]+=d[6+j]; return out;
}
Vector scaled(const Vector& x, double f) { Vector y=x; for(double& v:y) v*=f; return y; }
Vector add_scaled(const Vector& x, const Vector& y, double f) {
    Vector z=x; for(size_t i=0;i<z.size();++i) z[i]+=f*y[i]; return z;
}
void cable(const Parameters& p, const Vector& s, std::vector<Vec3>& q, std::vector<Vec3>& v) {
    size_t m=p.rest.size(); q.resize(m+1); v.resize(m+1);
    Mat3 R=rotation(s);
    q[0]=at(s,0)+mv(R,p.arm); v[0]=at(s,12)+mv(R,cross(at(s,15),p.arm));
    for(size_t i=0;i<m;++i) { q[i+1]=at(s,18+3*i); v[i+1]=at(s,18+3*m+3*i); }
}
struct Boundary {
    Mat3 B, coupling;
};
Boundary boundary(const Parameters& p, const Vector& s) {
    Boundary b{mm(rotation(s),skew(p.arm)),{}};
    for(int k=0;k<9;++k) { b.B[k]=-b.B[k]; b.coupling[k]=p.node_mass[0]*b.B[k]; }
    return b;
}
Vector body_solve(const Boundary& b, const Mat3& inv, double mass, Vec3 f, Vec3 torque) {
    Vec3 translation=f/mass;
    Vec3 angular=mv(inv,torque-mv(transpose(b.coupling),translation));
    translation=translation-mv(b.coupling,angular)/mass;
    Vector out(6); put(out,0,translation); put(out,3,angular); return out;
}
Vec3 boundary_product(const Boundary& b, const Vector& u) { return at(u,0)+mv(b.B,at(u,3)); }
struct Chain {
    Vector lengths, diagonal, off, root_response;
    std::vector<Vec3> tangent;
};
Chain chain(const Parameters& p, const std::vector<Vec3>& q, const Boundary& b,
            const Mat3& inv, double mass) {
    size_t m=p.rest.size(); Chain c{Vector(m),Vector(m),Vector(m-1),{},std::vector<Vec3>(m)};
    for(size_t i=0;i<m;++i) {
        Vec3 edge=q[i+1]-q[i]; c.lengths[i]=norm(edge);
        require(std::isfinite(c.lengths[i]) && c.lengths[i]>1e-12,"Collapsed/nonfinite cable edge; DER geometry is invalid");
        c.tangent[i]=edge/c.lengths[i];
        if(i) require(1.+dot(c.tangent[i-1],c.tangent[i])>1e-12,
                      "Antiparallel interior cable edges; DER curvature is singular");
    }
    Vec3 root_f=c.tangent[0]*-1., root_torque=mv(transpose(b.B),root_f);
    c.root_response=body_solve(b,inv,mass,root_f,root_torque);
    for(size_t i=0;i<m;++i) {
        Vec3 t=c.tangent[i];
        c.diagonal[i]=dot(t,t/p.node_mass[i+1]);
        if(i) c.diagonal[i]+=dot(t,t/p.node_mass[i]);
        else c.diagonal[i]+=dot(root_f,at(c.root_response,0))+dot(root_torque,at(c.root_response,3));
        if(i+1<m) c.off[i]=-dot(t,c.tangent[i+1]/p.node_mass[i+1]);
    }
    return c;
}
Vector solve(const Chain& c, Vector rhs) {
    Vector pivots=c.diagonal;
    for(size_t i=0;i<pivots.size();++i) {
        if(i) { double r=c.off[i-1]/pivots[i-1]; pivots[i]-=r*c.off[i-1]; rhs[i]-=r*rhs[i-1]; }
        if(!(pivots[i]>0.) || !std::isfinite(pivots[i])) throw std::runtime_error("Invalid constraint pivot");
    }
    rhs.back()/=pivots.back();
    for(size_t i=rhs.size()-1;i-->0;) rhs[i]=(rhs[i]-c.off[i]*rhs[i+1])/pivots[i];
    return rhs;
}
Vector constraint_product(const Chain& c, const Boundary& b, const Vector& u) {
    Vector out(c.lengths.size()); Vec3 left=boundary_product(b,u);
    for(size_t i=0;i<out.size();++i) { Vec3 right=at(u,6+3*i); out[i]=dot(c.tangent[i],right-left); left=right; }
    return out;
}
Vector impulse(const Parameters& p, const Chain& c, const Vector& lambda) {
    size_t m=p.rest.size(); Vector out(6+3*m);
    for(int j=0;j<6;++j) out[j]=c.root_response[j]*lambda[0];
    for(size_t i=0;i<m;++i) {
        Vec3 f=c.tangent[i]*lambda[i]; if(i+1<m) f=f-c.tangent[i+1]*lambda[i+1];
        put(out,6+3*i,f/p.node_mass[i+1]);
    }
    return out;
}
Vec3 drone_body_drag_acceleration(const Parameters& p,const Vector& s) {
    auto R=rotation(s);auto v=mv(transpose(R),at(s,12));
    return mv(R,Vec3{-p.body_drag[0]*v[0],-p.body_drag[1]*v[1],-p.body_drag[2]*v[2]});
}
void validate_wrench(const Wrench& w) {
    require(finite(w) && w[0]>=0., "Wrench must be finite: nonnegative collective N, body XYZ Nm");
}
struct CableForces { std::vector<Vec3> force; Vec3 moment; };
CableForces cable_forces(const Parameters& p,const Vector& s,const Chain& c,const std::vector<Vec3>& v) {
    size_t m=p.rest.size();
    std::vector<Vec3> edge_gradient(m,Vec3{}), f(m+1);
    for(size_t i=0;i+1<m;++i) {
        Vec3 a=c.tangent[i], next=c.tangent[i+1]; double cosine=dot(a,next), denominator=1.+cosine;
        double factor=(-4.*p.stiffness/((p.rest[i]+p.rest[i+1])/2.))/(denominator*denominator);
        edge_gradient[i]=edge_gradient[i]+(next-a*cosine)*factor/c.lengths[i];
        edge_gradient[i+1]=edge_gradient[i+1]+(a-next*cosine)*factor/c.lengths[i+1];
    }
    // Point clamp, not a welded first edge: the prescribed tangent lives at
    // s=0, the first edge tangent at its midpoint. The boundary dual length
    // is rest[0]/2, with no ghost mass/length or additional spring parameter.
    // E_root = EI/(2*d0) * |2(ta x t0)/(1+ta.t0)|^2.
    // Derive edge force AND body couple from this same rotationally invariant
    // energy. The first marker is free to bend; no tangent projection is used.
    Mat3 R=rotation(s);
    Vec3 ta=mv(R,p.root_tangent_body), t0=c.tangent[0];
    double cosine=dot(ta,t0), denominator=1.+cosine;
    require(denominator>1e-12,"Point-clamp tangent is antiparallel to first edge; DER curvature is singular");
    double factor=-4.*p.stiffness/(p.rest[0]/2.)/(denominator*denominator);
    edge_gradient[0]=edge_gradient[0]+(ta-t0*cosine)*factor/c.lengths[0];
    Vec3 root_moment=cross(ta,t0)*(-factor); // cable on drone, world frame
    for(size_t i=0;i<=m;++i) {
        Vec3 bending{}; if(i<m) bending=edge_gradient[i]; if(i) bending=bending-edge_gradient[i-1];
        f[i]=bending+p.gravity*p.node_mass[i];
    }
    double total=0.; for(double x:p.rest) total+=x;
    const auto& rs=p.residual_static; const auto& rn=p.residual_normal_log; const auto& rt=p.residual_tangential;
    bool residual=false; for(double x:rs) residual=residual||x!=0.; for(double x:rn) residual=residual||x!=0.; for(double x:rt) residual=residual||x!=0.;
    double arc=0.;
    for(size_t i=0;i<=m;++i) {
        f[i]=f[i]+v[i]*(-p.drag*p.node_mass[i]);
        // Quadratic normal drag on the cylinder (independence principle): only the velocity
        // normal to the local tangent loads the cable; tributary length is half of each edge.
        double coefficient=.5*p.air_density*p.normal_drag_coefficient*p.diameter;
        double length=((i?p.rest[i-1]:0.)+(i<m?p.rest[i]:0.))/2.;
        double st=arc/total; if(i<m) arc+=p.rest[i];
        if(residual) {
            // The residual (see Parameters): steady force, normal-drag multiplier, tangential drag.
            Vec3 steady{}; for(int a=0;a<3;++a) steady[a]=rs[3*a]+st*(rs[3*a+1]+st*rs[3*a+2]);
            f[i]=f[i]+(p.residual_body_frame?mv(R,steady):steady)*length;
            coefficient*=std::exp(rn[0]+rn[1]*st);
        }
        double tangential=residual?.5*p.air_density*p.diameter*(rt[0]+rt[1]*st):0.;
        if(coefficient>0.||tangential!=0.) {
            Vec3 tangent= i==0?ta:c.tangent[i-1];
            if(i>0 && i<m){tangent=tangent+c.tangent[i];tangent=tangent/norm(tangent);}
            Vec3 along=tangent*dot(tangent,v[i]), normal=v[i]-along;
            f[i]=f[i]-normal*(coefficient*length*norm(normal))-along*(tangential*length*norm(along));
        }
    }
    return {std::move(f),root_moment};
}
} // namespace

#include "material_frame.inc"
#include "prescribed_material.inc"

Model::Model(ModelInput input) : parameters(std::move(input.parameters)),com(input.tracking_to_com),
    attachment(input.tracking_to_attachment),intervals(input.intervals),output_step(input.output_step),delay(input.delay) {
    require(finite(com)&&finite(attachment),"Invalid COM/attachment geometry");
    parameters.arm=attachment-com;
    const size_t count=intervals.size();
    require(count>0 && input.subdivisions.size()==count && input.marker_mass.size()==count,
            "Cable intervals, subdivisions and marker masses must agree");
    require(finite(intervals)&&finite(input.subdivisions)&&finite(input.marker_mass),"Invalid cable mesh values");
    require(std::isfinite(output_step)&&output_step>0. && std::isfinite(delay)&&delay>=0.,"Invalid output step/command delay");
    double bare=input.total_cable_mass-std::accumulate(input.marker_mass.begin(),input.marker_mass.end(),0.);
    require(std::isfinite(bare)&&bare>0.,"Positive bare-cable mass required");
    parameters.rest.clear(); marker_indices={0};
    for(size_t i=0;i<count;++i) {
        double n=input.subdivisions[i];
        require(intervals[i]>0. && input.marker_mass[i]>=0. && n>=1. && n==std::floor(n) && n<=1000000.,"Invalid interval/subdivision/marker mass");
        parameters.rest.insert(parameters.rest.end(),static_cast<size_t>(n),intervals[i]/n);
        marker_indices.push_back(static_cast<int>(parameters.rest.size()));
    }
    const auto& rest=parameters.rest;
    double total=std::accumulate(rest.begin(),rest.end(),0.);
    parameters.node_mass.resize(rest.size()+1);
    for(size_t i=0;i<=rest.size();++i)
        parameters.node_mass[i]=((i?rest[i-1]:0.)+(i<rest.size()?rest[i]:0.))*(.5*bare/total);
    for(size_t i=0;i<count;++i) parameters.node_mass[marker_indices[i+1]]+=input.marker_mass[i];
    Engine validated(parameters);
    // Jacobi eigenvalues for the additional rigid-inertia triangle constraint.
    Mat3 A=parameters.inertia;
    for(int sweep=0;sweep<30;++sweep) {
        int i=0,j=1;
        for(auto pair: {std::pair<int,int>{0,2},{1,2}})
            if(std::abs(A[3*pair.first+pair.second])>std::abs(A[3*i+j])) {i=pair.first;j=pair.second;}
        if(std::abs(A[3*i+j])<=std::numeric_limits<double>::epsilon()*std::max({std::abs(A[0]),std::abs(A[4]),std::abs(A[8])})) break;
        double phi=.5*std::atan2(2*A[3*i+j],A[3*j+j]-A[3*i+i]);
        Mat3 Q=identity();Q[3*i+i]=Q[3*j+j]=std::cos(phi);Q[3*i+j]=std::sin(phi);Q[3*j+i]=-std::sin(phi);
        A=mm(transpose(Q),mm(A,Q));
    }
    require(2*std::max({A[0],A[4],A[8]})-(A[0]+A[4]+A[8])<=1e-12,"Inertia principal moments violate the triangle inequality");
}
Vector Model::initial_state(Vec3 tracking,Mat3 R,Vec3 tracking_v,Vec3 omega,
                             const Vector& positions,const Vector& velocities) const {
    require(finite(tracking)&&finite(R)&&finite(tracking_v)&&finite(omega),"Nonfinite initial body state");
    size_t m=parameters.rest.size();Vector s(18+6*m);
    put(s,0,tracking+mv(R,com));std::copy(R.begin(),R.end(),s.begin()+3);
    put(s,12,tracking_v+mv(R,cross(omega,com)));put(s,15,omega);
    Vec3 root=at(s,0)+mv(R,parameters.arm), root_v=at(s,12)+mv(R,cross(omega,parameters.arm));
    if(!positions.empty()) require(positions.size()==3*(m+1)&&finite(positions)&&norm(at(positions,0)-root)<=1e-6,"Initial cable root/shape mismatch");
    if(!velocities.empty()) require(velocities.size()==3*(m+1)&&finite(velocities)&&norm(at(velocities,0)-root_v)<=1e-6,"Initial cable root velocity/shape mismatch");
    double arc=0.;
    for(size_t i=0;i<m;++i) {
        arc+=parameters.rest[i];Vec3 q=root;q[2]-=arc;
        put(s,18+3*i,positions.empty()?q:at(positions,3*(i+1)));
        put(s,18+3*m+3*i,velocities.empty()?root_v:at(velocities,3*(i+1)));
    }
    Engine validator(parameters);validator.reset(s);return s;
}
Vector Model::observation(const Vector& s,Vec3 acceleration) const {
    size_t m=parameters.rest.size();require(s.size()==18+6*m&&finite(s),"Invalid observation state");
    Vector out(18+6*(m+1)+3);Mat3 R=rotation(s);
    put(out,0,at(s,0)-mv(R,com));put(out,3,at(s,12)+mv(R,cross(at(s,15),com*-1.)));
    std::copy(R.begin(),R.end(),out.begin()+6);put(out,15,at(s,15));
    std::vector<Vec3> q,v;cable(parameters,s,q,v);
    for(size_t i=0;i<=m;++i) {put(out,18+3*i,q[i]);put(out,18+3*(m+1)+3*i,v[i]);}
    put(out,out.size()-3,acceleration);return out;
}

Engine::Engine(Parameters parameters) : p_(std::move(parameters)) {
    require(std::isfinite(p_.mass) && p_.mass>0. && finite(p_.inertia) && finite(p_.arm) && finite(p_.gravity), "Invalid body parameters");
    require(finite(p_.root_tangent_body) && std::abs(norm(p_.root_tangent_body)-1.)<=1e-10,
            "Point-clamp body tangent must be a finite unit vector");
    for(int i=0;i<3;++i) for(int j=0;j<3;++j)
        require(std::abs(p_.inertia[3*i+j]-p_.inertia[3*j+i])<=1e-12,"Inertia must be symmetric");
    require(p_.inertia[0]>0. && p_.inertia[0]*p_.inertia[4]-p_.inertia[1]*p_.inertia[3]>0.
            && determinant(p_.inertia)>0.,"Inertia must be positive definite");
    require(!p_.rest.empty() && p_.node_mass.size()==p_.rest.size()+1 && finite(p_.rest) && finite(p_.node_mass), "Invalid cable mesh");
    for(double x:p_.rest) require(x>0., "Cable lengths must be positive");
    for(double x:p_.node_mass) require(x>0., "Node masses must be positive");
    require(std::isfinite(p_.maximum_step) && p_.maximum_step>0. && p_.projections>0, "Invalid integration settings");
    require(std::isfinite(p_.stiffness) && p_.stiffness>=0. && std::isfinite(p_.drag) && p_.drag>=0., "Invalid stiffness or drag");
    require(std::isfinite(p_.bending_damping) && p_.bending_damping>=0.,"Bending damping Cb must be finite and nonnegative");
    require(p_.bending_damping_saturation>0.,"Bending damping saturation rate must be positive (infinite: linear)");
    require(std::isfinite(p_.air_density) && p_.air_density>=0. && std::isfinite(p_.normal_drag_coefficient)
            && p_.normal_drag_coefficient>=0. && std::isfinite(p_.diameter) && p_.diameter>=0.,"Invalid quadratic air drag");
    require(finite(p_.residual_static) && finite(p_.residual_normal_log) && finite(p_.residual_tangential),"Invalid cable residual");
    require(6+3*p_.rest.size()<=material::capacity,"Material derivative capacity exceeded");
    Mat3 K=skew(p_.arm), K2=mm(transpose(K),K), inertia=p_.inertia;
    root_mass_=p_.mass+p_.node_mass[0]; double reduced=p_.mass*p_.node_mass[0]/root_mass_;
    for(int i=0;i<9;++i) inertia[i]+=reduced*K2[i];
    root_inverse_=inverse(inertia);
}
void Engine::validate_state(const Vector& s, bool constraints) const {
    size_t m=p_.rest.size(); require(s.size()==18+6*m && finite(s), "State shape or values are invalid");
    Mat3 R=rotation(s), gram=mm(transpose(R),R), I=identity();
    double tol=constraints?1e-8:1e-6;
    for(int i=0;i<9;++i) require(std::abs(gram[i]-I[i])<=tol, "State rotation is not orthogonal");
    require(std::abs(determinant(R)-1.)<=tol, "State rotation is not proper");
    std::vector<Vec3> q,v; cable(p_,s,q,v);
    Vec3 previous=mv(R,p_.root_tangent_body);
    for(size_t i=0;i<m;++i) {
        Vec3 edge=q[i+1]-q[i]; double l=norm(edge);
        require(std::isfinite(l) && l>1e-12,"Collapsed/nonfinite cable edge; DER geometry is invalid");
        Vec3 tangent=edge/l;
        require(1.+dot(previous,tangent)>1e-12,
                "Antiparallel root or interior cable directions; DER curvature is singular");
        previous=tangent;
    }
    if(constraints) {
        for(size_t i=0;i<m;++i) {
            Vec3 e=q[i+1]-q[i]; double l=norm(e);
            require(l>0. && std::abs(l-p_.rest[i])<=1e-6, "Initial cable length mismatch");
            require(std::abs(dot(e/l,v[i+1]-v[i]))<=1e-6, "Initial cable velocity violates length constraint");
        }
    }
}
void Engine::reset(const Vector& s) { Guard guard(busy_); validate_state(s,true); state_=previous_state_=s; previous_step_=0.; }
Vector Engine::checkpoint() const {
    Guard guard(busy_);require(!state_.empty(),"Reset engine before use");
    Vector out=state_;out.insert(out.end(),previous_state_.begin(),previous_state_.end());
    out.push_back(previous_step_);return out;
}
void Engine::restore(const Vector& saved) {
    Guard guard(busy_);size_t width=18+6*p_.rest.size();
    require(saved.size()==2*width+1 && finite(saved),"Invalid BDF2 checkpoint");
    Vector current(saved.begin(),saved.begin()+width),previous(saved.begin()+width,saved.end()-1);
    require(saved.back()>=0. && saved.back()<=p_.maximum_step*(1.+1e-9),"Invalid BDF2 history step");
    validate_state(current,true);validate_state(previous,true);
    state_=std::move(current);previous_state_=std::move(previous);previous_step_=saved.back();
}
Vector Engine::state() const { Guard guard(busy_); require(!state_.empty(),"Reset engine before use"); return state_; }
Vec3 Engine::drone_air_force() const {
    Guard guard(busy_);require(!state_.empty(),"Reset engine before use");
    return drone_body_drag_acceleration(p_,state_)*p_.mass;
}
Vector Engine::acceleration(const Wrench& w) const {
    Guard guard(busy_); require(!state_.empty(),"Reset engine before use"); validate_wrench(w); return add_scaled(acceleration_at(state_,w),material::instantaneous(p_,state_),1.);
}
Vector Engine::acceleration_at(const Vector& s, const Wrench& w) const {
    size_t m=p_.rest.size(); std::vector<Vec3> q,v; cable(p_,s,q,v);
    auto b=boundary(p_,s);auto c=chain(p_,q,b,root_inverse_,root_mass_);
    auto f=generalized_load(s,w);
    auto a=body_solve(b,root_inverse_,root_mass_,at(f,0),at(f,3));a.resize(6+3*m);
    for(size_t i=0;i<m;++i)put(a,6+3*i,at(f,6+3*i)/p_.node_mass[i+1]);
    auto rhs=constraint_product(c,b,a);
    for(size_t i=0;i<m;++i){auto dv=v[i+1]-v[i],transverse=dv-c.tangent[i]*dot(dv,c.tangent[i]);rhs[i]+=dot(transverse,transverse)/c.lengths[i];}
    auto omega=at(s,15);rhs[0]-=dot(c.tangent[0],mv(rotation(s),cross(omega,cross(omega,p_.arm))));
    return add_scaled(a,impulse(p_,c,solve(c,rhs)),-1.);
}
Vector Engine::generalized_load(const Vector& s, const Wrench& w) const {
    size_t m=p_.rest.size(); std::vector<Vec3> q,v; cable(p_,s,q,v);
    Boundary b=boundary(p_,s); Chain c=chain(p_,q,b,root_inverse_,root_mass_);
    auto forces=cable_forces(p_,s,c,v);const auto& f=forces.force;
    Vec3 root_moment=forces.moment;Mat3 R=rotation(s),Rt=transpose(R);
    Vec3 omega=at(s,15), gyro=cross(omega,mv(p_.inertia,omega));
    Vec3 centripetal=mv(R,cross(omega,cross(omega,p_.arm)));
    Vec3 force=p_.gravity*p_.mass+Vec3{R[2],R[5],R[8]}*w[0]+f[0]-centripetal*p_.node_mass[0];
    force=force+drone_body_drag_acceleration(p_,s)*p_.mass;
    Vec3 torque=Vec3{w[1],w[2],w[3]}-gyro+mv(transpose(b.B),f[0])-mv(transpose(b.B),centripetal)*p_.node_mass[0];
    torque=torque+mv(Rt,root_moment);
    Vector out(6+3*m);put(out,0,force);put(out,3,torque);
    for(size_t i=0;i<m;++i)put(out,6+3*i,f[i+1]);return out;
}
Vector Engine::integrate(const Vector& s, const Vector& previous, double previous_step, double h, const std::function<Wrench(const Vector&,double)>& control, double start) const {
    // Projected Lie-BDF2 endpoint. Backward Euler only for explicit order startup.
    using namespace material;int m=int(p_.rest.size()),v=6+3*m;
    auto coefficient=bdf_coefficients(h,previous_step);double beta=coefficient.beta,b=coefficient.memory;
    auto scale=saturation(p_,s);   // saturating bending damping: factors from the step start
    auto u0=velocity(s,m),prior_u=velocity(previous,m);Vector x=u0,offset(v);
    for(int i=0;i<v;++i)u0[i]+=b*(u0[i]-prior_u[i]);
    put(offset,0,(at(s,0)-at(previous,0))*b);
    if(b>0.)put(offset,3,log_rotation(mm(transpose(rotation(previous)),rotation(s)))*b);
    for(int i=0;i<3*m;++i)offset[6+i]=b*(s[18+i]-previous[18+i]);
    for(int iteration=0;iteration<20;++iteration){
        Vector out=displaced(s,add_scaled(offset,x,beta),m);
        std::vector<Vec3>q,qv;
        for(int k=0;k<p_.projections;++k){auto b=boundary(p_,out);cable(p_,out,q,qv);
            auto c=chain(p_,q,b,root_inverse_,root_mass_);Vector rhs(m);
            for(int i=0;i<m;++i)rhs[i]=p_.rest[i]-c.lengths[i];
            out=displaced(out,impulse(p_,c,solve(c,rhs)),m);}
        cable(p_,out,q,qv);for(int i=0;i<m;++i)require(std::abs(norm(q[i+1]-q[i])-p_.rest[i])<=1e-10,"Implicit position projection failed");
        with_velocity(out,x,m);validate_state(out,false);
        auto g=geometry(p_,out);auto A=operators(g,p_.bending_damping,scale);
        auto w=control(out,start+h);validate_wrench(w);auto f=generalized_load(out,w);
        // (M + beta D) u = M u0 + beta f, subject to the velocity-level length constraints.
        for(auto&a:A)a*=beta;
        Vector rhs=material::product(g.mass,u0,v,v);
        for(int i=0;i<v;++i){rhs[i]+=beta*f[i];for(int j=0;j<v;++j)A[i*v+j]+=g.mass[i*v+j];}
        auto next=prescribed::constrained(A,rhs,g.constraint,Vector(m),v,m);double err=0.;
        for(int i=0;i<v;++i)err=std::max(err,std::abs(next[i]-x[i])/(1.+std::max(std::abs(next[i]),std::abs(x[i]))));
        x=std::move(next);if(err>p_.solver_tolerance)continue;
        with_velocity(out,x,m);
        require(maxabs(material::product(g.constraint,x,m,v))<=1e-9,"Implicit velocity constraint failed");
        validate_state(out,false);return out;
    }
    throw std::runtime_error("Projected implicit endpoint iteration failed after 20 iterations");
}
void Engine::advance_impl(double duration, const std::function<Wrench(const Vector&,double)>& controller) {
    require(!state_.empty(),"Reset engine before use");
    require(std::isfinite(duration) && duration>0.,"Duration must be positive and finite");
    double ratio=duration/p_.maximum_step, nearest=std::nearbyint(ratio);
    require(std::isfinite(ratio) && ratio<=double(std::numeric_limits<int>::max()),"Duration is too large");
    int count; double h;
    if(nearest>=1. && std::abs(duration-nearest*p_.maximum_step)<=1e-12) { count=int(nearest); h=p_.maximum_step; }
    else { count=int(std::ceil(ratio)); h=duration/count; }
    Vector next=state_,prior=previous_state_;double previous_step=previous_step_;
    // Physical state AND history commit only after the whole call succeeds.
    for(int i=0;i<count;++i) {
        auto endpoint=integrate(next,prior,previous_step,h,controller,i*h);
        prior=std::move(next);next=std::move(endpoint);previous_step=h;
    }
    validate_state(next,false);state_=std::move(next);previous_state_=std::move(prior);previous_step_=previous_step;
}
void Engine::advance(double duration, const Wrench& w) {
    Guard guard(busy_); validate_wrench(w); advance_impl(duration,[&](const Vector&,double){return w;});
}
void Engine::advance_controlled(double duration, const std::function<Wrench(const Vector&)>& controller) {
    Guard guard(busy_); require(bool(controller),"Controller callback is required");
    advance_impl(duration,[&](const Vector& s,double){return controller(s);});
}
Motors Engine::advance_motors(double duration,const Wrench& requested,const Actuator& actuator,const Motors& motors) {
    Guard guard(busy_);
    const auto target=actuator.target(requested);
    const auto end=actuator.response(motors,target,duration);
    // Held motor target, exact first-order response at each implicit endpoint. No
    // motor state is advanced by a controller callback or telemetry read.
    advance_impl(duration,[&](const Vector&,double t){return actuator.wrench(actuator.response(motors,target,t));});
    return end; // Engine state and motor endpoint commit only on success.
}
} // namespace aerowhip
