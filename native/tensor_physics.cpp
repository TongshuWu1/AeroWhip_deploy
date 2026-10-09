// Differentiable ATen implementation. Python owns controllers/losses, not these equations.
// Composite registration preserves autograd, vmap and compiler decomposition.
#include <ATen/ATen.h>
#include <ATen/TensorSubclassLikeUtils.h>
#include <torch/library.h>
#include <algorithm>
#include <cmath>
#include <limits>
#include <tuple>
#include <vector>

namespace aerowhip_tensor {
using at::Tensor;
using List = std::vector<Tensor>;
constexpr double pi = 3.14159265358979323846;
Tensor part(const Tensor& x, int64_t a, int64_t b) { return x.slice(0,a,b); }
Tensor cat(const List& x, int64_t dim=0) { return at::cat(x,dim); }
Tensor vector_cross(const Tensor& a,const Tensor& b) { return at::linalg_cross(a,b,-1); }
Tensor norm(const Tensor& x,int64_t dim=-1,bool keep=false) {
    return at::linalg_vector_norm(x,2,{dim},keep);
}
Tensor skew(const Tensor& v) {
    auto x=v.select(-1,0), y=v.select(-1,1), z=v.select(-1,2), o=at::zeros_like(x);
    auto sizes=v.sizes().vec(); sizes.pop_back(); sizes.push_back(3); sizes.push_back(3);
    return at::stack({o,-z,y,z,o,-x,-y,x,o},-1).reshape(sizes);
}
Tensor rotation_exp(const Tensor& v) {
    auto angle=aerowhip_tensor::norm(v), k=skew(v), eye=at::eye(3,v.options());
    auto a=at::sinc(angle/pi).unsqueeze(-1).unsqueeze(-1);
    auto b=(.5*at::sinc(angle/(2*pi)).square()).unsqueeze(-1).unsqueeze(-1);
    return eye+a*k+b*at::matmul(k,k);
}
Tensor solve(const Tensor& A,const Tensor& rhs) {
    if(!A.is_cuda()) return at::linalg_solve(A,rhs);
    auto result=at::linalg_solve_ex(A,rhs,true,false);
    return at::where(std::get<1>(result)==0,std::get<0>(result),at::full_like(rhs,std::numeric_limits<double>::quiet_NaN()));
}
Tensor chain_solve(const Tensor& diagonal,const Tensor& off,const Tensor& rhs,bool elimination) {
    if(!elimination) return solve(at::diag(diagonal)+at::diag(off,1)+at::diag(off,-1),rhs);
    List pivots{diagonal[0]}, values{rhs[0]};
    for(int64_t i=1;i<diagonal.size(0);++i) {
        auto ratio=off[i-1]/pivots.back();
        pivots.push_back(diagonal[i]-ratio*off[i-1]); values.push_back(rhs[i]-ratio*values.back());
    }
    List reversed{values.back()/pivots.back()};
    for(int64_t i=diagonal.size(0)-2;i>=0;--i) reversed.push_back((values[i]-off[i]*reversed.back())/pivots[i]);
    std::reverse(reversed.begin(),reversed.end()); auto output=at::stack(reversed);
    return at::where(at::stack(pivots).gt(0).all(),output,at::full_like(output,std::numeric_limits<double>::quiet_NaN()));
}
// Parameters: mass, inertia, tracking->COM, COM->root, gravity, rest, node mass,
// EI, drag, root inertia inverse, root translation mass, base body mass matrix,
// prescribed root tangent in body coordinates (index12). Fixed geometry caches
// have the same lifetime as the model; fit parameters EI/drag are passed each call.
List setup(const List& p) {
    auto k=skew(p[3]), mass=p[0]+p[6][0], reduced=p[0]*p[6][0]/mass;
    return {solve(p[1]+reduced*at::matmul(k.t(),k),at::eye(3,p[0].options())),mass,
            at::block_diag({p[0]*at::eye(3,p[0].options()),p[1]})};
}
List observe(const List& s,const List& p) {
    auto tracking=s[0]-at::matmul(s[1],p[2]);
    auto tracking_v=s[2]+at::matmul(s[1],vector_cross(s[3],-p[2]));
    auto root=s[0]+at::matmul(s[1],p[3]);
    auto root_v=s[2]+at::matmul(s[1],vector_cross(s[3],p[3]));
    return {tracking,tracking_v,cat({root.unsqueeze(0),s[4]}),cat({root_v.unsqueeze(0),s[5]})};
}
Tensor positive_geometry_or_nan(const Tensor& value) {
    return at::where(value>1e-12,value,at::full_like(value,std::numeric_limits<double>::quiet_NaN()));
}
Tensor geometry_valid(const Tensor& q,const Tensor& root_tangent) {
    auto edges=q.slice(0,1)-q.slice(0,0,-1), lengths=norm(edges,-1,true);
    auto tangent=edges/positive_geometry_or_nan(lengths);
    auto previous=cat({root_tangent.unsqueeze(0),tangent.slice(0,0,-1)});
    return at::isfinite(q).all() & at::isfinite(root_tangent).all()
        & (lengths>1e-12).all() & (1+(previous*tangent).sum(-1)>1e-12).all();
}
List invalidate_geometry(List state,const Tensor& valid) {
    // Invalid lanes remain explicit failures, never finite surrogate physics.
    // Callers must reject nonfinite loss/gradients, not merely mask NaNs later.
    for(auto& value:state) value=at::where(valid,value,at::full_like(value,std::numeric_limits<double>::quiet_NaN()));
    return state;
}
Tensor bending(const Tensor& q,const Tensor& ei,const Tensor& dual) {
    auto edges=q.slice(1,1)-q.slice(1,0,-1), lengths=positive_geometry_or_nan(norm(edges,-1,true));
    auto t=edges/lengths, a=t.slice(1,0,-1), b=t.slice(1,1);
    auto cosine=(a*b).sum(-1,true);
    auto factor=-4*ei.unsqueeze(1).unsqueeze(2)/dual.unsqueeze(0).unsqueeze(-1);
    factor=factor/positive_geometry_or_nan(1+cosine).square();
    auto previous=factor*(b-cosine*a)/lengths.slice(1,0,-1);
    auto following=factor*(a-cosine*b)/lengths.slice(1,1);
    auto zero=at::zeros_like(t.slice(1,0,1));
    auto gradient=cat({previous,zero},1)+cat({zero,following},1);
    return cat({gradient,zero},1)-cat({zero,gradient},1);
}
List root_bending(const Tensor& q,const Tensor& tangent,const Tensor& ei,const Tensor& first_rest) {
    // Same point-clamp half-cell energy and variation as the scalar engine.
    // This endpoint closure is NOT a constraint locking the first edge.
    auto edge=q[1]-q[0], length=positive_geometry_or_nan(aerowhip_tensor::norm(edge)), t=edge/length;
    auto cosine=at::dot(tangent,t), denominator=1+cosine;
    // Do not regularize an antiparallel clamp into finite, apparently valid
    // forces. Tensor lanes propagate NaN to the existing validity checks.
    auto safe=positive_geometry_or_nan(denominator);
    auto factor=-4*ei/(first_rest/2)/safe.square();
    auto gradient=factor*(tangent-cosine*t)/length;
    auto force=cat({gradient.unsqueeze(0),-gradient.unsqueeze(0),at::zeros_like(q.slice(0,2))});
    return {force,-factor*vector_cross(tangent,t)}; // cable-on-support moment, world
}
Tensor network(Tensor x,const List& w) {
    for(size_t i=0;i<w.size();i+=2) { x=at::linear(x,w[i],w[i+1]); if(i+2<w.size()) x=at::tanh(x); }
    return x;
}
// Spec: cable log bound, drone max rate s^-1, velocity scale m/s,
// cable enabled, drone enabled. Six weight/bias tensors per branch.
Tensor cable_drag(const Tensor& q,const Tensor& v,const Tensor& root_tangent,
                  const List& p,const List& w,const std::vector<double>& spec) {
    if(w.empty() || spec[3]==0.)return -p[8]*v;
    auto edges=q.slice(0,1)-q.slice(0,0,-1),t=edges/positive_geometry_or_nan(norm(edges,-1,true));
    auto bisector=t.slice(0,0,-1)+t.slice(0,1);
    auto tangent=cat({root_tangent.unsqueeze(0),bisector/positive_geometry_or_nan(norm(bisector,-1,true)),t.slice(0,-1)});
    auto parallel=tangent*(tangent*v).sum(-1,true),perpendicular=v-parallel;
    auto features=cat({parallel.square().sum(-1,true),perpendicular.square().sum(-1,true)},-1)/(spec[2]*spec[2]);
    auto gamma=p[8]*at::exp(spec[0]*at::tanh(network(features,List(w.begin(),w.begin()+6))));
    return -gamma.slice(-1,0,1)*parallel-gamma.slice(-1,1,2)*perpendicular;
}
List aerodynamic_accelerations(const List& s,const List& p,const List& w,const std::vector<double>& spec) {
    TORCH_CHECK(w.empty() || (w.size()==12 && spec.size()==5),"Expected structured aerodynamic weights/specification");
    auto o=observe(s,p);
    auto cable=cable_drag(o[2],o[3],at::matmul(s[1],p[12]),p,w,spec);
    auto drone=at::zeros_like(s[2]);
    if(!w.empty() && spec[4]!=0.) {
        auto body_v=at::matmul(s[1].t(),s[2]);
        auto y=network(body_v.square()/(spec[2]*spec[2]),List(w.begin()+6,w.end()));
        auto gamma=spec[1]*.5*(1+at::tanh(y/2));
        drone=at::matmul(s[1],-gamma*body_v);
    }
    return {cable,drone};
}
Tensor boundary(const List& s,const List& p) {
    return cat({at::eye(3,p[0].options()),-at::matmul(s[1],skew(p[3]))},1);
}
Tensor body_solve(const Tensor& boundary,const Tensor& rhs,const List& p) {
    // Preserve the qualified tensor reference's arithmetic and compiler graph.
    auto body=p[11]+at::matmul(p[6][0]*boundary.t(),boundary);
    auto coupling=body.slice(0,0,3).slice(1,3,6), translation=part(rhs,0,3)/p[10];
    auto angular=at::matmul(p[9],part(rhs,3,6)-at::matmul(coupling.t(),translation));
    return cat({translation-at::matmul(coupling,angular)/p[10],angular});
}
List system(const Tensor& q,const Tensor& B,const List& p) {
    auto edges=q.slice(0,1)-q.slice(0,0,-1), lengths=norm(edges,1).clamp_min(1e-12);
    auto tangent=edges/lengths.unsqueeze(1), root_row=-at::matmul(tangent[0],B);
    auto response=body_solve(B,root_row,p);
    auto right=(tangent*(tangent/p[6].slice(0,1).unsqueeze(1))).sum(-1);
    auto left=(tangent.slice(0,1)*(tangent.slice(0,1)/p[6].slice(0,1,-1).unsqueeze(1))).sum(-1);
    auto diagonal=right+cat({at::dot(root_row,response).reshape({1}),left});
    auto off=-(tangent.slice(0,0,-1)*(tangent.slice(0,1)/p[6].slice(0,1,-1).unsqueeze(1))).sum(-1);
    return {lengths,tangent,response,diagonal,off};
}
Tensor product(const Tensor& tangent,const Tensor& B,const Tensor& u) {
    auto free=u.slice(0,6).reshape({-1,3});
    return (tangent*(free-cat({at::matmul(B,part(u,0,6)).unsqueeze(0),free.slice(0,0,-1)}))).sum(1);
}
Tensor impulse(const List& sys,const Tensor& lambda,const List& p) {
    auto edge=sys[1]*lambda.unsqueeze(1);
    auto free=cat({edge.slice(0,0,-1)-edge.slice(0,1),edge.slice(0,-1)});
    return cat({sys[2]*lambda[0],(free/p[6].slice(0,1).unsqueeze(1)).flatten()});
}
Tensor generalized_load(const List& s,const List& p,const Tensor& thrust,const Tensor& torque,
                    const List& w,const std::vector<double>& spec) {
    auto o=observe(s,p); auto q=o[2],v=o[3],B=boundary(s,p);
    auto dual=(p[5].slice(0,0,-1)+p[5].slice(0,1))/2;
    auto force=bending(q.unsqueeze(0),p[7].reshape({1}),dual)[0]+p[6].unsqueeze(1)*p[4];
    auto root=root_bending(q,at::matmul(s[1],p[12]),p[7],p[5][0]);
    force=force+root[0];
    auto aero=aerodynamic_accelerations(s,p,w,spec);
    force=force+p[6].unsqueeze(1)*aero[0];
    auto body_force=p[0]*(p[4]+aero[1])+thrust*s[1].select(1,2), gyro=vector_cross(s[3],at::matmul(p[1],s[3]));
    auto rhs=cat({body_force,torque-gyro+at::matmul(s[1].t(),root[1])})+at::matmul(B.t(),force[0]);
    auto centripetal=at::matmul(s[1],vector_cross(s[3],vector_cross(s[3],p[3])));
    rhs=rhs-p[6][0]*at::matmul(B.t(),centripetal);
    return cat({rhs,force.slice(0,1).flatten()});
}
Tensor elastic_acceleration(const List& s,const List& p,const Tensor& thrust,const Tensor& torque,
                    const List& w,const std::vector<double>& spec,bool elimination) {
    auto o=observe(s,p);auto q=o[2],v=o[3],B=boundary(s,p);
    auto f=generalized_load(s,p,thrust,torque,w,spec);
    auto a=cat({body_solve(B,f.slice(0,0,6),p),f.slice(0,6)/p[6].slice(0,1).repeat_interleave(3)});
    auto centripetal=at::matmul(s[1],vector_cross(s[3],vector_cross(s[3],p[3])));
    auto sys=system(q,B,p); auto relative=v.slice(0,1)-v.slice(0,0,-1);
    auto transverse=relative-sys[1]*(relative*sys[1]).sum(1,true);
    auto bias=transverse.square().sum(1)/sys[0];
    auto root_bias=cat({at::dot(sys[1][0],centripetal).reshape({1}),at::zeros({p[5].size(0)-1},p[0].options())});
    auto lambda=chain_solve(sys[3],sys[4],product(sys[1],B,a)+bias-root_bias,elimination);
    return a-impulse(sys,lambda,p);
}
Tensor velocity(const List& s) { return cat({s[2],s[3],s[5].flatten()}); }
List with_velocity(List s,const Tensor& u) { s[2]=part(u,0,3);s[3]=part(u,3,6);s[5]=u.slice(0,6).reshape({-1,3});return s; }
List displace(List s,const Tensor& d) {
    s[0]=s[0]+part(d,0,3);s[1]=at::matmul(s[1],rotation_exp(part(d,3,6)));s[4]=s[4]+d.slice(0,6).reshape({-1,3});return s;
}
List project_position(List s,const List& p,int64_t iterations,bool elimination) {
    for(int64_t i=0;i<iterations;++i) {
        auto B=boundary(s,p), q=observe(s,p)[2];auto sys=system(q,B,p);
        auto lambda=chain_solve(sys[3],sys[4],p[5]-sys[0],elimination);
        s=displace(s,impulse(sys,lambda,p));
    }
    return s;
}
List project(List s,const Tensor& u,const List& p,int64_t iterations,bool elimination) {
    s=project_position(s,p,iterations,elimination);
    auto B=boundary(s,p), q=observe(s,p)[2];auto sys=system(q,B,p);
    auto lambda=chain_solve(sys[3],sys[4],product(sys[1],B,u),elimination);
    auto output=with_velocity(s,u-impulse(sys,lambda,p));
    return invalidate_geometry(output,geometry_valid(q,at::matmul(s[1],p[12])));
}
#include "tensor_material.inc"
Tensor acceleration(const List&s,const List&p,const Tensor&thrust,const Tensor&torque,
                    const List&w,const std::vector<double>&spec,bool elimination){
    return elastic_acceleration(s,p,thrust,torque,w,spec,elimination)+material::instantaneous(s,p);
}
List material_operators(const List&s,const List&p){auto g=material::geometry(s,p);auto op=material::operators(g,p);
    return {g.k,g.jac,g.mass,g.constraint,op.damping,op.twist};}
// Physical tensors remain seven; integration adds packed previous state and h.
Tensor pack_physical(const List&s) {
    return cat({s[0],s[1].flatten(),s[2],s[3],s[4].flatten(),s[5].flatten(),s[6]});
}
List unpack_physical(const Tensor&s,int64_t m) {
    return {s.slice(0,0,3),s.slice(0,3,12).reshape({3,3}),s.slice(0,12,15),s.slice(0,15,18),
        s.slice(0,18,18+3*m).reshape({m,3}),s.slice(0,18+3*m,18+6*m).reshape({m,3}),s.slice(0,18+6*m)};
}
List bdf_coefficients(double h,const Tensor&previous) {
    auto old=previous.reshape({}),active=(old>0)&(h<=2*old*(1.+1e-12));
    auto r=h/at::where(active,old,at::ones_like(old));
    return {at::where(active,r*r/(1+2*r),at::zeros_like(r)),
            at::where(active,h*(1+r)/(1+2*r),at::full_like(r,h))};
}
Tensor rotation_log(const Tensor&R) {
    auto z=.5*at::stack({R[2][1]-R[1][2],R[0][2]-R[2][0],R[1][0]-R[0][1]});
    auto sine2=z.square().sum(),safe=at::sqrt(sine2.clamp_min(1e-16)),cosine=.5*(R.trace()-1);
    auto angle=at::atan2(safe,cosine);
    // Series avoids a zero-norm derivative at identity; general branch safe too.
    auto scale=at::where((sine2<1e-8)&(cosine>0),1+sine2/6+3*sine2*sine2/40,angle/safe);
    return at::where(angle<pi-1e-6,z*scale,at::full_like(z,std::numeric_limits<double>::quiet_NaN()));
}
// work = current velocity/angle-rate iterate, convergence mask, validity mask,
//        first-converged endpoint (seven state tensors).
List implicit_begin(const List&s){
    return {cat({velocity(s),at::zeros_like(s[6])}),at::zeros({},s[0].options().dtype(at::kBool)),
        at::ones({},s[0].options().dtype(at::kBool)),s[0],s[1],s[2],s[3],s[4],s[5],s[6]};
}
List implicit_finish(const List&s,const List&work,double h){
    List result(work.begin()+3,work.end());result.push_back(pack_physical(s));result.push_back(s[8]*0+h);
    return invalidate_geometry(result,work[1] & work[2]);
}
List implicit_iteration(const List&s,const List&p,const Tensor&thrust,const Tensor&torque,double h,
                   int64_t iterations,const List&weights,const std::vector<double>&spec,const List&work,bool elimination){
    TORCH_CHECK(s.size()==9,"BDF2 needs seven physical tensors, previous state and step");
    int64_t m=p[5].size(0),v=6+3*m;auto prior=unpack_physical(s[7],m),c=bdf_coefficients(h,s[8]);
    auto b=c[0],beta=c[1],u0=velocity(s)+b*(velocity(s)-velocity(prior)),theta=s[6]+b*(s[6]-prior[6]);
    auto offset=b*cat({s[0]-prior[0],rotation_log(at::matmul(prior[1].t(),s[1])),(s[4]-prior[4]).flatten()});
    auto x=work[0],done=work[1],valid=work[2];List result(work.begin()+3,work.end());
    auto current=with_velocity(project_position(displace(List(s.begin(),s.begin()+7),offset+beta*x.slice(0,0,v)),p,iterations,elimination),x.slice(0,0,v));
    auto q=observe(current,p)[2];
    auto pose_ok=geometry_valid(q,at::matmul(current[1],p[12])) & (aerowhip_tensor::norm(q.slice(0,1)-q.slice(0,0,-1))-p[5]).abs().le(1e-10).all();
    valid=valid & (done | pose_ok);
    auto g=material::geometry(current,p);auto op=material::operators(g,p);
    auto A=beta*op.damping+at::block_diag({g.mass,beta*beta*op.twist});
    auto f=generalized_load(current,p,thrust,torque,weights,spec);
    auto rhs=cat({at::matmul(g.mass,u0)+beta*f,-beta*at::matmul(op.twist,theta)});
    auto J=cat({g.constraint,at::zeros({m,m},p[0].options())},1);
    auto next=material::constrained_solve(A,rhs,J,at::zeros({m},p[0].options()));
    auto converged=((next-x).abs()/(1+at::maximum(next.abs(),x.abs()))).max().le(2e-11);
    current=with_velocity(current,next.slice(0,0,v));current[6]=theta+beta*next.slice(0,v);
    auto accept=(~done) & converged;
    valid=valid & (~accept | at::matmul(g.constraint,next.slice(0,0,v)).abs().le(1e-9).all());
    for(size_t i=0;i<result.size();++i)result[i]=at::where(accept,current[i],result[i]);
    x=at::where(done,x,next);done=done | converged;
    List output{x,done,valid};output.insert(output.end(),result.begin(),result.end());return output;
}
List implicit_step(const List&s,const List&p,const Tensor&thrust,const Tensor&torque,double h,
                   int64_t iterations,const List&weights,const std::vector<double>&spec,bool elimination){
    auto work=implicit_begin(s);
    for(int k=0;k<20;++k){
        work=implicit_iteration(s,p,thrust,torque,h,iterations,weights,spec,work,elimination);
        if(!at::isTensorSubclassLike(work[1]) && work[1].item<bool>())break;
    }
    return implicit_finish(s,work,h);
}
// Actuator pack: mixing, inverse mixing, minimum N, maximum N, response seconds.
// Configuration/rank checks are shared with the scalar C++ Actuator at setup.
List motor_allocate(const Tensor& requested,const List& a) {
    auto raw=at::matmul(a[1],requested);
    return {raw,at::minimum(at::maximum(raw,a[2]),a[3])};
}
template<typename Time>
Tensor motor_response_at(const Tensor& initial,const Tensor& target,const List& a,const Time& t) {
    auto positive=a[4]>0, safe=at::where(positive,a[4],at::ones_like(a[4]));
    auto forces=at::where(positive,initial+(target-initial)*(-at::expm1(-t/safe)),target);
    return at::minimum(at::maximum(forces,a[2]),a[3]);
}
Tensor motor_response(const Tensor& initial,const Tensor& target,const List& a,double t) {
    return motor_response_at(initial,target,a,t);
}
Tensor motor_response_elapsed(const Tensor& initial,const Tensor& target,const List& a,const Tensor& t) {
    return motor_response_at(initial,target,a,t);
}
Tensor motor_wrench(const Tensor& forces,const List& a) {return at::matmul(a[0],forces);}
List motor_substep(const List& s,const Tensor& initial,const Tensor& target,const List& a,
                   const List& p,double h,const Tensor& start,int64_t iterations,
                   const List& weights,const std::vector<double>& spec,bool elimination) {
    auto wrench=motor_wrench(motor_response_at(initial,target,a,start+h),a);
    return implicit_step(s,p,wrench[0],wrench.slice(0,1),h,iterations,weights,spec,elimination);
}
// Prescribed-attachment calibration uses the same force/constraint conventions.
List prescribed_system(const Tensor& q,const List& p) {
    auto edges=q.slice(0,1)-q.slice(0,0,-1), lengths=norm(edges,1).clamp_min(1e-12), t=edges/lengths.unsqueeze(1);
    auto inverse=p[6].slice(0,1).reciprocal(), left=cat({at::zeros({1},inverse.options()),inverse.slice(0,0,-1)});
    return {lengths,t,t.square().sum(-1)*(inverse+left),-(t.slice(0,0,-1)*t.slice(0,1)).sum(-1)*inverse.slice(0,0,-1)};
}
Tensor prescribed_impulse(const Tensor& t,const Tensor& lambda,const List& p) {
    auto e=t*lambda.unsqueeze(1);return cat({e.slice(0,0,-1)-e.slice(0,1),e.slice(0,-1)})/p[6].slice(0,1).unsqueeze(1);
}
Tensor prescribed_product(const Tensor& t,const Tensor& free,const Tensor& root) {
    return (t*(free-cat({root.unsqueeze(0),free.slice(0,0,-1)}))).sum(-1);
}
List prescribed_elastic_acceleration(const List& s,const List& b,const List& p,const List& w,const std::vector<double>& spec,bool elimination) {
    TORCH_CHECK(b.size()==5,"Prescribed material clamp requires root P/V/A, rotation and body angular rate");
    auto q=cat({b[0].unsqueeze(0),s[0]}),v=cat({b[1].unsqueeze(0),s[1]});
    auto root=root_bending(q,at::matmul(b[3],p[12]),p[7],p[5][0]);
    auto force=bending(q.unsqueeze(0),p[7].reshape({1}),(p[5].slice(0,0,-1)+p[5].slice(0,1))/2)[0]
        +root[0]+p[6].unsqueeze(1)*(p[4]+cable_drag(q,v,at::matmul(b[3],p[12]),p,w,spec));
    auto a=force.slice(0,1)/p[6].slice(0,1).unsqueeze(1);auto sys=prescribed_system(q,p);
    auto relative=v.slice(0,1)-v.slice(0,0,-1), transverse=relative-sys[1]*(relative*sys[1]).sum(-1,true);
    auto lambda=chain_solve(sys[2],sys[3],prescribed_product(sys[1],a,b[2])+transverse.square().sum(-1)/sys[0],elimination);
    return {a-prescribed_impulse(sys[1],lambda,p),force[0]+sys[1][0]*lambda[0]-p[6][0]*b[2],root[1]};
}
List prescribed_acceleration(const List&s,const List&b,const List&p,const List&w,const std::vector<double>&spec,bool elimination){
    auto out=prescribed_elastic_acceleration(s,b,p,w,spec,elimination);auto extra=material::prescribed_instantaneous(s,b,p);
    for(int i=0;i<3;++i)out[i]=out[i]+extra[i];return out;
}
List prescribed_step(const List& s,const List& samples,const List& p,double h,int64_t iterations,const List& w,const std::vector<double>&spec,bool elimination) {
    TORCH_CHECK(samples.size()==20,"Expected four P/V/A/rotation/angular-rate boundary samples");
    auto b=[&](int i){return List{samples[5*i],samples[5*i+1],samples[5*i+2],samples[5*i+3],samples[5*i+4]};};
    auto end=b(3);int64_t m=p[5].size(0),v=6+3*m;
    auto x=cat({s[1].flatten(),at::zeros_like(s[2])}),done=at::zeros({},p[0].options().dtype(at::kBool));
    TORCH_CHECK(s.size()==5,"Prescribed BDF2 needs physical state and history");
    auto c=bdf_coefficients(h,s[4]);auto bdf=c[0],beta=c[1];
    auto prior_position=s[3].slice(0,0,3*m).reshape({m,3}),prior_velocity=s[3].slice(0,3*m,6*m).reshape({m,3}),prior_angle=s[3].slice(0,6*m);
    auto position_bar=s[0]+bdf*(s[0]-prior_position),velocity_bar=s[1]+bdf*(s[1]-prior_velocity),theta=s[2]+bdf*(s[2]-prior_angle);
    auto valid=at::ones_like(done);List result(s.begin(),s.begin()+3);
    for(int k=0;k<20;++k){
        auto positions=position_bar+beta*x.slice(0,0,3*m).reshape({m,3});
        for(int64_t i=0;i<iterations;++i){auto sys=prescribed_system(cat({end[0].unsqueeze(0),positions}),p);
            positions=positions+prescribed_impulse(sys[1],chain_solve(sys[2],sys[3],p[5]-sys[0],elimination),p);}
        auto q=cat({end[0].unsqueeze(0),positions});
        valid=valid & (done | (geometry_valid(q,at::matmul(end[3],p[12])) & (aerowhip_tensor::norm(q.slice(0,1)-q.slice(0,0,-1))-p[5]).abs().le(1e-10).all()));
        List current{positions,x.slice(0,0,3*m).reshape({m,3}),s[2]};
        auto full=material::supported_state(current,end,p);auto g=material::geometry(full,p);auto op=material::operators(g,p);
        auto body=velocity(full).slice(0,0,6);
        // Free-node forces from the shared load include EI, drag and gravity;
        // body wrench/inertial bias drops out when support DOFs are eliminated.
        auto f=generalized_load(full,p,at::zeros_like(p[0]),at::zeros_like(p[4]),w,spec).slice(0,6);
        auto A=beta*op.damping+at::block_diag({g.mass,beta*beta*op.twist});
        auto rhs=cat({p[6].slice(0,1).repeat_interleave(3)*velocity_bar.flatten()+beta*f,-beta*at::matmul(op.twist,theta)});
        rhs=rhs-beta*at::matmul(op.damping.slice(0,6).slice(1,0,6),body);
        auto J=cat({g.constraint.slice(1,6),at::zeros({m,m},p[0].options())},1);
        auto target=-at::matmul(g.constraint.slice(1,0,6),body);
        auto next=material::constrained_solve(A.slice(0,6).slice(1,6),rhs,J,target);
        auto converged=((next-x).abs()/(1+at::maximum(next.abs(),x.abs()))).max().le(2e-11);
        current={positions,next.slice(0,0,3*m).reshape({m,3}),theta+beta*next.slice(0,3*m)};
        auto accept=(~done) & converged;
        valid=valid & (~accept | (at::matmul(J,next)-target).abs().le(1e-9).all());
        for(size_t i=0;i<result.size();++i)result[i]=at::where(accept,current[i],result[i]);
        x=at::where(done,x,next);done=done | converged;
        if(!at::isTensorSubclassLike(done) && done.item<bool>())break;
    }
    result.push_back(cat({s[0].flatten(),s[1].flatten(),s[2]}));result.push_back(s[4]*0+h);
    return invalidate_geometry(result,done & valid);
}
List prescribed_constraints(const List& s,const List& boundary,const List& p) {
    auto sys=prescribed_system(cat({boundary[0].unsqueeze(0),s[0]}),p);
    return {sys[0]-p[5],prescribed_product(sys[1],s[1],boundary[1])};
}
} // namespace

TORCH_LIBRARY(aerowhip, m) {
    m.def("implicit_begin(Tensor[] s) -> Tensor[]");
    m.def("implicit_finish(Tensor[] s, Tensor[] work, float h) -> Tensor[]");
    m.def("implicit_iteration(Tensor[] s, Tensor[] p, Tensor thrust, Tensor torque, float h, int iterations, Tensor[] weights, float[] specification, Tensor[] work, bool elimination=False) -> Tensor[]");
    m.def("implicit_step(Tensor[] s, Tensor[] p, Tensor thrust, Tensor torque, float h, int iterations, Tensor[] weights, float[] specification, bool elimination=False) -> Tensor[]");
    m.def("material_operators(Tensor[] s, Tensor[] p) -> Tensor[]");
    m.def("elastic_acceleration(Tensor[] s, Tensor[] p, Tensor thrust, Tensor torque, Tensor[] weights, float[] specification, bool elimination=False) -> Tensor");
    m.def("geometry_valid(Tensor q, Tensor root_tangent) -> Tensor");
    m.def("root_bending(Tensor q, Tensor tangent, Tensor ei, Tensor first_rest) -> Tensor[]");
    m.def("motor_allocate(Tensor requested, Tensor[] a) -> Tensor[]");
    m.def("motor_response(Tensor initial, Tensor target, Tensor[] a, float t) -> Tensor");
    m.def("motor_response_elapsed(Tensor initial, Tensor target, Tensor[] a, Tensor t) -> Tensor");
    m.def("motor_wrench(Tensor forces, Tensor[] a) -> Tensor");
    m.def("motor_substep(Tensor[] s, Tensor initial, Tensor target, Tensor[] a, Tensor[] p, float h, Tensor start, int iterations, Tensor[] weights, float[] specification, bool elimination=False) -> Tensor[]");
    m.def("setup(Tensor[] p) -> Tensor[]");
    m.def("observe(Tensor[] s, Tensor[] p) -> Tensor[]");
    m.def("acceleration(Tensor[] s, Tensor[] p, Tensor thrust, Tensor torque, Tensor[] weights, float[] specification, bool elimination=False) -> Tensor");
    m.def("project(Tensor[] s, Tensor velocity, Tensor[] p, int iterations, bool elimination=False) -> Tensor[]");
    m.def("aerodynamic_accelerations(Tensor[] s, Tensor[] p, Tensor[] weights, float[] specification) -> Tensor[]");
    m.def("prescribed_acceleration(Tensor[] s, Tensor[] boundary, Tensor[] p, Tensor[] weights, float[] specification, bool elimination=False) -> Tensor[]");
    m.def("prescribed_step(Tensor[] s, Tensor[] samples, Tensor[] p, float h, int iterations, Tensor[] weights, float[] specification, bool elimination=False) -> Tensor[]");
    m.def("prescribed_constraints(Tensor[] s, Tensor[] boundary, Tensor[] p) -> Tensor[]");
}
TORCH_LIBRARY_IMPL(aerowhip, CompositeImplicitAutograd, m) {
    using namespace aerowhip_tensor;
    m.impl("implicit_begin",implicit_begin);m.impl("implicit_finish",implicit_finish);m.impl("implicit_iteration",implicit_iteration);
    m.impl("implicit_step",implicit_step);m.impl("material_operators",material_operators);
    m.impl("elastic_acceleration",elastic_acceleration);
    m.impl("geometry_valid",geometry_valid);
    m.impl("root_bending",root_bending);
    m.impl("motor_allocate",motor_allocate);m.impl("motor_response",motor_response);m.impl("motor_response_elapsed",motor_response_elapsed);m.impl("motor_wrench",motor_wrench);m.impl("motor_substep",motor_substep);
    m.impl("setup",setup);m.impl("observe",observe);m.impl("acceleration",acceleration);
    m.impl("project",project);m.impl("aerodynamic_accelerations",aerodynamic_accelerations);
    m.impl("prescribed_acceleration",prescribed_acceleration);m.impl("prescribed_step",prescribed_step);
    m.impl("prescribed_constraints",prescribed_constraints);
}
// These are compositions of batching-aware ATen operations, with no raw data
// access. Decompose at the Batched key as well: generic list-argument fallback
// cannot infer the batch dimensions of our state/parameter packs.
TORCH_LIBRARY_IMPL(aerowhip, FuncTorchBatched, m) {
    using namespace aerowhip_tensor;
    m.impl("implicit_begin",implicit_begin);m.impl("implicit_finish",implicit_finish);m.impl("implicit_iteration",implicit_iteration);
    m.impl("implicit_step",implicit_step);m.impl("material_operators",material_operators);
    m.impl("elastic_acceleration",elastic_acceleration);
    m.impl("geometry_valid",geometry_valid);
    m.impl("root_bending",root_bending);
    m.impl("motor_allocate",motor_allocate);m.impl("motor_response",motor_response);m.impl("motor_response_elapsed",motor_response_elapsed);m.impl("motor_wrench",motor_wrench);m.impl("motor_substep",motor_substep);
    m.impl("setup",setup);m.impl("observe",observe);m.impl("acceleration",acceleration);
    m.impl("project",project);m.impl("aerodynamic_accelerations",aerodynamic_accelerations);
    m.impl("prescribed_acceleration",prescribed_acceleration);m.impl("prescribed_step",prescribed_step);
    m.impl("prescribed_constraints",prescribed_constraints);
}
