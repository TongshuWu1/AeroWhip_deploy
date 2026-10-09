// Objective material-frame Kelvin--Voigt operators. Included in tensor_physics.cpp.
// Analytic forward pose derivatives expressed in ATen: differentiable, vmap-safe,
// no host scalar extraction, numerical differentiation or Python physical solver.
namespace material {
struct V { Tensor x,j; };
struct D { Tensor x,j; };
V plus(const V&a,const V&b){return {a.x+b.x,a.j+b.j};}
V minus(const V&a,const V&b){return {a.x-b.x,a.j-b.j};}
V times(const V&a,const D&b){return {a.x*b.x,a.j*b.x+a.x.unsqueeze(-1)*b.j};}
D dp(const V&a,const V&b){return {(a.x*b.x).sum(),(a.j*b.x.unsqueeze(-1)+b.j*a.x.unsqueeze(-1)).sum(0)};}
V cp(const V&a,const V&b){return {vector_cross(a.x,b.x),
    (vector_cross(a.j.t(),b.x.expand_as(a.j.t()))+vector_cross(a.x.expand_as(b.j.t()),b.j.t())).t()};}
D inv(const D&a){return {1/a.x,-a.j/a.x.square()};}
V transport(const V&x,const V&axis,const D&denom){return plus(plus(x,cp(axis,x)),times(cp(axis,cp(axis,x)),inv(denom)));}
struct Geometry {Tensor k,jac,dual,mass,constraint;};
Geometry geometry(const List&s,const List&p){
    int64_t m=p[5].size(0),v=6+3*m;
    auto eye=at::eye(v,p[0].options()),zero=at::zeros({3,v},p[0].options());
    auto rotate=[&](const Tensor&x){return V{at::matmul(s[1],x),
        cat({at::zeros({3,3},x.options()),-at::matmul(s[1],skew(x)),at::zeros({3,3*m},x.options())},1)};};
    auto B=boundary(s,p);auto o=observe(s,p); auto q=o[2];
    List qjac{cat({B,at::zeros({3,3*m},p[0].options())},1)};
    for(int64_t i=0;i<m;++i)qjac.push_back(eye.slice(0,6+3*i,9+3*i));
    // Same deterministic root reference as the scalar implementation.
    auto e=at::eye(3,p[0].options()),reference=at::where(p[12][0].abs()>.8,e[1],e[0]);
    reference=reference-at::dot(reference,p[12])*p[12]; reference=reference/aerowhip_tensor::norm(reference);
    V f1=rotate(reference),f2=rotate(vector_cross(p[12],reference)),a=rotate(p[12]);
    List curvature,jacobian,constraints;
    for(int64_t i=0;i<m;++i){
        V edge{q[i+1]-q[i],qjac[i+1]-qjac[i]}; auto square=dp(edge,edge);
        auto length=positive_geometry_or_nan(at::sqrt(square.x));
        D len{length,square.j/(2*length)}; V t=times(edge,inv(len));
        auto cosine=dp(a,t);D den{positive_geometry_or_nan(1+cosine.x),cosine.j};
        auto axis=cp(a,t);auto inverse=inv(den);inverse.x=2*inverse.x;inverse.j=2*inverse.j;
        auto binormal=times(axis,inverse);auto k1=dp(f1,binormal),k2=dp(f2,binormal);
        curvature.push_back(at::stack({k1.x,k2.x}));jacobian.push_back(at::stack({k1.j,k2.j}));
        constraints.push_back(at::matmul(t.x,edge.j));
        f1=transport(f1,axis,den);f2=transport(f2,axis,den);a=t;
    }
    auto dual=cat({(p[5][0]/2).reshape({1}),(p[5].slice(0,0,-1)+p[5].slice(0,1))/2});
    auto body=p[11]+p[6][0]*at::matmul(B.t(),B);
    auto mass=at::block_diag({body,at::diag(p[6].slice(0,1).repeat_interleave(3))});
    return {at::stack(curvature),at::stack(jacobian),dual,mass,at::stack(constraints)};
}
struct Operators{Tensor damping,twist;};
Operators operators(const Geometry&g,const List&p){
    int64_t m=p[5].size(0);auto e=at::eye(m,p[0].options());
    auto previous=cat({at::zeros_like(e.slice(0,0,1)),e.slice(0,0,-1)});
    auto difference=e-previous;auto twist=at::matmul(difference.t(),(p[14]/g.dual).unsqueeze(1)*difference);
    // Preserve the original (edge, incident side, curvature component) row
    // order, but assemble all rows together. This is the same Rayleigh matrix
    // L^T W L, not a different damping law or a detached/approximate gradient.
    auto linear=g.jac.unsqueeze(1).expand({m,2,2,g.jac.size(2)});
    auto angles=at::stack({previous,e},1).unsqueeze(2);
    auto curvature=at::stack({g.k.select(1,1),-g.k.select(1,0)},1).unsqueeze(1).unsqueeze(-1);
    auto L=cat({linear,angles*curvature},-1).reshape({4*m,g.jac.size(2)+m});
    auto weights=(p[13]/(2*g.dual)).repeat_interleave(4);
    auto damping=at::matmul(L.t(),weights.unsqueeze(1)*L);
    auto angular=at::matmul(difference.t(),(p[15]/g.dual).unsqueeze(1)*difference);
    auto z=at::zeros({g.mass.size(0),g.mass.size(0)},p[0].options());
    return {damping+at::block_diag({z,angular}),twist};
}
Tensor constrained_solve(const Tensor&A,const Tensor&rhs,const Tensor&J,const Tensor&target){
    // Solve all RHS together, avoiding one factorization per constraint.
    auto all=solve(A,cat({rhs.unsqueeze(1),J.t()},1));auto x=all.select(1,0),response=all.slice(1,1);
    auto schur=at::matmul(J,response);
    auto lambda=solve((schur+schur.t())/2,at::matmul(J,x)-target);
    return x-at::matmul(response,lambda);
}
Tensor instantaneous(const List&s,const List&p){
    auto g=geometry(s,p);auto op=operators(g,p);int64_t v=g.mass.size(0);
    auto Cuu=op.damping.slice(0,0,v).slice(1,0,v),Cu=op.damping.slice(0,0,v).slice(1,v);
    auto Caa=op.damping.slice(0,v).slice(1,v),u=velocity(s);
    auto rate=solve(Caa,-at::matmul(Cu.t(),u)-at::matmul(op.twist,s[6]));
    auto f=-at::matmul(Cuu,u)-at::matmul(Cu,rate);
    return constrained_solve(g.mass,f,g.constraint,at::zeros({g.constraint.size(0)},p[0].options()));
}
// Prescribed root uses the identical operators. Body coordinates are support
// coordinates only: eliminating them analytically removes their mass/inertia.
List supported_state(const List&s,const List&b,const List&p){
    // b: root P/V/A, body-to-world R, body angular velocity.
    return {b[0]-at::matmul(b[3],p[3]),b[3],
            b[1]-at::matmul(b[3],vector_cross(b[4],p[3])),b[4],s[0],s[1],s[2]};
}
List prescribed_instantaneous(const List&s,const List&b,const List&p){
    auto full=supported_state(s,b,p);auto g=geometry(full,p);auto op=operators(g,p);
    int64_t v=g.mass.size(0),m=p[5].size(0);
    auto Cuu=op.damping.slice(0,0,v).slice(1,0,v),Cu=op.damping.slice(0,0,v).slice(1,v);
    auto rate=solve(op.damping.slice(0,v).slice(1,v),
        -at::matmul(Cu.t(),velocity(full))-at::matmul(op.twist,s[2]));
    auto f=-at::matmul(Cuu,velocity(full))-at::matmul(Cu,rate);
    auto J=g.constraint.slice(1,6),mass=g.mass.slice(0,6).slice(1,6);
    auto a=constrained_solve(mass,f.slice(0,6),J,at::zeros({m},p[0].options()));
    // Recover constraint multipliers to report the cable-on-support reaction.
    auto response=solve(mass,J.t()),schur=at::matmul(J,response);
    auto lambda=solve(schur,at::matmul(J,solve(mass,f.slice(0,6))));
    auto support=f.slice(0,0,6)-at::matmul(g.constraint.slice(1,0,6).t(),lambda);
    auto force=support.slice(0,0,3);
    // Report direct world couple, excluding force moment about the COM.
    auto moment=at::matmul(b[3],support.slice(0,3,6)-at::matmul(boundary(full,p).slice(1,3).t(),force));
    return {a.reshape({m,3}),force,moment};
}
} // namespace material
