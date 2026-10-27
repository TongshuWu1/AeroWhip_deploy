// Native prescribed-boundary execution of the SAME material model.
// Body coordinates are eliminated exactly, never approximated by a large mass.
namespace prescribed {
using material::Matrix;
struct BoundarySample { Vec3 p,v,a; Mat3 R; Vec3 omega; };
BoundarySample sample(const Vector& b,size_t k=0) {
    require(b.size()>=k+21,"Invalid prescribed boundary width");
    for(size_t i=k;i<k+21;++i)require(std::isfinite(b[i]),"Nonfinite prescribed boundary");
    BoundarySample out{at(b,k),at(b,k+3),at(b,k+6),{},at(b,k+18)};
    std::copy_n(b.begin()+k+9,9,out.R.begin());
    auto gram=mm(transpose(out.R),out.R),I=identity();
    for(int i=0;i<9;++i)require(std::abs(gram[i]-I[i])<=1e-8,"Invalid boundary rotation");
    require(std::abs(determinant(out.R)-1.)<=1e-8,"Improper boundary rotation");return out;
}
Vector supported(const Parameters&p,const Vector&s,const BoundarySample&b) {
    size_t m=p.rest.size();require(s.size()==6*m&&finite(s),"Invalid prescribed material state");
    Vector full(18+6*m);put(full,0,b.p-mv(b.R,p.arm));std::copy(b.R.begin(),b.R.end(),full.begin()+3);
    put(full,12,b.v-mv(b.R,cross(b.omega,p.arm)));put(full,15,b.omega);
    std::copy(s.begin(),s.end(),full.begin()+18);return full;
}
Chain system(const Parameters&p,const std::vector<Vec3>&q) {
    size_t m=p.rest.size();Chain c{Vector(m),Vector(m),Vector(m-1),Vector(6),std::vector<Vec3>(m)};
    for(size_t i=0;i<m;++i){auto e=q[i+1]-q[i];c.lengths[i]=norm(e);
        require(std::isfinite(c.lengths[i])&&c.lengths[i]>1e-12,"Collapsed prescribed edge");
        c.tangent[i]=e/c.lengths[i];
        if(i)require(1.+dot(c.tangent[i-1],c.tangent[i])>1e-12,"Antiparallel prescribed edge");
        c.diagonal[i]=dot(c.tangent[i],c.tangent[i])*(1./p.node_mass[i+1]+(i?1./p.node_mass[i]:0.));
        if(i)c.off[i-1]=-dot(c.tangent[i-1],c.tangent[i])/p.node_mass[i];
    }return c;
}
Vector product(const Chain&c,const Vector&free,Vec3 root) {
    Vector out(c.lengths.size());
    for(size_t i=0;i<out.size();++i){auto right=at(free,3*i);out[i]=dot(c.tangent[i],right-root);root=right;}return out;
}
Vector free_impulse(const Parameters&p,const Chain&c,const Vector&lambda) {
    Vector out(3*lambda.size());
    for(size_t i=0;i<lambda.size();++i){auto f=c.tangent[i]*lambda[i];if(i+1<lambda.size())f=f-c.tangent[i+1]*lambda[i+1];put(out,3*i,f/p.node_mass[i+1]);}return out;
}
struct Acceleration { Vector free; Vec3 force,moment; };
Acceleration elastic(const Parameters&p,const Vector&s,const BoundarySample&b) {
    size_t m=p.rest.size();auto full=supported(p,s,b);std::vector<Vec3>q,v;cable(p,full,q,v);
    auto c=system(p,q);auto f=cable_forces(p,full,c,v);Vector a(3*m);
    for(size_t i=0;i<m;++i)put(a,3*i,f.force[i+1]/p.node_mass[i+1]);
    auto rhs=product(c,a,b.a);
    for(size_t i=0;i<m;++i){auto relative=v[i+1]-v[i],transverse=relative-c.tangent[i]*dot(relative,c.tangent[i]);rhs[i]+=dot(transverse,transverse)/c.lengths[i];}
    auto lambda=solve(c,rhs);return {add_scaled(a,free_impulse(p,c,lambda),-1.),f.force[0]+c.tangent[0]*lambda[0]-b.a*p.node_mass[0],f.moment};
}
Vector constrained(const Matrix&A,const Vector&rhs,const Matrix&J,const Vector&target,int n,int m) {
    auto L=material::factor(A,n),x=material::cholsolve(L,rhs,n);Matrix response(n*m),S(m*m);
    for(int j=0;j<m;++j){Vector col(n);for(int i=0;i<n;++i)col[i]=J[j*n+i];col=material::cholsolve(L,col,n);
        for(int i=0;i<n;++i)response[i*m+j]=col[i];
        for(int k=0;k<m;++k)for(int i=0;i<n;++i)S[k*m+j]+=J[k*n+i]*col[i];}
    for(int i=0;i<m;++i)for(int j=0;j<i;++j)S[i*m+j]=S[j*m+i]=.5*(S[i*m+j]+S[j*m+i]);
    auto violation=add_scaled(material::product(J,x,m,n),target,-1.);
    auto lambda=material::cholsolve(material::factor(S,m),violation,m);
    for(int i=0;i<n;++i)for(int j=0;j<m;++j)x[i]-=response[i*m+j]*lambda[j];return x;
}
Vector endpoint_solve(const Parameters&p,const Vector&s,const Vector&old,const BoundarySample&b,double h) {
    auto full=supported(p,s,b);auto g=material::geometry(p,full);
    auto D=material::operators(g,p.bending_damping);
    int m=g.m,v=g.nv,nf=v-6;auto u=velocity(full,m);
    Vector rhs(nf),target(m);
    std::vector<Vec3>q,qv;cable(p,full,q,qv);auto c=system(p,q);
    auto f=cable_forces(p,full,c,qv);
    Matrix A(nf*nf),J(m*nf);
    // Eliminate known root/body velocity from the full implicit system.
    for(int i=6;i<v;++i){rhs[i-6]=p.node_mass[(i-6)/3+1]*old[3*m+i-6]+h*f.force[(i-6)/3+1][(i-6)%3];
        for(int j=0;j<6;++j)rhs[i-6]-=h*D[i*v+j]*u[j];
        for(int j=6;j<v;++j)A[(i-6)*nf+j-6]=h*D[i*v+j]+g.mass[i*v+j];}
    for(int i=0;i<m;++i){for(int j=0;j<6;++j)target[i]-=g.constraint[i*v+j]*u[j];
        for(int j=6;j<v;++j)J[i*nf+j-6]=g.constraint[i*v+j];}
    return constrained(A,rhs,J,target,nf,m);
}
Acceleration material_acceleration(const Parameters&p,const Vector&s,const BoundarySample&b) {
    auto full=supported(p,s,b);auto g=material::geometry(p,full);
    auto D=material::operators(g,p.bending_damping);
    int m=g.m,v=g.nv;auto u=velocity(full,m);Vector f(v);
    for(int i=0;i<v;++i)for(int j=0;j<v;++j)f[i]-=D[i*v+j]*u[j];
    std::vector<Vec3>q,qv;cable(p,full,q,qv);auto c=system(p,q);Vector a(3*m);
    for(int i=0;i<m;++i)put(a,3*i,at(f,6+3*i)/p.node_mass[i+1]);
    auto lambda=solve(c,product(c,a,Vec3{}));Vector support(f.begin(),f.begin()+6);
    for(int i=0;i<6;++i)for(int j=0;j<m;++j)support[i]-=g.constraint[j*v+i]*lambda[j];
    auto force=at(support,0),moment=mv(b.R,at(support,3)-mv(transpose(boundary(p,full).B),force));
    return {add_scaled(a,free_impulse(p,c,lambda),-1.),force,moment};
}
Vector step(const Parameters&p,const Vector&s,const std::array<BoundarySample,4>&b,double h,const Vector&previous,double previous_step) {
    // Same projected implicit endpoint, with support coordinates eliminated.
    // Keep the four-sample wire format; only the exact endpoint drives the step.
    int m=int(p.rest.size());Vector x(3*m),bar=s;std::copy_n(s.begin()+3*m,3*m,x.begin());
    auto coefficients=bdf_coefficients(h,previous_step);double beta=coefficients.beta;
    for(int i=0;i<6*m;++i)bar[i]+=coefficients.memory*(s[i]-previous[i]);
    for(int iteration=0;iteration<20;++iteration){Vector out=bar;
        for(int i=0;i<3*m;++i){out[i]+=beta*x[i];out[3*m+i]=x[i];}
        std::vector<Vec3>q,qv;
        for(int k=0;k<p.projections;++k){cable(p,supported(p,out,b[3]),q,qv);auto c=system(p,q);Vector rhs(m);
            for(int i=0;i<m;++i)rhs[i]=p.rest[i]-c.lengths[i];auto delta=free_impulse(p,c,solve(c,rhs));
            for(int i=0;i<3*m;++i)out[i]+=delta[i];}
        cable(p,supported(p,out,b[3]),q,qv);auto c=system(p,q);
        for(int i=0;i<m;++i)require(std::abs(c.lengths[i]-p.rest[i])<=1e-10,"Prescribed implicit position projection failed");
        auto next=endpoint_solve(p,out,bar,b[3],beta);double err=0.;
        for(int i=0;i<3*m;++i)err=std::max(err,std::abs(next[i]-x[i])/(1.+std::max(std::abs(next[i]),std::abs(x[i]))));
        x=std::move(next);if(err>2e-11)continue;
        std::copy_n(x.begin(),3*m,out.begin()+3*m);
        require(material::maxabs(product(c,x,b[3].v))<=1e-9,"Prescribed implicit velocity constraint failed");
        require(finite(out),"Nonfinite prescribed implicit endpoint");return out;
    }
    throw std::runtime_error("Prescribed implicit endpoint iteration failed after 20 iterations");
}
} // namespace prescribed

Vector prescribed_advance(const Model&model,Vector s,const Vector&samples,Vector&previous,double&previous_step) {
    const auto&p=model.parameters;
    require(!samples.empty()&&samples.size()%85==0,"Expected substep rows: h plus four21-value boundary samples");
    require(previous.size()==s.size()&&finite(previous)&&std::isfinite(previous_step)&&previous_step>=0.
            &&previous_step<=p.maximum_step*(1.+1e-9),"Invalid prescribed BDF2 history");
    Engine validator(p);validator.reset(prescribed::supported(p,s,prescribed::sample(samples,1)));
    for(size_t offset=0;offset<samples.size();offset+=85){double h=samples[offset];
        require(std::isfinite(h)&&h>0.&&h<=p.maximum_step*(1.+1e-9),"Invalid prescribed integration step");
        std::array<prescribed::BoundarySample,4>b;
        for(int i=0;i<4;++i)b[i]=prescribed::sample(samples,offset+1+21*i);
        auto next=prescribed::step(p,s,b,h,previous,previous_step);previous=std::move(s);s=std::move(next);previous_step=h;
    }
    validator.reset(prescribed::supported(p,s,prescribed::sample(samples,samples.size()-21)));return s;
}
Vector prescribed_acceleration(const Model&model,const Vector&s,const Vector&sample) {
    const auto&p=model.parameters;
    auto b=prescribed::sample(sample);Engine validator(p);validator.reset(prescribed::supported(p,s,b));
    auto a=prescribed::elastic(p,s,b),extra=prescribed::material_acceleration(p,s,b);
    Vector out=add_scaled(a.free,extra.free,1.);auto f=a.force+extra.force,t=a.moment+extra.moment;
    out.insert(out.end(),f.begin(),f.end());out.insert(out.end(),t.begin(),t.end());return out;
}
