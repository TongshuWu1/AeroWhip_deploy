// Objective isotropic bending damping in the twist-free (parallel-transport) material frame.
// Included inside aerowhip after engine math helpers. No Python dependency.
namespace material {
// Forward derivatives with respect to generalized pose increments, not finite differences.
// Current scalar derivative storage supports up to83edges (84nodes), checked explicitly.
constexpr int capacity=256;
thread_local int dim=0;
struct D {double v=0.;std::array<double,capacity> g{};D()=default;D(double x):v(x){}};
D operator+(const D&a,const D&b){D c(a.v+b.v);for(int k=0;k<dim;++k)c.g[k]=a.g[k]+b.g[k];return c;}
D operator-(const D&a,const D&b){D c(a.v-b.v);for(int k=0;k<dim;++k)c.g[k]=a.g[k]-b.g[k];return c;}
D operator*(const D&a,const D&b){D c(a.v*b.v);for(int k=0;k<dim;++k)c.g[k]=a.g[k]*b.v+a.v*b.g[k];return c;}
D operator/(const D&a,const D&b){D c(a.v/b.v);for(int k=0;k<dim;++k)c.g[k]=(a.g[k]-c.v*b.g[k])/b.v;return c;}
D root(const D&a){D c(std::sqrt(a.v));for(int k=0;k<dim;++k)c.g[k]=a.g[k]/(2*c.v);return c;}
using V=std::array<D,3>;
V plus(const V&a,const V&b){return {a[0]+b[0],a[1]+b[1],a[2]+b[2]};}
V minus(const V&a,const V&b){return {a[0]-b[0],a[1]-b[1],a[2]-b[2]};}
V times(const V&a,const D&s){return {a[0]*s,a[1]*s,a[2]*s};}
D dp(const V&a,const V&b){return a[0]*b[0]+a[1]*b[1]+a[2]*b[2];}
V cp(const V&a,const V&b){return {a[1]*b[2]-a[2]*b[1],a[2]*b[0]-a[0]*b[2],a[0]*b[1]-a[1]*b[0]};}
V transport(const V&x,const V&axis,const D&denom){return plus(plus(x,cp(axis,x)),times(cp(axis,cp(axis,x)),D(1)/denom));}
using Matrix=Vector;
double bilinear(const Vector&a,const Matrix&M,const Vector&b,int n){double z=0.;for(int i=0;i<n;++i)for(int j=0;j<n;++j)z+=a[i]*M[i*n+j]*b[j];return z;}
Vector product(const Matrix&M,const Vector&v,int rows,int cols){Vector r(rows);for(int i=0;i<rows;++i)for(int j=0;j<cols;++j)r[i]+=M[i*cols+j]*v[j];return r;}
double maxabs(const Vector&v){double x=0.;for(double y:v)x=std::max(x,std::abs(y));return x;}
Matrix factor(Matrix a,int n){for(int i=0;i<n;++i){for(int j=0;j<=i;++j){double x=a[i*n+j];for(int k=0;k<j;++k)x-=a[i*n+k]*a[j*n+k];if(i==j){require(std::isfinite(x)&&x>0.,"Nonpositive implicit-system pivot");a[i*n+j]=std::sqrt(x);}else a[i*n+j]=x/a[j*n+j];}for(int j=i+1;j<n;++j)a[i*n+j]=0.;}return a;}
Vector cholsolve(const Matrix&l,Vector b,int n){for(int i=0;i<n;++i){for(int k=0;k<i;++k)b[i]-=l[i*n+k]*b[k];b[i]/=l[i*n+i];}for(int i=n-1;i>=0;--i){for(int k=i+1;k<n;++k)b[i]-=l[k*n+i]*b[k];b[i]/=l[i*n+i];}return b;}
struct Geometry {int m,nv;Vector k,jac,dual;Matrix mass,constraint;};
Geometry geometry(const Parameters&p,const Vector&s){
    int m=int(p.rest.size()), nv=6+3*m;require(nv<=capacity,"Scalar material derivatives support at most83edges");dim=nv;
    Geometry g{m,nv,Vector(2*m),Vector(2*m*nv),Vector(m),Matrix(nv*nv),Matrix(m*nv)};
    Mat3 R=rotation(s);std::array<D,9> rd;
    for(int i=0;i<9;++i)rd[i]=D(R[i]);
    for(int k=0;k<3;++k){Vec3 axis{};axis[k]=1.;Mat3 derivative=mm(R,skew(axis));for(int i=0;i<9;++i)rd[i].g[3+k]=derivative[i];}
    auto rotate=[&](Vec3 x){V y{};for(int i=0;i<3;++i)for(int j=0;j<3;++j)y[i]=y[i]+rd[3*i+j]*D(x[j]);return y;};
    V com{};for(int i=0;i<3;++i){com[i]=D(s[i]);com[i].g[i]=1.;}
    std::vector<V> q(m+1);q[0]=plus(com,rotate(p.arm));
    for(int i=0;i<m;++i)for(int k=0;k<3;++k){q[i+1][k]=D(s[18+3*i+k]);q[i+1][k].g[6+3*i+k]=1.;}
    Vec3 reference{1.,0.,0.};if(std::abs(dot(reference,p.root_tangent_body))>.8)reference={0.,1.,0.};
    double projection=dot(reference,p.root_tangent_body);for(int j=0;j<3;++j)reference[j]-=p.root_tangent_body[j]*projection;double reference_length=norm(reference);for(double& x:reference)x/=reference_length;
    V f1=rotate(reference),f2=rotate(cross(p.root_tangent_body,reference)),a=rotate(p.root_tangent_body);
    Boundary bound=boundary(p,s);
    for(int i=0;i<m;++i){V edge=minus(q[i+1],q[i]);D len=root(dp(edge,edge));require(std::isfinite(len.v)&&len.v>1e-12,"Collapsed frame edge");V t=times(edge,D(1)/len);D denominator=D(1)+dp(a,t);require(denominator.v>1e-12,"Antiparallel material frame transport");V axis=cp(a,t),binormal=times(axis,D(2)/denominator);std::array<D,2> k{dp(f1,binormal),dp(f2,binormal)};
        for(int j=0;j<2;++j){g.k[2*i+j]=k[j].v;for(int c=0;c<nv;++c)g.jac[(2*i+j)*nv+c]=k[j].g[c];}
        g.dual[i]=i?(p.rest[i-1]+p.rest[i])/2.:p.rest[0]/2.;
        for(int j=0;j<3;++j){g.constraint[i*nv+6+3*i+j]=t[j].v;if(i)g.constraint[i*nv+6+3*(i-1)+j]=-t[j].v;else{g.constraint[j]=-t[j].v;for(int k0=0;k0<3;++k0)g.constraint[3+k0]-=t[j].v*bound.B[3*j+k0];}}
        f1=transport(f1,axis,denominator);f2=transport(f2,axis,denominator);a=t;
    }
    // Root node's translational inertia is coupled exactly once to the body.
    for(int i=0;i<3;++i){g.mass[i*nv+i]=p.mass+p.node_mass[0];for(int j=0;j<3;++j){g.mass[i*nv+3+j]=p.node_mass[0]*bound.B[3*i+j];g.mass[(3+j)*nv+i]=g.mass[i*nv+3+j];g.mass[(3+i)*nv+3+j]=p.inertia[3*i+j];for(int k=0;k<3;++k)g.mass[(3+i)*nv+3+j]+=p.node_mass[0]*bound.B[3*k+i]*bound.B[3*k+j];}}
    for(int i=0;i<m;++i)for(int j=0;j<3;++j)g.mass[(6+3*i+j)*nv+6+3*i+j]=p.node_mass[i+1];return g;
}
// Kelvin-Voigt bending damping, Rayleigh dissipation sum_i Cb/(2 l_i) |dk_i/dt|^2 on the
// material curvature (k1, k2) in the frame parallel transported from the clamp. Objective:
// a rigid motion of drone and cable together leaves (k1, k2) unchanged. Returns D (nv x nv).
// `scale` (optional, one factor per joint) multiplies each joint's coefficient.
Matrix operators(const Geometry&g,double cb,const Vector&scale={}){int m=g.m,v=g.nv;Matrix damping(v*v);
    for(int i=0;i<m;++i){double weight=cb/g.dual[i]*(scale.empty()?1.:scale[i]);
        for(int component=0;component<2;++component){const double*row=&g.jac[(2*i+component)*v];
            for(int r=0;r<v;++r)if(row[r]!=0.)for(int c=0;c<v;++c)damping[r*v+c]+=weight*row[r]*row[c];}}
    return damping;}
// Saturating bending damping: with a finite saturation rate r_s (rad/s) each joint's coefficient
// is Cb / sqrt(1 + (|dk_i/dt| / r_s)^2) -- linear for slow bending, a bounded damping moment
// (Cb r_s / l_i) for fast bending, as friction-like internal damping of a braided cable behaves.
// The factors are evaluated on a given state (the start of an integration step: the damping stays
// implicit in velocity with lagged coefficients). Empty when the damping is linear.
Vector saturation(const Parameters&p,const Vector&s){
    if(!std::isfinite(p.bending_damping_saturation))return {};
    auto g=geometry(p,s);int m=g.m,v=g.nv;auto u=velocity(s,m);Vector scale(m);
    for(int i=0;i<m;++i){double r2=0.;for(int component=0;component<2;++component){const double*row=&g.jac[(2*i+component)*v];
            double r=0.;for(int c=0;c<v;++c)r+=row[c]*u[c];r2+=r*r;}
        scale[i]=1./std::sqrt(1.+r2/(p.bending_damping_saturation*p.bending_damping_saturation));}
    return scale;}
// Continuous material contribution for telemetry, not a separate integration kick.
Vector instantaneous(const Parameters&p,const Vector&s){
    auto g=geometry(p,s); int v=g.nv,m=g.m; auto u=velocity(s,m);
    auto D=operators(g,p.bending_damping,saturation(p,s)); Vector f(v);
    for(int i=0;i<v;++i)for(int j=0;j<v;++j)f[i]-=D[i*v+j]*u[j];
    auto L=factor(g.mass,v),a=cholsolve(L,f,v); Matrix response(v*m),schur(m*m);
    for(int j=0;j<m;++j){Vector col(v);for(int i=0;i<v;++i)col[i]=g.constraint[j*v+i];
        col=cholsolve(L,col,v);for(int i=0;i<v;++i)response[i*m+j]=col[i];
        for(int k=0;k<m;++k)for(int i=0;i<v;++i)schur[k*m+j]+=g.constraint[k*v+i]*col[i];}
    auto lambda=cholsolve(factor(schur,m),product(g.constraint,a,m,v),m);
    for(int i=0;i<v;++i)for(int j=0;j<m;++j)a[i]-=response[i*m+j]*lambda[j];
    return a;
}
}
