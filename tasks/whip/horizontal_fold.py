"""Ten-link material fold geometry shared by current reward and diagnostics."""
import numpy as np
from .fold_geometry import regions

class FoldGeometry:
    """Causal port of the reviewed unique half-height-core overlap tracker."""
    def __init__(self,rest):
        self.rest=np.asarray(rest,float)
        if self.rest.shape!=(10,) or not np.isfinite(self.rest).all() or np.any(self.rest<=0):
            raise ValueError('Horizontal fold requires ten positive finite rest lengths')
        self.L=float(self.rest.sum());self.nodes=np.r_[0.,np.cumsum(self.rest)]
        self.grid=np.linspace(self.nodes[1],self.nodes[-2],1001)
        self.bandwidth=.075*self.L/.9525
        kernel=np.exp(-.5*((self.grid[None]-self.nodes[1:-1,None])/self.bandwidth)**2)
        self.kernel=kernel/np.trapezoid(kernel,self.grid,axis=1)[:,None]
        self.previous=[];self.next_id=0

    def observe(self,p):
        p=np.asarray(p,float)
        if p.shape!=(11,3) or not np.isfinite(p).all():raise ValueError('Finite eleven-node geometry required')
        edge=np.diff(p,axis=0);length=np.linalg.norm(edge,axis=-1)
        if np.any(length<=0):raise ValueError('Collapsed cable segment')
        u=edge/length[:,None];cross=np.cross(u[:-1],u[1:]);norm=np.linalg.norm(cross,axis=-1)
        dot=np.sum(u[:-1]*u[1:],axis=-1);turn=np.arctan2(norm,dot)
        axes=np.divide(cross,norm[:,None],out=np.zeros_like(cross),where=norm[:,None]>1e-10)
        current=regions(self.grid,turn@self.kernel,self.bandwidth)
        left=[[j for j,b in enumerate(current) if max(a['core_m'][0],b['core_m'][0])<min(a['core_m'][1],b['core_m'][1])] for a in self.previous]
        right=[[i for i,js in enumerate(left) if j in js] for j in range(len(current))]
        for j,b in enumerate(current):
            unique=len(right[j])==1 and len(left[right[j][0]])==1
            clean=unique and 'plateau' not in b['flags'] and 'plateau' not in self.previous[right[j][0]]['flags']
            if clean:b['track_id']=self.previous[right[j][0]]['track_id']
            else:
                b['track_id']=self.next_id;self.next_id+=1
                if right[j]:b['flags'].append('ambiguous_identity')
            ix=np.flatnonzero((self.nodes[1:-1]>=b['basin_m'][0])&(self.nodes[1:-1]<=b['basin_m'][1]))
            theta=float(turn[ix].sum());unresolved=bool(np.any((norm[ix]<=1e-10)&(dot[ix]<0)))
            c=float(np.clip(np.linalg.norm((turn[ix,None]*axes[ix]).sum(0))/theta,0,1)) if theta>0 and not unresolved else 0.
            b.update(joint_indices=(ix+1).tolist(),theta=theta,coherence=c,unresolved=unresolved)
        self.previous=current
        return turn,current
