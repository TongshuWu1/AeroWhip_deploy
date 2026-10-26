"""Material-coordinate fold measurement on the ten-link cable mesh.

regions() finds bend basins in a smoothed turning-angle profile; FoldGeometry
tracks them causally through time. This is a measurement of the simulated or
recorded mesh, not a finer physical model. Bandwidth and grid are fixed parts of
this measurement definition, not task settings.
"""
import numpy as np

def regions(grid, row, bandwidth):
    """Local basins separated by minima; centre/width weighted by turning.

    The half-height interval within each basin is used only for matching. Flat
    whole profiles have no localized bend. Numeric flat maxima use their
    midpoint and are explicitly flagged. No physical amplitude cutoff exists.
    """
    x, y = np.asarray(grid), np.asarray(row)
    floor = 128*np.finfo(float).eps*max(1., float(y.max()))
    if y.max() <= floor or np.ptp(y) <= floor:
        return []
    groups = []; begin = 0
    for end in range(1, len(y)+1):
        if end == len(y) or abs(y[end]-y[end-1]) > floor:
            groups.append((begin, end)); begin = end
    peaks = []
    for lo, hi in groups:
        level = float(np.min(y[lo:hi]))
        if (level > floor and (lo == 0 or level > y[lo-1]+floor)
                and (hi == len(y) or level > y[hi]+floor)):
            peaks.append(((lo+hi-1)//2, hi-lo > 1))
    if not peaks:
        return []
    cuts = [0]+[a+int(np.argmin(y[a:b+1])) for (a, _), (b, _) in zip(peaks, peaks[1:])]+[len(y)-1]
    result = []
    for k, (peak, plateau) in enumerate(peaks):
        lo, hi = cuts[k:k+2]
        xx, yy = x[lo:hi+1], y[lo:hi+1]
        mass = float(np.trapezoid(yy, xx))
        if mass <= floor*(x[-1]-x[0]):
            continue
        centre = float(np.trapezoid(xx*yy, xx)/mass)
        width = float(np.sqrt(max(0., np.trapezoid((xx-centre)**2*yy, xx)/mass)))
        left = peak
        while left > lo and y[left-1] >= .5*y[peak]:
            left -= 1
        right = peak
        while right < hi and y[right+1] >= .5*y[peak]:
            right += 1
        flags = []
        if plateau:
            flags.append('plateau')
        if x[peak]-x[0] < 2*bandwidth or x[-1]-x[peak] < 2*bandwidth:
            flags.append('boundary_affected')
        result.append(dict(centre_m=centre, width_m=width, turn_rad=mass,
                           peak_m=float(x[peak]), core_m=[float(x[left]), float(x[right])],
                           basin_m=[float(xx[0]), float(xx[-1])], flags=flags))
    return result


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
