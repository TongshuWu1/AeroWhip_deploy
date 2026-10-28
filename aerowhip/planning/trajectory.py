"""Free excitation trajectories: a sum of sines per axis under a smooth window, so the drone starts
and ends at rest at the hover point. The coefficients (amplitude, frequency, phase per term and
axis) are what the optimiser varies; `samples()` writes the file the flight script flies
(`--trajectory FILE`), and `motion()` is the same trajectory as a motion object for the simulator."""
import json
import math

import numpy as np


class FourierTrajectory:
    """Per axis K terms: p(t) = w(t) sum_k A_k sin(2 pi f_k t + phi_k), with w the quintic fade in and
    out over `fade` s. theta is the flat vector [A, f, phi] x 3 axes x K terms."""
    def __init__(self, terms=4, duration=60., fade=4., f_range=(.1, 2.5), amplitude_max=(.6, .6, .3), name='excitation'):
        self.K, self.duration, self.fade, self.f_range = terms, float(duration), float(fade), f_range
        self.amplitude_max = np.asarray(amplitude_max, float)
        self.name = name

    @property
    def size(self):
        return 3*self.K*3

    def bounds(self):
        lo, hi = [], []
        for axis in range(3):
            lo += [0.]*self.K+[self.f_range[0]]*self.K+[-math.pi]*self.K
            hi += [self.amplitude_max[axis]]*self.K+[self.f_range[1]]*self.K+[math.pi]*self.K
        return np.array(lo), np.array(hi)

    def random(self, rng, scale=.5):
        lo, hi = self.bounds()
        theta = lo+(hi-lo)*rng.uniform(0., 1., self.size)
        for axis in range(3):                        # amplitudes: smaller on average, falling with frequency
            k = axis*3*self.K
            theta[k:k+self.K] *= scale*rng.uniform(.2, 1., self.K)
        return theta

    def unpack(self, theta):
        theta = np.asarray(theta, float).reshape(3, 3, self.K)
        return theta[:, 0], theta[:, 1], theta[:, 2]            # A (3,K), f (3,K), phi (3,K)

    def window(self, t):
        T, F = self.duration, self.fade
        u = np.clip(t/F, 0., 1.); d = np.clip((T-t)/F, 0., 1.)
        def q(x): return 10*x**3-15*x**4+6*x**5
        def dq(x): return (30*x**2-60*x**3+30*x**4)/F
        def ddq(x): return (60*x-180*x**2+120*x**3)/F**2
        w = np.where(t < F, q(u), np.where(t > T-F, q(d), 1.))
        dw = np.where(t < F, dq(u), np.where(t > T-F, -dq(d), 0.))
        ddw = np.where(t < F, ddq(u), np.where(t > T-F, ddq(d), 0.))
        return w, dw, ddw

    def samples(self, theta, dt=.02):
        """(t, p, v, a) arrays, p relative to the hover point."""
        A, f, phi = self.unpack(theta)
        t = np.arange(0., self.duration+dt/2, dt)
        w, dw, ddw = self.window(t)
        p = np.zeros((len(t), 3)); v = np.zeros_like(p); a = np.zeros_like(p)
        for axis in range(3):
            om = 2*math.pi*f[axis][:, None]
            arg = om*t[None, :]+phi[axis][:, None]
            q = (A[axis][:, None]*np.sin(arg)).sum(0)
            dq = (A[axis][:, None]*om*np.cos(arg)).sum(0)
            ddq = -(A[axis][:, None]*om**2*np.sin(arg)).sum(0)
            p[:, axis] = w*q; v[:, axis] = dw*q+w*dq; a[:, axis] = ddw*q+2*dw*dq+w*ddq
        return t, p, v, a

    def normalise(self, theta, level, vertical_max=3., speed_max=3.):
        """Scale the horizontal amplitudes so the peak horizontal acceleration equals `level` (m/s^2),
        the vertical ones so the vertical acceleration stays within `vertical_max`, then all so the
        speed stays within `speed_max`."""
        theta = np.asarray(theta, float).reshape(3, 3, self.K).copy()
        t, p, v, a = self.samples(theta.ravel())
        h = np.linalg.norm(a[:, :2], axis=1).max()
        if h > 0:
            theta[:2, 0] *= level/h
        t, p, v, a = self.samples(theta.ravel())
        z = np.abs(a[:, 2]).max()
        if z > vertical_max:
            theta[2, 0] *= vertical_max/z
        t, p, v, a = self.samples(theta.ravel())
        sp = np.linalg.norm(v, axis=1).max()
        if sp > speed_max:
            theta[:, 0] *= speed_max/sp
        return theta.ravel()

    def motion(self, theta):
        from aerowhip.planning.simulate import flight_script
        _, trajectories = flight_script()
        t, p, v, a = self.samples(theta)
        return trajectories.TrajectoryFile(dict(t=t, position=p, velocity=v, acceleration=a), self.name)

    def write(self, theta, path, extra=None):
        t, p, v, a = self.samples(theta)
        data = dict(name=self.name, t=t.tolist(), position=p.tolist(), velocity=v.tolist(), acceleration=a.tolist(),
                    theta=np.asarray(theta, float).tolist(), terms=self.K, duration=self.duration, fade=self.fade)
        if extra:
            data.update(extra)
        path.write_text(json.dumps(data))

    def summary(self, theta):
        t, p, v, a = self.samples(theta)
        return dict(peak_speed=float(np.linalg.norm(v, axis=1).max()), peak_acceleration=float(np.linalg.norm(a[:, :2], axis=1).max()),
                    peak_vertical=float(np.abs(a[:, 2]).max()), extent=np.round(p.max(0)-p.min(0), 2).tolist())
