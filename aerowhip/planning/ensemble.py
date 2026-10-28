"""The model ensemble: parameter sets around the adopted fits, drawn from the fits' clustered
uncertainties (standard deviations and correlations of their search vectors), plus deliberate
bracket members that widen what is in doubt. Every member is a full model configuration the
batch runtime and the replays accept.

The ensemble must bracket the measured behaviour (scripts/bracket_check.py): on the flights that
flipped the cable some members must put the cable over the drone, or the ensemble is too narrow
to be trusted for planning near the limit.
"""
from copy import deepcopy
import json
from pathlib import Path
import sys

import numpy as np

from aerowhip.fitting import flight_drone
from aerowhip.fitting.flight_cable import with_cable
from aerowhip.paths import ROOT

sys.path.insert(0, str(ROOT/'scripts'))

# Bracket members: multiplicative changes of the cable's physics that make the cable swing more or
# less, applied on top of the adopted values. Named so the check can say which bracket reached the
# measured rise.
BRACKETS = {
    'adopted': {},
    'damping/3': dict(Cb_n_m2_s=1/3),
    'damping/10': dict(Cb_n_m2_s=1/10),
    'stiff x3': dict(EI_n_m2=3.),
    'stiff x3, damping/3': dict(EI_n_m2=3., Cb_n_m2_s=1/3),
    'drag/2': dict(normal_drag_coefficient=.5, external_drag_s_inv=.5),
    'drag/2, damping/3': dict(normal_drag_coefficient=.5, external_drag_s_inv=.5, Cb_n_m2_s=1/3),
    'residual x0': dict(residual=0.),
    'residual x2': dict(residual=2.),
}


def load_fit(path):
    return json.loads(Path(path).read_text())


class Ensemble:
    def __init__(self, base_config, cable_fit, drone_fit=None):
        """base_config: the model with cf_6's geometry; cable_fit / drone_fit: the fit JSONs whose
        adopted values and uncertainties define the distribution."""
        self.base = deepcopy(base_config)
        self.cable_fit, self.drone_fit = cable_fit, drone_fit
        import fit_cable_segments as fcs
        self.parameters = fcs.Parameters(cable_fit['terms'], cable_fit['frame'], vertical=cable_fit.get('vertical', 'quadratic'))
        self.parameters.start(self.base)          # records the held (zeroed) terms the search vector leaves out

    def cable_config(self, x, scale=None):
        """Config at cable search vector x; `scale` multiplies physical values (and the residual)."""
        c = self.parameters.config(self.base, np.asarray(x, float))
        if scale:
            for key, k in scale.items():
                if key == 'residual':
                    r = c['cable'].get('residual')
                    if r:
                        r['static_n_per_m'] = (np.asarray(r['static_n_per_m'])*k).tolist()
                        for name in ('normal_drag_log', 'tangential_drag'):
                            if name in r:
                                r[name] = (np.asarray(r[name])*k).tolist()
                else:
                    c['cable'][key] = float(c['cable'][key])*k
        return c

    def drone_config(self, c, x):
        """x = (inertia scale, motor response s, body drag xy, body drag z); the delay from the fit."""
        if self.drone_fit is None:
            return c
        delay = self.drone_fit['best']['command_delay_s']
        c = flight_drone.vehicle(c, [x[0], 1., x[1], delay], cable=True)
        c['drone']['body_drag_s_inv'] = [float(x[2]), float(x[2]), float(x[3])]
        return c

    def samples(self, count, seed=0, sigma=1., inertia_range=(.7, 1.4)):
        """`count` members drawn from the fits' uncertainties (Gaussian in the search space, `sigma`
        standard deviations), the roll/pitch inertia uniform over `inertia_range` (not identifiable).
        Returns [(label, config)]; the first member is the adopted model."""
        rng = np.random.default_rng(seed)
        out = [('adopted', self.member_config(self.cable_fit['best']['x'], self.drone_best()))]
        cov = self.covariance(self.cable_fit)
        dcov = self.covariance(self.drone_fit) if self.drone_fit else None
        for k in range(count-1):
            x = rng.multivariate_normal(np.asarray(self.cable_fit['best']['x'], float), sigma**2*cov)
            d = self.drone_best()
            if dcov is not None:
                d = rng.multivariate_normal(d, sigma**2*dcov)
                d[1] = max(d[1], .003); d[2] = max(d[2], 0.); d[3] = max(d[3], 0.)
            d[0] = rng.uniform(*inertia_range)
            out.append((f'sample {k+1}', self.member_config(x, d)))
        return out

    def brackets(self):
        """The bracket members around the adopted model."""
        return [(name, self.drone_config(self.cable_config(self.cable_fit['best']['x'], scale), self.drone_best()))
                for name, scale in BRACKETS.items()]

    def member_config(self, x, d):
        return self.drone_config(self.cable_config(x), d)

    def drone_best(self):
        if self.drone_fit is None:
            return None
        return np.asarray(self.drone_fit['best']['x'], float).copy()

    @staticmethod
    def covariance(fit):
        u = fit['uncertainty']
        sd = np.asarray(u['standard_deviation'], float)
        corr = np.asarray(u.get('correlation', np.eye(len(sd))), float)
        return sd[:, None]*corr*sd[None, :]
