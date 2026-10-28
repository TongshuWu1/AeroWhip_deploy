"""The online estimator's update and robust filter on a linear stand-in for the replays: windows
that agree move the estimate, a one-off wild window (a flip) is held and does not, and a change
that keeps happening is adopted."""
from concurrent.futures import Future

import numpy as np

from aerowhip.online.estimator import OnlineEstimator, Parameter


class LinearSpace:
    """Two parameters; residual r = A (theta - truth) + noise, in place of a replay."""
    def __init__(self):
        self.parameters = [Parameter('a', 'cable', 0., 1., .5, -np.inf, np.inf), Parameter('b', 'cable', 0., 1., .5, -np.inf, np.inf)]
        self.prior = np.eye(2)
        self.typical = dict(cable=.1)
        self.names = ['a', 'b']

    def family(self, name):
        return [0, 1]


def window(estimator, A, truth, noise, offset=0.):
    def submit(theta):
        f = Future()
        f.set_result(A@(np.asarray(theta)[[0, 1]]-truth)+noise+offset)
        return f
    thetas = [estimator.theta]+[estimator.theta+np.eye(2)[k]*.5 for k in range(2)]
    return estimator._update('cable', [submit(t) for t in thetas], submit)


def setup(seed=0):
    rng = np.random.default_rng(seed)
    est = OnlineEstimator(LinearSpace(), families=('cable',), threads=1, effective_samples=8.)
    A = rng.normal(size=(200, 2))*.1
    return est, A, rng


def test_agreeing_windows_converge():
    est, A, rng = setup()
    truth = np.array([.8, -.5])
    statuses = [window(est, A, truth, rng.normal(size=200)*.05)['status'] for _ in range(30)]
    assert statuses.count('accepted') >= 28
    assert np.allclose(est.theta, truth, atol=.15)


def test_one_wild_window_is_held():
    est, A, rng = setup(1)
    truth = np.array([.3, .2])
    for _ in range(10):
        window(est, A, truth, rng.normal(size=200)*.05)
    before = est.theta.copy()
    out = window(est, A, truth+np.array([4., -4.]), rng.normal(size=200)*.05, offset=1.)   # a flip: unexplained and far
    assert out['status'] == 'outlier'
    assert np.allclose(est.theta, before)
    for _ in range(8):                                     # it does not recur: it expires, nothing changes
        window(est, A, truth, rng.normal(size=200)*.05)
    assert not est.families['cable'].pending
    assert np.allclose(est.theta, truth, atol=.2)


def test_a_change_that_keeps_happening_is_adopted():
    est, A, rng = setup(2)
    truth = np.array([0., 0.])
    for _ in range(15):
        window(est, A, truth, rng.normal(size=200)*.05)
    new = np.array([3., 2.])                               # the drone changed (e.g. a different battery)
    statuses = [window(est, A, new, rng.normal(size=200)*.05)['status'] for _ in range(12)]
    assert 'change' in statuses
    assert np.allclose(est.theta, new, atol=.5)
