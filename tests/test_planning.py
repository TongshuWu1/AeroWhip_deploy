"""The planning pieces that run without a flight: free trajectories and information scores."""
import numpy as np

from aerowhip.planning.information import scores
from aerowhip.planning.trajectory import FourierTrajectory


def test_fourier_trajectory_rests_at_both_ends_and_normalises_to_the_level():
    T = FourierTrajectory(terms=3, duration=20.)
    theta = T.normalise(T.random(np.random.default_rng(2)), level=3.)
    t, p, v, a = T.samples(theta)
    assert np.linalg.norm(p[0]) < 1e-9 and np.linalg.norm(p[-1]) < 1e-6 and np.linalg.norm(v[0]) < 1e-9 and np.linalg.norm(v[-1]) < 1e-6
    assert abs(np.linalg.norm(a[:, :2], axis=1).max()-3.) < 1e-6
    assert np.abs(a[:, 2]).max() <= 3.+1e-9 and np.linalg.norm(v, axis=1).max() <= 3.+1e-9
    m = T.motion(theta)
    assert abs(m.duration-20.) < 1e-9 and np.allclose(m.state(10.)[0], p[np.argmin(np.abs(t-10.))], atol=1e-6)


def test_information_score_is_the_weakest_parameter_and_grows_with_information():
    parameters = [('a', 'cable', 0, 1.), ('b', 'cable', 1, 1.)]
    weak = scores(np.diag([100., 1.]), parameters)
    assert weak['weakest'] == 'b' and abs(weak['score']-0.) < 1e-9
    assert scores(np.diag([100., 10.]), parameters)['score'] > weak['score']
    coupled = scores(np.array([[100., 30.], [30., 10.]]), parameters)   # correlation makes b harder to tell apart
    assert coupled['marginal_log10']['b'] < scores(np.diag([100., 10.]), parameters)['marginal_log10']['b']
