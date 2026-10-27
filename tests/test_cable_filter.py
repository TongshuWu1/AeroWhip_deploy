"""The live cable filter's building blocks: the length projection (scalar rewrite against the
vector reference, bit for bit), candidate preprocessing, and labelling of a hanging cable."""
import numpy as np

from aerowhip.deployment.cable_filter import MARKERS, CableFilter, cable_candidates

ARC = np.cumsum([.095]+[.1]*9)


def reference_project(filt, root, positions):
    chain = np.vstack([root, positions])
    for _ in range(filt.iterations):
        for k in range(1, MARKERS+1):
            gap = chain[k]-chain[k-1]
            length = np.linalg.norm(gap)
            limit = filt.segments[k-1]
            if length > limit:
                chain[k] = chain[k-1]+gap*(limit/length)
    return chain[1:]


def test_projection_matches_the_vector_reference_exactly():
    filt = CableFilter(ARC)
    rng = np.random.default_rng(0)
    for _ in range(200):
        root = rng.normal(size=3)
        positions = root+np.cumsum(rng.normal(scale=.12, size=(MARKERS, 3)), 0)
        expected = reference_project(filt, root, positions.copy())
        got = positions.copy()
        filt._project(root, got)
        assert np.array_equal(got, expected)
        steps = np.linalg.norm(np.diff(np.vstack([root, got]), axis=0), axis=1)
        assert (steps <= filt.segments+1e-12).all()


def test_candidates_drop_drone_markers_and_duplicates():
    drone = np.array([0., 0., 2.])
    cloud = np.array([[0., .05, 2.], [.3, 0., 1.5], [.3, 0., 1.505], [np.nan, 0., 0.], [1., 1., 1.]])
    out = cable_candidates(cloud, drone)
    assert out.shape == (2, 3) and np.array_equal(out[0], [.3, 0., 1.5]) and np.array_equal(out[1], [1., 1., 1.])


def test_hanging_cable_is_labelled_in_order():
    filt = CableFilter(ARC)
    root = np.array([0., 0., 2.])
    markers = root+np.outer(ARC, [0., 0., -1.])
    rng = np.random.default_rng(1)
    order = rng.permutation(MARKERS)
    state = filt.observe(0., root, markers[order])
    assert state['initialized'] and (state['confidence'] == 1.).all()
    assert np.array_equal(state['index'], np.argsort(order))


def test_frame_clock_absorbs_receipt_jitter():
    # A marker moving at 2 m/s observed on jittered receipt stamps (frames exactly 10 ms apart):
    # the filter's velocity follows the motion, not the jitter.
    rng = np.random.default_rng(2)
    filt = CableFilter(ARC)
    root0 = np.array([0., 0., 2.])
    velocity = np.array([2., 0., 0.])
    for frame in range(60):
        stamp = frame*.01+rng.uniform(-.004, .004)
        root = root0+velocity*frame*.01
        state = filt.observe(stamp, root, root+np.outer(ARC, [0., 0., -1.]))
    np.testing.assert_allclose(state['velocities'], np.tile(velocity, (10, 1)), atol=1e-6)


def test_frame_clock_counts_a_dropped_frame():
    filt = CableFilter(ARC)
    root0 = np.array([0., 0., 2.])
    velocity = np.array([1., 0., 0.])
    for frame in [*range(30), *range(31, 60)]:              # frame 30 never arrives
        root = root0+velocity*frame*.01
        state = filt.observe(frame*.01, root, root+np.outer(ARC, [0., 0., -1.]))
    np.testing.assert_allclose(state['velocities'], np.tile(velocity, (10, 1)), atol=1e-6)
