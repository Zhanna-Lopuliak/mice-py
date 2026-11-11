"""Item 2: seeds (C1, A5).

``impute(seed=...)``: the same seed gives identical output, a different seed
gives different output, ``seed=None`` draws fresh entropy. Chain k gets the
same random stream whatever ``n_imputations`` is. Every imputer, midas
included, draws only from the ``rng`` it is given.
"""

import numpy as np
import pytest

from ._helpers import METHODS, impute, numeric_frame


def _same(a, b):
    return len(a) == len(b) and all(x.equals(y) for x, y in zip(a, b))


def _run(method, seed, n_imputations=2, **kwargs):
    df = numeric_frame(n=40, seed=11)
    _, out = impute(df, method, n_imputations=n_imputations, maxit=2, seed=seed, **kwargs)
    return out


@pytest.mark.parametrize("method", METHODS)
def test_same_seed_gives_identical_output(method):
    assert _same(_run(method, seed=7), _run(method, seed=7))


@pytest.mark.parametrize("method", METHODS)
def test_different_seeds_give_different_output(method):
    assert not _same(_run(method, seed=7), _run(method, seed=8))


@pytest.mark.parametrize("method", METHODS)
def test_seed_none_draws_fresh_entropy(method):
    assert not _same(_run(method, seed=None), _run(method, seed=None))


@pytest.mark.parametrize("method", METHODS)
def test_chains_are_not_copies_of_each_other(method):
    out = _run(method, seed=7)
    assert not out[0].equals(out[1])


@pytest.mark.parametrize("method", METHODS)
def test_chain_stream_does_not_depend_on_n_imputations(method):
    """The first k datasets of ``impute(n_imputations=m, seed=s)`` are the same
    for every m >= k. The random visit sequence draws from its own stream."""
    three = _run(method, seed=7, n_imputations=3, visit_sequence="random")
    for m in (1, 2):
        assert _same(_run(method, seed=7, n_imputations=m, visit_sequence="random"), three[:m]), m


def test_random_visit_sequence_follows_seed():
    df = numeric_frame(n=40, seed=11)

    def order(seed):
        mice, _ = impute(df, "sample", n_imputations=1, maxit=1, seed=seed, visit_sequence="random")
        return tuple(mice.visit_sequence)

    assert order(3) == order(3)
    # three incomplete columns, six orders: twelve seeds must not all agree
    assert len({order(seed) for seed in range(12)}) > 1


@pytest.mark.parametrize("method", METHODS)
def test_univariate_imputer_is_reproducible_with_rng(method):
    from imputation.utils import get_imputer_func

    rng = np.random.default_rng(5)
    n = 40
    x = rng.normal(size=(n, 2))
    y = x.sum(axis=1) + rng.normal(size=n)
    ry = np.ones(n, dtype=bool)
    ry[::5] = False
    y[~ry] = np.nan
    f = get_imputer_func(method)
    first = f(y, ry, x, rng=np.random.default_rng(99))
    second = f(y, ry, x, rng=np.random.default_rng(99))
    np.testing.assert_array_equal(first, second)


def test_midas_does_not_touch_numpy_global_rng():
    from imputation.midas import midas

    rng = np.random.default_rng(5)
    n = 40
    x = rng.normal(size=(n, 2))
    y = x.sum(axis=1) + rng.normal(size=n)
    ry = np.ones(n, dtype=bool)
    ry[::5] = False
    np.random.seed(2024)
    before = np.random.get_state()
    midas(y, ry, x, rng=np.random.default_rng(1))
    after = np.random.get_state()
    np.testing.assert_array_equal(after[1], before[1])
    assert after[2:] == before[2:]
