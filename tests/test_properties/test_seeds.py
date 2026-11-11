"""Item 2: seeds (C1, A5).

``impute(seed=...)``: the same seed gives identical output, a different seed
gives different output, ``seed=None`` draws fresh entropy. Chain k gets the
same random stream whatever ``n_imputations`` is. Every imputer, midas
included, draws only from the ``rng`` it is given.

The categorical methods run on a frame with boolean, categorical and text
columns (where the columns they cannot impute get pmm), the others on numeric
data.

``mean`` is the one method whose output does not depend on the seed at all,
so the tests that expect randomness skip it and its own test expects the
opposite. ``norm.predict`` draws no random number either, but in a chain its
predictors hold the random initial imputations of the other incomplete
columns, so its datasets still depend on the seed (on these data, with three
incomplete columns); with a single incomplete column they do not.
"""

import numpy as np
import pytest

from ..test_parity._helpers import ScriptedGenerator
from ._helpers import (
    CATEGORICAL_METHODS,
    DETERMINISTIC_METHODS,
    METHODS,
    impute,
    numeric_frame,
    regression_frame,
    typed_frame,
)

# Methods whose completed datasets do not depend on the seed
SEED_INDEPENDENT = ("mean",)
STOCHASTIC = [m for m in METHODS if m not in SEED_INDEPENDENT]


def _same(a, b):
    return len(a) == len(b) and all(x.equals(y) for x, y in zip(a, b))


def _run(method, seed, n_imputations=2, **kwargs):
    df = typed_frame(n=40, seed=11) if method in CATEGORICAL_METHODS else numeric_frame(n=40, seed=11)
    _, out = impute(df, method, n_imputations=n_imputations, maxit=2, seed=seed, **kwargs)
    return out


@pytest.mark.parametrize("method", METHODS)
def test_same_seed_gives_identical_output(method):
    assert _same(_run(method, seed=7), _run(method, seed=7))


@pytest.mark.parametrize("method", STOCHASTIC)
def test_different_seeds_give_different_output(method):
    assert not _same(_run(method, seed=7), _run(method, seed=8))


@pytest.mark.parametrize("method", STOCHASTIC)
def test_seed_none_draws_fresh_entropy(method):
    assert not _same(_run(method, seed=None), _run(method, seed=None))


@pytest.mark.parametrize("method", STOCHASTIC)
def test_chains_are_not_copies_of_each_other(method):
    out = _run(method, seed=7)
    assert not out[0].equals(out[1])


@pytest.mark.parametrize("method", SEED_INDEPENDENT)
def test_seed_independent_method_ignores_the_seed(method):
    """Different seeds, ``seed=None`` and different chains: all identical."""
    out = _run(method, seed=7)
    assert _same(out, _run(method, seed=8))
    assert _same(out, _run(method, seed=None))
    assert out[0].equals(out[1])


def test_norm_predict_ignores_the_seed_with_one_incomplete_column():
    """Its predictors are then complete, so nothing random enters."""
    df = regression_frame()
    _, first = impute(df, "norm.predict", seed=7)
    _, second = impute(df, "norm.predict", seed=8)
    assert _same(first, second)
    assert first[0].equals(first[1])


@pytest.mark.parametrize("method", DETERMINISTIC_METHODS)
def test_deterministic_imputer_draws_no_random_number(method):
    """A strict generator with nothing scripted refuses every draw."""
    from imputation.utils import get_imputer_func

    rng = np.random.default_rng(5)
    x = rng.normal(size=(40, 2))
    y = x.sum(axis=1) + rng.normal(size=40)
    ry = np.ones(40, dtype=bool)
    ry[::5] = False
    scripted = ScriptedGenerator(strict=True)
    out = get_imputer_func(method)(y, ry, x, rng=scripted)
    assert out.shape == (8,)
    assert scripted.calls == []


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
    if method == "polyreg":  # a three-category target
        y = np.where(ry, np.digitize(y, [-0.5, 0.5]), None)
    elif method in CATEGORICAL_METHODS:  # a two-category target
        y = np.where(ry, y > 0, None)
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
    # the legacy global generator is what the test watches
    np.random.seed(2024)  # noqa: NPY002
    before = np.random.get_state()  # noqa: NPY002
    midas(y, ry, x, rng=np.random.default_rng(1))
    after = np.random.get_state()  # noqa: NPY002
    np.testing.assert_array_equal(after[1], before[1])
    assert after[2:] == before[2:]
