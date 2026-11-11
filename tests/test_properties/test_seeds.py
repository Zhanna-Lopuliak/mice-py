"""Item 2: seeds (C1, A5).

Planned: ``impute(seed=...)``; the same seed gives identical output, a different
seed gives different output, ``seed=None`` draws fresh entropy. Today ``seed``
is swallowed by ``**kwargs`` and every chain uses ``default_rng(42 + chain)``,
and midas draws from the global NumPy RNG.
"""

import numpy as np
import pytest

from ._helpers import METHODS, impute, method_params, numeric_frame


def _same(a, b):
    return len(a) == len(b) and all(x.equals(y) for x, y in zip(a, b))


def _run(method, seed):
    df = numeric_frame(n=40, seed=11)
    _, out = impute(df, method, n_imputations=2, maxit=2, seed=seed)
    return out


@pytest.mark.parametrize(
    "method", method_params({"midas": "A5: midas draws from the global NumPy RNG and ignores rng"})
)
def test_same_seed_gives_identical_output(method):
    assert _same(_run(method, seed=7), _run(method, seed=7))


C1_FIXED_SEED = "C1: seed is ignored, chains always use default_rng(42 + chain)"


@pytest.mark.parametrize(
    "method",
    # midas passes today only because it uses the global RNG (A5).
    method_params({m: C1_FIXED_SEED for m in ("pmm", "cart", "rf", "sample")}),
)
def test_different_seeds_give_different_output(method):
    assert not _same(_run(method, seed=7), _run(method, seed=8))


@pytest.mark.parametrize(
    "method",
    method_params({m: C1_FIXED_SEED for m in ("pmm", "cart", "rf", "sample")}),
)
def test_seed_none_draws_fresh_entropy(method):
    assert not _same(_run(method, seed=None), _run(method, seed=None))


@pytest.mark.parametrize("method", METHODS)
def test_chains_are_not_copies_of_each_other(method):
    out = _run(method, seed=7)
    assert not out[0].equals(out[1])


@pytest.mark.parametrize(
    "method", method_params({"midas": "A5: midas draws from the global NumPy RNG and ignores rng"})
)
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
