"""Item 3: argument handling (C2, C4).

Planned: the imputers have no ``**kwargs``, so an unknown argument raises
``TypeError``; ``impute`` validates ``<method>_<param>`` names against the
imputer signatures and raises on unknown names; NumPy integers are accepted
for ``n_imputations`` and ``maxit``.
"""

import numpy as np
import pytest

from ._helpers import impute, method_params, numeric_frame, pmm_reference_frame

C2_SWALLOWED = "C2: unknown keyword arguments are swallowed by **kwargs"


def _problem():
    rng = np.random.default_rng(0)
    n = 30
    x = rng.normal(size=(n, 2))
    y = x.sum(axis=1) + rng.normal(size=n)
    ry = np.ones(n, dtype=bool)
    ry[::6] = False
    y[~ry] = np.nan
    return y, ry, x


@pytest.mark.parametrize("method", method_params({"midas": C2_SWALLOWED, "sample": C2_SWALLOWED}))
def test_imputer_rejects_unknown_keyword(method):
    from imputation.utils import get_imputer_func

    y, ry, x = _problem()
    with pytest.raises(TypeError):
        get_imputer_func(method)(y, ry, x, not_a_parameter=1, rng=np.random.default_rng(0))


@pytest.mark.parametrize("method", method_params({"midas": C2_SWALLOWED, "sample": C2_SWALLOWED}))
def test_impute_rejects_misspelled_method_parameter(method):
    df = numeric_frame(n=30)
    with pytest.raises((TypeError, ValueError)):
        impute(df, method, n_imputations=1, maxit=1, **{f"{method}_not_a_parameter": 3})


@pytest.mark.xfail(strict=True, reason=C2_SWALLOWED + " (unknown method prefix)")
def test_impute_rejects_unknown_method_prefix():
    df = numeric_frame(n=30)
    with pytest.raises((TypeError, ValueError)):
        impute(df, "pmm", n_imputations=1, maxit=1, nosuchmethod_donors=3)


@pytest.mark.xfail(strict=True, reason=C2_SWALLOWED + " (plain unknown keyword)")
def test_impute_rejects_unknown_plain_keyword():
    df = numeric_frame(n=30)
    with pytest.raises((TypeError, ValueError)):
        impute(df, "pmm", n_imputations=1, maxit=1, n_imputation=3)


def _nearest_fitted_donor(df, mis):
    """PMM with one donor and matchtype 0: the observed y whose OLS fit is closest."""
    X = np.c_[np.ones(len(df)), df[["x1", "x2"]].to_numpy()]
    y = df["y"].to_numpy()
    coef, *_ = np.linalg.lstsq(X[~mis], y[~mis], rcond=None)
    yhat = X @ coef
    dist = np.abs(yhat[mis][:, None] - yhat[~mis][None, :])
    gaps = np.sort(dist, axis=1)
    assert (gaps[:, 1] - gaps[:, 0] > 1e-3).all(), "test data must not have near ties"
    return y[~mis][dist.argmin(axis=1)]


def test_prefixed_parameters_reach_the_imputer():
    """``pmm_donors=1`` with ``pmm_matchtype=0`` makes PMM deterministic: every
    imputation is the observed value whose fitted value is closest."""
    df, mis = pmm_reference_frame()
    expected = _nearest_fitted_donor(df, mis)
    _, out = impute(df, "pmm", n_imputations=3, maxit=1, pmm_donors=1, pmm_matchtype=0)
    for completed in out:
        np.testing.assert_allclose(completed.loc[mis, "y"].to_numpy(), expected, rtol=0, atol=0)


def test_default_donors_are_not_deterministic():
    """Control for the test above: with the default 5 donors the result varies."""
    df, mis = pmm_reference_frame()
    expected = _nearest_fitted_donor(df, mis)
    _, out = impute(df, "pmm", n_imputations=3, maxit=1)
    assert not all(np.array_equal(c.loc[mis, "y"].to_numpy(), expected) for c in out)


C4_NUMPY_INT = "C4: validators require a Python int and reject NumPy integers"


@pytest.mark.xfail(strict=True, reason=C4_NUMPY_INT)
@pytest.mark.parametrize("dtype", [np.int64, np.int32], ids=["int64", "int32"])
def test_numpy_integer_n_imputations(dtype):
    df = numeric_frame(n=30)
    mice, out = impute(df, "sample", n_imputations=dtype(3), maxit=2)
    assert len(out) == 3
    assert all(v.shape == (2, 3) for v in mice.chain_mean.values())


@pytest.mark.xfail(strict=True, reason=C4_NUMPY_INT)
@pytest.mark.parametrize("dtype", [np.int64, np.int32], ids=["int64", "int32"])
def test_numpy_integer_maxit(dtype):
    df = numeric_frame(n=30)
    mice, out = impute(df, "sample", n_imputations=2, maxit=dtype(3))
    assert len(out) == 2
    assert all(v.shape == (3, 2) for v in mice.chain_mean.values())
