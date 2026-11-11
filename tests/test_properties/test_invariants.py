"""Item 1: invariants that every completed dataset must satisfy.

Each (dataset, method) pair is imputed once and the result is cached, so the
individual invariants can be reported separately without re-running MICE.
"""

import functools

import numpy as np
import pandas as pd
import pytest

from ._helpers import (
    MAXIT,
    METHODS,
    N_IMP,
    assert_chain_stats,
    assert_dtypes_preserved,
    assert_imputed_are_observed,
    assert_no_missing,
    assert_observed_unchanged,
    assert_structure_preserved,
    impute,
    mixed_frame,
    numeric_frame,
    xfail,
)

DATASETS = {
    "numeric": lambda: numeric_frame(index="noncontiguous"),
    "numeric_strindex": lambda: numeric_frame(seed=4, index="string"),
    "mixed": lambda: mixed_frame(index="noncontiguous"),
}

KNOWN = {}

CASES = [
    pytest.param(ds, m, marks=xfail(KNOWN[(ds, m)]), id=f"{ds}-{m}")
    if (ds, m) in KNOWN
    else pytest.param(ds, m, id=f"{ds}-{m}")
    for ds in DATASETS
    for m in METHODS
]


@functools.lru_cache(maxsize=None)
def _run(dataset, method):
    df = DATASETS[dataset]()
    before = df.copy(deep=True)
    mice, out = impute(df, method, n_imputations=N_IMP, maxit=MAXIT)
    return df, before, mice, out


@pytest.mark.parametrize("dataset,method", CASES)
def test_number_of_datasets(dataset, method):
    _, _, mice, out = _run(dataset, method)
    assert len(out) == N_IMP
    assert len(mice.imputed_datasets) == N_IMP
    assert all(isinstance(d, pd.DataFrame) for d in out)


@pytest.mark.parametrize("dataset,method", CASES)
def test_observed_cells_unchanged(dataset, method):
    df, _, _, out = _run(dataset, method)
    for completed in out:
        assert_observed_unchanged(df, completed)


@pytest.mark.parametrize("dataset,method", CASES)
def test_no_missing_values_left(dataset, method):
    _, _, _, out = _run(dataset, method)
    for completed in out:
        assert_no_missing(completed)


@pytest.mark.parametrize("dataset,method", CASES)
def test_shape_columns_and_index_preserved(dataset, method):
    df, _, _, out = _run(dataset, method)
    for completed in out:
        assert_structure_preserved(df, completed)


@pytest.mark.parametrize("dataset,method", CASES)
def test_dtypes_preserved(dataset, method):
    df, _, _, out = _run(dataset, method)
    for completed in out:
        assert_dtypes_preserved(df, completed)


@pytest.mark.parametrize("dataset,method", CASES)
def test_input_not_mutated(dataset, method):
    df, before, _, _ = _run(dataset, method)
    pd.testing.assert_frame_equal(df, before)


@pytest.mark.parametrize("dataset,method", CASES)
def test_imputed_values_are_observed_values(dataset, method):
    """All five methods are donor based: imputations are observed values or labels."""
    df, _, _, out = _run(dataset, method)
    for completed in out:
        assert_imputed_are_observed(df, completed)


@pytest.mark.parametrize("dataset,method", CASES)
def test_chain_statistics_shape_and_finite(dataset, method):
    df, _, mice, _ = _run(dataset, method)
    assert_chain_stats(mice, df, N_IMP, MAXIT)


# ---------------------------------------------------------------------------
# The univariate imputers, called directly
# ---------------------------------------------------------------------------


def _univariate_problem(seed=7, n=40, x_as_frame=False):
    rng = np.random.default_rng(seed)
    x = rng.normal(size=(n, 2))
    y = x @ np.array([1.0, -0.5]) + rng.normal(scale=0.5, size=n)
    ry = np.ones(n, dtype=bool)
    ry[[0, 3, 8, 15, 22, 31]] = False
    y[~ry] = np.nan
    if x_as_frame:
        x = pd.DataFrame(x, columns=["x1", "x2"])
    return y, ry, x


def _imputer(method):
    from imputation.utils import get_imputer_func

    return get_imputer_func(method)


@pytest.mark.parametrize("x_as_frame", [False, True], ids=["ndarray", "DataFrame"])
@pytest.mark.parametrize("method", METHODS)
def test_univariate_returns_one_observed_value_per_missing_cell(method, x_as_frame):
    y, ry, x = _univariate_problem(x_as_frame=x_as_frame)
    y_before, ry_before = y.copy(), ry.copy()
    x_before = x.copy()
    out = np.asarray(_imputer(method)(y, ry, x, rng=np.random.default_rng(1)))
    assert out.shape == ((~ry).sum(),)
    assert np.isin(out, y[ry]).all()
    # inputs are not mutated
    np.testing.assert_array_equal(y, y_before)
    np.testing.assert_array_equal(ry, ry_before)
    if x_as_frame:
        pd.testing.assert_frame_equal(x, x_before)
    else:
        np.testing.assert_array_equal(x, x_before)


@pytest.mark.parametrize("method", METHODS)
def test_univariate_imputes_only_the_requested_cells(method):
    """``id_mis`` may select a subset of the missing cells (R's ``wy``)."""
    y, ry, x = _univariate_problem()
    id_mis = np.zeros_like(ry)
    id_mis[[3, 22]] = True
    out = np.asarray(_imputer(method)(y, ry, x, id_mis=id_mis, rng=np.random.default_rng(2)))
    assert out.shape == (2,)
    assert np.isin(out, y[ry]).all()
