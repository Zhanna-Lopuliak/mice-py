"""Item 1: invariants that every completed dataset must satisfy.

Each (dataset, method) pair is imputed once and the result is cached, so the
individual invariants can be reported separately without re-running MICE.
"""

import functools
import re

import numpy as np
import pandas as pd
import pytest

from ._helpers import (
    CATEGORICAL_METHODS,
    DONOR_METHODS,
    MAXIT,
    METHODS,
    N_IMP,
    NUMERIC_MODEL_METHODS,
    assert_chain_stats,
    assert_dtypes_preserved,
    assert_imputed_are_observed,
    assert_model_imputations,
    assert_no_missing,
    assert_observed_unchanged,
    assert_structure_preserved,
    impute,
    imputes_observed_values,
    mixed_frame,
    numeric_frame,
    typed_frame,
    xfail,
)

DATASETS = {
    "numeric": lambda: numeric_frame(index="noncontiguous"),
    "numeric_strindex": lambda: numeric_frame(seed=4, index="string"),
    "mixed": lambda: mixed_frame(index="noncontiguous"),
    "typed": lambda: typed_frame(index="noncontiguous"),
}

KNOWN = {}

CASES = [
    pytest.param(ds, m, marks=xfail(KNOWN[(ds, m)]), id=f"{ds}-{m}")
    if (ds, m) in KNOWN
    else pytest.param(ds, m, id=f"{ds}-{m}")
    for ds in DATASETS
    for m in METHODS
]
# Donor methods, and categorical methods with pmm for the other columns
OBSERVED_VALUE_CASES = [case for case in CASES if imputes_observed_values(case.values[1])]
MODEL_CASES = [case for case in CASES if case.values[1] in NUMERIC_MODEL_METHODS]


@functools.cache
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
    """Kept, except a nullable integer column under a numeric-model method,
    which becomes Float64 (none in these datasets)."""
    df, _, mice, out = _run(dataset, method)
    for completed in out:
        assert_dtypes_preserved(df, completed, mice)


@pytest.mark.parametrize("dataset,method", CASES)
def test_input_not_mutated(dataset, method):
    df, before, _, _ = _run(dataset, method)
    pd.testing.assert_frame_equal(df, before)


@pytest.mark.parametrize("dataset,method", OBSERVED_VALUE_CASES)
def test_imputed_values_are_observed_values(dataset, method):
    """Donor methods impute observed values or labels; so do the categorical
    methods (labels of their columns) with pmm for the other columns, except
    that polr may give an ordered Categorical any declared category."""
    df, _, mice, out = _run(dataset, method)
    for completed in out:
        assert_imputed_are_observed(df, completed, mice=mice)


# The columns of the typed frame each categorical method imputes
TYPED_COLUMNS = {
    "logreg": ["flag", "two"],
    "logreg.boot": ["flag", "two"],
    "polyreg": ["flag", "two", "three", "grade"],
    "polr": ["flag", "two", "three", "grade"],
}


@pytest.mark.parametrize("method", CATEGORICAL_METHODS)
def test_categorical_method_imputes_its_columns(method):
    """On the typed frame logreg and logreg.boot impute the boolean and the
    two-level column, polyreg and polr also the three-category text column
    and the ordered column (polr the boolean and the two-level column
    through its fallback), with labels observed there: every declared level
    of the ordered column is observed."""
    df, _, mice, out = _run("typed", method)
    assert [c for c in df.columns if mice.method[c] == method] == TYPED_COLUMNS[method]
    assert set(df["grade"].dropna()) == set(df["grade"].cat.categories)
    for completed in out:
        assert_imputed_are_observed(df, completed, columns=TYPED_COLUMNS[method])


@pytest.mark.parametrize("dataset,method", MODEL_CASES)
def test_model_imputations_are_finite_floats(dataset, method):
    """Numeric-model methods impute finite floats; mean imputes the observed mean."""
    df, _, mice, out = _run(dataset, method)
    assert {mice.method[c] for c in df.columns if df[c].isna().any() and df[c].dtype == float} == {method}
    for completed in out:
        assert_model_imputations(df, completed, mice)


@pytest.mark.parametrize("dataset,method", CASES)
def test_chain_statistics_shape_and_finite(dataset, method):
    df, _, mice, _ = _run(dataset, method)
    assert_chain_stats(mice, df, N_IMP, MAXIT)


# ---------------------------------------------------------------------------
# The univariate imputers, called directly
# ---------------------------------------------------------------------------


def _univariate_problem(seed=7, n=40, x_as_frame=False, method=None):
    """(y, ry, x); y is numeric, or for a categorical method an object array of
    labels (three for polyreg and polr, two for the others) with None where
    it is missing, as MICE passes a text column."""
    rng = np.random.default_rng(seed)
    x = rng.normal(size=(n, 2))
    y = x @ np.array([1.0, -0.5]) + rng.normal(scale=0.5, size=n)
    ry = np.ones(n, dtype=bool)
    ry[[0, 3, 8, 15, 22, 31]] = False
    y[~ry] = np.nan
    if method in ("polyreg", "polr"):
        y = np.where(ry, np.where(y > 0.5, "hi", np.where(y > -0.5, "mid", "lo")), None)
    elif method in CATEGORICAL_METHODS:
        y = np.where(ry, np.where(y > 0, "yes", "no"), None)
    if x_as_frame:
        x = pd.DataFrame(x, columns=["x1", "x2"])
    return y, ry, x


def _imputer(method):
    from imputation.utils import get_imputer_func

    return get_imputer_func(method)


def _assert_imputations_of_kind(method, out, y, ry):
    """Donor methods: observed values. Categorical methods: observed labels.
    Numeric-model methods: finite floats, the observed mean for mean."""
    if method in DONOR_METHODS:
        assert np.isin(out, y[ry]).all()
    if method in CATEGORICAL_METHODS:
        assert out.dtype == object and set(out) <= set(y[ry])
    if method in NUMERIC_MODEL_METHODS:
        assert out.dtype == np.float64 and np.isfinite(out).all()
    if method == "mean":
        np.testing.assert_allclose(out, y[ry].mean(), rtol=1e-14)


@pytest.mark.parametrize("x_as_frame", [False, True], ids=["ndarray", "DataFrame"])
@pytest.mark.parametrize("method", METHODS)
def test_univariate_returns_one_value_per_missing_cell(method, x_as_frame):
    """One imputation per missing cell, of the method's kind; inputs untouched."""
    y, ry, x = _univariate_problem(x_as_frame=x_as_frame, method=method)
    y_before, ry_before = y.copy(), ry.copy()
    x_before = x.copy()
    out = np.asarray(_imputer(method)(y, ry, x, rng=np.random.default_rng(1)))
    assert out.shape == ((~ry).sum(),)
    _assert_imputations_of_kind(method, out, y, ry)
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
    y, ry, x = _univariate_problem(method=method)
    id_mis = np.zeros_like(ry)
    id_mis[[3, 22]] = True
    out = np.asarray(_imputer(method)(y, ry, x, id_mis=id_mis, rng=np.random.default_rng(2)))
    assert out.shape == (2,)
    _assert_imputations_of_kind(method, out, y, ry)


# Methods whose imputer reads no row outside its two masks: all but those
# with R's augmentation (logreg, polyreg, polr)
NOT_AUGMENTED = (*DONOR_METHODS, *NUMERIC_MODEL_METHODS, "logreg.boot")
# Methods that read the predictors (mean and sample, like R's, ignore x)
READING_X = tuple(m for m in METHODS if m not in ("mean", "sample"))


def _with_rows_in_neither_mask(y, ry, x):
    """The problem (y, ry, x) with five rows in neither mask inserted among its rows; also returns id_mis.

    The added rows have NaN in one predictor or in both, predictors far
    outside their range elsewhere, and a decoy target: an extreme number,
    or for labels a label of its own, or a missing value.
    """
    positions = [0, 7, 7, 19, len(ry)]
    x_added = np.array([[np.nan, 50.0], [np.nan, np.nan], [-40.0, np.nan], [60.0, -70.0], [np.nan, np.nan]])
    labels = y.dtype == object
    y_added = ["decoy", None, "decoy", "zzz", None] if labels else [1e6, np.nan, -1e6, 2e6, np.nan]
    y = np.insert(y, positions, y_added)
    x = np.insert(x, positions, x_added, axis=0)
    return y, np.insert(ry, positions, False), x, np.insert(~ry, positions, False)


@pytest.mark.parametrize("x_as_frame", [False, True], ids=["ndarray", "DataFrame"])
@pytest.mark.parametrize("method", NOT_AUGMENTED)
def test_univariate_ignores_rows_in_neither_mask(method, x_as_frame):
    """Rows outside both masks (NaN or extreme predictors, decoy targets)
    change nothing for a method without augmentation: the same seed gives
    the same imputations."""
    y, ry, x = _univariate_problem(method=method)
    plain = _imputer(method)(y, ry, x, id_mis=~ry, rng=np.random.default_rng(3))
    y2, ry2, x2, id_mis2 = _with_rows_in_neither_mask(y, ry, x)
    if x_as_frame:
        x2 = pd.DataFrame(x2, columns=["x1", "x2"])
    out = _imputer(method)(y2, ry2, x2, id_mis=id_mis2, rng=np.random.default_rng(3))
    np.testing.assert_array_equal(np.asarray(out), np.asarray(plain))


@pytest.mark.parametrize("method", METHODS)
def test_univariate_imputes_every_row_of_overlapping_masks(method):
    """``id_mis`` may include fit rows (R's ``where = "all"``): one imputation
    per row of ``id_mis``, of the method's kind."""
    y, ry, x = _univariate_problem(method=method)
    id_mis = np.ones_like(ry)
    out = np.asarray(_imputer(method)(y, ry, x, id_mis=id_mis, rng=np.random.default_rng(4)))
    assert out.shape == (len(ry),)
    _assert_imputations_of_kind(method, out, y, ry)


@pytest.mark.parametrize("x_as_frame", [False, True], ids=["ndarray", "DataFrame"])
@pytest.mark.parametrize("row", ["fit", "imputed"])
@pytest.mark.parametrize("method", READING_X)
def test_univariate_rejects_a_missing_predictor_in_a_row_it_uses(method, row, x_as_frame):
    """NaN in x in a fit row or in a row to impute raises ``ValueError``
    naming the method (cart and rf too, whose scikit-learn trees would take
    it); NaN in a row in neither mask does not."""
    y, ry, x = _univariate_problem(method=method)
    target_row = np.flatnonzero(ry if row == "fit" else ~ry)[0]
    x = x.copy()
    x[target_row, 1] = np.nan
    id_mis = ~ry
    if x_as_frame:
        x = pd.DataFrame(x, columns=["x1", "x2"])
    message = f"Predictors must not contain missing values for {re.escape(method)} in the rows it fits or imputes"
    if x_as_frame:
        message += r".* Columns with missing values: \['x2'\]"
    with pytest.raises(ValueError, match=message):
        _imputer(method)(y, ry, x, id_mis=id_mis, rng=np.random.default_rng(5))
    # the same row left out of both masks is not checked
    used = np.ones_like(ry)
    used[target_row] = False
    out = _imputer(method)(y, ry & used, x, id_mis=id_mis & used, rng=np.random.default_rng(5))
    assert np.asarray(out).shape == ((id_mis & used).sum(),)


@pytest.mark.parametrize("method", ["mean", "sample"])
def test_univariate_mean_and_sample_do_not_read_the_predictors(method):
    """mean and sample, like R's, do not read x: NaN anywhere changes nothing."""
    y, ry, x = _univariate_problem(method=method)
    x_nan = np.full_like(x, np.nan)
    plain = _imputer(method)(y, ry, x, rng=np.random.default_rng(6))
    out = _imputer(method)(y, ry, x_nan, rng=np.random.default_rng(6))
    np.testing.assert_array_equal(out, plain)
