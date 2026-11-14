"""Invariants of ``impute(where=...)`` over the methods, the column types and random masks (L4).

- Cells that are not where-cells are unchanged in every completed dataset.
- ``where=None``, ``"missing"`` and ``data.isna()`` give identical results for
  the same seed, for every method.
- Which cells of the completed data are missing equals the mask-only oracle
  `_helpers.nan_cascade`, the specification of the NaN cascade: a where-cell
  becomes missing when its column has no method, or when a predictor is
  missing in its row at the visit (a missing cell that ``where`` leaves out,
  or a where-cell that is missing itself).
- All completed datasets of a run have the same dtypes.
- ``where`` on complete integer, boolean, categorical, ordered and text
  columns works with the methods that suit them.
- Under the donor methods every imputation of a where-cell is an observed
  value of its column.
"""

import warnings

import numpy as np
import pandas as pd
import pytest

from imputation import MICE

from ._helpers import (
    DONOR_METHODS,
    METHODS,
    assert_cells_outside_where_unchanged,
    label_key,
    method_spec,
    nan_cascade,
    numeric_frame,
    predictors_of,
    typed_frame,
)
from .test_continue_chains import _differences, _record

SEED = 20261114


def _random_where(df, rng, left_out=0.3, added=0.1):
    """The missing cells of `df`, with a fraction `left_out` of them left out and a fraction `added` of the
    observed cells added."""
    missing = df.isna().to_numpy()
    draws = rng.random(missing.shape)
    return pd.DataFrame(np.where(missing, draws >= left_out, draws < added), index=df.index, columns=df.columns)


def _impute(df, **kwargs):
    """``MICE(df).impute`` with two chains and two iterations; warnings about lost values are expected and
    ignored here, any FutureWarning (a dtype pandas would change) is an error."""
    kwargs = {"n_imputations": 2, "maxit": 2, "seed": SEED, **kwargs}
    mice = MICE(df)
    with warnings.catch_warnings():
        warnings.simplefilter("error", FutureWarning)
        warnings.filterwarnings("ignore", "Observed values replaced by missing values", UserWarning)
        warnings.filterwarnings("ignore", "Number of logged events", UserWarning)
        out = mice.impute(**kwargs)
    return mice, out


def _assert_same_dtypes(out):
    for completed in out[1:]:
        assert completed.dtypes.to_dict() == out[0].dtypes.to_dict()


@pytest.mark.parametrize("method", METHODS)
def test_cells_outside_where_are_unchanged(method):
    df = typed_frame(n=40, seed=2)
    before = df.copy(deep=True)
    where = _random_where(df, np.random.default_rng(1))
    mice, out = _impute(df, method=method_spec(method, df), where=where)
    pd.testing.assert_frame_equal(df, before)
    pd.testing.assert_frame_equal(mice.where, where)
    _assert_same_dtypes(out)
    for completed in out:
        assert_cells_outside_where_unchanged(df, where, completed)


@pytest.mark.parametrize("method", METHODS)
def test_default_where_in_every_form_gives_identical_results(method):
    df = typed_frame(n=40, seed=3)
    records = []
    for where in (None, "missing", df.isna()):
        mice, out = _impute(df, method=method_spec(method, df), where=where, maxit=1)
        records.append(_record(mice, out))
    assert _differences(records[0], records[1]) == []
    assert _differences(records[0], records[2]) == []


def _random_predictor_matrix(df, rng):
    """Each column predicts each other column with probability 1/2."""
    matrix = pd.DataFrame(
        (rng.random((df.shape[1], df.shape[1])) < 0.5).astype(int), index=df.columns, columns=df.columns
    )
    np.fill_diagonal(matrix.values, 0)
    return matrix


N_ORACLE_CASES = 16


def _oracle_case(seed):
    """A random mask and predictor matrix, methods "" for some columns, a visit sequence and a number of
    iterations (0 to 3); returns (data, arguments of impute)."""
    rng = np.random.default_rng([SEED, seed])
    df = numeric_frame(n=30, seed=seed)
    kwargs = {
        "where": _random_where(df, rng, left_out=0.4, added=0.15),
        "method": {col: ("" if rng.random() < 0.2 else ("norm.predict", "pmm")[seed % 2]) for col in df.columns},
        "visit_sequence": ("roman", "arabic", "monotone", "random")[seed % 4],
        "maxit": seed % 4,
        "predictor_matrix": _random_predictor_matrix(df, rng),
    }
    return df, kwargs


@pytest.mark.parametrize("seed", range(N_ORACLE_CASES))
def test_missing_cells_follow_the_mask_oracle(seed):
    df, kwargs = _oracle_case(seed)
    mice, out = _impute(df, **kwargs)
    predictors = {col: predictors_of(mice, col) for col in mice.visit_sequence}
    expected = nan_cascade(df, mice.where, mice.method, mice.visit_sequence, predictors, kwargs["maxit"])
    _assert_same_dtypes(out)
    for completed in out:
        pd.testing.assert_frame_equal(completed.isna(), expected)
        assert_cells_outside_where_unchanged(df, kwargs["where"], completed)


def test_the_oracle_cases_lose_observed_values():
    """So that the oracle test cannot pass with masks under which nothing goes missing: most cases lose
    observed where-cells, a few of them in iterations (not only by a column without method)."""
    by_iterations = without_method = 0
    for seed in range(N_ORACLE_CASES):
        df, kwargs = _oracle_case(seed)
        mice, out = _impute(df, **kwargs)
        lost = out[0].isna() & mice.where & df.notna()
        imputed = [col for col in df.columns if mice.method[col] != ""]
        without_method += int(lost.drop(columns=imputed).to_numpy().any())
        by_iterations += int(lost[imputed].to_numpy().any())
    assert without_method >= 8 and by_iterations >= 4, (without_method, by_iterations)


def _complete_columns():
    """``y`` incomplete and ``x``; complete columns of every type: int64, bool, unordered and ordered
    Categorical, text with three values."""
    rng = np.random.default_rng(9)
    n = 40
    x = rng.normal(size=n)
    latent = x + rng.normal(scale=0.7, size=n)
    df = pd.DataFrame(
        {
            "y": latent + rng.normal(scale=0.5, size=n),
            "x": x,
            "k": np.round(3 + 2 * latent).astype(np.int64),
            "flag": latent > 0,
            "cat": pd.Categorical(np.where(latent > 0.5, "p", np.where(latent < -0.5, "q", "r"))),
            "grade": pd.Categorical(
                np.where(latent > 0.5, "hi", np.where(latent < -0.5, "lo", "mid")),
                categories=["lo", "mid", "hi"],
                ordered=True,
            ),
            "txt": pd.Series(np.where(latent + rng.normal(size=n) > 0, "u", np.where(x > 0, "v", "w")), dtype=object),
        }
    )
    df.loc[[0, 5, 10, 15], "y"] = np.nan
    return df


@pytest.mark.parametrize(
    "column, method, dtype",
    [
        ("k", "pmm", "Int64"),
        ("k", "sample", "Int64"),
        ("k", "cart", "Int64"),
        ("k", "norm", "Float64"),
        ("k", "mean", "Float64"),
        ("flag", "logreg", "boolean"),
        ("flag", "pmm", "boolean"),
        ("flag", "sample", "boolean"),
        ("cat", "polyreg", "category"),
        ("cat", "cart", "category"),
        ("grade", "polr", "category"),
        ("grade", "pmm", "category"),
        ("txt", "polyreg", "object"),
        ("txt", "sample", "object"),
        ("txt", "rf", "object"),
    ],
)
def test_where_on_a_complete_column_of_each_type(column, method, dtype):
    df = _complete_columns()
    where = df.isna()
    where.loc[[1, 2, 3, 20, 21, 22], column] = True
    mice, out = _impute(df, method={column: method}, where=where)
    assert mice.method[column] == method and column in mice.visit_sequence
    _assert_same_dtypes(out)
    observed = {label_key(v) for v in df[column].tolist()}
    for completed in out:
        assert str(completed[column].dtype) == dtype
        if dtype == "category":
            assert completed[column].dtype == df[column].dtype
        assert completed.notna().all().all()
        imputed = completed.loc[where[column], column].tolist()
        if method in DONOR_METHODS or dtype in ("boolean", "category", "object"):
            assert {label_key(v) for v in imputed} <= observed
        assert_cells_outside_where_unchanged(df, where, completed)


@pytest.mark.parametrize("method", DONOR_METHODS)
def test_donor_imputations_of_where_cells_are_observed_values(method):
    df = typed_frame(n=40, seed=4)
    where = _random_where(df, np.random.default_rng(2), left_out=0.2, added=0.2)
    spec = method_spec(method, df)
    mice, out = _impute(df, method=spec, where=where)
    for completed in out:
        for col in df.columns:
            observed = {label_key(v) for v in df[col].dropna().tolist()}
            imputed = completed.loc[where[col], col].dropna().tolist()
            assert {label_key(v) for v in imputed} <= observed, col
