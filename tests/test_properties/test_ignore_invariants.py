"""Invariants of ``impute(ignore=...)`` over the methods, the column types and random masks (L4).

- ``ignore`` all False, ``ignore=None`` and no ``ignore`` give identical
  results for the same seed, for every method.
- Ignored rows are imputed: the completed data are missing in the same cells
  as without ``ignore`` (under the default ``where``, in none), and every
  cell that is not a where-cell is unchanged.
- Under the donor methods every imputation and every start value is an
  observed value of a row that is not ignored, also where the ignored rows
  hold values that occur in no other row.
- Which cells of the completed data are missing does not depend on
  ``ignore``: it equals the mask-only oracle `_helpers.nan_cascade`, which
  knows nothing of ``ignore``, also under random ``where`` masks.
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
    predictors_of,
    typed_frame,
)
from .test_continue_chains import _differences, _record
from .test_where_invariants import N_ORACLE_CASES, _oracle_case

SEED = 20261114


def _random_ignore(n_rows, rng, share=0.25):
    """A random `share` of the rows."""
    return rng.random(n_rows) < share


def _impute(df, **kwargs):
    """``MICE(df).impute`` with two chains and two iterations; warnings about lost values and logged events are
    expected and ignored here, any FutureWarning (a dtype pandas would change) is an error."""
    kwargs = {"n_imputations": 2, "maxit": 2, "seed": SEED, **kwargs}
    mice = MICE(df)
    with warnings.catch_warnings():
        warnings.simplefilter("error", FutureWarning)
        warnings.filterwarnings("ignore", "Observed values replaced by missing values", UserWarning)
        warnings.filterwarnings("ignore", "Number of logged events", UserWarning)
        out = mice.impute(**kwargs)
    return mice, out


@pytest.mark.parametrize("method", METHODS)
def test_no_row_ignored_in_every_form_gives_identical_results(method):
    df = typed_frame(n=40, seed=3)
    records = []
    for kwargs in ({}, {"ignore": None}, {"ignore": np.zeros(40, dtype=bool)}, {"ignore": [False] * 40}):
        mice, out = _impute(df, method=method_spec(method, df), maxit=1, **kwargs)
        records.append(_record(mice, out))
    for record in records[1:]:
        assert _differences(records[0], record) == []


@pytest.mark.parametrize("method", METHODS)
def test_ignored_rows_are_imputed(method):
    """About a third of the rows ignored, half of the incomplete rows among them: no cell is left missing, as
    without ignore."""
    df = typed_frame(n=40, seed=2)
    before = df.copy(deep=True)
    incomplete = df.isna().any(axis=1).to_numpy()
    # every fifth row and every second incomplete row
    ignore = (np.arange(40) % 5 == 0) | (incomplete & (np.cumsum(incomplete) % 2 == 0))
    assert 10 <= ignore.sum() <= 20 and (ignore & incomplete).sum() >= 5
    mice, out = _impute(df, method=method_spec(method, df), ignore=ignore)
    pd.testing.assert_frame_equal(df, before)
    np.testing.assert_array_equal(mice.ignore, ignore)
    _, without = _impute(df, method=method_spec(method, df))
    for completed, other in zip(out, without):
        pd.testing.assert_frame_equal(completed.isna(), other.isna())
        assert not completed.isna().any().any()
        assert completed.dtypes.to_dict() == other.dtypes.to_dict()
        assert_cells_outside_where_unchanged(df, mice.where, completed)


def _decoy_frame(n=40, seed=6):
    """Incomplete columns of four types and a complete predictor ``x``. The ignored rows (every fourth; returned
    with the frame) hold values that occur in no other row: ``num`` its own floats (all distinct), ``count``
    (Int64) 50 + i, ``txt`` "decoy i", ``cat`` the category "decoy", which no other row has. Only ``count`` is
    far from the other rows, so that no two columns are collinear through the decoys."""
    rng = np.random.default_rng(seed)
    x = rng.normal(size=n)
    latent = x + rng.normal(scale=0.8, size=n)
    ignore = np.arange(n) % 4 == 1
    num = 0.6 * latent + rng.normal(scale=0.7, size=n)
    count = np.round(5 + 2 * latent).astype(int)
    txt = np.where(latent < -0.5, "lo", np.where(latent < 0.5, "mid", "hi")).astype(object)
    cat = np.where(latent + rng.normal(size=n) > 0, "yes", "no").astype(object)
    decoys = np.flatnonzero(ignore)
    count[decoys] = 50 + decoys
    txt[decoys] = [f"decoy {i}" for i in decoys]
    cat[decoys] = "decoy"
    df = pd.DataFrame(
        {
            "num": num,
            "count": pd.array(count, dtype="Int64"),
            "txt": pd.Series(txt, dtype=object),
            "cat": pd.Categorical(cat, categories=["no", "yes", "decoy"]),
            "x": x,
        }
    )
    # missing cells in rows of both kinds
    for col, rows in (("num", [0, 1, 6, 9]), ("count", [2, 5, 11]), ("txt", [3, 13, 20]), ("cat", [4, 17, 30])):
        df.loc[rows, col] = None
    return df, ignore


@pytest.mark.parametrize("maxit", [0, 2], ids=["start", "chains"])
@pytest.mark.parametrize("method", DONOR_METHODS)
def test_donor_imputations_are_observed_values_of_rows_that_are_not_ignored(method, maxit):
    df, ignore = _decoy_frame()
    mice, out = _impute(df, method=method_spec(method, df), ignore=ignore, maxit=maxit)
    where = mice.where
    for col in ("num", "count", "txt", "cat"):
        assert mice.method[col] != "" and where.loc[ignore, col].any() and where.loc[~ignore, col].any(), col
        allowed = {label_key(v) for v in df.loc[~ignore, col].dropna().tolist()}
        for completed in out:
            imputed = [label_key(v) for v in completed.loc[where[col], col].tolist()]
            strangers = [v for v in imputed if v not in allowed]
            assert not strangers, f"{col}: imputed values of ignored rows {strangers}"
    # the decoys are where they were
    for completed in out:
        assert_cells_outside_where_unchanged(df, where, completed)


@pytest.mark.parametrize("seed", range(N_ORACLE_CASES))
def test_missing_cells_follow_the_mask_oracle_whatever_is_ignored(seed):
    """The oracle cases of test_where_invariants with a random fifth of the rows ignored."""
    df, kwargs = _oracle_case(seed)
    ignore = _random_ignore(len(df), np.random.default_rng([SEED, seed]), share=0.2)
    mice, out = _impute(df, ignore=ignore, **kwargs)
    predictors = {col: predictors_of(mice, col) for col in mice.visit_sequence}
    expected = nan_cascade(df, mice.where, mice.method, mice.visit_sequence, predictors, kwargs["maxit"])
    _, without = _impute(df, **kwargs)
    for completed, other in zip(out, without):
        pd.testing.assert_frame_equal(completed.isna(), expected)
        pd.testing.assert_frame_equal(other.isna(), expected)
        assert completed.dtypes.to_dict() == out[0].dtypes.to_dict()
        assert_cells_outside_where_unchanged(df, kwargs["where"], completed)
