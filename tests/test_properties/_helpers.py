"""Shared data builders and assertions for the L4 invariant tests.

The tests in this folder describe how the library must behave after the
overhaul (see the interface contract). Checks that fail today because of a
known finding carry ``xfail(strict=True, reason="<ID>: ...")``.
"""

import numbers

import numpy as np
import pandas as pd
import pytest

from .._logged_events import impute_checking_log_warning

METHODS = ("pmm", "cart", "rf", "midas", "sample")

# Small and quick: the whole folder has to run in about 20 seconds.
N_IMP = 2
MAXIT = 2
SEED = 20240611


def xfail(reason):
    """A strict xfail marker; ``reason`` starts with the finding ID."""
    return pytest.mark.xfail(strict=True, reason=reason)


def method_params(xfails=None, methods=METHODS):
    """Parametrize over ``methods``; ``xfails`` maps a method to its reason."""
    xfails = xfails or {}
    return [pytest.param(m, marks=xfail(xfails[m]), id=m) if m in xfails else pytest.param(m, id=m) for m in methods]


def full_predictor_matrix(df):
    """All ones with a zero diagonal: R's default and the default of ``impute``.

    Passing it explicitly keeps these tests independent of that default.
    """
    pm = pd.DataFrame(1, index=df.columns, columns=df.columns)
    np.fill_diagonal(pm.values, 0)
    return pm


def is_numeric_column(series):
    return pd.api.types.is_numeric_dtype(series) and not pd.api.types.is_bool_dtype(series)


def method_spec(method, df):
    """Method argument for ``impute``.

    midas only imputes numeric targets (as R's midastouch), so with midas the
    non-numeric columns are imputed by ``sample``. Every other method is used
    for all columns.
    """
    if method != "midas":
        return method
    return {c: ("midas" if is_numeric_column(df[c]) else "sample") for c in df.columns}


def impute(
    df, method, n_imputations=N_IMP, maxit=MAXIT, seed=SEED, predictor_matrix="full", check_log_warning=True, **kwargs
):
    """Run ``MICE(df).impute`` with the full predictor matrix; return (mice, datasets).

    With `check_log_warning`, the "Number of logged events" warning is
    checked against ``mice.logged_events`` instead of being passed on.
    """
    from imputation import MICE

    if isinstance(predictor_matrix, str) and predictor_matrix == "full":
        predictor_matrix = full_predictor_matrix(df)
    mice = MICE(df)
    run = impute_checking_log_warning if check_log_warning else (lambda mice, **kw: mice.impute(**kw))
    out = run(
        mice,
        n_imputations=n_imputations,
        maxit=maxit,
        method=method_spec(method, df),
        predictor_matrix=predictor_matrix,
        seed=seed,
        **kwargs,
    )
    return mice, out


# ---------------------------------------------------------------------------
# Data builders
# ---------------------------------------------------------------------------


def _knock_out(df, rng, fractions):
    n = len(df)
    for col, frac in fractions.items():
        rows = rng.choice(n, size=max(2, int(round(frac * n))), replace=False)
        df.iloc[rows, df.columns.get_loc(col)] = np.nan
    return df


def numeric_frame(n=60, seed=0, index="noncontiguous"):
    """Three incomplete correlated numeric columns, one complete float, one complete int."""
    rng = np.random.default_rng(seed)
    a = rng.normal(size=n)
    b = 0.6 * a + rng.normal(scale=0.8, size=n)
    c = -0.5 * a + 0.4 * b + rng.normal(scale=0.7, size=n)
    d = rng.normal(size=n)
    k = rng.integers(0, 10, size=n)
    df = pd.DataFrame({"a": a, "b": b, "c": c, "d": d, "k": k})
    df = _knock_out(df, rng, {"a": 0.15, "b": 0.2, "c": 0.25})
    df.index = make_index(n, index, rng)
    return df


def mixed_frame(n=60, seed=1, index="noncontiguous"):
    """Numeric columns plus an object column and a pandas ``category`` column, all incomplete."""
    rng = np.random.default_rng(seed)
    a = rng.normal(size=n)
    g = np.where(a + rng.normal(scale=0.7, size=n) > 0.3, "hi", np.where(a < -0.5, "lo", "mid"))
    b = 0.5 * a + (g == "hi") + rng.normal(scale=0.7, size=n)
    h = pd.Categorical(rng.choice(["x", "y", "z"], size=n), categories=["z", "x", "y"])
    d = rng.normal(size=n)
    df = pd.DataFrame({"a": a, "b": b, "g": pd.Series(g, dtype=object), "h": h, "d": d})
    df = _knock_out(df, rng, {"a": 0.15, "b": 0.2, "g": 0.15, "h": 0.1})
    df.index = make_index(n, index, rng)
    return df


def make_index(n, kind, rng):
    if kind == "range":
        return pd.RangeIndex(n)
    if kind == "noncontiguous":
        # Non-default, with gaps, not sorted.
        return pd.Index(rng.permutation(np.arange(n) * 7 + 3))
    if kind == "string":
        return pd.Index([f"r{i:03d}" for i in rng.permutation(n)])
    raise ValueError(kind)


def pmm_reference_frame():
    """``y`` linear in two complete uniform predictors, six ``y`` cells missing.

    The fitted values have no near ties, so PMM with one donor and
    ``matchtype=0`` is deterministic. Returns ``(df, missing_mask)``.
    """
    rng = np.random.default_rng(11)
    n = 50
    x1 = rng.uniform(0, 10, n)
    x2 = rng.uniform(0, 5, n)
    y = 1 + 2 * x1 - x2 + rng.normal(0, 1, n)
    df = pd.DataFrame({"y": y, "x1": x1, "x2": x2})
    mis = np.zeros(n, dtype=bool)
    mis[[3, 10, 17, 25, 33, 41]] = True
    df.loc[mis, "y"] = np.nan
    return df, mis


def regression_frame(n=40, seed=3):
    """``y`` depends on two complete predictors; four ``y`` cells are missing."""
    rng = np.random.default_rng(seed)
    x1 = rng.normal(size=n)
    x2 = rng.normal(size=n)
    y = x1 + 0.5 * x2 + rng.normal(scale=0.5, size=n)
    df = pd.DataFrame({"y": y, "x1": x1, "x2": x2})
    df.loc[[1, 5, 9, 20], "y"] = np.nan
    return df


# ---------------------------------------------------------------------------
# Assertions
# ---------------------------------------------------------------------------


def label_key(value):
    """Comparable key that keeps booleans apart from the numbers 0 and 1."""
    if isinstance(value, (bool, np.bool_)):
        return ("bool", bool(value))
    if isinstance(value, str):
        return ("str", str(value))
    if isinstance(value, numbers.Number):
        return ("num", float(value))
    return ("other", value)


def assert_observed_unchanged(original, completed):
    for col in original.columns:
        obs = original[col].notna().to_numpy()
        expected = original[col][obs].tolist()
        got = completed[col][obs].tolist()
        assert [label_key(v) for v in got] == [label_key(v) for v in expected], col


def assert_no_missing(completed):
    missing = completed.isna().sum()
    assert missing.sum() == 0, missing[missing > 0].to_dict()


def assert_structure_preserved(original, completed):
    assert completed.shape == original.shape
    assert list(completed.columns) == list(original.columns)
    assert completed.index.equals(original.index)
    assert type(completed.index) is type(original.index)


def assert_dtypes_preserved(original, completed):
    assert completed.dtypes.to_dict() == original.dtypes.to_dict()


def assert_imputed_are_observed(original, completed, columns=None):
    """Donor property: every imputed cell holds a value observed in that column."""
    for col in columns if columns is not None else original.columns:
        mis = original[col].isna().to_numpy()
        if not mis.any():
            continue
        observed = {label_key(v) for v in original[col][~mis].tolist()}
        imputed = [label_key(v) for v in completed[col][mis].tolist()]
        strangers = [v for v in imputed if v not in observed]
        assert not strangers, (col, strangers[:5])


def assert_chain_stats(mice, original, n_imputations, maxit):
    for col in original.columns:
        n_mis = int(original[col].isna().sum())
        if n_mis == 0 or not is_numeric_column(original[col]):
            continue
        mean = np.asarray(mice.chain_mean[col], dtype=float)
        var = np.asarray(mice.chain_var[col], dtype=float)
        assert mean.shape == (maxit, n_imputations), col
        assert var.shape == (maxit, n_imputations), col
        assert np.isfinite(mean).all(), col
        if n_mis >= 2:  # the variance of a single imputed value is undefined, as in R
            assert np.isfinite(var).all(), col


def constant_columns(df):
    """Incomplete columns whose observed values are all equal.

    R's setup step treats such a column as constant and leaves it unimputed;
    filling it with its only observed value is equally acceptable here. R's
    rule is a variance below ``1000 * eps``, so numeric columns whose
    observed values differ by less than that count as constant too.
    """
    out = set()
    for c in df.columns:
        if not df[c].isna().any():
            continue
        observed = df[c].dropna()
        if observed.nunique() <= 1:
            out.add(c)
        elif is_numeric_column(df[c]) and observed.astype(float).var() < 1000 * np.finfo(float).eps:
            out.add(c)
    return out


def setup_skipped_columns(df):
    """Incomplete columns that R's setup step (mice.edit.setup) does not impute.

    These are the constant columns and, with the full predictor matrix, the
    incomplete columns that ``find_collinear`` (R's ``find.collinear``,
    checked against R in test_l1_setup) reports among the other columns.
    Leaving them missing is R's behaviour; imputing them is also accepted.
    """
    from imputation.design import find_collinear

    constant = constant_columns(df)
    rest = df.drop(columns=list(constant))
    collinear = {c for c in find_collinear(rest) if df[c].isna().any()} if rest.shape[1] > 1 else set()
    return constant | collinear


def assert_all_invariants(original, before, mice, out, n_imputations, maxit, donor=True, allow_unimputed=()):
    """Item 1 invariants. Columns in ``allow_unimputed`` and the columns R's setup
    step skips (constant or collinear, see ``setup_skipped_columns``) may keep
    their missing cells; filled cells must still hold observed values."""
    assert len(out) == n_imputations
    pd.testing.assert_frame_equal(original, before)
    constant = setup_skipped_columns(original) | set(allow_unimputed)
    regular = [c for c in original.columns if c not in constant]
    for completed in out:
        assert_structure_preserved(original, completed)
        assert_dtypes_preserved(original, completed)
        assert_observed_unchanged(original, completed)
        assert_no_missing(completed[regular])
        if donor:
            # NaN left in a skipped column is allowed; any filled value must be observed.
            # (An entirely missing column has nothing to compare with.)
            filler = {c: original[c].dropna().iloc[0] for c in constant if original[c].notna().any()}
            checked = [c for c in original.columns if original[c].notna().any()]
            assert_imputed_are_observed(original, completed.fillna(filler), columns=checked)
    assert_chain_stats(mice, original[regular], n_imputations, maxit)
