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

# The methods by kind.
# Donor methods impute observed values or labels.
DONOR_METHODS = ("pmm", "cart", "rf", "midas", "sample")
# Model-based methods for numeric targets only; they impute new floats.
NUMERIC_MODEL_METHODS = ("norm", "norm.nob", "norm.boot", "norm.predict", "mean")
# Model-based methods for categorical targets only; they impute labels of
# the column. logreg and logreg.boot impute columns with two categories
# (TWO_CATEGORY_METHODS), polyreg and polr columns with any number of
# categories (polr boolean, two-category and numeric ones through its
# fallback to the model of polyreg). polr can also impute a declared
# category of an ordered Categorical that is not observed (see
# `assert_imputed_are_observed`).
TWO_CATEGORY_METHODS = ("logreg", "logreg.boot")
CATEGORICAL_METHODS = (*TWO_CATEGORY_METHODS, "polyreg", "polr")
# Methods whose univariate imputer draws no random number; a subset of the
# others. Inside a chain, norm.predict still depends on the seed through the
# (random) imputations of the other incomplete columns; mean does not.
DETERMINISTIC_METHODS = ("norm.predict", "mean")
METHODS = DONOR_METHODS + NUMERIC_MODEL_METHODS + CATEGORICAL_METHODS

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


def has_two_categories(series):
    """True for a non-numeric column (boolean, categorical, text) with at most two observed values."""
    return not is_numeric_column(series) and series.dropna().nunique() <= 2


def can_impute(method, series):
    """True if the categorical ``method`` takes ``series`` as its target:
    logreg and logreg.boot a boolean, categorical or text column with two
    categories, polyreg and polr every boolean, categorical or text column
    (polr where its model does not apply through its fallback)."""
    if method in TWO_CATEGORY_METHODS:
        return has_two_categories(series)
    return not is_numeric_column(series)


def method_spec(method, df):
    """Method argument for ``impute``.

    A method restricted to numeric targets (midas, as R's midastouch, and
    the ``NUMERIC_MODEL_METHODS``) imputes the numeric columns and
    ``sample`` the others; a method restricted to categorical targets
    (``CATEGORICAL_METHODS``) imputes the columns it can impute (see
    `can_impute`) and ``pmm`` the others. Every other method is used for
    all columns.
    """
    if method == "midas" or method in NUMERIC_MODEL_METHODS:
        return {c: (method if is_numeric_column(df[c]) else "sample") for c in df.columns}
    if method in CATEGORICAL_METHODS:
        return {c: (method if can_impute(method, df[c]) else "pmm") for c in df.columns}
    return method


def imputes_observed_values(method):
    """True if every cell imputed with ``method_spec(method, df)`` holds a value observed in its column.

    So for the donor methods, and for the categorical methods, which impute
    labels of the column and leave the other columns to pmm; under polr an
    ordered Categorical column may also get a declared category that is not
    observed (`assert_imputed_are_observed` with ``mice`` allows that).
    """
    return method in DONOR_METHODS or method in CATEGORICAL_METHODS


def register_imputer(monkeypatch, name, imputer):
    """Make ``imputer`` the method ``name`` of ``impute`` for one test."""
    import imputation.utils as utils
    from imputation.constants import ImputationMethod

    monkeypatch.setitem(utils._IMPUTER_MAP, name, imputer)
    # the supported methods are then the registered ones
    monkeypatch.setattr(ImputationMethod, "get_all_methods", classmethod(lambda cls: list(utils._IMPUTER_MAP)))


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


def typed_frame(n=60, seed=5, index="noncontiguous"):
    """One incomplete column of each type the methods tell apart, and a complete predictor.

    ``num`` (float), ``flag`` (booleans with missing values, object dtype),
    ``two`` (Categorical with two levels, declared as yes, no), ``three``
    (text with three values, object dtype) and ``grade`` (ordered
    Categorical with four levels), all incomplete and related to the
    complete float ``x``.
    """
    rng = np.random.default_rng(seed)
    x = rng.normal(size=n)
    latent = x + rng.normal(scale=0.8, size=n)
    num = 0.6 * latent + rng.normal(scale=0.7, size=n)
    flag = pd.Series(latent + rng.normal(size=n) > 0, dtype=object)
    two = pd.Categorical(np.where(latent + rng.normal(size=n) > 0.2, "yes", "no"), categories=["yes", "no"])
    three = np.where(latent < -0.5, "lo", np.where(latent < 0.5, "mid", "hi"))
    levels = ["low", "mid", "high", "top"]
    grades = np.array(levels)[np.digitize(latent + rng.normal(scale=0.5, size=n), [-0.8, 0.0, 0.8])]
    grade = pd.Categorical(grades, categories=levels, ordered=True)
    df = pd.DataFrame(
        {"num": num, "flag": flag, "two": two, "three": pd.Series(three, dtype=object), "grade": grade, "x": x}
    )
    df = _knock_out(df, rng, {"num": 0.15, "flag": 0.15, "two": 0.15, "three": 0.15, "grade": 0.15})
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


def is_nullable_integer(dtype):
    """pandas' integer extension dtypes (``Int64`` ...), which can hold missing values."""
    return isinstance(dtype, pd.api.extensions.ExtensionDtype) and pd.api.types.is_integer_dtype(dtype)


def expected_dtypes(original, mice=None):
    """The dtypes of a completed dataset: those of ``original``, except that a
    nullable integer column imputed by a numeric-model method (R's ``norm``
    family and ``mean`` impute doubles) becomes ``Float64``."""
    dtypes = original.dtypes.to_dict()
    if mice is not None:
        for col, dtype in dtypes.items():
            if mice.method.get(col, "") in NUMERIC_MODEL_METHODS and is_nullable_integer(dtype):
                dtypes[col] = pd.Float64Dtype()
    return dtypes


def assert_dtypes_preserved(original, completed, mice=None):
    """Dtypes are kept; with ``mice``, see `expected_dtypes` for the one change."""
    assert completed.dtypes.to_dict() == expected_dtypes(original, mice)


def assert_model_imputations(original, completed, mice):
    """Columns imputed by a numeric-model method hold finite floats in their
    imputed cells; under ``mean`` they all equal the observed mean.

    Cells left missing (a skipped column, or a row whose predictors are
    incomplete) are not checked here. (``mean`` averages the observed rows
    in the model, so with an incomplete predictor that is not imputed the
    expected value would be the mean over fewer rows; no test here has one.)
    """
    for col in original.columns:
        method = mice.method.get(col, "")
        if method not in NUMERIC_MODEL_METHODS:
            continue
        dtype = completed[col].dtype
        assert pd.api.types.is_float_dtype(dtype), (col, dtype)
        mis = original[col].isna().to_numpy()
        imputed = completed[col].to_numpy(dtype=float, na_value=np.nan)[mis]
        imputed = imputed[~np.isnan(imputed)]
        assert np.isfinite(imputed).all(), (col, imputed[~np.isfinite(imputed)])
        if method == "mean":
            observed = original[col].dropna().to_numpy(dtype=float)
            eps = np.finfo(getattr(dtype, "numpy_dtype", dtype)).eps
            np.testing.assert_allclose(imputed, observed.mean(), rtol=100 * eps, atol=0, err_msg=str(col))


def declares_unobserved_categories(method, series):
    """True if ``method`` may impute a category of ``series`` that is not observed:
    polr on an ordered Categorical, which gives every declared level a
    positive probability, as R's ``MASS::polr``."""
    return method == "polr" and isinstance(series.dtype, pd.CategoricalDtype) and series.dtype.ordered


def assert_imputed_are_observed(original, completed, columns=None, mice=None):
    """Donor property: every imputed cell holds a value observed in that column.

    With ``mice``, a column that ``mice.method`` imputes with polr and that
    is an ordered Categorical may also hold any of its declared categories
    (see `declares_unobserved_categories`).
    """
    for col in columns if columns is not None else original.columns:
        mis = original[col].isna().to_numpy()
        if not mis.any():
            continue
        method = mice.method.get(col, "") if mice is not None else ""
        if declares_unobserved_categories(method, original[col]):
            observed = {label_key(v) for v in original[col].cat.categories}
        else:
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
    their missing cells. With ``donor`` (pass ``imputes_observed_values(method)``),
    filled cells must hold observed values, or under polr declared categories
    of an ordered Categorical (`assert_imputed_are_observed`). Columns imputed
    by a numeric-model method are checked by `assert_model_imputations`."""
    assert len(out) == n_imputations
    pd.testing.assert_frame_equal(original, before)
    constant = setup_skipped_columns(original) | set(allow_unimputed)
    regular = [c for c in original.columns if c not in constant]
    for completed in out:
        assert_structure_preserved(original, completed)
        assert_dtypes_preserved(original, completed, mice)
        assert_observed_unchanged(original, completed)
        assert_no_missing(completed[regular])
        assert_model_imputations(original, completed, mice)
        if donor:
            # NaN left in a skipped column is allowed; any filled value must be observed.
            # (An entirely missing column has nothing to compare with.)
            filler = {c: original[c].dropna().iloc[0] for c in constant if original[c].notna().any()}
            checked = [c for c in original.columns if original[c].notna().any()]
            assert_imputed_are_observed(original, completed.fillna(filler), columns=checked, mice=mice)
    assert_chain_stats(mice, original[regular], n_imputations, maxit)
