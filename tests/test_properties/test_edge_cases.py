"""Item 4: edge-case inputs (B1 to B5, C4, C6 and new findings B6 to B9 and A12).

Unless a test says otherwise, ``impute`` runs with the full predictor matrix
(the planned default), two imputations and two iterations, and the completed
datasets must satisfy every invariant of item 1.
"""

import re
import warnings

import numpy as np
import pandas as pd
import pytest

from .._logged_events import impute_checking_log_warning
from ._helpers import (
    CATEGORICAL_METHODS,
    MAXIT,
    METHODS,
    N_IMP,
    NUMERIC_MODEL_METHODS,
    TWO_CATEGORY_METHODS,
    assert_all_invariants,
    full_predictor_matrix,
    impute,
    imputes_observed_values,
    method_params,
    numeric_frame,
    pmm_reference_frame,
    regression_frame,
    typed_frame,
)

NON_MIDAS = ("pmm", "cart", "rf", "sample")


def _check(df, method, n_imputations=N_IMP, maxit=MAXIT, **kwargs):
    before = df.copy(deep=True)
    mice, out = impute(df, method, n_imputations=n_imputations, maxit=maxit, **kwargs)
    assert_all_invariants(df, before, mice, out, n_imputations, maxit, donor=imputes_observed_values(method))
    return mice, out


# ---------------------------------------------------------------------------
# B1: categorical target with pmm
# ---------------------------------------------------------------------------


def _categorical_target_frame(n=48, seed=21):
    rng = np.random.default_rng(seed)
    x1 = rng.normal(size=n)
    x2 = rng.normal(size=n)
    score = x1 + 0.5 * x2 + rng.normal(scale=0.5, size=n)
    g = np.where(score > 0.5, "high", np.where(score < -0.5, "low", "mid")).astype(object)
    df = pd.DataFrame({"x1": x1, "x2": x2, "g": pd.Series(g, dtype=object)})
    df.loc[[2, 7, 13, 30, 41], "g"] = np.nan
    return df


def test_pmm_categorical_target_direct():
    from imputation.PMM import pmm

    df = _categorical_target_frame()
    y = df["g"].to_numpy()
    ry = df["g"].notna().to_numpy()
    out = pmm(y, ry, df[["x1", "x2"]].to_numpy(), rng=np.random.default_rng(0))
    assert len(out) == (~ry).sum()
    assert set(out) <= set(y[ry])


@pytest.mark.parametrize("method", NON_MIDAS)
def test_categorical_target(method):
    _check(_categorical_target_frame(), method)


def test_midas_rejects_categorical_target_clearly():
    """midas, like R's midastouch, is for numeric targets: it must refuse clearly."""
    from imputation import MICE

    df = _categorical_target_frame()
    with pytest.raises((ValueError, TypeError), match=r"(?i)numeric|categor"):
        MICE(df).impute(n_imputations=1, maxit=1, method="midas", predictor_matrix=full_predictor_matrix(df), seed=1)


# ---------------------------------------------------------------------------
# B2: categorical predictor
# ---------------------------------------------------------------------------


def _categorical_predictor_frame():
    df = _categorical_target_frame()
    df["g"] = df["g"].fillna("mid")
    df["y"] = df["x1"] - df["x2"] + (df["g"] == "high") * 1.5
    df.loc[[0, 4, 9, 22, 35], "y"] = np.nan
    return df


@pytest.mark.parametrize("method", method_params())
def test_categorical_predictor(method):
    _check(_categorical_predictor_frame(), method)


# ---------------------------------------------------------------------------
# B3: more donors than observed values
# ---------------------------------------------------------------------------


def test_pmm_more_donors_than_observed_direct():
    from imputation.PMM import pmm

    y = np.array([1.0, 2.0, 3.0, np.nan, np.nan])
    ry = ~np.isnan(y)
    x = np.array([[0.1], [0.5], [0.9], [0.3], [0.7]])
    out = pmm(y, ry, x, donors=10, rng=np.random.default_rng(0))
    assert len(out) == 2
    assert np.isin(out, y[ry]).all()


def test_pmm_more_donors_than_observed_through_impute():
    df = regression_frame(n=30)
    df.loc[10:, "y"] = np.nan  # 9 observed values
    _check(df, "pmm", pmm_donors=50)


@pytest.mark.parametrize("k", [0, 1, 3, 10])
def test_matchindex_clamps_k(k):
    from imputation.PMM import matchindex

    d = np.array([0.1, 0.5, 0.9])
    t = np.array([0.2, 0.8, 2.0])
    idx = np.asarray(matchindex(d, t, k=k, rng=np.random.default_rng(0)))
    assert idx.shape == (3,)
    assert np.issubdtype(idx.dtype, np.integer)
    assert ((idx >= 0) & (idx < len(d))).all()
    if k <= 1:  # one donor: the nearest
        np.testing.assert_array_equal(idx, [0, 2, 2])


def test_matchindex_breaks_ties_at_random():
    from imputation.PMM import matchindex

    d = np.array([0.0, 1.0, 1.0, 1.0, 5.0])
    t = np.array([1.0])
    rng = np.random.default_rng(0)
    picked = {int(np.asarray(matchindex(d, t, k=1, rng=rng))[0]) for _ in range(200)}
    assert picked == {1, 2, 3}


# ---------------------------------------------------------------------------
# B4: duplicated, collinear and constant predictors
# ---------------------------------------------------------------------------

REDUNDANT = {
    "duplicate": lambda df: df["x1"].copy(),
    "collinear": lambda df: df["x1"] - 2 * df["x2"],
    "constant0": lambda df: pd.Series(0.0, index=df.index),
    "constant1": lambda df: pd.Series(1.0, index=df.index),
}


@pytest.mark.parametrize("kind", list(REDUNDANT))
@pytest.mark.parametrize("method", METHODS)
def test_redundant_predictor(method, kind):
    """A duplicated, exactly collinear or constant complete predictor.

    Before the fix of B4, whether pmm crashed on these depended on rounding
    (on other data a constant of 3.0 or an exactly collinear column could slip
    through with meaningless coefficients); on this data all four crashed.
    """
    df, _ = pmm_reference_frame()
    df["extra"] = REDUNDANT[kind](df)
    _check(df, method)


@pytest.mark.parametrize("kind", list(REDUNDANT))
def test_redundant_predictor_leaves_pmm_fit_unchanged(kind):
    """A redundant column does not change the column space, so with one donor
    and matchtype 0 the imputations must equal those without it (R drops the
    column in the setup step or in remove.lindep)."""
    df, mis = pmm_reference_frame()
    _, base = impute(df, "pmm", n_imputations=1, maxit=1, pmm_donors=1, pmm_matchtype=0)
    df["extra"] = REDUNDANT[kind](df)
    _, out = impute(df, "pmm", n_imputations=1, maxit=1, pmm_donors=1, pmm_matchtype=0)
    np.testing.assert_array_equal(out[0].loc[mis, "y"].to_numpy(), base[0].loc[mis, "y"].to_numpy())


def test_pmm_direct_with_duplicated_columns():
    from imputation.PMM import pmm

    df = regression_frame()
    y = df["y"].to_numpy()
    ry = ~np.isnan(y)
    x = np.c_[df["x1"], df["x1"], df["x2"]]
    out = pmm(y, ry, x, rng=np.random.default_rng(0))
    assert np.isin(out, y[ry]).all()


@pytest.mark.parametrize("method", METHODS)
def test_incomplete_duplicated_column(method):
    """Two identical incomplete columns. R's setup step (find.collinear) drops the
    copy from imputation and leaves it missing; imputing it is also accepted."""
    df = regression_frame()
    df.loc[[2, 6, 13, 27], "x1"] = np.nan
    df["x1_copy"] = df["x1"]
    before = df.copy(deep=True)
    mice, out = impute(df, method)
    assert_all_invariants(
        df, before, mice, out, N_IMP, MAXIT, donor=imputes_observed_values(method), allow_unimputed={"x1_copy"}
    )


@pytest.mark.parametrize("method", METHODS)
def test_more_predictors_than_observed_rows(method):
    """Six complete predictors, five observed ``y`` values.

    norm.nob and norm.boot divide by ``n_obs - q - 1``, which is not positive
    here even after remove_lindep (R imputes non-finite values): they raise.
    """
    rng = np.random.default_rng(8)
    n = 30
    df = pd.DataFrame({f"p{i}": rng.normal(size=n) for i in range(6)})
    df.insert(0, "y", df.sum(axis=1) + rng.normal(size=n))
    df.loc[5:, "y"] = np.nan
    if method in ("norm.nob", "norm.boot"):
        with pytest.raises(ValueError, match=rf"{re.escape(method)} needs more than \d+ observed values"):
            impute(df, method, check_log_warning=False)
        return
    _check(df, method)


# ---------------------------------------------------------------------------
# B5: midas with a perfect fit
# ---------------------------------------------------------------------------


def _perfect_fit():
    x = np.arange(1.0, 21.0)
    y = 2 * x + 1
    y[[3, 8, 15]] = np.nan
    return y, ~np.isnan(y), x.reshape(-1, 1)


def test_midas_perfect_fit_direct():
    from imputation.midas import midas

    y, ry, x = _perfect_fit()
    out = midas(y, ry, x, ridge=0.0, rng=np.random.default_rng(0))
    assert len(out) == 3
    assert np.isin(out, y[ry]).all()


def _perfect_fit_frame():
    """``y`` is exactly ``1 + x1 + x2``, but no predictor is collinear with it.

    With a single predictor (as in ``_perfect_fit``) ``y`` would duplicate
    ``x``: R's setup step then stops imputing ``y`` (and R stops with "No
    predictors were left"). Two weakly related predictors (correlation 0.37)
    keep the perfect fit while each correlates 0.83 with ``y``, below the
    thresholds of ``find_collinear`` and ``remove_lindep``.
    """
    x1 = np.arange(1.0, 21.0)
    x2 = (np.arange(20) * 7 % 20).astype(float)
    y = 1 + x1 + x2
    y[[3, 8, 15]] = np.nan
    return pd.DataFrame({"y": y, "x1": x1, "x2": x2})


def test_midas_perfect_fit_through_impute(monkeypatch):
    import imputation.midas as midas_module

    used = []
    original = midas_module._donor_probabilities

    def recording(*args, **kwargs):
        probs, kappa = original(*args, **kwargs)
        used.append(kappa)
        return probs, kappa

    monkeypatch.setattr(midas_module, "_donor_probabilities", recording)
    df = _perfect_fit_frame()
    _check(df, "midas", midas_ridge=0.0)
    assert used and all(k == 100 for k in used)  # the fit is perfect in every call


def test_target_collinear_with_its_only_predictor_is_not_imputed():
    """``y = 2x + 1``: the setup step finds ``y`` collinear with ``x`` and stops
    imputing it, as R's mice.edit.setup does. R then also stops with an error
    because no predictor is left; the port goes on and logs the event."""
    from imputation import MICE

    y, _, x = _perfect_fit()
    df = pd.DataFrame({"y": y, "x": x[:, 0]})
    mice = MICE(df)
    with pytest.warns(UserWarning, match=r"Number of logged events: 1\b.*\['y'\]"):
        out = mice.impute(n_imputations=2, maxit=2, method="midas", predictor_matrix=full_predictor_matrix(df), seed=1)
    assert mice.method["y"] == ""
    assert mice.visit_sequence == []
    assert mice.logged_events.to_dict("list") == {
        "it": [0],
        "im": [0],
        "dep": [""],
        "meth": ["collinear"],
        "out": ["y"],
    }
    for completed in out:
        pd.testing.assert_frame_equal(completed, df)


def test_midas_perfect_fit_uses_kappa_100():
    from imputation.midas import _donor_probabilities

    y, ry, x = _perfect_fit()
    omega = np.ones(ry.sum())
    probs, kappa = _donor_probabilities(y, ry, x, ~ry, omega, ridge=0.0)
    assert kappa == 100
    assert probs.shape == (ry.sum(), (~ry).sum())
    np.testing.assert_allclose(probs.sum(axis=0), 1.0)


# ---------------------------------------------------------------------------
# Few observed or few missing values
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("method", METHODS)
def test_column_with_one_missing_value(method):
    df = regression_frame()
    df["y"] = df["y"].fillna(0.25)
    df.loc[7, "y"] = np.nan
    _check(df, method)


@pytest.mark.parametrize("method", METHODS)
def test_column_with_one_observed_value(method):
    """A column with a single observed value is constant: R leaves it unimputed,
    filling it with that value is also accepted (see constant_columns)."""
    df = regression_frame(n=30)
    df.loc[1:, "y"] = np.nan
    _check(df, method)


@pytest.mark.parametrize("method", METHODS)
def test_column_with_two_observed_values(method):
    df = regression_frame(n=30)
    df.loc[2:, "y"] = np.nan
    _check(df, method)


@pytest.mark.parametrize("method", METHODS)
def test_constant_incomplete_column(method):
    """All observed values equal: left unimputed (R) or filled with 4.0."""
    df = regression_frame(n=30)
    df["y"] = 4.0
    df.loc[[0, 5, 10, 15, 20, 25], "y"] = np.nan
    _check(df, method)


@pytest.mark.parametrize("method", method_params())
def test_target_without_predictors(method):
    df = regression_frame(n=30)
    pm = full_predictor_matrix(df)
    pm.loc["y", :] = 0
    _check(df, method, predictor_matrix=pm)


# ---------------------------------------------------------------------------
# Rows, columns and sizes
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("method", method_params())
def test_all_empty_row_is_kept_and_imputed(method):
    df = numeric_frame(n=30)
    df["k"] = df["k"].astype(float)  # the int column is about to hold a NaN
    df.iloc[4, :] = np.nan
    with pytest.warns(UserWarning, match=r"1 row\(s\) have no observed value"):
        _check(df, method)


def test_validation_does_not_print(capsys):
    from imputation import MICE

    df = numeric_frame(n=30)
    df["k"] = df["k"].astype(float)
    df.iloc[4, :] = np.nan
    with pytest.warns(UserWarning, match=r"1 row\(s\) have no observed value"):
        MICE(df)
    assert capsys.readouterr().out == ""


@pytest.mark.parametrize("method", METHODS)
def test_fully_observed_data(method):
    df = regression_frame().dropna()
    before = df.copy(deep=True)
    _, out = impute(df, method)
    assert len(out) == N_IMP
    for completed in out:
        pd.testing.assert_frame_equal(completed, before)
    pd.testing.assert_frame_equal(df, before)


@pytest.mark.parametrize("method", METHODS)
def test_one_incomplete_column(method):
    df = regression_frame()
    assert df.isna().sum().astype(bool).sum() == 1
    _check(df, method)


@pytest.mark.parametrize("method", METHODS)
def test_single_imputation_single_iteration(method):
    _check(numeric_frame(n=30), method, n_imputations=1, maxit=1)


# ---------------------------------------------------------------------------
# Column types
# ---------------------------------------------------------------------------


def _boolean_target_frame(n=40, seed=7):
    rng = np.random.default_rng(seed)
    x1 = rng.normal(size=n)
    x2 = rng.normal(size=n)
    flag = pd.Series(x1 + rng.normal(size=n) > 0, dtype=object)
    df = pd.DataFrame({"x1": x1, "x2": x2, "flag": flag})
    df.loc[[2, 6, 11, 30], "flag"] = np.nan
    return df


@pytest.mark.parametrize("method", method_params(methods=NON_MIDAS + CATEGORICAL_METHODS))
def test_boolean_target(method):
    _check(_boolean_target_frame(), method)


def test_midas_boolean_target_keeps_labels_or_refuses():
    from imputation import MICE

    df = _boolean_target_frame()
    try:
        out = MICE(df).impute(
            n_imputations=2, maxit=1, method="midas", predictor_matrix=full_predictor_matrix(df), seed=1
        )
    except (ValueError, TypeError) as exc:  # refusing a non-numeric target is acceptable
        assert "numeric" in str(exc).lower() or "bool" in str(exc).lower()
        return
    mis = df["flag"].isna().to_numpy()
    for completed in out:
        imputed = completed["flag"].to_numpy()[mis]
        assert all(isinstance(v, (bool, np.bool_)) for v in imputed), list(imputed)


def test_midas_imputes_boolean_target_through_impute():
    """A12: midas models a boolean target as 0/1 and imputes observed booleans."""
    from imputation import MICE

    df = _boolean_target_frame()
    before = df.copy(deep=True)
    mice = MICE(df)
    out = mice.impute(
        n_imputations=N_IMP, maxit=MAXIT, method="midas", predictor_matrix=full_predictor_matrix(df), seed=1
    )
    assert_all_invariants(df, before, mice, out, N_IMP, MAXIT)
    mis = df["flag"].isna().to_numpy()
    for completed in out:
        imputed = completed["flag"].to_numpy()[mis]
        assert all(isinstance(v, (bool, np.bool_)) for v in imputed), list(imputed)


@pytest.mark.parametrize("kind", ["object", "numpy"])
def test_midas_boolean_target_direct(kind):
    from imputation.midas import midas

    df = _boolean_target_frame()
    ry = df["flag"].notna().to_numpy()
    y = df["flag"].to_numpy()
    if kind == "numpy":
        y = np.where(ry, y, False).astype(bool)
    out = midas(y, ry, df[["x1", "x2"]].to_numpy(), rng=np.random.default_rng(0))
    assert len(out) == (~ry).sum()
    assert all(isinstance(v, (bool, np.bool_)) for v in out), list(out)
    if kind == "numpy":
        assert out.dtype == bool


def test_midas_returns_observed_values_of_integer_target():
    """A12: the imputations are the donors' observed values, not float copies."""
    from imputation.midas import midas

    rng = np.random.default_rng(5)
    x = rng.normal(size=(30, 2))
    y = np.round(10 + 3 * x[:, 0] + rng.normal(size=30)).astype(np.int64)
    ry = np.ones(30, dtype=bool)
    ry[[2, 9, 17, 25]] = False
    out = midas(y, ry, x, rng=np.random.default_rng(0))
    assert out.dtype == np.int64
    assert np.isin(out, y[ry]).all()


@pytest.mark.parametrize("kind", ["object", "string", "category", "numeric_category"])
def test_midas_rejects_non_numeric_target_direct(kind):
    """Like R's midastouch (which fails for factors), midas needs a numeric target."""
    from imputation.midas import midas

    values = ["u", "v", None, "w", "u", "v", None, "w", "u", "w"]
    y = {
        "object": pd.Series(values, dtype=object),
        "string": pd.Series(values, dtype="string"),
        "category": pd.Series(values, dtype="category"),
        "numeric_category": pd.Series([1, 2, None, 3, 1, 2, None, 3, 1, 3], dtype="category"),
    }[kind]
    ry = y.notna().to_numpy()
    x = np.random.default_rng(0).normal(size=(len(y), 2))
    with pytest.raises(ValueError, match=r"Target y must be numeric for midas"):
        midas(y, ry, x, rng=np.random.default_rng(0))


@pytest.mark.parametrize("method", method_params())
def test_boolean_predictor(method):
    df = regression_frame()
    df["flag"] = df["x1"] > 0
    assert df["flag"].dtype == bool
    _check(df, method)


def _category_target_frame():
    df = _categorical_target_frame()
    df["g"] = pd.Categorical(df["g"], categories=["mid", "low", "high"])
    return df


@pytest.mark.parametrize("method", NON_MIDAS)
def test_category_dtype_target(method):
    _check(_category_target_frame(), method)


@pytest.mark.parametrize("method", method_params())
def test_category_dtype_predictor(method):
    df = _categorical_predictor_frame()
    df["g"] = pd.Categorical(df["g"], categories=["mid", "low", "high"])
    _check(df, method)


def _works_or_names_dtype(df, method, dtype_name, column="zcol"):
    """Nullable dtypes must either work (all invariants) or fail with a clear
    ``ValueError``/``TypeError`` that names the dtype and the column."""
    before = df.copy(deep=True)
    try:
        mice, out = impute(df, method)
    except Exception as exc:
        message = str(exc)
        clear = isinstance(exc, (ValueError, TypeError)) and dtype_name in message and column in message
        assert clear, repr(exc)
        return
    assert_all_invariants(df, before, mice, out, N_IMP, MAXIT, donor=imputes_observed_values(method))


def _nullable_column(dtype, n, rng):
    if dtype == "Int64":
        return pd.array(rng.integers(0, 50, size=n), dtype="Int64")
    if dtype == "Float64":
        return pd.array(np.round(rng.normal(size=n), 3), dtype="Float64")
    return pd.array(rng.choice(["u", "v", "w"], size=n), dtype="string")


@pytest.mark.parametrize("dtype", ["Int64", "Float64"])
@pytest.mark.parametrize("method", METHODS)
def test_nullable_numeric_target(method, dtype):
    rng = np.random.default_rng(13)
    df = regression_frame().drop(columns="y")
    df["zcol"] = _nullable_column(dtype, len(df), rng)
    df.loc[[2, 6, 11, 30], "zcol"] = pd.NA
    _works_or_names_dtype(df, method, dtype)


@pytest.mark.parametrize("dtype", ["Int64", "Float64"])
@pytest.mark.parametrize("method", METHODS)
def test_nullable_numeric_predictor(method, dtype):
    rng = np.random.default_rng(13)
    df = regression_frame()
    df["zcol"] = _nullable_column(dtype, len(df), rng)
    _works_or_names_dtype(df, method, dtype)


@pytest.mark.parametrize("method", NON_MIDAS)
def test_string_dtype_target(method):
    rng = np.random.default_rng(13)
    df = regression_frame().drop(columns="y")
    df["zcol"] = _nullable_column("string", len(df), rng)
    df.loc[[2, 6, 11, 30], "zcol"] = pd.NA
    _works_or_names_dtype(df, method, "string")


@pytest.mark.parametrize("method", method_params())
def test_string_dtype_predictor(method):
    rng = np.random.default_rng(13)
    df = regression_frame()
    df["zcol"] = _nullable_column("string", len(df), rng)
    _works_or_names_dtype(df, method, "string")


# ---------------------------------------------------------------------------
# B6: midas with tied predictor rows
# ---------------------------------------------------------------------------


def test_midas_tied_predictor_rows_direct():
    """Minimal reproduction: a binary predictor puts several donors at distance 0."""
    from imputation.midas import midas

    y = np.array([1.0, 2, 3, 4, 5, 6, 7, 8, np.nan, np.nan])
    x = np.array([0.0, 0, 0, 0, 1, 1, 1, 1, 0, 1]).reshape(-1, 1)
    ry = ~np.isnan(y)
    for seed in range(5):
        out = midas(y, ry, x, rng=np.random.default_rng(seed))
        assert np.isin(out, y[ry]).all()


def test_midas_binary_predictor_through_impute():
    rng = np.random.default_rng(4)
    n = 40
    x = rng.integers(0, 2, size=n).astype(float)
    df = pd.DataFrame({"y": 2 * x + rng.normal(size=n), "x": x})
    df.loc[[1, 6, 12, 19, 33], "y"] = np.nan
    _check(df, "midas")


def test_midas_donor_probabilities_are_distributions_with_ties():
    from imputation.midas import _donor_probabilities

    y = np.array([1.0, 2, 3, 4, 5, 6, 7, 8, np.nan, np.nan])
    x = np.array([0.0, 0, 0, 0, 1, 1, 1, 1, 0, 1]).reshape(-1, 1)
    ry = ~np.isnan(y)
    omega = np.array([2, 1, 0, 1, 1, 2, 0, 1])
    probs, _ = _donor_probabilities(y, ry, x, ~ry, omega)
    assert probs.shape == (8, 2)
    assert np.isfinite(probs).all() and (probs >= 0).all()
    np.testing.assert_allclose(probs.sum(axis=0), 1.0)
    assert (probs[omega == 0] == 0).all()  # donors outside the bootstrap sample get no weight


def test_midas_integer_binary_predictor_through_impute():
    """An int 0/1 predictor and two incomplete numeric columns."""
    rng = np.random.default_rng(12)
    n = 40
    group = rng.integers(0, 2, size=n)
    a = group + rng.normal(size=n)
    b = 0.5 * a - group + rng.normal(size=n)
    df = pd.DataFrame({"a": a, "b": b, "group": group})
    df.loc[[0, 7, 15, 22, 31], "a"] = np.nan
    df.loc[[3, 7, 18, 26, 35], "b"] = np.nan
    assert df["group"].dtype == np.int64
    _check(df, "midas", n_imputations=5, maxit=5)


def _three_incomplete_columns_frame(n=40, seed=7):
    """Three incomplete standard normal columns; no row is entirely missing."""
    rng = np.random.default_rng(seed)
    df = pd.DataFrame(rng.normal(size=(n, 3)), columns=["a", "b", "c"])
    mis = rng.random((n, 3)) < 0.25
    mis[mis.all(axis=1), 0] = False
    return df.mask(mis)


def test_midas_three_incomplete_numeric_columns():
    """Imputed predictors are copies of observed values, so a missing row's
    predictors can equal an observed row's. This crashed before B6 was fixed."""
    df = _three_incomplete_columns_frame()
    _check(df, "midas", n_imputations=5, maxit=5, seed=7, visit_sequence="random")


# ---------------------------------------------------------------------------
# midas: kappa through impute
# ---------------------------------------------------------------------------


def test_midas_kappa_through_impute(monkeypatch):
    """``midas_kappa`` reaches midas as ``kappa``; without it kappa is estimated."""
    import imputation.midas as midas_module

    used = []
    original = midas_module._donor_probabilities

    def recording(*args, **kwargs):
        probs, kappa = original(*args, **kwargs)
        used.append(kappa)
        return probs, kappa

    monkeypatch.setattr(midas_module, "_donor_probabilities", recording)
    df = regression_frame()
    _check(df, "midas", midas_kappa=3)
    assert len(used) == N_IMP * MAXIT
    assert all(k == 3 for k in used)

    used.clear()
    _check(df, "midas")
    assert len(used) == N_IMP * MAXIT
    assert all(k != 3 for k in used)

    with pytest.raises(TypeError, match="midas_midas_kappa"):
        impute(df, "midas", midas_midas_kappa=3)


# ---------------------------------------------------------------------------
# C6: the setup functions of imputation.design
# ---------------------------------------------------------------------------


def test_obtain_design_dummy_codes_object_columns():
    from imputation.design import obtain_design

    data = pd.DataFrame(
        {"num": [1, 2, 3, 4, 5], "g": ["b", "a", "c", "a", "b"], "y": [0.1, 0.2, 0.3, 0.4, 0.5]},
        index=[10, 3, 7, 1, 0],
    )
    x = obtain_design(data, ["num", "g"])
    assert list(x.columns) == ["num", "gb", "gc"]  # first level "a" dropped, sorted levels
    assert x.index.equals(data.index)
    assert all(pd.api.types.is_float_dtype(t) for t in x.dtypes)
    np.testing.assert_array_equal(x["gb"].to_numpy(), [1.0, 0.0, 0.0, 0.0, 1.0])
    np.testing.assert_array_equal(x["gc"].to_numpy(), [0.0, 0.0, 1.0, 0.0, 0.0])


def test_remove_lindep_drops_duplicate_and_constant_columns():
    from imputation.design import remove_lindep

    rng = np.random.default_rng(3)
    n = 40
    x1, x2 = rng.normal(size=n), rng.normal(size=n)
    x = np.c_[x1, x1, np.full(n, 2.0), x2]
    y = x1 + x2 + rng.normal(size=n)
    ry = np.ones(n, dtype=bool)
    ry[::4] = False
    keep = np.asarray(remove_lindep(x, y, ry))
    assert keep.dtype == bool and keep.shape == (4,)
    assert not keep[2]
    assert keep[0] != keep[1]
    assert keep[3]


def test_find_collinear_reports_one_of_a_duplicated_pair():
    from imputation.design import find_collinear

    df = regression_frame()
    df["x1_copy"] = df["x1"]
    found = set(find_collinear(df))
    assert len(found & {"x1", "x1_copy"}) == 1
    assert not found & {"x2", "y"}


# ---------------------------------------------------------------------------
# B11: methods without any observed target value
# ---------------------------------------------------------------------------

TREE_METHODS = ("cart", "rf")


@pytest.mark.parametrize("method", TREE_METHODS)
@pytest.mark.parametrize("kind", ["numeric", "categorical"])
def test_tree_method_without_observed_target_raises_value_error(method, kind):
    from imputation.utils import get_imputer_func

    rng = np.random.default_rng(0)
    x = rng.normal(size=(10, 2))
    y = np.full(10, np.nan) if kind == "numeric" else np.full(10, None, dtype=object)
    with pytest.raises(ValueError, match="no observed value"):
        get_imputer_func(method)(y, np.zeros(10, dtype=bool), x, rng=rng)


@pytest.mark.parametrize("method", ["pmm", "midas", *NUMERIC_MODEL_METHODS, *CATEGORICAL_METHODS])
@pytest.mark.parametrize("kind", ["numeric", "categorical"])
def test_pmm_midas_without_observed_target_raise_value_error(method, kind):
    """Like cart and rf: a clear error, not a crash deep inside the method
    (R's mean imputes NaN here)."""
    from imputation.utils import get_imputer_func

    rng = np.random.default_rng(0)
    x = rng.normal(size=(10, 2))
    y = np.full(10, np.nan) if kind == "numeric" else np.full(10, None, dtype=object)
    with pytest.raises(ValueError, match=f"{method}: the target has no observed value"):
        get_imputer_func(method)(y, np.zeros(10, dtype=bool), x, rng=rng)


@pytest.mark.parametrize("method", ["pmm", "midas", "cart", "rf", *NUMERIC_MODEL_METHODS, *CATEGORICAL_METHODS])
def test_no_observed_target_and_nothing_to_impute_returns_empty(method):
    from imputation.utils import get_imputer_func

    x = np.random.default_rng(0).normal(size=(10, 2))
    out = get_imputer_func(method)(np.full(10, np.nan), np.zeros(10, dtype=bool), x, id_mis=np.zeros(10, dtype=bool))
    assert len(out) == 0


@pytest.mark.parametrize("method", ["pmm", "midas", *TREE_METHODS, "norm", "logreg"])
def test_tree_method_without_usable_observed_row_through_impute(method):
    """``x`` has method "" and is missing wherever ``y`` is observed, so no row
    with an observed ``y`` has a complete design. (For logreg ``y`` holds two
    labels: on a numeric column with more than two values it would be refused
    before the chains start.)"""
    n = 20
    i = np.arange(n, dtype=float)
    y = np.where(np.sin(i) > 0, "a", "b").astype(object) if method == "logreg" else np.sin(i)
    df = pd.DataFrame({"y": y, "x": np.cos(i), "z": np.sin(2 * i)})
    df.loc[: n // 2 - 1, "y"] = np.nan
    df.loc[n // 2 :, "x"] = np.nan
    pm = full_predictor_matrix(df)
    with pytest.raises(ValueError, match="no observed value"):
        impute(df, {"y": method, "x": ""}, n_imputations=1, maxit=1, predictor_matrix=pm)


@pytest.mark.parametrize("method", TREE_METHODS)
@pytest.mark.parametrize("kind", ["numeric", "categorical"])
def test_tree_method_with_one_observed_target_value_imputes_it(method, kind):
    from imputation.utils import get_imputer_func

    rng = np.random.default_rng(0)
    x = rng.normal(size=(10, 2))
    if kind == "numeric":
        y = np.full(10, np.nan)
        y[3] = 2.5
    else:
        y = np.full(10, None, dtype=object)
        y[3] = "a"
    ry = np.zeros(10, dtype=bool)
    ry[3] = True
    out = get_imputer_func(method)(y, ry, x, rng=rng)
    assert list(out) == [y[3]] * 9


# ---------------------------------------------------------------------------
# B10: initial="meanobs" with non-numeric columns
# ---------------------------------------------------------------------------


def _meanobs_frame():
    """Incomplete numeric, object, category, string and boolean columns."""
    df = pd.DataFrame(
        {
            "num": [1.0, 2.0, 10.0, np.nan, 4.0, np.nan, 3.5, 2.5],
            "obj": pd.Series(["b", "a", "b", None, "c", "b", None, "a"], dtype=object),
            "cat": pd.Categorical(["y", None, "x", "x", "y", "z", None, "z"], categories=["z", "y", "x"]),
            "str": pd.Series(["p", "q", None, "q", "p", "q", None, "p"], dtype="string"),
            "bool": pd.Series([True, False, None, False, False, True, None, False], dtype=object),
            "d": np.linspace(0.0, 1.0, 8),
        }
    )
    return df


def test_meanobs_initial_imputation_of_non_numeric_columns():
    """Numeric: the observed value closest to the mean (mean 3.83, so 4.0).
    Other columns: the most frequent observed value; a tie (cat: y, x, z
    twice each) goes to the first level in category order."""
    from imputation import MICE

    df = _meanobs_frame()
    mice = MICE(df)
    impute_checking_log_warning(mice, n_imputations=1, maxit=1, initial="meanobs", method="sample", seed=1)
    data = df.copy()
    mice._initial_imputation(data, np.random.default_rng(0))
    assert data.loc[[3, 5], "num"].tolist() == [4.0, 4.0]
    assert data.loc[[3, 6], "obj"].tolist() == ["b", "b"]
    assert data.loc[[1, 6], "cat"].tolist() == ["z", "z"]
    assert data.loc[[2, 6], "str"].tolist() == ["p", "p"]  # p and q three times each: sorted first
    assert data.loc[[2, 6], "bool"].tolist() == [False, False]
    assert data.dtypes.to_dict() == df.dtypes.to_dict()


@pytest.mark.parametrize("method", NON_MIDAS + CATEGORICAL_METHODS)
def test_meanobs_initial_through_impute(method):
    df = _meanobs_frame()
    _check(df, method, initial="meanobs")


def test_sample_with_one_observed_value_and_a_series_target():
    """The single observed value is taken by position, whatever the index."""
    from imputation.sample import sample

    y = pd.Series([np.nan, 5.0, np.nan], index=[10, 11, 12])
    out = sample(y, np.array([False, True, False]), None, rng=np.random.default_rng(0))
    assert list(out) == [5.0, 5.0]


# ---------------------------------------------------------------------------
# The numeric-model methods: norm, norm.nob, norm.boot, norm.predict, mean
# ---------------------------------------------------------------------------


def test_norm_turns_nullable_integer_column_into_float64():
    """R turns an integer column imputed by norm into double: Int64 becomes
    Float64, without a pandas warning, and the observed integers are kept."""
    rng = np.random.default_rng(13)
    df = regression_frame()
    df["count"] = pd.array(np.round(10 + 3 * df["x1"] + rng.normal(size=len(df))).astype(int), dtype="Int64")
    df.loc[[0, 7, 19, 33], "count"] = pd.NA
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        _, out = _check(df, "norm")
    for completed in out:
        assert completed["count"].dtype == pd.Float64Dtype()
        imputed = completed.loc[[0, 7, 19, 33], "count"].to_numpy(dtype=float)
        assert (imputed != np.round(imputed)).any()  # real floats were written


def test_norm_keeps_float32_column():
    df = regression_frame()
    df["y"] = df["y"].astype(np.float32)
    with warnings.catch_warnings():
        warnings.simplefilter("error")  # pandas warns when it has to upcast float32
        _, out = _check(df, "norm")
    for completed in out:
        assert completed["y"].dtype == np.float32


@pytest.mark.parametrize("method", ["norm.nob", "norm.boot"])
def test_norm_nob_and_boot_need_more_observed_values_than_coefficients_plus_one(method):
    """They divide by ``n_obs - q - 1`` (R imputes infinite values where it is
    not positive). Two predictors, so q = 3: 3 and 4 observed values raise, 5
    are enough."""
    from imputation.utils import get_imputer_func

    imputer = get_imputer_func(method)
    df = regression_frame()
    y, x = df["y"].to_numpy(), df[["x1", "x2"]].to_numpy()
    observed = np.flatnonzero(~np.isnan(y))
    for n_obs in (3, 4, 5):
        ry = np.zeros(len(y), dtype=bool)
        ry[observed[:n_obs]] = True
        if n_obs <= 4:
            with pytest.raises(ValueError, match=rf"{re.escape(method)} needs more than 4 observed values"):
                imputer(y, ry, x, rng=np.random.default_rng(0))
        else:
            assert np.isfinite(imputer(y, ry, x, rng=np.random.default_rng(0))).all()


def _non_numeric_target(kind, n, rng):
    """A categorical or boolean target as MICE passes it, missing every fourth cell."""
    ry = np.arange(n) % 4 != 0
    if kind == "categorical":
        y = pd.Categorical(np.where(ry, rng.choice(["a", "b", "c"], size=n), None))
    else:
        y = np.where(ry, rng.random(n) > 0.5, None)
    return y, ry


@pytest.mark.parametrize("kind, label", [("categorical", "categorical or text"), ("boolean", "boolean")])
@pytest.mark.parametrize("method", NUMERIC_MODEL_METHODS)
def test_numeric_model_method_rejects_non_numeric_target(method, kind, label):
    """R's versions impute NA there (with warnings); the port refuses."""
    from imputation.utils import get_imputer_func

    rng = np.random.default_rng(0)
    y, ry = _non_numeric_target(kind, 20, rng)
    with pytest.raises(ValueError, match=rf"numeric for {re.escape(method)}, got {label} values"):
        get_imputer_func(method)(y, ry, rng.normal(size=(20, 2)), rng=np.random.default_rng(1))


def test_norm_rejects_categorical_target_through_impute():
    """Before the chains start, with the column named."""
    df = _categorical_target_frame()
    with pytest.raises(ValueError, match=r"Column 'g' is categorical or text .* norm imputes numeric columns only"):
        impute(df, {"g": "norm"}, n_imputations=1, maxit=1)


@pytest.mark.parametrize("method", NUMERIC_MODEL_METHODS)
def test_numeric_model_method_without_predictors(method):
    """The intercept-only model, also for norm.boot (where R stops with an
    error): norm.predict then imputes the observed mean, as mean does."""
    from imputation.utils import get_imputer_func

    imputer = get_imputer_func(method)
    rng = np.random.default_rng(3)
    y = rng.normal(loc=5.0, size=30)
    ry = np.arange(30) % 3 != 0
    for x in (np.empty((30, 0)), pd.DataFrame(index=range(30)), None):
        out = imputer(y, ry, x, rng=np.random.default_rng(2))
        assert out.shape == (10,) and np.isfinite(out).all()
        if method in ("norm.predict", "mean"):
            np.testing.assert_allclose(out, y[ry].mean(), rtol=1e-12)


def test_norm_boot_ridge_and_norm_ridge_reach_their_own_method(monkeypatch):
    """``norm_boot_ridge`` starts with the prefix of norm too; it must reach
    norm.boot only, and ``norm_ridge`` and ``norm_ls_meth`` norm only."""
    import imputation.norm
    import imputation.sampler

    seen = []

    def recording(where, original):
        def wrapper(x, y, ls_meth="qr", ridge=1e-5):
            seen.append((where, ls_meth, ridge))
            return original(x, y, ls_meth=ls_meth, ridge=ridge)

        return wrapper

    # norm fits through norm_draw (imputation.sampler), norm.boot directly
    monkeypatch.setattr(imputation.sampler, "estimice", recording("norm", imputation.sampler.estimice))
    monkeypatch.setattr(imputation.norm, "estimice", recording("norm.boot", imputation.norm.estimice))
    df = numeric_frame(n=30)
    impute(
        df,
        {"a": "norm", "b": "norm.boot", "c": "sample"},
        norm_ls_meth="ridge",
        norm_ridge=0.25,
        norm_boot_ridge=0.5,
    )
    calls = N_IMP * MAXIT
    assert sorted(seen) == [("norm", "ridge", 0.25)] * calls + [("norm.boot", "qr", 0.5)] * calls


# ---------------------------------------------------------------------------
# The categorical methods: logreg, logreg.boot, polyreg and polr
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("method", TWO_CATEGORY_METHODS)
def test_logreg_on_three_categories_raises_naming_polyreg(method):
    """R stops with "invalid 'labels'"; the port refuses clearly."""
    df = typed_frame()
    with pytest.raises(ValueError, match=rf"{re.escape(method)} imputes a target with two categories, .*polyreg"):
        impute(df, {"three": method}, check_log_warning=False)


@pytest.mark.parametrize("method", TWO_CATEGORY_METHODS)
def test_one_observed_category_is_imputed_without_random_numbers(method):
    """R imputes invented labels (logreg) or fits anyway (logreg.boot); the
    port imputes the one category and draws nothing. (Through impute such a
    column is constant and not imputed, as in R.)"""
    from imputation.utils import get_imputer_func

    from ..test_parity._helpers import ScriptedGenerator

    rng = np.random.default_rng(3)
    x = rng.normal(size=(12, 2))
    y = pd.Categorical(["no"] * 9 + [None] * 3, categories=["no", "yes"])
    scripted = ScriptedGenerator(strict=True)
    out = get_imputer_func(method)(y, y.notna(), x, rng=scripted)
    assert out.tolist() == ["no"] * 3
    assert scripted.calls == []


def _two_category_column(kind, n, rng):
    """A two-category column of the given dtype, related to nothing."""
    flags = rng.random(n) > 0.4
    if kind == "object_bool":
        return pd.Series(flags, dtype=object)
    if kind == "boolean":
        return pd.Series(flags, dtype="boolean")
    labels = np.where(flags, "yes", "no")
    if kind == "string":
        return pd.Series(labels, dtype="string")
    return pd.Series(pd.Categorical(labels, categories=["yes", "no"]))


@pytest.mark.parametrize("kind", ["object_bool", "boolean", "string", "category"])
@pytest.mark.parametrize("method", CATEGORICAL_METHODS)
def test_two_category_column_types_keep_their_dtype(method, kind):
    rng = np.random.default_rng(17)
    df = regression_frame()
    df["zcol"] = _two_category_column(kind, len(df), rng)
    df.loc[[2, 6, 11, 30], "zcol"] = None
    mice, _ = _check(df, method)
    assert mice.method["zcol"] == method


@pytest.mark.parametrize("method", CATEGORICAL_METHODS)
def test_numeric_zero_one_column(method):
    """R's logreg takes a numeric 0/1 target and imputes 0 and 1; polyreg
    imputes the two values too (R's returns them as text). As R, impute warns
    about a categorical method on a numeric column."""
    df = regression_frame()
    df["z01"] = (df["x1"] + np.random.default_rng(2).normal(size=len(df)) > 0).astype(float)
    df.loc[[3, 8, 14, 27, 33], "z01"] = np.nan
    mismatch = rf"Type mismatch for variable\(s\): z01\nImputation method {re.escape(method)} is for categorical"
    with pytest.warns(UserWarning, match=mismatch):
        mice, out = _check(df, {"z01": method})
    assert mice.method["z01"] == method
    for completed in out:
        assert completed["z01"].dtype == np.float64
        assert set(completed.loc[[3, 8, 14, 27, 33], "z01"]) <= {0.0, 1.0}


@pytest.mark.parametrize("method", CATEGORICAL_METHODS)
def test_unused_declared_category_is_never_imputed(method):
    """A two-level Categorical with a third declared category that does not
    occur: two categories to impute, the declared categories kept."""
    rng = np.random.default_rng(23)
    df = regression_frame()
    labels = np.where(df["x1"] + rng.normal(size=len(df)) > 0, "yes", "no")
    df["zcol"] = pd.Categorical(labels, categories=["maybe", "no", "yes"])
    mis = [1, 4, 9, 16, 25, 36]
    df.loc[mis, "zcol"] = np.nan
    mice, out = _check(df, method, n_imputations=4)
    assert mice.method["zcol"] == method
    for completed in out:
        assert list(completed["zcol"].cat.categories) == ["maybe", "no", "yes"]
        assert set(completed.loc[mis, "zcol"]) <= {"no", "yes"}


@pytest.mark.parametrize("key, method", [("logreg_donors", "logreg"), ("logreg_boot_ridge", "logreg.boot")])
def test_logreg_keywords_raise_takes_no_parameters(key, method):
    df = typed_frame()
    with pytest.raises(TypeError, match=rf"'{key}'\. Method '{re.escape(method)}' takes no parameters\."):
        impute(df, method, **{key: 1})


def test_polyreg_imputes_one_observed_category_everywhere():
    """R draws one uniform number per cell first, then returns the category;
    so does the port. (Through impute such a column is constant and not
    imputed, as in R.)"""
    from imputation.polyreg import polyreg

    from ..test_parity._helpers import ScriptedGenerator

    x = np.random.default_rng(3).normal(size=(12, 2))
    y = pd.Categorical(["b"] * 9 + [None] * 3, categories=["a", "b", "c"])
    scripted = ScriptedGenerator(uniforms=[0.1, 0.5, 0.9], strict=True)
    assert polyreg(y, y.notna(), x, rng=scripted).tolist() == ["b"] * 3
    scripted.assert_exhausted()
    assert scripted.order == [("uniform", 3)]


def test_polyreg_more_than_50_categories_raise():
    """R's augment stops: "Maximum number of categories (50) exceeded"."""
    from imputation.polyreg import polyreg

    y = np.array([f"l{i % 51:02d}" for i in range(110)], dtype=object)
    ry = np.arange(110) % 10 != 0
    y[~ry] = None
    x = np.random.default_rng(0).normal(size=(110, 1))
    with pytest.raises(ValueError, match=r"51 categories, more than the maximum of 50 .*exceeded"):
        polyreg(y, ry, x, rng=np.random.default_rng(1))


def test_polyreg_two_categories_with_one_missing_value():
    """R stops ("dim(X) must have a positive length"); the port imputes."""
    df = regression_frame()
    df["zcol"] = pd.Categorical(np.where(df["x1"] > 0, "yes", "no"))
    df.loc[[7], "zcol"] = np.nan
    mice, out = _check(df, {"zcol": "polyreg"})
    assert mice.method["zcol"] == "polyreg"
    assert {completed.loc[7, "zcol"] for completed in out} <= {"no", "yes"}


def test_polyreg_ordered_column_with_unused_level():
    """R stops ("dims ... do not match"); the port imputes the levels present
    and keeps the declared ones."""
    rng = np.random.default_rng(29)
    df = regression_frame()
    grades = np.array(["low", "mid", "high"])[np.digitize(df["x1"] + rng.normal(scale=0.5, size=len(df)), [-0.5, 0.5])]
    df["grade"] = pd.Categorical(grades, categories=["low", "mid", "high", "top"], ordered=True)
    mis = [0, 5, 12, 18, 27, 35]
    df.loc[mis, "grade"] = np.nan
    mice, out = _check(df, {"grade": "polyreg"}, n_imputations=4)
    assert mice.method["grade"] == "polyreg"
    for completed in out:
        assert completed["grade"].dtype == df["grade"].dtype
        assert set(completed.loc[mis, "grade"]) <= {"low", "mid", "high"}


def test_polyreg_keywords_reach_the_imputer(monkeypatch):
    """``polyreg_maxit`` and ``polyreg_max_weights`` are R's ``nnet.maxit`` and ``nnet.MaxNWts``."""
    import imputation.polyreg

    seen = []
    original = imputation.polyreg.multinom_fit

    def recording(*args, **kwargs):
        seen.append((kwargs["maxit"], kwargs["max_weights"]))
        return original(*args, **kwargs)

    monkeypatch.setattr(imputation.polyreg, "multinom_fit", recording)
    impute(typed_frame(), {"three": "polyreg"}, polyreg_maxit=7, polyreg_max_weights=99)
    assert seen and set(seen) == {(7, 99)}


def test_polyreg_too_many_weights_raise():
    """R: "too many (N) weights"."""
    with pytest.raises(ValueError, match=r"too many \(\d+\) weights.*max_weights=5"):
        impute(typed_frame(), {"three": "polyreg"}, polyreg_max_weights=5, check_log_warning=False)


def _three_category_column(kind, n, rng):
    """A three-category column of the given dtype, related to nothing."""
    labels = rng.choice(["lo", "mid", "hi"], size=n)
    if kind == "object":
        return pd.Series(labels, dtype=object)
    if kind == "string":
        return pd.Series(labels, dtype="string")
    return pd.Series(pd.Categorical(labels, categories=["mid", "lo", "hi"]))


@pytest.mark.parametrize("kind", ["object", "string", "category"])
def test_polyreg_column_types_keep_their_dtype(kind):
    """Three categories; boolean columns, which have two, are in
    test_two_category_column_types_keep_their_dtype."""
    rng = np.random.default_rng(19)
    df = regression_frame()
    df["zcol"] = _three_category_column(kind, len(df), rng)
    df.loc[[2, 6, 11, 30, 33], "zcol"] = None
    mice, _ = _check(df, "polyreg")
    assert mice.method["zcol"] == "polyreg"


def test_polr_imputes_one_observed_category_everywhere():
    """R stops on an unordered target and fits a degenerate model to an
    ordered one; the port imputes the category without a fit, after drawing
    one uniform number per cell where R draws them. (Through impute such a
    column is constant and not imputed, as in R.)"""
    from imputation.polr import polr

    from ..test_parity._helpers import ScriptedGenerator

    x = np.random.default_rng(3).normal(size=(12, 2))
    for ordered in (True, False):
        y = pd.Categorical(["b"] * 9 + [None] * 3, categories=["a", "b", "c", "d"], ordered=ordered)
        scripted = ScriptedGenerator(uniforms=[0.1, 0.5, 0.9], strict=True)
        assert polr(y, y.notna(), x, rng=scripted).tolist() == ["b"] * 3
        scripted.assert_exhausted()
        assert scripted.order == [("uniform", 3)]


def _polr_fallback_frame():
    """A two-level ordered column and a boolean column, which polr imputes by
    its fallback to the multinomial model, and complete predictors."""
    rng = np.random.default_rng(31)
    df = regression_frame()
    latent = df["x1"] + rng.normal(scale=0.7, size=len(df))
    df["level"] = pd.Categorical(np.where(latent > 0, "high", "low"), categories=["low", "high"], ordered=True)
    df["flag"] = pd.Series(latent + rng.normal(size=len(df)) > 0.3, dtype=object)
    df.loc[[0, 4, 9, 17, 26], "level"] = np.nan
    df.loc[[2, 9, 15, 33], "flag"] = np.nan
    return df


def test_polr_two_levels_and_booleans_through_the_fallback():
    """R's MASS::polr refuses both (fewer than 3 levels, not a factor) and
    mice falls back to nnet::multinom; with ``polr_log_fallback`` every call
    logs "polr falls back to multinom", as R's ``polr.to.loggedEvents``."""
    df = _polr_fallback_frame()
    mice, out = _check(df, {"level": "polr", "flag": "polr"}, polr_log_fallback=True)
    assert mice.method["level"] == mice.method["flag"] == "polr"
    for completed in out:
        assert completed["level"].dtype == df["level"].dtype
        assert set(completed["flag"]) <= {False, True}
    events = mice.logged_events[mice.logged_events["meth"] == "polr"]
    assert len(events) == 2 * N_IMP * MAXIT  # two columns, every call
    assert set(events["dep"]) == {"level", "flag"} and set(events["out"]) == {"polr falls back to multinom"}
    quiet, _ = _check(df, {"level": "polr", "flag": "polr"})
    assert not (quiet.logged_events["meth"] == "polr").any()


def test_polr_keywords_reach_the_fallback(monkeypatch):
    """``polr_maxit`` and ``polr_max_weights`` are R's ``nnet.maxit`` and
    ``nnet.MaxNWts``: they apply to the fallback fit."""
    import imputation.polr

    seen = []
    original = imputation.polr.multinom_fit

    def recording(*args, **kwargs):
        seen.append((kwargs["maxit"], kwargs["max_weights"]))
        return original(*args, **kwargs)

    monkeypatch.setattr(imputation.polr, "multinom_fit", recording)
    impute(typed_frame(), {"flag": "polr", "grade": "polr"}, polr_maxit=7, polr_max_weights=99)
    assert len(seen) == N_IMP * MAXIT and set(seen) == {(7, 99)}  # only the boolean column falls back


def test_polr_more_than_50_categories_raise():
    """R's augment stops: "Maximum number of categories (50) exceeded"."""
    from imputation.polr import polr

    y = np.array([f"l{i % 51:02d}" for i in range(110)], dtype=object)
    ry = np.arange(110) % 10 != 0
    y[~ry] = None
    x = np.random.default_rng(0).normal(size=(110, 1))
    with pytest.raises(ValueError, match=r"51 categories, more than the maximum of 50 .*exceeded"):
        polr(y, ry, x, rng=np.random.default_rng(1))


def test_polr_without_predictors_imputes():
    """R's mice.impute.polr stops without predictors (both of its fits see
    no response); the port fits the model without predictors, and for a
    two-level column the fallback on the intercept, which it logs."""
    df = _polr_fallback_frame()
    levels = ["low", "mid", "high", "top"]
    grades = np.array(levels)[np.digitize(df["x2"], [-0.7, 0.0, 0.7])]
    df["grade"] = pd.Categorical(grades, categories=levels, ordered=True)
    df.loc[[1, 3, 8, 21], "grade"] = np.nan
    pm = full_predictor_matrix(df)
    pm.loc[["grade", "level"], :] = 0
    mice, out = _check(df, {"grade": "polr", "level": "polr"}, predictor_matrix=pm, polr_log_fallback=True)
    assert mice.method["grade"] == mice.method["level"] == "polr"
    assert isinstance(df["grade"].dtype, pd.CategoricalDtype) and df["grade"].dtype.ordered
    for completed in out:
        assert completed["grade"].dtype == df["grade"].dtype
    events = mice.logged_events[mice.logged_events["meth"] == "polr"]
    assert set(events["dep"]) == {"level"} and len(events) == N_IMP * MAXIT


@pytest.mark.parametrize("method", CATEGORICAL_METHODS)
def test_categorical_method_without_predictors(method):
    """Every categorical method imputes labels without predictors (the
    intercept-only model; polr the cut points alone), whatever form the
    empty predictors take; R's polr stops there."""
    from imputation.utils import get_imputer_func

    imputer = get_imputer_func(method)
    rng = np.random.default_rng(5)
    labels = ["lo", "hi"] if method in TWO_CATEGORY_METHODS else ["lo", "mid", "hi"]
    ry = np.arange(30) % 3 != 0
    y = pd.Categorical(np.where(ry, rng.choice(labels, size=30), None), categories=labels, ordered=True)
    for x in (np.empty((30, 0)), pd.DataFrame(index=range(30)), None):
        out = imputer(y, ry, x, rng=np.random.default_rng(2))
        assert out.shape == (10,) and set(out) <= set(labels)


def test_polr_ordered_column_keeps_its_dtype_and_category_order():
    """An ordered ``category`` column declared in an order that is not
    alphabetical, with an unused level: the completed column has the same
    dtype (categories, their order, ordered), and its imputations are
    declared categories (the unused one has a small positive probability)."""
    rng = np.random.default_rng(29)
    df = regression_frame()
    grades = np.array(["low", "high", "top"])[np.digitize(df["x1"] + rng.normal(scale=0.5, size=len(df)), [-0.5, 0.5])]
    df["grade"] = pd.Categorical(grades, categories=["low", "mid", "high", "top"], ordered=True)
    mis = [0, 5, 12, 18, 27, 35]
    df.loc[mis, "grade"] = np.nan
    mice, out = _check(df, {"grade": "polr"}, n_imputations=4)
    assert mice.method["grade"] == "polr"
    for completed in out:
        assert completed["grade"].dtype == df["grade"].dtype
        assert list(completed["grade"].cat.categories) == ["low", "mid", "high", "top"]
        assert completed["grade"].cat.ordered
        assert set(completed.loc[mis, "grade"]) <= {"low", "mid", "high", "top"}
