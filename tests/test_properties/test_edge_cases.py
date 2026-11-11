"""Item 4: edge-case inputs (B1 to B5, C4, C6 and new findings B6 to B9 and A12).

Unless a test says otherwise, ``impute`` runs with the full predictor matrix
(the planned default), two imputations and two iterations, and the completed
datasets must satisfy every invariant of item 1.
"""

import numpy as np
import pandas as pd
import pytest

from ._helpers import (
    MAXIT,
    METHODS,
    N_IMP,
    assert_all_invariants,
    full_predictor_matrix,
    impute,
    method_params,
    numeric_frame,
    pmm_reference_frame,
    regression_frame,
)

NON_MIDAS = ("pmm", "cart", "rf", "sample")

C4_EMPTY = "C4: validate_dataframe drops all-empty rows"
C4_PRINT = "C4: validators print to stdout"


def _check(df, method, n_imputations=N_IMP, maxit=MAXIT, **kwargs):
    before = df.copy(deep=True)
    mice, out = impute(df, method, n_imputations=n_imputations, maxit=maxit, **kwargs)
    assert_all_invariants(df, before, mice, out, n_imputations, maxit)
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
        MICE(df).impute(
            n_imputations=1, maxit=1, method="midas", predictor_matrix=full_predictor_matrix(df), seed=1
        )


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
    assert_all_invariants(df, before, mice, out, N_IMP, MAXIT, allow_unimputed={"x1_copy"})


@pytest.mark.parametrize("method", METHODS)
def test_more_predictors_than_observed_rows(method):
    """Six complete predictors, five observed ``y`` values."""
    rng = np.random.default_rng(8)
    n = 30
    df = pd.DataFrame({f"p{i}": rng.normal(size=n) for i in range(6)})
    df.insert(0, "y", df.sum(axis=1) + rng.normal(size=n))
    df.loc[5:, "y"] = np.nan
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
        "it": [0], "im": [0], "dep": [""], "meth": ["collinear"], "out": ["y"]
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

@pytest.mark.parametrize("method", method_params({m: C4_EMPTY for m in METHODS}))
def test_all_empty_row_is_kept_and_imputed(method):
    df = numeric_frame(n=30)
    df["k"] = df["k"].astype(float)  # the int column is about to hold a NaN
    df.iloc[4, :] = np.nan
    _check(df, method)


@pytest.mark.xfail(strict=True, reason=C4_PRINT)
def test_validation_does_not_print(capsys):
    from imputation import MICE

    df = numeric_frame(n=30)
    df["k"] = df["k"].astype(float)
    df.iloc[4, :] = np.nan
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


@pytest.mark.parametrize("method", method_params(methods=NON_MIDAS))
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
    assert_all_invariants(df, before, mice, out, N_IMP, MAXIT)


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
