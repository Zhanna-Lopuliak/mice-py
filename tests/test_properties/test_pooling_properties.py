"""Item 6: properties of pooling with Rubin's rules (A4).

``imputation.pooling.pool_estimates(estimates, variances, dfcom,
param_names, alpha)`` returns one row per term with columns ``m, estimate,
ubar, b, t, dfcom, df, riv, lambda, fmi, std_error, statistic, p_value,
conf_low, conf_high``, and ``barnard_rubin(m, b, t, dfcom)`` returns the
Barnard-Rubin degrees of freedom. The older ``apply_rubins_rules`` is checked
where it applies.
"""

import numpy as np
import pandas as pd
import pytest
from hypothesis import HealthCheck, given, settings
from hypothesis import strategies as st
from hypothesis.extra.numpy import arrays

PLANNED_COLUMNS = [
    "m", "estimate", "ubar", "b", "t", "dfcom", "df", "riv", "lambda", "fmi",
    "std_error", "statistic", "p_value", "conf_low", "conf_high",
]
SCALE_FREE = ["df", "riv", "lambda", "fmi", "statistic", "p_value"]


def _settings(n):
    return settings(
        max_examples=n,
        deadline=None,
        derandomize=True,
        database=None,
        suppress_health_check=[HealthCheck.too_slow],
    )


@st.composite
def pooling_inputs(draw, finite_dfcom=None):
    m = draw(st.integers(2, 10))
    p = draw(st.integers(1, 3))
    est = draw(arrays(np.float64, (m, p), elements=st.floats(-1e3, 1e3, allow_subnormal=False)))
    var = draw(arrays(np.float64, (m, p), elements=st.floats(1e-4, 1e3, allow_subnormal=False)))
    if finite_dfcom is None:
        finite_dfcom = draw(st.booleans())
    dfcom = draw(st.floats(1.0, 1e5)) if finite_dfcom else np.inf
    return est, var, dfcom


def _pool(est, var, dfcom):
    from imputation.pooling import pool_estimates

    return pool_estimates(est, var, dfcom=dfcom)


# ---------------------------------------------------------------------------
# pool_estimates
# ---------------------------------------------------------------------------

def test_pool_estimates_table_layout():
    from imputation.pooling import pool_estimates

    est = np.array([[1.0, 2.0], [1.5, 2.5], [0.5, 1.0]])
    var = np.array([[0.1, 0.2], [0.1, 0.3], [0.2, 0.2]])
    table = pool_estimates(est, var, dfcom=50, param_names=["Intercept", "x"])
    assert isinstance(table, pd.DataFrame)
    assert list(table.index) == ["Intercept", "x"]
    assert list(table.columns) == PLANNED_COLUMNS
    assert (table["m"] == 3).all()
    assert (table["dfcom"] == 50).all()
    np.testing.assert_allclose(table["estimate"], est.mean(axis=0))
    np.testing.assert_allclose(table["ubar"], var.mean(axis=0))
    np.testing.assert_allclose(table["b"], est.var(axis=0, ddof=1))


def test_pool_estimates_bounds():
    @_settings(80)
    @given(pooling_inputs())
    def check(inp):
        est, var, dfcom = inp
        m = est.shape[0]
        tab = _pool(est, var, dfcom)
        assert (tab["t"] >= tab["ubar"]).all()
        assert ((tab["lambda"] >= 0) & (tab["lambda"] <= 1)).all()
        assert (tab["riv"] >= 0).all()
        assert ((tab["fmi"] >= 0) & (tab["fmi"] <= 1)).all()
        assert (tab["df"] > 0).all()
        assert (tab["df"] <= dfcom * (1 + 1e-12)).all()
        lam = tab["lambda"].to_numpy()
        with np.errstate(divide="ignore"):
            bound = np.where(lam > 0, (m - 1) / lam**2, np.inf)
        assert (tab["df"].to_numpy() <= bound * (1 + 1e-9)).all()
        np.testing.assert_allclose(tab["std_error"], np.sqrt(tab["t"]), rtol=1e-12)
        assert ((tab["conf_low"] <= tab["estimate"]) & (tab["estimate"] <= tab["conf_high"])).all()
        assert ((tab["p_value"] >= 0) & (tab["p_value"] <= 1)).all()

    check()


@pytest.mark.parametrize("dfcom", [np.inf, 25.0], ids=["dfcom_inf", "dfcom_25"])
def test_pool_estimates_zero_between_variance(dfcom):
    """Identical estimates in every imputation: b = 0, t = ubar, lambda = riv = 0."""
    est = np.tile([1.5, -2.25, 0.0], (4, 1))
    var = np.array([[0.5, 1.0, 2.0], [0.25, 1.0, 2.0], [0.5, 2.0, 1.0], [1.0, 1.0, 1.0]])
    tab = _pool(est, var, dfcom)
    assert (tab["b"] == 0).all()
    np.testing.assert_array_equal(tab["t"], tab["ubar"])
    assert (tab["lambda"] == 0).all()
    assert (tab["riv"] == 0).all()
    assert (tab["df"] <= dfcom).all()


def test_pool_estimates_ignores_order_of_imputations():
    @_settings(40)
    @given(pooling_inputs(), st.randoms(use_true_random=False))
    def check(inp, rnd):
        est, var, dfcom = inp
        order = list(range(est.shape[0]))
        rnd.shuffle(order)
        a = _pool(est, var, dfcom)
        b = _pool(est[order], var[order], dfcom)
        pd.testing.assert_frame_equal(a, b, check_exact=False, rtol=1e-10, atol=1e-12)

    check()


def test_pool_estimates_scale_equivariance():
    """Estimates times c and variances times c^2: estimate and std_error scale
    by c; df, riv, lambda, fmi, statistic and p_value do not change."""
    @_settings(40)
    @given(pooling_inputs(), st.floats(0.01, 100.0))
    def check(inp, c):
        est, var, dfcom = inp
        a = _pool(est, var, dfcom)
        b = _pool(est * c, var * c**2, dfcom)
        np.testing.assert_allclose(b["estimate"], a["estimate"] * c, rtol=1e-9, atol=1e-9)
        np.testing.assert_allclose(b["std_error"], a["std_error"] * c, rtol=1e-9)
        np.testing.assert_allclose(b["conf_low"], a["conf_low"] * c, rtol=1e-9, atol=1e-9)
        np.testing.assert_allclose(b["conf_high"], a["conf_high"] * c, rtol=1e-9, atol=1e-9)
        for col in SCALE_FREE:
            np.testing.assert_allclose(b[col], a[col], rtol=1e-7, atol=1e-12, err_msg=col)

    check()


# ---------------------------------------------------------------------------
# barnard_rubin
# ---------------------------------------------------------------------------

def test_barnard_rubin_bounds():
    from imputation.pooling import barnard_rubin

    @_settings(80)
    @given(
        st.integers(2, 50),
        st.floats(0.0, 1e3),
        st.floats(1e-4, 1e3),
        st.one_of(st.just(np.inf), st.floats(1.0, 1e5)),
    )
    def check(m, b, ubar, dfcom):
        t = ubar + (1 + 1 / m) * b
        lam = (1 + 1 / m) * b / t
        df = float(barnard_rubin(m, b, t, dfcom))
        assert df > 0
        assert df <= dfcom * (1 + 1e-12)
        if lam > 0:
            # NumPy floats: for a tiny b, lam**2 underflows to 0 and the
            # bound is infinite (Python floats would raise ZeroDivisionError)
            with np.errstate(divide="ignore", over="ignore"):
                bound = np.float64(m - 1) / np.float64(lam) ** 2
            assert df <= bound * (1 + 1e-9)
        # more complete-data information never lowers df
        assert float(barnard_rubin(m, b, t, np.inf)) >= df * (1 - 1e-12)
        # scale free. The factor is a power of two, so b and t are scaled
        # exactly: with lambda near 1, 1 - lambda cancels and one rounding of
        # t (as t * 9.0 would add) moves df by more than 1e-9 relative.
        np.testing.assert_allclose(float(barnard_rubin(m, b * 4.0, t * 4.0, dfcom)), df, rtol=1e-9)

    check()


# ---------------------------------------------------------------------------
# MICE.pool()["table"]
# ---------------------------------------------------------------------------

def test_mice_pool_table():
    from imputation import MICE

    from ._helpers import full_predictor_matrix, regression_frame

    df = regression_frame()
    mice = MICE(df)
    mice.impute(n_imputations=3, maxit=2, method="sample", predictor_matrix=full_predictor_matrix(df), seed=3)
    fits = mice.fit("y ~ x1 + x2")
    table = mice.pool()["table"]
    assert list(table.index) == list(fits[0].params.index)
    assert (table["dfcom"] == fits[0].df_resid).all()
    assert (table["df"] <= table["dfcom"]).all()
    assert (table["t"] >= table["ubar"]).all()
    np.testing.assert_allclose(table["estimate"], np.mean([f.params for f in fits], axis=0))


def test_mice_pool_multi_equation_model():
    """MNLogit has one column of coefficients per equation. Each coefficient
    is pooled on its own, matched with its variance by the covariance labels."""
    import statsmodels.formula.api as smf

    from imputation import MICE

    rng = np.random.default_rng(5)
    n = 90
    base = pd.DataFrame({"x": rng.normal(size=n), "y": rng.choice([0.0, 1.0, 2.0], size=n)})
    datasets = []
    for _ in range(3):
        d = base.copy()
        d.loc[:9, "x"] = rng.normal(size=10)
        datasets.append(d)
    mice = MICE(base)
    mice.imputed_datasets = datasets
    fits = mice.fit("y ~ x", model=smf.mnlogit, disp=False)
    table = mice.pool()["table"]
    assert list(table.index) == ["1:Intercept", "1:x", "2:Intercept", "2:x"]
    for name in table.index:
        eq, term = name.split(":")
        col = int(eq) - 1
        est = np.array([f.params.loc[term].iloc[col] for f in fits])
        var = np.array([f.bse.loc[term].iloc[col] ** 2 for f in fits])
        np.testing.assert_allclose(table.loc[name, "estimate"], est.mean(), rtol=1e-12)
        np.testing.assert_allclose(table.loc[name, "ubar"], var.mean(), rtol=1e-12)
        np.testing.assert_allclose(table.loc[name, "b"], est.var(ddof=1), rtol=1e-10)


def test_mice_pool_single_imputation_reports_the_single_analysis():
    """m = 1: nothing to pool (R's pool() warns and returns the fit). b, riv,
    lambda and fmi are NaN, t = ubar and df = dfcom, so the inference is that
    of the one complete-data analysis."""
    from imputation import MICE

    from ._helpers import full_predictor_matrix, regression_frame

    df = regression_frame()
    mice = MICE(df)
    mice.impute(n_imputations=1, maxit=1, method="sample", predictor_matrix=full_predictor_matrix(df), seed=3)
    fits = mice.fit("y ~ x1 + x2")
    with pytest.warns(UserWarning, match="m = 1"):
        out = mice.pool()
    table = out["table"]
    for col in ["b", "riv", "lambda", "fmi"]:
        assert table[col].isna().all(), col
    np.testing.assert_array_equal(table["t"], table["ubar"])
    assert (table["df"] == fits[0].df_resid).all()
    np.testing.assert_allclose(table[["conf_low", "conf_high"]], fits[0].conf_int(), rtol=1e-10)
    np.testing.assert_allclose(table["p_value"], fits[0].pvalues, rtol=1e-10)
    np.testing.assert_allclose(out["pooled_result"].conf_int(), fits[0].conf_int(), rtol=1e-10)
    np.testing.assert_allclose(out["pooled_covariance"], fits[0].cov_params(), rtol=1e-12)


def test_pooled_result_refuses_joint_tests():
    """A test of a linear combination needs its own pooling rule (R: D1);
    the pooled result says so instead of using one df for all terms."""
    from imputation import MICE

    from ._helpers import full_predictor_matrix, regression_frame

    df = regression_frame()
    mice = MICE(df)
    mice.impute(n_imputations=3, maxit=1, method="sample", predictor_matrix=full_predictor_matrix(df), seed=3)
    mice.fit("y ~ x1 + x2")
    res = mice.pool()["pooled_result"]
    for test in (res.t_test, res.f_test, res.wald_test):
        with pytest.raises(NotImplementedError, match="D1"):
            test("x1 = 0")


# ---------------------------------------------------------------------------
# Descriptive pooling: pool_descriptive_statistics and PoolingResult
# ---------------------------------------------------------------------------

def _descriptive_datasets(m=4, n=30, seed=11):
    """m completed versions of one dataset that differ in 8 'imputed' rows."""
    rng = np.random.default_rng(seed)
    base = pd.DataFrame({"x": rng.normal(size=n), "g": rng.choice(["a", "b", "c"], size=n)})
    datasets = []
    for _ in range(m):
        d = base.copy()
        rows = rng.choice(n, size=8, replace=False)
        d.loc[rows, "x"] = rng.normal(size=8)
        d.loc[rows, "g"] = rng.choice(["a", "b", "c"], size=8)
        datasets.append(d)
    return datasets


def test_pooled_mean_equals_pooled_intercept_only_regression():
    """The mean of x with variance var/n and dfcom = n - 1 is the intercept of
    the OLS fit of ``x ~ 1``, so descriptive pooling and MICE.pool agree."""
    from imputation import MICE
    from imputation.pooling import pool_descriptive_statistics

    datasets = _descriptive_datasets()
    res = pool_descriptive_statistics(datasets)
    mice = MICE(datasets[0])
    mice.imputed_datasets = datasets
    mice.fit("x ~ 1")
    ref = mice.pool()["table"].loc["Intercept"]
    row = res.table().loc["x"]
    for col in PLANNED_COLUMNS:
        np.testing.assert_allclose(row[col], ref[col], rtol=1e-10, err_msg=col)
    summ = res.summary().set_index("Parameter").loc["x"]
    np.testing.assert_allclose(summ["DF"], ref["df"], rtol=1e-10)
    np.testing.assert_allclose(summ["CI_Lower"], ref["conf_low"], rtol=1e-10)
    np.testing.assert_allclose(summ["CI_Upper"], ref["conf_high"], rtol=1e-10)
    np.testing.assert_allclose(summ["Lambda"], ref["lambda"], rtol=1e-10)
    np.testing.assert_allclose(summ["FMI"], ref["fmi"], rtol=1e-10)
    np.testing.assert_allclose(res.frac_miss_info[0], ref["fmi"], rtol=1e-10)


def test_pooled_proportion_follows_rubin_and_barnard_rubin():
    """A proportion, worked out by hand: Rubin's rules, Barnard-Rubin df with
    dfcom = n - 1, R's fmi, and a t interval."""
    from scipy import stats

    from imputation.pooling import pool_descriptive_statistics

    datasets = _descriptive_datasets()
    m, n = len(datasets), len(datasets[0])
    p = np.array([np.mean(d["g"].to_numpy() == "a") for d in datasets])
    qbar, ubar, b = p.mean(), np.mean(p * (1 - p) / n), p.var(ddof=1)
    t = ubar + (1 + 1 / m) * b
    lam, riv = (1 + 1 / m) * b / t, (1 + 1 / m) * b / ubar
    dfcom = n - 1
    df_obs = (dfcom + 1) / (dfcom + 3) * dfcom * (1 - lam)
    df = 1 / (lam**2 / (m - 1) + 1 / df_obs)
    fmi = (riv + 2 / (df + 3)) / (riv + 1)
    half = stats.t.ppf(0.975, df) * np.sqrt(t)

    summ = pool_descriptive_statistics(datasets).summary().set_index("Parameter").loc["g[a]"]
    expected = {
        "Estimate": qbar, "Std_Error": np.sqrt(t), "DF": df, "CI_Lower": qbar - half,
        "CI_Upper": qbar + half, "Within_Var": ubar, "Between_Var": b, "Total_Var": t,
        "Lambda": lam, "FMI": fmi,
    }
    for col, value in expected.items():
        np.testing.assert_allclose(summ[col], value, rtol=1e-10, err_msg=col)


# ---------------------------------------------------------------------------
# apply_rubins_rules (kept for existing callers)
# ---------------------------------------------------------------------------

def _rubin(est, var):
    from imputation.pooling import apply_rubins_rules

    qbar, t, ubar, b, lam = apply_rubins_rules(est, var)
    return {"qbar": qbar, "t": t, "ubar": ubar, "b": b, "lambda": lam}


@_settings(80)
@given(pooling_inputs())
def test_apply_rubins_rules_bounds(inp):
    est, var, _ = inp
    r = _rubin(est, var)
    assert (r["t"] >= r["ubar"]).all()
    assert ((r["lambda"] >= 0) & (r["lambda"] <= 1)).all()  # documented as FMI before 0.2
    np.testing.assert_allclose(r["qbar"], est.mean(axis=0), rtol=1e-12, atol=1e-9)
    np.testing.assert_allclose(r["ubar"], var.mean(axis=0), rtol=1e-12)


def test_apply_rubins_rules_zero_between_variance():
    est = np.tile([1.5, -2.25, 0.0], (4, 1))
    var = np.array([[0.5, 1.0, 2.0], [0.25, 1.0, 2.0], [0.5, 2.0, 1.0], [1.0, 1.0, 1.0]])
    r = _rubin(est, var)
    assert (r["b"] == 0).all()
    np.testing.assert_array_equal(r["t"], r["ubar"])
    assert (r["lambda"] == 0).all()


@_settings(40)
@given(pooling_inputs(), st.randoms(use_true_random=False))
def test_apply_rubins_rules_ignores_order_of_imputations(inp, rnd):
    est, var, _ = inp
    order = list(range(est.shape[0]))
    rnd.shuffle(order)
    a = _rubin(est, var)
    b = _rubin(est[order], var[order])
    for key in a:
        np.testing.assert_allclose(b[key], a[key], rtol=1e-10, atol=1e-9, err_msg=key)


@_settings(40)
@given(pooling_inputs(), st.floats(0.01, 100.0))
def test_apply_rubins_rules_scale_equivariance(inp, c):
    est, var, _ = inp
    a = _rubin(est, var)
    b = _rubin(est * c, var * c**2)
    np.testing.assert_allclose(b["qbar"], a["qbar"] * c, rtol=1e-9, atol=1e-9)
    np.testing.assert_allclose(np.sqrt(b["t"]), np.sqrt(a["t"]) * c, rtol=1e-9)
    np.testing.assert_allclose(b["lambda"], a["lambda"], rtol=1e-7, atol=1e-12)
