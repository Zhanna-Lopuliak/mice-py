"""Item 6: properties of pooling with Rubin's rules (A4).

Planned: ``imputation.pooling.pool_estimates(estimates, variances, dfcom,
param_names, alpha)`` returns one row per term with columns ``m, estimate,
ubar, b, t, dfcom, df, riv, lambda, fmi, std_error, statistic, p_value,
conf_low, conf_high``, and ``barnard_rubin(m, b, t, dfcom)`` returns the
Barnard-Rubin degrees of freedom. Neither exists today, so those tests are
xfailed with A4. Today's ``apply_rubins_rules`` is checked where it applies.
"""

import numpy as np
import pandas as pd
import pytest
from hypothesis import HealthCheck, given, settings
from hypothesis import strategies as st
from hypothesis.extra.numpy import arrays

A4 = "A4: no pool_estimates / barnard_rubin; pooling uses the normal reference"

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
# Planned pool_estimates
# ---------------------------------------------------------------------------

@pytest.mark.xfail(strict=True, reason=A4)
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


@pytest.mark.xfail(strict=True, reason=A4)
def test_pool_estimates_bounds():
    _pool(np.ones((2, 1)), np.ones((2, 1)), np.inf)  # fail fast while the function is missing

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


@pytest.mark.xfail(strict=True, reason=A4)
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


@pytest.mark.xfail(strict=True, reason=A4)
def test_pool_estimates_ignores_order_of_imputations():
    _pool(np.ones((2, 1)), np.ones((2, 1)), np.inf)

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


@pytest.mark.xfail(strict=True, reason=A4)
def test_pool_estimates_scale_equivariance():
    """Estimates times c and variances times c^2: estimate and std_error scale
    by c; df, riv, lambda, fmi, statistic and p_value do not change."""
    _pool(np.ones((2, 1)), np.ones((2, 1)), np.inf)

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
# Planned barnard_rubin
# ---------------------------------------------------------------------------

@pytest.mark.xfail(strict=True, reason=A4)
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
            assert df <= (m - 1) / lam**2 * (1 + 1e-9)
        # more complete-data information never lowers df
        assert float(barnard_rubin(m, b, t, np.inf)) >= df * (1 - 1e-12)
        # scale free
        np.testing.assert_allclose(float(barnard_rubin(m, b * 9.0, t * 9.0, dfcom)), df, rtol=1e-9)

    check()


# ---------------------------------------------------------------------------
# Planned MICE.pool()["table"]
# ---------------------------------------------------------------------------

@pytest.mark.xfail(strict=True, reason=A4 + " (MICE.pool has no 'table')")
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


# ---------------------------------------------------------------------------
# Today's apply_rubins_rules
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
    assert ((r["lambda"] >= 0) & (r["lambda"] <= 1)).all()  # reported today as frac_miss_info
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
