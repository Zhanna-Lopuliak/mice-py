"""L1 parity: pooling with Rubin's rules and Barnard-Rubin degrees of freedom.

Reference: tests/reference/r/l1_pooling.R. The Python side refits the models
on R's own completed datasets, so the inputs to pooling agree up to rounding.
"""

import numpy as np
import pandas as pd
import pytest

from imputation import MICE

from ._helpers import as_array, decode_frame, load_fixture, str_list

pytestmark = pytest.mark.parity

RTOL = 1e-8
POOL = load_fixture("l1_pooling")
MODELS = {case["name"]: case for case in POOL["models"]}

# R column name -> planned Python column name of pool_estimates / out["table"]
COLUMNS = {
    "m": "m",
    "estimate": "estimate",
    "ubar": "ubar",
    "b": "b",
    "t": "t",
    "dfcom": "dfcom",
    "df": "df",
    "riv": "riv",
    "lambda": "lambda",
    "fmi": "fmi",
    "std.error": "std_error",
    "statistic": "statistic",
    "p.value": "p_value",
    "conf.low": "conf_low",
    "conf.high": "conf_high",
}


def _py_term(term):
    return "Intercept" if term == "(Intercept)" else term


def _assert_table_matches(table, summary):
    terms = [_py_term(t) for t in str_list(summary["term"])]
    assert [str(i) for i in table.index] == terms
    for r_name, py_name in COLUMNS.items():
        np.testing.assert_allclose(
            np.asarray(table[py_name], dtype=float),
            as_array(summary[r_name]),
            rtol=RTOL,
            atol=0,
            err_msg=py_name,
        )


def _fitted_mice(model_case, **fit_kwargs):
    """MICE on nhanes with R's completed datasets in place of its own."""
    mice = MICE(decode_frame(POOL["data"]))
    mice.imputed_datasets = [decode_frame(c) for c in POOL["completed"]]
    mice.fit(model_case["formula_py"], **fit_kwargs)
    return mice


# ---------------------------------------------------------------------------
# (a) barnard_rubin and pool_estimates
# ---------------------------------------------------------------------------


def test_barnard_rubin_matches_r_on_grid():
    from imputation.pooling import barnard_rubin

    grid = POOL["barnard_rubin"]
    m, b, t, dfcom = (as_array(grid[k]) for k in ("m", "b", "t", "dfcom"))
    expected = as_array(grid["df"])
    got = np.array([barnard_rubin(*args) for args in zip(m, b, t, dfcom)], dtype=float)
    np.testing.assert_allclose(got, expected, rtol=RTOL, atol=0)


def _pool_inputs(case):
    if "per_imputation" in case:
        est = np.array([as_array(p["estimate"]) for p in case["per_imputation"]])
        se = np.array([as_array(p["std_error"]) for p in case["per_imputation"]])
        dfcom = float(case["per_imputation"][0]["df_residual"])
        names = [_py_term(t) for t in str_list(case["terms"])]
    else:
        est = np.atleast_2d(as_array(case["estimates"]))
        se = np.atleast_2d(as_array(case["std_errors"]))
        dfcom = float(as_array(case["dfcom"])[0])
        names = str_list(case["terms"])
    return est, se**2, dfcom, names


POOL_CASES = POOL["models"] + POOL["edges"]


@pytest.mark.parametrize("case", POOL_CASES, ids=[c["name"] for c in POOL_CASES])
def test_pool_estimates_matches_r(case):
    from imputation.pooling import pool_estimates

    est, var, dfcom, names = _pool_inputs(case)
    table = pool_estimates(est, var, dfcom=dfcom, param_names=names, alpha=0.05)
    _assert_table_matches(table, case["summary"])


# ---------------------------------------------------------------------------
# (b) end to end: MICE.fit / MICE.pool on R's completed datasets
# ---------------------------------------------------------------------------


def test_pool_table_linear_model_matches_r():
    mice = _fitted_mice(MODELS["lm"])
    out = mice.pool()
    _assert_table_matches(out["table"], MODELS["lm"]["summary"])


def test_pool_table_logistic_model_matches_r():
    import statsmodels.formula.api as smf

    mice = _fitted_mice(MODELS["logistic"], model=smf.logit, disp=False)
    out = mice.pool()
    _assert_table_matches(out["table"], MODELS["logistic"]["summary"])


@pytest.mark.parametrize("model_name", ["glm", "GLM"])
def test_pool_table_glm_through_model_kwargs_matches_r(model_name):
    """A binomial GLM: the family goes through ``model_kwargs``, ``tol`` and
    ``maxiter`` through to ``.fit()``. The R side uses
    ``glm.control(epsilon = 1e-14, maxit = 100)``. Both the formula function
    ``smf.glm`` and the model class ``sm.GLM`` are accepted."""
    import statsmodels.api as sm
    import statsmodels.formula.api as smf

    model = smf.glm if model_name == "glm" else sm.GLM
    mice = _fitted_mice(
        MODELS["logistic"],
        model=model,
        model_kwargs={"family": sm.families.Binomial()},
        tol=1e-14,
        maxiter=100,
    )
    assert all(isinstance(f.model.family, sm.families.Binomial) for f in mice.model_results)
    out = mice.pool()
    _assert_table_matches(out["table"], MODELS["logistic"]["summary"])


def _fit_kwargs(name):
    import statsmodels.formula.api as smf

    return {} if name == "lm" else {"model": smf.logit, "disp": False}


@pytest.mark.parametrize("name", ["lm", "logistic"])
def test_pooled_result_inference_uses_t_with_barnard_rubin_df(name):
    """Standard errors, t statistics, p-values and the 95% interval of the
    MICEresult are R's, which use a t reference with each term's own df."""
    from scipy import stats

    summary = MODELS[name]["summary"]
    res = _fitted_mice(MODELS[name], **_fit_kwargs(name)).pool()["pooled_result"]
    assert res.use_t
    np.testing.assert_allclose(res.df, as_array(summary["df"]), rtol=RTOL, atol=0)
    np.testing.assert_allclose(res.bse, as_array(summary["std.error"]), rtol=RTOL, atol=0)
    np.testing.assert_allclose(res.tvalues, as_array(summary["statistic"]), rtol=RTOL, atol=0)
    np.testing.assert_allclose(res.pvalues, as_array(summary["p.value"]), rtol=RTOL, atol=0)
    ci = np.asarray(res.conf_int())
    np.testing.assert_allclose(ci[:, 0], as_array(summary["conf.low"]), rtol=RTOL, atol=0)
    np.testing.assert_allclose(ci[:, 1], as_array(summary["conf.high"]), rtol=RTOL, atol=0)
    # other levels: the same t quantile rule on R's df and standard error.
    # Compared as midpoint and half-width: a limit close to zero would turn
    # the 3e-9 relative difference of the logistic standard errors (see
    # l1_pooling.R) into a much larger relative one.
    half = stats.t.ppf(0.95, as_array(summary["df"])) * as_array(summary["std.error"])
    ci90 = np.asarray(res.conf_int(alpha=0.10))
    np.testing.assert_allclose((ci90[:, 1] - ci90[:, 0]) / 2, half, rtol=RTOL, atol=0)
    np.testing.assert_allclose(ci90.mean(axis=1), as_array(summary["estimate"]), rtol=RTOL, atol=0)


def test_pool_summary_shows_df_and_fmi():
    """``pool(summ=True)`` returns the statsmodels summary; its parameter
    table holds R's estimates, inference, df and fmi."""
    summary = MODELS["lm"]["summary"]
    smry = _fitted_mice(MODELS["lm"]).pool(summ=True)
    params = smry.tables[1]
    assert list(params.index) == [_py_term(t) for t in str_list(summary["term"])]
    columns = {
        "Coef.": "estimate",
        "Std.Err.": "std.error",
        "t": "statistic",
        "P>|t|": "p.value",
        "[0.025": "conf.low",
        "0.975]": "conf.high",
        "df": "df",
        "FMI": "fmi",
    }
    for py_name, r_name in columns.items():
        np.testing.assert_allclose(
            params[py_name].to_numpy(dtype=float),
            as_array(summary[r_name]),
            rtol=RTOL,
            atol=0,
            err_msg=py_name,
        )
    text = str(smry)
    assert " df " in text and "FMI" in text


# ---------------------------------------------------------------------------
# (c) the other keys of pool() for the linear model
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def pooled_lm():
    mice = _fitted_mice(MODELS["lm"])
    return mice.pool()


def test_pool_linear_model_estimates_match_r(pooled_lm):
    np.testing.assert_allclose(pooled_lm["pooled_params"], as_array(MODELS["lm"]["pooled"]["estimate"]), rtol=RTOL)
    names = [_py_term(t) for t in str_list(MODELS["lm"]["pooled"]["term"])]
    assert list(pooled_lm["parameter_names"]) == names


def test_pool_linear_model_standard_errors_match_r(pooled_lm):
    np.testing.assert_allclose(
        np.sqrt(np.diag(pooled_lm["pooled_covariance"])),
        as_array(MODELS["lm"]["summary"]["std.error"]),
        rtol=RTOL,
    )


def test_pool_linear_model_within_and_between_variance_match_r(pooled_lm):
    pooled = MODELS["lm"]["pooled"]
    np.testing.assert_allclose(np.diag(pooled_lm["within_covariance"]), as_array(pooled["ubar"]), rtol=RTOL)
    np.testing.assert_allclose(np.diag(pooled_lm["between_covariance"]), as_array(pooled["b"]), rtol=RTOL)


def test_pool_linear_model_fmi_matches_r(pooled_lm):
    np.testing.assert_allclose(pooled_lm["fraction_missing_info"], as_array(MODELS["lm"]["pooled"]["fmi"]), rtol=RTOL)


def test_completed_datasets_keep_observed_values():
    """Sanity check of the fixture: R's completed datasets agree with nhanes
    on every observed cell."""
    data = decode_frame(POOL["data"])
    for payload in POOL["completed"]:
        completed = decode_frame(payload)
        observed = data.notna()
        pd.testing.assert_frame_equal(completed[observed], data[observed])
        assert not completed.isna().any().any()
