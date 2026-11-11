"""Tests of the layer-1 tests: introduce a defect into the library, check that it was applied and that L1 fails.

Each mutant patches one function for one test (it is undone afterwards) and
counts its calls, so that a test cannot pass because the mutant was never
reached. The checks are those of test_l1_norm, run on one of its cases:

* (a) ``norm`` without the parameter draw: least-squares coefficients and
  residual standard deviation instead of ``norm_draw``. The imputations
  differ and the chi-square and parameter normals are left over.
* (b) ``chisquare`` asked with ``n_obs`` instead of ``n_obs - q`` degrees of
  freedom. The scripted value is the same, so the imputations still match;
  only the logged ``df`` shows it.
* (c) ``norm.nob`` with the textbook residual df ``n_obs - q`` instead of R's
  ``n_obs - q - 1``. The imputations differ.
* (d) ``norm.boot`` without the bootstrap (the observed rows themselves).
  The imputations differ and the bootstrap indices are left over.
* (e) ``norm.boot`` drawing its indices from ``integers(0, n_obs - 1)``.
  Where R's indices all lie inside that range, the imputations still match
  and the logged bounds show the defect; where one is ``n_obs - 1``, the
  scripted generator refuses it.
"""

import numpy as np
import pytest

import imputation.norm
import imputation.sampler

from .test_l1_norm import CASES, check_draws, check_imputations, replay

pytestmark = pytest.mark.parity


def _counted(calls, function):
    """`function`, appending to `calls` on every call."""

    def wrapper(*args, **kwargs):
        calls.append(1)
        return function(*args, **kwargs)

    return wrapper


def test_norm_without_parameter_draw_is_caught(monkeypatch):
    calls = []

    def least_squares(y, ry, x, rng=None, ls_meth="qr", ridge=1e-5, rank_adjust=True):
        p = imputation.sampler.estimice(x[ry], y[ry], ls_meth=ls_meth, ridge=ridge)
        coef = np.nan_to_num(p["c"], nan=0.0)
        sigma = float(np.sqrt(np.sum(p["r"] ** 2) / p["df"]))
        return {"coef": coef, "beta": coef, "sigma": sigma, "estimation": ls_meth}

    monkeypatch.setattr(imputation.norm, "norm_draw", _counted(calls, least_squares))
    case = CASES["norm_qr_seed1"]
    imputed, rng = replay(case)
    assert calls, "mutant not applied: norm does not call imputation.norm.norm_draw"
    with pytest.raises(AssertionError, match="Not equal to tolerance"):
        check_imputations(case, imputed)
    with pytest.raises(AssertionError, match=r"scripted values left over: 1 chisq, 3 normal"):
        check_draws(case, rng)


def test_chisquare_with_n_obs_degrees_of_freedom_is_caught(monkeypatch):
    calls = []
    original = imputation.sampler.estimice

    def wrong_df(x, y, **kwargs):
        p = original(x, y, **kwargs)
        p["df"] = len(y)  # n_obs instead of n_obs - q
        return p

    # norm_draw looks estimice up in imputation.sampler
    monkeypatch.setattr(imputation.sampler, "estimice", _counted(calls, wrong_df))
    case = CASES["norm_qr_seed1"]
    imputed, rng = replay(case)
    assert calls, "mutant not applied: norm_draw does not call imputation.sampler.estimice"
    assert [call.args["df"] for call in rng.calls if call.method == "chisquare"] == [case["n_obs"]]
    check_imputations(case, imputed)  # the same scripted chi-square value: invisible here
    with pytest.raises(AssertionError, match=r"assert \[14\] == \[11\]"):
        check_draws(case, rng)


def test_norm_nob_with_textbook_residual_df_is_caught(monkeypatch):
    calls = []

    def textbook(residuals, n_obs, n_coef):
        return float(np.sqrt(np.sum(residuals**2) / (n_obs - n_coef)))

    monkeypatch.setattr(imputation.norm, "_residual_sd", _counted(calls, textbook))
    case = CASES["norm_nob"]
    imputed, rng = replay(case)
    assert calls, "mutant not applied: norm_nob does not call imputation.norm._residual_sd"
    with pytest.raises(AssertionError, match="Not equal to tolerance"):
        check_imputations(case, imputed)
    check_draws(case, rng)  # the draws themselves are right


def test_norm_boot_without_bootstrap_is_caught(monkeypatch):
    calls = []
    monkeypatch.setattr(imputation.norm, "_bootstrap_sample", _counted(calls, lambda n, rng: np.arange(n)))
    case = CASES["norm_boot"]
    imputed, rng = replay(case)
    assert calls, "mutant not applied: norm_boot does not call imputation.norm._bootstrap_sample"
    with pytest.raises(AssertionError, match="Not equal to tolerance"):
        check_imputations(case, imputed)
    with pytest.raises(AssertionError, match=r"scripted values left over: 14 integer"):
        check_draws(case, rng)


def _short_bootstrap(n, rng):
    return rng.integers(0, n - 1, size=n)


def test_norm_boot_with_short_index_range_is_caught_by_the_logged_bounds(monkeypatch):
    """R's indices of ``norm_boot`` all lie below ``n_obs - 1``, so the run
    goes through with the right imputations; the logged bounds of
    ``integers`` show the defect."""
    calls = []
    monkeypatch.setattr(imputation.norm, "_bootstrap_sample", _counted(calls, _short_bootstrap))
    case = CASES["norm_boot"]
    assert max(case["integers"]) < case["n_obs"] - 1
    imputed, rng = replay(case)
    assert calls, "mutant not applied: norm_boot does not call imputation.norm._bootstrap_sample"
    check_imputations(case, imputed)  # the same scripted indices: invisible here
    with pytest.raises(AssertionError, match=r"'high': 13"):
        check_draws(case, rng)


def test_norm_boot_with_short_index_range_is_caught_by_the_generator(monkeypatch):
    """R's indices of ``norm_boot_one_missing`` include ``n_obs - 1``, which
    the defective range excludes: the scripted generator refuses it."""
    calls = []
    monkeypatch.setattr(imputation.norm, "_bootstrap_sample", _counted(calls, _short_bootstrap))
    case = CASES["norm_boot_one_missing"]
    assert case["n_obs"] - 1 in case["integers"]
    with pytest.raises(AssertionError, match=r"outside the requested range \[0, 18\)"):
        replay(case)
    assert calls, "mutant not applied: norm_boot does not call imputation.norm._bootstrap_sample"
