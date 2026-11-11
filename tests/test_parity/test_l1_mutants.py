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

The mutants of ``logreg``, ``logreg.boot`` and ``augment`` run the checks of
test_l1_augment and test_l1_logreg:

* (f) no augmentation. The added rows are missing, and the fit of the
  separated observed rows diverges.
* (g) the added rows with weight 1 instead of ``(p + 1) / (2 p k)``.
* (h) the added rows not clamped to the range of their predictor. Only the
  rare dummy, where the clamp binds, shows it.
* (i) augmentation although exactly one value is missing (R skips it).
* (j) the covariance of the coefficient draw multiplied by the quasibinomial
  dispersion (R's ``cov.scaled`` instead of ``cov.unscaled``).
* (k) the covariance recomputed at the final coefficients instead of taken
  from the last least-squares step.
* (l) ``logreg`` without the draw of the coefficients: the normals are left
  over.
* (m) ``logreg.boot`` without the bootstrap: the indices are left over.
* (n) the reference category taken in the declared order of an unordered
  factor instead of the sorted order: on a factor declared as (yes, no) the
  fit models the other category.
"""

import numpy as np
import pandas as pd
import pytest

import imputation.logreg
import imputation.norm
import imputation.sampler

from . import test_l1_augment, test_l1_logreg
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


# ---------------------------------------------------------------------------
# logreg, logreg.boot and augment
# ---------------------------------------------------------------------------

AUGMENT = test_l1_augment.CASES
LOGREG = test_l1_logreg.CASES


def _augment_mismatch(name):
    """Replay augment case `name`; return the AssertionError of `check_augmented`."""
    case = AUGMENT[name]
    y, aug = test_l1_augment.replay(case)
    with pytest.raises(AssertionError) as info:
        test_l1_augment.check_augmented(case, y, aug)
    return str(info.value)


def test_logreg_without_augmentation_is_caught(monkeypatch):
    calls = []

    def unchanged(y, id_obs, x, id_mis, max_categories=50):
        y, x = np.asarray(y), np.asarray(x, dtype=float).reshape(len(y), -1)
        return imputation.logreg.Augmented(y, np.asarray(id_obs), x, np.asarray(id_mis), np.ones(len(y)))

    monkeypatch.setattr(imputation.logreg, "augment", _counted(calls, unchanged))
    assert "shapes (12,), (20,) mismatch" in _augment_mismatch("levels2")
    case = LOGREG["logreg_separated"]
    steps, rng = test_l1_logreg.replay(case)
    assert calls, "mutant not applied: logreg does not call imputation.logreg.augment"
    assert np.abs(steps.fit.coef).max() > 10  # the separated fit diverges
    with pytest.raises(AssertionError, match="Not equal to tolerance"):
        test_l1_logreg.check_fit(case, steps)
    test_l1_logreg.check_draws(case, rng)  # the draws themselves are right


def test_augmentation_with_weight_one_is_caught(monkeypatch):
    calls = []
    original = imputation.logreg.augment

    def weight_one(*args, **kwargs):
        aug = original(*args, **kwargs)
        return aug._replace(weights=np.ones_like(aug.weights))

    monkeypatch.setattr(imputation.logreg, "augment", _counted(calls, weight_one))
    assert "Not equal to tolerance" in _augment_mismatch("levels2")
    case = LOGREG["logreg_seed1"]
    steps, _ = test_l1_logreg.replay(case)
    assert calls, "mutant not applied: logreg does not call imputation.logreg.augment"
    with pytest.raises(AssertionError, match="Not equal to tolerance"):
        test_l1_logreg.check_fit(case, steps)


def test_augmentation_without_clamping_is_caught(monkeypatch):
    """Visible where the clamp binds (the rare dummy), and only there."""
    calls = []
    monkeypatch.setattr(imputation.logreg, "_clamp", _counted(calls, lambda values, low, high: values))
    test_l1_augment.check_case(AUGMENT["levels2"])  # x1 and x2 stay inside their ranges
    y, aug = test_l1_augment.replay(AUGMENT["rare_dummy"])
    assert calls, "mutant not applied: augment does not call imputation.logreg._clamp"
    assert aug.x[:, 1].min() < 0  # below the dummy's minimum
    with pytest.raises(AssertionError, match="Not equal to tolerance"):
        test_l1_augment.check_augmented(AUGMENT["rare_dummy"], y, aug)


def test_augmentation_with_one_missing_value_is_caught(monkeypatch):
    calls = []
    monkeypatch.setattr(
        imputation.logreg, "_augmentation_skipped", _counted(calls, lambda n_predictors, n_not_observed: False)
    )
    assert "shapes (20,), (12,) mismatch" in _augment_mismatch("one_missing")
    case = LOGREG["logreg_one_missing"]
    steps, _ = test_l1_logreg.replay(case)
    assert calls, "mutant not applied: augment does not call imputation.logreg._augmentation_skipped"
    with pytest.raises(AssertionError, match="Not equal to tolerance"):
        test_l1_logreg.check_fit(case, steps)


def _logit_fit_mutant(monkeypatch, calls, change):
    """Make logreg's `logit_fit` return ``change(fit, x, y, weights)``."""
    original = imputation.logreg.logit_fit

    def mutant(x, y, weights=None, **kwargs):
        fit = original(x, y, weights=weights, **kwargs)
        w = np.ones(len(y)) if weights is None else np.asarray(weights)
        return change(fit, np.asarray(x), np.asarray(y, dtype=float), w)

    monkeypatch.setattr(imputation.logreg, "logit_fit", _counted(calls, mutant))


def _fitted_means(fit, x):
    return 1 / (1 + np.exp(-(x @ np.nan_to_num(fit.coef))))


def test_covariance_scaled_by_the_dispersion_is_caught(monkeypatch):
    calls, dispersions = [], []

    def scaled(fit, x, y, w):
        mu = _fitted_means(fit, x)
        dispersion = np.sum(w * (y - mu) ** 2 / (mu * (1 - mu))) / (len(y) - np.sum(~fit.aliased))
        dispersions.append(dispersion)
        return fit._replace(cov_unscaled=fit.cov_unscaled * dispersion)

    _logit_fit_mutant(monkeypatch, calls, scaled)
    case = LOGREG["logreg_seed1"]
    steps, rng = test_l1_logreg.replay(case)
    assert calls, "mutant not applied: logreg does not call imputation.logreg.logit_fit"
    assert abs(dispersions[0] - 1) > 0.05
    with pytest.raises(AssertionError, match="Not equal to tolerance"):
        test_l1_logreg.check_fit(case, steps)
    with pytest.raises(AssertionError, match="Not equal to tolerance"):
        test_l1_logreg.check_prediction(case, steps)
    test_l1_logreg.check_draws(case, rng)


def test_covariance_at_the_final_coefficients_is_caught(monkeypatch):
    calls = []

    def at_final(fit, x, y, w):
        kept = x[:, ~fit.aliased]
        mu = _fitted_means(fit, x)
        return fit._replace(cov_unscaled=np.linalg.inv(kept.T @ (kept * (w * mu * (1 - mu))[:, None])))

    _logit_fit_mutant(monkeypatch, calls, at_final)
    case = LOGREG["logreg_seed1"]
    steps, _ = test_l1_logreg.replay(case)
    assert calls, "mutant not applied: logreg does not call imputation.logreg.logit_fit"
    with pytest.raises(AssertionError, match="Not equal to tolerance"):
        test_l1_logreg.check_fit(case, steps)


def test_logreg_without_coefficient_draw_is_caught(monkeypatch):
    calls = []
    monkeypatch.setattr(
        imputation.logreg, "_coefficient_draw", _counted(calls, lambda fit, rng: np.nan_to_num(fit.coef))
    )
    case = LOGREG["logreg_seed1"]
    steps, rng = test_l1_logreg.replay(case)
    assert calls, "mutant not applied: logreg does not call imputation.logreg._coefficient_draw"
    test_l1_logreg.check_fit(case, steps)  # the fit itself is right
    with pytest.raises(AssertionError, match="Not equal to tolerance"):
        test_l1_logreg.check_prediction(case, steps)
    with pytest.raises(AssertionError, match=r"scripted values left over: 3 normal"):
        test_l1_logreg.check_draws(case, rng)


def test_logreg_boot_without_bootstrap_is_caught(monkeypatch):
    calls = []
    monkeypatch.setattr(imputation.logreg, "_bootstrap_sample", _counted(calls, lambda n, rng: np.arange(n)))
    case = LOGREG["logreg_boot_seed1"]
    steps, rng = test_l1_logreg.replay(case)
    assert calls, "mutant not applied: logreg_boot does not call imputation.logreg._bootstrap_sample"
    with pytest.raises(AssertionError, match="Not equal to tolerance"):
        test_l1_logreg.check_fit(case, steps)
    with pytest.raises(AssertionError, match=r"scripted values left over: 23 integer"):
        test_l1_logreg.check_draws(case, rng)


def test_reference_category_in_declared_order_is_caught(monkeypatch):
    """With the factor declared as (yes, no), "yes" becomes the first category:
    the fit models "no", its coefficients change sign and the labels swap."""
    calls = []
    original = imputation.logreg.target_categories

    def declared_order(y, keep_unused=False):
        if isinstance(getattr(y, "dtype", None), pd.CategoricalDtype) and not y.dtype.ordered:
            y = pd.Categorical(y, ordered=True)  # declared order, as for an ordered factor
        return original(y, keep_unused)

    monkeypatch.setattr(imputation.logreg, "target_categories", _counted(calls, declared_order))
    test_l1_logreg.check_case(LOGREG["logreg_seed1"])  # declared (no, yes): sorted already
    case = LOGREG["logreg_declared_yes_no"]
    steps, _ = test_l1_logreg.replay(case)
    assert calls, "mutant not applied: logreg does not call imputation.logreg.target_categories"
    np.testing.assert_allclose(steps.fit.coef, -np.asarray(case["coef"]), rtol=1e-8)
    with pytest.raises(AssertionError, match="Not equal to tolerance"):
        test_l1_logreg.check_fit(case, steps)
