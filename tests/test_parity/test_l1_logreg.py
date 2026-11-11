"""L1 parity: imputation of a two-category target by logistic regression (logreg, logreg.boot).

Reference: tests/reference/r/l1_logreg.R (mice.impute.logreg and
mice.impute.logreg.boot with every random number captured, the glm.fit,
the coefficient draw and the probabilities).

Each case is replayed with a strict ScriptedGenerator that serves R's draws
and refuses any other, through the step-by-step versions of the imputers
(`imputation.logreg._logreg` and `_logreg_boot`), which also return the fit,
the coefficients used for the prediction and the probabilities. The checks
are split so that test_l1_mutants can run them against a defective library:
`replay` runs the port, `check_labels` compares the imputations with R's,
`check_fit` the glm.fit, `check_prediction` the drawn coefficients and the
probabilities, `check_draws` the draws the port asked for (all used, in R's
order, bounds of the bootstrap).
"""

import numpy as np
import pandas as pd
import pytest

import imputation.logreg
from imputation.logreg import logreg, logreg_boot, target_categories

from ._helpers import ScriptedGenerator, as_array, as_bool, load_fixture, str_list

pytestmark = pytest.mark.parity

RTOL = 1e-8
ATOL = 1e-10
# A fit that ran into the iteration limit (a separated bootstrap sample) is
# badly conditioned: its working weights are near 0 and the entries of
# cov.unscaled are about 1e10. Its coefficients agree with R's to fewer
# digits, and how many depends on the linear algebra library, so they get a
# looser tolerance and its cov.unscaled, which logreg.boot never uses, is not
# compared. The probabilities (all 0 or 1) are compared as everywhere.
RTOL_NOT_CONVERGED = 1e-6

FIXTURE = load_fixture("l1_logreg")
CASES = {case["name"]: case for case in FIXTURE["cases"]}
STEPS = {"logreg": "_logreg", "logreg.boot": "_logreg_boot"}


def _listed(values):
    return values if isinstance(values, list) else [values]


def target(inp):
    """The target as MICE passes it: a Categorical, floats, or an object array of booleans with None."""
    values = _listed(inp["y"])
    if inp["kind"] in ("factor", "ordered"):
        return pd.Categorical(values, categories=str_list(inp["levels"]), ordered=inp["kind"] == "ordered")
    if inp["kind"] == "logical":
        return np.array(values, dtype=object)
    return as_array(values)


def _inputs(case):
    """(y, ry, wy, x) of a case; wy is None where R used ``!ry``."""
    inp = FIXTURE["inputs"][case["input"]]
    y = target(inp)
    ry = as_bool(inp["ry"])
    wy = None if inp["wy"] is None else as_bool(inp["wy"])
    x = np.empty((len(ry), 0)) if inp["p"] == 0 else as_array(inp["x"]).reshape(len(ry), inp["p"])
    return y, ry, wy, x


def _scripted(case):
    """A strict generator with R's draws: normals for logreg, integers for logreg.boot, uniforms for both."""
    integers = None if case["integers"] is None else as_array(case["integers"], dtype=int)
    return ScriptedGenerator(
        normals=as_array(case["normals"]), uniforms=as_array(case["uniforms"]), integers=integers, strict=True
    )


def replay(case):
    """Run the port on the inputs of `case` with R's draws; return (steps, rng).

    The step function is looked up when called, so that a patched module
    attribute takes effect.
    """
    y, ry, wy, x = _inputs(case)
    rng = _scripted(case)
    steps = getattr(imputation.logreg, STEPS[case["method"]])(y, ry, x, wy, rng)
    return steps, rng


def _kind(case):
    return FIXTURE["inputs"][case["input"]]["kind"]


def check_labels(case, steps):
    """The port's labels are R's (the port's form of them: booleans for a logical target)."""
    expected = _listed(case["imputed"])
    got = steps.labels.tolist()
    assert got == expected
    kind = _kind(case)
    if kind in ("factor", "ordered"):
        assert all(isinstance(v, str) for v in got)
    elif kind == "logical":
        assert all(isinstance(v, bool) for v in got)
    else:
        assert steps.labels.dtype == np.float64


def _coef_rtol(case):
    return RTOL if case["converged"] else RTOL_NOT_CONVERGED


def check_fit(case, steps):
    """glm.fit: coefficients (NaN where aliased) and cov.unscaled within tolerance, same iterations and convergence.

    For a fit that did not converge see `RTOL_NOT_CONVERGED`.
    """
    fit = steps.fit
    q = case["n_coef"]
    np.testing.assert_allclose(fit.coef, as_array(case["coef"]), rtol=_coef_rtol(case), atol=ATOL)
    rank = int(np.sum(~np.isnan(as_array(case["coef"]))))
    assert fit.cov_unscaled.shape == (rank, rank)
    if case["converged"]:
        np.testing.assert_allclose(
            fit.cov_unscaled, as_array(case["cov_unscaled"]).reshape(rank, rank), rtol=RTOL, atol=ATOL
        )
    assert fit.coef.shape == (q,)
    assert (fit.iterations, fit.converged) == (case["iter"], case["converged"])


def check_prediction(case, steps):
    """The coefficients of the prediction (beta.star) and the probabilities within tolerance."""
    np.testing.assert_allclose(steps.beta, as_array(case["beta_star"]), rtol=_coef_rtol(case), atol=ATOL)
    np.testing.assert_allclose(steps.prob, as_array(case["prob"]), rtol=RTOL, atol=ATOL)


def check_draws(case, rng):
    """The port drew what R drew: every value, in R's order, with R's bootstrap bounds."""
    rng.assert_exhausted()
    assert rng.order == [tuple(pair) for pair in case["order"]]
    bounds = [call.args for call in rng.calls if call.method == "integers"]
    expected = {"low": 0, "high": case["n_obs"], "endpoint": False}
    assert bounds == ([] if case["integers"] is None else [expected])


def check_case(case):
    """Every L1 check of one case."""
    steps, rng = replay(case)
    check_labels(case, steps)
    if case["port"] in ("match", "rebuilt"):
        check_fit(case, steps)
        check_prediction(case, steps)
    else:
        assert steps.fit is None and steps.beta is None and steps.prob is None
    check_draws(case, rng)


def _names(*ports):
    return [name for name, case in CASES.items() if case["port"] in ports]


FITTED = _names("match", "rebuilt")
IMPUTED = _names("match", "rebuilt", "one_category")


@pytest.mark.parametrize("name", IMPUTED)
def test_imputations_match_r(name):
    case = CASES[name]
    steps, _ = replay(case)
    check_labels(case, steps)


@pytest.mark.parametrize("name", FITTED)
def test_fit_matches_r(name):
    case = CASES[name]
    steps, _ = replay(case)
    check_fit(case, steps)


@pytest.mark.parametrize("name", FITTED)
def test_coefficients_and_probabilities_match_r(name):
    case = CASES[name]
    steps, _ = replay(case)
    check_prediction(case, steps)


@pytest.mark.parametrize("name", IMPUTED)
def test_draws_match_r(name):
    case = CASES[name]
    _, rng = replay(case)
    check_draws(case, rng)


@pytest.mark.parametrize("name", IMPUTED)
def test_public_imputers_return_the_labels(name):
    """`logreg` and `logreg_boot` return the labels of the step functions."""
    case = CASES[name]
    y, ry, wy, x = _inputs(case)
    imputer = logreg if case["method"] == "logreg" else logreg_boot
    imputed = imputer(y, ry, x, id_mis=wy, rng=_scripted(case))
    assert imputed.tolist() == _listed(case["imputed"])


@pytest.mark.parametrize("name", list(CASES))
def test_level_order_is_the_ports(name):
    """The fixture's level order is that of `target_categories`."""
    case = CASES[name]
    y, *_ = _inputs(case)
    assert target_categories(y).levels == _listed(case["levels"])


def test_fixture_covers_the_planned_cases():
    logreg_inputs = {c["input"] for c in CASES.values() if c["method"] == "logreg" and c["port"] == "match"}
    assert logreg_inputs >= {
        "base",
        "filled_target",
        "numeric01",
        "logical",
        "declared_yes_no",
        "ordered",
        "one_missing",
        "no_predictor",
        "wy_subset",
        "separated",
    }
    assert sum(c["input"] == "base" and c["method"] == "logreg" for c in CASES.values()) == 2
    boot_inputs = {c["input"] for c in CASES.values() if c["method"] == "logreg.boot"}
    assert boot_inputs >= {"base", "one_predictor", "no_predictor", "one_missing", "near_separated"}
    assert sum(c["input"] == "base" and c["method"] == "logreg.boot" for c in CASES.values()) == 2


def test_values_at_missing_cells_are_ignored():
    """Inside MICE the missing cells hold the current imputations; with both
    categories observed they change nothing."""
    filled, _ = replay(CASES["logreg_filled_target"])
    plain, _ = replay(CASES["logreg_seed1"])
    np.testing.assert_array_equal(filled.labels, plain.labels)
    np.testing.assert_array_equal(filled.beta, plain.beta)


def test_augmentation_keeps_a_separated_fit_finite():
    """The observed rows are separated by x1; with the added rows the fit
    converges to moderate coefficients (R checks that the plain fit does not)."""
    case = CASES["logreg_separated"]
    steps, _ = replay(case)
    assert steps.fit.converged
    assert np.abs(steps.fit.coef).max() < 10


def test_separated_bootstrap_sample_runs_into_the_iteration_limit():
    """R does not stop either: 25 iterations, not converged, large coefficients."""
    case = CASES["logreg_boot_separated"]
    steps, _ = replay(case)
    assert (steps.fit.iterations, steps.fit.converged) == (25, False)
    assert np.abs(steps.fit.coef).max() > 10
    check_case(case)


def test_more_than_two_categories_raise_before_drawing():
    """R draws everything and then stops ("invalid 'labels'"); the port
    refuses at once and names polyreg."""
    case = CASES["logreg_three_levels"]
    assert case["r_error"] == "invalid 'labels'; length 3 should be 1 or 2"
    y, ry, wy, x = _inputs(case)
    for imputer in (logreg, logreg_boot):
        rng = ScriptedGenerator(strict=True)
        with pytest.raises(ValueError, match=r"two categories, but y has 3 \('a', 'b', 'c'\)\. Use polyreg"):
            imputer(y, ry, x, id_mis=wy, rng=rng)
        assert rng.calls == []


@pytest.mark.parametrize("name", _names("one_category"))
def test_one_category_is_imputed_without_fit_or_draws(name):
    """R fits anyway and, for logreg, returns invented labels such as "no1"."""
    case = CASES[name]
    assert case["r_imputed"] is not None
    check_case(case)
    assert set(_listed(case["imputed"])) == {"no"}


def test_r_failures_are_recorded():
    """The cases where the port goes on and R does not (see l1_logreg.R)."""
    errors = {name: CASES[name]["r_error"] for name in _names("rebuilt") if CASES[name]["r_error"]}
    assert errors == {
        "logreg_numeric_12": "y values must be 0 <= y <= 1",
        "logreg_constant_column": "dims [product 3] do not match the length of object [4]",
        "logreg_ordered_unused": "invalid 'labels'; length 3 should be 1 or 2",
        "logreg_boot_no_predictor": "non-conformable arguments",
    }
    assert all(v is None for v in CASES["logreg_boot_duplicate_column"]["r_imputed"])


@pytest.mark.parametrize("name", ["logreg_constant_column", "logreg_boot_duplicate_column"])
def test_aliased_column_gets_coefficient_zero(name):
    """R stops (logreg) or imputes NA (logreg.boot); the port predicts with 0
    for the aliased coefficient and draws only the others."""
    case = CASES[name]
    steps, _ = replay(case)
    assert steps.fit.aliased.tolist() == [False, False, False, True]
    assert np.isnan(steps.fit.coef[3]) and steps.beta[3] == 0.0
    check_case(case)
