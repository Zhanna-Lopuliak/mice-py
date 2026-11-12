"""L1 parity: imputation of a categorical target by multinomial logistic regression (polyreg).

Reference: tests/reference/r/l1_polyreg.R (mice.impute.polyreg with its
uniform numbers captured, the nnet::multinom fit and the probabilities of
the imputed cells; direct nnet::multinom fits).

Each case is replayed with a strict ScriptedGenerator that serves R's
uniform numbers and refuses any other draw, through the step-by-step version
of the imputer (`imputation.polyreg._polyreg`), which also returns the fit
and the probabilities. The checks are split so that test_l1_mutants can run
them against a defective library: `replay` runs the port, `check_labels`
compares the imputations with R's, `check_fit` the objective, convergence
and coefficients of the fit, `check_prediction` the probabilities,
`check_draws` the draws the port asked for.

Tolerances. The fixtures come from R on macOS; the minimiser takes up to 100
steps, along which rounding differences between R's C code and NumPy (and
between platforms) can grow. The probabilities are compared within 1e-6,
the objective within 1e-8 (relative), and the coefficients within 1e-6 only
where the fit converged and the predictors are well scaled. The uniform
numbers of every case are at least 1e-4 away from every cumulative
probability (checked in R), so the labels must be identical.
"""

import numpy as np
import pandas as pd
import pytest

import imputation.polyreg
from imputation.fitting import multinom_fit
from imputation.logreg import target_categories
from imputation.polyreg import polyreg

from ._helpers import ScriptedGenerator, as_array, as_bool, load_fixture, str_list

pytestmark = pytest.mark.parity

PROB_ATOL = 1e-6
VALUE_RTOL = 1e-8
COEF_ATOL = 1e-6

FIXTURE = load_fixture("l1_polyreg")
CASES = {case["name"]: case for case in FIXTURE["cases"]}
MULTINOM = {case["name"]: case for case in FIXTURE["multinom_cases"]}
# predictors on scales 1000, 1 and 0.001: the coefficients are not compared
BADLY_SCALED = {"polyreg_five_levels"}


def _listed(values):
    return values if isinstance(values, list) else [values]


def target(inp):
    """The target as MICE passes it: a Categorical with R's declared levels."""
    values = _listed(inp["y"])
    return pd.Categorical(values, categories=str_list(inp["levels"]), ordered=inp["kind"] == "ordered")


def _inputs(case):
    """(y, ry, wy, x) of a case; wy is None where R used ``!ry``."""
    inp = FIXTURE["inputs"][case["input"]]
    y = target(inp)
    ry = as_bool(inp["ry"])
    wy = None if inp["wy"] is None else as_bool(inp["wy"])
    x = np.empty((len(ry), 0)) if inp["p"] == 0 else as_array(inp["x"]).reshape(len(ry), inp["p"])
    return y, ry, wy, x


def _scripted(case):
    """A strict generator that serves R's uniform numbers and nothing else."""
    return ScriptedGenerator(uniforms=as_array(case["uniforms"]), strict=True)


def replay(case):
    """Run the port on the inputs of `case` with R's draws; return (steps, rng).

    The step function is looked up when called, so that a patched module
    attribute takes effect.
    """
    y, ry, wy, x = _inputs(case)
    rng = _scripted(case)
    steps = imputation.polyreg._polyreg(y, ry, x, wy, case["maxit"], case["max_weights"], rng)
    return steps, rng


def check_labels(case, steps):
    """The port's labels are R's."""
    got = steps.labels.tolist()
    assert got == _listed(case["imputed"])
    assert all(isinstance(v, str) for v in got)


def _compare_coef(case):
    return case["convergence"] == 0 and case["name"] not in BADLY_SCALED


def check_fit(case, steps):
    """nnet::multinom: the objective within 1e-8 (relative), the same convergence
    code and fitted classes; the coefficients within 1e-6 where the fit
    converged and is well scaled."""
    fit = steps.fit
    np.testing.assert_allclose(fit.value, case["value"], rtol=VALUE_RTOL, atol=0)
    assert fit.converged == (case["convergence"] == 0)
    levels = _listed(case["levels"])
    assert [levels[k] for k in fit.classes] == _listed(case["fitted_levels"])
    expected = as_array(case["coef"]).reshape(len(fit.classes) - 1, -1)
    assert fit.coef.shape == expected.shape
    if _compare_coef(case):
        np.testing.assert_allclose(fit.coef, expected, rtol=0, atol=COEF_ATOL)


def check_prediction(case, steps):
    """The probabilities of the port's levels in every imputed cell, within 1e-6."""
    expected = as_array(case["prob"]).reshape(case["n_imp"], len(_listed(case["levels"])))
    np.testing.assert_allclose(steps.prob, expected, rtol=0, atol=PROB_ATOL)


def check_draws(case, rng):
    """The port drew what R drew: one uniform number per imputed cell, all of them."""
    rng.assert_exhausted()
    assert rng.order == [tuple(pair) for pair in case["order"]]


def check_case(case):
    """Every L1 check of one case."""
    steps, rng = replay(case)
    check_labels(case, steps)
    if case["value"] is not None:
        check_fit(case, steps)
        check_prediction(case, steps)
    else:
        assert steps.fit is None and steps.prob is None
    check_draws(case, rng)


def _names(*ports):
    return [name for name, case in CASES.items() if case["port"] in ports]


IMPUTED = _names("match", "rebuilt")
FITTED = [name for name in IMPUTED if CASES[name]["value"] is not None]


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
def test_probabilities_match_r(name):
    case = CASES[name]
    steps, _ = replay(case)
    check_prediction(case, steps)


@pytest.mark.parametrize("name", IMPUTED)
def test_draws_match_r(name):
    case = CASES[name]
    _, rng = replay(case)
    check_draws(case, rng)


@pytest.mark.parametrize("name", IMPUTED)
def test_public_imputer_returns_the_labels(name):
    """`polyreg` returns the labels of the step function."""
    case = CASES[name]
    y, ry, wy, x = _inputs(case)
    imputed = polyreg(y, ry, x, id_mis=wy, maxit=case["maxit"], max_weights=case["max_weights"], rng=_scripted(case))
    assert imputed.tolist() == _listed(case["imputed"])


@pytest.mark.parametrize("name", list(CASES))
def test_level_order_is_the_ports(name):
    """The fixture's level order is that of `target_categories`."""
    case = CASES[name]
    y, *_ = _inputs(case)
    assert target_categories(y).levels == _listed(case["levels"])


def test_fixture_covers_the_planned_cases():
    inputs = {c["input"] for c in CASES.values() if c["port"] == "match"}
    assert inputs >= {
        "base",
        "two_levels",
        "unsorted",
        "numeric_labels",
        "unused_level",
        "ordered",
        "one_category",
        "no_predictor",
        "one_missing",
        "wy_subset",
        "separated",
        "five_levels",
    }
    assert sum(c["input"] == "base" and c["port"] == "match" and c["maxit"] == 100 for c in CASES.values()) == 2
    assert CASES["polyreg_maxit"]["maxit"] == 5
    assert FIXTURE["nnet"] == "7.3.21"


def test_the_iteration_limit_stops_the_fit():
    """With nnet.maxit = 5 R's fit stops at the limit (convergence 1), without
    an error; the port stops at the same point."""
    case = CASES["polyreg_maxit"]
    assert case["convergence"] == 1
    steps, _ = replay(case)
    assert not steps.fit.converged and steps.fit.gr_count == 5
    check_case(case)


def test_one_observed_category_draws_the_uniforms_but_fits_nothing():
    """R draws runif(n_imp) before it looks at the categories."""
    case = CASES["polyreg_one_category"]
    steps, rng = replay(case)
    assert steps.fit is None and set(steps.labels) == {"b"}
    assert rng.order == [("uniform", 10)]
    check_case(case)


def test_unused_declared_level_is_dropped():
    """augment() rebuilds an unordered factor without its unused levels."""
    case = CASES["polyreg_unused_level"]
    steps, _ = replay(case)
    assert steps.prob.shape == (10, 3)
    check_case(case)


def test_without_predictors_the_model_matrix_has_two_constant_columns():
    """mice adds a column ``int = 1`` next to the intercept; both get the same coefficients."""
    case = CASES["polyreg_no_predictor"]
    steps, _ = replay(case)
    assert steps.fit.coef.shape == (2, 2)
    np.testing.assert_array_equal(steps.fit.coef[:, 0], steps.fit.coef[:, 1])


def test_r_failures_are_recorded():
    """The cases where the port goes on and R does not (see l1_polyreg.R)."""
    errors = {name: CASES[name]["r_error"] for name in _names("rebuilt")}
    assert errors == {
        "polyreg_two_levels_one_missing": "dim(X) must have a positive length",
        "polyreg_ordered_unused": "dims [product 30] do not match the length of object [40]",
        "polyreg_imputed_only_level": "dims [product 30] do not match the length of object [40]",
    }


def test_level_without_observed_row_gets_probability_zero():
    """Without augmentation, "d" is held only by the cells to impute: R's
    multinom drops it and polyreg stops; the port fits a, b and c and never
    imputes d."""
    case = CASES["polyreg_imputed_only_level"]
    steps, _ = replay(case)
    assert steps.fit.classes.tolist() == [0, 1, 2] and steps.fit.n_classes == 4
    assert np.all(steps.prob[:, 3] == 0)
    assert "d" not in steps.labels
    check_case(case)


@pytest.mark.parametrize(
    "name, message",
    [
        ("polyreg_too_many_weights", r"too many \(12\) weights"),
        ("polyreg_many_categories", r"Maximum number of categories \(50\) exceeded"),
    ],
)
def test_r_errors_are_value_errors(name, message):
    """R stops, and so does the port, with R's message in its own; it draws
    what R drew before (the uniforms, before the fit; nothing before augment)."""
    case = CASES[name]
    assert case["port"] == "value_error"
    assert case["r_error"].replace("(", r"\(").replace(")", r"\)") == message
    with pytest.raises(ValueError, match=message):
        replay(case)
    y, ry, wy, x = _inputs(case)
    rng = _scripted(case)
    with pytest.raises(ValueError, match=message):
        imputation.polyreg._polyreg(y, ry, x, wy, case["maxit"], case["max_weights"], rng)
    check_draws(case, rng)


# ---------------------------------------------------------------------------
# Direct nnet::multinom fits
# ---------------------------------------------------------------------------


def fit_multinom(case):
    x = as_array(case["x"]).reshape(len(_listed(case["y"])), -1)
    y = as_array(case["y"], dtype=int)
    return multinom_fit(x, y, weights=as_array(case["weights"]), n_classes=case["n_classes"])


def check_multinom(case, fit):
    """Objective, convergence, coefficients and the probabilities of new rows."""
    np.testing.assert_allclose(fit.value, case["value"], rtol=VALUE_RTOL, atol=0)
    assert fit.converged == (case["convergence"] == 0)
    np.testing.assert_allclose(fit.coef, as_array(case["coef"]).reshape(fit.coef.shape), rtol=0, atol=COEF_ATOL)
    new_x = as_array(case["new_x"]).reshape(-1, fit.coef.shape[1])
    prob = as_array(case["prob"]).reshape(new_x.shape[0], case["n_classes"])
    np.testing.assert_allclose(fit.predict(new_x), prob, rtol=0, atol=PROB_ATOL)


@pytest.mark.parametrize("name", list(MULTINOM))
def test_multinom_fit_matches_r(name):
    case = MULTINOM[name]
    check_multinom(case, fit_multinom(case))


def test_logistic_unit_is_cut_off_beyond_15():
    """Two classes: nnet's logistic unit returns exactly 0 and 1 far from the boundary."""
    case = MULTINOM["two_classes_separated"]
    fit = fit_multinom(case)
    assert fit.logistic
    prob = fit.predict(as_array(case["new_x"]).reshape(-1, 3))
    assert prob[0, 1] == 0.0 and prob[-1, 1] == 1.0
