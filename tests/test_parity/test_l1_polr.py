"""L1 parity: imputation of an ordered categorical target by the proportional-odds model (polr).

Reference: tests/reference/r/l1_polr.R (mice.impute.polr with its uniform
numbers captured, the class of the fit R made and polr's error, the fit and
the probabilities of the imputed cells; direct MASS::polr fits).

Each case is replayed with a strict ScriptedGenerator that serves R's
uniform numbers and refuses any other draw, through the step-by-step version
of the imputer (`imputation.polr._polr`), which also returns the fit, the
probabilities and the condition that made polr fall back to multinom. The
checks are split so that test_l1_mutants can run them against a defective
library: `replay` runs the port, `check_labels` compares the imputations
with R's, `check_fit` which fit was made and its deviance (objective),
convergence, coefficients and cut points, `check_prediction` the
probabilities, `check_draws` the draws the port asked for.

Tolerances. The fixtures come from R on macOS; the minimiser takes up to 100
steps, along which rounding differences between R's C code and NumPy (and
between platforms) can grow. The probabilities are compared within 1e-6,
the deviance (the objective of a multinom fit) within 1e-8 (relative), and
the coefficients and cut points within 1e-6 only where the fit converged
and is well conditioned. The evaluation counts are not compared: the
function count of a search that ends on a restart depends on the last bits
of the values (with every input perturbed by 1e-15, relative, it moves by
up to 7 here; the gradient count stays R's). The uniform numbers of every
case are at least 1e-4 away from every cumulative probability (checked in
R), so the labels must be identical.
"""

import numpy as np
import pandas as pd
import pytest

import imputation.polr
from imputation.fitting import MultinomFit, PolrError, PolrFit, polr_fit
from imputation.logreg import target_categories
from imputation.polr import FALLBACK_EVENT, polr

from ._helpers import ScriptedGenerator, as_array, as_bool, load_fixture, str_list

pytestmark = pytest.mark.parity

PROB_ATOL = 1e-6
VALUE_RTOL = 1e-8
COEF_ATOL = 1e-6

FIXTURE = load_fixture("l1_polr")
CASES = {case["name"]: case for case in FIXTURE["cases"]}
DIRECT = {case["name"]: case for case in FIXTURE["direct_cases"]}
# the fallback on observed rows whose middle split is separated: its
# coefficients drift along the separation until the minimiser stops, so
# they are not compared (the probabilities are)
SEPARATED = {"polr_separated", "polr_separated_unused"}


def _listed(values):
    return values if isinstance(values, list) else [values]


def target(inp):
    """The target as MICE passes it: a Categorical with R's declared levels,
    an object array for text and booleans, floats for numbers."""
    values = _listed(inp["y"])
    if inp["kind"] in ("ordered", "factor"):
        return pd.Categorical(values, categories=str_list(inp["levels"]), ordered=inp["kind"] == "ordered")
    if inp["kind"] in ("character", "logical"):
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
    """A strict generator that serves R's uniform numbers and nothing else."""
    return ScriptedGenerator(uniforms=as_array(case["uniforms"]), strict=True)


def replay(case, log_fallback=False, log_event=None):
    """Run the port on the inputs of `case` with R's draws; return (steps, rng).

    The step function is looked up when called, so that a patched module
    attribute takes effect.
    """
    y, ry, wy, x = _inputs(case)
    rng = _scripted(case)
    steps = imputation.polr._polr(y, ry, x, wy, case["maxit"], case["max_weights"], log_fallback, rng, log_event)
    return steps, rng


def r_text(label):
    """A label as R prints it: TRUE / FALSE for booleans, numbers without a trailing ``.0``."""
    if isinstance(label, (bool, np.bool_)):
        return "TRUE" if label else "FALSE"
    if isinstance(label, (float, np.floating)):
        return format(float(label), "g")
    return label


def check_labels(case, steps):
    """The port's labels are R's, of the type of the target's values."""
    got = steps.labels.tolist()
    assert [r_text(v) for v in got] == _listed(case["imputed"])
    kind = FIXTURE["inputs"][case["input"]]["kind"]
    expected_type = {"numeric": float, "logical": bool}.get(kind, str)
    assert all(isinstance(v, expected_type) for v in got)


def fit_kind(steps):
    """The fit the port made: "polr", "multinom" or "none"."""
    if steps.fit is None:
        return "none"
    return "polr" if isinstance(steps.fit, PolrFit) else "multinom"


def _compare_coef(case):
    return case["convergence"] == 0 and case["name"] not in SEPARATED


def check_fit(case, steps):
    """The fit the port made is the fit R made (or, where R does something
    else, the rebuilt one), for the same reason; its deviance (objective)
    within 1e-8 (relative), the same convergence code; the coefficients and
    cut points within 1e-6 where the fit converged and is well conditioned."""
    assert fit_kind(steps) == case["port_fit"]
    assert steps.fallback == case["port_fallback"]
    fit = steps.fit
    assert fit.converged == (case["convergence"] == 0)
    if isinstance(fit, PolrFit):
        np.testing.assert_allclose(fit.deviance, case["deviance"], rtol=VALUE_RTOL, atol=0)
        expected_zeta = as_array(case["zeta"])
        expected_coef = as_array(case["coef"])
        assert fit.zeta.shape == expected_zeta.shape and fit.coef.shape == expected_coef.shape
        if _compare_coef(case):
            np.testing.assert_allclose(fit.zeta, expected_zeta, rtol=0, atol=COEF_ATOL)
            np.testing.assert_allclose(fit.coef, expected_coef, rtol=0, atol=COEF_ATOL)
        return
    np.testing.assert_allclose(fit.value, case["value"], rtol=VALUE_RTOL, atol=0)
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
    if case["port_fit"] != "none":
        check_fit(case, steps)
        check_prediction(case, steps)
    else:
        assert steps.fit is None and steps.prob is None and steps.fallback is None
    check_draws(case, rng)


def _names(*ports):
    return [name for name, case in CASES.items() if case["port"] in ports]


IMPUTED = _names("match", "rebuilt")
FITTED = [name for name in IMPUTED if CASES[name]["port_fit"] != "none"]


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


@pytest.mark.parametrize("name", _names("match"))
def test_the_fit_is_the_one_r_made(name):
    """Where the port gives R's result, it made the fit R made, for the same reason."""
    case = CASES[name]
    assert (case["port_fit"], case["port_fallback"]) == (case["fit"], case["polr_error"])
    assert _listed(case["imputed"]) == _listed(case["r_imputed"])


@pytest.mark.parametrize("name", IMPUTED)
def test_public_imputer_returns_the_labels(name):
    """`polr` returns the labels of the step function."""
    case = CASES[name]
    y, ry, wy, x = _inputs(case)
    imputed = polr(y, ry, x, id_mis=wy, maxit=case["maxit"], max_weights=case["max_weights"], rng=_scripted(case))
    assert [r_text(v) for v in imputed.tolist()] == _listed(case["imputed"])


@pytest.mark.parametrize("name", [name for name, case in CASES.items() if case["input"] != "many_categories"])
def test_level_order_is_the_ports(name):
    """The fixture's level order is that of `target_categories` with the unused levels kept."""
    case = CASES[name]
    y, *_ = _inputs(case)
    assert [r_text(level) for level in target_categories(y, keep_unused=True).levels] == _listed(case["levels"])


def test_fixture_covers_the_planned_cases():
    inputs = {c["input"] for c in CASES.values() if c["port"] == "match"}
    assert inputs >= {
        "base",
        "declared_order",
        "unordered",
        "unused_middle",
        "unused_end",
        "one_missing",
        "wy_subset",
        "two_levels",
        "numeric",
        "logical",
        "separated",
    }
    assert sum(c["input"] == "base" and c["port"] == "match" for c in CASES.values()) == 2
    fallbacks = {c["polr_error"] for c in CASES.values() if c["port"] == "match" and c["fit"] == "multinom"}
    assert fallbacks == {PolrError.FEW_LEVELS, PolrError.NOT_A_FACTOR, PolrError.START_NOT_CONVERGED}
    assert CASES["polr_fallback_maxit"]["maxit"] == 5 and CASES["polr_fallback_maxit"]["convergence"] == 1
    assert {c["name"] for c in DIRECT.values() if "r_error" in c} == {
        "two_levels",
        "start_not_converged",
        "start_not_finite",
    }
    assert (FIXTURE["mass"], FIXTURE["nnet"]) == ("7.3.66", "7.3.21")


def test_declared_order_is_the_model_order():
    """The same codes with the labels (low, mid, high, top), which sort
    otherwise, and as an unordered factor declared (c, a, d, b), which sorts
    to (a, b, c, d): the same fit and probabilities as the base case (up to
    rounding: R adds the rows of the augmentation in the declared order)."""
    base = CASES["polr_seed1"]
    for name in ("polr_declared_order", "polr_unordered"):
        case = CASES[name]
        np.testing.assert_allclose(as_array(case["prob"]), as_array(base["prob"]), rtol=0, atol=1e-12)
        steps, _ = replay(case)
        np.testing.assert_allclose(steps.fit.zeta, as_array(base["zeta"]), rtol=0, atol=COEF_ATOL)
    assert _listed(CASES["polr_declared_order"]["levels"]) == ["low", "mid", "high", "top"]


def test_unused_declared_level_can_be_imputed():
    """MASS::polr counts every declared level: one without rows gets a small
    positive probability between its neighbours, so it can be imputed."""
    for name, column in (("polr_unused_middle", 1), ("polr_unused_end", 3)):
        case = CASES[name]
        steps, _ = replay(case)
        assert steps.prob.shape == (10, 4) and steps.fit.zeta.shape == (3,)
        assert np.all(steps.prob[:, column] > 0)
        check_case(case)


def test_the_fallback_gives_an_unused_level_probability_zero():
    """R's multinom drops the unused level "b" and mice then stops; the port
    gives it probability 0."""
    case = CASES["polr_separated_unused"]
    steps, _ = replay(case)
    assert isinstance(steps.fit, MultinomFit) and steps.fit.classes.tolist() == [0, 2, 3]
    assert np.all(steps.prob[:, 1] == 0)
    check_case(case)


def test_aliased_column_gets_coefficient_zero():
    """R's polr drops the constant column (aliased with the intercept in the
    start fit) and its predict then stops; the port gives it the coefficient 0."""
    case = CASES["polr_constant_column"]
    steps, _ = replay(case)
    assert steps.fit.aliased.tolist() == [False, False, True] and steps.fit.coef[2] == 0
    check_case(case)


def test_one_category_is_imputed_without_a_fit():
    """R fits a degenerate model to an ordered target (whose start fit
    converges) and stops on an unordered one; the port imputes the category
    after drawing the uniform numbers, where R draws them."""
    for name in ("polr_one_category", "polr_one_category_unordered"):
        case = CASES[name]
        steps, rng = replay(case)
        assert steps.fit is None and set(steps.labels) == {"b"}
        assert rng.order == [("uniform", 10)]
    assert _listed(CASES["polr_one_category"]["r_imputed"]) == ["b"] * 10


def test_r_failures_are_recorded():
    """The cases where the port goes on and R stops (see l1_polr.R)."""
    errors = {name: CASES[name]["r_error"] for name in _names("rebuilt")}
    assert errors == {
        "polr_text": None,
        "polr_one_category": None,
        "polr_one_category_unordered": "need two or more classes to fit a multinom model",
        "polr_two_levels_one_missing": "dim(X) must have a positive length",
        "polr_separated_unused": "dims [product 3] do not match the length of object [4]",
        "polr_constant_column": "non-conformable arguments",
        "polr_no_predictor": "need two or more classes to fit a multinom model",
        "polr_no_predictor_two_levels": "need two or more classes to fit a multinom model",
    }
    # R's polr refuses a character target; the port takes text as an unordered factor
    text = CASES["polr_text"]
    assert (text["fit"], text["polr_error"], text["port_fit"]) == ("multinom", PolrError.NOT_A_FACTOR, "polr")


@pytest.mark.parametrize(
    "name, message",
    [
        ("polr_too_many_weights", r"too many \(16\) weights"),
        ("polr_many_categories", r"Maximum number of categories \(50\) exceeded"),
    ],
)
def test_r_errors_are_value_errors(name, message):
    """R stops, and so does the port, before any draw (R draws after the fit)."""
    case = CASES[name]
    assert case["port"] == "value_error" and case["order"] == []
    y, ry, wy, x = _inputs(case)
    rng = _scripted(case)
    with pytest.raises(ValueError, match=message):
        imputation.polr._polr(y, ry, x, wy, case["maxit"], case["max_weights"], False, rng, None)
    check_draws(case, rng)


def test_without_predictors_the_port_fits_where_r_stops():
    """R's ``cbind.data.frame`` drops the zero-column x, so polr sees no
    response and multinom no classes, and mice stops before its draws. The
    port fits polr without predictors (the cut points alone; the same
    probabilities in every cell), and where polr does not apply (two levels)
    multinom on the intercept alone."""
    for name in ("polr_no_predictor", "polr_no_predictor_two_levels"):
        case = CASES[name]
        assert (case["fit"], case["polr_error"]) == (None, PolrError.NOT_A_FACTOR)
        assert not case["r_draws"]
        steps, _ = replay(case)
        assert np.all(steps.prob == steps.prob[0])
        check_case(case)
    steps, _ = replay(CASES["polr_no_predictor"])
    assert isinstance(steps.fit, PolrFit) and steps.fit.coef.shape == (0,) and steps.fit.zeta.shape == (3,)
    steps, _ = replay(CASES["polr_no_predictor_two_levels"])
    assert isinstance(steps.fit, MultinomFit) and steps.fit.coef.shape == (1, 1)
    assert steps.fallback == PolrError.FEW_LEVELS


@pytest.mark.parametrize("name", ["polr_numeric", "polr_two_levels", "polr_separated", "polr_no_predictor_two_levels"])
def test_fallback_logs_rs_event(name):
    """With log_fallback, a fallback logs "polr falls back to multinom", as
    R's polr.to.loggedEvents = TRUE; without it, or without a fallback,
    nothing is logged."""
    case = CASES[name]
    events = []
    replay(case, log_fallback=True, log_event=events.append)
    assert events == [FALLBACK_EVENT] == ["polr falls back to multinom"]
    events.clear()
    replay(case, log_fallback=False, log_event=events.append)
    replay(CASES["polr_seed1"], log_fallback=True, log_event=events.append)
    assert events == []
    replay(case, log_fallback=True, log_event=None)  # nothing to log to


# ---------------------------------------------------------------------------
# Direct MASS::polr fits
# ---------------------------------------------------------------------------

FITS = [name for name, case in DIRECT.items() if "r_error" not in case]


def _direct_inputs(case):
    n = len(_listed(case["y"]))
    p = case["p"]
    x = np.empty((n, 0)) if p == 0 else as_array(case["x"]).reshape(n, p)
    new_x = np.empty((case["n_new"], 0)) if p == 0 else as_array(case["new_x"]).reshape(case["n_new"], p)
    return x, as_array(case["y"], dtype=int), as_array(case["weights"]), new_x


def fit_direct(case):
    x, y, weights, _ = _direct_inputs(case)
    return polr_fit(x, y, weights=weights, n_levels=case["n_levels"])


def check_direct(case, fit):
    """Deviance, convergence, aliased columns, coefficients, cut points and the probabilities of new rows."""
    _, _, _, new_x = _direct_inputs(case)
    np.testing.assert_allclose(fit.deviance, case["deviance"], rtol=VALUE_RTOL, atol=0)
    assert fit.converged == (case["convergence"] == 0)
    assert fit.aliased.tolist() == [bool(v) for v in (case["aliased"] or [])]
    np.testing.assert_allclose(fit.coef, as_array(case["coef"] or [], dtype=float), rtol=0, atol=COEF_ATOL)
    np.testing.assert_allclose(fit.zeta, as_array(case["zeta"]), rtol=0, atol=COEF_ATOL)
    prob = as_array(case["prob"]).reshape(case["n_new"], case["n_levels"])
    np.testing.assert_allclose(fit.predict(new_x), prob, rtol=0, atol=PROB_ATOL)


@pytest.mark.parametrize("name", FITS)
def test_polr_fit_matches_r(name):
    case = DIRECT[name]
    check_direct(case, fit_direct(case))


@pytest.mark.parametrize("name", [name for name in DIRECT if name not in FITS])
def test_polr_fit_stops_where_r_stops(name):
    """polr_fit raises PolrError with the condition R's polr stops on."""
    case = DIRECT[name]
    with pytest.raises(PolrError) as info:
        fit_direct(case)
    assert info.value.condition == case["r_error"]
    assert isinstance(info.value, ValueError)
