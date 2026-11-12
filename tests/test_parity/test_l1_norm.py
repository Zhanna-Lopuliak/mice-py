"""L1 parity: imputation by linear regression (norm, norm.nob, norm.boot, norm.predict) and by the mean.

Reference: tests/reference/r/l1_norm.R (mice.impute.norm, .norm.nob,
.norm.boot, .norm.predict, .mean with every random number captured).

Each case is replayed with a strict ScriptedGenerator that serves R's draws
and refuses any other. The checks are split so that test_l1_mutants can run
them against a defective library: `replay` runs the port, `check_imputations`
compares the result with R's, `check_draws` the draws the port asked for
(all used, in R's order, ``df`` of the chi-square, bounds of the bootstrap).
"""

import numpy as np
import pytest

from imputation.mean import mean
from imputation.norm import norm, norm_boot, norm_nob, norm_predict

from ._helpers import ScriptedGenerator, as_array, as_bool, load_fixture

pytestmark = pytest.mark.parity

RTOL = 1e-8
ATOL = 1e-10

FIXTURE = load_fixture("l1_norm")
CASES = {case["name"]: case for case in FIXTURE["cases"]}
IMPUTERS = {
    "norm": norm,
    "norm.nob": norm_nob,
    "norm.boot": norm_boot,
    "norm.predict": norm_predict,
    "mean": mean,
}


def _inputs(case):
    """(y, ry, wy, x) of a case; wy is None where R used ``!ry``."""
    inp = FIXTURE["inputs"][case["input"]]
    y = as_array(inp["y"])
    ry = as_bool(inp["ry"])
    wy = None if inp["wy"] is None else as_bool(inp["wy"])
    x = np.empty((len(y), 0)) if inp["p"] == 0 else as_array(inp["x"]).reshape(len(y), inp["p"])
    return y, ry, wy, x


def _scripted(case):
    """A strict generator with R's draws: integers only for norm.boot."""
    integers = None if case["integers"] is None else as_array(case["integers"], dtype=int)
    return ScriptedGenerator(
        chisq=as_array(case["chisq"]), normals=as_array(case["normals"]), integers=integers, strict=True
    )


def replay(case):
    """Run the port on the inputs of `case` with R's draws; return (imputed, rng)."""
    y, ry, wy, x = _inputs(case)
    rng = _scripted(case)
    params = {} if case["method"] == "mean" else {"ls_meth": case["ls_meth"], "ridge": case["ridge"]}
    imputed = IMPUTERS[case["method"]](y, ry, x, id_mis=wy, rng=rng, **params)
    return np.asarray(imputed), rng


def check_imputations(case, imputed):
    """The port's imputations equal R's (or, where R fails, R's intercept-only rebuild)."""
    expected = as_array(case["port_imputed"] if case["port"] == "intercept_only" else case["imputed"])
    assert imputed.dtype == np.float64
    np.testing.assert_allclose(imputed, expected, rtol=RTOL, atol=ATOL)


def check_draws(case, rng):
    """The port drew what R drew: every value, in R's order, with R's df and bounds."""
    rng.assert_exhausted()
    assert rng.order == [tuple(pair) for pair in case["order"]]
    chisq = [call.args["df"] for call in rng.calls if call.method == "chisquare"]
    assert chisq == ([] if case["df"] is None else [case["df"]])
    bounds = [call.args for call in rng.calls if call.method == "integers"]
    expected = {"low": 0, "high": case["n_obs"], "endpoint": False}
    assert bounds == ([] if case["integers"] is None else [expected])


def check_case(case):
    """Every L1 check of one case."""
    imputed, rng = replay(case)
    check_imputations(case, imputed)
    check_draws(case, rng)


def _names(port):
    return [name for name, case in CASES.items() if case["port"] == port]


@pytest.mark.parametrize("name", _names("match"))
def test_imputations_match_r(name):
    case = CASES[name]
    imputed, _ = replay(case)
    check_imputations(case, imputed)


@pytest.mark.parametrize("name", _names("match") + _names("intercept_only"))
def test_draws_match_r(name):
    case = CASES[name]
    _, rng = replay(case)
    check_draws(case, rng)


def test_fixture_covers_every_method_and_variant():
    """Each method has the base data, one predictor, none, one missing value,
    a subset of the missing cells and a target filled at the missing cells."""
    variants = {"base", "one_predictor", "no_predictor", "one_missing", "wy_subset", "filled_target"}
    for method in IMPUTERS:
        assert {c["input"] for c in CASES.values() if c["method"] == method} >= variants, method
    assert {c["ls_meth"] for c in CASES.values() if c["method"] == "norm"} == {"qr", "ridge"}


@pytest.mark.parametrize("method", list(IMPUTERS))
def test_values_at_missing_cells_are_ignored(method):
    """Inside MICE the missing cells hold the current imputations; they must
    not enter the fit. The same draws give the same imputations."""
    label = method.replace(".", "_")
    filled, _ = replay(CASES[f"{label}_filled_target"])
    plain, _ = replay(CASES["norm_qr_seed1" if method == "norm" else label])
    np.testing.assert_array_equal(filled, plain)


def test_norm_boot_without_predictors_fits_the_intercept():
    """R stops in lm.fit ("'x' must be a matrix") after drawing the bootstrap
    sample; the port fits the intercept-only model to that sample."""
    case = CASES["norm_boot_no_predictor"]
    assert case["r_error"] == "'x' must be a matrix"
    check_case(case)


@pytest.mark.parametrize("method", ["norm.nob", "norm.boot"])
def test_too_few_observed_values_raise(method):
    """With n_obs = q + 1, R divides by zero and imputes infinite values; the
    port refuses before drawing anything."""
    case = CASES["norm_nob_tiny"]
    assert not np.isfinite(as_array(case["imputed"])).all()
    assert case["n_obs"] == case["n_coef"] + 1
    y, ry, wy, x = _inputs(case)
    rng = ScriptedGenerator(strict=True)
    with pytest.raises(ValueError, match=rf"{method} needs more than 4 observed values with 3 coefficients"):
        IMPUTERS[method](y, ry, x, id_mis=wy, rng=rng)
    assert rng.calls == []
