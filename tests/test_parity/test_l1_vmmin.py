"""L1 parity: R's BFGS minimiser (vmmin), which nnet::multinom and MASS::polr call.

Reference: tests/reference/r/l1_vmmin.R (optim(method = "BFGS") on small
functions, with its parameters, value, evaluation counts and convergence
code).

The functions below are those of the R script, written with elementwise
arithmetic and left-to-right sums as R evaluates them, so that both sides
take the same steps: the numbers of function and gradient evaluations and
the convergence flag must be equal, the parameters and the value agree
within 1e-8. The start values of the R script keep every case away from a
stop whose function count depends on the last bits of the values (see
there); `test_counts_do_not_depend_on_the_last_bits` checks that here.
"""

import numpy as np
import pytest

from imputation.fitting import vmmin

from ._helpers import as_array, load_fixture

pytestmark = pytest.mark.parity

TOL = 1e-8

FIXTURE = load_fixture("l1_vmmin")
CASES = {case["name"]: case for case in FIXTURE["cases"]}
DATA = {key: as_array(values) for key, values in FIXTURE["logit_data"].items()}


def _sum(values):
    """R's sum(): left to right in double precision."""
    return float(np.add.accumulate(np.asarray(values, dtype=float))[-1])


QUAD_CENTER = np.array([1.0, -2.0, 0.5])
QUAD_SCALE = np.array([1.0, 10.0, 0.1])


def _quadratic_fn(p):
    return _sum(QUAD_SCALE * (p - QUAD_CENTER) ** 2) + (p[0] - p[1] - 3) ** 2


def _quadratic_gr(p):
    cross = 2 * (p[0] - p[1] - 3)
    return 2 * QUAD_SCALE * (p - QUAD_CENTER) + np.array([cross, -cross, 0.0])


def _rosenbrock_fn(p):
    return 100 * (p[1] - p[0] ** 2) ** 2 + (1 - p[0]) ** 2


def _rosenbrock_gr(p):
    return np.array([-400 * p[0] * (p[1] - p[0] ** 2) - 2 * (1 - p[0]), 200 * (p[1] - p[0] ** 2)])


def _eta(p):
    return p[0] + p[1] * DATA["x1"] + p[2] * DATA["x2"]


def _logistic_fn(p):
    eta = _eta(p)
    return _sum(np.log(1 + np.exp(eta)) - DATA["y"] * eta)


def _logistic_gr(p):
    r = 1 / (1 + np.exp(-_eta(p))) - DATA["y"]
    return np.array([_sum(r), _sum(r * DATA["x1"]), _sum(r * DATA["x2"])])


def _barrier_fn(p):
    with np.errstate(invalid="ignore"):  # NaN for a negative parameter, as R's log
        return _sum(10 * p - np.log(p))


def _barrier_gr(p):
    return 10 - 1 / p


FUNCTIONS = {
    "quadratic": (_quadratic_fn, _quadratic_gr),
    "rosenbrock": (_rosenbrock_fn, _rosenbrock_gr),
    "logistic": (_logistic_fn, _logistic_gr),
    "barrier": (_barrier_fn, _barrier_gr),
    "squares": (lambda p: _sum(p**2), lambda p: 2 * p),
}


def _number(value):
    """A tolerance from the fixture ("-Inf" for R's default abstol)."""
    return float(value)


def run(case):
    fn, gr = FUNCTIONS[case["fun"]]
    return vmmin(
        fn,
        gr,
        as_array(case["start"]),
        maxit=case["maxit"],
        abstol=_number(case["abstol"]),
        reltol=_number(case["reltol"]),
    )


@pytest.mark.parametrize("name", list(CASES))
def test_counts_and_convergence_match_r(name):
    case = CASES[name]
    res = run(case)
    assert [res.fn_count, res.gr_count] == case["counts"]
    assert res.converged == (case["convergence"] == 0)


@pytest.mark.parametrize("name", list(CASES))
def test_parameters_and_value_match_r(name):
    case = CASES[name]
    res = run(case)
    np.testing.assert_allclose(res.par, as_array(case["par"]), rtol=TOL, atol=TOL)
    np.testing.assert_allclose(res.value, case["value"], rtol=TOL, atol=TOL)


def _perturbed(function, seed):
    """`function` with every value multiplied by ``1 + e``, ``|e| <= 1e-15`` (a few ulps)."""
    rng = np.random.default_rng(seed)

    def perturbed(p):
        value = function(p)
        return value * (1 + 1e-15 * rng.uniform(-1, 1, size=np.shape(value)))

    return perturbed


@pytest.mark.parametrize("name", list(CASES))
def test_counts_do_not_depend_on_the_last_bits(name):
    """R's C code and NumPy round differently (fused multiply-adds, exp and
    log); the counts of these cases must not depend on that."""
    case = CASES[name]
    fn, gr = FUNCTIONS[case["fun"]]
    for seed in range(5):
        res = vmmin(
            _perturbed(fn, seed),
            _perturbed(gr, seed + 100),
            as_array(case["start"]),
            maxit=case["maxit"],
            abstol=_number(case["abstol"]),
            reltol=_number(case["reltol"]),
        )
        assert [res.fn_count, res.gr_count] == case["counts"], seed


def test_fixture_covers_the_stopping_rules():
    """The iteration limit, abstol, a looser reltol, nnet's tolerances, rejected non-finite points, a zero gradient."""
    assert {name for name, case in CASES.items() if case["convergence"] == 1} == {"rosenbrock_maxit", "logistic_maxit"}
    assert CASES["quadratic_abstol"]["value"] <= 1e-3 < CASES["quadratic"]["counts"][0]
    assert CASES["logistic_reltol"]["counts"][1] < CASES["logistic"]["counts"][1]
    assert CASES["start_at_minimum"]["counts"] == [1, 1]


def test_start_value_must_be_finite():
    with pytest.raises(ValueError, match="not finite at the start values"):
        vmmin(_barrier_fn, _barrier_gr, [-1.0, 1.0])


@pytest.mark.parametrize("maxit", [0, -3])
def test_maxit_not_positive_only_evaluates_the_start(maxit):
    """R's vmmin: no iteration, counts 0, convergence 0."""
    res = vmmin(_rosenbrock_fn, _rosenbrock_gr, [-1.2, 1.0], maxit=maxit)
    np.testing.assert_array_equal(res.par, [-1.2, 1.0])
    assert (res.value, res.converged, res.fn_count, res.gr_count) == (_rosenbrock_fn([-1.2, 1.0]), True, 0, 0)
