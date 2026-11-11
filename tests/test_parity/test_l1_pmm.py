"""L1 parity: predictive mean matching pieces.

Reference: tests/reference/r/l1_pmm.R (mice:::quantify, mice.impute.pmm with
donors = 1, matchindex).
"""

import numpy as np
import pytest

from imputation.PMM import pmm

from ._helpers import ScriptedGenerator, as_array, as_bool, load_fixture

pytestmark = pytest.mark.parity

RTOL = 1e-8
ATOL = 1e-10

QUANTIFY = load_fixture("l1_quantify")
PMM_ONE = load_fixture("l1_pmm_one_donor")
MATCH = load_fixture("l1_matchindex")


def _labels(values):
    """Factor values as an object array with NaN for missing, as MICE passes them."""
    return np.array([np.nan if v is None else v for v in values], dtype=object)


# ---------------------------------------------------------------------------
# quantify
# ---------------------------------------------------------------------------


@pytest.mark.xfail(
    strict=True,
    reason="A3: quantify() not implemented; quantify_cca uses predictor-side scores",
)
@pytest.mark.parametrize("case", QUANTIFY["cases"], ids=[c["name"] for c in QUANTIFY["cases"]])
def test_quantify_matches_r_up_to_sign(case):
    from imputation.PMM import quantify

    y = _labels(case["y"])
    ry = as_bool(case["ry"])
    x = as_array(case["x"])  # includes the intercept column, as in R
    expected = as_array(case["ynum"])[ry]

    ynum = np.asarray(quantify(y, ry, x), dtype=float)
    assert ynum.shape == (len(y),)
    got = ynum[ry]
    sign = np.sign(np.dot(got, expected))
    np.testing.assert_allclose(sign * got, expected, rtol=RTOL, atol=ATOL)


# ---------------------------------------------------------------------------
# PMM with one donor and R's random numbers
# ---------------------------------------------------------------------------

PMM_KNOWN = {
    "numeric_matchtype1": "A1: beta uses the transposed Cholesky factor",
    "numeric_matchtype2": "A1: beta uses the transposed Cholesky factor",
    "numeric_matchtype1_seed2": "A1: beta uses the transposed Cholesky factor",
    "factor_matchtype0": "B1: pmm crashes on a categorical target (parity also needs A3)",
}


def _pmm_params():
    for case in PMM_ONE["cases"]:
        reason = PMM_KNOWN.get(case["name"])
        marks = [pytest.mark.xfail(strict=True, reason=reason)] if reason else []
        yield pytest.param(case, id=case["name"], marks=marks)


@pytest.mark.parametrize("case", list(_pmm_params()))
def test_pmm_one_donor_matches_r(case):
    is_factor = case["levels"] is not None
    y = _labels(case["y"]) if is_factor else as_array(case["y"])
    ry = as_bool(case["ry"])
    x = as_array(case["x"])  # without intercept; pmm adds it
    rng = ScriptedGenerator(chisq=case["chisq"], normals=case["normals"])

    imputed = pmm(y, ry, x, donors=1, matchtype=case["matchtype"], rng=rng)

    assert rng.unused == (0, 0), "not every captured R draw was consumed"
    if is_factor:
        np.testing.assert_array_equal(
            np.asarray(imputed, dtype=object), np.array(case["imputed"], dtype=object)
        )
    else:
        # donors are observed values, so the match must be exact
        np.testing.assert_array_equal(np.asarray(imputed, dtype=float), as_array(case["imputed"]))


# ---------------------------------------------------------------------------
# Donor matching with k > 1
# ---------------------------------------------------------------------------

N_CALLS = 500  # P(missing one of 12 donors in 500 draws) < 1e-17 per target


def _expected_reached(case):
    return [set(np.atleast_1d(r).astype(int).tolist()) for r in case["reached"]]


def _reached(draw):
    t = as_array(MATCH["t"])
    reached = [set() for _ in t]
    for _ in range(N_CALLS):
        for j, i in enumerate(np.asarray(draw(), dtype=int)):
            reached[j].add(int(i))
    return reached


def _match_params(today):
    for case in MATCH["cases"]:
        n_donors = len(MATCH["d"])
        if case["k"] > n_donors:
            reason = "B3: k larger than the number of donors is not clamped"
        elif today:
            reason = None
        else:
            reason = (
                "A9: matchindex() not implemented; KD-tree matcherid breaks ties deterministically"
            )
        marks = [pytest.mark.xfail(strict=True, reason=reason)] if reason else []
        yield pytest.param(case, id=case["name"], marks=marks)


@pytest.mark.parametrize("case", list(_match_params(today=False)))
def test_matchindex_reaches_r_donor_set(case):
    from imputation.PMM import matchindex

    d = as_array(MATCH["d"])
    t = as_array(MATCH["t"])
    rng = np.random.default_rng(20261116)
    reached = _reached(lambda: matchindex(d, t, k=case["k"], rng=rng))
    assert reached == _expected_reached(case)


@pytest.mark.parametrize("case", list(_match_params(today=True)))
def test_matcherid_today_reaches_r_donor_set(case):
    """Today's KD-tree matcher, without ties. Remove together with matcherid."""
    from imputation.PMM import matcherid

    d = as_array(MATCH["d"])
    t = as_array(MATCH["t"])
    rng = np.random.default_rng(20261116)
    reached = _reached(lambda: matcherid(d, t, matcher="NN", k=case["k"], rng=rng))
    assert reached == _expected_reached(case)
