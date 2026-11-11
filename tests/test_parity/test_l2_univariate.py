"""Layer 2: distribution parity of pmm, midastouch and sample with R mice.

For each method and fixed input (F1 to F4, see tests/reference/r/_l2_common.R)
R called ``mice.impute.<method>`` 20,000 times; here the Python imputer is
called 3,000 times (midas: 1,500) with one seeded generator. Per missing
cell we compare the donor counts (T1) and the mean and variance of the
imputed value (T2); per OLS coefficient of the completed y on ``[1, x]`` the
mean and variance across calls (T3). Thresholds are in ``_l2_stats``.

F2 (one binary predictor, many tied predictions) is tested with T1 as the
recipe asks, plus T3 for pmm: R's ``matchindex`` shuffles the donors once per
call, so the cells of one call share their tie-break, which only the
coefficient moments can see (a per-cell random tie-break passes T1 and T2
on F2 but fails T3 with |z| of about 27).
"""

import pytest

from ._l2_harness import cached_comparison

pytestmark = pytest.mark.parity


def _xfail(reason, raises=None):
    return pytest.mark.xfail(strict=True, reason=reason, raises=raises)


A1 = _xfail("A1: transposed Cholesky factor in norm_draw shrinks/misplaces coefficient draws")
A9_A1 = _xfail(
    "A9, A1: KD-tree matching breaks donor ties deterministically "
    "(and the coefficient draw is wrong)"
)
B1 = _xfail(
    "B1: pmm crashes on a factor target (parity also needs A3, R's quantify)",
    raises=AttributeError,
)
NEW1 = _xfail(
    "B6: midas crashes when several donors are at distance 0 (tied predictors): "
    "minmax caps the column sums, so p no longer sums to 1 for np.random.choice; "
    "R's sample() renormalises",
    raises=ValueError,
)

T1_CASES = [
    pytest.param("pmm", "F1", marks=A1),
    pytest.param("pmm", "F2", marks=A9_A1),
    pytest.param("pmm", "F3", marks=B1),
    pytest.param("pmm", "F4"),
    pytest.param("midastouch", "F1"),
    pytest.param("midastouch", "F2", marks=NEW1),
    pytest.param("midastouch", "F4"),
    pytest.param("sample", "F1"),
    pytest.param("sample", "F3"),
]

T2_CASES = [
    pytest.param("pmm", "F1", marks=A1),
    pytest.param("pmm", "F4"),
    pytest.param("midastouch", "F1"),
    pytest.param("midastouch", "F4"),
    pytest.param("sample", "F1"),
]

T3_CASES = T2_CASES[:1] + [pytest.param("pmm", "F2", marks=A9_A1)] + T2_CASES[1:]


@pytest.mark.parametrize("method, input_name", T1_CASES)
def test_t1_donor_counts_match_r(method, input_name):
    comp = cached_comparison(method, input_name)
    assert comp.t1_ok(), comp.t1_message()


@pytest.mark.parametrize("method, input_name", T2_CASES)
def test_t2_cell_mean_and_variance_match_r(method, input_name):
    comp = cached_comparison(method, input_name)
    assert comp.t2_ok(), comp.t2_message()


@pytest.mark.parametrize("method, input_name", T3_CASES)
def test_t3_coefficient_mean_and_variance_match_r(method, input_name):
    comp = cached_comparison(method, input_name)
    assert comp.t3_ok(), comp.t3_message()
