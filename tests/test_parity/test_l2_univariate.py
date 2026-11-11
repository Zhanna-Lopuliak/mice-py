"""Layer 2: distribution parity of pmm, midastouch, sample, the norm family, logreg and polyreg with R mice.

For each method and fixed input (F1 to F6, see tests/reference/r/_l2_common.R)
R called ``mice.impute.<method>`` 20,000 times; here the Python imputer is
called 3,000 times (midas: 1,500) with one seeded generator. Per missing
cell we compare the donor counts (T1) and the mean and variance of the
imputed value (T2); per OLS coefficient of the completed y on ``[1, x]`` the
mean and variance across calls (T3). Thresholds are in ``_l2_stats``.

``norm``, ``norm.nob`` and ``norm.boot`` impute new numbers rather than
observed values (fixtures of the continuous kind): their T1 compares, per
cell, the counts in the ten bins cut at the deciles of R's 20,000 draws.
T2 and T3 are the same as for the donor methods. F1 puts half of the missing
rows where the coefficients are poorly determined, so the parameter draw of
``norm`` and the bootstrap of ``norm.boot`` matter there.

F2 (one binary predictor, many tied predictions) is tested with T1 as the
recipe asks, plus T3 for pmm: R's ``matchindex`` shuffles the donors once per
call, so the cells of one call share their tie-break, which only the
coefficient moments can see (a per-cell random tie-break passes T1 and T2
on F2 but fails T3 with |z| of about 27).

``logreg`` and ``logreg.boot`` draw labels from a fitted model (fixtures of
the category kind) on binary targets: T1 compares the level counts per
cell; T3 the mean and variance across calls of each OLS coefficient of the
0/1 "yes" indicator of the completed target on ``[1, x]``, which sees the
dependence between the cells of one call that the shared coefficient draw
(or bootstrap sample) creates. There is no T2 for labels. On F5 half of the
missing rows lie far out where the labels depend on the poorly determined
slopes; on F6 the observed rows are separated, so ``logreg`` depends on its
augmentation (``logreg.boot``, which does not augment, is run on F5 only).

``polyreg`` draws labels from a fitted multinomial model (category kind) on
F3, a target with three levels: T1 compares the level counts per cell, T3
the moments of the coefficients of the "b" and "c" indicators. R called it
20,000 times, Python 3,000 times.
"""

import pytest

from ._l2_harness import cached_comparison

pytestmark = pytest.mark.parity


T1_CASES = [
    pytest.param("pmm", "F1"),
    pytest.param("pmm", "F2"),
    pytest.param("pmm", "F3"),
    pytest.param("pmm", "F4"),
    pytest.param("midastouch", "F1"),
    pytest.param("midastouch", "F2"),
    pytest.param("midastouch", "F4"),
    pytest.param("sample", "F1"),
    pytest.param("sample", "F3"),
]

# continuous kind: T1 on R's decile bins, T2 and T3 on each
NORM_CASES = [
    pytest.param("norm", "F1"),
    pytest.param("norm", "F4"),
    pytest.param("norm.nob", "F1"),
    pytest.param("norm.boot", "F1"),
]

T2_CASES = [
    pytest.param("pmm", "F1"),
    pytest.param("pmm", "F4"),
    pytest.param("midastouch", "F1"),
    pytest.param("midastouch", "F4"),
    pytest.param("sample", "F1"),
] + NORM_CASES

# category kind: T1 on the level counts, T3 on the level indicator
CATEGORY_CASES = [
    pytest.param("logreg", "F5"),
    pytest.param("logreg", "F6"),
    pytest.param("logreg.boot", "F5"),
    pytest.param("polyreg", "F3"),
]

T3_CASES = T2_CASES[:1] + [pytest.param("pmm", "F2")] + T2_CASES[1:] + CATEGORY_CASES


@pytest.mark.parametrize("method, input_name", T1_CASES)
def test_t1_donor_counts_match_r(method, input_name):
    comp = cached_comparison(method, input_name)
    assert comp.t1_ok(), comp.t1_message()


@pytest.mark.parametrize("method, input_name", NORM_CASES)
def test_t1_decile_bin_counts_match_r(method, input_name):
    comp = cached_comparison(method, input_name)
    assert comp.kind == "continuous"
    assert comp.t1_ok(), comp.t1_message()


@pytest.mark.parametrize("method, input_name", CATEGORY_CASES)
def test_t1_level_counts_match_r(method, input_name):
    comp = cached_comparison(method, input_name)
    assert comp.kind == "category"
    assert comp.t1_ok(), comp.t1_message()


@pytest.mark.parametrize("method, input_name", T2_CASES)
def test_t2_cell_mean_and_variance_match_r(method, input_name):
    comp = cached_comparison(method, input_name)
    assert comp.t2_ok(), comp.t2_message()


@pytest.mark.parametrize("method, input_name", T3_CASES)
def test_t3_coefficient_mean_and_variance_match_r(method, input_name):
    comp = cached_comparison(method, input_name)
    assert comp.t3_ok(), comp.t3_message()
