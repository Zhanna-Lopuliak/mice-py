"""Layer 2 with general masks: pmm, midastouch, cart, rf and sample on F8.

R's ``sampler.univ`` calls an imputer with every row: ``ry``, the fit rows,
and ``wy``, the rows to impute, may overlap (under ``where``), and a row may
be in neither (an ignored row, a missing cell that ``where`` leaves out, a
row with incomplete predictors). F8 (tests/reference/r/_l2_common.R) has 40
fit rows, 20 rows to impute of which 8 are fit rows too, and 8 rows in
neither mask: 4 with an observed decoy y far above the y of the fit rows,
4 with NA in y, two of those with NA in x; the other rows in neither mask
have predictors far outside the range of the rows used.

R called ``mice.impute.<method>(y, ry, x, wy)`` as often as for F1 to F4
(rf 5,000 times, the others 20,000; fixtures ``l2_masks_<method>``, written
by tests/reference/r/l2_masks.R, which stops if R ever imputes a value that
is not the value of a fit row). Python calls each imputer with ``id_mis=wy``
as often as for F1 to F4 (``N_CALLS`` in ``_l2_harness``) and must pass T1,
T2 and T3 with the thresholds of the other inputs. A fit row to impute can
be its own donor (pmm draws it in about 10 to 20 percent of R's calls); R is
the reference for how often. What the statistics count under general masks
is described in ``_l2_harness``. As on F1 to F4, cart must also reach the
donor sets of rpart's leaves.
"""

import numpy as np
import pytest

from ._l2_harness import cached_comparison, cached_run, donor_set_mismatches, r_reference

pytestmark = pytest.mark.parity

METHODS = ("pmm", "midastouch", "cart", "rf", "sample")


def test_f8_has_the_planned_masks():
    """60 rows: 40 fit rows, 20 rows to impute (8 of them fit rows), 8 rows
    in neither mask, 4 of them with an observed y far above every y of the
    fit rows. NaN in x only in two rows in neither mask with NaN in y, in
    one predictor and in all three; the other predictors of those rows lie
    far outside the range of the rows used. Every fixture has this input."""
    payloads = [r_reference(method, "F8")[1]["input"] for method in METHODS]
    assert all(payload == payloads[0] for payload in payloads[1:])
    inp, _ = r_reference("pmm", "F8")
    neither = ~inp.used
    assert len(inp.ry) == 60
    assert (inp.ry.sum(), inp.wy.sum(), (inp.ry & inp.wy).sum(), neither.sum()) == (40, 20, 8, 8)
    observed = ~np.isnan(inp.y)
    assert observed[inp.ry].all() and not observed[inp.wy & ~inp.ry].any()
    assert (neither & observed).sum() == 4
    y_fit = inp.y[inp.ry]
    assert inp.y[neither & observed].min() > y_fit.max() + 10 * y_fit.std(ddof=1)
    n_nan = np.isnan(inp.x).sum(axis=1)
    assert not n_nan[inp.used].any()
    assert sorted(n_nan[n_nan > 0].tolist()) == [1, 3]
    assert neither[n_nan > 0].all() and np.isnan(inp.y[n_nan > 0]).all()
    decoys = inp.x[neither]
    low, high = inp.x[inp.used].min(axis=0), inp.x[inp.used].max(axis=0)
    assert ((decoys < low) | (decoys > high) | np.isnan(decoys)).all()


@pytest.mark.parametrize("statistic", ["T1", "T2", "T3"])
@pytest.mark.parametrize("method", ["pmm", "midastouch", "cart", pytest.param("rf", marks=pytest.mark.slow), "sample"])
def test_f8_matches_r(method, statistic):
    """Donor counts (T1), cell moments (T2) and coefficient moments (T3)
    against R, with the thresholds of F1 to F4."""
    comp = cached_comparison(method, "F8")
    passed = {"T1": comp.t1_ok, "T2": comp.t2_ok, "T3": comp.t3_ok}[statistic]()
    message = {"T1": comp.t1_message, "T2": comp.t2_message, "T3": comp.t3_message}[statistic]
    assert passed, message()


def test_cart_f8_donor_sets_equal_rpart_leaves():
    """cart with its defaults (R's settings) reaches, for every cell, the
    donors R reached in its first 2,000 calls."""
    inp, r_sum, py_sum = cached_run("cart", "F8")
    assert py_sum["n_not_donor"] == 0, "imputations must be values of fit rows"
    bad = donor_set_mismatches(r_sum, py_sum)
    assert not bad, "donor sets differ from rpart's: " + "; ".join(bad[:5])
