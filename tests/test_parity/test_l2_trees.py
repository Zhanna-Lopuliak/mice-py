"""Layer 2 for the tree methods cart and rf.

R builds its trees with rpart and ranger, Python with scikit-learn. What is
tested:

* CART, numeric target: rpart's tree is deterministic and only the draw
  within the leaf is random. With R's settings, which are cart's defaults
  (``bootstrap=False, cp=1e-4, min_samples_leaf=5``, hence
  ``min_samples_split=15, max_depth=30, ccp_alpha=1e-4 * var(y_obs)``), the
  set of donors reachable for each missing cell must equal the set R reached
  in 2,000 calls. Because the leaves are the same, the strict tests T1, T2
  and T3 of test_l2_univariate apply as well.
* RF: the draw rule (pool the donors of all trees, then draw one) on the
  helper ``imputation.rf._draw_pooled_donors``.
* RF, numeric target: the forests differ (scikit-learn is not ranger), but
  with ranger-like settings the draws agree in distribution, so T1, T2 and
  T3 apply, next to the weaker checks below.
* CART and RF with a factor target, and RF with a numeric target: imputations
  are observed values or levels; numeric: per-cell mean within 0.25 R
  standard deviations (of that cell) of R's mean and pooled variance ratio
  in [0.5, 2]; factor: overall level proportions within 0.10 of R's.
"""

import numpy as np
import pytest

from ._l2_harness import (
    cached_comparison,
    cached_run,
    donor_set_mismatches,
    run_python,
    weak_factor,
    weak_numeric,
)

pytestmark = pytest.mark.parity

# R's settings for mice.impute.cart; also the defaults of cart
CART_R_SETTINGS = {"bootstrap": False, "cp": 1e-4, "min_samples_leaf": 5}



@pytest.mark.parametrize("input_name", ["F1", "F2", "F4"])
def test_cart_numeric_donor_sets_equal_rpart_leaves(input_name):
    inp, r_sum, py_sum = run_python("cart", input_name, **CART_R_SETTINGS)
    assert py_sum["n_not_donor"] == 0, "imputations must be observed values"
    bad = donor_set_mismatches(r_sum, py_sum)
    assert not bad, "donor sets differ from rpart's: " + "; ".join(bad[:5])


@pytest.mark.parametrize("input_name", ["F1", "F2", "F4"])
@pytest.mark.parametrize("statistic", ["T1", "T2", "T3"])
def test_cart_numeric_target_strict(input_name, statistic):
    """cart with its defaults: donor counts (T1), cell moments (T2) and
    coefficient moments (T3) against R, with the thresholds of pmm."""
    comp = cached_comparison("cart", input_name)
    passed = {"T1": comp.t1_ok, "T2": comp.t2_ok, "T3": comp.t3_ok}[statistic]()
    message = {"T1": comp.t1_message, "T2": comp.t2_message, "T3": comp.t3_message}[statistic]
    assert passed, message()


def test_cart_factor_target_level_proportions_close_to_r():
    inp, r_sum, py_sum = cached_run("cart", "F3")
    assert py_sum["n_not_donor"] == 0, "imputations must be observed levels"
    diff = weak_factor(r_sum, py_sum)
    assert diff <= 0.10, f"largest level-proportion difference {diff:.3f} (pass: <= 0.10)"


@pytest.mark.slow
@pytest.mark.parametrize("input_name", ["F1", "F2", "F4"])
def test_rf_numeric_target_close_to_r(input_name):
    inp, r_sum, py_sum = cached_run("rf", input_name)
    assert py_sum["n_not_donor"] == 0, "imputations must be observed values"
    dev, ratio = weak_numeric(r_sum, py_sum)
    assert dev <= 0.25 and 0.5 <= ratio <= 2, (
        f"largest |mean_P - mean_R| = {dev:.3f} R sd (pass: <= 0.25); "
        f"pooled variance ratio {ratio:.3f} (pass: in [0.5, 2])"
    )


@pytest.mark.slow
@pytest.mark.parametrize("input_name", ["F1", "F2", "F4"])
@pytest.mark.parametrize("statistic", ["T1", "T2", "T3"])
def test_rf_numeric_target_strict(input_name, statistic):
    """rf with its defaults (10 trees, ranger-like node sizes and mtry,
    donors pooled over all trees) against ranger: T1, T2 and T3 with the
    thresholds of pmm, on 300 Python calls."""
    comp = cached_comparison("rf", input_name)
    passed = {"T1": comp.t1_ok, "T2": comp.t2_ok, "T3": comp.t3_ok}[statistic]()
    message = {"T1": comp.t1_message, "T2": comp.t2_message, "T3": comp.t3_message}[statistic]
    assert passed, message()


@pytest.mark.slow
def test_rf_factor_target_level_proportions_close_to_r():
    inp, r_sum, py_sum = cached_run("rf", "F3")
    assert py_sum["n_not_donor"] == 0, "imputations must be observed levels"
    diff = weak_factor(r_sum, py_sum)
    assert diff <= 0.10, f"largest level-proportion difference {diff:.3f} (pass: <= 0.10)"


def test_rf_draw_rule_pools_donors_of_all_trees():
    from imputation.rf import _draw_pooled_donors

    # observed rows a, b, c, d: tree 1 puts a alone in leaf 1 (b, c, d in
    # leaf 2); tree 2 puts b, c, d in leaf 5 (a in leaf 9). A missing row in
    # leaf 1 of tree 1 and leaf 5 of tree 2 pools {a} and {b, c, d}, so R
    # draws a with probability 1/4.
    nodes_obs = np.array([[1, 9], [2, 5], [2, 5], [2, 5]])
    nodes_mis = np.tile([1, 5], (4000, 1))
    y_obs = np.array([10.0, 20.0, 30.0, 40.0])
    draws = np.asarray(
        _draw_pooled_donors(nodes_obs, nodes_mis, y_obs, np.random.default_rng(2025))
    )
    assert draws.shape == (4000,)
    assert set(np.unique(draws)) <= set(y_obs)
    p_hat = np.mean(draws == 10.0)
    # sd of p_hat is 0.0068; 0.038 is about 5.5 sd
    assert abs(p_hat - 0.25) <= 0.038, f"P(a) = {p_hat:.3f}, R's rule gives 0.25"
