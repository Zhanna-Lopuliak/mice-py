"""Layer 2 for the tree methods cart and rf.

R builds its trees with rpart and ranger, Python with scikit-learn, so the
draws cannot be compared as strictly as for pmm. What is tested instead:

* CART, numeric target: rpart's tree is deterministic and only the draw
  within the leaf is random. With the planned arguments
  ``cart(..., bootstrap=False, cp=1e-4, min_samples_leaf=5)`` the set of
  donors reachable for each missing cell must equal the set R reached in
  2,000 calls. (A scratch experiment fitting scikit-learn's tree with
  ``min_samples_split=15, min_samples_leaf=5, max_depth=30,
  ccp_alpha=1e-4 * var(y_obs)`` reproduced rpart's leaves exactly on F1, F2
  and F4.)
* RF: the draw rule (pool the donors of all trees, then draw one) on the
  planned helper ``imputation.rf._draw_pooled_donors``.
* CART and RF with a factor target, and RF with a numeric target: imputations
  are observed values or levels; numeric: per-cell mean within 0.25 R
  standard deviations (of that cell) of R's mean and pooled variance ratio
  in [0.5, 2]; factor: overall level proportions within 0.10 of R's.
"""

import numpy as np
import pytest

from ._l2_harness import (
    cached_run,
    donor_set_mismatches,
    run_python,
    weak_factor,
    weak_numeric,
)

pytestmark = pytest.mark.parity

CART_PLANNED = {"bootstrap": False, "cp": 1e-4, "min_samples_leaf": 5}

A6_CART_ARGS = pytest.mark.xfail(
    strict=True,
    raises=TypeError,
    reason="A6: cart has no bootstrap/cp arguments yet; it bootstraps and uses ccp_alpha=1e-4",
)
A6_RF = pytest.mark.xfail(
    strict=True,
    reason="A6: rf draws one donor per tree and uses scikit-learn defaults, not ranger's",
)


@pytest.mark.parametrize(
    "input_name",
    [pytest.param(name, marks=A6_CART_ARGS) for name in ("F1", "F2", "F4")],
)
def test_cart_numeric_donor_sets_equal_rpart_leaves(input_name):
    inp, r_sum, py_sum = run_python("cart", input_name, **CART_PLANNED)
    assert py_sum["n_not_donor"] == 0, "imputations must be observed values"
    bad = donor_set_mismatches(r_sum, py_sum)
    assert not bad, "donor sets differ from rpart's: " + "; ".join(bad[:5])


def test_cart_factor_target_level_proportions_close_to_r():
    inp, r_sum, py_sum = cached_run("cart", "F3")
    assert py_sum["n_not_donor"] == 0, "imputations must be observed levels"
    diff = weak_factor(r_sum, py_sum)
    assert diff <= 0.10, f"largest level-proportion difference {diff:.3f} (pass: <= 0.10)"


@pytest.mark.slow
@pytest.mark.parametrize(
    "input_name",
    [pytest.param("F1", marks=A6_RF), "F2", pytest.param("F4", marks=A6_RF)],
)
def test_rf_numeric_target_close_to_r(input_name):
    inp, r_sum, py_sum = cached_run("rf", input_name)
    assert py_sum["n_not_donor"] == 0, "imputations must be observed values"
    dev, ratio = weak_numeric(r_sum, py_sum)
    assert dev <= 0.25 and 0.5 <= ratio <= 2, (
        f"largest |mean_P - mean_R| = {dev:.3f} R sd (pass: <= 0.25); "
        f"pooled variance ratio {ratio:.3f} (pass: in [0.5, 2])"
    )


@pytest.mark.slow
def test_rf_factor_target_level_proportions_close_to_r():
    inp, r_sum, py_sum = cached_run("rf", "F3")
    assert py_sum["n_not_donor"] == 0, "imputations must be observed levels"
    diff = weak_factor(r_sum, py_sum)
    assert diff <= 0.10, f"largest level-proportion difference {diff:.3f} (pass: <= 0.10)"


@pytest.mark.xfail(
    strict=True,
    raises=ImportError,
    reason="A6: rf has no _draw_pooled_donors yet; today it draws one donor per tree "
    "and then one of those, so a donor alone in its leaf gets probability 1/2",
)
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
