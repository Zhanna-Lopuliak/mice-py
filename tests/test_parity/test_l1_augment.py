"""L1 parity: the augmentation of categorical imputation models (mice:::augment).

Reference: tests/reference/r/l1_augment.R (mice:::augment on small inputs,
no random numbers).

R's augment works on the integer codes R sees (``unclass(y)``): the
declared level codes of a factor, the values of a numeric or logical target.
`replay` passes those codes to the port and turns the result back into
labels; `check_augmented` compares rows, weights, targets and masks with R's,
and `check_levels` the level order after the augmentation with
`target_categories`. test_l1_mutants runs them against a defective library.
"""

import numpy as np
import pandas as pd
import pytest

import imputation.logreg

from ._helpers import as_array, as_bool, load_fixture, str_list

pytestmark = pytest.mark.parity

TOL = 1e-12

FIXTURE = load_fixture("l1_augment")
CASES = {case["name"]: case for case in FIXTURE["cases"]}


def target(case):
    """The target of `case` as MICE passes it: a Categorical, floats or an object array of booleans."""
    values = case["y"] if isinstance(case["y"], list) else [case["y"]]
    if case["kind"] in ("factor", "ordered"):
        return pd.Categorical(values, categories=str_list(case["levels"]), ordered=case["kind"] == "ordered")
    if case["kind"] == "logical":
        return np.array(values, dtype=object)
    return as_array(values)


def r_codes(y):
    """The codes R's augment sees, 0-based, -1 for NA."""
    if isinstance(y, pd.Categorical):
        return np.asarray(y.codes, dtype=int)
    missing = pd.isna(y)
    codes = np.full(len(y), -1)
    codes[~missing] = np.asarray(y[~missing], dtype=float).astype(int)
    return codes


def labels(y, codes):
    """Codes of `y`'s kind back as the labels R writes (None for -1)."""
    if isinstance(y, pd.Categorical):
        return [None if c < 0 else str(y.categories[c]) for c in codes]
    if y.dtype == object:  # logical
        return [None if c < 0 else bool(c) for c in codes]
    return [None if c < 0 else float(c) for c in codes]


def _inputs(case):
    """(y, ry, wy, x) of a case; wy is ``!ry`` where R used the default."""
    y = target(case)
    ry = as_bool(case["ry"])
    wy = ~ry if case["wy"] is None else as_bool(case["wy"])
    x = np.empty((len(ry), 0)) if case["p"] == 0 else as_array(case["x"]).reshape(len(ry), case["p"])
    return y, ry, wy, x


def replay(case):
    """Run the port's augment on the codes R sees; return (y, augmented)."""
    y, ry, wy, x = _inputs(case)
    return y, imputation.logreg.augment(r_codes(y), ry, x, wy)


def check_augmented(case, y, aug):
    """Rows and weights within 1e-12 of R's, targets and masks equal."""
    expected = case["augmented"]
    n_rows = len(expected["w"])
    np.testing.assert_allclose(aug.weights, as_array(expected["w"]), rtol=TOL, atol=TOL)
    if case["p"]:
        np.testing.assert_allclose(aug.x, as_array(expected["x"]).reshape(n_rows, case["p"]), rtol=TOL, atol=TOL)
    else:
        assert aug.x.shape == (n_rows, 0)
    assert labels(y, aug.y) == [None if v is None else _plain(case, v) for v in expected["y"]]
    np.testing.assert_array_equal(aug.id_obs, as_bool(expected["ry"]))
    np.testing.assert_array_equal(aug.id_mis, as_bool(expected["wy"]))


def _plain(case, value):
    if case["kind"] in ("factor", "ordered"):
        return str(value)
    return bool(value) if case["kind"] == "logical" else float(value)


def check_levels(case, y):
    """The level order R leaves a factor in after augmenting it is the order
    of `target_categories` (all declared levels of an ordered factor)."""
    categories = imputation.logreg.target_categories(y, keep_unused=case["kind"] == "ordered")
    assert [str(level) for level in categories.levels] == str_list(case["augmented"]["levels"])


def check_case(case):
    y, aug = replay(case)
    check_augmented(case, y, aug)


AUGMENTED = [name for name, case in CASES.items() if case["r_error"] is None]
# factors that augment rebuilds (it returns the others unchanged, declared levels included)
REBUILT_FACTORS = [
    name
    for name in AUGMENTED
    if CASES[name]["kind"] in ("factor", "ordered") and len(CASES[name]["augmented"]["w"]) > len(CASES[name]["ry"])
]


@pytest.mark.parametrize("name", AUGMENTED)
def test_augmented_rows_match_r(name):
    check_case(CASES[name])


@pytest.mark.parametrize("name", REBUILT_FACTORS)
def test_level_order_after_augmentation_matches_r(name):
    case = CASES[name]
    y, _ = replay(case)
    check_levels(case, y)


def test_fixture_covers_the_planned_cases():
    assert set(CASES) >= {
        "levels2",
        "levels3",
        "levels4",
        "numeric01",
        "logical",
        "rare_dummy",
        "no_predictor",
        "one_missing",
        "unsorted_labels",
        "unused_unordered",
        "unused_ordered",
        "imputed_only",
        "too_many",
    }


@pytest.mark.parametrize("name", ["no_predictor", "one_missing"])
def test_skipped_augmentation_returns_the_input(name):
    """Without predictors, or with exactly one entry not observed, R returns
    the data unchanged with weights 1."""
    case = CASES[name]
    y, aug = replay(case)
    assert len(aug.y) == len(case["ry"])
    np.testing.assert_array_equal(aug.weights, 1.0)
    check_augmented(case, y, aug)


def test_clamping_binds_for_a_rare_dummy():
    """mean - sd / 2 of the dummy lies below its minimum 0: R clamps it."""
    case = CASES["rare_dummy"]
    _, ry, _, x = _inputs(case)
    dummy = x[:, 1]
    assert dummy.mean() - dummy.std(ddof=1) / 2 < 0
    _, aug = replay(case)
    assert aug.x[len(ry) :, 1].min() == 0.0


def test_category_only_in_imputed_cells_is_augmented():
    """The categories present in all of y count, also the current imputations."""
    case = CASES["imputed_only"]
    y, aug = replay(case)
    observed = {y[i] for i in np.flatnonzero(as_bool(case["ry"]))}
    assert "c" not in observed
    added = labels(y, aug.y[len(case["ry"]) :])
    assert added.count("c") == 2 * case["p"]


def test_more_than_fifty_categories_raise():
    case = CASES["too_many"]
    assert case["r_error"] == "Maximum number of categories (50) exceeded"
    with pytest.raises(ValueError, match=r"51 categories, more than the maximum of 50"):
        replay(case)
