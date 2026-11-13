"""L1 parity: the univariate imputers called with every row, as R's sampler.univ calls them.

Reference: tests/reference/r/l1_masks.R (direct calls of mice's imputers
with the masks sampler.univ produces: every row to impute, so that the fit
rows are imputed too, as under ``where = "all"``; rows in neither mask with
an observed target, as under ``ignore``, or a missing one that ``where``
leaves out; such rows with NA in x; and direct calls of mice:::augment with
such masks).

The rows in neither mask hold decoys (predictors far outside their range,
an extreme target, a label that no fit row has), so that any use of them
shows. Only the augmentation of logreg, polyreg and polr reads them, as in
R: its means, standard deviations and ranges over all rows, column by
column without NA, and its categories from all of y.

The cases have the formats of the per-method fixtures and are replayed and
checked with the helpers of their tests, given the inputs of this fixture: a
strict ScriptedGenerator that serves R's draws and refuses any other, the
order of the draws, every scripted value used (`replay` and the `check_*`
functions of test_l1_norm, test_l1_logreg, test_l1_polyreg, test_l1_polr
and test_l1_augment). pmm with one donor and the donor probabilities of
midastouch are replayed here as in test_l1_pmm and test_l1_midastouch.
test_l1_mutants runs these checks against a defective library.
"""

import numpy as np
import pytest

from imputation.logreg import logreg, logreg_boot
from imputation.midas import _donor_probabilities
from imputation.PMM import pmm

from . import test_l1_augment, test_l1_logreg, test_l1_norm, test_l1_polr, test_l1_polyreg
from ._helpers import ScriptedGenerator, as_array, as_bool, load_fixture

pytestmark = pytest.mark.parity

RTOL = 1e-8
ATOL = 1e-10
PATTERNS = ("overlap", "outside", "na_rows")

FIXTURE = load_fixture("l1_masks")
INPUTS = FIXTURE["inputs"]


def _by_name(cases):
    return {case["name"]: case for case in cases}


NORM = _by_name(FIXTURE["norm_cases"])
LOGREG = _by_name(FIXTURE["logreg_cases"])
POLYREG = _by_name(FIXTURE["polyreg_cases"])
POLR = _by_name(FIXTURE["polr_cases"])
PMM = _by_name(FIXTURE["pmm_cases"])
MIDAS = _by_name(FIXTURE["midastouch_cases"])
AUGMENT = _by_name(FIXTURE["augment_cases"])


def _numeric_inputs(case):
    """(y, ry, wy, x) of a case with a numeric target."""
    return test_l1_norm._inputs(case, INPUTS)


# ---------------------------------------------------------------------------
# The checks of the per-method tests
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("name", list(NORM))
def test_norm_methods_match_r(name):
    """norm, norm.nob, norm.boot, norm.predict and mean: R's imputations and draws."""
    case = NORM[name]
    imputed, rng = test_l1_norm.replay(case, INPUTS)
    test_l1_norm.check_imputations(case, imputed)
    test_l1_norm.check_draws(case, rng)


LOGREG_FITTED = [name for name, case in LOGREG.items() if case["port"] in ("match", "rebuilt")]


@pytest.mark.parametrize("name", LOGREG_FITTED)
def test_logreg_methods_match_r(name):
    """logreg and logreg.boot: labels, glm.fit, coefficients, probabilities and draws."""
    test_l1_logreg.check_case(LOGREG[name], INPUTS)


@pytest.mark.parametrize("name", list(POLYREG))
def test_polyreg_matches_r(name):
    """polyreg: labels, the multinom fit, probabilities and draws."""
    test_l1_polyreg.check_case(POLYREG[name], INPUTS)


@pytest.mark.parametrize("name", list(POLR))
def test_polr_matches_r(name):
    """polr: labels, the polr fit, probabilities and draws."""
    test_l1_polr.check_case(POLR[name], INPUTS)


def replay_pmm(case):
    """Run pmm with one donor on the inputs of `case` with R's draws of ``.norm.draw``; return (imputed, rng).

    As in test_l1_pmm, matchindex draws from the generator's own stream:
    with one donor and no ties its draws do not change the donor.
    """
    y, ry, wy, x = _numeric_inputs(case)
    rng = ScriptedGenerator(chisq=case["chisq"], normals=case["normals"])
    imputed = pmm(y, ry, x, id_mis=wy, donors=1, matchtype=case["matchtype"], rng=rng)
    return np.asarray(imputed), rng


def check_pmm(case, imputed, rng):
    """R's donors (observed values, so exactly), and R's chi-square and normals, all of them, in R's order."""
    np.testing.assert_array_equal(imputed, as_array(case["imputed"]))
    rng.assert_exhausted()
    assert rng.order == [("chisq", 1), ("normal", len(case["normals"]))]


@pytest.mark.parametrize("name", list(PMM))
def test_pmm_one_donor_matches_r(name):
    case = PMM[name]
    check_pmm(case, *replay_pmm(case))


def midas_probabilities(case):
    """The port's donor probabilities and kappa of `case`, with R's bootstrap weights."""
    y, ry, wy, x = _numeric_inputs(case)
    return _donor_probabilities(y, ry, x, wy, as_array(case["omega"]), kappa=case["midas_kappa"], outout=case["outout"])


def check_midas(case, probs, kappa):
    """R's donor probabilities (rescaled as R's sample() rescales them) and kappa."""
    expected = as_array(case["probs"])
    assert np.shape(probs) == expected.shape
    np.testing.assert_allclose(float(kappa), case["kappa"], rtol=1e-8)
    # as in test_l1_midastouch: an absolute floor of 1e-14 keeps the check relative
    np.testing.assert_allclose(probs, expected, rtol=1e-8, atol=1e-14)
    np.testing.assert_allclose(np.sum(probs, axis=0), 1.0, rtol=1e-12)


@pytest.mark.parametrize("name", list(MIDAS))
def test_midastouch_donor_probabilities_match_r(name):
    case = MIDAS[name]
    check_midas(case, *midas_probabilities(case))


@pytest.mark.parametrize("name", list(AUGMENT))
def test_augment_matches_r(name):
    """mice:::augment: the added rows, targets, weights and masks (1e-12)."""
    test_l1_augment.check_case(AUGMENT[name])


def test_fixture_covers_every_method_and_pattern():
    for method in ("norm", "norm.nob", "norm.boot", "norm.predict", "mean"):
        inputs = {c["input"] for c in NORM.values() if c["method"] == method}
        assert inputs == {f"num_{pattern}" for pattern in PATTERNS}
    for method, prefix in (("logreg", "logreg_"), ("logreg.boot", "logreg_boot_")):
        assert {f"{prefix}{pattern}" for pattern in PATTERNS} <= {
            name for name, c in LOGREG.items() if c["method"] == method
        }
    for cases, prefix in ((POLYREG, "polyreg_"), (POLR, "polr_"), (PMM, "pmm_"), (MIDAS, "midastouch_")):
        assert {f"{prefix}{pattern}" for pattern in PATTERNS} <= set(cases)
    assert set(AUGMENT) == {
        "na_one_predictor",
        "outside_only_category",
        "one_outside_not_imputed",
        "one_imputed_plus_ignored",
        "all_observed_all_imputed",
    }


# ---------------------------------------------------------------------------
# The inputs
# ---------------------------------------------------------------------------


def test_inputs_have_the_planned_masks():
    """overlap: every row imputed, the fit rows among them; outside and
    na_rows: rows in neither mask with an observed and with a missing
    target; na_rows: NA in x in one predictor only and in both, outside
    both masks only."""
    for family in ("num", "bin", "cat", "ord"):
        overlap = INPUTS[f"{family}_overlap"]
        assert all(as_bool(overlap["wy"])) and 0 < sum(as_bool(overlap["ry"])) < len(overlap["ry"])
        for pattern in ("outside", "na_rows"):
            inp = INPUTS[f"{family}_{pattern}"]
            ry, wy = as_bool(inp["ry"]), as_bool(inp["wy"])
            neither = ~ry & ~wy
            assert not (ry & wy).any()
            missing_y = np.array([v is None for v in inp["y"]])
            assert (neither & missing_y).sum() == 3 and (neither & ~missing_y).sum() == 3
            x = as_array(inp["x"]).reshape(len(ry), inp["p"])
            nan_rows = np.isnan(x)
            assert not nan_rows[ry | wy].any()
            if pattern == "na_rows":
                assert sorted(nan_rows.sum(axis=1)[nan_rows.any(axis=1)].tolist()) == [1, 1, 2]
            else:
                assert not nan_rows.any()


@pytest.mark.parametrize("name", [n for n in NORM if not n.endswith("overlap")])
def test_rows_in_neither_mask_do_not_change_the_norm_methods(name):
    """Without the rows in neither mask, the same draws give the same
    imputations (the R script shows the same for R)."""
    case = NORM[name]
    y, ry, wy, x = _numeric_inputs(case)
    keep = ry | wy
    small = {**INPUTS[case["input"]], "y": y[keep].tolist(), "ry": ry[keep].tolist(), "wy": wy[keep].tolist()}
    small["x"] = x[keep].tolist()
    full, _ = test_l1_norm.replay(case, INPUTS)
    without, _ = test_l1_norm.replay(case, {case["input"]: small})
    np.testing.assert_array_equal(without, full)


# ---------------------------------------------------------------------------
# What the cases show, and where the port differs from R
# ---------------------------------------------------------------------------


def test_logreg_boot_predicts_a_fit_row_from_its_own_predictors():
    """With overlapping masks R's logreg.boot predicts a row that is also a
    fit row from the predictors of the bootstrap row that replaced it (its
    ``dotx``); the port predicts every row from its own predictors. The fit
    and the rows that are not fit rows are R's."""
    case = LOGREG["logreg_boot_overlap"]
    assert case["port"] == "rebuilt"
    steps, _ = test_l1_logreg.replay(case, INPUTS)
    ry = as_bool(INPUTS[case["input"]]["ry"])
    r_prob = as_array(case["r_prob"])
    np.testing.assert_allclose(steps.prob[~ry], r_prob[~ry], rtol=RTOL, atol=ATOL)
    assert np.abs(steps.prob[ry] - r_prob[ry]).max() > 0.1
    assert case["r_imputed"] != case["imputed"]
    test_l1_logreg.check_case(case, INPUTS)


def test_a_third_value_outside_the_masks_stops_logreg_but_not_logreg_boot():
    """The value 2 in an ignored row of a 0/1 target: augment counts it as a
    category, so R's logreg stops in glm.fit before any draw and the port
    raises, naming polyreg. logreg.boot does not read the row, in R and here."""
    case = LOGREG["logreg_third_value"]
    assert case["port"] == "value_error"
    assert case["r_error"] == "y values must be 0 <= y <= 1"
    y, ry, wy, x = test_l1_logreg._inputs(case, INPUTS)
    rng = ScriptedGenerator(strict=True)
    with pytest.raises(ValueError, match=r"two categories, but y has 3 \(0\.0, 1\.0, 2\.0\)\. Use polyreg"):
        logreg(y, ry, x, id_mis=wy, rng=rng)
    assert rng.calls == []
    boot = LOGREG["logreg_boot_third_value"]
    assert boot["port"] == "match" and boot["levels"] == [0, 1]
    imputed = logreg_boot(y, ry, x, id_mis=wy, rng=test_l1_logreg._scripted(boot))
    assert imputed.tolist() == boot["imputed"]


def test_category_only_outside_the_fit_rows_is_augmented_and_fitted():
    """A category that no fit row holds enters through the added rows, as in R."""
    case = LOGREG["logreg_outside_category"]
    y, ry, *_ = test_l1_logreg._inputs(case, INPUTS)
    assert set(y[ry]) == {"no"} and case["levels"] == ["no", "yes"]
    steps, _ = test_l1_logreg.replay(case, INPUTS)
    assert steps.fit.converged
    cases = [POLYREG[f"polyreg_{pattern}"] for pattern in PATTERNS] + [POLR[f"polr_{pattern}"] for pattern in PATTERNS]
    for case in cases:
        y = INPUTS[case["input"]]["y"]
        ry = as_bool(INPUTS[case["input"]]["ry"])
        assert "d" not in {y[i] for i in np.flatnonzero(ry)} and "d" in y
        prob = as_array(case["prob"]).reshape(case["n_imp"], len(case["levels"]))
        assert case["levels"][3] == "d" and (prob[:, 3] > 0).all()


def test_augment_skip_rule_counts_the_rows_outside_the_fit_rows():
    """R skips the augmentation when exactly one row lies outside the fit
    rows, whatever the number of rows to impute."""
    skipped = AUGMENT["one_outside_not_imputed"]
    assert sum(not v for v in skipped["ry"]) == 1 and sum(skipped["wy"]) == 2
    assert len(skipped["augmented"]["w"]) == len(skipped["ry"])
    for name in ("one_imputed_plus_ignored", "all_observed_all_imputed"):
        case = AUGMENT[name]
        assert len(case["augmented"]["w"]) > len(case["ry"])
    assert sum(AUGMENT["one_imputed_plus_ignored"]["wy"]) == 1


def test_augment_takes_each_column_without_its_missing_values():
    """A row with NA in one predictor counts for the other: a1 has 10
    values, a2 11, and the standard deviation divides by each count minus 1."""
    case = AUGMENT["na_one_predictor"]
    _, ry, wy, x = test_l1_augment._inputs(case)
    assert (~np.isnan(x)).sum(axis=0).tolist() == [10, 11]
    _, aug = test_l1_augment.replay(case)
    added = aug.x[len(ry) :]
    block = 2 * len(case["augmented"]["levels"])  # the added rows of one predictor
    for j in range(2):
        v = x[~np.isnan(x[:, j]), j]
        expected = np.clip(v.mean() + 0.5 * v.std(ddof=1), v.min(), v.max())
        np.testing.assert_allclose(added[block * j, j], expected, rtol=1e-12)
    test_l1_augment.check_augmented(case, *test_l1_augment.replay(case))
