"""Tests of the layer-2 tests: re-introduce an old defect, check it is caught.

Each mutant is written so that it is valid once the library is fixed:

* A1: ``np.linalg.cholesky`` patched to return the transpose. With the
  draw ``beta = coef + L @ z * sigma`` this reproduces the old draw, and T1,
  T2 and T3 on F1 must each flag it.
* A9: ``imputation.PMM.matchindex`` replaced by a matcher with
  deterministic tie-breaking (lowest index first, as the KD-tree did); T1 on
  F2 must flag it.
* A6: ``cart(..., bootstrap=True)``, the old bootstrapped tree; the donor-set
  test on F1 must flag it.
* ``norm`` without the parameter draw: ``imputation.norm.norm_nob`` (least
  squares coefficients, residual sd from the fit) compared with R's ``norm``
  on F1. Ignoring the parameter uncertainty narrows the imputations in the
  cells where the coefficients are poorly determined: T1 (decile bins), T2
  (cell variance) and T3 (coefficient variance) must each flag it. Measured
  with 3,000 calls: smallest T1 p about 1e-206, largest |z| about 43 (T2) and
  100 (T3). On F4, where every missing row lies in the bulk of the observed
  ones, only T3 flags it clearly (|z| about 17; T2 about 5.5, T1 p 0.002).
* ``logreg`` without augmentation: ``imputation.logreg.augment`` patched to
  return its input with weights 1, compared with R's ``logreg`` on F6, whose
  observed rows are separated. The fit then diverges and its huge covariance
  scatters the coefficient draw, so the labels far from the boundary are
  close to a coin flip instead of nearly certain: T1 (level counts) and T3
  (coefficients of the level indicator) must each flag it. Measured with
  1,500 calls (fewer than the 3,000 of the parity test, because every
  diverging fit runs to the iteration limit): T1 p 0 (below the smallest
  double) in most cells, largest |z| about 66 (T3). On F5, which is not
  separated, it is not flagged (T1 p 8e-5, T3 |z| 4.9 with 3,000 calls).
* ``logreg`` without the coefficient draw: ``imputation.logreg._coefficient_draw``
  patched to return the fitted coefficients, compared with R's ``logreg``
  on F5. The imputations of one call no longer share a drawn slope, so the
  coefficients of the level indicator vary less across calls (T3: largest
  |z| about 50, on the log-variance). F5 puts missing rows far from the
  observed ones, where the draw also moves the label probabilities towards
  1/2, so T1 flags it too (smallest p about 1e-20). Both must flag it.
* ``polyreg`` taking the most probable category instead of a draw:
  ``imputation.polyreg._draw_categories`` patched to return the category
  with the largest probability, compared with R's ``polyreg`` on F3. Every
  call then imputes the same labels. Measured with 1,000 calls (fewer than
  the 3,000 of the parity test; the effect is large): T1 smallest p about
  1e-294 and below 1e-8 in every cell, T3 largest |z| about 284 (mean) and
  7,300 (log-variance; the coefficients no longer vary). The test asserts
  p < 1e-12 and |z| > 8.
* ``polr`` replaced by the multinomial model: the Python ``polyreg``
  compared with R's ``polr`` on F7, an ordered target drawn from a model
  that is not a proportional-odds model. The multinomial fit predicts other
  probabilities (up to 0.5 apart in R). Measured with 1,000 calls (fewer
  than the 2,000 of the parity test; the effect is large): T1 p below 1e-12
  in all 20 cells (the smallest 0), T3 largest |z| about 114 (mean) and 26
  (log-variance). The test asserts p < 1e-12 and |z| > 8.
* Fit rows taken from ``y`` instead of from ``id_obs``: pmm and cart on F8
  wrapped so that every row with an observed y is a fit row, which adds
  the 4 rows in neither mask that hold a decoy y. Measured with the calls
  of the parity test: pmm, whose regression the decoys pull, T1 p 0 in
  every cell, largest |z| about 119 (T2) and 244 (T3); cart puts two cells
  into a leaf with the decoys and draws a decoy 2,381 times in 1,500 calls,
  T1 p below 1e-7 in 18 of the 20 cells (counting the values of fit rows
  only), and 18 of the 20 donor sets differ from rpart's. pmm must be
  flagged by T1, T2 and T3, cart by T1 and the donor-set test (its T2 and
  T3 are not computed when a decoy is drawn).
* A fit row to impute excluded from its own donors: pmm on F8 with
  ``imputation.PMM.matchindex`` drawing the donor of each such cell again
  from the other fit rows. R draws the own value in 10 to 20 percent of the
  calls (the unpatched matcher picked it in 4,067 of the 24,000 draws for
  the 8 cells). Measured with 3,000 calls: T1 p below 1e-69 in the 8 cells
  (the smallest 0) and 0.03 or more in the others, largest |z| about 35
  (T2) and 14 (T3). T1, T2 and T3 must each flag it, T1 in those 8 cells
  and no other.
"""

import functools

import numpy as np
import pytest

from ._l2_harness import compare, donor_set_mismatches, imputer, r_reference, run_python
from ._l2_stats import P_MIN

pytestmark = pytest.mark.parity


@functools.cache
def transposed_cholesky_comparison():
    """pmm on F1 with np.linalg.cholesky returning the transposed factor."""
    calls = []
    original = np.linalg.cholesky

    def transposed(a, *args, **kwargs):
        calls.append(1)
        return np.swapaxes(original(a, *args, **kwargs), -1, -2)

    with pytest.MonkeyPatch.context() as mp:
        mp.setattr(np.linalg, "cholesky", transposed)
        inp, r_sum, py_sum = run_python("pmm", "F1")
    return compare(r_sum, py_sum, inp), len(calls)


@pytest.mark.parametrize("statistic", ["T1", "T2", "T3"])
def test_transposed_cholesky_is_flagged(statistic):
    comp, n_calls = transposed_cholesky_comparison()
    assert n_calls > 0, "mutant not applied: pmm no longer calls np.linalg.cholesky"
    passed = {"T1": comp.t1_ok, "T2": comp.t2_ok, "T3": comp.t3_ok}[statistic]()
    message = {"T1": comp.t1_message, "T2": comp.t2_message, "T3": comp.t3_message}
    assert not passed, f"{statistic} does not flag the mutant: " + message[statistic]()


def deterministic_matchindex(d, t, k=5, rng=None):
    """k nearest donors with ties broken by position (KD-tree behaviour)."""
    d = np.asarray(d, dtype=float).ravel()
    t = np.asarray(t, dtype=float).ravel()
    rng = np.random.default_rng() if rng is None else rng
    k = int(min(max(k, 1), len(d)))
    nearest = np.argsort(np.abs(d[None, :] - t[:, None]), axis=1, kind="stable")[:, :k]
    return nearest[np.arange(len(t)), rng.integers(k, size=len(t))]


def test_deterministic_ties_are_flagged_by_t1(monkeypatch):
    import imputation.PMM as pmm_module

    calls = []

    def counting(*args, **kwargs):
        calls.append(1)
        return deterministic_matchindex(*args, **kwargs)

    monkeypatch.setattr(pmm_module, "matchindex", counting)
    inp, r_sum, py_sum = run_python("pmm", "F2")
    comp = compare(r_sum, py_sum, inp)
    assert calls, "mutant not applied: pmm does not call imputation.PMM.matchindex"
    assert not comp.t1_ok(), "T1 does not flag deterministic ties: " + comp.t1_message()


@functools.cache
def norm_without_parameter_draw_comparison():
    """Python norm_nob in place of norm, against R's norm on F1."""
    from imputation.norm import norm_nob

    calls = []

    def counting(*args, **kwargs):
        calls.append(1)
        return norm_nob(*args, **kwargs)

    inp, r_sum, py_sum = run_python("norm", "F1", func=counting)
    return compare(r_sum, py_sum, inp), len(calls)


@pytest.mark.parametrize("statistic", ["T1", "T2", "T3"])
def test_norm_without_parameter_draw_is_flagged(statistic):
    comp, n_calls = norm_without_parameter_draw_comparison()
    assert n_calls > 0, "mutant not applied: norm_nob was not called"
    assert comp.kind == "continuous"
    passed = {"T1": comp.t1_ok, "T2": comp.t2_ok, "T3": comp.t3_ok}[statistic]()
    message = {"T1": comp.t1_message, "T2": comp.t2_message, "T3": comp.t3_message}
    assert not passed, f"{statistic} does not flag the mutant: " + message[statistic]()


@functools.cache
def logreg_without_augmentation_comparison():
    """logreg on F6 with imputation.logreg.augment returning its input, weights 1."""
    import imputation.logreg as logreg_module

    calls = []

    def no_augmentation(y, id_obs, x, id_mis, *args, **kwargs):
        calls.append(1)
        y = np.asarray(y, dtype=np.intp).ravel()
        return logreg_module.Augmented(
            y=y,
            id_obs=np.asarray(id_obs, dtype=bool),
            x=np.asarray(x, dtype=float),
            id_mis=np.asarray(id_mis, dtype=bool),
            weights=np.ones(len(y)),
        )

    with pytest.MonkeyPatch.context() as mp:
        mp.setattr(logreg_module, "augment", no_augmentation)
        inp, r_sum, py_sum = run_python("logreg", "F6", n_calls=1500)
    return compare(r_sum, py_sum, inp), len(calls)


@pytest.mark.parametrize("statistic", ["T1", "T3"])
def test_logreg_without_augmentation_is_flagged(statistic):
    comp, n_calls = logreg_without_augmentation_comparison()
    assert n_calls > 0, "mutant not applied: logreg no longer calls imputation.logreg.augment"
    assert comp.kind == "category"
    passed = {"T1": comp.t1_ok, "T3": comp.t3_ok}[statistic]()
    message = {"T1": comp.t1_message, "T3": comp.t3_message}
    assert not passed, f"{statistic} does not flag the mutant: " + message[statistic]()


@functools.cache
def logreg_without_coefficient_draw_comparison():
    """logreg on F5 with imputation.logreg._coefficient_draw returning the fitted coefficients."""
    import imputation.logreg as logreg_module

    calls = []

    def fitted_coefficients(fit, rng):
        calls.append(1)
        return np.where(fit.aliased, 0.0, fit.coef)

    with pytest.MonkeyPatch.context() as mp:
        mp.setattr(logreg_module, "_coefficient_draw", fitted_coefficients)
        inp, r_sum, py_sum = run_python("logreg", "F5")
    return compare(r_sum, py_sum, inp), len(calls)


@pytest.mark.parametrize("statistic", ["T1", "T3"])
def test_logreg_without_coefficient_draw_is_flagged(statistic):
    comp, n_calls = logreg_without_coefficient_draw_comparison()
    assert n_calls > 0, "mutant not applied: logreg no longer calls imputation.logreg._coefficient_draw"
    assert comp.kind == "category"
    passed = {"T1": comp.t1_ok, "T3": comp.t3_ok}[statistic]()
    message = {"T1": comp.t1_message, "T3": comp.t3_message}
    assert not passed, f"{statistic} does not flag the mutant: " + message[statistic]()


def test_bootstrapped_cart_is_flagged_by_donor_sets():
    inp, r_sum, py_sum = run_python("cart", "F1", bootstrap=True, cp=1e-4, min_samples_leaf=5)
    assert donor_set_mismatches(r_sum, py_sum), "the donor-set test does not flag a bootstrapped tree"


@functools.cache
def polyreg_most_probable_category_comparison():
    """polyreg on F3 with imputation.polyreg._draw_categories returning the most probable category."""
    import imputation.polyreg as polyreg_module

    calls = []

    def most_probable(u, prob):
        calls.append(1)
        return np.argmax(prob, axis=1)

    with pytest.MonkeyPatch.context() as mp:
        mp.setattr(polyreg_module, "_draw_categories", most_probable)
        inp, r_sum, py_sum = run_python("polyreg", "F3", n_calls=1000)
    return compare(r_sum, py_sum, inp), len(calls)


def test_polyreg_taking_the_most_probable_category_is_flagged_by_t1():
    comp, n_calls = polyreg_most_probable_category_comparison()
    assert n_calls > 0, "mutant not applied: polyreg no longer calls imputation.polyreg._draw_categories"
    assert comp.kind == "category"
    assert comp.t1_min_p < 1e-12, "T1 does not flag the mutant clearly: " + comp.t1_message()


def test_polyreg_taking_the_most_probable_category_is_flagged_by_t3():
    comp, n_calls = polyreg_most_probable_category_comparison()
    assert n_calls > 0, "mutant not applied: polyreg no longer calls imputation.polyreg._draw_categories"
    assert np.abs(comp.t3_z_mean).max() > 8, "T3 (mean) does not flag the mutant clearly: " + comp.t3_message()
    assert np.abs(comp.t3_z_var).max() > 8, "T3 (log-variance) does not flag the mutant clearly: " + comp.t3_message()


@functools.cache
def polr_replaced_by_polyreg_comparison():
    """The Python polyreg in place of polr, against R's polr on F7."""
    from imputation.polyreg import polyreg

    calls = []

    def counting(*args, **kwargs):
        calls.append(1)
        return polyreg(*args, **kwargs)

    inp, r_sum, py_sum = run_python("polr", "F7", func=counting, n_calls=1000)
    return compare(r_sum, py_sum, inp), len(calls)


def test_polr_replaced_by_polyreg_is_flagged_by_t1():
    comp, n_calls = polr_replaced_by_polyreg_comparison()
    assert n_calls > 0, "mutant not applied: polyreg was not called"
    assert comp.kind == "category"
    assert comp.t1_min_p < 1e-12, "T1 does not flag the mutant clearly: " + comp.t1_message()


def test_polr_replaced_by_polyreg_is_flagged_by_t3():
    comp, n_calls = polr_replaced_by_polyreg_comparison()
    assert n_calls > 0, "mutant not applied: polyreg was not called"
    assert np.abs(comp.t3_z_mean).max() > 8, "T3 (mean) does not flag the mutant clearly: " + comp.t3_message()
    assert np.abs(comp.t3_z_var).max() > 8, "T3 (log-variance) does not flag the mutant clearly: " + comp.t3_message()


def fit_rows_from_y(func, added):
    """``func`` with its fit rows taken from ``y`` (every row with an
    observed value) instead of from ``id_obs``. Appends the number of rows
    this adds to ``added``, once per call."""

    def defective(y, id_obs, x, id_mis=None, **kwargs):
        observed = ~np.isnan(np.asarray(y, dtype=float))
        added.append(int((observed & ~id_obs).sum()))
        return func(y, id_obs | observed, x, id_mis=id_mis, **kwargs)

    return defective


@functools.cache
def fit_rows_from_y_run(method):
    """pmm or cart on F8 with the rows that hold an observed y used as fit rows."""
    added = []
    inp, r_sum, py_sum = run_python(method, "F8", func=fit_rows_from_y(imputer(method), added))
    return compare(r_sum, py_sum, inp), r_sum, py_sum, added


@pytest.mark.parametrize("statistic", ["T1", "T2", "T3"])
def test_pmm_with_fit_rows_from_y_is_flagged(statistic):
    comp, _, _, added = fit_rows_from_y_run("pmm")
    assert added and set(added) == {4}, f"mutant not applied as planned: rows added per call {sorted(set(added))}"
    passed = {"T1": comp.t1_ok, "T2": comp.t2_ok, "T3": comp.t3_ok}[statistic]()
    message = {"T1": comp.t1_message, "T2": comp.t2_message, "T3": comp.t3_message}
    assert not passed, f"{statistic} does not flag the mutant: " + message[statistic]()


def test_cart_with_fit_rows_from_y_is_flagged_by_t1():
    comp, _, _, added = fit_rows_from_y_run("cart")
    assert added and set(added) == {4}, f"mutant not applied as planned: rows added per call {sorted(set(added))}"
    assert not comp.t1_ok(), "T1 does not flag the mutant: " + comp.t1_message()


def test_cart_with_fit_rows_from_y_is_flagged_by_donor_sets():
    _, r_sum, py_sum, added = fit_rows_from_y_run("cart")
    assert added and set(added) == {4}, f"mutant not applied as planned: rows added per call {sorted(set(added))}"
    assert donor_set_mismatches(r_sum, py_sum), "the donor-set test does not flag the mutant"


@functools.cache
def pmm_without_own_donor_comparison():
    """pmm on F8 with a matcher that never gives a fit row to impute its own value.

    The library's matcher runs as usual; then the donor of each cell whose
    row is a fit row is drawn again, by the same matcher, from the other fit
    rows. Returns the comparison, the number of matcher calls, the number of
    draws in which the unpatched matcher had picked the cell's own row, and
    the cells concerned.
    """
    import imputation.PMM as pmm_module

    inp, _ = r_reference("pmm", "F8")
    fit_rows = np.flatnonzero(inp.ry)
    # per cell, the position of its row among the fit rows (the donors), -1 if it is not a fit row
    own = np.array([np.searchsorted(fit_rows, row) if inp.ry[row] else -1 for row in np.flatnonzero(inp.wy)])
    cells = np.flatnonzero(own >= 0)
    original = pmm_module.matchindex
    calls, own_drawn = [], []

    def without_own_donor(d, t, k=5, rng=None):
        calls.append(1)
        d, t = np.asarray(d, dtype=float), np.asarray(t, dtype=float)
        assert (len(d), len(t)) == (len(fit_rows), len(own)), "pmm no longer matches the cells of wy with the fit rows"
        idx = original(d, t, k=k, rng=rng)
        for j in cells:
            own_drawn.append(int(idx[j] == own[j]))
            others = np.delete(np.arange(len(d)), own[j])
            idx[j] = others[original(d[others], t[j : j + 1], k=k, rng=rng)[0]]
        return idx

    with pytest.MonkeyPatch.context() as mp:
        mp.setattr(pmm_module, "matchindex", without_own_donor)
        inp, r_sum, py_sum = run_python("pmm", "F8")
    return compare(r_sum, py_sum, inp), len(calls), sum(own_drawn), cells


def _check_own_donor_mutant_applied(n_calls, n_own_drawn, cells):
    assert n_calls > 0, "mutant not applied: pmm no longer calls imputation.PMM.matchindex"
    assert len(cells) == 8, f"F8 has {len(cells)} fit rows to impute, expected 8"
    assert n_own_drawn > 0, "mutant not applied: the matcher never picked a cell's own row"


@pytest.mark.parametrize("statistic", ["T1", "T2", "T3"])
def test_pmm_without_own_donor_is_flagged(statistic):
    comp, *applied = pmm_without_own_donor_comparison()
    _check_own_donor_mutant_applied(*applied)
    passed = {"T1": comp.t1_ok, "T2": comp.t2_ok, "T3": comp.t3_ok}[statistic]()
    message = {"T1": comp.t1_message, "T2": comp.t2_message, "T3": comp.t3_message}
    assert not passed, f"{statistic} does not flag the mutant: " + message[statistic]()


def test_pmm_without_own_donor_is_flagged_by_t1_in_the_fit_rows_to_impute_only():
    """The mutant changes the cells of the 8 fit rows to impute and no other."""
    comp, *applied = pmm_without_own_donor_comparison()
    _check_own_donor_mutant_applied(*applied)
    cells = applied[2]
    flagged = comp.t1_p < P_MIN
    assert flagged[cells].all(), f"T1 p-values of the fit rows to impute: {comp.t1_p[cells]}"
    others = np.delete(np.arange(len(flagged)), cells)
    assert not flagged[others].any(), f"T1 p-values of the other cells: {comp.t1_p[others]}"
