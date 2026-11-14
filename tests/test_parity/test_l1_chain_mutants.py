"""Tests of the chain tests: introduce a defect into the chain loop, check that it was applied and that l1_chain fails.

Each mutant patches one function of ``imputation.MICE`` for one test (it is
undone afterwards) and counts where it was reached, so that a test cannot
pass because the mutant was never applied. Some defects sit inside
``MICE._iterate`` and not in a helper of their own; their mutant is
``MICE._iterate`` compiled from its own source with the lines of the defect
put in (`_edit_iterate`). The comparisons are those of test_l1_chain
(`_l1_chain.CHECKS`), run on the named case:

* rows with incomplete predictors dropped before the imputer, as MICE 0.3
  did: the spy receives fewer rows than R hands its imputer, and logreg's
  augmentation takes its statistics over fewer rows, which changes a label;
* rows in neither mask dropped before the imputer: the spy receives fewer
  rows. Under the base cases (the missing cells imputed, no row ignored) the
  rows in neither mask are exactly those with incomplete predictors, so this
  and the mutant above drop the same rows; the cases of phase 5 with
  ``ignore`` (ignored rows have complete predictors) and with missing cells
  left out of ``where`` tell them apart;
* ``cx`` taken after ``_remove_lindep``: ``_remove_lindep`` gets every
  observed row, whatever its predictors, so a predictor that is missing in
  such a row is dropped, and ``cx`` and the masks are taken from the
  predictors left. The spy receives other predictors and other masks;
* a where-cell outside ``cx`` keeping its start value instead of NaN
  (``_store_imputations``);
* the working column not refreshed after a visit
  (``_refresh_working_column``): the next columns are predicted from the
  start values;
* ``_check_df`` run in every iteration instead of the first: "df set to 1"
  is logged again;
* ``continue_chains`` with a fresh generator, and with the iteration counter
  restarted (events numbered from 1 again, "df set to 1" logged again, the
  rows of the chain statistics overwritten).

Not visible yet: chain statistics over the missing where-cells only instead
of all where-cells. While ``where`` is the missing cells the two are the
same cells; the cases of phase 5 with observed where-cells (``where="all"``)
will show it. Chain statistics taken right after each visit from the
imputer's output, instead of from ``imp`` at the end of the iteration, are
the same values as long as no column is visited twice in an iteration and
no ``post`` changes the imputations; the ``post`` cases and the visit
sequence with a repeated column of phase 6 will show it.
"""

import __future__

import importlib
import inspect
import sys
import textwrap

import numpy as np
import pytest

from imputation import MICE

from ._l1_chain import (
    CASES,
    check_calls,
    check_chain_statistics,
    check_completed,
    check_draws,
    check_imputations,
    check_iteration,
    check_logged_events,
    run_case,
)

pytestmark = pytest.mark.parity

MICE_MODULE = importlib.import_module("imputation.MICE")


def _edit_iterate(monkeypatch, old, new):
    """Make ``MICE._iterate`` its own source with the lines `old` replaced by `new`.

    `old` must occur once in the dedented source (the loop body is indented
    by 8 spaces there). `new` may call ``_reached()``; returns the list that
    collects those calls.
    """
    reached = []
    source = textwrap.dedent(inspect.getsource(MICE._iterate))
    assert source.count(old) == 1, f"mutant not applied: MICE._iterate has no line {old!r}"
    namespace = dict(vars(MICE_MODULE), _reached=lambda: reached.append(1))
    code = compile(source.replace(old, new), MICE_MODULE.__file__, "exec", flags=__future__.annotations.compiler_flag)
    exec(code, namespace)
    monkeypatch.setattr(MICE, "_iterate", namespace["_iterate"])
    return reached


# Where the imputer's target is taken: the rows it receives can be cut there
_TARGET = "        y = _target_values(working[col])\n"


def test_rows_with_incomplete_predictors_dropped_as_in_0_3_are_caught(monkeypatch):
    reached = _edit_iterate(
        monkeypatch,
        _TARGET,
        _TARGET
        + "        if not cx.all():\n"
        + "            _reached()\n"
        + "            y, x_values, ry, wy = y[cx], x_values[cx], ry[cx], wy[cx]\n",
    )
    case = CASES["spy_empty_method_predictor"]
    run = run_case(case)
    assert reached, "mutant not applied: no column had rows with incomplete predictors"
    assert len(run.calls[0].y) == 17  # 25 rows, 8 with hyp missing
    with pytest.raises(AssertionError, match=r"call 1 \(bmi, iteration 1, imputation 1\): ry"):
        check_calls(case, run)

    # logreg: augment() takes the standard deviation of x1 over fewer rows,
    # which moves the rows it adds; the slope, the probabilities and a label
    # of the first iteration change. The draws are the same.
    case = CASES["logreg_empty_method_predictor"]
    run = run_case(case)
    with pytest.raises(AssertionError, match=r"step 1, chain 1: imputations of 'g': labels"):
        check_imputations(case, run)
    check_draws(case, run)


def test_rows_in_neither_mask_dropped_are_caught(monkeypatch):
    reached = _edit_iterate(
        monkeypatch,
        _TARGET,
        _TARGET
        + "        used = ry | wy\n"
        + "        if not used.all():\n"
        + "            _reached()\n"
        + "            y, x_values, ry, wy = y[used], x_values[used], ry[used], wy[used]\n",
    )
    case = CASES["spy_empty_method_predictor"]
    run = run_case(case)
    assert reached, "mutant not applied: no column had rows in neither mask"
    with pytest.raises(AssertionError, match=r"call 1 \(bmi, iteration 1, imputation 1\): ry"):
        check_calls(case, run)


def test_cx_taken_after_remove_lindep_is_caught(monkeypatch):
    """chl is predicted by age, bmi and hyp; bmi and hyp are missing in row 6,
    where chl is observed. Screened on all observed rows, both are dropped,
    and chl is imputed from age alone, in every row."""
    reached = _edit_iterate(
        monkeypatch,
        "        keep, event = _remove_lindep(x, working[col], ry)\n",
        "        keep, event = _remove_lindep(x, working[col], _fit_rows(np.ones_like(cx), observed, self._ignore))\n"
        + "        if (np.isnan(x_values[:, keep]).any(axis=1) == cx).any():\n"
        + "            _reached()\n"
        + "        cx = ~np.isnan(x_values[:, keep]).any(axis=1)\n"
        + "        ry = _fit_rows(cx, observed, self._ignore)\n"
        + "        wy = _rows_to_impute(cx, where)\n",
    )
    case = CASES["spy_empty_method_predictor"]
    run = run_case(case)
    assert reached, "mutant not applied: no cx changed"
    with pytest.raises(AssertionError, match=r"call 2 \(chl, iteration 1, imputation 1\): the predictors \['age'\]"):
        check_calls(case, run)
    assert run.calls[1].wy.sum() == 10  # every missing cell of chl, R's 3
    # in iteration 2 bmi is predicted from those values of chl
    with pytest.raises(AssertionError, match=r"step 1, chain 1: imputations of 'bmi'"):
        check_imputations(case, run)


def test_start_value_kept_outside_cx_is_caught(monkeypatch):
    calls = []

    def keeping_start_values(imp, imputable, values):
        calls.append(not imputable.all())
        if imputable.any():
            imp.iloc[np.flatnonzero(imputable)] = values

    monkeypatch.setattr(MICE_MODULE, "_store_imputations", keeping_start_values)
    case = CASES["spy_empty_method_predictor"]
    run = run_case(case)
    assert any(calls), "mutant not applied: no where-cell outside cx"
    with pytest.raises(AssertionError, match=r"step 1, chain 1: imputations of 'bmi'"):
        check_imputations(case, run)
    with pytest.raises(AssertionError, match=r"step 1, chain 1: completed data, 'bmi'"):
        check_completed(case, run)
    with pytest.raises(AssertionError, match=r"step 1: chain_mean of 'bmi'"):
        check_chain_statistics(case, run)
    case = CASES["logreg_empty_method_predictor"]
    run = run_case(case)
    with pytest.raises(AssertionError, match=r"step 1, chain 1: imputations of 'g': labels"):
        check_imputations(case, run)


def test_working_column_not_refreshed_after_a_visit_is_caught(monkeypatch):
    calls = []
    original = MICE_MODULE._refresh_working_column

    def not_after_a_visit(working, column, imp, where, observed):
        if sys._getframe(1).f_code.co_name == "_iterate":
            calls.append(1)
            return
        original(working, column, imp, where, observed)

    monkeypatch.setattr(MICE_MODULE, "_refresh_working_column", not_after_a_visit)
    case = CASES["norm_predict_nhanes"]
    run = run_case(case)
    assert calls, "mutant not applied: MICE._iterate does not call _refresh_working_column"
    # in iteration 2 bmi is predicted from the start values of hyp and chl
    with pytest.raises(AssertionError, match=r"step 1, chain 1: imputations of 'bmi'"):
        check_imputations(case, run)
    # visited chl, hyp, bmi: bmi is predicted by chl, whose start values stay in the working data
    case = CASES["spy_arabic_predictor_matrix"]
    run = run_case(case)
    with pytest.raises(AssertionError, match=r"call 3 \(bmi, iteration 1, imputation 1\): x"):
        check_calls(case, run)


def test_check_df_in_every_iteration_is_caught(monkeypatch):
    reached = _edit_iterate(
        monkeypatch,
        "        if iter_idx == 0:\n",
        "        if iter_idx > 0:\n            _reached()\n        if True:\n",
    )
    for name in ("spy_df_set_to_1", "continue_events"):
        case = CASES[name]
        run = run_case(case)
        with pytest.raises(AssertionError, match=rf"{name}, step 1: logged events"):
            check_logged_events(case, run)
        check_imputations(case, run)  # it only logs
    assert reached, "mutant not applied: no second iteration"


def test_chain_statistics_over_the_missing_cells_only_are_not_seen_before_where(monkeypatch):
    """The where-cells are the missing cells until phase 5 adds ``where``."""
    calls = []
    original = MICE._record_chain_statistics

    def missing_cells_only(self, imp, iter_idx, chain_idx):
        calls.append(1)
        missing = {col: values[self.id_mis[col].loc[values.index].to_numpy()] for col, values in imp.items()}
        original(self, missing, iter_idx, chain_idx)

    monkeypatch.setattr(MICE, "_record_chain_statistics", missing_cells_only)
    for name in ("spy_empty_method_predictor", "boys_mean_norm_predict"):
        case = CASES[name]
        check_chain_statistics(case, run_case(case))
    assert calls, "mutant not applied: MICE._iterate does not call _record_chain_statistics"


# ---------------------------------------------------------------------------
# continue_chains
# ---------------------------------------------------------------------------


def _wrapping_continue(monkeypatch, defect):
    """``continue_chains`` that applies ``defect(mice)`` first when the chains have run before
    (R's ``mice.mids``); returns the list of those calls."""
    calls = []
    original = MICE.continue_chains

    def mutant(self, maxit=1):
        if self.iteration > 0:
            calls.append(defect(self))
        return original(self, maxit)

    monkeypatch.setattr(MICE, "continue_chains", mutant)
    return calls


def test_continuing_with_a_fresh_generator_is_caught(monkeypatch):
    def fresh_generators(mice):
        for chain, state in enumerate(mice._chain_states):
            state.rng = np.random.default_rng(chain)
        return True

    calls = _wrapping_continue(monkeypatch, fresh_generators)
    case = CASES["continue_norm"]
    run = run_case(case)
    assert calls == [True], "mutant not applied"
    with pytest.raises(AssertionError, match=r"continue_norm, step 2, chain 1: imputations of 'bmi'"):
        check_imputations(case, run)
    with pytest.raises(AssertionError, match=r"continue_norm, step 2: drawn so far"):
        check_draws(case, run)


def test_continuing_with_the_iteration_counter_restarted_is_caught(monkeypatch):
    def restart(mice):
        mice.iteration = 0
        return True

    calls = _wrapping_continue(monkeypatch, restart)
    case = CASES["continue_events"]
    run = run_case(case)
    assert calls == [True], "mutant not applied"
    check_imputations(case, run)  # the same values
    with pytest.raises(AssertionError, match=r"continue_events, step 2: iteration 3, R's 5"):
        check_iteration(case, run)
    with pytest.raises(AssertionError, match=r"continue_events, step 2: logged events"):
        check_logged_events(case, run)
    with pytest.raises(AssertionError, match=r"continue_events, step 2: chain_mean of 'bmi'"):
        check_chain_statistics(case, run)
    # numbered from 1 again, and check.df run again
    events = run.steps[1].logged_events
    assert events["it"].max() == 3
    assert events["out"].str.startswith("df set to 1").sum() == 2 * case["m"]
