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
  and the mutant above drop the same rows; in ``spy_left_out`` missing
  cells of bmi left out of ``where`` put rows with complete predictors in
  neither mask, which only this mutant drops;
* ``cx`` taken after ``_remove_lindep``: ``_remove_lindep`` gets every
  observed row, whatever its predictors, so a predictor that is missing in
  such a row is dropped, and ``cx`` and the masks are taken from the
  predictors left. The spy receives other predictors and other masks;
* a where-cell outside ``cx`` keeping its start value instead of NaN
  (``_store_imputations``), for missing cells and, in
  ``incomplete_predictors``, for an observed where-cell, which then keeps a
  value where R's completed data lose the observed one;
* the working column not refreshed after a visit
  (``_refresh_working_column``): the next columns are predicted from the
  start values;
* ``_check_df`` run in every iteration instead of the first: "df set to 1"
  is logged again;
* ``continue_chains`` with a fresh generator, and with the iteration counter
  restarted (events numbered from 1 again, "df set to 1" logged again, the
  rows of the chain statistics overwritten).

``where``:

* ``where`` not used, only the missing cells imputed (``check_where``):
  MICE then has other where-cells than R and cannot take R's start, which
  `_l1_chain.run_case` compares first (the columns and rows of the start);
* the completed data built from the working data (``MICE._impute_once``):
  observed where-cells keep their observed values;
* the imputations of observed where-cells written to the working data
  (``_refresh_working_column`` called by ``MICE._iterate``): the next column
  is predicted from them (``x`` of the spy);
* the chain statistics over the missing where-cells only
  (``MICE._record_chain_statistics``): a complete column under
  ``where="all"`` has none;
* method ``""`` by completeness instead of by where-cells (``make_method``),
  ``"monotone"`` by the number of missing values (``_monotone``) and the
  monotone edit of the predictor matrix left out
  (``_monotone_predictor_matrix``): the comparisons of test_l1_setup.

``ignore``:

* ``ignore`` not applied to the fit rows (``_fit_rows``): the spy receives
  other fit rows, and ``norm`` draws its chi-square with other degrees of
  freedom;
* ``ignore`` applied to the rows to impute: the where-cells of ignored rows
  are not imputed;
* ``_check_df`` or ``_remove_lindep`` given the fit rows without ``ignore``:
  "df set to 1" is not logged, and a predictor that is constant in the rows
  that are not ignored is kept;
* rows in neither mask dropped before the imputer, here ignored rows with an
  observed target: the spy receives fewer rows, and where a category of a
  ``logreg`` target is held only by ignored rows, it is not a category any
  more, so nothing is drawn.

Not visible yet: chain statistics taken right after each visit from the
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

from ._helpers import decode_frame
from ._l1_chain import (
    CASES,
    check_calls,
    check_chain_statistics,
    check_completed,
    check_draws,
    check_imputations,
    check_iteration,
    check_logged_events,
    check_lost_observed_values,
    run_case,
)
from .test_l1_setup import MONOTONE, WHERE, check_monotone_case, check_where_methods, check_where_visit_sequences

pytestmark = pytest.mark.parity

MICE_MODULE = importlib.import_module("imputation.MICE")


def _edit_iterate(monkeypatch, old, new, *more):
    """Make ``MICE._iterate`` its own source with the lines `old` replaced by `new`.

    `old` must occur once in the dedented source (the loop body is indented
    by 8 spaces there). `new` may call ``_reached()``; returns the list that
    collects those calls. `more` holds further pairs ``(old, new)``, replaced
    in the same way.
    """
    reached = []
    source = textwrap.dedent(inspect.getsource(MICE._iterate))
    for old_lines, new_lines in ((old, new), *more):
        assert source.count(old_lines) == 1, f"mutant not applied: MICE._iterate has no line {old_lines!r}"
        source = source.replace(old_lines, new_lines)
    namespace = dict(vars(MICE_MODULE), _reached=lambda: reached.append(1))
    code = compile(source, MICE_MODULE.__file__, "exec", flags=__future__.annotations.compiler_flag)
    exec(code, namespace)
    monkeypatch.setattr(MICE, "_iterate", namespace["_iterate"])
    return reached


# Where the imputer's target is taken: the rows it receives can be cut there
_TARGET = "        y = _target_values(working[col])\n"
# Where the imputations are stored, by the where-cells inside wy (R's cc). A
# mutant that cuts the rows passed to the imputer keeps the wy of all rows
# for it (all_wy)
_STORE = (
    "        _store_imputations(imp, wy[where], imputed_values)\n",
    "        _store_imputations(imp, all_wy[where], imputed_values)\n",
)


def test_rows_with_incomplete_predictors_dropped_as_in_0_3_are_caught(monkeypatch):
    reached = _edit_iterate(
        monkeypatch,
        _TARGET,
        _TARGET
        + "        all_wy = wy\n"
        + "        if not cx.all():\n"
        + "            _reached()\n"
        + "            y, x_values, ry, wy = y[cx], x_values[cx], ry[cx], wy[cx]\n",
        _STORE,
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

    # only row 6 of bmi has an incomplete predictor here; the rows in neither mask with complete predictors
    # stay (see test_rows_in_neither_mask_with_complete_predictors_dropped_are_caught)
    run = run_case(CASES["spy_left_out"])
    assert len(run.calls[0].y) == 24


def test_rows_in_neither_mask_dropped_are_caught(monkeypatch):
    reached = _edit_iterate(
        monkeypatch,
        _TARGET,
        _TARGET
        + "        all_wy, used = wy, ry | wy\n"
        + "        if not used.all():\n"
        + "            _reached()\n"
        + "            y, x_values, ry, wy = y[used], x_values[used], ry[used], wy[used]\n",
        _STORE,
    )
    case = CASES["spy_empty_method_predictor"]
    run = run_case(case)
    assert reached, "mutant not applied: no column had rows in neither mask"
    with pytest.raises(AssertionError, match=r"call 1 \(bmi, iteration 1, imputation 1\): ry"):
        check_calls(case, run)


def test_rows_in_neither_mask_with_complete_predictors_dropped_are_caught(monkeypatch):
    """bmi is left out of where in rows 1 and 3 (R's numbers), where its predictors are complete: those rows are in
    neither mask, and the mutant of 0.3 (rows with incomplete predictors dropped) keeps them."""
    reached = _edit_iterate(
        monkeypatch,
        _TARGET,
        _TARGET
        + "        all_wy, used = wy, ry | wy\n"
        + "        if (~used & cx).any():\n"
        + "            _reached()\n"
        + "        y, x_values, ry, wy = y[used], x_values[used], ry[used], wy[used]\n",
        _STORE,
    )
    case = CASES["spy_left_out"]
    run = run_case(case)
    assert reached, "mutant not applied: no row in neither mask with complete predictors"
    # rows 1 and 3 (complete predictors) and 6 (hyp missing) of 25
    assert len(run.calls[0].y) == 22
    with pytest.raises(AssertionError, match=r"call 1 \(bmi, iteration 1, imputation 1\): ry"):
        check_calls(case, run)


def test_cx_taken_after_remove_lindep_is_caught(monkeypatch):
    """chl is predicted by age, bmi and hyp; bmi and hyp are missing in row 6,
    where chl is observed. Screened on all observed rows, both are dropped,
    and chl is imputed from age alone, in every row."""
    reached = _edit_iterate(
        monkeypatch,
        "        keep, event = _remove_lindep(x, working[col], ry)\n",
        "        keep, event = _remove_lindep(x, working[col], _fit_rows(np.ones_like(cx), observed, self.ignore))\n"
        + "        if (np.isnan(x_values[:, keep]).any(axis=1) == cx).any():\n"
        + "            _reached()\n"
        + "        cx = ~np.isnan(x_values[:, keep]).any(axis=1)\n"
        + "        ry = _fit_rows(cx, observed, self.ignore)\n"
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


def test_start_value_kept_for_an_observed_where_cell_outside_cx_is_caught(monkeypatch):
    """The observed where-cell of bmi in row 15 (R's number) has chl, left out of where, missing: R's completed
    data lose its observed value. With its start value kept MICE loses nothing there."""
    case = CASES["incomplete_predictors"]
    data = decode_frame(case["data"])
    kept_observed = []

    def keeping_start_values(imp, imputable, values):
        kept_observed.append(data.loc[imp.index[~imputable], imp.name].notna().any())
        if imputable.any():
            imp.iloc[np.flatnonzero(imputable)] = values

    monkeypatch.setattr(MICE_MODULE, "_store_imputations", keeping_start_values)
    run = run_case(case)
    assert any(kept_observed), "mutant not applied: no observed where-cell outside cx"
    with pytest.raises(AssertionError, match=r"incomplete_predictors, step 1: observed values lost per column \{\}"):
        check_lost_observed_values(case, run)
    with pytest.raises(AssertionError, match=r"incomplete_predictors, step 1, chain 1: completed data, 'bmi'"):
        check_completed(case, run)


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


# ---------------------------------------------------------------------------
# where
# ---------------------------------------------------------------------------


def test_where_not_used_is_caught(monkeypatch):
    """Only the missing cells imputed: age, complete, has no where-cell and gets no start, unlike R's."""
    calls = []

    def missing_cells(where, data):
        calls.append(where)
        return data.isna()

    monkeypatch.setattr(MICE_MODULE, "check_where", missing_cells)
    columns = r"\['bmi', 'hyp', 'chl'\], R's \['age', 'bmi', 'hyp', 'chl'\]"
    with pytest.raises(AssertionError, match=rf"all_mean_norm_predict: the start has the columns {columns}"):
        run_case(CASES["all_mean_norm_predict"])
    assert calls == ["all"], "mutant not applied"


def test_completed_data_from_the_working_data_are_caught(monkeypatch):
    """The observed where-cells keep their observed values: age is not its observed mean (R's mean imputations)."""
    calls = []

    def from_working_data(self, state, chain_idx, iterations):
        calls.append(len(iterations))
        working = self._working_frame(state.imp)
        for iter_idx in iterations:
            self._iterate(working, state, iter_idx, chain_idx)
        return working

    monkeypatch.setattr(MICE, "_impute_once", from_working_data)
    case = CASES["all_mean_norm_predict"]
    run = run_case(case)
    assert any(calls), "mutant not applied: no iteration ran"
    with pytest.raises(AssertionError, match=r"all_mean_norm_predict, step 1, chain 1: completed data, 'age'"):
        check_completed(case, run)
    check_imputations(case, run)  # imp is R's


def test_imputations_of_observed_where_cells_in_the_working_data_are_caught(monkeypatch):
    """age is visited first, its where-cells in rows 3, 10 and 20 (R's numbers) are observed: bmi, visited next,
    must be predicted from the observed age."""
    calls = []
    original = MICE_MODULE._refresh_working_column

    def every_where_cell_after_a_visit(working, column, imp, where, observed):
        if sys._getframe(1).f_code.co_name != "_iterate":
            original(working, column, imp, where, observed)
            return
        calls.append((where & observed).any())
        working.loc[where, column] = MICE_MODULE._values(imp)

    monkeypatch.setattr(MICE_MODULE, "_refresh_working_column", every_where_cell_after_a_visit)
    case = CASES["spy_observed_where_cells"]
    run = run_case(case)
    assert any(calls), "mutant not applied: no observed where-cell refreshed"
    with pytest.raises(AssertionError, match=r"call 2 \(bmi, iteration 1, imputation 1\): x"):
        check_calls(case, run)


def test_chain_statistics_over_the_missing_cells_only_are_caught(monkeypatch):
    """Under where="all" the complete age has where-cells but no missing cell: its chain mean would be NaN."""
    calls = []
    original = MICE._record_chain_statistics

    def missing_cells_only(self, imp, iter_idx, chain_idx):
        calls.append(1)
        missing = {col: values[self.id_mis[col].loc[values.index].to_numpy()] for col, values in imp.items()}
        original(self, missing, iter_idx, chain_idx)

    monkeypatch.setattr(MICE, "_record_chain_statistics", missing_cells_only)
    case = CASES["all_mean_norm_predict"]
    run = run_case(case)
    assert calls, "mutant not applied: MICE._iterate does not call _record_chain_statistics"
    with pytest.raises(AssertionError, match=r"all_mean_norm_predict, step 1: chain_mean of 'age'"):
        check_chain_statistics(case, run)
    check_completed(case, run)  # only the statistics


# ---------------------------------------------------------------------------
# ignore
# ---------------------------------------------------------------------------


def test_ignore_not_applied_to_the_fit_rows_is_caught(monkeypatch):
    """The observed bmi of the ignored rows 14, 17, 22 and 24 (R's numbers) become fit rows."""
    calls = []

    def without_ignore(cx, observed, ignore):
        calls.append(bool((cx & observed & ignore).any()))
        return cx & observed

    monkeypatch.setattr(MICE_MODULE, "_fit_rows", without_ignore)
    case = CASES["spy_ignore"]
    run = run_case(case)
    assert any(calls), "mutant not applied: no ignored fit row"
    with pytest.raises(AssertionError, match=r"spy_ignore, call 1 \(bmi, iteration 1, imputation 1\): ry"):
        check_calls(case, run)
    # norm draws its chi-square with sum(ry) - q degrees of freedom
    case = CASES["norm_ignore"]
    run = run_case(case)
    with pytest.raises(AssertionError, match=r"norm_ignore: degrees of freedom \["):
        check_draws(case, run)


def test_ignore_applied_to_the_rows_to_impute_is_caught(monkeypatch):
    """The missing bmi of the ignored rows 1 and 6 (R's numbers) is not imputed."""
    reached = _edit_iterate(
        monkeypatch,
        "        wy = _rows_to_impute(cx, where)\n",
        "        wy = _rows_to_impute(cx, where) & ~self.ignore\n"
        + "        if (_rows_to_impute(cx, where) & self.ignore).any():\n"
        + "            _reached()\n",
    )
    case = CASES["spy_ignore"]
    run = run_case(case)
    assert reached, "mutant not applied: no where-cell in an ignored row"
    with pytest.raises(AssertionError, match=r"spy_ignore, call 1 \(bmi, iteration 1, imputation 1\): wy"):
        check_calls(case, run)
    case = CASES["ignore_norm_predict_mean"]
    run = run_case(case)
    with pytest.raises(AssertionError, match=r"ignore_norm_predict_mean, step 1, chain 1: imputations of 'bmi'"):
        check_imputations(case, run)


# The fit rows as they would be without ignore
_FIT_ROWS_WITHOUT_IGNORE = "_fit_rows(cx, observed, np.zeros_like(cx))"


def test_check_df_given_the_fit_rows_without_ignore_is_caught(monkeypatch):
    """u has 5 observed values, 3 of them in rows that are not ignored: R logs "df set to 1" for 3."""
    reached = _edit_iterate(
        monkeypatch,
        "            self._check_df(x_values.shape[1], int(ry.sum()), col, method_name, chain_idx)\n",
        f"            n_rows = int({_FIT_ROWS_WITHOUT_IGNORE}.sum())\n"
        + "            if n_rows != ry.sum():\n"
        + "                _reached()\n"
        + "            self._check_df(x_values.shape[1], n_rows, col, method_name, chain_idx)\n",
    )
    case = CASES["spy_ignore_few_fit_rows"]
    run = run_case(case)
    assert reached, "mutant not applied: no ignored fit row"
    events = run.steps[0].logged_events
    assert not events["out"].str.startswith("df set to 1").any()
    with pytest.raises(AssertionError, match=r"spy_ignore_few_fit_rows, step 1: logged events"):
        check_logged_events(case, run)
    check_calls(case, run)  # it only logs


def test_remove_lindep_given_the_fit_rows_without_ignore_is_caught(monkeypatch):
    """p is 1 in every row but the ignored rows 14 and 17 (R's numbers): R drops it from the model of bmi."""
    reached = _edit_iterate(
        monkeypatch,
        "        keep, event = _remove_lindep(x, working[col], ry)\n",
        f"        keep, event = _remove_lindep(x, working[col], {_FIT_ROWS_WITHOUT_IGNORE})\n"
        + f"        if ({_FIT_ROWS_WITHOUT_IGNORE} != ry).any():\n"
        + "            _reached()\n",
    )
    case = CASES["spy_ignore_few_fit_rows"]
    run = run_case(case)
    assert reached, "mutant not applied: no ignored fit row"
    with pytest.raises(
        AssertionError,
        match=r"spy_ignore_few_fit_rows, call 1 \(bmi, iteration 1, imputation 1\): the predictors "
        r"\['age', 'hyp', 'chl', 'p'\], R's \['age', 'hyp', 'chl'\]",
    ):
        check_calls(case, run)
    with pytest.raises(AssertionError, match=r"spy_ignore_few_fit_rows, step 1: logged events"):
        check_logged_events(case, run)


def test_ignored_rows_dropped_before_the_imputer_are_caught(monkeypatch):
    """Rows in neither mask dropped, as in test_rows_in_neither_mask_dropped_are_caught, on the cases with
    ignore: the spy receives 21 of 25 rows for bmi (rows 14, 17, 22 and 24 are ignored and observed). Under
    logreg, with one cell to impute, the augmentation is skipped and a label changes. Where yes is held only by
    ignored rows, without them g has one category, so MICE draws nothing and imputes no, where R draws and
    imputes yes in the second iteration (no in the first, by the sign of the normal draw)."""
    reached = _edit_iterate(
        monkeypatch,
        _TARGET,
        _TARGET
        + "        all_wy, used = wy, ry | wy\n"
        + "        if (~used & self.ignore).any():\n"
        + "            _reached()\n"
        + "        y, x_values, ry, wy = y[used], x_values[used], ry[used], wy[used]\n",
        _STORE,
    )
    case = CASES["spy_ignore"]
    run = run_case(case)
    assert reached, "mutant not applied: no ignored row in neither mask"
    assert len(run.calls[0].y) == 21
    with pytest.raises(AssertionError, match=r"spy_ignore, call 1 \(bmi, iteration 1, imputation 1\): ry"):
        check_calls(case, run)

    # one cell to impute: without the four ignored rows it is the only row outside the fit rows, so augment()
    # is skipped, and the label of the first iteration changes; the draws are the same
    case = CASES["logreg_one_cell_ignored"]
    run = run_case(case)
    with pytest.raises(
        AssertionError,
        match=r"logreg_one_cell_ignored, step 1, chain 1: imputations of 'g': labels \['no'\] differ from R's "
        r"\['yes'\]",
    ):
        check_imputations(case, run)
    check_draws(case, run)

    case = CASES["logreg_category_only_ignored"]
    run = run_case(case)
    with pytest.raises(AssertionError, match=r"logreg_category_only_ignored, step 1: drawn so far"):
        check_draws(case, run)
    with pytest.raises(
        AssertionError, match=r"logreg_category_only_ignored, step 2, chain 1: imputations of 'g': labels"
    ):
        check_imputations(case, run)


# ---------------------------------------------------------------------------
# where in the setup (test_l1_setup)
# ---------------------------------------------------------------------------


def _setup_case(cases, name):
    return next(case for case in cases if case["name"] == name)


def test_method_by_completeness_is_caught(monkeypatch):
    """make_method without where: the complete age gets "" under where="all", R's polyreg."""
    calls = []
    original = MICE_MODULE.make_method

    def by_completeness(data, default_method, where=None):
        calls.append(where)
        return original(data, default_method)

    monkeypatch.setattr(MICE_MODULE, "make_method", by_completeness)
    # without a method, the 25 where-cells of age are missing in the completed data
    with pytest.warns(UserWarning, match=r"'age': 25 \(no method\)"):
        with pytest.raises(AssertionError, match=r"nhanes2_all: methods \{'age': ''"):
            check_where_methods(_setup_case(WHERE, "nhanes2_all"))
    assert any(where is not None for where in calls), "mutant not applied"


def test_monotone_by_the_number_of_missing_values_is_caught(monkeypatch):
    """R orders by where-cells: hyp (5) age (6) chl (12), not age (0) hyp (8) chl (10)."""
    calls = []
    original = MICE_MODULE._monotone
    case = _setup_case(WHERE, "nhanes2_matrix")
    missing = decode_frame(case["data"]).isna()

    def by_missing_values(where, columns):
        calls.append(1)
        return original(missing, columns)

    monkeypatch.setattr(MICE_MODULE, "_monotone", by_missing_values)
    with pytest.raises(AssertionError, match=r"nhanes2_matrix: visit sequence monotone \['age', 'hyp', 'chl'\]"):
        check_where_visit_sequences(case)
    assert calls, "mutant not applied"


def test_monotone_edit_left_out_is_caught(monkeypatch):
    calls = []

    def unedited(predictor_matrix, where):
        calls.append(1)
        return predictor_matrix

    monkeypatch.setattr(MICE_MODULE, "_monotone_predictor_matrix", unedited)
    with pytest.raises(AssertionError, match=r"nhanes_monotone: the predictor matrix differs from R's"):
        check_monotone_case(_setup_case(MONOTONE, "nhanes_monotone"))
    assert calls, "mutant not applied"


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
