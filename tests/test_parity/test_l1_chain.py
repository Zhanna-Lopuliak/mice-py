"""L1 parity: whole chains of MICE against R's mice, from R's start and with R's draws.

Reference: tests/reference/r/l1_chain.R. Each case runs ``impute(maxit=0)``,
gets R's start values and a strict generator in every chain state and is
continued with ``continue_chains`` as R ran ``mice()`` and ``mice.mids()``
(`_l1_chain.run_case`). Every comparison of `_l1_chain.CHECKS` runs on every
case: ``imp`` cell by cell, the completed datasets, the chain statistics,
the logged events, ``iteration``, method, visit sequence and predictor
matrix (with ``post``, the columns whose function is in effect), the draws
(none, or R's in R's order), for the spy cases what the imputer received at
every call, and the numbers of observed values that the completed data lose
under ``where`` (R silently, MICE with a warning).
test_l1_chain_mutants runs the same comparisons against a defective
library.

Where R's ``mice()`` stops (``r_errors`` of the fixture), the tests at the
end record R's error and test what MICE does instead: it raises its own
``ValueError`` before the chains start, or, for ``polyreg`` with a label
held only by an ignored row where the augmentation is skipped, it imputes
(the label gets probability 0) where R stops.

``post`` and passive columns: where MICE deliberately differs from R, the
last tests test MICE's own result. A passive function that is not row-wise
gets all rows, while R evaluates the formula on the rows of the where-cells
(``differences``: the other columns are R's, the passive one is the
function of all rows); a ``post`` for a passive column raises, while R
keeps it and never runs it (``post_never_run``).
"""

import functools
import re
import warnings

import numpy as np
import pandas as pd
import pytest

from imputation import MICE

from ._helpers import as_array, as_bool, decode_frame, passive_function, str_list
from ._l1_chain import (
    CASES,
    CHECKS,
    DIFFERENCES,
    POST_NEVER_RUN,
    R_ERRORS,
    check_passive_function_gets_all_rows,
    impute_kwargs,
    run_case,
    without_column,
)

pytestmark = pytest.mark.parity


@functools.cache
def _run(name):
    """The run of case `name`, once per session: the comparisons only read it."""
    return run_case(CASES[name])


@pytest.mark.parametrize("check", list(CHECKS))
@pytest.mark.parametrize("name", list(CASES))
def test_chain_matches_r(name, check):
    CHECKS[check](CASES[name], _run(name))


# ---------------------------------------------------------------------------
# what the fixture covers
# ---------------------------------------------------------------------------


def _methods(case):
    return {meth for meth in case["method"].values() if meth}


def _predictor_keeps_missing_values(case):
    """A column without a method and with missing values predicts a column that is imputed."""
    data = decode_frame(case["data"])
    rows = case["predictor_matrix"]["rows"]
    matrix = dict(zip(rows, np.array(case["predictor_matrix"]["values"], dtype=int)))
    columns = case["predictor_matrix"]["cols"]
    imputed = [col for col, meth in case["method"].items() if meth]
    return any(
        not case["method"][col] and data[col].isna().any() and any(matrix[target][j] for target in imputed)
        for j, col in enumerate(columns)
    )


def _events(case, step=-1):
    """The logged events after a step as (it, im, out) triples."""
    events = case["steps"][step]["logged_events"]
    if events is None:
        return []
    return list(zip(np.atleast_1d(events["it"]), np.atleast_1d(events["im"]), np.atleast_1d(events["out"])))


def _lindep_in_every_iteration(case):
    """remove.lindep drops a column in every iteration of every chain."""
    dropped = {(it, im) for it, im, out in _events(case) if it > 0 and not str(out).startswith("df set to 1")}
    iterations = case["steps"][-1]["iteration"]
    return dropped == {(it, im) for it in range(1, iterations + 1) for im in range(1, case["m"] + 1)}


def _df_set_to_1_once(case):
    """check.df logs in the first iteration, and later iterations run."""
    its = [it for it, _, out in _events(case) if str(out).startswith("df set to 1")]
    return bool(its) and set(its) == {1} and sum(case["maxit"]) > 1


def _events_in_both_steps(case):
    """A continued run that logs events before and after it is continued."""
    return len(case["steps"]) > 1 and 0 < len(_events(case, 0)) < len(_events(case, 1))


BASE_CASES = {
    "norm, scripted, nhanes, 3 iterations": lambda c: (
        c["kind"] == "scripted" and _methods(c) == {"norm"} and c["data"]["nrow"] == 25 and c["maxit"] == [3]
    ),
    "norm.predict, deterministic, nhanes, m = 2": lambda c: (
        c["kind"] == "deterministic" and _methods(c) == {"norm.predict"} and c["m"] == 2 and c["maxit"] == [2]
    ),
    "mean and norm.predict, deterministic, boys, m = 2": lambda c: (
        c["kind"] == "deterministic" and _methods(c) == {"mean", "norm.predict"} and c["m"] == 2
    ),
    "a predictor that keeps missing values, spy": lambda c: c["kind"] == "spy" and _predictor_keeps_missing_values(c),
    "a predictor that keeps missing values, scripted logreg": lambda c: (
        c["kind"] == "scripted" and _methods(c) == {"logreg"} and _predictor_keeps_missing_values(c)
    ),
    "remove.lindep drops a column in every iteration": _lindep_in_every_iteration,
    "df set to 1 in the first iteration only": _df_set_to_1_once,
    "visit sequence and predictor matrix given, spy or deterministic": lambda c: (
        c["kind"] in ("spy", "deterministic")
        and c["args"]["visit_sequence"] is not None
        and c["args"]["predictor_matrix"] is not None
    ),
    "mice() with 2, then mice.mids() with 3 iterations, scripted norm": lambda c: (
        c["kind"] == "scripted" and _methods(c) == {"norm"} and c["maxit"] == [2, 3]
    ),
    "continued, with events in both steps": _events_in_both_steps,
}


def _where_keyword(case):
    where = case["args"].get("where")
    return where if isinstance(where, str) else None


def _where(case):
    """The where-cells of the case as a boolean frame in data order."""
    data = decode_frame(case["data"])
    where = case["args"].get("where", "missing")
    if isinstance(where, dict):
        return pd.DataFrame(np.array(where["values"], dtype=bool), columns=str_list(where["cols"]))[data.columns]
    missing = data.isna()
    keywords = {"missing": missing, "observed": ~missing, "all": missing | True, "none": missing & False}
    return keywords[where]


def _predictors(case, col):
    """The columns that `col` predicts among those with a method."""
    rows = case["predictor_matrix"]["rows"]
    j = case["predictor_matrix"]["cols"].index(col)
    values = np.array(case["predictor_matrix"]["values"], dtype=int)
    return [target for target, row in zip(rows, values) if row[j] and case["method"][target]]


def _left_out_in_a_predictor(case):
    """A missing cell outside where in a column that predicts a column with a method."""
    data, where = decode_frame(case["data"]), _where(case)
    return any((data[col].isna() & ~where[col]).any() and _predictors(case, col) for col in data.columns)


def _complete_column_with_where_cells(case):
    data, where = decode_frame(case["data"]), _where(case)
    return any(data[col].notna().all() and where[col].any() and case["method"][col] for col in data.columns)


def _observed_where_cell_in_a_predictor(case):
    data, where = decode_frame(case["data"]), _where(case)
    return any((data[col].notna() & where[col]).any() and _predictors(case, col) for col in data.columns)


def _imp_missing(case, col, step=-1):
    """Per where-cell of `col` (row order): NA in R's imp of every chain after the step."""
    return np.array([[v is None for v in chain] for chain in case["steps"][step]["imp"][col]["values"]]).all(axis=0)


def _no_imputation_for_observed_and_missing_cells(case):
    """A column with a method has an observed and a missing where-cell without imputation."""
    data, where = decode_frame(case["data"]), _where(case)
    for col in data.columns:
        if not case["method"][col] or not where[col].any():
            continue
        observed = data[col].notna().to_numpy()[where[col].to_numpy()]
        missing = _imp_missing(case, col)
        if (missing & observed).any() and (missing & ~observed).any():
            return True
    return False


def _no_imputation_at_all(case):
    """Every where-cell of a column with a method is NA in imp after every step."""
    where = _where(case)
    steps = range(len(case["steps"]))
    return any(
        case["method"][col] and where[col].any() and all(_imp_missing(case, col, step).all() for step in steps)
        for col in where.columns
    )


def _where_cells_without_method(case, user):
    """A column with observed where-cells and method "": given by the user, or set by the setup step."""
    data, where = decode_frame(case["data"]), _where(case)
    method = case["args"]["method"]
    for col in data.columns:
        if case["method"][col] or not (data[col].notna() & where[col]).any():
            continue
        by_user = isinstance(method, dict) and method.get(col) == ""
        if by_user == user:
            return True
    return False


def _chain_statistics_over_observed_where_cells(case):
    data, where = decode_frame(case["data"]), _where(case)
    return any((data[col].notna() & where[col]).any() for col in case["steps"][-1]["chain_mean"] or {})


WHERE_CASES = {
    'where "all", deterministic, mean and norm.predict': lambda c: (
        c["kind"] == "deterministic" and _where_keyword(c) == "all" and _methods(c) == {"mean", "norm.predict"}
    ),
    'where "all", spy': lambda c: c["kind"] == "spy" and _where_keyword(c) == "all",
    'where "all", scripted norm': lambda c: (
        c["kind"] == "scripted" and _where_keyword(c) == "all" and _methods(c) == {"norm"}
    ),
    'where "observed"': lambda c: _where_keyword(c) == "observed",
    'where "none"': lambda c: _where_keyword(c) == "none",
    "missing cells left out of where in a predictor column, spy": lambda c: (
        c["kind"] == "spy" and _left_out_in_a_predictor(c)
    ),
    "a complete column with where-cells": _complete_column_with_where_cells,
    "an observed where-cell in a predictor column, spy": lambda c: (
        c["kind"] == "spy" and _where_keyword(c) is None and _observed_where_cell_in_a_predictor(c)
    ),
    "an observed and a missing where-cell without imputation": _no_imputation_for_observed_and_missing_cells,
    "no where-cell of a column with a method imputed": _no_imputation_at_all,
    'observed where-cells and method "" given by the user': lambda c: _where_cells_without_method(c, user=True),
    'observed where-cells and method "" set by the setup step': lambda c: _where_cells_without_method(c, user=False),
    "chain statistics over observed where-cells": _chain_statistics_over_observed_where_cells,
    "continued under a where matrix": lambda c: len(c["steps"]) > 1 and isinstance(c["args"].get("where"), dict),
    'scripted logreg under where "all"': lambda c: (
        c["kind"] == "scripted" and _where_keyword(c) == "all" and _methods(c) == {"logreg"}
    ),
}


def test_fixture_covers_the_kinds_and_the_base_cases():
    assert {case["kind"] for case in CASES.values()} == {"deterministic", "scripted", "spy"}
    for case in CASES.values():
        assert case["kind"] == "deterministic" or case["m"] == 1, case["name"]
        assert (case["calls"] is not None) == (case["kind"] == "spy"), case["name"]
        assert (case["draws"] is not None) == (case["kind"] == "scripted"), case["name"]
    missing = [name for name, covers in BASE_CASES.items() if not any(covers(case) for case in CASES.values())]
    assert missing == []


def test_fixture_covers_the_where_cases():
    missing = [name for name, covers in WHERE_CASES.items() if not any(covers(case) for case in CASES.values())]
    assert missing == []
    # lost_observed is written only for a case that loses observed values (l1_chain.R checks the converse)
    for case in CASES.values():
        lost = case.get("lost_observed")
        if lost is not None:
            assert any(lost["start"]) or any(any(step) for step in lost["steps"]), case["name"]


# ---------------------------------------------------------------------------
# what the fixture covers: ignore
# ---------------------------------------------------------------------------


def _ignored(case):
    """The rows the case leaves out of the fits, one boolean per row (all False without ignore)."""
    ignore = case["args"].get("ignore")
    return np.zeros(case["data"]["nrow"], dtype=bool) if ignore is None else as_bool(ignore)


def _imputed_columns(case, method=None):
    return [col for col, meth in case["method"].items() if meth and (method is None or meth == method)]


def _imputed_and_neither(case):
    """An ignored row with a missing where-cell (imputed) and one with an observed cell outside where (in
    neither mask), in one column with a method."""
    data, where, ignored = decode_frame(case["data"]), _where(case), _ignored(case)
    return any(
        (ignored & where[col] & data[col].isna()).any() and (ignored & ~where[col] & data[col].notna()).any()
        for col in _imputed_columns(case)
    )


def _df_set_counts_fit_rows(case):
    """'df set to 1' with the number of observed values in the rows that are not ignored, fewer than all."""
    data, ignored = decode_frame(case["data"]), _ignored(case)
    for _, _, out in _events(case):
        if not str(out).startswith("df set to 1"):
            continue
        counted = int(str(out).split("# observed cases: ")[1].split()[0])
        for col in _imputed_columns(case):
            observed = data[col].notna().to_numpy()
            if counted == (observed & ~ignored).sum() < observed.sum():
                return True
    return False


def _lindep_drops_constant_in_fit_rows(case):
    """remove.lindep drops a predictor in the iterations that is constant in the rows that are not ignored and
    varies over all rows (so the setup step keeps it)."""
    data, ignored = decode_frame(case["data"]), _ignored(case)
    dropped = {name for it, _, out in _events(case) if it > 0 for name in str(out).split(", ")}
    return any(
        col in dropped and data.loc[~ignored, col].nunique() == 1 and data[col].nunique() > 1 for col in data.columns
    )


def _logreg_one_cell(case):
    data, where, ignored = decode_frame(case["data"]), _where(case), _ignored(case)
    return any(
        where[col].sum() == 1 and (ignored & data[col].notna()).any() for col in _imputed_columns(case, "logreg")
    )


def _label_only_in_ignored_rows(case, method):
    data, ignored = decode_frame(case["data"]), _ignored(case)
    return any(
        bool(set(data.loc[ignored, col].dropna()) - set(data.loc[~ignored, col].dropna()))
        for col in _imputed_columns(case, method)
    )


def _clamp_lifted_by_ignored_row(case):
    """A predictor whose values in the rows that are not ignored have mean + sd / 2 above their maximum (augment
    clamps), while an ignored row holds a larger value."""
    data, ignored = decode_frame(case["data"]), _ignored(case)
    for col in data.columns:
        if data[col].dtype.kind != "f" or not data[col].notna().all():
            continue
        kept = data.loc[~ignored, col]
        if kept.mean() + kept.std() / 2 > kept.max() and (data.loc[ignored, col] > kept.max()).any():
            return True
    return False


def _one_row_outside_fit_rows(case, method):
    """Augment skipped: every where-cell of the column is observed, and one ignored row is the only row
    outside its fit rows."""
    data, where, ignored = decode_frame(case["data"]), _where(case), _ignored(case)
    for col in _imputed_columns(case, method):
        observed = data[col].notna().to_numpy()
        outside = ~(observed & ~ignored)
        if observed.all() and where[col].any() and outside.sum() == (outside & ignored).sum() == 1:
            return True
    return False


IGNORE_CASES = {
    "ignored rows with a missing where-cell (imputed) and an observed cell (in neither mask), spy": lambda c: (
        c["kind"] == "spy" and _imputed_and_neither(c)
    ),
    "ignore, deterministic, norm.predict and mean, m = 2": lambda c: (
        c["kind"] == "deterministic" and _ignored(c).any() and _methods(c) == {"mean", "norm.predict"} and c["m"] == 2
    ),
    "ignore, scripted norm": lambda c: c["kind"] == "scripted" and _ignored(c).any() and _methods(c) == {"norm"},
    "df set to 1 counts the fit rows, not the ignored ones": _df_set_counts_fit_rows,
    "remove.lindep drops a predictor constant in the rows that are not ignored": _lindep_drops_constant_in_fit_rows,
    "scripted logreg, one cell to impute and ignored rows": lambda c: c["kind"] == "scripted" and _logreg_one_cell(c),
    "scripted logreg, a category only in ignored rows": lambda c: (
        c["kind"] == "scripted" and _label_only_in_ignored_rows(c, "logreg")
    ),
    "scripted logreg, a predictor value of an ignored row outside the range of the others": lambda c: (
        c["kind"] == "scripted" and _methods(c) == {"logreg"} and _clamp_lifted_by_ignored_row(c)
    ),
    "scripted polr, augment skipped, a level only in an ignored row": lambda c: (
        c["kind"] == "scripted" and _one_row_outside_fit_rows(c, "polr") and _label_only_in_ignored_rows(c, "polr")
    ),
    "ignore with a where matrix": lambda c: _ignored(c).any() and isinstance(c["args"].get("where"), dict),
    "continued with ignore": lambda c: len(c["steps"]) > 1 and _ignored(c).any() and _methods(c) == {"norm"},
    "fewer than 10 rows left for the fits, with R's warning": lambda c: (
        (~_ignored(c)).sum() < 10 and str_list(c.get("warnings")) == [WARNING_FEWER_ROWS]
    ),
}
WARNING_FEWER_ROWS = "Fewer than 10 rows for fitting the imputation model. Are you sure?"


def test_fixture_covers_the_ignore_cases():
    missing = [name for name, covers in IGNORE_CASES.items() if not any(covers(case) for case in CASES.values())]
    assert missing == []
    # a case writes warnings only when R warned
    assert all(str_list(case["warnings"]) for case in CASES.values() if "warnings" in case)


# ---------------------------------------------------------------------------
# what the fixture covers: post and passive
# ---------------------------------------------------------------------------


def _post(case):
    """The post recipes of the case by column ({} without post)."""
    return case["args"].get("post") or {}


def _passive(case):
    """The passive recipes of the case by column ({} without passive columns)."""
    return case["args"].get("passive") or {}


def _post_recipes(case):
    return {recipe["recipe"] for recipe in _post(case).values()}


def _bounds_bind(case):
    """Bounds on a column, with an imputation of R's last step at a bound: the chain statistics are those after
    post."""
    for col, recipe in _post(case).items():
        if recipe["recipe"] == "bounds" and col in (case["steps"][-1]["chain_mean"] or {}):
            values = np.concatenate([as_array(chain) for chain in case["steps"][-1]["imp"][col]["values"]])
            if np.isin(values, [recipe["lower"], recipe["upper"]]).any():
                return True
    return False


def _later_column_predicted_from_post(case):
    """A column with post predicts a column visited after it in the same iteration."""
    visits = str_list(case["effective_visit_sequence"])
    return any(
        set(_predictors(case, col)) & set(visits[visits.index(col) + 1 :]) for col in _post(case) if col in visits
    )


def _post_fills_where_cells_without_imputation(case):
    """fill_missing on a column none of whose where-cells has an imputation: R's imp holds the value in every
    where-cell after every step."""
    for col, recipe in _post(case).items():
        if recipe["recipe"] == "fill_missing":
            values = [v for step in case["steps"] for chain in step["imp"][col]["values"] for v in chain]
            if all(v == recipe["value"] for v in values):
                return True
    return False


def _setup_events(case):
    """The events of the setup step (iteration 0) as (out, meth) pairs."""
    events = case["steps"][-1]["logged_events"]
    if events is None:
        return set()
    rows = zip(*(np.atleast_1d(events[name]) for name in ("it", "out", "meth")))
    return {(str(out), str(meth)) for it, out, meth in rows if it == 0}


def _post_cleared_by_the_setup_step(case):
    """post given for a column whose method the setup step removed (logged): R clears it."""
    removed = {out for out, _ in _setup_events(case)}
    return any(col in removed and col not in str_list(case.get("post_columns")) for col in _post(case))


def _inputs(recipe):
    return [str(recipe[key]) for key in ("weight", "height", "left", "right", "column") if key in recipe]


def _before_its_inputs(case, col, order):
    return all(order.index(col) < order.index(name) for name in _inputs(_passive(case)[col]) if name in order)


def _function_holds(case, col, step):
    """In R's completed data of every chain after the step, the passive column is its function in its where-cells."""
    function, where = passive_function(_passive(case)[col]), _where(case)[col].to_numpy()
    for payload in case["steps"][step]["completed"]:
        completed = decode_frame(payload)
        values, expected = completed[col].to_numpy()[where], function(completed).to_numpy()[where]
        if not np.allclose(values, expected, rtol=1e-8, atol=1e-10, equal_nan=True):
            return False
    return True


def _passive_last_by_default(case):
    """No visit sequence given, the passive column before its inputs in the data, visited last, and the function
    of its row in the completed data."""
    columns = str_list(case["data"]["columns"])
    return case["args"]["visit_sequence"] is None and any(
        _before_its_inputs(case, col, columns)
        and str_list(case["effective_visit_sequence"])[-1] == col
        and _function_holds(case, col, -1)
        for col in _passive(case)
    )


def _roman_passive_before_its_inputs(case):
    """Under "roman" the passive column is visited before its inputs, and after one iteration it is not their
    function."""
    visits = str_list(case["effective_visit_sequence"])
    return case["args"]["visit_sequence"] == "roman" and any(
        _before_its_inputs(case, col, visits) and case["maxit"][0] == 1 and not _function_holds(case, col, 0)
        for col in _passive(case)
    )


def _imp_has_missing(case, col, step=-1):
    return any(v is None for chain in case["steps"][step]["imp"][col]["values"] for v in chain)


def _input_without_method(case):
    """An input of a passive column has method "" and missing values: the passive column, and a column it
    predicts, have missing imputations."""
    data = decode_frame(case["data"])
    return any(
        any(case["method"][name] == "" and data[name].isna().any() for name in _inputs(recipe))
        and _imp_has_missing(case, col)
        and any(_imp_has_missing(case, target) for target in _predictors(case, col))
        for col, recipe in _passive(case).items()
    )


def _passive_copy(case):
    """A passive column that the setup step removes as collinear: method "", logged."""
    return any(case["method"][col] == "" and (col, "collinear") in _setup_events(case) for col in _passive(case))


POST_PASSIVE_CASES = {
    "post bounds, scripted norm, 3 iterations": lambda c: (
        c["kind"] == "scripted" and _post_recipes(c) == {"bounds"} and _methods(c) == {"norm"} and sum(c["maxit"]) >= 3
    ),
    "post bounds, norm.predict": lambda c: (
        c["kind"] == "deterministic" and "bounds" in _post_recipes(c) and "norm.predict" in _methods(c)
    ),
    "post shift": lambda c: "shift" in _post_recipes(c),
    "post rule on another column": lambda c: "rule" in _post_recipes(c),
    "a later column predicted from the processed values, spy": lambda c: (
        c["kind"] == "spy" and _later_column_predicted_from_post(c)
    ),
    "post, 2 or more iterations": lambda c: bool(_post(c)) and sum(c["maxit"]) >= 2,
    'post under where "all"': lambda c: bool(_post(c)) and _where_keyword(c) == "all",
    "post with ignore": lambda c: bool(_post(c)) and _ignored(c).any(),
    "post fills where-cells that could not be imputed": _post_fills_where_cells_without_imputation,
    "post bounds that bind, chain statistics after post": _bounds_bind,
    "continued with post": lambda c: bool(_post(c)) and len(c["steps"]) > 1,
    "post of a column the setup step removes, cleared as in R": _post_cleared_by_the_setup_step,
    "passive bmi, visited last by default": lambda c: (
        any(r["recipe"] == "bmi" for r in _passive(c).values()) and _passive_last_by_default(c)
    ),
    '"roman" with the passive column before its inputs': _roman_passive_before_its_inputs,
    "a visit sequence that names the passive column twice": lambda c: any(
        str_list(c["effective_visit_sequence"]).count(col) == 2 for col in _passive(c)
    ),
    "a passive column as predictor, spy": lambda c: (
        c["kind"] == "spy" and any(_predictors(c, col) for col in _passive(c))
    ),
    "observed where-cells of a passive column": lambda c: any(
        (_where(c)[col] & decode_frame(c["data"])[col].notna()).any() for col in _passive(c)
    ),
    'an input with method "": the passive value is missing and spreads': _input_without_method,
    "observed values of a passive column lost": lambda c: any(
        col in step for step in (c.get("lost_observed") or {"steps": []})["steps"] for col in _passive(c)
    ),
    'a passive copy of a column: collinear, method ""': _passive_copy,
    'a complete passive column: method ""': lambda c: any(
        decode_frame(c["data"])[col].notna().all() and c["method"][col] == "" for col in _passive(c)
    ),
    "a passive column without observed values": lambda c: any(
        decode_frame(c["data"])[col].isna().all() and c["method"][col].startswith("~") for col in _passive(c)
    ),
    "continued with a passive column": lambda c: bool(_passive(c)) and len(c["steps"]) > 1,
    "post and passive in one run": lambda c: bool(_post(c)) and bool(_passive(c)),
}


def test_fixture_covers_the_post_and_passive_cases():
    missing = [name for name, covers in POST_PASSIVE_CASES.items() if not any(covers(case) for case in CASES.values())]
    assert missing == []
    # post_columns is written for every case with post, and only then
    assert all(("post_columns" in case) == bool(_post(case)) for case in CASES.values())


# ---------------------------------------------------------------------------
# where R stops
# ---------------------------------------------------------------------------

# What MICE does where R's mice() stops: None where it imputes, else the
# beginning of its ValueError
PORT_WHERE_R_STOPS = {
    "polyreg_no_predictor_label_only_ignored": None,
    "polyreg_one_row_outside_label_only_ignored": None,
    "all_observed_ignored": r"^Every observed value of the column 'u' lies in a row that ignore leaves out",
    "ignore_na": r"^ignore contains missing values: every entry must be True or False\.$",
    "ignore_numeric": r"^ignore must hold booleans, True or False; it has dtype int64\.",
    "ignore_short": r"^ignore has 24 entries, the data has 25 rows",
}
R_STOPS = {
    "polyreg_no_predictor_label_only_ignored": r"^dims \[product 6\] do not match the length of object \[9\]$",
    "polyreg_one_row_outside_label_only_ignored": r"^dims \[product 8\] do not match the length of object \[12\]$",
    "all_observed_ignored": r"^0 \(non-NA\) cases$",
    "ignore_na": r"^missing value where TRUE/FALSE needed$",
    "ignore_numeric": r"^Argument ignore not a logical\.$",
    "ignore_short": r"^length\(ignore\) \(24\) does not match nrow\(data\) \(25\)\.$",
}


def test_fixture_r_errors_are_those_tested():
    assert sorted(R_ERRORS) == sorted(PORT_WHERE_R_STOPS) == sorted(R_STOPS)
    for name, error in R_STOPS.items():
        assert pd.Series([R_ERRORS[name]["error"]]).str.match(error).all(), (name, R_ERRORS[name]["error"])


def _impute_r_error_case(case, n_imputations=None):
    kwargs = impute_kwargs(case)
    if n_imputations is not None:
        kwargs["n_imputations"] = n_imputations
    return MICE(decode_frame(case["data"])), kwargs


@pytest.mark.parametrize("name", [name for name, error in PORT_WHERE_R_STOPS.items() if error is not None])
def test_where_r_stops_mice_raises_before_the_chains(name):
    case = R_ERRORS[name]
    mice, kwargs = _impute_r_error_case(case)
    with pytest.raises(ValueError, match=PORT_WHERE_R_STOPS[name]):
        mice.impute(maxit=case["maxit"], **kwargs)
    assert not getattr(mice, "imputed_datasets", None)


@pytest.mark.parametrize("name", [name for name, error in PORT_WHERE_R_STOPS.items() if error is None])
def test_where_r_stops_polyreg_imputes_without_the_label_of_the_ignored_row(name):
    """R: multinom() drops the level c, held only by the ignored row 3, with the warning "group 'c' is empty",
    and polyreg stops comparing its uniform numbers with the probabilities of the two other levels. MICE takes
    the labels present in all of y (a, b, c), fits a and b, and gives c the probability 0: 40 chains impute a
    and b, never c."""
    case = R_ERRORS[name]
    assert str_list(case["warnings"])[0] == "group 'c' is empty"
    mice, kwargs = _impute_r_error_case(case, n_imputations=40)
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        out = mice.impute(maxit=case["maxit"], **kwargs)
    where = mice.where["g"].to_numpy()
    imputed = pd.Series(np.concatenate([completed["g"].to_numpy()[where] for completed in out]))
    assert set(imputed) == {"a", "b"}, imputed.value_counts().to_dict()
    assert mice.method["g"] == "polyreg" and mice.ignore[2]


# ---------------------------------------------------------------------------
# post and passive: where MICE deliberately differs
# ---------------------------------------------------------------------------


def test_fixture_differences_are_those_tested():
    assert sorted(DIFFERENCES) == ["passive_aggregate"]
    assert sorted(POST_NEVER_RUN) == ["post_for_passive"]


def test_passive_function_that_is_not_row_wise_gets_all_rows_unlike_r():
    """q = (chl - mean(chl))^2 is passive, its where-cells are those where chl is missing, and it predicts
    nothing. R evaluates the formula on the rows of the where-cells, so its mean of chl is that of the imputed
    values only; MICE passes all rows to the function, so its mean is that of the whole completed column. Every
    other column is R's; q is the function of all rows and differs from R's."""
    case = DIFFERENCES["passive_aggregate"]
    run = run_case(case)
    other_case, other_run = without_column(case, run, "q")
    for check in CHECKS.values():
        check(other_case, other_run)
    check_passive_function_gets_all_rows(case, run)


@pytest.mark.parametrize("name", list(POST_NEVER_RUN))
def test_post_for_a_passive_column_raises_where_r_never_runs_it(name):
    """R keeps the post command of a passive column after the setup step (post_columns) and never runs it:
    l1_chain.R requires that the run equals the run without it. MICE raises before the chains start."""
    case = POST_NEVER_RUN[name]
    kwargs = impute_kwargs(case)
    both = [col for col in kwargs["post"] if col in kwargs["passive"]]
    assert both and set(both) <= set(str_list(case["post_columns"]))
    mice = MICE(decode_frame(case["data"]))
    with pytest.raises(ValueError, match=re.escape(f"post names the passive columns {both}: post is not applied")):
        mice.impute(maxit=case["maxit"], **kwargs)
    assert not getattr(mice, "imputed_datasets", None)
