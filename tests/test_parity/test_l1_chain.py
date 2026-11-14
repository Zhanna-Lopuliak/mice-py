"""L1 parity: whole chains of MICE against R's mice, from R's start and with R's draws.

Reference: tests/reference/r/l1_chain.R. Each case runs ``impute(maxit=0)``,
gets R's start values and a strict generator in every chain state and is
continued with ``continue_chains`` as R ran ``mice()`` and ``mice.mids()``
(`_l1_chain.run_case`). Every comparison of `_l1_chain.CHECKS` runs on every
case: ``imp`` cell by cell, the completed datasets, the chain statistics,
the logged events, ``iteration``, method, visit sequence and predictor
matrix, the draws (none, or R's in R's order), for the spy cases what the
imputer received at every call, and the numbers of observed values that the
completed data lose under ``where`` (R silently, MICE with a warning).
test_l1_chain_mutants runs the same comparisons against a defective
library.
"""

import functools

import numpy as np
import pandas as pd
import pytest

from ._helpers import decode_frame, str_list
from ._l1_chain import CASES, CHECKS, run_case

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
