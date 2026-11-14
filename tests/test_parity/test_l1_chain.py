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

Where R's ``mice()`` stops (``r_errors`` of the fixture), the tests at the
end record R's error and test what MICE does instead: it raises its own
``ValueError`` before the chains start, or, for ``polyreg`` with a label
held only by an ignored row where the augmentation is skipped, it imputes
(the label gets probability 0) where R stops.
"""

import functools
import warnings

import numpy as np
import pandas as pd
import pytest

from imputation import MICE

from ._helpers import as_bool, decode_frame, str_list
from ._l1_chain import CASES, CHECKS, R_ERRORS, impute_kwargs, run_case

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
