"""L1 parity: whole chains of MICE against R's mice, from R's start and with R's draws.

Reference: tests/reference/r/l1_chain.R. Each case runs ``impute(maxit=0)``,
gets R's start values and a strict generator in every chain state and is
continued with ``continue_chains`` as R ran ``mice()`` and ``mice.mids()``
(`_l1_chain.run_case`). Every comparison of `_l1_chain.CHECKS` runs on every
case: ``imp`` cell by cell, the completed datasets, the chain statistics,
the logged events, ``iteration``, method, visit sequence and predictor
matrix, the draws (none, or R's in R's order) and, for the spy cases, what
the imputer received at every call. test_l1_chain_mutants runs the same
comparisons against a defective library.
"""

import functools

import numpy as np
import pytest

from ._helpers import decode_frame
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


def test_fixture_covers_the_kinds_and_the_base_cases():
    assert {case["kind"] for case in CASES.values()} == {"deterministic", "scripted", "spy"}
    for case in CASES.values():
        assert case["kind"] == "deterministic" or case["m"] == 1, case["name"]
        assert (case["calls"] is not None) == (case["kind"] == "spy"), case["name"]
        assert (case["draws"] is not None) == (case["kind"] == "scripted"), case["name"]
    missing = [name for name, covers in BASE_CASES.items() if not any(covers(case) for case in CASES.values())]
    assert missing == []
