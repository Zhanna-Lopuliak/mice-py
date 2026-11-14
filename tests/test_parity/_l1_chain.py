"""Whole chains of ``MICE`` against R's mice: the helper of test_l1_chain and test_l1_chain_mutants.

Reference: tests/reference/r/l1_chain.R. R and Python draw different random
numbers, so a chain is compared exactly from R's start and with R's draws.
`run_case` runs ``MICE.impute(maxit=0)`` with the arguments of the case,
writes R's start values into the chain states (``MICE._chain_states[i].imp``)
and gives each state a strict `ScriptedGenerator`, then continues the chains
with ``MICE.continue_chains`` step by step, as R ran ``mice()`` and
``mice.mids()``. No function of the library is patched.

- deterministic cases: each chain gets a generator without scripted values,
  so that any draw fails;
- scripted cases (one chain): R's draws, served in R's order;
- spy cases (one chain): the method "spy" is registered for the run. It
  records what MICE hands it and imputes as `imputation.norm.norm_predict`.
  MICE passes an imputer neither the column nor the names of the
  predictors, so the spy reads the column, the iteration, the chain and the
  names of the predictors kept from the frame of the ``MICE._iterate`` that
  called it.

Where R's completed data lose observed values (a where-cell without an
imputation is NA, also over an observed value), MICE warns with the number
of them per column. `run_case` records that warning after the start and
after every step, and `check_lost_observed_values` compares its numbers with
R's (``lost_observed`` of the case; a case without it loses none). It also
records the warnings that MICE gives with R's text (`R_WARNINGS`, such as
"Fewer than 10 rows ..." for ``ignore``), which `check_warnings` compares
with R's (``warnings`` of the case; a case without it has none).

``R_ERRORS`` holds the calls of R's ``mice()`` that stop, with R's error, for
the tests of what MICE does there (test_l1_chain).

The comparisons are separate functions, so that test_l1_chain_mutants can
run them against a defective library; each raises ``AssertionError`` with a
message that names the case and what differs. Tolerances as in the other L1
tests: values ``rtol 1e-8``, ``atol 1e-10``; labels, masks, the positions of
missing values, events and draws identical.
"""

import re
import sys
import warnings
from dataclasses import dataclass, field
from typing import NamedTuple

import numpy as np
import pandas as pd
import pytest

from imputation import MICE
from imputation.norm import norm_predict

from .._logged_events import continue_checking_log_warning, impute_checking_log_warning
from ..test_properties._helpers import register_imputer
from ._helpers import ScriptedGenerator, as_array, as_bool, decode_frame, load_fixture, str_list

RTOL = 1e-8
ATOL = 1e-10

FIXTURE = load_fixture("l1_chain")
CASES = {case["name"]: case for case in FIXTURE["cases"]}
R_ERRORS = {case["name"]: case for case in FIXTURE["r_errors"]}

# The arguments of mice() a case may pass, under the names of the matching arguments of impute
_ARGUMENTS = ("method", "predictor_matrix", "visit_sequence", "where", "ignore")
_LOG_COLUMNS = ("it", "im", "dep", "meth", "out")
_POOLS = ("chisq", "normal", "uniform")

# MICE's warning about observed where-cells that are missing in the completed data, and its count per column
_LOST_WARNING = "Observed values replaced by missing values in the completed data"
_LOST_COUNT = re.compile(
    r"'([^']*)': (\d+) \((no method|predictors incomplete in these rows|passive function missing in these rows)\)"
)

# Warnings that MICE gives with R's text; `run_case` records them for `check_warnings`
R_WARNINGS = ("Fewer than 10 rows for fitting the imputation model. Are you sure?",)


class SpyCall(NamedTuple):
    """What the spy received at one call, with the call's iteration, chain and column (1-based, as R's)."""

    it: int
    im: int
    dep: str
    y: np.ndarray
    ry: np.ndarray
    wy: np.ndarray
    x_names: list
    x: np.ndarray


@dataclass
class Step:
    """A run after one call of ``continue_chains``: copies of what it compares with R."""

    imp: list  # per chain: column -> pd.Series over the where-cells
    datasets: list
    chain_mean: dict
    chain_var: dict
    logged_events: pd.DataFrame
    iteration: int
    draws_used: dict  # pool -> number of values drawn so far, over all chains
    lost_observed: dict  # the warning about lost observed values of the step: column -> (number, reason)
    warnings: list  # the messages of the step's warnings with R's text (R_WARNINGS)


@dataclass
class ChainRun:
    """A case run by `run_case`."""

    mice: MICE
    generators: list  # the ScriptedGenerator of each chain
    steps: list = field(default_factory=list)
    calls: list = field(default_factory=list)  # SpyCall, in the order of the calls
    start_lost_observed: dict = field(default_factory=dict)  # as Step.lost_observed, for impute(maxit=0)
    start_warnings: list = field(default_factory=list)  # as Step.warnings, for impute(maxit=0)


def impute_kwargs(case):
    """The arguments of ``impute`` that match those of R's ``mice()`` in the case (without ``maxit``)."""
    args = case["args"]
    assert set(args) <= set(_ARGUMENTS), f"arguments the helper does not know: {sorted(set(args) - set(_ARGUMENTS))}"
    kwargs = {"n_imputations": case["m"], "seed": case["seed"]}
    method = args["method"]
    if method is not None:
        kwargs["method"] = method if isinstance(method, str) else {col: str(meth) for col, meth in method.items()}
    if args["predictor_matrix"] is not None:
        kwargs["predictor_matrix"] = _matrix(args["predictor_matrix"])
    visit_sequence = args["visit_sequence"]
    if visit_sequence is not None:
        kwargs["visit_sequence"] = visit_sequence if isinstance(visit_sequence, str) else str_list(visit_sequence)
    where = args.get("where")
    if where is not None:
        # R's matrix, its columns named, as a frame with the index of the data (decode_frame: 0, 1, ...)
        kwargs["where"] = (
            where
            if isinstance(where, str)
            else pd.DataFrame(np.array(where["values"], dtype=bool), columns=str_list(where["cols"]))
        )
    ignore = args.get("ignore")
    if ignore is not None:
        # R's vector as given: a list of booleans, or of what R was given where it stops
        kwargs["ignore"] = ignore if isinstance(ignore, list) else [ignore]
    return kwargs


def run_case(case):
    """Run the case from R's start with R's draws; return a `ChainRun` with one `Step` per value of ``maxit``.

    The warning about lost observed values and the warnings with R's text are recorded
    (`_recording_warnings`); any other warning but the checked "Number of logged events" fails the run.
    """
    mice = MICE(decode_frame(case["data"]))
    run = ChainRun(mice=mice, generators=[])
    with pytest.MonkeyPatch.context() as patch, warnings.catch_warnings():
        warnings.simplefilter("error")
        if case["kind"] == "spy":
            register_imputer(patch, "spy", _spy(run.calls))
        run.start_lost_observed, run.start_warnings = _recording_warnings(
            impute_checking_log_warning, mice, maxit=0, **impute_kwargs(case)
        )
        run.generators = _start_from_r(mice, case)
        for maxit in case["maxit"]:
            lost, r_warnings = _recording_warnings(continue_checking_log_warning, mice, maxit=maxit)
            run.steps.append(_snapshot(mice, run.generators, lost, r_warnings))
    return run


def _recording_warnings(call, *args, **kwargs):
    """``call(*args, **kwargs)``; returns what MICE's warning about lost observed values says, and the messages
    of its warnings with R's text (`R_WARNINGS`), in their order.

    The first result maps each column the warning about lost values names to the pair (number of observed
    values lost, reason); it is empty without the warning. Every other warning is passed on.
    """
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        call(*args, **kwargs)
    lost, r_warnings = {}, []
    for warning in caught:
        message = str(warning.message)
        if message in R_WARNINGS and warning.category is UserWarning:
            r_warnings.append(message)
            continue
        if not message.startswith(_LOST_WARNING):
            warnings.warn_explicit(warning.message, warning.category, warning.filename, warning.lineno)
            continue
        assert not lost, "the warning about lost observed values appeared twice"
        lost = {col: (int(n), reason) for col, n, reason in _LOST_COUNT.findall(message)}
        assert lost, f"no column read from the warning {message!r}"
    return lost, r_warnings


def _matrix(payload):
    """A named matrix written by ``named_matrix()`` in l1_chain.R as a DataFrame."""
    return pd.DataFrame(
        np.array(payload["values"], dtype=int), index=str_list(payload["rows"]), columns=str_list(payload["cols"])
    )


def _r_values(series, values):
    """R's values of the cells of `series` (a JSON list, null for NA) in the dtype of `series`."""
    if isinstance(series.dtype, pd.CategoricalDtype):
        labels = [None if v is None else str(v) for v in values]
        return pd.Categorical(labels, categories=series.cat.categories, ordered=series.cat.ordered)
    return as_array(values)


def _generator(case):
    """A strict generator with the draws of the case; without any for a deterministic or spy case."""
    draws = case["draws"]
    if draws is None:
        return ScriptedGenerator(strict=True)
    uniforms = as_array(draws["uniforms"]) if draws["uniforms"] else None
    return ScriptedGenerator(
        chisq=as_array(draws["chisq"]), normals=as_array(draws["normals"]), uniforms=uniforms, strict=True
    )


def _start_from_r(mice, case):
    """Write R's start values into the chain states and give each state its generator; return the generators."""
    start = case["start"]
    generators = []
    for chain, state in enumerate(mice._chain_states):
        columns = [str(col) for col in state.imp]
        assert sorted(columns) == sorted(start), (
            f"{case['name']}: the start has the columns {columns}, R's {list(start)}"
        )
        for col, payload in start.items():
            series = state.imp[col]
            np.testing.assert_array_equal(
                series.index.to_numpy(),
                as_array(payload["rows"], dtype=int),
                err_msg=f"{case['name']}: rows of the start of {col!r}",
            )
            series.iloc[:] = _r_values(series, payload["values"][chain])
        state.rng = _generator(case)
        generators.append(state.rng)
    return generators


def _drawn(generators):
    """The number of values drawn from each pool by all `generators`."""
    counts = dict.fromkeys(_POOLS, 0)
    for rng in generators:
        for pool, n in rng.order:
            counts[pool] = counts.get(pool, 0) + n
    return counts


def _snapshot(mice, generators, lost_observed, r_warnings):
    """The `Step` of `mice` as it is now; `lost_observed` and `r_warnings` from `_recording_warnings`."""
    return Step(
        imp=[{col: values.copy() for col, values in state.imp.items()} for state in mice._chain_states],
        datasets=[dataset.copy() for dataset in mice.imputed_datasets],
        chain_mean={col: values.copy() for col, values in mice.chain_mean.items()},
        chain_var={col: values.copy() for col, values in mice.chain_var.items()},
        logged_events=mice.logged_events.copy(),
        iteration=mice.iteration,
        draws_used=_drawn(generators),
        lost_observed=lost_observed,
        warnings=r_warnings,
    )


def _loop_locals():
    """The local variables of the ``MICE._iterate`` that called the imputer."""
    frame = sys._getframe(1)
    while frame is not None and frame.f_code.co_name != "_iterate":
        frame = frame.f_back
    if frame is None:
        raise AssertionError("the spy was not called from MICE._iterate")
    return frame.f_locals


def _spy(calls):
    """The imputer "spy": appends a `SpyCall` to `calls` and imputes as norm_predict."""

    def spy(y, id_obs, x, id_mis=None, rng=None):
        loop = _loop_locals()
        y = np.array(y, dtype=float)
        calls.append(
            SpyCall(
                it=loop["iter_idx"] + 1,
                im=loop["chain_idx"] + 1,
                dep=str(loop["col"]),
                y=y,
                ry=np.array(id_obs, dtype=bool),
                wy=np.array(id_mis, dtype=bool),
                x_names=[str(name) for name in loop["x"].columns[loop["keep"]]],
                x=np.array(x, dtype=float).reshape(len(y), -1),
            )
        )
        return norm_predict(y, id_obs, x, id_mis=id_mis)

    return spy


# ---------------------------------------------------------------------------
# The comparisons
# ---------------------------------------------------------------------------


def _assert_values(got, expected, message):
    """`got` (a Series) equals R's `expected` (a JSON list): labels identical, numbers to the L1 tolerance.

    Missing values must be at the same positions.
    """
    if isinstance(got.dtype, pd.CategoricalDtype) or got.dtype == object:
        labels = [None if pd.isna(v) else str(v) for v in got]
        r_labels = [None if v is None else str(v) for v in expected]
        assert labels == r_labels, f"{message}: labels {labels} differ from R's {r_labels}"
        return
    np.testing.assert_allclose(
        got.to_numpy(dtype=float, na_value=np.nan), as_array(expected), rtol=RTOL, atol=ATOL, err_msg=message
    )


def _steps(case, run):
    """Pairs (label, R's step, the run's step); the label names the case and the step."""
    assert len(run.steps) == len(case["steps"]), f"{case['name']}: {len(run.steps)} steps, R has {len(case['steps'])}"
    for number, (expected, got) in enumerate(zip(case["steps"], run.steps), start=1):
        yield f"{case['name']}, step {number}", expected, got


def check_imputations(case, run):
    """``imp`` of every chain after every step: R's ``imp``, cell by cell (rows, values, NaN)."""
    for label, expected, got in _steps(case, run):
        for chain, imp in enumerate(got.imp):
            columns = sorted(str(col) for col in imp)
            assert columns == sorted(expected["imp"]), f"{label}, chain {chain + 1}: imp has the columns {columns}"
            for col, payload in expected["imp"].items():
                where = f"{label}, chain {chain + 1}: imputations of {col!r}"
                np.testing.assert_array_equal(
                    imp[col].index.to_numpy(), as_array(payload["rows"], dtype=int), err_msg=f"{where}, rows"
                )
                _assert_values(imp[col], payload["values"][chain], where)


def check_completed(case, run):
    """The completed datasets after every step: R's ``complete(imp, i)``."""
    for label, expected, got in _steps(case, run):
        assert len(got.datasets) == case["m"], f"{label}: {len(got.datasets)} datasets"
        for chain, (dataset, payload) in enumerate(zip(got.datasets, expected["completed"])):
            r_frame = decode_frame(payload)
            columns = [str(col) for col in dataset.columns]
            assert columns == list(r_frame.columns), f"{label}, chain {chain + 1}: columns {columns}"
            for col in r_frame.columns:
                r_values = [None if pd.isna(v) else v for v in r_frame[col].astype(object)]
                _assert_values(dataset[col], r_values, f"{label}, chain {chain + 1}: completed data, {col!r}")


def check_chain_statistics(case, run):
    """``chain_mean`` and ``chain_var`` after every step: R's ``chainMean`` and ``chainVar`` (iteration x chain)."""
    for label, expected, got in _steps(case, run):
        for name, statistics, r_statistics in (
            ("chain_mean", got.chain_mean, expected["chain_mean"]),
            ("chain_var", got.chain_var, expected["chain_var"]),
        ):
            r_statistics = r_statistics or {}
            columns = sorted(str(col) for col in statistics)
            assert columns == sorted(r_statistics), f"{label}: {name} has the columns {columns}"
            for col, r_matrix in r_statistics.items():
                r_values = as_array(r_matrix).reshape(expected["iteration"], case["m"])
                values = statistics[col]
                assert values.shape == r_values.shape, f"{label}: {name} of {col!r} has the shape {values.shape}"
                np.testing.assert_allclose(
                    values, r_values, rtol=RTOL, atol=ATOL, err_msg=f"{label}: {name} of {col!r}"
                )


def _events(frame):
    return {
        name: [int(v) for v in frame[name]] if name in ("it", "im") else [str(v) for v in frame[name]]
        for name in _LOG_COLUMNS
    }


def check_logged_events(case, run):
    """``logged_events`` after every step: R's ``loggedEvents``, row by row."""
    for label, expected, got in _steps(case, run):
        r_events = expected["logged_events"] or {name: [] for name in _LOG_COLUMNS}
        r_events = {
            name: [int(v) for v in np.atleast_1d(r_events[name])] if name in ("it", "im") else str_list(r_events[name])
            for name in _LOG_COLUMNS
        }
        events = _events(got.logged_events)
        assert events == r_events, f"{label}: logged events {events} differ from R's {r_events}"


def check_iteration(case, run):
    """``iteration`` after every step: R's ``imp$iteration``."""
    for label, expected, got in _steps(case, run):
        assert got.iteration == expected["iteration"], (
            f"{label}: iteration {got.iteration}, R's {expected['iteration']}"
        )


def check_setup(case, run):
    """The method of every column, the visit sequence and the predictor matrix: R's."""
    mice = run.mice
    method = {str(col): meth for col, meth in mice.method.items()}
    assert method == case["method"], f"{case['name']}: methods {method} differ from R's {case['method']}"
    visits = [str(col) for col in mice.visit_sequence]
    r_visits = str_list(case["effective_visit_sequence"])
    assert visits == r_visits, f"{case['name']}: visit sequence {visits} differs from R's {r_visits}"
    r_matrix = _matrix(case["predictor_matrix"])
    matrix = mice.predictor_matrix.loc[r_matrix.index, r_matrix.columns].astype(int)
    np.testing.assert_array_equal(matrix.to_numpy(), r_matrix.to_numpy(), err_msg=f"{case['name']}: predictor matrix")


def check_draws(case, run):
    """The draws: none in a deterministic or spy case; in a scripted one every value of R, in R's order, with
    R's degrees of freedom, and after every step as many as R had drawn."""
    for label, expected, got in _steps(case, run):
        r_used = expected["draws_used"] or dict.fromkeys(_POOLS, 0)
        assert got.draws_used == r_used, f"{label}: drawn so far {got.draws_used}, by R {r_used}"
    draws = case["draws"]
    if draws is None:
        return
    (rng,) = run.generators
    rng.assert_exhausted()
    r_order = [(str(pool), int(n)) for pool, n in draws["order"]]
    assert rng.order == r_order, f"{case['name']}: the order of the draws differs from R's"
    df = [call.args["df"] for call in rng.calls if call.method == "chisquare"]
    r_df = [int(v) for v in draws["chisq_df"]]
    assert df == r_df, f"{case['name']}: degrees of freedom {df} of the chi-square draws, R's {r_df}"


def check_calls(case, run):
    """What the spy received, call by call: R's ``y``, ``ry``, ``x`` (with its column names) and ``wy``."""
    r_calls = case["calls"] or []
    # R runs the chains inside each iteration, MICE one chain after the other
    calls = sorted(run.calls, key=lambda call: (call.it, call.im))
    assert len(calls) == len(r_calls), f"{case['name']}: {len(calls)} calls of the imputer, R made {len(r_calls)}"
    for number, (call, r_call) in enumerate(zip(calls, r_calls), start=1):
        label = f"{case['name']}, call {number} ({r_call['dep']}, iteration {r_call['it']}, imputation {r_call['im']})"
        assert (call.it, call.im, call.dep) == (r_call["it"], r_call["im"], r_call["dep"]), (
            f"{label}: called for {call.dep}, iteration {call.it}, imputation {call.im}"
        )
        r_names = str_list(r_call["x_names"])
        assert call.x_names == r_names, f"{label}: the predictors {call.x_names}, R's {r_names}"
        np.testing.assert_array_equal(call.ry, as_bool(r_call["ry"]), err_msg=f"{label}: ry")
        np.testing.assert_array_equal(call.wy, as_bool(r_call["wy"]), err_msg=f"{label}: wy")
        np.testing.assert_allclose(call.y, as_array(r_call["y"]), rtol=RTOL, atol=ATOL, err_msg=f"{label}: y")
        r_x = as_array(r_call["x"]).reshape(len(call.y), len(r_names))
        np.testing.assert_allclose(call.x, r_x, rtol=RTOL, atol=ATOL, err_msg=f"{label}: x")


def check_lost_observed_values(case, run):
    """MICE's warning about lost observed values after the start and after every step: R's number of observed
    where-cells that are NA in the completed data of some chain, per column, and the reason ("no method" for a
    column whose method is ``""``)."""
    expected = case.get("lost_observed") or {"start": {}, "steps": [{}] * len(case["steps"])}
    labelled = [(f"{case['name']}, start", expected["start"], run.start_lost_observed)]
    for (label, _, got), r_lost in zip(_steps(case, run), expected["steps"]):
        labelled.append((label, r_lost, got.lost_observed))
    for label, r_lost, lost in labelled:
        r_counts = {str(col): int(n) for col, n in (r_lost or {}).items()}
        counts = {col: n for col, (n, _) in lost.items()}
        assert counts == r_counts, f"{label}: observed values lost per column {counts}, R's {r_counts}"
        for col, (_, reason) in lost.items():
            r_reason = "no method" if case["method"][col] == "" else "predictors incomplete in these rows"
            assert reason == r_reason, f"{label}: observed values of {col!r} lost for {reason!r}, by R for {r_reason!r}"


def check_warnings(case, run):
    """The warnings with R's text (`R_WARNINGS`): ``impute`` gives those of R's ``mice()`` (``warnings`` of the
    case), ``continue_chains`` none, as R's ``mice.mids()`` does not check its arguments again."""
    r_warnings = str_list(case.get("warnings"))
    assert run.start_warnings == r_warnings, (
        f"{case['name']}: impute warned {run.start_warnings}, R's mice() {r_warnings}"
    )
    for label, _, got in _steps(case, run):
        assert got.warnings == [], f"{label}: continue_chains warned {got.warnings}, R's mice.mids() nothing"


CHECKS = {
    "imputations": check_imputations,
    "completed": check_completed,
    "chain_statistics": check_chain_statistics,
    "logged_events": check_logged_events,
    "iteration": check_iteration,
    "setup": check_setup,
    "draws": check_draws,
    "calls": check_calls,
    "lost_observed_values": check_lost_observed_values,
    "warnings": check_warnings,
}


def check_case(case, run):
    """Every comparison of one case."""
    for check in CHECKS.values():
        check(case, run)
