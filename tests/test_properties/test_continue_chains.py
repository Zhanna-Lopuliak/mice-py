"""``MICE.continue_chains``: more iterations of the chains of the last ``impute``, as R's ``mice.mids``.

The requirement that defines it: for every way to call ``impute``,
``impute(maxit=a + b, seed=s)`` and ``impute(maxit=a, seed=s)`` followed by
``continue_chains(b)`` give identical results: the completed datasets (every
cell, dtypes, index), ``chain_mean``, ``chain_var``, ``logged_events``,
``visit_sequence``, ``method``, ``predictor_matrix`` and ``iteration``, also
for ``a = 0`` and for a continuation split into several calls. The grid below
runs it over the methods, visit sequences, starts and data that change what a
chain carries from one iteration to the next. The mutants at the end
introduce the defects a continuation can have and check which comparisons
see each of them.

``impute`` has no ``where`` or ``ignore`` argument yet (version 0.4 adds
them). The case with observed where-cells sets the private masks
``MICE._where`` and ``MICE._ignore`` after ``impute(maxit=0)`` and starts the
chains again by hand.
"""

import functools
import importlib
import operator
import warnings

import numpy as np
import pandas as pd
import pytest

from imputation import MICE
from imputation.MICE import _ChainState, _complete

from .._logged_events import continue_checking_log_warning, impute_checking_log_warning
from ._helpers import (
    MAXIT,
    N_IMP,
    full_predictor_matrix,
    impute,
    method_spec,
    numeric_frame,
    register_imputer,
    regression_frame,
    typed_frame,
)
from .test_chain_loop import _empty_method_frame

# The module, which the name imputation.MICE (the class) hides
MICE_MODULE = importlib.import_module("imputation.MICE")
NAN = np.nan
SEED = 20261113


# ---------------------------------------------------------------------------
# the comparison
# ---------------------------------------------------------------------------


def _record(mice, out):
    """The results of a run that the comparison compares: the datasets the last call returned, and attributes."""
    return {
        "datasets": out,
        "chain_mean": mice.chain_mean,
        "chain_var": mice.chain_var,
        "logged_events": mice.logged_events,
        "visit_sequence": mice.visit_sequence,
        "method": mice.method,
        "predictor_matrix": mice.predictor_matrix,
        "iteration": mice.iteration,
    }


def _identical_frames(x, y):
    """Every cell (NaN equal to NaN), dtypes, index and column labels and their types; in object columns
    also the type of every value (True is not 1, None is not NaN)."""
    try:
        pd.testing.assert_frame_equal(x, y, check_exact=True, check_index_type=True, check_column_type=True)
    except AssertionError:
        return False
    return all([type(v) for v in x[col]] == [type(v) for v in y[col]] for col in x.columns if x[col].dtype == object)


def _identical_statistics(x, y):
    return list(x) == list(y) and all(
        x[col].shape == y[col].shape and np.array_equal(x[col], y[col], equal_nan=True) for col in x
    )


COMPARISONS = {
    "datasets": lambda x, y: len(x) == len(y) and all(_identical_frames(a, b) for a, b in zip(x, y)),
    "chain_mean": _identical_statistics,
    "chain_var": _identical_statistics,
    "logged_events": _identical_frames,
    "visit_sequence": operator.eq,
    "method": operator.eq,
    "predictor_matrix": _identical_frames,
    "iteration": operator.eq,
}


def _differences(x, y):
    """The names of the comparisons in which the records `x` and `y` differ."""
    return [name for name, same in COMPARISONS.items() if not same(x[name], y[name])]


# ---------------------------------------------------------------------------
# the cases
# ---------------------------------------------------------------------------


def _numeric(method, **kwargs):
    df = numeric_frame(n=40, seed=11)
    return df, {"method": method_spec(method, df), **kwargs}


def _logged_events_frame():
    """Five observed values of ``y``, six predictors: "df set to 1" in the first iteration, and
    ``remove_lindep`` drops p0 and p4 in every call (see test_setup_step)."""
    i = np.arange(1, 41)
    p = np.column_stack([np.sin((k + 2) * i) for k in range(6)])
    df = pd.DataFrame(p, columns=[f"p{k}" for k in range(6)])
    df.insert(0, "y", p.sum(axis=1) + 0.3 * np.cos(5.3 * i))
    df.loc[5:, "y"] = NAN
    return df


def _int64_frame():
    df = numeric_frame(n=40, seed=11)
    count = pd.array(np.arange(40) % 9, dtype="Int64")
    count[[3, 11, 19, 27, 35]] = pd.NA
    df["count"] = count
    return df


CASES = {
    **{method: functools.partial(_numeric, method) for method in ("pmm", "midas", "cart", "rf", "sample")},
    "norm": functools.partial(_numeric, "norm"),
    "norm.boot": functools.partial(_numeric, "norm.boot"),
    # numeric, boolean, two categories, text with three values, ordered with four levels
    "defaults": lambda: (typed_frame(n=40, seed=11), {}),
    "random_visit": functools.partial(_numeric, "pmm", visit_sequence="random"),
    "list_visit": functools.partial(_numeric, "norm", visit_sequence=["c", "a", "b"]),
    "meanobs": functools.partial(_numeric, "pmm", initial="meanobs"),
    "logged_events": lambda: (_logged_events_frame(), {"method": "pmm"}),
    # z has method "" and predicts the others: NaN where-cells in y and w
    "empty_method_predictor": lambda: (
        _empty_method_frame(),
        {"method": {"y": "norm", "z": "", "w": "norm", "v": "pmm"}},
    ),
    "int64_norm": lambda: (_int64_frame(), {"method": method_spec("norm", _int64_frame())}),
}
N_ITER = 5
SPLITS = [(0, 5), (1, 4), (2, 3), (2, 1, 2)]
GRID = [
    pytest.param(case, splits, SEED, id=f"{case}-{'+'.join(map(str, splits))}") for case in CASES for splits in SPLITS
]
GRID += [
    pytest.param(case, (2, 3), seed, id=f"{case}-2+3-seed{seed}")
    for case in ("pmm", "norm", "defaults", "random_visit")
    for seed in (1, 2)
]


def _run(case, splits, seed):
    """``impute(maxit=splits[0])``, then ``continue_chains`` with each further entry; the record of the run."""
    df, kwargs = CASES[case]()
    mice = MICE(df)
    out = impute_checking_log_warning(
        mice, n_imputations=N_IMP, maxit=splits[0], seed=seed, predictor_matrix=full_predictor_matrix(df), **kwargs
    )
    for maxit in splits[1:]:
        out = continue_checking_log_warning(mice, maxit=maxit)
        assert out is mice.imputed_datasets
    return _record(mice, out)


@functools.cache
def _reference(case, seed):
    """One call of ``impute(maxit=N_ITER)``."""
    return _run(case, (N_ITER,), seed)


@pytest.mark.parametrize("case, splits, seed", GRID)
def test_continuing_equals_one_longer_run(case, splits, seed):
    assert sum(splits) == N_ITER
    assert _differences(_reference(case, seed), _run(case, splits, seed)) == []


def test_the_cases_have_what_they_are_named_after():
    """So that the grid cannot pass because a case lost its point."""
    assert set(_reference("defaults", SEED)["method"].values()) == {"pmm", "logreg", "polyreg", "polr", ""}
    assert _reference("random_visit", SEED)["visit_sequence"] != ["a", "b", "c"]
    assert _reference("list_visit", SEED)["visit_sequence"] == ["c", "a", "b"]
    events = _reference("logged_events", SEED)["logged_events"]
    assert set(events["it"]) == {1, 2, 3, 4, 5}
    df_set = events["out"].str.startswith("df set to 1")
    assert df_set.sum() == N_IMP and set(events.loc[df_set, "it"]) == {1}
    empty = _reference("empty_method_predictor", SEED)
    assert all(completed.loc[[0, 5, 10], "y"].isna().all() for completed in empty["datasets"])
    assert np.isnan(empty["chain_mean"]["w"]).all()
    assert all(
        completed["count"].dtype == pd.Float64Dtype() for completed in _reference("int64_norm", SEED)["datasets"]
    )
    for case in CASES:
        assert all(statistics.shape == (N_ITER, N_IMP) for statistics in _reference(case, SEED)["chain_mean"].values())


# Observed where-cells, set through the private masks: y is imputed also in
# rows 1, 2, 20, 21 and 22, where it is observed; z (method "") is missing in
# rows 0 to 3, so the where-cells of y in rows 1 and 2 cannot be imputed and
# hold NaN in the state, while the working data keep their observed values.
# w is predicted by y. Rows 36 to 39 are ignored.
OBSERVED_WHERE_ROWS = [1, 2, 20, 21, 22]
IGNORED_ROWS = [36, 37, 38, 39]


def _masked_frame():
    rng = np.random.default_rng(7)
    x = rng.normal(size=40)
    z = x + rng.normal(scale=0.5, size=40)
    y = x - z + rng.normal(scale=0.5, size=40)
    df = pd.DataFrame({"y": y, "x": x, "z": z, "w": y + rng.normal(scale=0.5, size=40)})
    df.loc[[0, 1, 2, 3], "z"] = NAN
    df.loc[[10, 11, 12], "y"] = NAN
    df.loc[[30, 31, 32, 33], "w"] = NAN
    return df


def _masked_run(splits, seed=SEED):
    """``impute(maxit=0)`` for the setup; then the masks, a new start of each chain, and ``continue_chains``
    with each entry of `splits`; the record of the run."""
    df = _masked_frame()
    mice = MICE(df)
    mice.impute(n_imputations=N_IMP, maxit=0, method={"y": "norm", "z": "", "w": "norm"}, seed=seed)
    where = df.isna()
    where.loc[OBSERVED_WHERE_ROWS, "y"] = True
    mice._where = where
    mice._ignore = np.isin(np.arange(len(df)), IGNORED_ROWS)
    rngs = [np.random.default_rng([seed, chain]) for chain in range(N_IMP)]
    mice._chain_states = [_ChainState(imp=mice._initial_imputation(rng), rng=rng) for rng in rngs]
    for maxit in splits:
        out = continue_checking_log_warning(mice, maxit=maxit)
    return _record(mice, out)


@pytest.mark.parametrize("splits", [(2, 3), (1, 1, 3)], ids=["2+3", "1+1+3"])
def test_continuing_with_observed_where_cells_equals_one_longer_run(splits):
    reference = _masked_run((N_ITER,))
    for completed in reference["datasets"]:
        # observed where-cells: replaced by imputations, or NaN where z is missing
        assert completed.loc[[1, 2], "y"].isna().all() and completed.loc[[20, 21, 22], "y"].notna().all()
    assert _differences(reference, _masked_run(splits)) == []


# ---------------------------------------------------------------------------
# behaviour
# ---------------------------------------------------------------------------


def test_continue_before_impute_raises():
    mice = MICE(numeric_frame(n=20))
    with pytest.raises(ValueError, match=r"^There are no chains to continue: call impute\(\) first\."):
        mice.continue_chains()


@pytest.mark.parametrize("maxit", [-1, 1.5, "2"])
def test_invalid_maxit_raises_and_changes_nothing(maxit):
    mice, out = impute(numeric_frame(n=30), "pmm")
    with pytest.raises(ValueError, match="maxit must be"):
        mice.continue_chains(maxit)
    assert mice.iteration == MAXIT and mice.imputed_datasets is out


def test_maxit_zero_runs_nothing():
    mice, out = impute(regression_frame(), "pmm")
    mice.fit("y ~ x1")
    assert mice.continue_chains(0) is out
    assert mice.iteration == MAXIT and len(mice.model_results) == N_IMP
    assert mice.chain_mean["y"].shape == (MAXIT, N_IMP)


def test_iteration_counts_the_iterations_of_every_call():
    mice, _ = impute(regression_frame(), "norm", maxit=2)
    assert mice.iteration == 2
    mice.continue_chains()
    assert mice.iteration == 3
    mice.continue_chains(2)
    assert mice.iteration == 5 and mice.chain_mean["y"].shape == (5, N_IMP)
    mice.impute(n_imputations=N_IMP, maxit=0, seed=1)
    assert mice.iteration == 0 and mice.chain_mean["y"].shape == (0, N_IMP)
    mice.continue_chains()
    assert mice.iteration == 1


def test_rows_of_the_chain_statistics_are_appended():
    mice, _ = impute(numeric_frame(n=40, seed=11), "norm")
    first = {col: (mice.chain_mean[col].copy(), mice.chain_var[col].copy()) for col in mice.chain_mean}
    mice.continue_chains(3)
    for col, (mean, var) in first.items():
        assert mice.chain_mean[col].shape == mice.chain_var[col].shape == (MAXIT + 3, N_IMP)
        np.testing.assert_array_equal(mice.chain_mean[col][:MAXIT], mean)
        np.testing.assert_array_equal(mice.chain_var[col][:MAXIT], var)
        assert np.isfinite(mice.chain_mean[col]).all() and np.isfinite(mice.chain_var[col]).all()


def test_events_of_the_continuation_carry_its_iteration_numbers():
    """The check "df set to 1" runs in the first iteration of the run only; "p0, p4" is logged in every call."""
    mice, _ = impute(_logged_events_frame(), "pmm", maxit=2)
    continue_checking_log_warning(mice, maxit=3)
    events = mice.logged_events
    assert list(events.columns) == ["it", "im", "dep", "meth", "out"]
    df_set = events[events["out"].str.startswith("df set to 1")]
    assert df_set["it"].tolist() == [1, 1] and df_set["im"].tolist() == [1, 2]
    lindep = events[events["out"] == "p0, p4"]
    assert list(zip(lindep["it"], lindep["im"])) == [(it, im) for it in range(1, 6) for im in (1, 2)]
    # sorted by iteration and imputation, as impute sorts them
    assert list(zip(events["it"], events["im"])) == sorted(zip(events["it"], events["im"]))


def _caught(call, **kwargs):
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        call(**kwargs)
    return [w for w in caught if "logged events" in str(w.message)]


def test_warning_only_when_the_continuation_logs_an_event():
    # events of the setup step only (a constant column): logged_events is not empty, the continuation adds none
    df = regression_frame()
    df["const"] = 3.0
    mice = MICE(df)
    assert len(_caught(mice.impute, n_imputations=N_IMP, maxit=1, seed=1)) == 1
    assert len(mice.logged_events) == 1
    assert _caught(mice.continue_chains, maxit=2) == []
    # events in every iteration: the warning gives the number of all events and points at this call
    mice = MICE(_logged_events_frame())
    _caught(mice.impute, n_imputations=N_IMP, maxit=1, method="pmm", seed=1)
    (warning,) = _caught(mice.continue_chains, maxit=2)
    assert str(warning.message) == f"Number of logged events: {len(mice.logged_events)}. See MICE.logged_events."
    assert len(mice.logged_events) == 8
    assert warning.filename == __file__


def test_continue_clears_the_fitted_models():
    mice, _ = impute(regression_frame(), "pmm")
    mice.fit("y ~ x1")
    mice.pool()
    assert hasattr(mice, "pooled_result")
    mice.continue_chains()
    assert mice.model_results == [] and not hasattr(mice, "pooled_result")
    with pytest.raises(ValueError, match=r"^No model results found\. Please run \.fit\(\) first\.$"):
        mice.pool()


def test_a_run_that_stops_with_an_error_cannot_be_continued(monkeypatch):
    calls = []

    def stops_at_the_fifth_call(y, id_obs, x, id_mis=None, rng=None):
        calls.append(1)
        if len(calls) == 5:
            raise ValueError("stopped")
        return np.full(int(np.sum(id_mis)), 0.5)

    register_imputer(monkeypatch, "stops", stops_at_the_fifth_call)
    mice = MICE(regression_frame())
    mice.impute(n_imputations=2, maxit=1, method="stops", seed=1)  # calls 1 and 2
    # chain 1 runs iterations 2 and 3 (calls 3 and 4), chain 2 stops in iteration 2
    with pytest.raises(ValueError, match="stopped"):
        mice.continue_chains(2)
    assert mice.iteration == 1
    with pytest.raises(ValueError, match="no chains to continue"):
        mice.continue_chains()
    # a new impute can be continued; one that stops before its chains cannot
    mice.impute(n_imputations=2, maxit=1, method="stops", seed=1)
    mice.continue_chains()
    with pytest.raises(ValueError, match="nope"):
        mice.impute(method={"y": "nope"})
    with pytest.raises(ValueError, match="no chains to continue"):
        mice.continue_chains()


# ---------------------------------------------------------------------------
# mutants
# ---------------------------------------------------------------------------
#
# Each applies a defect of the continuation by patching one function for one
# test, checks that it was applied, and states which comparisons of the
# requirement see it (none may pass unnoticed):
#
# * a fresh generator per chain instead of the one in the state;
# * a new start instead of the imputations in the state;
# * the visit sequence drawn again (visit_sequence="random");
# * the iteration counter restarted: events numbered from 1 again and
#   "df set to 1" logged again; the datasets do not change;
# * the working data rebuilt from the completed datasets instead of the
#   state. It shows only with observed where-cells, which the completed data
#   replace and the working data keep, here through the private masks. Under
#   the masks of impute (the missing cells) the two frames are equal, also
#   where a where-cell is NaN in the state (a predictor with method ""): the
#   defect cannot be seen before impute has a where argument;
# * the chain statistics of the earlier iterations overwritten instead of
#   kept.


def _wrapping_continue(monkeypatch, defect):
    """``continue_chains`` that first applies ``defect(mice)``; returns the list of its calls."""
    calls = []
    original = MICE.continue_chains

    def mutant(self, maxit=1):
        calls.append(defect(self))
        return original(self, maxit)

    monkeypatch.setattr(MICE, "continue_chains", mutant)
    return calls


def test_continuing_with_fresh_generators_is_caught(monkeypatch):
    def fresh_generators(mice):
        for chain, state in enumerate(mice._chain_states):
            state.rng = np.random.default_rng(chain)
        return True

    reference = _reference("norm", SEED)
    calls = _wrapping_continue(monkeypatch, fresh_generators)
    mutant = _run("norm", (2, 3), SEED)
    assert calls == [True], "mutant not applied"
    assert _differences(reference, mutant) == ["datasets", "chain_mean", "chain_var"]


def test_continuing_from_a_new_start_is_caught(monkeypatch):
    def new_start(mice):
        for state in mice._chain_states:
            state.imp = mice._initial_imputation(state.rng)
        return True

    reference = _reference("pmm", SEED)
    calls = _wrapping_continue(monkeypatch, new_start)
    mutant = _run("pmm", (2, 3), SEED)
    assert calls == [True], "mutant not applied"
    assert _differences(reference, mutant) == ["datasets", "chain_mean", "chain_var"]


def test_continuing_with_a_redrawn_visit_sequence_is_caught(monkeypatch):
    def redraw(mice):
        before = list(mice.visit_sequence)
        mice._set_visit_sequence("random", np.random.default_rng(0))
        return mice.visit_sequence != before

    reference = _reference("random_visit", SEED)
    calls = _wrapping_continue(monkeypatch, redraw)
    mutant = _run("random_visit", (2, 3), SEED)
    assert calls == [True], "mutant not applied: the redrawn sequence is the same"
    assert _differences(reference, mutant) == [
        "datasets",
        "chain_mean",
        "chain_var",
        "visit_sequence",
    ]


def test_continuing_with_the_iteration_counter_restarted_is_caught(monkeypatch):
    def restart(mice):
        mice.iteration = 0
        return True

    reference = _reference("logged_events", SEED)
    calls = _wrapping_continue(monkeypatch, restart)
    mutant = _run("logged_events", (2, 3), SEED)
    assert calls == [True], "mutant not applied"
    assert _differences(reference, mutant) == ["chain_mean", "chain_var", "logged_events", "iteration"]
    # numbered from 1 again, and check_df run again
    events = mutant["logged_events"]
    assert events["it"].max() == 3
    assert events["out"].str.startswith("df set to 1").sum() == 2 * N_IMP
    assert reference["logged_events"]["out"].str.startswith("df set to 1").sum() == N_IMP


def _working_frame_from_completed_data(monkeypatch):
    """``_working_frame`` that, when a run is continued, returns the completed data; returns, per call,
    whether that frame differs from the right one."""
    calls = []
    original = MICE._working_frame

    def from_completed_data(self, imp):
        if self.iteration == 0:
            return original(self, imp)
        completed = _complete(self._chain_data, imp, self._where)
        calls.append(not completed.equals(original(self, imp)))
        return completed

    monkeypatch.setattr(MICE, "_working_frame", from_completed_data)
    return calls


def test_continuing_from_the_completed_data_is_caught(monkeypatch):
    reference = _masked_run((N_ITER,))
    calls = _working_frame_from_completed_data(monkeypatch)
    mutant = _masked_run((2, 3))
    assert calls == [True] * N_IMP, "mutant not applied, or the case does not show it"
    assert _differences(reference, mutant) == ["datasets", "chain_mean", "chain_var"]


def test_continuing_from_the_completed_data_is_invisible_under_the_masks_of_impute(monkeypatch):
    """The completed data equal the working data while impute imputes the missing cells only, also where a
    where-cell is NaN in the state: the case above needs the private masks."""
    reference = _reference("empty_method_predictor", SEED)
    calls = _working_frame_from_completed_data(monkeypatch)
    mutant = _run("empty_method_predictor", (2, 3), SEED)
    assert calls == [False] * N_IMP
    assert _differences(reference, mutant) == []


def test_chain_statistics_overwritten_instead_of_appended_is_caught(monkeypatch):
    reference = _reference("pmm", SEED)
    calls = []
    original = MICE_MODULE._with_rows

    def overwritten(statistics, n_rows):
        if len(statistics) == 0:
            return original(statistics, n_rows)
        calls.append(1)
        return np.full((len(statistics) + n_rows, statistics.shape[1]), np.nan)

    monkeypatch.setattr(MICE_MODULE, "_with_rows", overwritten)
    mutant = _run("pmm", (2, 3), SEED)
    assert calls, "mutant not applied: continue_chains does not call _with_rows"
    assert _differences(reference, mutant) == ["chain_mean", "chain_var"]
