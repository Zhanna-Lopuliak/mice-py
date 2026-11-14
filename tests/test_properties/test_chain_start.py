"""The start of the chains, R's ``initialize.imp``, through ``impute(maxit=0)``.

With ``maxit=0`` no iteration runs, as in R: each completed dataset holds the
start of its chain, the chain states are kept for ``continue_chains``, and
the chain statistics have no rows. Everything decided before the chains start
is the same as for any other ``maxit``.

R's ``sample`` draws differently from NumPy, so the start cannot be compared
with R draw by draw (the exact chain cases inject R's start values instead).
Its properties are tested here: which cells are started and from which
values, and that ``initial="sample"`` draws with replacement, uniformly over
the observed values (a chi-square test over many chains, and a mutant that
draws from half of the observed values, which the test must catch).

``impute`` has no ``ignore`` argument yet (version 0.4 adds it). The test of
ignored rows sets the private mask ``MICE._ignore`` right before each start.
"""

import importlib

import numpy as np
import pandas as pd
import pytest
from scipy import stats

from imputation import MICE
from imputation.MICE import _complete

from .._logged_events import impute_checking_log_warning
from ._helpers import (
    assert_dtypes_preserved,
    assert_imputed_are_observed,
    assert_observed_unchanged,
    assert_structure_preserved,
    full_predictor_matrix,
    numeric_frame,
    typed_frame,
)

# The module, which the name imputation.MICE (the class) hides
MICE_MODULE = importlib.import_module("imputation.MICE")
NAN = np.nan
N_CHAINS = 4
SEED = 20261113


def _start(df, n_imputations=N_CHAINS, seed=SEED, **kwargs):
    """``impute(maxit=0)`` with the full predictor matrix; returns (mice, datasets)."""
    mice = MICE(df)
    kwargs.setdefault("predictor_matrix", full_predictor_matrix(df))
    out = impute_checking_log_warning(mice, n_imputations=n_imputations, maxit=0, seed=seed, **kwargs)
    return mice, out


def _typed():
    """``typed_frame`` (float, boolean, two categories, text, ordered categories) and an incomplete ``Int64``."""
    df = typed_frame(n=40, seed=11)
    count = pd.array(np.arange(40) % 7, dtype="Int64")
    count[[2, 9, 17, 30]] = pd.NA
    df["count"] = count
    return df


# ---------------------------------------------------------------------------
# maxit=0
# ---------------------------------------------------------------------------


def test_maxit_zero_returns_the_start_and_keeps_the_chains():
    df = _typed()
    before = df.copy(deep=True)
    mice, out = _start(df)
    assert mice.iteration == 0
    assert out is mice.imputed_datasets and len(out) == N_CHAINS
    assert sorted(mice.chain_mean) == sorted(mice.chain_var) == ["count", "num"]
    for statistics in (*mice.chain_mean.values(), *mice.chain_var.values()):
        assert statistics.shape == (0, N_CHAINS)
    assert len(mice._chain_states) == N_CHAINS
    for state, completed in zip(mice._chain_states, out):
        pd.testing.assert_frame_equal(_complete(mice._chain_data, state.imp, mice.where), completed)
    pd.testing.assert_frame_equal(df, before)


def _setup_events_frame():
    """``numeric_frame`` with an incomplete constant column and an incomplete copy of ``d``."""
    df = numeric_frame(n=40, seed=11)
    df["const"] = 3.0
    df["dup"] = df["d"]
    df.iloc[[1, 2], df.columns.get_loc("const")] = NAN
    df.iloc[[3, 4], df.columns.get_loc("dup")] = NAN
    return df


def test_maxit_zero_decides_everything_before_the_chains_as_any_maxit():
    """Methods, setup step, visit sequence (also a random one) and the events
    of the setup step, with their warning (checked by the helper)."""
    df = _setup_events_frame()
    start, _ = _start(df, visit_sequence="random")
    run = MICE(df)
    impute_checking_log_warning(
        run,
        n_imputations=N_CHAINS,
        maxit=2,
        seed=SEED,
        visit_sequence="random",
        predictor_matrix=full_predictor_matrix(df),
    )
    assert start.method == run.method and start.method["const"] == start.method["dup"] == ""
    pd.testing.assert_frame_equal(start.predictor_matrix, run.predictor_matrix)
    assert start.visit_sequence == run.visit_sequence
    assert sorted(start.visit_sequence) == ["a", "b", "c"]
    # only the events of the setup step, as in the run
    assert (start.logged_events["it"] == 0).all() and len(start.logged_events) == 2
    setup = run.logged_events[run.logged_events["it"] == 0]
    pd.testing.assert_frame_equal(start.logged_events, setup.reset_index(drop=True))


# ---------------------------------------------------------------------------
# which cells, which values
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("method", [None, {"num": "norm", "count": "norm"}], ids=["defaults", "norm"])
def test_exactly_the_cells_to_impute_are_started_from_observed_values(method):
    """Every method starts from the observed values, also norm (whose Int64 column becomes Float64)."""
    df = _typed()
    mice, out = _start(df, n_imputations=10, method=method)
    for completed in out:
        assert_structure_preserved(df, completed)
        assert_dtypes_preserved(df, completed, mice)
        assert_observed_unchanged(df, completed)
        assert not completed.isna().any().any()
        assert_imputed_are_observed(df, completed)
    if method is not None:
        assert out[0]["count"].dtype == pd.Float64Dtype()


def _without_method_frame():
    """``y`` is imputed; ``const`` is constant and ``dup`` a copy of ``x1``, so the
    setup step removes their methods; ``u`` gets method "" from the user. All
    four are incomplete."""
    i = np.arange(1, 41)
    x1, x2 = np.sin(i), np.cos(1.3 * i)
    df = pd.DataFrame(
        {
            "y": x1 - x2 + 0.5 * np.sin(3.7 * i),
            "x1": x1,
            "x2": x2,
            "const": 3.0,
            "dup": x1.copy(),
            "u": x2 + 0.5 * np.sin(5.1 * i),
        }
    )
    df.loc[[1, 8, 16], "y"] = NAN
    df.loc[[2, 9], "const"] = NAN
    df.loc[[3, 4], "dup"] = NAN
    df.loc[[5, 6], "u"] = NAN
    return df


def test_columns_without_a_method_are_not_started():
    df = _without_method_frame()
    mice, out = _start(df, method={"u": ""})
    assert mice.method == {"y": "pmm", "x1": "", "x2": "", "const": "", "dup": "", "u": ""}
    assert list(mice.logged_events["meth"]) == ["constant", "collinear"]
    for state, completed in zip(mice._chain_states, out):
        for col in ("const", "dup", "u"):
            assert state.imp[col].isna().all(), col
            pd.testing.assert_series_equal(completed[col], df[col])
        assert completed["y"].notna().all()


def test_meanobs_start():
    """A numeric column gets the observed value closest to its mean (the first
    in row order if two are as close), any other column its most frequent
    observed value (ties: the first category, or the first in sorted order)."""
    df = pd.DataFrame(
        {
            "num": [1.0, 2.0, 10.0, NAN, 4.0, NAN, 3.5, 2.5],  # mean 23 / 6, so 4.0
            "tie": [3.0, 1.0, NAN, 3.0, 1.0, NAN, 1.0, 3.0],  # mean 2: 3.0 (row 0) and 1.0 as close
            "cat": pd.Categorical(["y", None, "x", "x", "y", "z", None, "z"], categories=["z", "y", "x"]),
            "txt": pd.Series(["p", "q", None, "q", "p", "r", None, "r"], dtype=object),
            "flag": pd.Series([True, False, None, True, False, True, None, False], dtype=object),
            "d": [0.3, 0.9, 0.1, 0.7, 0.5, 0.2, 0.8, 0.4],
        }
    )
    mice, out = _start(df, method="sample", initial="meanobs")
    assert mice.visit_sequence == ["num", "tie", "cat", "txt", "flag"]
    for completed in out:
        assert completed.loc[[3, 5], "num"].tolist() == [4.0, 4.0]
        assert completed.loc[[2, 5], "tie"].tolist() == [3.0, 3.0]
        assert completed.loc[[1, 6], "cat"].tolist() == ["z", "z"]
        assert completed.loc[[2, 6], "txt"].tolist() == ["p", "p"]
        assert completed.loc[[2, 6], "flag"].tolist() == [False, False]
        assert completed.dtypes.to_dict() == df.dtypes.to_dict()


# Rows 0 to 9 hold the observed values 0 to 9; rows 10 to 14 decoys 1000 to
# 1004 and are ignored; rows 15 to 20 are missing, 19 and 20 ignored.
IGNORE = np.isin(np.arange(21), [10, 11, 12, 13, 14, 19, 20])


@pytest.fixture
def ignored_rows(monkeypatch):
    """Set ``MICE._ignore`` to ``IGNORE`` right before each start (``impute`` sets it to all False)."""
    calls = []
    original = MICE._initial_imputation

    def with_ignore(self, rng):
        calls.append(1)
        self._ignore = IGNORE.copy()
        return original(self, rng)

    monkeypatch.setattr(MICE, "_initial_imputation", with_ignore)
    return calls


@pytest.mark.parametrize("initial", ["sample", "meanobs"])
def test_start_values_come_from_rows_that_are_not_ignored(ignored_rows, initial):
    """Ignored rows are started too. meanobs: the mean of 0 to 9 is 4.5, so 4
    (the first of 4 and 5); with the decoys it would be 9 or 1000."""
    i = np.arange(21)
    y = np.concatenate([np.arange(10.0), 1000.0 + np.arange(5), np.full(6, NAN)])
    df = pd.DataFrame({"y": y, "x": np.sin(i)})
    mice, out = _start(df, n_imputations=10, method={"y": "pmm"}, initial=initial)
    assert len(ignored_rows) == 10, "the mask was not set before the start"
    missing = df["y"].isna().to_numpy()
    starts = np.concatenate([completed["y"].to_numpy()[missing] for completed in out])
    assert np.isin(starts, np.arange(10.0)).all(), starts[~np.isin(starts, np.arange(10.0))]
    if initial == "meanobs":
        assert (starts == 4.0).all()
    else:
        assert len(np.unique(starts)) > 5


# ---------------------------------------------------------------------------
# the draws of initial="sample"
# ---------------------------------------------------------------------------

# 100 chains of 20 missing cells each, 10 distinct observed values: 200
# expected draws per value. The test asks p >= 1e-6; the start of this seed
# gives p = 0.084 (166 to 229 draws per value), the mutant below p = 0 (no
# draw of five values).
CHI_SQUARE_CHAINS = 100
MIN_P = 1e-6


def _start_value_counts():
    """How often each observed value of ``y`` is a start value, over the chains of ``impute(maxit=0)``."""
    observed = np.arange(10) + 0.5
    df = pd.DataFrame({"y": np.concatenate([observed, np.full(20, NAN)]), "x": np.cos(np.arange(30))})
    mice = MICE(df)
    out = mice.impute(n_imputations=CHI_SQUARE_CHAINS, maxit=0, method={"y": "pmm"}, seed=SEED)
    missing = df["y"].isna().to_numpy()
    starts = np.concatenate([completed["y"].to_numpy()[missing] for completed in out])
    counts = np.array([np.sum(starts == value) for value in observed])
    assert counts.sum() == starts.size == 20 * CHI_SQUARE_CHAINS  # observed values only
    return counts


def test_start_values_are_uniform_draws_with_replacement_from_the_observed_values():
    """20 draws per chain from 10 values: they repeat, so with replacement; and uniform over the values."""
    counts = _start_value_counts()
    assert stats.chisquare(counts).pvalue >= MIN_P, counts


def test_start_from_half_of_the_observed_values_is_caught(monkeypatch):
    calls = []

    def first_half(observed_values, size, rng):
        calls.append(1)
        return rng.choice(observed_values[: len(observed_values) // 2], size=size)

    monkeypatch.setattr(MICE_MODULE, "_sample_start", first_half)
    counts = _start_value_counts()
    assert len(calls) == CHI_SQUARE_CHAINS, "mutant not applied: the start does not call _sample_start"
    assert (counts[5:] == 0).all()
    assert stats.chisquare(counts).pvalue < MIN_P
