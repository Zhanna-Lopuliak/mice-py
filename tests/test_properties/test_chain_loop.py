"""The chain loop of ``MICE.impute``, after R's ``sampler`` and ``sampler.univ``.

Each chain keeps a state ``(imp, rng)``: ``imp[column]`` holds the current
imputation of every where-cell of the column (a cell that gets an
imputation, R's ``where``). The working data are the data with the missing
where-cells filled from ``imp``; an observed where-cell keeps its observed
value there. For a visited column, the model is fitted on the rows with
complete predictors (``cx``), an observed cell and no ``ignore``; the
imputer fills the where-cells inside ``cx`` and receives all rows. A
where-cell outside ``cx`` becomes NaN in ``imp``. A completed dataset is a
copy of the data with every where-cell replaced by ``imp``, and the chain
statistics are computed from ``imp`` at the end of each iteration.

``impute`` has no ``where`` or ``ignore`` argument yet (version 0.4 adds
them): it uses the missing cells and no row. The tests with other masks set
the private attributes ``MICE._where`` and ``MICE._ignore`` and call the
loop's methods directly.
"""

import warnings

import numpy as np
import pandas as pd
import pytest

from imputation.MICE import (
    _ChainState,
    _complete,
    _fit_rows,
    _refresh_working_column,
    _rows_to_impute,
    _store_imputations,
)

from ._helpers import full_predictor_matrix, register_imputer

T, F = True, False
NAN = np.nan

# One target column on eight rows, by hand:
#   row  observed  where  ignore  cx (predictors complete)
#   10   yes       yes    -       yes   observed where-cell: fitted on and imputed
#   11   yes       -      yes     yes   ignored: in neither mask
#   12   -         yes    -       yes   missing where-cell: imputed
#   13   -         -      -       yes   missing, not a where-cell: in neither mask
#   14   yes       yes    -       -     observed where-cell outside cx
#   15   -         yes    -       -     missing where-cell outside cx
#   16   yes       -      -       yes   fitted on
#   17   yes       -      -       yes   fitted on
INDEX = [10, 11, 12, 13, 14, 15, 16, 17]
OBSERVED = np.array([T, T, F, F, T, F, T, T])
WHERE = np.array([T, F, T, F, T, T, F, F])
IGNORE = np.array([F, T, F, F, F, F, F, F])
CX = np.array([T, T, T, T, F, F, T, T])


def _frame():
    """``y`` the target, ``x`` complete, ``z`` (method "") missing in rows 14 and 15, so ``cx`` is as above."""
    return pd.DataFrame(
        {
            "y": [1.0, 40.0, NAN, NAN, 5.0, NAN, 7.0, 8.0],
            "x": [0.1, 0.4, 0.3, 0.2, 0.5, 0.6, 0.9, 0.7],
            "z": [3.0, 2.0, 4.0, 3.0, NAN, NAN, 1.0, 5.0],
        },
        index=INDEX,
    )


# ---------------------------------------------------------------------------
# the helpers, on the masks above
# ---------------------------------------------------------------------------


def test_fit_rows_are_complete_observed_and_not_ignored():
    np.testing.assert_array_equal(_fit_rows(CX, OBSERVED, IGNORE), [T, F, F, F, F, F, T, T])


def test_rows_to_impute_are_where_cells_with_complete_predictors():
    np.testing.assert_array_equal(_rows_to_impute(CX, WHERE), [T, F, T, F, F, F, F, F])


def test_store_imputations_writes_nan_outside_cx():
    imp = pd.Series([5.0, 5.0, 5.0, 5.0], index=[10, 12, 14, 15])
    _store_imputations(imp, CX[WHERE], np.array([100.0, 102.0]))
    pd.testing.assert_series_equal(imp, pd.Series([100.0, 102.0, NAN, NAN], index=[10, 12, 14, 15]))


def test_store_imputations_without_any_imputable_cell_clears_the_start_values():
    imp = pd.Series([5.0, 5.0], index=[14, 15])
    _store_imputations(imp, np.array([F, F]), None)
    assert imp.isna().all()


def test_refresh_fills_only_the_missing_where_cells():
    working = _frame()
    working.loc[[12, 15], "y"] = [5.0, 5.0]  # start values
    imp = pd.Series([100.0, 102.0, 104.0, NAN], index=[10, 12, 14, 15])
    _refresh_working_column(working, "y", imp, WHERE, OBSERVED)
    # 10 and 14 are observed where-cells: they keep their observed values;
    # 15 gets the missing imputation; 13 is not a where-cell
    expected = _frame()
    expected["y"] = [1.0, 40.0, 102.0, NAN, 5.0, NAN, 7.0, 8.0]
    pd.testing.assert_frame_equal(working, expected)


def test_complete_replaces_every_where_cell():
    data = _frame()
    data["g"] = pd.Categorical(["a", "b", None, "a", "b", "a", "b", "a"], categories=["b", "a"])
    before = data.copy(deep=True)
    where = pd.DataFrame(False, index=data.index, columns=data.columns)
    where["y"] = WHERE
    where.loc[[14, 15], "z"] = True
    where.loc[12, "g"] = True
    imp = {
        "y": pd.Series([100.0, 102.0, NAN, 106.0], index=[10, 12, 14, 15]),
        "z": pd.Series([NAN, 9.0], index=[14, 15]),
        "g": pd.Series(pd.Categorical(["a"], categories=["b", "a"]), index=[12]),
    }
    completed = _complete(data, imp, where)
    expected = before.copy()
    # the observed where-cell 10 gets its imputation, the observed where-cell 14 becomes missing
    expected["y"] = [100.0, 40.0, 102.0, NAN, NAN, 106.0, 7.0, 8.0]
    expected["z"] = [3.0, 2.0, 4.0, 3.0, NAN, 9.0, 1.0, 5.0]
    expected.loc[12, "g"] = "a"
    pd.testing.assert_frame_equal(completed, expected)
    pd.testing.assert_frame_equal(data, before)
    # a new frame, independent of imp
    imp["y"].iloc[0] = -1.0
    assert completed.loc[10, "y"] == 100.0


# ---------------------------------------------------------------------------
# the loop with an observed where-cell and an ignored row
# ---------------------------------------------------------------------------


@pytest.fixture
def spy(monkeypatch):
    """The method "spy": records what it receives and imputes 100 plus the position of each row to impute."""
    calls = []

    def imputer(y, id_obs, x, id_mis=None, rng=None):
        calls.append(
            {
                "y": np.array(y, dtype=float),
                "id_obs": np.array(id_obs, dtype=bool),
                "id_mis": np.array(id_mis, dtype=bool),
                "x": np.array(x, dtype=float),
            }
        )
        return 100.0 + np.flatnonzero(id_mis)

    register_imputer(monkeypatch, "spy", imputer)
    return calls


def _mice_with_masks(spy):
    """A run of ``impute`` sets up method, predictors and visit sequence; then the masks of the table above."""
    from imputation import MICE

    df = _frame()
    mice = MICE(df)
    mice.impute(
        n_imputations=1,
        maxit=1,
        method={"y": "spy", "z": ""},
        initial="meanobs",
        predictor_matrix=full_predictor_matrix(df),
        seed=1,
    )
    assert mice.visit_sequence == ["y"] and mice._predictors["y"] == ["x", "z"]
    spy.clear()
    where = pd.DataFrame(False, index=df.index, columns=df.columns)
    where["y"] = WHERE
    where["z"] = df["z"].isna()
    mice._where = where
    mice._ignore = IGNORE.copy()
    mice._log = []
    return mice


def test_start_takes_values_of_rows_that_are_not_ignored(spy):
    """meanobs over rows 10, 14, 16, 17 (mean 5.25) gives 5.0; with the ignored row 11 (40.0) it would give 8.0.
    The column without a method is not started."""
    mice = _mice_with_masks(spy)
    imp = mice._initial_imputation(np.random.default_rng(0))
    assert list(imp) == ["y", "z"]
    pd.testing.assert_series_equal(imp["y"], pd.Series([5.0] * 4, index=[10, 12, 14, 15], name="y"))
    pd.testing.assert_series_equal(imp["z"], pd.Series([NAN, NAN], index=[14, 15], name="z"))
    working = mice._working_frame(imp)
    expected = _frame()
    expected["y"] = [1.0, 40.0, 5.0, NAN, 5.0, 5.0, 7.0, 8.0]
    pd.testing.assert_frame_equal(working, expected)


def test_one_iteration_with_an_observed_where_cell_and_an_ignored_row(spy):
    mice = _mice_with_masks(spy)
    rng = np.random.default_rng(0)
    state = _ChainState(imp=mice._initial_imputation(rng), rng=rng)
    working = mice._working_frame(state.imp)
    mice._iterate(working, state, 0, 0)

    # the imputer gets all rows: y of the working data (15 still holds its
    # start value), R's ry and wy, and NaN in z in rows 14 and 15
    (call,) = spy
    np.testing.assert_array_equal(call["y"], [1.0, 40.0, 5.0, NAN, 5.0, 5.0, 7.0, 8.0])
    np.testing.assert_array_equal(call["id_obs"], [T, F, F, F, F, F, T, T])
    np.testing.assert_array_equal(call["id_mis"], [T, F, T, F, F, F, F, F])
    np.testing.assert_array_equal(call["x"], _frame()[["x", "z"]].to_numpy())
    # check.df counts the three fit rows
    assert mice._log == [(1, 1, "y", "spy", "df set to 1. # observed cases: 3  # predictors: 3")]

    expected_imp = pd.Series([100.0, 102.0, NAN, NAN], index=[10, 12, 14, 15], name="y")
    pd.testing.assert_series_equal(state.imp["y"], expected_imp)
    expected_working = _frame()
    expected_working["y"] = [1.0, 40.0, 102.0, NAN, 5.0, NAN, 7.0, 8.0]
    pd.testing.assert_frame_equal(working, expected_working)

    completed = _complete(mice._chain_data, state.imp, mice._where)
    expected_completed = _frame()
    expected_completed["y"] = [100.0, 40.0, 102.0, NAN, NAN, NAN, 7.0, 8.0]
    pd.testing.assert_frame_equal(completed, expected_completed)

    # over all where-cells of imp, the missing ones skipped: 100 and 102
    assert mice.chain_mean["y"][0, 0] == 101.0
    assert mice.chain_var["y"][0, 0] == 2.0


def test_second_iteration_sees_nan_instead_of_the_start_value(spy):
    mice = _mice_with_masks(spy)
    rng = np.random.default_rng(0)
    state = _ChainState(imp=mice._initial_imputation(rng), rng=rng)
    mice.chain_mean = {"y": np.full((2, 1), NAN)}
    mice.chain_var = {"y": np.full((2, 1), NAN)}
    completed = mice._impute_once(state, 0, range(2))
    assert len(spy) == 2
    np.testing.assert_array_equal(spy[1]["y"], [1.0, 40.0, 102.0, NAN, 5.0, NAN, 7.0, 8.0])
    assert completed["y"].tolist()[:3] == [100.0, 40.0, 102.0]
    assert completed.loc[[13, 14, 15], "y"].isna().all()
    np.testing.assert_array_equal(mice.chain_mean["y"][:, 0], [101.0, 101.0])


# ---------------------------------------------------------------------------
# where-cells outside cx under impute (a predictor with method "")
# ---------------------------------------------------------------------------


def _empty_method_frame():
    """``z`` (method "") predicts ``y`` and ``w``. ``y`` is missing in rows 0, 5, 10, 15, 20, 25, three of
    them where ``z`` is missing; ``w`` only where ``z`` is missing, so none of its cells can be imputed.
    ``v`` has a single missing cell."""
    rng = np.random.default_rng(31)
    n = 60
    x = rng.normal(size=n)
    z = x + rng.normal(scale=0.5, size=n)
    df = pd.DataFrame(
        {
            "y": x - z + rng.normal(scale=0.5, size=n),
            "x": x,
            "z": z,
            "w": x + rng.normal(size=n),
            "v": x + rng.normal(size=n),
        }
    )
    df.loc[[0, 5, 10, 15, 20, 25], "y"] = NAN
    df.loc[[0, 5, 10, 30, 35, 40, 45], "z"] = NAN
    df.loc[[30, 35], "w"] = NAN
    df.loc[[50], "v"] = NAN
    return df


@pytest.mark.parametrize("method", ["pmm", "norm"])
def test_where_cells_outside_cx_end_as_nan_not_as_start_values(method, monkeypatch):
    from imputation import MICE

    starts = []
    original = MICE._initial_imputation

    def recording(self, rng):
        imp = original(self, rng)
        starts.append({col: values.copy() for col, values in imp.items()})
        return imp

    monkeypatch.setattr(MICE, "_initial_imputation", recording)
    df = _empty_method_frame()
    mice = MICE(df)
    with warnings.catch_warnings():
        warnings.simplefilter("error", RuntimeWarning)
        out = mice.impute(n_imputations=2, maxit=2, method={"y": method, "z": "", "w": method, "v": method}, seed=3)
    assert mice.visit_sequence == ["y", "w", "v"]
    for start, state, completed in zip(starts, mice._chain_states, out):
        # every where-cell had a start value
        assert all(values.notna().all() for col, values in start.items() if col != "z")
        # y: the three cells with z missing are NaN in imp and in the data
        assert state.imp["y"].loc[[0, 5, 10]].isna().all()
        assert state.imp["y"].loc[[15, 20, 25]].notna().all()
        assert completed.loc[[0, 5, 10], "y"].isna().all()
        # w: no cell can be imputed
        assert state.imp["w"].isna().all()
        assert completed.loc[[30, 35], "w"].isna().all()
        # z has no method: its where-cells stay missing
        assert state.imp["z"].isna().all()
    # chain statistics without a value are NaN, as R's NA, and without a RuntimeWarning
    assert np.isnan(mice.chain_mean["w"]).all() and np.isnan(mice.chain_var["w"]).all()
    assert np.isfinite(mice.chain_mean["v"]).all() and np.isnan(mice.chain_var["v"]).all()
    assert np.isfinite(mice.chain_mean["y"]).all() and np.isfinite(mice.chain_var["y"]).all()


# ---------------------------------------------------------------------------
# chain state, copies, chain statistics
# ---------------------------------------------------------------------------


def test_impute_keeps_one_state_per_chain_and_completes_from_it():
    from imputation import MICE

    df = _empty_method_frame()
    mice = MICE(df)
    out = mice.impute(n_imputations=3, maxit=2, method={"z": ""}, seed=7)
    assert len(mice._chain_states) == 3
    for state, completed in zip(mice._chain_states, out):
        assert isinstance(state.rng, np.random.Generator)
        assert sorted(state.imp) == ["v", "w", "y", "z"]
        for col, values in state.imp.items():
            assert list(values.index) == list(df.index[df[col].isna()])
            assert values.dtype == df[col].dtype
        pd.testing.assert_frame_equal(_complete(mice._chain_data, state.imp, mice._where), completed)
    # the generators went on independently
    assert len({state.rng.bit_generator.state["state"]["state"] for state in mice._chain_states}) == 3


def test_completed_datasets_are_independent_copies():
    from imputation import MICE

    df = _empty_method_frame()
    df["k"] = pd.array([1, 2, None, 4] * 15, dtype="Int64")
    df["g"] = pd.Categorical(["a", "b", "c", None, "a", "b"] * 10)
    before = df.copy(deep=True)
    mice = MICE(df)
    out = mice.impute(n_imputations=2, maxit=1, method={"z": "", "g": "pmm"}, seed=2)
    snapshot = [completed.copy(deep=True) for completed in out]
    imp_before = {col: values.copy() for col, values in mice._chain_states[0].imp.items()}

    # change every cell of the first dataset in place
    changed = {"y": -99.0, "x": -99.0, "z": -99.0, "w": -99.0, "v": -99.0, "k": -99, "g": "c"}
    for col, value in changed.items():
        out[0].loc[:, col] = pd.Series(value, index=out[0].index, dtype=out[0][col].dtype)
    assert (out[0]["g"] == "c").all() and (out[0]["k"] == -99).all()
    pd.testing.assert_frame_equal(out[1], snapshot[1])
    for col, values in mice._chain_states[0].imp.items():
        pd.testing.assert_series_equal(values, imp_before[col])
    pd.testing.assert_frame_equal(mice.data, before)
    pd.testing.assert_frame_equal(mice._chain_data, before)

    # and every imputation of the second chain's state
    for values in mice._chain_states[1].imp.values():
        values.iloc[:] = NAN
    pd.testing.assert_frame_equal(out[1], snapshot[1])


def test_chain_statistics_are_computed_after_the_iteration_from_imp(monkeypatch):
    """The imputer of each column is called first; then the statistics of
    every column, from ``imp`` as it is at the end of the iteration."""
    from imputation import MICE

    events = []

    def imputer(y, id_obs, x, id_mis=None, rng=None):
        events.append("impute")
        return np.arange(int(np.sum(id_mis)), dtype=float)

    register_imputer(monkeypatch, "fixed", imputer)
    original = MICE._record_chain_statistics

    def recording(self, imp, iter_idx, chain_idx):
        events.append(("statistics", iter_idx, chain_idx))
        original(self, imp, iter_idx, chain_idx)
        for col in self.chain_mean:
            values = imp[col].to_numpy(dtype=float)
            assert self.chain_mean[col][iter_idx, chain_idx] == np.nanmean(values)
            assert self.chain_var[col][iter_idx, chain_idx] == np.nanvar(values, ddof=1)

    monkeypatch.setattr(MICE, "_record_chain_statistics", recording)
    df = _empty_method_frame()[["y", "x", "v"]]
    df.loc[[1, 2], "v"] = NAN
    mice = MICE(df)
    mice.impute(n_imputations=2, maxit=2, method="fixed", seed=1)
    assert mice.visit_sequence == ["y", "v"]
    expected = []
    for chain in range(2):
        for iteration in range(2):
            expected += ["impute", "impute", ("statistics", iteration, chain)]
    assert events == expected
    # y: 0 to 5, v: 0 to 2
    np.testing.assert_array_equal(mice.chain_mean["y"], np.full((2, 2), 2.5))
    np.testing.assert_array_equal(mice.chain_mean["v"], np.full((2, 2), 1.0))


# ---------------------------------------------------------------------------
# impute clears the fitted models
# ---------------------------------------------------------------------------


def test_pool_after_a_new_impute_needs_a_new_fit():
    from imputation import MICE

    df = _empty_method_frame()[["y", "x", "v"]]
    mice = MICE(df)
    mice.impute(n_imputations=2, maxit=1, seed=1)
    mice.fit("y ~ x")
    mice.pool()
    assert hasattr(mice, "pooled_result")

    mice.impute(n_imputations=2, maxit=1, seed=2)
    assert mice.model_results == []
    assert not hasattr(mice, "pooled_result")
    with pytest.raises(ValueError, match=r"^No model results found\. Please run \.fit\(\) first\.$"):
        mice.pool()
    mice.fit("y ~ x")
    assert len(mice.pool()["pooled_params"]) == 2
