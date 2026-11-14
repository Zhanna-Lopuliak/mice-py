"""``impute(where=...)``: the cells that get an imputation, as R's ``where``.

A where-cell gets an imputation in the completed data, also where it is
observed; a missing cell that ``where`` leaves out stays missing. The data
the chains work on hold the imputations of the missing where-cells only, so
an observed where-cell keeps its observed value there and the other columns
are predicted from it. A where-cell becomes missing when its column has no
method or its predictors are incomplete in its row at the visit, also over
an observed value, and ``impute`` then warns.

The expectations below are computed by hand or by a few lines of NumPy with
the deterministic methods ``mean`` and ``norm.predict``: the mean of the
observed values, and the least-squares prediction from the observed rows.
R's results for the same cases are fixtures of a later phase; the rules
tested here are those of R's ``check.where``, ``make.method``,
``check.visitSequence``, ``mice.edit.predictorMatrix``, ``sampler.univ`` and
``complete`` (mice 3.19.0).

The mutants at the end introduce the defects that ``where`` can have, each
by patching one function, and check that a named test of this file fails.
"""

import importlib
import warnings

import numpy as np
import pandas as pd
import pytest

from imputation import MICE, make_method
from imputation.validators import check_where

from ._helpers import numeric_frame

MICE_MODULE = importlib.import_module("imputation.MICE")
NAN = np.nan
T, F = True, False


def _run(df, **kwargs):
    """``MICE(df).impute(**kwargs)`` with two chains and two iterations by default; returns (mice, datasets,
    the messages of the UserWarnings)."""
    kwargs = {"n_imputations": 2, "maxit": 2, "seed": 1, **kwargs}
    mice = MICE(df)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        out = mice.impute(**kwargs)
    return mice, out, [str(w.message) for w in caught if issubclass(w.category, UserWarning)]


def _fitted(y, x, fit_rows, rows):
    """The least-squares prediction of `y` from `x` with an intercept, fitted on `fit_rows`, at `rows`
    (what ``norm.predict`` imputes)."""
    design = np.column_stack([np.ones(len(y)), x])
    coef, *_ = np.linalg.lstsq(design[fit_rows], y[fit_rows], rcond=None)
    return design[rows] @ coef


# ---------------------------------------------------------------------------
# check_where: the accepted forms
# ---------------------------------------------------------------------------


def _small():
    return pd.DataFrame({"x": [1.0, NAN, 3.0], "y": [NAN, 2.0, 5.0], "g": ["a", None, "b"]}, index=[7, 3, 5])


@pytest.mark.parametrize(
    "where, expected",
    [
        (None, [[F, T, F], [T, F, T], [F, F, F]]),
        ("missing", [[F, T, F], [T, F, T], [F, F, F]]),
        ("all", [[T, T, T], [T, T, T], [T, T, T]]),
        ("observed", [[T, F, T], [F, T, F], [T, T, T]]),
        ("none", [[F, F, F], [F, F, F], [F, F, F]]),
    ],
    ids=["None", "missing", "all", "observed", "none"],
)
def test_check_where_keywords(where, expected):
    df = _small()
    got = check_where(where, df)
    pd.testing.assert_frame_equal(got, pd.DataFrame(expected, index=df.index, columns=df.columns))


def test_check_where_aligns_a_frame_by_column_label():
    df = _small()
    where = pd.DataFrame({"g": [T, F, F], "x": [F, F, T], "y": [F, T, F]}, index=df.index)
    got = check_where(where, df)
    assert list(got.columns) == ["x", "y", "g"] and got.index.equals(df.index)
    assert got["x"].tolist() == [F, F, T] and got["y"].tolist() == [F, T, F] and got["g"].tolist() == [T, F, F]
    # a new frame
    where.iloc[:, :] = True
    assert not got.to_numpy().all()


def test_check_where_takes_nullable_and_object_booleans():
    df = _small()
    where = pd.DataFrame(
        {
            "x": pd.array([T, F, T], dtype="boolean"),
            "y": np.array([np.True_, F, T], dtype=object),
            "g": np.array([F, F, T]),
        },
        index=df.index,
    )
    got = check_where(where, df)
    assert all(dtype == np.dtype(bool) for dtype in got.dtypes)
    assert got.to_numpy().tolist() == [[T, T, F], [F, F, F], [T, T, T]]


def test_check_where_takes_a_boolean_array_in_data_order():
    df = _small()
    array = np.array([[T, F, F], [F, T, F], [F, F, T]])
    got = check_where(array, df)
    pd.testing.assert_frame_equal(got, pd.DataFrame(array, index=df.index, columns=df.columns))
    array[:] = True
    assert not got.to_numpy().all()


# ---------------------------------------------------------------------------
# check_where: the errors
# ---------------------------------------------------------------------------


def _frame_like(df, values, **kwargs):
    return pd.DataFrame(values, index=kwargs.get("index", df.index), columns=kwargs.get("columns", df.columns))


@pytest.mark.parametrize(
    "where, message",
    [
        (
            "missings",
            r"^where must be None, one of 'missing', 'all', 'observed', 'none', a boolean DataFrame with "
            r"the index and the columns of the data, or a boolean NumPy array of the shape of the data; got the "
            r"string 'missings'\.$",
        ),
        ([[True] * 3] * 3, r"^where must be None, .*; got list\.$"),
        (
            np.ones((3, 2), dtype=bool),
            r"^where has the shape \(3, 2\), the data \(3, 3\): the array must have one entry per cell of the data\.$",
        ),
        (np.ones(9, dtype=bool), r"^where has the shape \(9,\), the data \(3, 3\)"),
        (
            np.ones((3, 3), dtype=int),
            r"^where must hold booleans, True or False; the array has dtype int64\. "
            r"Numbers, also 0 and 1, are not accepted: convert them with \.astype\(bool\)\.$",
        ),
        (
            np.array([[1.0, 0.0, NAN]] * 3),
            r"^where contains missing values \(the array\): every cell must be True or False\.$",
        ),
        (np.array([[T, F, None]] * 3, dtype=object), r"^where contains missing values \(the array\)"),
        (
            np.array([[T, F, "yes"]] * 3, dtype=object),
            r"^where must hold booleans, True or False; the array has dtype object\.",
        ),
    ],
    ids=["keyword", "list", "array-shape", "array-1d", "array-numbers", "array-nan", "array-none", "array-text"],
)
def test_check_where_errors(where, message):
    with pytest.raises(ValueError, match=message):
        check_where(where, _small())


def test_check_where_frame_errors():
    df = _small()
    with pytest.raises(ValueError, match=r"^where has the shape \(2, 3\), the data \(3, 3\): where must have one "):
        check_where(_frame_like(df, [[T, T, T]] * 2, index=[7, 3]), df)
    with pytest.raises(
        ValueError,
        match=r"^where must have the column labels of the data, in any order; it has the columns \['z'\], which "
        r"are not in the data, and not the columns \['g'\]\.$",
    ):
        check_where(_frame_like(df, [[T, T, T]] * 3, columns=["x", "y", "z"]), df)
    other_index = "where has another index than the data: it must have the index of the data, the same labels in the"
    with pytest.raises(ValueError, match=rf"^{other_index} same order\.$"):
        check_where(_frame_like(df, [[T, T, T]] * 3, index=[0, 1, 2]), df)
    with pytest.raises(ValueError, match=rf"^{other_index}"):
        check_where(_frame_like(df, [[T, T, T]] * 3, index=[3, 7, 5]), df)
    with pytest.raises(ValueError, match=r"^where must hold booleans, True or False; column 'y' has dtype int64\."):
        check_where(pd.DataFrame({"x": [T, F, T], "y": [1, 0, 1], "g": [T, T, T]}, index=df.index), df)
    with pytest.raises(ValueError, match=r"^where contains missing values \(column 'x'\): every cell must be True "):
        check_where(
            pd.DataFrame({"x": pd.array([T, None, T], dtype="boolean"), "y": [T] * 3, "g": [T] * 3}, index=df.index), df
        )
    with pytest.raises(ValueError, match=r"^where contains missing values \(column 'g'\)"):
        check_where(pd.DataFrame({"x": [T] * 3, "y": [T] * 3, "g": [T, None, F]}, index=df.index), df)


def test_impute_and_make_method_check_where():
    df = _small()
    with pytest.raises(ValueError, match="^where contains missing values"):
        MICE(df).impute(where=np.array([[T, F, NAN]] * 3))
    with pytest.raises(ValueError, match="^where must be None"):
        make_method(df, where="everything")


# ---------------------------------------------------------------------------
# setup: methods, visit sequence, predictor matrix
# ---------------------------------------------------------------------------


def _nhanes_like():
    """25 rows shaped like R's nhanes: ``age`` complete (int64), ``bmi``, ``hyp`` and ``chl`` with 9, 8 and 10
    missing cells."""
    rng = np.random.default_rng(7)
    n = 25
    age = rng.integers(1, 4, size=n)
    df = pd.DataFrame(
        {
            "age": age,
            "bmi": 22 + 2 * age + rng.normal(scale=2, size=n),
            "hyp": 1 + 0.3 * age + rng.normal(scale=0.4, size=n),
            "chl": 150 + 20 * age + rng.normal(scale=20, size=n),
        }
    )
    for col, n_missing in (("bmi", 9), ("hyp", 8), ("chl", 10)):
        df.loc[rng.choice(n, n_missing, replace=False), col] = NAN
    return df


@pytest.mark.parametrize(
    "where, methods, visits",
    [
        ("missing", ["", "pmm", "pmm", "pmm"], ["bmi", "hyp", "chl"]),
        ("all", ["pmm"] * 4, ["age", "bmi", "hyp", "chl"]),
        ("observed", ["pmm"] * 4, ["age", "bmi", "hyp", "chl"]),
        ("none", [""] * 4, []),
    ],
)
def test_methods_and_visit_sequence_of_each_keyword(where, methods, visits):
    df = _nhanes_like()
    mice, out, _ = _run(df, maxit=0, where=where)
    assert list(mice.method.values()) == methods
    assert mice.visit_sequence == visits
    pd.testing.assert_frame_equal(mice.where, check_where(where, df))
    assert make_method(df, where=where) == dict(zip(df.columns, methods))


def test_make_method_gives_where_cells_the_default_of_their_type():
    df = pd.DataFrame(
        {
            "num": [1.0, NAN, 3.0, 4.0],
            "flag": [T, F, T, F],
            "grade": pd.Categorical(["lo", "mid", "hi", "mid"], categories=["lo", "mid", "hi"], ordered=True),
            "txt": ["a", "b", "c", None],
        }
    )
    where = df.isna()
    where["num"] = False  # incomplete, no where-cell
    where.loc[0, ["flag", "grade"]] = True  # complete, with a where-cell
    assert make_method(df, where=where) == {"num": "", "flag": "logreg", "grade": "polr", "txt": "polyreg"}
    assert make_method(df) == {"num": "pmm", "flag": "", "grade": "", "txt": "polyreg"}


def _visit_frame():
    """The nhanes-like frame with ``where`` the missing cells, except that only 2 of the 10 missing cells of
    ``chl`` are where-cells: chl 2, hyp 8, bmi 9 where-cells, against 10, 8, 9 missing cells."""
    df = _nhanes_like()
    where = df.isna()
    where.loc[where.index[where["chl"]][2:], "chl"] = False
    return df, where


@pytest.mark.parametrize(
    "keyword, expected",
    [
        ("roman", ["bmi", "hyp", "chl"]),
        ("arabic", ["chl", "hyp", "bmi"]),
        ("monotone", ["chl", "hyp", "bmi"]),
        ("revmonotone", ["bmi", "hyp", "chl"]),
    ],
)
def test_visit_keywords_order_by_the_number_of_where_cells(keyword, expected):
    """R's orders for the same counts (R also lists age, never imputed, first under monotone)."""
    df, where = _visit_frame()
    mice, _, _ = _run(df, maxit=0, where=where, visit_sequence=keyword)
    assert mice.visit_sequence == expected


def test_monotone_orders_by_the_number_of_where_cells():
    """By the number of missing cells it would be hyp (8), bmi (9), chl (10)."""
    df, where = _visit_frame()
    mice, _, _ = _run(df, maxit=0, where=where, visit_sequence="monotone")
    assert mice.visit_sequence == ["chl", "hyp", "bmi"], "monotone does not order by the number of where-cells"


def test_complete_column_with_where_cells_gets_the_default_method_and_is_imputed():
    """age (complete, int64) has three where-cells: pmm, visited, its three imputations in the state; as R."""
    df = _nhanes_like()
    where = df.isna()
    where.loc[[0, 1, 2], "age"] = True
    mice, out, _ = _run(df, maxit=1, where=where)
    assert mice.method["age"] == "pmm", "a complete column with where-cells has no method"
    assert mice.visit_sequence == ["age", "bmi", "hyp", "chl"]
    for state, completed in zip(mice._chain_states, out):
        assert list(state.imp["age"].index) == [0, 1, 2]
        assert completed["age"].tolist()[:3] == state.imp["age"].tolist()
        assert completed["age"].tolist()[3:] == df["age"].tolist()[3:]
    assert mice.chain_mean["age"].shape == (1, 2)


@pytest.mark.parametrize("method", [None, "norm", {"bmi": "norm"}], ids=["default", "str", "dict"])
def test_incomplete_column_without_where_cells_gets_no_method(method):
    """As R: method "" for bmi whatever method says, no logged event; its missing cells stay missing, and the
    where-cells of hyp and chl in the rows where bmi is missing too."""
    df = _nhanes_like()
    where = df.isna()
    where["bmi"] = False
    mice, out, messages = _run(df, method=method, where=where)
    assert mice.method["bmi"] == ""
    assert "bmi" not in mice.visit_sequence and mice.logged_events.empty and messages == []
    bmi_missing = df["bmi"].isna()
    for completed in out:
        assert completed["bmi"].isna().equals(bmi_missing)
        for col in ("hyp", "chl"):
            np.testing.assert_array_equal(completed[col].isna(), df[col].isna() & bmi_missing)


def test_predictor_matrix_needs_a_row_for_each_column_that_is_imputed():
    df = _nhanes_like()
    where = df.isna()
    where["bmi"] = False
    # bmi (no where-cells) predicts without a row; chl, imputed, has none
    matrix = pd.DataFrame(1, index=["hyp"], columns=["age", "bmi"])
    with pytest.raises(ValueError, match=r"^predictor_matrix has no row for the columns that are imputed \['chl'\]\."):
        MICE(df).impute(predictor_matrix=matrix, where=where, maxit=0)
    matrix = pd.DataFrame(1, index=["hyp", "chl"], columns=["age", "bmi"])
    mice, out, _ = _run(df, predictor_matrix=matrix, where=where, maxit=1)
    assert mice.visit_sequence == ["hyp", "chl"]
    for completed in out:
        assert completed["bmi"].isna().equals(df["bmi"].isna())
    # a column with method "" needs no row either, also when incomplete
    mice, _, _ = _run(df, predictor_matrix=matrix, method={"bmi": ""}, maxit=0)
    assert mice.visit_sequence == ["hyp", "chl"]


# The predictor matrix that R's mice.edit.predictorMatrix makes from the
# default one for nhanes with visitSequence "monotone" and maxit 1 (the order
# of all columns is age, hyp, bmi, chl): bmi from age and hyp, hyp from age,
# chl from age, bmi and hyp
MONOTONE_MATRIX = [[0, 0, 0, 0], [1, 0, 1, 0], [1, 0, 0, 0], [1, 1, 1, 0]]


def test_monotone_with_one_iteration_edits_the_predictor_matrix():
    df = _nhanes_like()
    mice, _, _ = _run(df, maxit=1, visit_sequence="monotone")
    assert mice.visit_sequence == ["hyp", "bmi", "chl"]
    assert mice.predictor_matrix.to_numpy().tolist() == MONOTONE_MATRIX, "the predictor matrix is not edited"
    assert mice._predictors == {"hyp": ["age"], "bmi": ["age", "hyp"], "chl": ["age", "bmi", "hyp"]}


def test_monotone_edit_orders_by_the_number_of_where_cells():
    """chl has 2 where-cells: the order of all columns is age, chl, hyp, bmi."""
    df, where = _visit_frame()
    mice, _, _ = _run(df, maxit=1, where=where, visit_sequence="monotone")
    expected = pd.DataFrame(
        [[0, 0, 0, 0], [1, 0, 1, 1], [1, 0, 0, 1], [1, 0, 0, 0]], index=df.columns, columns=df.columns
    )
    pd.testing.assert_frame_equal(mice.predictor_matrix, expected)


@pytest.mark.parametrize(
    "maxit, visit_sequence",
    [(2, "monotone"), (0, "monotone"), (1, "revmonotone"), (1, "roman"), (1, ["hyp", "bmi", "chl"])],
    ids=["maxit2", "maxit0", "revmonotone", "roman", "list"],
)
def test_no_monotone_edit_otherwise(maxit, visit_sequence):
    df = _nhanes_like()
    mice, _, _ = _run(df, maxit=maxit, visit_sequence=visit_sequence)
    full = np.ones((4, 4), dtype=int) - np.eye(4, dtype=int)
    assert mice.predictor_matrix.to_numpy().tolist() == full.tolist()


# ---------------------------------------------------------------------------
# the chains, with deterministic methods
# ---------------------------------------------------------------------------


def _xy_frame():
    """``x`` complete; ``y = 1 + 2 x + noise``, missing in rows 0, 3, 6 and 9."""
    rng = np.random.default_rng(3)
    x = rng.normal(size=20)
    df = pd.DataFrame({"x": x, "y": 1 + 2 * x + rng.normal(size=20)})
    df.loc[[0, 3, 6, 9], "y"] = NAN
    return df


def test_observed_where_cell_is_replaced_and_predicts_with_its_observed_value():
    """Every cell of x is a where-cell, imputed by mean; y by norm.predict from x. As in R, the completed x is
    the mean of x everywhere, and y is predicted from the observed x, not from the mean."""
    df = _xy_frame()
    where = df.isna()
    where["x"] = True
    mice, out, messages = _run(df, where=where, method={"x": "mean", "y": "norm.predict"})
    x, y = df["x"].to_numpy(), df["y"].to_numpy()
    missing = np.isnan(y)
    expected_y = y.copy()
    expected_y[missing] = _fitted(y, x, ~missing, missing)
    for completed in out:
        np.testing.assert_allclose(
            completed["y"], expected_y, rtol=1e-10, err_msg="y not predicted from the observed x"
        )
        np.testing.assert_allclose(
            completed["x"], np.full(20, x.mean()), rtol=1e-12, err_msg="observed where-cells of x not replaced"
        )
    assert messages == []


def _left_out_frame():
    """``x`` complete; ``y`` missing in rows 0, 3, 6, 9; ``z`` missing in rows 1 and 12; ``w`` complete.

    ``where``: the missing cells, except y in row 3; plus the observed z in rows 3 and 5 and the observed w in
    row 3. The predictor matrix: y from x, z from x and y, w from y; so the where-cells of z and w in row 3
    cannot be imputed. Returns (data, where, predictor matrix).
    """
    rng = np.random.default_rng(5)
    x = rng.normal(size=20)
    y = 1 + 2 * x + rng.normal(size=20)
    df = pd.DataFrame({"x": x, "y": y, "z": x - y + rng.normal(scale=0.5, size=20), "w": rng.normal(size=20)})
    df.loc[[0, 3, 6, 9], "y"] = NAN
    df.loc[[1, 12], "z"] = NAN
    where = df.isna()
    where.loc[3, "y"] = False
    where.loc[[3, 5], "z"] = True
    where.loc[3, "w"] = True
    predictors = pd.DataFrame(0, index=["y", "z", "w"], columns=df.columns)
    predictors.loc["y", "x"] = 1
    predictors.loc["z", ["x", "y"]] = 1
    predictors.loc["w", "y"] = 1
    return df, where, predictors


LEFT_OUT_METHODS = {"y": "norm.predict", "z": "norm.predict", "w": "mean"}


def test_missing_cell_left_out_of_where_spreads_to_the_where_cells_of_its_row():
    df, where, predictors = _left_out_frame()
    mice, out, messages = _run(df, where=where, method=LEFT_OUT_METHODS, predictor_matrix=predictors)
    assert mice.visit_sequence == ["y", "z", "w"]
    assert messages == [
        "Observed values replaced by missing values in the completed data, because where marks them and they could "
        "not be imputed (as in R); per column: 'z': 1 (predictors incomplete in these rows), 'w': 1 (predictors "
        "incomplete in these rows). See MICE.where."
    ]
    x, y, z = (df[col].to_numpy() for col in ("x", "y", "z"))
    # y from x, in rows 0, 6 and 9; z from x and y (with those imputations) on its observed rows but row 3,
    # imputed in rows 1, 5 (an observed where-cell) and 12
    y_missing = np.isin(np.arange(20), [0, 6, 9])
    y_working = y.copy()
    y_working[y_missing] = _fitted(y, x, ~np.isnan(y), y_missing)
    fit = ~np.isnan(z) & (np.arange(20) != 3)
    rows = np.array([1, 5, 12])
    expected_z = _fitted(z, np.column_stack([x, y_working]), fit, rows)
    for state, completed in zip(mice._chain_states, out):
        np.testing.assert_allclose(completed.loc[y_missing, "y"], y_working[y_missing], rtol=1e-10)
        assert completed.loc[3, ["y", "z", "w"]].isna().all(), "the where-cells of row 3 are not missing"
        assert completed.drop(index=3).notna().all().all()
        np.testing.assert_allclose(completed.loc[rows, "z"], expected_z, rtol=1e-10)
        assert state.imp["z"].index.tolist() == [1, 3, 5, 12] and np.isnan(state.imp["z"][3])
    # w: its only where-cell cannot be imputed in any iteration
    assert np.isnan(mice.chain_mean["w"]).all() and np.isnan(mice.chain_var["w"]).all()


def test_all_where_cells_without_complete_predictors_are_missing_without_error():
    """Every where-cell of z lies in a row where y, its predictor, is missing and left out: imp is all NaN."""
    df, where, predictors = _left_out_frame()
    where["z"] = False
    where.loc[[3], "z"] = True
    mice, out, messages = _run(df, where=where, method={**LEFT_OUT_METHODS, "z": "norm"}, predictor_matrix=predictors)
    for state, completed in zip(mice._chain_states, out):
        assert state.imp["z"].isna().all() and np.isnan(completed.loc[3, "z"])
    assert np.isnan(mice.chain_mean["z"]).all()
    assert len(messages) == 1 and "'z': 1 (predictors incomplete in these rows)" in messages[0]


def test_where_cells_of_a_column_without_method_are_missing():
    """where="all": x has the user's method "", const is removed by the setup step. All their cells are
    missing in the completed data, as in R; y is still predicted from the observed x."""
    df = _xy_frame()
    df["const"] = 4.0
    mice, out, messages = _run(df, where="all", method={"x": "", "y": "norm.predict"}, maxit=1)
    assert mice.method == {"x": "", "y": "norm.predict", "const": ""}
    assert messages == [
        "Number of logged events: 1. Not imputed because constant or collinear, their where-cells are missing in "
        "the completed data, the observed ones too: ['const']. See MICE.logged_events.",
        "Observed values replaced by missing values in the completed data, because where marks them and they could "
        "not be imputed (as in R); per column: 'x': 20 (no method), 'const': 20 (no method). See MICE.where.",
    ]
    x, y = df["x"].to_numpy(), df["y"].to_numpy()
    missing = np.isnan(y)
    for completed in out:
        assert completed["x"].isna().all() and completed["const"].isna().all()
        np.testing.assert_allclose(completed["y"], _fitted(y, x, ~missing, np.ones(20, dtype=bool)), rtol=1e-10)


def test_setup_message_unchanged_without_observed_where_cells():
    df = _xy_frame()
    df["const"] = 4.0
    df.loc[[2, 5], "const"] = NAN
    _, _, messages = _run(df, maxit=1)
    assert messages == [
        "Number of logged events: 1. Not imputed because constant or collinear, their missing values are kept: "
        "['const']. See MICE.logged_events."
    ]


def test_where_all_with_mean_gives_chain_means_equal_to_the_observed_means():
    """As in R: every where-cell counts, all hold the observed mean, so the variance is 0."""
    df = numeric_frame(n=40, seed=11)
    mice, out, messages = _run(df, where="all", method="mean")
    assert messages == []
    for col in df.columns:
        observed_mean = df[col].dropna().to_numpy(dtype=float).mean()
        np.testing.assert_allclose(mice.chain_mean[col], np.full((2, 2), observed_mean), rtol=1e-12)
        np.testing.assert_allclose(mice.chain_var[col], 0.0, atol=1e-20)
        for completed in out:
            np.testing.assert_allclose(completed[col].to_numpy(dtype=float), observed_mean, rtol=1e-12)


def test_chain_statistics_are_taken_over_all_where_cells():
    """y: missing and observed where-cells, imputed by norm.predict; x by mean. The statistics of the last
    iteration are those of the where-cells of the completed data."""
    df = _xy_frame()
    where = df.isna()
    where.loc[[1, 2, 4, 5], "y"] = True
    where.loc[[0, 1], "x"] = True
    mice, out, _ = _run(df, where=where, method={"x": "mean", "y": "norm.predict"})
    for chain, completed in enumerate(out):
        for col in ("x", "y"):
            values = completed.loc[where[col], col].to_numpy(dtype=float)
            assert np.allclose(mice.chain_mean[col][-1, chain], values.mean(), rtol=1e-12), (
                f"chain_mean of {col!r} is not taken over all where-cells"
            )
            assert np.allclose(mice.chain_var[col][-1, chain], values.var(ddof=1), rtol=1e-10), (
                f"chain_var of {col!r} is not taken over all where-cells"
            )


def test_where_none_imputes_nothing():
    df = _nhanes_like()
    mice, out, messages = _run(df, where="none")
    assert messages == []
    assert set(mice.method.values()) == {""} and mice.visit_sequence == []
    assert mice.chain_mean == {} and mice.chain_var == {} and mice.iteration == 2
    assert not mice.where.to_numpy().any()
    for completed in out:
        pd.testing.assert_frame_equal(completed, df)
        assert completed is not df
    out[0].iloc[0, 1] = -1.0
    assert df.iloc[0, 1] != -1.0 and out[1].iloc[0, 1] != -1.0
    assert all(completed.equals(df) for completed in mice.continue_chains(2))


def test_continue_chains_warns_about_lost_observed_values():
    """The start fills every where-cell; the first iteration makes the where-cells of row 3 missing."""
    df, where, predictors = _left_out_frame()
    mice, out, messages = _run(df, where=where, method=LEFT_OUT_METHODS, predictor_matrix=predictors, maxit=0)
    assert messages == [] and all(completed.loc[3, ["z", "w"]].notna().all() for completed in out)
    with pytest.warns(UserWarning, match=r"per column: 'z': 1 \(predictors incomplete in these rows\), 'w': 1 "):
        mice.continue_chains()


# ---------------------------------------------------------------------------
# dtypes
# ---------------------------------------------------------------------------


def _complete_typed_frame():
    """``y`` incomplete; the complete columns ``k`` (int64), ``u`` (uint8), ``flag`` (bool) and ``x``."""
    rng = np.random.default_rng(4)
    n = 30
    x = rng.normal(size=n)
    df = pd.DataFrame(
        {
            "y": x + rng.normal(scale=0.5, size=n),
            "k": rng.integers(0, 10, size=n),
            "u": rng.integers(0, 5, size=n).astype(np.uint8),
            "flag": x + rng.normal(size=n) > 0,
            "x": x,
        }
    )
    df.loc[[0, 4, 8], "y"] = NAN
    return df


@pytest.mark.parametrize(
    "method, expected",
    [
        ({}, {"k": "Int64", "u": "UInt8", "flag": "boolean"}),
        ({"k": "norm", "u": "mean"}, {"k": "Float64", "u": "Float64", "flag": "boolean"}),
        ({"k": "", "u": "", "flag": ""}, {"k": "Int64", "u": "UInt8", "flag": "boolean"}),
    ],
    ids=["default", "float-methods", "no-method"],
)
def test_complete_integer_and_boolean_columns_with_where_cells_become_nullable(method, expected):
    df = _complete_typed_frame()
    where = df.isna()
    where.loc[[1, 2, 3], ["k", "u", "flag"]] = True
    with warnings.catch_warnings():
        # pandas warns before it changes the dtype of a column a value is written into
        warnings.simplefilter("error", FutureWarning)
        warnings.filterwarnings("ignore", "Observed values replaced", UserWarning)
        out = MICE(df).impute(n_imputations=2, maxit=2, seed=1, where=where, method=method)
    expected = {**df.dtypes.astype(str).to_dict(), **expected}
    for completed in out:
        assert completed.dtypes.astype(str).to_dict() == expected
        if method.get("k") == "":
            assert completed.loc[[1, 2, 3], ["k", "u", "flag"]].isna().all().all()
        else:
            assert completed.notna().all().all()
        pd.testing.assert_frame_equal(
            completed.loc[4:, ["k", "u", "flag"]], df.loc[4:, ["k", "u", "flag"]], check_dtype=False
        )


def test_dtypes_unchanged_without_where_cells_on_them():
    df = _complete_typed_frame()
    _, out, _ = _run(df, maxit=1)
    assert all(completed.dtypes.to_dict() == df.dtypes.to_dict() for completed in out)


# ---------------------------------------------------------------------------
# mutants
# ---------------------------------------------------------------------------
#
# Each applies one defect by patching one function for one test, checks that
# it was applied, and runs a named test of this file, which must fail with
# its own message:
#
# * where not used, only the missing cells imputed (check_where);
# * the completed data built from the working data (MICE._impute_once);
# * the imputations of observed where-cells written to the working data
#   (_refresh_working_column);
# * the chain statistics taken over the missing where-cells only
#   (MICE._record_chain_statistics);
# * method "" by completeness instead of by where-cells (make_method);
# * "monotone" by the number of missing values (_monotone);
# * the monotone edit of the predictor matrix left out
#   (_monotone_predictor_matrix).


def _counting(calls, function):
    def counted(*args, **kwargs):
        calls.append(1)
        return function(*args, **kwargs)

    return counted


def test_where_not_used_is_caught(monkeypatch):
    calls = []
    monkeypatch.setattr(MICE_MODULE, "check_where", _counting(calls, lambda where, data: data.isna()))
    with pytest.raises(AssertionError, match="observed where-cells of x not replaced"):
        test_observed_where_cell_is_replaced_and_predicts_with_its_observed_value()
    assert calls, "mutant not applied"


def test_completed_data_from_the_working_data_is_caught(monkeypatch):
    calls = []

    def from_working_data(self, state, chain_idx, iterations):
        calls.append(1)
        working = self._working_frame(state.imp)
        for iter_idx in iterations:
            self._iterate(working, state, iter_idx, chain_idx)
        return working

    monkeypatch.setattr(MICE, "_impute_once", from_working_data)
    with pytest.raises(AssertionError, match="observed where-cells of x not replaced"):
        test_observed_where_cell_is_replaced_and_predicts_with_its_observed_value()
    assert calls, "mutant not applied"


def test_imputations_of_observed_where_cells_in_the_working_data_are_caught(monkeypatch):
    calls = []

    def every_where_cell(working, column, imp, where, observed):
        calls.append((where & observed).any())
        working.loc[where, column] = MICE_MODULE._values(imp)

    monkeypatch.setattr(MICE_MODULE, "_refresh_working_column", every_where_cell)
    with pytest.raises(AssertionError, match="y not predicted from the observed x"):
        test_observed_where_cell_is_replaced_and_predicts_with_its_observed_value()
    assert any(calls), "mutant not applied: no observed where-cell refreshed"


def test_chain_statistics_over_the_missing_cells_only_are_caught(monkeypatch):
    calls = []
    original = MICE._record_chain_statistics

    def missing_cells_only(self, imp, iter_idx, chain_idx):
        calls.append(1)
        missing = {col: values[self.id_mis[col].loc[values.index].to_numpy()] for col, values in imp.items()}
        original(self, missing, iter_idx, chain_idx)

    monkeypatch.setattr(MICE, "_record_chain_statistics", missing_cells_only)
    with pytest.raises(AssertionError, match="chain_mean of 'x' is not taken over all where-cells"):
        test_chain_statistics_are_taken_over_all_where_cells()
    assert calls, "mutant not applied"


def test_method_by_completeness_is_caught(monkeypatch):
    calls = []
    original = MICE_MODULE.make_method
    monkeypatch.setattr(
        MICE_MODULE,
        "make_method",
        _counting(calls, lambda data, default_method, where=None: original(data, default_method)),
    )
    with pytest.raises(AssertionError, match="a complete column with where-cells has no method"):
        test_complete_column_with_where_cells_gets_the_default_method_and_is_imputed()
    assert calls, "mutant not applied"


def test_monotone_by_the_number_of_missing_values_is_caught(monkeypatch):
    calls = []
    original = MICE_MODULE._monotone
    missing = _visit_frame()[0].isna()
    monkeypatch.setattr(MICE_MODULE, "_monotone", _counting(calls, lambda where, columns: original(missing, columns)))
    with pytest.raises(AssertionError, match="monotone does not order by the number of where-cells"):
        test_monotone_orders_by_the_number_of_where_cells()
    assert calls, "mutant not applied"


def test_monotone_edit_left_out_is_caught(monkeypatch):
    calls = []
    monkeypatch.setattr(
        MICE_MODULE, "_monotone_predictor_matrix", _counting(calls, lambda predictor_matrix, where: predictor_matrix)
    )
    with pytest.raises(AssertionError, match="the predictor matrix is not edited"):
        test_monotone_with_one_iteration_edits_the_predictor_matrix()
    assert calls, "mutant not applied"
