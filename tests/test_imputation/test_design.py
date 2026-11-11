"""Unit tests for imputation.design: edge cases the R fixtures do not cover.

The parity cases are in tests/test_parity/test_l1_setup.py. Unless a test
says otherwise, the expected values below were checked by hand against
mice 3.19.0 (mice:::obtain.design, mice:::remove.lindep,
mice:::find.collinear) on the same inputs; the deterministic data use sin
and cos so that R can rebuild them exactly.
"""

import numpy as np
import pandas as pd
import pytest

from imputation.design import (
    ALL_PREDICTORS_DROPPED,
    _remove_lindep,
    find_collinear,
    obtain_design,
    remove_lindep,
)

N = 12
IDX = np.arange(1, N + 1)
X1 = np.sin(IDX)
X2 = np.cos(0.7 * IDX)
Y = X1 + 0.5 * X2 + 0.3 * np.sin(3 * IDX)
RY = np.ones(N, dtype=bool)
RY[[1, 6]] = False
K = np.tile([1, 2, 3], 4)


# ---------------------------------------------------------------------------
# obtain_design
# ---------------------------------------------------------------------------


def test_obtain_design_empty_predictor_list():
    data = pd.DataFrame({"y": [1.0, 2.0, 3.0], "x": [4.0, 5.0, 6.0]}, index=[7, 3, 5])
    x = obtain_design(data, [])
    assert x.shape == (3, 0)
    assert x.index.equals(data.index)


@pytest.mark.parametrize(
    "column",
    [
        pd.Categorical(["a", "a", "a", "a"]),
        pd.Categorical(["a", None, "a", "a"], ordered=True),
        pd.Series(["s", "s", None, "s"], dtype=object),
        pd.Series(["s", "s", "s", "s"], dtype="string"),
    ],
    ids=["category", "ordered", "object", "string"],
)
def test_obtain_design_one_level_column_gives_no_column(column):
    # R stops with "contrasts can be applied only to factors with 2 or more
    # levels"; the port documents that it adds no column instead.
    data = pd.DataFrame({"num": [1, 2, 3, 4], "g": column})
    x = obtain_design(data, ["num", "g"])
    assert list(x.columns) == ["num"]
    np.testing.assert_array_equal(x["num"].to_numpy(), [1.0, 2.0, 3.0, 4.0])


def test_obtain_design_unused_categories_give_zero_columns():
    # model.frame keeps unused levels, so model.matrix has a column of zeros
    data = pd.DataFrame({"f": pd.Categorical(["a", "c", "a"], categories=["a", "b", "c"])})
    x = obtain_design(data, ["f"])
    assert list(x.columns) == ["fb", "fc"]
    np.testing.assert_array_equal(x.to_numpy(), [[0.0, 0.0], [0.0, 1.0], [0.0, 0.0]])


@pytest.mark.parametrize(
    "column",
    [
        pd.Series([True, None, False, True], dtype=object),
        pd.Series([True, pd.NA, False, True], dtype="boolean"),
    ],
    ids=["object", "boolean"],
)
def test_obtain_design_boolean_with_missing_values(column):
    x = obtain_design(pd.DataFrame({"b": column}), ["b"])
    assert list(x.columns) == ["bTRUE"]
    np.testing.assert_array_equal(x["bTRUE"].to_numpy(), [1.0, np.nan, 0.0, 1.0])


def test_obtain_design_boolean_with_one_value_still_has_its_column():
    # R codes a logical with the levels FALSE, TRUE even if one never occurs
    x = obtain_design(pd.DataFrame({"b": [True, True, True]}), ["b"])
    assert list(x.columns) == ["bTRUE"]
    np.testing.assert_array_equal(x["bTRUE"].to_numpy(), [1.0, 1.0, 1.0])


def test_obtain_design_missing_category_is_missing_in_every_column():
    data = pd.DataFrame({"f": pd.Categorical(["a", None, "b", "c"])})
    x = obtain_design(data, ["f"])
    np.testing.assert_array_equal(x.to_numpy(), [[0.0, 0.0], [np.nan, np.nan], [1.0, 0.0], [0.0, 1.0]])


def test_obtain_design_ordered_uses_polynomial_contrasts():
    data = pd.DataFrame({"g": pd.Categorical(["a", None, "c", "b"], categories=["a", "b", "c"], ordered=True)})
    x = obtain_design(data, ["g"])
    assert list(x.columns) == ["g.L", "g.Q"]
    linear = np.array([-1.0, 0.0, 1.0]) / np.sqrt(2.0)
    quadratic = np.array([1.0, -2.0, 1.0]) / np.sqrt(6.0)
    expected = np.array(
        [[linear[0], quadratic[0]], [np.nan, np.nan], [linear[2], quadratic[2]], [linear[1], quadratic[1]]]
    )
    np.testing.assert_allclose(x.to_numpy(), expected, rtol=0, atol=1e-15)


def test_obtain_design_ordered_names_and_basis_for_six_levels():
    levels = list("abcdef")
    data = pd.DataFrame({"g": pd.Categorical(levels, categories=levels, ordered=True)})
    design = obtain_design(data, ["g"])
    assert list(design.columns) == ["g.L", "g.Q", "g.C", "g^4", "g^5"]
    x = design.to_numpy()
    np.testing.assert_allclose(x.T @ x, np.eye(5), atol=1e-14)  # orthonormal
    np.testing.assert_allclose(x.sum(axis=0), 0.0, atol=1e-14)  # contrasts
    np.testing.assert_allclose(x[:, 0], np.arange(-5.0, 6.0, 2.0) / np.sqrt(70.0), atol=1e-15)
    # contr.poly(6)[, 5] in R, rounded as R prints it
    np.testing.assert_allclose(
        x[:, 4],
        [-0.06299408, 0.31497039, -0.62994079, 0.62994079, -0.31497039, 0.06299408],
        atol=5e-9,
    )


def test_obtain_design_object_levels_sorted_by_code_point():
    # R's default factor order in the C locale: "B" < "a" < "b" < "c"
    data = pd.DataFrame({"s": pd.Series(["b", "a", "B", "c"], dtype=object)})
    x = obtain_design(data, ["s"])
    assert list(x.columns) == ["sa", "sb", "sc"]
    np.testing.assert_array_equal(x.to_numpy(), [[0, 1, 0], [1, 0, 0], [0, 0, 0], [0, 0, 1]])


def test_obtain_design_string_dtype_behaves_like_object():
    # port behaviour: R has no separate string type
    values = ["y", None, "x", "z"]
    as_object = obtain_design(pd.DataFrame({"s": pd.Series(values, dtype=object)}), ["s"])
    as_string = obtain_design(pd.DataFrame({"s": pd.Series(values, dtype="string")}), ["s"])
    pd.testing.assert_frame_equal(as_object, as_string)
    assert list(as_object.columns) == ["sy", "sz"]


def test_obtain_design_numeric_columns_become_float():
    data = pd.DataFrame({"i": pd.Series([1, pd.NA, 3], dtype="Int64"), "n": np.array([4, 5, 6], dtype=np.int32)})
    x = obtain_design(data, ["i", "n"])
    assert all(pd.api.types.is_float_dtype(t) for t in x.dtypes)
    np.testing.assert_array_equal(x.to_numpy(), [[1.0, 4.0], [np.nan, 5.0], [3.0, 6.0]])


def test_obtain_design_follows_predictor_order_and_uses_each_once():
    data = pd.DataFrame({"a": [1.0, 2.0], "g": ["p", "q"], "b": [3.0, 4.0]})
    x = obtain_design(data, ["b", "g", "a", "b"])
    assert list(x.columns) == ["b", "gq", "a"]


def test_obtain_design_leaves_input_unchanged():
    # port behaviour
    data = pd.DataFrame({"g": pd.Categorical(["a", None, "b"]), "b": [True, None, False]})
    before = data.copy(deep=True)
    obtain_design(data, ["g", "b"])
    pd.testing.assert_frame_equal(data, before)


def test_obtain_design_rejects_unsupported_dtype():
    # port behaviour
    data = pd.DataFrame({"t": pd.date_range("2020-01-01", periods=3)})
    with pytest.raises(TypeError, match="unsupported dtype"):
        obtain_design(data, ["t"])


# ---------------------------------------------------------------------------
# remove_lindep
# ---------------------------------------------------------------------------


def _frame(**columns):
    return pd.DataFrame(columns)


def test_remove_lindep_single_column():
    keep, event = _remove_lindep(_frame(x1=X1), Y, RY)
    assert keep.dtype == bool and keep.tolist() == [True]
    assert event is None


def test_remove_lindep_single_column_too_correlated_with_y():
    keep, event = _remove_lindep(_frame(x1=X1), X1 + 1e-6 * X2, RY)
    assert keep.tolist() == [False]
    assert event == ALL_PREDICTORS_DROPPED


def test_remove_lindep_zero_columns():
    keep = remove_lindep(np.empty((N, 0)), Y, RY)
    assert keep.dtype == bool and keep.shape == (0,)


def test_remove_lindep_constant_column():
    keep, event = _remove_lindep(_frame(c=np.full(N, 3.0), x1=X1, x2=X2), Y, RY)
    assert keep.tolist() == [False, True, True]
    assert event == "c"


def test_remove_lindep_constant_column_with_one_other_is_not_logged():
    # R returns before logging when at most one column survives the screen
    keep, event = _remove_lindep(_frame(c=np.full(N, 3.0), x1=X1), Y, RY)
    assert keep.tolist() == [False, True]
    assert event is None


def test_remove_lindep_only_a_constant_column():
    keep, event = _remove_lindep(_frame(c=np.full(N, 3.0)), Y, RY)
    assert keep.tolist() == [False]
    assert event == ALL_PREDICTORS_DROPPED


def test_remove_lindep_ry_all_false_keeps_every_column():
    keep, event = _remove_lindep(_frame(x1=X1, x2=X2), Y, np.zeros(N, dtype=bool))
    assert keep.tolist() == [True, True]
    assert event is None


def test_remove_lindep_single_observed_value_keeps_nothing():
    # Not R: R stops with "missing value where TRUE/FALSE needed".
    ry = np.zeros(N, dtype=bool)
    ry[4] = True
    keep, event = _remove_lindep(_frame(x1=X1, x2=X2), Y, ry)
    assert keep.tolist() == [False, False]
    assert event is None


def test_remove_lindep_near_copy_is_dropped_and_named():
    keep, event = _remove_lindep(_frame(x1=X1, x1b=X1 + 1e-4 * X2, x2=X2), Y, RY)
    assert keep.tolist() == [True, False, True]
    assert event == "x1b"


def test_remove_lindep_array_and_frame_agree():
    frame = _frame(c=np.full(N, 3.0), x1=X1, x1b=X1 + 1e-4 * X2, x2=X2)
    np.testing.assert_array_equal(remove_lindep(frame, Y, RY), remove_lindep(frame.to_numpy(), Y, RY))
    assert _remove_lindep(frame.to_numpy(), Y, RY)[1] == ""  # R logs "" without names


@pytest.mark.parametrize(
    "categories, expected",
    [(["lo", "mid", "hi"], [False, True]), (["mid", "lo", "hi"], [True, True])],
    ids=["codes_follow_k", "codes_scrambled"],
)
def test_remove_lindep_categorical_y_counts_by_level_codes(categories, expected):
    labels = np.array(["lo", "mid", "hi"])[K - 1]
    y = pd.Categorical(labels, categories=categories)
    keep = remove_lindep(_frame(k=K.astype(float), x1=X1), y, RY)
    assert keep.tolist() == expected


def test_remove_lindep_eps_zero_keeps_every_column():
    keep = remove_lindep(_frame(a=X1, b=X1), Y, RY, eps=0)
    assert keep.tolist() == [True, True]


def test_remove_lindep_negative_eps_raises():
    with pytest.raises(ValueError, match="eps"):
        remove_lindep(_frame(x1=X1, x2=X2), Y, RY, eps=-1)


def test_remove_lindep_eps_too_large_raises():
    # R stops here too: the eigenvalue loop runs past the first column
    x = _frame(x1=10 * X1, x2=10 * X2, x3=10 * X1 + 10 * X2)
    with pytest.raises(ValueError, match="too large"):
        remove_lindep(x, 10 * Y, RY, eps=2)


# ---------------------------------------------------------------------------
# find_collinear
# ---------------------------------------------------------------------------


def test_find_collinear_orders_by_observed_count():
    # a, c and d are complete (column order kept), b has two missing values
    data = _frame(a=X1, b=np.r_[2 * X1[:10], np.nan, np.nan], c=X2, d=-X1)
    assert find_collinear(data) == ["d", "b"]


@pytest.mark.parametrize(
    "categories, expected",
    [(["u", "v", "w"], ["g"]), (["w", "v", "u"], ["g"]), (["v", "u", "w"], [])],
    ids=["same_order", "reversed", "scrambled"],
)
def test_find_collinear_uses_level_codes(categories, expected):
    g = pd.Categorical(np.array(["u", "v", "w"])[K - 1], categories=categories)
    assert find_collinear(_frame(z=K.astype(float), g=g)) == expected


def test_find_collinear_boolean_counts_as_zero_one():
    data = _frame(z=(X1 > 0).astype(float), b=X1 > 0, w=X2)
    assert find_collinear(data) == ["b"]


def test_find_collinear_object_column_uses_sorted_levels():
    s = np.array(["b", "a", "c"], dtype=object)[K - 1]  # codes 2, 1, 3: not linear in K
    assert find_collinear(_frame(z=K.astype(float), s=s)) == []


def test_find_collinear_never_reports_constant_columns():
    data = _frame(z=X1, c=np.ones(N), c2=np.ones(N))
    assert find_collinear(data) == []


def test_find_collinear_threshold():
    data = _frame(a=X1, b=X1 + 0.1 * X2)  # correlation about 0.9976
    assert find_collinear(data, threshold=0.99) == ["b"]
    assert find_collinear(data) == []


def test_find_collinear_empty_frame():
    # Not R: R stops with "'x' is empty" (mice never calls it without predictors)
    assert find_collinear(pd.DataFrame(index=range(3))) == []


# ---------------------------------------------------------------------------
# quickpred
# ---------------------------------------------------------------------------


def _quickpred_frame():
    """Numeric, categorical (level order mid, lo, hi), object and boolean columns,
    all incomplete. R: the same with ``factor(..., levels = c("mid", "lo", "hi"))``,
    a character column ``s`` and a logical column ``b``."""
    i = np.arange(1, 31)
    a = np.sin(i)
    cg = np.cos(1.7 * i)
    g = pd.Categorical(np.where(cg > 0.3, "hi", np.where(cg < -0.3, "lo", "mid")), categories=["mid", "lo", "hi"])
    data = pd.DataFrame(
        {
            "y": a + 0.4 * np.cos(2.3 * i),
            "a": a,
            "g": g,
            "s": pd.Series(np.array(["r", "p", "q"], dtype=object)[i % 3], dtype=object),
            "b": pd.Series(np.sin(0.9 * i) > 0, dtype=object),
        }
    )
    data.loc[np.array([2, 5, 9, 14, 22, 27]) - 1, "y"] = np.nan
    data.loc[np.array([3, 9, 17, 25]) - 1, "g"] = np.nan
    data.loc[a > 0.85, "a"] = np.nan
    data.loc[np.array([4, 11]) - 1, "s"] = np.nan
    data.loc[np.array([6, 21, 30]) - 1, "b"] = np.nan
    return data


# R: quickpred(d, ...) rows y, a, g, s, b, each row written as a string of 0/1
@pytest.mark.parametrize(
    "arguments, expected",
    [
        ({}, "01110 10111 10011 11101 01110"),
        ({"mincor": 0.25}, "01010 10000 00000 11101 00010"),
        ({"mincor": 0.25, "minpuc": 0.7}, "00010 00000 00000 11101 00010"),
        ({"mincor": 0.25, "method": "spearman"}, "01000 10000 00000 11101 00010"),
        ({"mincor": 0.25, "method": "kendall"}, "01000 10000 00000 11101 00010"),
        ({"mincor": [0.5, 0.05, 0.3, 0.2, 0.9]}, "01000 10111 00000 11101 00000"),
        ({"mincor": 0.25, "include": "s", "exclude": ["a", "s"]}, "00010 10010 00010 10101 00010"),
    ],
    ids=["default", "mincor", "minpuc", "spearman", "kendall", "mincor_per_target", "include_exclude"],
)
def test_quickpred_with_categorical_object_and_boolean_columns_matches_r(arguments, expected):
    from imputation import quickpred

    data = _quickpred_frame()
    got = quickpred(data, **arguments)
    assert list(got.index) == list(data.columns) and list(got.columns) == list(data.columns)
    assert all(dtype.kind == "i" for dtype in got.dtypes)
    assert " ".join("".join(map(str, row)) for row in got.to_numpy()) == expected


def test_quickpred_uses_the_response_indicator():
    """Row s with mincor 0.25: every predictor enters through the correlation of
    s's response indicator with it (0.30 to 0.34 in R), none through the data
    correlation (at most 0.27, with y)."""
    from imputation import quickpred

    data = _quickpred_frame()
    got = quickpred(data, mincor=0.25)
    assert got.loc["s"].tolist() == [1, 1, 1, 0, 1]


def test_quickpred_threshold_is_strict():
    """A constant predictor has no correlation (R: NA, counted as 0); 0 > 0 is
    false, so it is not selected even with mincor 0."""
    from imputation import quickpred

    data = _frame(y=np.where(RY, Y, np.nan), x=X1, c=np.ones(N))
    got = quickpred(data, mincor=0.0)
    assert got.loc["y"].tolist() == [0, 1, 0]
    assert got.loc[["x", "c"]].eq(0).all().all()


def test_quickpred_rejects_unknown_columns_and_methods():
    from imputation import quickpred

    data = _quickpred_frame()
    with pytest.raises(ValueError, match="include"):
        quickpred(data, include=["nope"])
    with pytest.raises(ValueError, match="exclude"):
        quickpred(data, exclude="nope")
    with pytest.raises(ValueError, match="method"):
        quickpred(data, method="cosine")


def test_quickpred_is_exported():
    import imputation
    from imputation.design import quickpred

    assert imputation.quickpred is quickpred
    assert "quickpred" in imputation.__all__
