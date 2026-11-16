"""``impute(passive=...)`` and the visit sequence: columns computed from other columns, as R's passive imputation.

A passive column is computed, not imputed: at each visit its function gets a
copy of the data the chains work on (all rows) and returns one value per row;
the where-cells of the column take the values of their rows, and the working
data are refreshed in its missing where-cells only. ``MICE.method`` shows
``"passive"``. As in R, a passive column starts from sampled observed values,
is visited after the imputed columns when no visit sequence is given (a
keyword or a list is used as given, and a list may name a column twice), is
not checked for being constant but is checked for being collinear, and
counts for the chain statistics. Each function is called once on the data
before the chains start, and the dtype of the column is fixed from what it
returns.

The expectations below are computed by hand or by a few lines of NumPy with
the deterministic methods ``mean`` and ``norm.predict``. The rules tested
are those of R's ``sampler``, ``initialize.imp``, ``mice.edit.setup`` and
``check.visitSequence`` (mice 3.19.0), with the differences that the plan of
0.4 decides: the function gets all rows (R evaluates the formula on the rows
being imputed), and a function that fails, a ``post`` for a passive column
and a list that leaves a passive column out raise.

The mutants at the end introduce the defects that passive columns and the
visit sequence can have, each by patching one function, and check that a
named test of this file fails.
"""

import importlib
import sys
import warnings

import numpy as np
import pandas as pd
import pytest

from imputation import MICE
from imputation.validators import check_passive

from ._helpers import edit_iterate, full_predictor_matrix, nan_cascade, numeric_frame, predictors_of, typed_frame
from .test_continue_chains import _differences, _record
from .test_where_invariants import _random_predictor_matrix, _random_where

MICE_MODULE = importlib.import_module("imputation.MICE")
NAN = np.nan


def _run(df, **kwargs):
    """``MICE(df).impute(**kwargs)`` with two chains and two iterations by default; returns (mice, datasets,
    the messages of the UserWarnings). A FutureWarning fails the test."""
    kwargs = {"n_imputations": 2, "maxit": 2, "seed": 1, **kwargs}
    mice = MICE(df)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        out = mice.impute(**kwargs)
    # pandas warns where it would change the dtype of a column a value is written into
    assert [str(w.message) for w in caught if issubclass(w.category, FutureWarning)] == []
    return mice, out, [str(w.message) for w in caught if issubclass(w.category, UserWarning)]


def _continue(mice):
    """``mice.continue_chains()``; a FutureWarning fails the test, as in `_run`."""
    with warnings.catch_warnings():
        warnings.simplefilter("error", FutureWarning)
        return mice.continue_chains()


def _fitted(y, x, fit_rows, rows):
    """The least-squares prediction of `y` from `x` with an intercept, fitted on `fit_rows`, at `rows`
    (what ``norm.predict`` imputes)."""
    design = np.column_stack([np.ones(len(y)), x])
    coef, *_ = np.linalg.lstsq(design[fit_rows], y[fit_rows], rcond=None)
    return design[rows] @ coef


def _bmi(d):
    return d["wgt"] / (d["hgt"] / 100) ** 2


BMI = {"bmi": _bmi}


def _body_frame():
    """``age`` complete; ``bmi``, ``hgt``, ``wgt``, in this order (bmi before its inputs). bmi is missing where
    hgt or wgt is, and in rows 0, 30 and 31. Returns (data, a predictor matrix in which bmi predicts neither of
    its inputs)."""
    rng = np.random.default_rng(12)
    n = 40
    age = rng.uniform(1, 18, n)
    hgt = 75 + 6 * age + rng.normal(scale=6, size=n)
    wgt = 8 + 3 * age + rng.normal(scale=4, size=n)
    df = pd.DataFrame({"age": age, "bmi": wgt / (hgt / 100) ** 2, "hgt": hgt, "wgt": wgt})
    df.loc[[1, 5, 9, 13, 17], "hgt"] = NAN
    df.loc[[2, 5, 10, 14, 22], "wgt"] = NAN
    df.loc[df["hgt"].isna() | df["wgt"].isna(), "bmi"] = NAN
    df.loc[[0, 30, 31], "bmi"] = NAN
    predictors = full_predictor_matrix(df)
    predictors.loc[["hgt", "wgt"], "bmi"] = 0
    return df, predictors


BMI_MISSING = [0, 1, 2, 5, 9, 10, 13, 14, 17, 22, 30, 31]
INPUT_MISSING = [1, 2, 5, 9, 10, 13, 14, 17, 22]


def _assert_bmi_of_its_row(completed, rows=BMI_MISSING):
    np.testing.assert_allclose(
        completed.loc[rows, "bmi"], _bmi(completed.loc[rows]), rtol=1e-12, err_msg="bmi is not the function of its row"
    )


# ---------------------------------------------------------------------------
# check_passive
# ---------------------------------------------------------------------------


def _small():
    return pd.DataFrame({"a": [1.0, NAN, 3.0], "b": [2.0, 4.0, NAN], "s": [NAN, NAN, 5.0]}, index=[7, 3, 5])


@pytest.mark.parametrize(
    "function",
    [
        lambda d: d["a"] + d["b"],
        lambda d: (d["a"] + d["b"]).to_list(),
        lambda d: tuple(d["a"] + d["b"]),
        lambda d: (d["a"] + d["b"]).to_numpy(),
        lambda d: pd.array(d["a"] + d["b"], dtype="Float64"),
        lambda d: pd.Index(d["a"] + d["b"]),
    ],
    ids=["series", "list", "tuple", "array", "pandas-array", "index"],
)
def test_check_passive_takes_one_value_per_row(function):
    df = _small()
    got = check_passive({"s": function}, df)
    assert list(got) == ["s"]
    assert got["s"].index.equals(df.index)
    np.testing.assert_allclose(got["s"].to_numpy(dtype=float, na_value=NAN), [3.0, NAN, NAN])


def test_check_passive_calls_each_function_once_on_a_copy_of_the_data():
    df = _small()
    before = df.copy()
    calls = []

    def overwriting(d):
        calls.append(d.index.tolist())
        d.iloc[:, :] = -1.0
        return d["a"]

    assert check_passive(None, df) == {} and check_passive({}, df) == {}
    check_passive({"s": overwriting, "b": overwriting}, df)
    assert calls == [[7, 3, 5], [7, 3, 5]]
    pd.testing.assert_frame_equal(df, before)


_PASSIVE_FORM = (
    r"a mapping from column labels to functions f\(data\), for example \{'bmi': lambda d: d\['wgt'\] / "
    r"\(d\['hgt'\] / 100\) \*\* 2\}"
)
_ONE_PER_ROW = r"it must return one value per row: a Series with the index of the data, or a sequence of its length"
_CHECK_CALL = r"Column 's', passive \(called once on the data before the chains start\)"


@pytest.mark.parametrize(
    "passive, message",
    [
        ([_bmi], rf"^passive must be None or {_PASSIVE_FORM}; got list\.$"),
        ("~ I(a + b)", r"^passive must be None or a mapping .*; got str\.$"),
        ({"s": _bmi, "nope": _bmi}, r"^passive names columns that are not in the data: \['nope'\]\.$"),
        (
            {"s": "a + b"},
            r"^passive must map each column to a function f\(data\); the entry of the column 's' is a str, which "
            r"cannot be called\.$",
        ),
        ({"s": lambda d: d["nope"]}, rf"^{_CHECK_CALL}: the function raised KeyError: 'nope'$"),
        ({"s": lambda d: 1.0}, rf"^{_CHECK_CALL}: the function returned a float; {_ONE_PER_ROW}\.$"),
        ({"s": lambda d: "a"}, rf"^{_CHECK_CALL}: the function returned a str; "),
        ({"s": lambda d: d[["a", "b"]]}, rf"^{_CHECK_CALL}: the function returned a DataFrame; {_ONE_PER_ROW}\.$"),
        (
            {"s": lambda d: d[["a", "b"]].to_numpy()},
            rf"^{_CHECK_CALL}: the function returned an array of shape \(3, 2\); {_ONE_PER_ROW}\.$",
        ),
        (
            {"s": lambda d: d["a"].iloc[:2]},
            rf"^{_CHECK_CALL}: the function returned 2 values for the 3 rows of the data; {_ONE_PER_ROW}\.$",
        ),
        ({"s": lambda d: [1.0, 2.0, 3.0, 4.0]}, r"the function returned 4 values for the 3 rows of the data"),
        (
            {"s": lambda d: d["a"].sort_index()},
            rf"^{_CHECK_CALL}: the function returned a Series with another index; {_ONE_PER_ROW} \(the same labels in "
            r"the same order\)\.$",
        ),
        ({"s": lambda d: d["a"].reset_index(drop=True)}, r"the function returned a Series with another index"),
    ],
    ids=[
        "list",
        "formula",
        "unknown-column",
        "text",
        "raises",
        "scalar",
        "string-result",
        "two-columns",
        "2d",
        "short",
        "long",
        "reordered",
        "other-index",
    ],
)
def test_check_passive_errors(passive, message):
    with pytest.raises(ValueError, match=message):
        check_passive(passive, _small())


def test_check_passive_keeps_the_exception_of_the_function():
    with pytest.raises(ValueError, match="the function raised ZeroDivisionError") as error:
        check_passive({"s": lambda d: 1 / 0}, _small())
    assert isinstance(error.value.__cause__, ZeroDivisionError)


# ---------------------------------------------------------------------------
# the reserved name and the other arguments
# ---------------------------------------------------------------------------

_NOT_A_METHOD = (
    r"^'passive' is not an imputation method: a passive column is given with the argument passive, as a function "
    r"of the data, for example passive=\{'bmi': lambda d: d\['wgt'\] / \(d\['hgt'\] / 100\) \*\* 2\}; MICE\.method "
    r"then shows 'passive' for it\.$"
)


@pytest.mark.parametrize("method", ["passive", {"bmi": "passive"}, {"hgt": "passive"}], ids=["str", "dict", "other"])
def test_passive_is_not_a_method_name(method):
    df, _ = _body_frame()
    with pytest.raises(ValueError, match=_NOT_A_METHOD):
        MICE(df).impute(method=method)
    with pytest.raises(ValueError, match=_NOT_A_METHOD):
        MICE(df).impute(method=method, passive=BMI)


@pytest.mark.parametrize("method", ["pmm", ""])
def test_a_method_dict_must_not_name_a_passive_column(method):
    df, _ = _body_frame()
    with pytest.raises(
        ValueError,
        match=rf"^method names the passive columns \['bmi'\] \(with the methods \['{method}'\]\): a column given in "
        r"passive is computed by its function and has no other method\. Remove it from method, or from passive to "
        r"impute it\.$",
    ):
        MICE(df).impute(method={"hgt": "norm", "bmi": method}, passive=BMI)


def test_a_method_str_applies_to_the_other_columns():
    df, predictors = _body_frame()
    mice, _, _ = _run(df, method="norm", passive=BMI, predictor_matrix=predictors, maxit=0)
    assert mice.method == {"age": "", "bmi": "passive", "hgt": "norm", "wgt": "norm"}


def test_post_for_a_passive_column_raises():
    df, _ = _body_frame()
    with pytest.raises(
        ValueError,
        match=r"^post names the passive columns \['bmi'\]: post is not applied to a passive column \(R ignores it "
        r"silently\)\. Remove them from post, or put the change into their function in passive\.$",
    ):
        MICE(df).impute(passive=BMI, post={"hgt": lambda imp, data: imp, "bmi": lambda imp, data: imp})


def test_a_passive_column_needs_no_row_in_the_predictor_matrix():
    df, predictors = _body_frame()
    mice, out, _ = _run(df, passive=BMI, predictor_matrix=predictors.drop(index="bmi"))
    assert mice.method["bmi"] == "passive" and "bmi" not in mice._predictors
    for completed in out:
        _assert_bmi_of_its_row(completed)


# ---------------------------------------------------------------------------
# the relation, the order, the start
# ---------------------------------------------------------------------------


def test_every_completed_dataset_holds_the_function_of_its_rows():
    df, predictors = _body_frame()
    mice, out, messages = _run(df, passive=BMI, predictor_matrix=predictors, n_imputations=3)
    for completed in out:
        _assert_bmi_of_its_row(completed)
        observed = df["bmi"].notna()
        pd.testing.assert_series_equal(completed.loc[observed, "bmi"], df.loc[observed, "bmi"])
        assert completed.notna().all().all()
    assert messages == []
    assert mice.method == {"age": "", "bmi": "passive", "hgt": "pmm", "wgt": "pmm"}
    assert mice.visit_sequence == ["hgt", "wgt", "bmi"]
    assert mice.passive == BMI and mice.post == {}
    # the chain statistics of bmi, over its where-cells at the end of each iteration
    for chain, completed in enumerate(out):
        assert np.isclose(mice.chain_mean["bmi"][-1, chain], completed.loc[BMI_MISSING, "bmi"].mean(), rtol=1e-12)


def test_roman_with_the_passive_column_before_its_inputs_breaks_the_relation():
    """As in R: under "roman" bmi is computed before hgt and wgt are imputed again, so after one iteration it is
    the function of the previous values of its inputs where one of them is imputed."""
    df, predictors = _body_frame()
    mice, out, _ = _run(df, passive=BMI, predictor_matrix=predictors, visit_sequence="roman", maxit=1)
    assert mice.visit_sequence == ["bmi", "hgt", "wgt"], "the passive column is moved under a keyword"
    for completed in out:
        assert not np.allclose(completed.loc[INPUT_MISSING, "bmi"], _bmi(completed.loc[INPUT_MISSING]))
        # where both inputs are observed the relation holds
        _assert_bmi_of_its_row(completed, rows=[0, 30, 31])


def _counting(calls, function):
    def counted(data):
        calls.append(len(data))
        return function(data)

    return counted


def test_a_list_may_name_the_passive_column_twice():
    df, predictors = _body_frame()
    calls = []
    order = ["hgt", "bmi", "wgt", "bmi"]
    mice, out, _ = _run(
        df, passive={"bmi": _counting(calls, _bmi)}, predictor_matrix=predictors, visit_sequence=order, maxit=2
    )
    assert mice.visit_sequence == order
    assert list(mice.chain_mean) == ["hgt", "bmi", "wgt"]
    # the check call, then two visits in each of two iterations of two chains, each with all 40 rows
    assert calls == [40] * 9
    for completed in out:
        _assert_bmi_of_its_row(completed)


def test_start_values_are_sampled_observed_values():
    """maxit=0: the start of bmi is drawn from its observed values (not computed by its function), also with
    ignored rows, from the rows that are not ignored."""
    df, predictors = _body_frame()
    mice, out, _ = _run(df, passive=BMI, predictor_matrix=predictors, maxit=0, n_imputations=10)
    observed = set(df["bmi"].dropna())
    starts = [v for completed in out for v in completed.loc[BMI_MISSING, "bmi"]]
    assert set(starts) <= observed, "a start value of bmi is not an observed value"
    assert len(set(starts)) > 5
    assert not all(
        np.allclose(completed.loc[BMI_MISSING, "bmi"], _bmi(completed.loc[BMI_MISSING])) for completed in out
    )
    ignore = np.zeros(40, dtype=bool)
    ignore[3:30] = True
    _, out, _ = _run(df, passive=BMI, predictor_matrix=predictors, maxit=0, n_imputations=10, ignore=ignore)
    allowed = set(df.loc[~ignore, "bmi"].dropna())
    assert {v for completed in out for v in completed.loc[BMI_MISSING, "bmi"]} <= allowed


def test_meanobs_start_of_a_passive_column_is_the_observed_value_closest_to_the_mean():
    df, predictors = _body_frame()
    _, out, _ = _run(df, passive=BMI, predictor_matrix=predictors, maxit=0, initial="meanobs")
    observed = df["bmi"].dropna()
    closest = observed.iloc[int(np.argmin(np.abs(observed.to_numpy() - observed.mean())))]
    for completed in out:
        assert completed.loc[BMI_MISSING, "bmi"].tolist() == [closest] * len(BMI_MISSING)


def test_without_observed_values_in_the_rows_that_are_not_ignored_it_starts_from_normal_draws():
    """Every observed bmi lies in an ignored row: an imputed column would stop the run, a passive one starts
    from standard normal draws, as R's rnorm, and holds its function after one iteration."""
    df, predictors = _body_frame()
    ignore = df["bmi"].notna().to_numpy()
    mice, out, _ = _run(df, passive=BMI, predictor_matrix=predictors, maxit=0, n_imputations=3, ignore=ignore)
    starts = np.concatenate([completed.loc[BMI_MISSING, "bmi"].to_numpy() for completed in out])
    assert np.isfinite(starts).all() and np.abs(starts).max() < 5 and not set(starts) & set(df["bmi"].dropna())
    for completed in _continue(mice):
        _assert_bmi_of_its_row(completed)


def _product_frame():
    """``x1`` complete; ``x2`` missing in rows 0, 4, 8, 12; ``p`` (passive, ``x1 * x2``) measured with a small
    error, missing with x2 and in rows 16, 17; ``z = 1 + x1 x2 + noise``, missing in rows 2, 4, 6, 10, 18.
    Returns (data, predictor matrix: x2 from x1, z from p only)."""
    rng = np.random.default_rng(21)
    n = 24
    x1, x2 = rng.normal(size=n), rng.normal(size=n)
    df = pd.DataFrame(
        {"x1": x1, "x2": x2, "p": x1 * x2 + rng.normal(scale=0.2, size=n), "z": 1 + x1 * x2 + rng.normal(size=n)}
    )
    df.loc[[0, 4, 8, 12], "x2"] = NAN
    df.loc[[0, 4, 8, 12, 16, 17], "p"] = NAN
    df.loc[[2, 4, 6, 10, 18], "z"] = NAN
    predictors = pd.DataFrame(0, index=["x2", "z"], columns=df.columns)
    predictors.loc["x2", "x1"] = 1
    predictors.loc["z", "p"] = 1
    return df, predictors


PRODUCT = {"p": lambda d: d["x1"] * d["x2"]}
PRODUCT_METHODS = {"x2": "mean", "z": "norm.predict"}


def _product_expectation(df, x2_method="mean", where_all=False):
    """The completed p and z after two iterations: x2 holds its mean (or stays missing without a method), p the
    product in its where-cells, z norm.predict from p as the chains see it (observed p kept there)."""
    x1, x2, p, z = (df[col].to_numpy() for col in ("x1", "x2", "p", "z"))
    x2_work = x2.copy()
    if x2_method == "mean":
        x2_work[np.isnan(x2)] = np.nanmean(x2)
    product = x1 * x2_work
    p_where = np.ones(len(df), dtype=bool) if where_all else np.isnan(p)
    p_work = np.where(np.isnan(p), product, p)
    z_missing = np.isnan(z)
    fit = ~z_missing & ~np.isnan(p_work)
    rows = z_missing & ~np.isnan(p_work)
    completed_p = np.where(p_where, product, p)
    return completed_p, rows, _fitted(z, p_work, fit, rows)


def test_a_passive_column_predicts_another_column():
    """z is visited before p (the default order): from iteration 2 on it is fitted on the passive values."""
    df, predictors = _product_frame()
    mice, out, _ = _run(df, method=PRODUCT_METHODS, passive=PRODUCT, predictor_matrix=predictors)
    assert mice.visit_sequence == ["x2", "z", "p"]
    completed_p, rows, expected_z = _product_expectation(df)
    for completed in out:
        np.testing.assert_allclose(completed["p"], completed_p, rtol=1e-12)
        np.testing.assert_allclose(
            completed.loc[rows, "z"], expected_z, rtol=1e-10, err_msg="z is not predicted from the passive values"
        )


def test_an_input_that_keeps_missing_values_makes_the_passive_values_missing():
    """x2 has no method: p is missing where x2 is, and those rows drop out of the model of z, whose where-cell in
    row 4 stays missing."""
    df, predictors = _product_frame()
    mice, out, messages = _run(df, method={**PRODUCT_METHODS, "x2": ""}, passive=PRODUCT, predictor_matrix=predictors)
    assert messages == []
    completed_p, rows, expected_z = _product_expectation(df, x2_method="")
    assert rows.sum() == 4
    for completed in out:
        assert completed.loc[[0, 4, 8, 12], "p"].isna().all() and completed.loc[[16, 17], "p"].notna().all()
        np.testing.assert_allclose(completed["p"], completed_p, rtol=1e-12)
        assert np.isnan(completed.loc[4, "z"])
        np.testing.assert_allclose(completed.loc[rows, "z"], expected_z, rtol=1e-10)
    np.testing.assert_allclose(mice.chain_mean["p"], completed_p[[16, 17]].mean(), rtol=1e-12)


def test_where_on_observed_cells_gives_the_function_in_the_completed_data_and_keeps_the_observed_value():
    """Every cell of p is a where-cell: the completed data hold x1 * x2 in every row, while the chains keep the
    observed p (measured with an error), from which z is predicted, as in R."""
    df, predictors = _product_frame()
    where = df.isna()
    where["p"] = True
    _, out, messages = _run(df, method=PRODUCT_METHODS, passive=PRODUCT, predictor_matrix=predictors, where=where)
    assert messages == []
    completed_p, rows, expected_z = _product_expectation(df, where_all=True)
    for completed in out:
        np.testing.assert_allclose(completed["p"], completed_p, rtol=1e-12)
        assert not np.allclose(completed["p"], df["p"], equal_nan=True)
        np.testing.assert_allclose(
            completed.loc[rows, "z"], expected_z, rtol=1e-10, err_msg="z is not predicted from the observed p"
        )


def test_observed_passive_cells_lost_where_the_function_is_missing():
    """p is observed in row 0, where x2 (no method) is missing: under where on all cells of p it is lost."""
    df, predictors = _product_frame()
    df.loc[0, "p"] = 0.5
    where = df.isna()
    where["p"] = True
    _, out, messages = _run(
        df, method={**PRODUCT_METHODS, "x2": ""}, passive=PRODUCT, predictor_matrix=predictors, where=where
    )
    assert messages == [
        "Observed values replaced by missing values in the completed data, because where marks them and they could "
        "not be imputed (as in R); per column: 'p': 1 (passive function missing in these rows). See MICE.where."
    ]
    assert all(completed.loc[[0, 4, 8, 12], "p"].isna().all() for completed in out)


def test_a_function_that_is_not_row_wise_sees_all_rows():
    """c = x - mean(x): the mean is that of all 20 rows (R takes it over the rows being imputed)."""
    rng = np.random.default_rng(4)
    x = rng.normal(size=20)
    df = pd.DataFrame({"x": x, "c": x - x.mean() + rng.normal(scale=0.1, size=20)})
    rows = [0, 1, 2, 7]
    df.loc[rows, "c"] = NAN
    _, out, _ = _run(df, passive={"c": lambda d: d["x"] - d["x"].mean()})
    for completed in out:
        np.testing.assert_allclose(
            completed.loc[rows, "c"], x[rows] - x.mean(), rtol=1e-12, err_msg="the function did not see all rows"
        )


# ---------------------------------------------------------------------------
# setup, dtypes
# ---------------------------------------------------------------------------


def test_a_passive_copy_of_a_column_is_removed_as_collinear():
    """h2 = 2 x1 is collinear with x1: method "", logged, its missing values kept (as R's hgt2 = 2 * hgt)."""
    df, predictors = _product_frame()
    df["h2"] = 2 * df["x1"]
    df.loc[[3, 7], "h2"] = NAN
    mice, out, messages = _run(df, method=PRODUCT_METHODS, passive={**PRODUCT, "h2": lambda d: 2 * d["x1"]})
    assert mice.method["h2"] == "" and mice.method["p"] == "passive"
    assert list(mice.passive) == ["p"] and "h2" not in mice.visit_sequence
    assert mice.logged_events.to_dict("records") == [{"it": 0, "im": 0, "dep": "", "meth": "collinear", "out": "h2"}]
    assert messages == [
        "Number of logged events: 1. Not imputed because constant or collinear, their missing values are kept: "
        "['h2']. See MICE.logged_events."
    ]
    for completed in out:
        assert completed.loc[[3, 7], "h2"].isna().all()


def test_a_complete_passive_column_is_not_computed():
    df, predictors = _product_frame()
    calls = []
    mice, out, messages = _run(
        df,
        method=PRODUCT_METHODS,
        predictor_matrix=predictors,
        passive={**PRODUCT, "x1": _counting(calls, PRODUCT["p"])},
    )
    assert mice.method["x1"] == "" and list(mice.passive) == ["p"] and messages == []
    assert calls == [24]  # the check call only
    for completed in out:
        pd.testing.assert_series_equal(completed["x1"], df["x1"])


@pytest.mark.parametrize(
    "column, function, dtype",
    [
        (lambda: pd.Series(NAN, index=range(30)), lambda d: d["x1"] + 1, "float64"),
        (lambda: pd.Series([None] * 30, dtype=object), lambda d: d["x1"] + 1, "float64"),
        (lambda: pd.Series(NAN, index=range(30)), lambda d: (d["x1"] + 1).astype("float32"), "float32"),
        (lambda: pd.Series(pd.NA, index=range(30), dtype="Int64"), lambda d: (d["x1"] > 0).astype(int), "Float64"),
    ],
    ids=["nan", "none", "float32", "integers"],
)
def test_an_entirely_missing_passive_column_starts_from_normal_draws(column, function, dtype):
    """As in R: kept by the setup step, started from standard normal draws, its function after one iteration;
    it takes the dtype of the result (Float64 for integers, since the draws are floats)."""
    rng = np.random.default_rng(5)
    df = pd.DataFrame({"x1": rng.normal(size=30), "q": column()})
    mice, out, messages = _run(df, passive={"q": function}, maxit=0, n_imputations=3)
    assert mice.method["q"] == "passive" and messages == []
    starts = np.concatenate([completed["q"].to_numpy(dtype=float) for completed in out])
    assert np.isfinite(starts).all() and starts.std() > 0.5
    expected = function(df).to_numpy(dtype=float)
    for completed in _continue(mice):
        assert str(completed["q"].dtype) == dtype
        np.testing.assert_allclose(completed["q"].to_numpy(dtype=float), expected, rtol=1e-6)
    assert mice.chain_mean["q"].shape == (1, 3)


def test_an_entirely_missing_passive_column_with_another_function_raises():
    rng = np.random.default_rng(5)
    df = pd.DataFrame({"x1": rng.normal(size=30), "q": NAN})
    with pytest.raises(
        ValueError,
        match=r"^The passive column 'q' has no observed value at all to start from, and its function returns values "
        r"of dtype object, not numbers: R starts such a column from standard normal draws, which only a numeric "
        r"column can take\. Give it observed values to start from, or impute it with a method instead\.$",
    ):
        MICE(df).impute(passive={"q": lambda d: np.where(d["x1"] > 0, "hi", "lo")})


def test_an_integer_column_with_a_float_function_becomes_float64():
    """k (Int64) is passive with a function that returns floats, k64 (int64, complete) under where too; m (Int64)
    with a function that returns integers stays Int64."""
    rng = np.random.default_rng(9)
    x1 = rng.normal(size=20)
    df = pd.DataFrame(
        {
            "x1": x1,
            "k": pd.array(rng.integers(0, 10, size=20), dtype="Int64"),
            "k64": rng.integers(0, 10, size=20),
            "m": pd.array(rng.integers(0, 10, size=20), dtype="Int64"),
        }
    )
    df.loc[[1, 2], "k"] = pd.NA
    df.loc[[3, 4], "m"] = pd.NA
    where = df.isna()
    where.loc[[5, 6], "k64"] = True
    passive = {"k": lambda d: d["x1"] / 2, "k64": lambda d: d["x1"] * 3, "m": lambda d: d["k64"] * 2}
    with warnings.catch_warnings():
        warnings.simplefilter("error", FutureWarning)
        _, out, _ = _run(df, passive=passive, where=where)
    for completed in out:
        assert completed.dtypes.astype(str).to_dict() == {
            "x1": "float64",
            "k": "Float64",
            "k64": "Float64",
            "m": "Int64",
        }
        np.testing.assert_allclose(completed.loc[[1, 2], "k"].to_numpy(dtype=float), x1[[1, 2]] / 2)
        np.testing.assert_allclose(completed.loc[[5, 6], "k64"].to_numpy(dtype=float), x1[[5, 6]] * 3)
        # from the observed k64 of rows 3 and 4, which are not where-cells
        assert completed.loc[[3, 4], "m"].tolist() == (df.loc[[3, 4], "k64"] * 2).tolist()


def test_values_the_passive_column_cannot_hold_raise_before_the_chains_start():
    df, predictors = _body_frame()
    with pytest.raises(
        ValueError,
        match=r"^Column 'bmi', passive \(called once on the data before the chains start\): the function returned "
        r"values that are not numbers, which the column \(dtype float64\) cannot hold: '(thin|heavy)', ",
    ):
        MICE(df).impute(passive={"bmi": lambda d: np.where(_bmi(d) < 18, "thin", "heavy")})


@pytest.mark.parametrize(
    "function, message",
    [
        (lambda d: _bmi(d)[::-1], "the function returned a Series with another index; it must"),
        (lambda d: _bmi(d).iloc[1:], "the function returned 39 values for the 40 rows of the data; it must"),
        (lambda d: d[["hgt", "wgt"]], "the function returned a DataFrame; it must return one value per row"),
        (lambda d: d["nope"], "the function raised KeyError: 'nope'"),
    ],
    ids=["other-index", "short", "two-columns", "raises"],
)
def test_impute_stops_before_the_chains_for_a_function_that_fails_on_the_data(function, message):
    df, predictors = _body_frame()
    mice = MICE(df)
    with pytest.raises(
        ValueError, match=rf"^Column 'bmi', passive \(called once on the data before the chains start\): {message}"
    ):
        mice.impute(passive={"bmi": function}, predictor_matrix=predictors)
    assert not hasattr(mice, "imputed_datasets")


def test_errors_of_the_function_inside_the_chains_name_column_iteration_and_imputation():
    df, predictors = _body_frame()
    calls = []

    def failing_at_the_third_call(d):
        calls.append(1)
        if len(calls) == 3:  # the check call, chain 1 iteration 1, chain 1 iteration 2
            raise RuntimeError("third call")
        return _bmi(d)

    mice = MICE(df)
    with pytest.raises(
        ValueError, match=r"^Column 'bmi', passive \(iteration 2, imputation 1\): the function raised RuntimeError: "
    ) as error:
        mice.impute(passive={"bmi": failing_at_the_third_call}, predictor_matrix=predictors, n_imputations=2, maxit=2)
    assert isinstance(error.value.__cause__, RuntimeError)
    calls.clear()

    def short_at_the_second_call(d):
        calls.append(1)
        return _bmi(d).iloc[: 40 - len(calls) + 1]

    with pytest.raises(
        ValueError,
        match=r"^Column 'bmi', passive \(iteration 1, imputation 1\): the function returned 39 values for the 40 "
        r"rows of the data; it must return one value per row",
    ):
        mice.impute(passive={"bmi": short_at_the_second_call}, predictor_matrix=predictors)


# ---------------------------------------------------------------------------
# the visit sequence
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "frame, method",
    [(lambda: numeric_frame(n=40, seed=11), "pmm"), (lambda: typed_frame(n=40, seed=3), None)],
    ids=["numeric", "typed"],
)
def test_visit_sequence_none_equals_roman_without_passive_columns(frame, method):
    df = frame()
    records = []
    for visit_sequence in (None, "roman"):
        mice, out, _ = _run(df, visit_sequence=visit_sequence, method=method)
        records.append(_record(mice, out))
    assert _differences(*records) == []
    assert records[0]["visit_sequence"] == [col for col in df.columns if df[col].isna().any()]


def _order_frame():
    """Columns a (complete), p (passive, h + w, 2 where-cells), t (6), h (3), w (5)."""
    rng = np.random.default_rng(2)
    n = 30
    h, w = rng.normal(size=n), rng.normal(size=n)
    df = pd.DataFrame({"a": rng.normal(size=n), "p": h + w + 0.1 * rng.normal(size=n), "t": rng.normal(size=n)})
    df["h"], df["w"] = h, w
    df.loc[[20, 21], "p"] = NAN
    df.loc[10:15, "t"] = NAN
    df.loc[0:2, "h"] = NAN
    df.loc[3:7, "w"] = NAN
    return df


SUM = {"p": lambda d: d["h"] + d["w"]}


@pytest.mark.parametrize(
    "visit_sequence, expected",
    [
        (None, ["t", "h", "w", "p"]),
        ("roman", ["p", "t", "h", "w"]),
        ("arabic", ["w", "h", "t", "p"]),
        ("monotone", ["p", "h", "w", "t"]),
        ("revmonotone", ["t", "w", "h", "p"]),
    ],
)
def test_the_order_of_a_passive_column_under_each_keyword(visit_sequence, expected):
    """Only None moves the passive column last; a keyword places it like any other column, as in R."""
    mice, _, _ = _run(_order_frame(), passive=SUM, visit_sequence=visit_sequence, maxit=0)
    assert mice.visit_sequence == expected, "the order of the passive column is not that of the keyword"


def test_random_order_places_the_passive_column_anywhere():
    orders = [
        _run(_order_frame(), passive=SUM, visit_sequence="random", maxit=0, seed=s)[0].visit_sequence for s in range(8)
    ]
    assert all(sorted(order) == ["h", "p", "t", "w"] for order in orders)
    assert {order.index("p") for order in orders} != {3}


def test_a_list_must_contain_the_passive_column():
    with pytest.raises(
        ValueError,
        match=r"^Visit sequence is missing columns that are imputed: \['p'\]\. Every column that is imputed \(a "
        r"column with where-cells, by default an incomplete column, and a method other than '', passive columns "
        r"included\) must be in the visit sequence\.",
    ):
        MICE(_order_frame()).impute(passive=SUM, visit_sequence=["t", "h", "w"])


def test_a_list_with_repeats_and_columns_that_are_not_visited():
    mice, _, _ = _run(_order_frame(), passive=SUM, visit_sequence=["a", "t", "h", "p", "a", "w", "p"], maxit=1)
    assert mice.visit_sequence == ["t", "h", "p", "w", "p"]
    assert list(mice.chain_mean) == ["t", "h", "p", "w"]


def _twice_frame():
    """``x`` complete; ``y = x + z + noise`` missing in rows 0, 3, 6; ``z`` missing in rows 1, 2, 8, 9, 12."""
    rng = np.random.default_rng(14)
    x, z = rng.normal(size=20), rng.normal(size=20)
    df = pd.DataFrame({"x": x, "y": x + z + rng.normal(scale=0.3, size=20), "z": z})
    df.loc[[0, 3, 6], "y"] = NAN
    df.loc[[1, 2, 8, 9, 12], "z"] = NAN
    return df


def test_chain_statistics_of_a_column_visited_twice_are_those_of_its_last_visit():
    """y is visited before and after z: its first visit predicts from the start of z, the second from the mean
    imputations of z. The chain statistics hold the imputations at the end of the iteration."""
    df = _twice_frame()
    mice, out, _ = _run(df, method={"y": "norm.predict", "z": "mean"}, visit_sequence=["y", "z", "y"], maxit=1)
    assert mice.visit_sequence == ["y", "z", "y"]
    z_work = df["z"].fillna(df["z"].mean()).to_numpy()
    y = df["y"].to_numpy()
    y_missing = np.isnan(y)
    expected = _fitted(y, np.column_stack([df["x"], z_work]), ~y_missing, y_missing)
    for chain, completed in enumerate(out):
        np.testing.assert_allclose(completed.loc[y_missing, "y"], expected, rtol=1e-10)
        assert np.isclose(mice.chain_mean["y"][0, chain], expected.mean(), rtol=1e-12), (
            "chain_mean of 'y' is not taken at the end of the iteration"
        )
        assert np.isclose(mice.chain_var["y"][0, chain], expected.var(ddof=1), rtol=1e-10)


# ---------------------------------------------------------------------------
# the NaN cascade with a passive column
# ---------------------------------------------------------------------------

N_ORACLE_CASES = 8


def _passive_oracle_case(seed):
    """numeric_frame with p = a + b (missing where a or b is, and in three more rows), a random mask, predictor
    matrix and methods ("" for some columns), a visit sequence and 0 to 3 iterations."""
    rng = np.random.default_rng([20261114, seed])
    df = numeric_frame(n=30, seed=seed)
    df["p"] = df["a"] + df["b"]
    df.loc[df.index[rng.choice(30, 3, replace=False)], "p"] = NAN
    method = {col: ("" if rng.random() < 0.2 else ("norm.predict", "pmm")[seed % 2]) for col in df.columns[:-1]}
    kwargs = {
        "where": _random_where(df, rng, left_out=0.3, added=0.15),
        "method": method,
        "passive": {"p": lambda d: d["a"] + d["b"]},
        "visit_sequence": (None, "roman", "arabic", "monotone")[seed % 4],
        "maxit": seed % 4,
        "predictor_matrix": _random_predictor_matrix(df, rng),
    }
    return df, kwargs


@pytest.mark.parametrize("seed", range(N_ORACLE_CASES))
def test_missing_cells_follow_the_mask_oracle_with_a_passive_column(seed):
    """`nan_cascade` with p reading a and b: a passive value is missing exactly where one of its inputs is
    missing in the data the chains work on (this holds for a sum)."""
    df, kwargs = _passive_oracle_case(seed)
    mice, out, _ = _run(df, **kwargs)
    predictors = {col: predictors_of(mice, col) for col in mice.visit_sequence if mice.method[col] != "passive"}
    expected = nan_cascade(
        df, mice.where, mice.method, mice.visit_sequence, predictors, kwargs["maxit"], passive={"p": ["a", "b"]}
    )
    for completed in out:
        pd.testing.assert_frame_equal(completed.isna(), expected)


def test_the_oracle_cases_have_a_passive_column_with_missing_values():
    """So that the oracle test cannot pass without a passive column, or with one that never goes missing."""
    visited = with_missing = 0
    for seed in range(N_ORACLE_CASES):
        df, kwargs = _passive_oracle_case(seed)
        mice, out, _ = _run(df, **kwargs)
        visited += int("p" in mice.visit_sequence and kwargs["maxit"] > 0)
        with_missing += int(out[0].loc[mice.where["p"], "p"].isna().any() and kwargs["maxit"] > 0)
    assert visited >= 5 and with_missing >= 3, (visited, with_missing)


# ---------------------------------------------------------------------------
# mutants
# ---------------------------------------------------------------------------
#
# Each applies one defect by patching one function for one test, checks that
# it was applied, and runs a named test of this file, which must fail with
# its own message:
#
# * the chain statistics of a column visited twice taken at its first visit
#   (MICE._iterate, edited);
# * the passive columns moved last also under an explicit keyword
#   (MICE._set_visit_sequence);
# * the passive columns not moved last without a visit sequence, so computed
#   before their inputs where data order puts them first (_passive_last);
# * passive values written over the observed cells of the working data
#   (_refresh_working_column called by MICE._visit_passive);
# * the start of a passive column computed by its function
#   (MICE._initial_imputation);
# * the passive function given only the rows being imputed, as R's
#   model.frame on data[wy, ] (MICE._passive_values).


def test_chain_statistics_at_the_first_visit_are_caught(monkeypatch):
    reached = edit_iterate(
        monkeypatch,
        ("    for col in self.visit_sequence:\n", "    first_visit = {}\n    for col in self.visit_sequence:\n"),
        (
            "        self._apply_post(working, imp, col, iter_idx, chain_idx)\n\n"
            "    self._record_chain_statistics(state.imp,",
            "        self._apply_post(working, imp, col, iter_idx, chain_idx)\n"
            "        if col in first_visit:\n"
            "            _reached()\n"
            "        first_visit.setdefault(col, imp.copy())\n\n"
            "    self._record_chain_statistics({**state.imp, **first_visit},",
        ),
    )
    with pytest.raises(AssertionError, match="chain_mean of 'y' is not taken at the end of the iteration"):
        test_chain_statistics_of_a_column_visited_twice_are_those_of_its_last_visit()
    assert reached, "mutant not applied: no column visited twice"


def test_passive_columns_last_under_a_keyword_are_caught(monkeypatch):
    calls = []
    original = MICE._set_visit_sequence

    def passive_last(self, visit_sequence, rng=None):
        original(self, visit_sequence, rng)
        moved = MICE_MODULE._passive_last(self.visit_sequence, self.method)
        calls.append(moved != self.visit_sequence)
        self.visit_sequence = moved

    monkeypatch.setattr(MICE, "_set_visit_sequence", passive_last)
    with pytest.raises(AssertionError, match="the order of the passive column is not that of the keyword"):
        test_the_order_of_a_passive_column_under_each_keyword("roman", ["p", "t", "h", "w"])
    with pytest.raises(AssertionError, match="the passive column is moved under a keyword"):
        test_roman_with_the_passive_column_before_its_inputs_breaks_the_relation()
    assert any(calls), "mutant not applied"


def test_passive_columns_not_moved_last_by_default_are_caught(monkeypatch):
    calls = []

    def data_order(columns, method):
        calls.append(columns)
        return list(columns)

    monkeypatch.setattr(MICE_MODULE, "_passive_last", data_order)
    # data order puts bmi before hgt and wgt, which are imputed after it
    with pytest.raises(AssertionError, match="bmi is not the function of its row"):
        test_every_completed_dataset_holds_the_function_of_its_rows()
    assert calls, "mutant not applied"


def test_passive_values_over_observed_cells_of_the_working_data_are_caught(monkeypatch):
    calls = []
    original = MICE_MODULE._refresh_working_column

    def every_where_cell(working, column, imp, where, observed):
        if sys._getframe(1).f_code.co_name != "_visit_passive":
            original(working, column, imp, where, observed)
            return
        calls.append((where & observed).any())
        working.loc[where, column] = MICE_MODULE._values(imp)

    monkeypatch.setattr(MICE_MODULE, "_refresh_working_column", every_where_cell)
    with pytest.raises(AssertionError, match="z is not predicted from the observed p"):
        test_where_on_observed_cells_gives_the_function_in_the_completed_data_and_keeps_the_observed_value()
    assert any(calls), "mutant not applied: no observed where-cell of a passive column"


def test_start_computed_by_the_function_is_caught(monkeypatch):
    calls = []
    original = MICE._initial_imputation

    def by_the_function(self, rng):
        imp = original(self, rng)
        working = self._working_frame(imp)
        for col, function in self.passive.items():
            calls.append(col)
            imp[col].iloc[:] = function(working).to_numpy()[self.where[col].to_numpy()]
        return imp

    monkeypatch.setattr(MICE, "_initial_imputation", by_the_function)
    with pytest.raises(AssertionError, match="a start value of bmi is not an observed value"):
        test_start_values_are_sampled_observed_values()
    assert calls, "mutant not applied"


def test_function_given_only_the_rows_being_imputed_is_caught(monkeypatch):
    calls = []

    def where_rows_only(self, col, working, when):
        where = self.where[col].to_numpy()
        calls.append(int(where.sum()))
        values = self.passive[col](working[where].copy())
        return values.reindex(working.index).to_numpy()

    monkeypatch.setattr(MICE, "_passive_values", where_rows_only)
    with pytest.raises(AssertionError, match="the function did not see all rows"):
        test_a_function_that_is_not_row_wise_sees_all_rows()
    assert calls and all(n < 20 for n in calls), "mutant not applied"
