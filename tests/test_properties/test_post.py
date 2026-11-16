"""``impute(post=...)``: functions that change the imputations of a column right after they are drawn, as R's ``post``.

After each univariate imputation of a column, in every iteration and chain,
its function ``f(imp, data)`` gets copies of the imputations of all its
where-cells (``imp``) and of the data the chains work on in the same rows
(``data``), and returns the values to keep; they replace the imputations,
and the working data are refreshed from them before the next column is
visited. The chain statistics are taken at the end of the iteration, so
after ``post``. A column that is not imputed never calls its function, and
``MICE.post`` holds the functions in effect.

The expectations below are computed by hand or by a few lines of NumPy with
the deterministic methods ``mean`` and ``norm.predict``. The rules tested
are those of R's ``sampler`` and ``mice.edit.setup`` (mice 3.19.0); R's
results for the same cases are fixtures of a later phase.

The mutants at the end introduce the defects that ``post`` can have, each by
patching one function, and check that a named test of this file fails.
"""

import functools
import importlib
import sys
import types
import warnings

import numpy as np
import pandas as pd
import pytest

from imputation import MICE
from imputation.validators import check_post

from ._helpers import edit_iterate, method_spec, typed_frame
from .test_continue_chains import _differences, _record

MICE_MODULE = importlib.import_module("imputation.MICE")
NAN = np.nan
SHIFT = 10.0
Y_MISSING = [0, 3, 6, 9]


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


def _fitted(y, x, fit_rows, rows):
    """The least-squares prediction of `y` from `x` with an intercept, fitted on `fit_rows`, at `rows`
    (what ``norm.predict`` imputes)."""
    design = np.column_stack([np.ones(len(y)), x])
    coef, *_ = np.linalg.lstsq(design[fit_rows], y[fit_rows], rcond=None)
    return design[rows] @ coef


def _xy_frame():
    """``x`` complete; ``y = 1 + 2 x + noise``, missing in rows 0, 3, 6 and 9 (Y_MISSING)."""
    rng = np.random.default_rng(3)
    x = rng.normal(size=20)
    df = pd.DataFrame({"x": x, "y": 1 + 2 * x + rng.normal(size=20)})
    df.loc[Y_MISSING, "y"] = NAN
    return df


def _y_fitted(df):
    """What ``norm.predict`` imputes in the missing cells of ``y`` of `_xy_frame`, from ``x``."""
    x, y = df["x"].to_numpy(), df["y"].to_numpy()
    missing = np.isnan(y)
    return _fitted(y, x, ~missing, missing)


# ---------------------------------------------------------------------------
# check_post
# ---------------------------------------------------------------------------


def _clip(imp, data):
    return imp.clip(1, 25)


class _Callable:
    def __call__(self, imp, data):
        return imp


@pytest.mark.parametrize(
    "post",
    [
        {"y": _clip},
        {"y": lambda imp, data: imp, "x": _clip},
        types.MappingProxyType({"y": _clip}),
        {"y": _Callable()},
        {"y": functools.partial(lambda imp, data, low: imp.clip(low), low=0)},
        {},
    ],
    ids=["dict", "two", "mapping", "callable-object", "partial", "empty"],
)
def test_check_post_takes_a_mapping_of_functions(post):
    df = _xy_frame()
    got = check_post(post, df)
    assert type(got) is dict and got == dict(post)
    # a new dict
    got["other"] = _clip
    assert "other" not in post


def test_check_post_none_is_no_function():
    assert check_post(None, _xy_frame()) == {}


_POST_FORM = (
    r"a mapping from column labels to functions f\(imp, data\), for example \{'tv': lambda imp, data: "
    r"imp\.clip\(1, 25\)\}"
)


@pytest.mark.parametrize(
    "post, message",
    [
        ([_clip], rf"^post must be None or {_POST_FORM}; got list\.$"),
        (_clip, r"^post must be None or a mapping .*; got function\.$"),
        ("y", r"^post must be None or a mapping .*; got str\.$"),
        ({"y": _clip, "nope": _clip, 3: _clip}, r"^post names columns that are not in the data: \['nope', 3\]\.$"),
        (
            {"y": 1.5},
            r"^post must map each column to a function f\(imp, data\); the entry of the column 'y' is a float, which "
            r"cannot be called\.$",
        ),
        ({"x": _clip, "y": "imp.clip(1, 25)"}, r"^post must map each column .*; the entry of the column 'y' is a str"),
    ],
    ids=["list", "function", "string", "unknown-columns", "number", "text"],
)
def test_check_post_errors(post, message):
    with pytest.raises(ValueError, match=message):
        check_post(post, _xy_frame())


def test_impute_checks_post_before_it_changes_anything():
    df = _xy_frame()
    mice, out, _ = _run(df, maxit=1)
    with pytest.raises(ValueError, match=r"^post names columns that are not in the data: \['z'\]"):
        mice.impute(post={"z": _clip})
    assert mice.imputed_datasets is out
    mice.continue_chains()


# ---------------------------------------------------------------------------
# what post does
# ---------------------------------------------------------------------------

BOUNDS = (0.9, 1.1)


def test_bounds_hold_in_every_imputation():
    """norm draws spread around the fitted values; the bounds clip them, R's squeeze."""
    df = _xy_frame()
    mice, out, messages = _run(df, method="norm", n_imputations=3, post={"y": lambda imp, data: imp.clip(*BOUNDS)})
    assert messages == []
    clipped = 0
    for completed in out:
        imputed = completed.loc[Y_MISSING, "y"]
        assert imputed.between(*BOUNDS).all(), "an imputation of y lies outside the bounds"
        clipped += int(imputed.isin(BOUNDS).sum())
        pd.testing.assert_frame_equal(completed.drop(index=Y_MISSING), df.drop(index=Y_MISSING))
    assert clipped > 0
    assert list(mice.post) == ["y"]


def test_a_shift_is_added_to_every_imputation():
    df = _xy_frame()
    _, out, _ = _run(df, method="norm.predict", post={"y": lambda imp, data: imp + SHIFT})
    for completed in out:
        np.testing.assert_allclose(completed.loc[Y_MISSING, "y"], _y_fitted(df) + SHIFT, rtol=1e-12)


def test_a_rule_can_read_another_column():
    df = _xy_frame()
    _, out, _ = _run(df, method="norm.predict", post={"y": lambda imp, data: imp.where(data["x"] > 0, -100.0)})
    x = df.loc[Y_MISSING, "x"].to_numpy()
    expected = np.where(x > 0, _y_fitted(df), -100.0)
    assert (x > 0).any() and (x <= 0).any()
    for completed in out:
        np.testing.assert_allclose(completed.loc[Y_MISSING, "y"], expected, rtol=1e-12)


def test_what_the_function_receives():
    """y has where-cells in its missing rows and in the observed rows 1 and 2. In every call, imp holds what
    norm.predict just stored for all six where-cells, data the working data in the same rows: the new
    imputations in the missing cells of y, the observed values in rows 1 and 2."""
    df = _xy_frame()
    where = df.isna()
    where.loc[[1, 2], "y"] = True
    calls = []

    def recording(imp, data):
        calls.append((imp.copy(), data.copy()))
        return imp

    _run(df, method="norm.predict", where=where, post={"y": recording})
    rows = [0, 1, 2, 3, 6, 9]
    x, y = df["x"].to_numpy(), df["y"].to_numpy()
    fitted = _fitted(y, x, ~np.isnan(y), np.isin(np.arange(20), rows))
    assert len(calls) == 4  # two chains, two iterations
    for imp, data in calls:
        assert isinstance(imp, pd.Series) and imp.index.tolist() == rows
        np.testing.assert_allclose(imp.to_numpy(), fitted, rtol=1e-12)
        assert isinstance(data, pd.DataFrame) and data.index.tolist() == rows and list(data.columns) == ["x", "y"]
        pd.testing.assert_series_equal(data["x"], df.loc[rows, "x"])
        np.testing.assert_allclose(data.loc[Y_MISSING, "y"], imp[Y_MISSING], rtol=0)
        assert data.loc[[1, 2], "y"].tolist() == df.loc[[1, 2], "y"].tolist()


def test_the_arguments_are_copies():
    """A function that overwrites both arguments in place and returns their values as it got them gives the
    results of a run without post."""

    def overwriting(imp, data):
        kept = imp.copy()
        imp.iloc[:] = -1e6
        data.iloc[:, :] = -1e6
        return kept

    df = _xy_frame()
    with_post = _run(df, method="norm", post={"y": overwriting})
    without = _run(df, method="norm")
    assert _differences(_record(*with_post[:2]), _record(*without[:2])) == []


def _chain_frame():
    """``x`` complete; ``y = 1 + 2 x + noise``, missing in rows 0, 3, 6, 9; ``z = 3 - y + noise``, missing in
    rows 1, 4, 7. y is imputed from x, z from y: z is fitted on the rows in which y was imputed. Returns (data,
    predictor matrix)."""
    rng = np.random.default_rng(8)
    x = rng.normal(size=20)
    y = 1 + 2 * x + rng.normal(size=20)
    df = pd.DataFrame({"x": x, "y": y, "z": 3 - y + rng.normal(scale=0.5, size=20)})
    df.loc[Y_MISSING, "y"] = NAN
    df.loc[[1, 4, 7], "z"] = NAN
    predictors = pd.DataFrame(0, index=["y", "z"], columns=df.columns)
    predictors.loc["y", "x"] = 1
    predictors.loc["z", "y"] = 1
    return df, predictors


@pytest.mark.parametrize("order, maxit", [(["y", "z"], 1), (["z", "y"], 2)], ids=["y-first", "z-first"])
def test_the_next_columns_are_imputed_from_the_processed_values(order, maxit):
    """y gets norm.predict plus a shift. Visited first, it is shifted before z is fitted on it; visited after z,
    the shift of iteration 1 is in the data z is fitted on in iteration 2. The function runs once per visit, in
    every iteration and chain, on what norm.predict stored."""
    df, predictors = _chain_frame()
    calls = []

    def shift(imp, data):
        calls.append(imp.copy())
        return imp + SHIFT

    _, out, _ = _run(
        df, method="norm.predict", predictor_matrix=predictors, visit_sequence=order, maxit=maxit, post={"y": shift}
    )
    x, y, z = (df[col].to_numpy() for col in ("x", "y", "z"))
    y_missing, z_missing = np.isnan(y), np.isnan(z)
    y_fitted = _fitted(y, x, ~y_missing, y_missing)
    y_processed = y.copy()
    y_processed[y_missing] = y_fitted + SHIFT
    expected_z = _fitted(z, y_processed, ~z_missing, z_missing)
    for completed in out:
        np.testing.assert_allclose(completed.loc[y_missing, "y"], y_fitted + SHIFT, rtol=1e-12)
        np.testing.assert_allclose(
            completed.loc[z_missing, "z"],
            expected_z,
            rtol=1e-10,
            err_msg="z is not predicted from the processed values of y",
        )
    assert len(calls) == 2 * maxit
    for imp in calls:
        np.testing.assert_allclose(imp.to_numpy(), y_fitted, rtol=1e-12)


def test_chain_statistics_are_taken_after_post():
    df = _xy_frame()
    mice, out, _ = _run(df, method="norm.predict", post={"y": lambda imp, data: imp + SHIFT})
    for chain, completed in enumerate(out):
        values = completed.loc[Y_MISSING, "y"].to_numpy()
        for iteration in range(2):
            assert np.isclose(mice.chain_mean["y"][iteration, chain], values.mean(), rtol=1e-12), (
                "chain_mean of 'y' is not taken after post"
            )
            assert np.isclose(mice.chain_var["y"][iteration, chain], values.var(ddof=1), rtol=1e-10)


def test_post_is_not_called_for_a_column_that_is_not_imputed():
    """x is complete (no where-cells), w has the user's method "", c is removed by the setup step (constant):
    their functions are never called and are not in MICE.post, as in R."""
    df = _xy_frame()
    df["w"] = np.cos(np.arange(20.0))
    df.loc[[1, 2], "w"] = NAN
    df["c"] = 4.0
    df.loc[[2, 5], "c"] = NAN
    calls = []

    def recording(imp, data):
        calls.append(imp.name)
        return imp

    keep = functools.partial(lambda imp, data: imp)
    mice, out, messages = _run(
        df, method={"y": "norm.predict", "w": ""}, post={"x": recording, "w": recording, "c": recording, "y": keep}
    )
    assert calls == []
    assert mice.post == {"y": keep}
    assert mice.method == {"x": "", "y": "norm.predict", "w": "", "c": ""}
    assert messages == [
        "Number of logged events: 1. Not imputed because constant or collinear, their missing values are kept: "
        "['c']. See MICE.logged_events."
    ]
    for completed in out:
        assert completed["w"].isna().sum() == 2 and completed["c"].isna().sum() == 2


def test_under_where_all_post_gets_every_where_cell():
    df = _xy_frame()
    seen = []

    def recording(imp, data):
        seen.append((imp.name, imp.index.tolist(), data.index.tolist()))
        return imp + 1.0

    _, out, _ = _run(df, where="all", method={"x": "mean", "y": "norm.predict"}, post={"x": recording})
    assert seen == [("x", list(range(20)), list(range(20)))] * 4
    for completed in out:
        np.testing.assert_allclose(completed["x"], df["x"].mean() + 1.0, rtol=1e-12)


def test_post_is_called_when_no_where_cell_could_be_imputed():
    """y is predicted by z alone (method ""), which is missing in every row in which y is: no where-cell of y can
    be imputed, imp is all missing, and the function is called all the same, as in R."""
    df = _xy_frame()
    df["z"] = df["x"] * 3.0
    df.loc[Y_MISSING, "z"] = NAN
    predictors = pd.DataFrame(0, index=["y"], columns=df.columns)
    predictors.loc["y", "z"] = 1
    seen = []

    def filling(imp, data):
        seen.append(imp.isna().all())
        return imp.fillna(7.0)

    mice, out, _ = _run(df, method={"y": "norm", "z": ""}, predictor_matrix=predictors, post={"y": filling})
    assert seen == [True] * 4
    for completed in out:
        assert completed.loc[Y_MISSING, "y"].tolist() == [7.0] * 4
    np.testing.assert_array_equal(mice.chain_mean["y"], np.full((2, 2), 7.0))


def test_post_on_integer_categorical_boolean_and_text_columns_keeps_their_dtypes():
    rng = np.random.default_rng(6)
    n = 30
    x = rng.normal(size=n)
    df = pd.DataFrame(
        {
            "x": x,
            "k": pd.array(np.round(5 + 3 * x).astype(int), dtype="Int64"),
            "g": pd.Categorical(np.where(x > 0, "hi", "lo"), categories=["lo", "hi", "top"]),
            "b": pd.array(rng.random(n) > 0.5, dtype="boolean"),
            "s": pd.array(np.where(rng.random(n) > 0.5, "u", "v"), dtype="string"),
        }
    )
    df.loc[[0, 5, 10], "k"] = pd.NA
    df.loc[[1, 6, 11], "g"] = NAN
    df.loc[[2, 7], "b"] = pd.NA
    df.loc[[3, 8], "s"] = pd.NA
    post = {
        "k": lambda imp, data: imp.clip(4, 6),
        "g": lambda imp, data: pd.Series("top", index=imp.index),
        "b": lambda imp, data: [True, None],
        "s": lambda imp, data: imp.str.upper(),
    }
    _, out, _ = _run(df, method={"k": "pmm", "g": "polyreg", "b": "logreg", "s": "sample"}, post=post)
    for completed in out:
        assert completed.dtypes.to_dict() == df.dtypes.to_dict()
        assert completed.loc[[0, 5, 10], "k"].between(4, 6).all()
        assert completed.loc[[1, 6, 11], "g"].tolist() == ["top"] * 3
        assert completed.loc[2, "b"] and completed.loc[[7], "b"].isna().all()
        assert set(completed.loc[[3, 8], "s"]) <= {"U", "V"}


@pytest.mark.parametrize("method", ["defaults", "pmm", "cart", "sample", "norm", "mean", "logreg.boot", "polr"])
def test_post_that_returns_imp_gives_the_results_of_no_post(method):
    df = typed_frame(n=40, seed=8)
    kwargs = {} if method == "defaults" else {"method": method_spec(method, df)}
    identity = {col: (lambda imp, data: imp) for col in df.columns}
    records = []
    for post in (identity, None):
        mice, out, _ = _run(df, post=post, **kwargs)
        records.append(_record(mice, out))
    assert _differences(*records) == []


# ---------------------------------------------------------------------------
# errors of the function
# ---------------------------------------------------------------------------

_RESULT = r"it must return one value per entry of imp: a Series with the index of imp, or a sequence of its length\."


@pytest.mark.parametrize(
    "function, message",
    [
        (
            lambda imp, data: imp.iloc[:-1],
            rf"^Column 'y', post \(iteration 1, imputation 1\): the function returned 3 values for the 4 where-cells "
            rf"of the column; {_RESULT}$",
        ),
        (lambda imp, data: [1.0, 2.0], r"^Column 'y', post .*: the function returned 2 values for the 4 where-cells"),
        (lambda imp, data: imp.to_numpy()[:, None], r"the function returned an array of shape \(4, 1\); it must"),
        (
            lambda imp, data: imp[::-1],
            rf"^Column 'y', post \(iteration 1, imputation 1\): the function returned a Series with another index; "
            rf"{_RESULT[:-2]} \(the same labels in the same order\)\.$",
        ),
        (lambda imp, data: imp.reset_index(drop=True), r"the function returned a Series with another index"),
        (lambda imp, data: 1.0, r"^Column 'y', post \(iteration 1, imputation 1\): the function returned a float; "),
        (lambda imp, data: data, r"the function returned a DataFrame; it must return one value per entry of imp"),
        (lambda imp, data: None, r"the function returned a NoneType;"),
        (
            lambda imp, data: ["a", "b", "a", "b"],
            r"^Column 'y', post \(iteration 1, imputation 1\): the function returned values that are not numbers, "
            r"which the column \(dtype float64\) cannot hold: 'a', 'b'\.$",
        ),
        (lambda imp, data: imp > 0, r"the function returned values that are not numbers, which the column"),
    ],
    ids=["short", "list", "2d", "reordered", "other-index", "scalar", "frame", "none", "text", "booleans"],
)
def test_wrong_results_raise(function, message):
    with pytest.raises(ValueError, match=message):
        MICE(_xy_frame()).impute(n_imputations=1, maxit=1, seed=1, method="norm", post={"y": function})


def _typed_post_frame():
    rng = np.random.default_rng(2)
    n = 20
    x = rng.normal(size=n)
    df = pd.DataFrame(
        {
            "x": x,
            "k": pd.array(np.round(5 + 3 * x).astype(int), dtype="Int64"),
            "u": pd.array(np.arange(n) % 4, dtype="UInt8"),
            "g": pd.Categorical(np.where(x > 0, "hi", "lo")),
            "b": pd.array(rng.random(n) > 0.5, dtype="boolean"),
        }
    )
    for col in ("k", "u", "g", "b"):
        df.loc[[0, 5, 10], col] = None
    return df


@pytest.mark.parametrize(
    "column, function, message",
    [
        ("k", lambda imp, data: imp + 0.5, r"numbers that are not whole, which the column \(dtype Int64\) cannot hold"),
        ("u", lambda imp, data: imp.astype(int) - 9, r"numbers outside the range of uint8, which the column"),
        ("k", lambda imp, data: imp.astype(str), r"values that are not numbers, which the column \(dtype Int64\)"),
        (
            "g",
            lambda imp, data: pd.Series("mid", index=imp.index),
            r"^Column 'g', post \(iteration 1, imputation 1\): the function returned values that are not categories "
            r"of the column \(dtype category\): 'mid'\.$",
        ),
        ("b", lambda imp, data: [1, 0, 1], r"values that are not booleans, which the column \(dtype boolean\)"),
    ],
    ids=["not-whole", "range", "text-in-integers", "unknown-category", "numbers-for-booleans"],
)
def test_values_the_column_cannot_hold_raise(column, function, message):
    df = _typed_post_frame()
    with pytest.raises(ValueError, match=message):
        MICE(df).impute(n_imputations=1, maxit=1, seed=1, method="pmm", post={column: function})


def test_an_exception_inside_the_function_names_column_iteration_and_imputation():
    calls = []

    def failing(imp, data):
        calls.append(1)
        if len(calls) == 2:
            return data["nope"]
        return imp

    mice = MICE(_xy_frame())
    with pytest.raises(
        ValueError, match=r"^Column 'y', post \(iteration 2, imputation 1\): the function raised KeyError: 'nope'$"
    ) as error:
        mice.impute(n_imputations=2, maxit=2, seed=1, method="norm", post={"y": failing})
    assert isinstance(error.value.__cause__, KeyError)
    with pytest.raises(ValueError, match="no chains to continue"):
        mice.continue_chains()


# ---------------------------------------------------------------------------
# mutants
# ---------------------------------------------------------------------------
#
# Each applies one defect by patching one function for one test, checks that
# it was applied, and runs a named test of this file, which must fail with
# its own message:
#
# * post not applied (MICE._apply_post);
# * post applied, but the working data not refreshed from it
#   (_refresh_working_column called by MICE._apply_post);
# * post applied in the last iteration only, once per column and chain, as
#   a user who processes the final imputations (MICE._apply_post);
# * the chain statistics taken before post: right after the imputer instead
#   of at the end of the iteration (MICE._iterate, edited).


def test_post_not_applied_is_caught(monkeypatch):
    calls = []
    monkeypatch.setattr(MICE, "_apply_post", lambda self, *args: calls.append(self.post))
    with pytest.raises(AssertionError, match="an imputation of y lies outside the bounds"):
        test_bounds_hold_in_every_imputation()
    assert any(calls), "mutant not applied"


def test_working_data_not_refreshed_after_post_is_caught(monkeypatch):
    calls = []
    original = MICE_MODULE._refresh_working_column

    def not_after_post(working, column, imp, where, observed):
        if sys._getframe(1).f_code.co_name == "_apply_post":
            calls.append(column)
            return
        original(working, column, imp, where, observed)

    monkeypatch.setattr(MICE_MODULE, "_refresh_working_column", not_after_post)
    with pytest.raises(AssertionError, match="z is not predicted from the processed values of y"):
        test_the_next_columns_are_imputed_from_the_processed_values(["y", "z"], 1)
    assert calls, "mutant not applied: MICE._apply_post does not refresh the working data"


def test_post_in_the_last_iteration_only_is_caught(monkeypatch):
    calls = []
    original = MICE._apply_post

    def last_iteration_only(self, working, imp, col, iter_idx, chain_idx):
        if iter_idx == self.maxit - 1:
            calls.append(iter_idx)
            original(self, working, imp, col, iter_idx, chain_idx)

    monkeypatch.setattr(MICE, "_apply_post", last_iteration_only)
    with pytest.raises(AssertionError, match="z is not predicted from the processed values of y"):
        test_the_next_columns_are_imputed_from_the_processed_values(["z", "y"], 2)
    assert calls and set(calls) == {1}, "mutant not applied"


def test_chain_statistics_before_post_are_caught(monkeypatch):
    reached = edit_iterate(
        monkeypatch,
        (
            "        # R's post, right after the univariate imputation\n",
            "        _reached()\n        self._record_chain_statistics(state.imp, iter_idx, chain_idx)\n",
        ),
        ("    self._record_chain_statistics(state.imp, iter_idx, chain_idx)\n    iteration_time", "    iteration_time"),
    )
    with pytest.raises(AssertionError, match="chain_mean of 'y' is not taken after post"):
        test_chain_statistics_are_taken_after_post()
    assert reached, "mutant not applied"
