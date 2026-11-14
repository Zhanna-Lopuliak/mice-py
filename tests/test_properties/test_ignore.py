"""``impute(ignore=...)``: the rows that the imputation models leave out, as R's ``ignore``.

An ignored row is left out of the fit rows of every column (R's ``ry =
complete.cases(x, y) & r & !ignore``), in every iteration and at the start,
where the values are drawn from the observed values of the other rows. Its
where-cells are imputed like any other, and the imputer still receives it.
The setup step, the methods and the visit sequence use all rows; the checks
of each univariate imputation ("df set to 1", the predictors dropped by
``remove_lindep``) count the fit rows only. A column that is imputed and has
no observed value in the rows that are not ignored stops the run before the
chains start (R stops in the first iteration).

The expectations below are computed by hand or by a few lines of NumPy with
the deterministic methods ``mean`` and ``norm.predict``. R's results for the
same rules are in the ``l1_chain`` fixture (test_l1_chain); the rules tested
here are those of R's ``check.ignore``, ``initialize.imp``, ``sampler.univ``,
``check.df`` and ``remove.lindep`` (mice 3.19.0).

The mutants at the end introduce the defects that ``ignore`` can have, each
by patching one function, and check that a named test of this file fails.
"""

import importlib
import warnings

import numpy as np
import pandas as pd
import pytest

from imputation import MICE
from imputation.validators import check_ignore

MICE_MODULE = importlib.import_module("imputation.MICE")
NAN = np.nan
T, F = True, False
FEW_ROWS = "Fewer than 10 rows for fitting the imputation model. Are you sure?"


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
# check_ignore: the accepted forms
# ---------------------------------------------------------------------------


def _twelve():
    """12 rows with a non-default index."""
    return pd.DataFrame({"x": [1.0, NAN, 3.0] * 4}, index=[30 - 2 * i for i in range(12)])


ROWS = [F, T, F, F, F, F, F, F, F, F, F, T]


@pytest.mark.parametrize(
    "make",
    [
        lambda df: list(ROWS),
        lambda df: tuple(ROWS),
        lambda df: [np.bool_(v) for v in ROWS],
        lambda df: np.array(ROWS),
        lambda df: np.array(ROWS, dtype=object),
        lambda df: pd.array(ROWS, dtype="boolean"),
        lambda df: pd.Series(ROWS, index=df.index),
        lambda df: pd.Series(ROWS, index=df.index, dtype="boolean"),
        lambda df: pd.Series(ROWS, index=df.index, dtype=object),
    ],
    ids=[
        "list",
        "tuple",
        "list-numpy-bools",
        "array",
        "array-object",
        "pandas-array",
        "series",
        "series-boolean",
        "series-object",
    ],
)
def test_check_ignore_takes_every_boolean_sequence(make):
    df = _twelve()
    given = make(df)
    got = check_ignore(given, df)
    assert got.dtype == np.dtype(bool) and got.tolist() == ROWS
    # a new array
    got[:] = True
    assert check_ignore(given, df).tolist() == ROWS


def test_check_ignore_none_is_no_row_and_does_not_warn():
    df = _twelve().iloc[:5]
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        got = check_ignore(None, df)
    assert got.dtype == np.dtype(bool) and got.tolist() == [F] * 5


# ---------------------------------------------------------------------------
# check_ignore: the errors and the warning
# ---------------------------------------------------------------------------

_FORMS = (
    r"None or a sequence of booleans with one entry per row of the data: a list, a tuple, a NumPy or pandas "
    r"array, or a Series with the index of the data"
)


@pytest.mark.parametrize(
    "ignore, message",
    [
        ("all", rf"^ignore must be {_FORMS}; got str\.$"),
        (True, r"^ignore must be None or a sequence of booleans .*; got bool\.$"),
        (pd.DataFrame({"x": ROWS}), r"^ignore must be .*; got DataFrame\.$"),
        ({1: True}, r"^ignore must be .*; got dict\.$"),
        (
            ROWS[:-1],
            r"^ignore has 11 entries, the data has 12 rows: ignore must have one entry per row of the data\.$",
        ),
        (np.array(ROWS + [F]), r"^ignore has 13 entries, the data has 12 rows"),
        (np.array([ROWS, ROWS]), r"^ignore has the shape \(2, 12\), the data has 12 rows"),
        (np.array(ROWS).reshape(12, 1), r"^ignore has the shape \(12, 1\), the data has 12 rows"),
        (
            [1, 0] * 6,
            r"^ignore must hold booleans, True or False; it has dtype int64\. Numbers, also 0 and 1, are "
            r"not accepted, as in R: convert them to booleans first\.$",
        ),
        (np.array([1.0, 0.0] * 6), r"^ignore must hold booleans, True or False; it has dtype float64\."),
        (pd.array([1, 0] * 6, dtype="Int64"), r"^ignore must hold booleans, True or False; it has dtype Int64\."),
        (["yes", "no"] * 6, r"^ignore must hold booleans, True or False; it has dtype <U3\."),
        ([T, F] * 5 + [T, None], r"^ignore contains missing values: every entry must be True or False\.$"),
        ([T, F] * 5 + [T, NAN], r"^ignore contains missing values"),
        (pd.array([T, F] * 5 + [T, None], dtype="boolean"), r"^ignore contains missing values"),
    ],
    ids=[
        "string",
        "scalar",
        "frame",
        "dict",
        "short",
        "long",
        "2d",
        "column",
        "numbers",
        "floats",
        "Int64",
        "text",
        "None",
        "nan",
        "pandas-na",
    ],
)
def test_check_ignore_errors(ignore, message):
    with pytest.raises(ValueError, match=message):
        check_ignore(ignore, _twelve())


def test_check_ignore_series_errors():
    df = _twelve()
    with pytest.raises(ValueError, match=r"^ignore has 11 entries, the data has 12 rows"):
        check_ignore(pd.Series(ROWS[:-1], index=df.index[:-1]), df)
    other_index = (
        r"^ignore has another index than the data: a Series must have the index of the data, the same labels in "
        r"the same order\.$"
    )
    with pytest.raises(ValueError, match=other_index):
        check_ignore(pd.Series(ROWS), df)  # 0 to 11
    with pytest.raises(ValueError, match=other_index):
        check_ignore(pd.Series(ROWS, index=df.index[::-1]), df)
    with pytest.raises(ValueError, match=r"^ignore must hold booleans, True or False; it has dtype int64\."):
        check_ignore(pd.Series([0, 1] * 6, index=df.index), df)
    with pytest.raises(ValueError, match=r"^ignore contains missing values"):
        check_ignore(pd.Series([T, F] * 5 + [T, None], index=df.index), df)


@pytest.mark.parametrize(
    "n_rows, ignored, warns",
    [(12, [1, 11], False), (12, [0, 1, 2], True), (10, [], False), (9, [], True)],
    ids=["10-left", "9-left", "10-rows", "9-rows"],
)
def test_check_ignore_warns_with_fewer_than_10_rows_left(n_rows, ignored, warns):
    """R's rule: given an ignore, fewer than 10 rows that are not ignored (also without any row ignored)."""
    df = _twelve().iloc[:n_rows]
    rows = np.isin(np.arange(n_rows), ignored)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        assert check_ignore(rows, df).tolist() == rows.tolist()
    assert [str(w.message) for w in caught] == ([FEW_ROWS] if warns else [])
    assert all(w.category is UserWarning for w in caught)


# ---------------------------------------------------------------------------
# through impute
# ---------------------------------------------------------------------------


def _xy_frame():
    """``x`` complete; ``y = 1 + 2 x + noise``, missing in rows 0, 3, 6 and 9. Rows 3, 12 and 15 are ignored
    (IGNORED): 3 is missing, 12 and 15 hold the decoys 50 and 60."""
    rng = np.random.default_rng(3)
    x = rng.normal(size=20)
    df = pd.DataFrame({"x": x, "y": 1 + 2 * x + rng.normal(size=20)})
    df.loc[[0, 3, 6, 9], "y"] = NAN
    df.loc[[12, 15], "y"] = [50.0, 60.0]
    return df


IGNORED = np.isin(np.arange(20), [3, 12, 15])


def test_impute_keeps_the_ignore_it_used():
    df = _xy_frame()
    mice, _, messages = _run(df, ignore=pd.Series(IGNORED, index=df.index), maxit=0)
    assert isinstance(mice.ignore, np.ndarray) and mice.ignore.dtype == np.dtype(bool)
    np.testing.assert_array_equal(mice.ignore, IGNORED)
    assert messages == []
    mice, _, _ = _run(df, maxit=0)
    np.testing.assert_array_equal(mice.ignore, np.zeros(20, dtype=bool))


def test_impute_checks_ignore_before_it_changes_anything():
    df = _xy_frame()
    mice, out, _ = _run(df, ignore=IGNORED, maxit=1)
    with pytest.raises(ValueError, match=r"^ignore has 19 entries, the data has 20 rows"):
        mice.impute(ignore=IGNORED[:-1])
    with pytest.raises(ValueError, match=r"^ignore must hold booleans"):
        mice.impute(ignore=IGNORED.astype(int))
    # the run before can still be continued, with its ignore
    assert mice.imputed_datasets is out
    np.testing.assert_array_equal(mice.ignore, IGNORED)
    mice.continue_chains()
    assert mice.iteration == 2


def test_impute_warns_with_fewer_than_10_rows_left_and_goes_on():
    """12 rows, 3 ignored: R's warning, and the run as usual."""
    df = _xy_frame().iloc[:12]
    ignore = np.isin(np.arange(12), [3, 4, 5])
    mice, out, messages = _run(df, ignore=ignore, method="mean", maxit=1)
    assert messages == [FEW_ROWS]
    y = df["y"].to_numpy()
    expected = np.nanmean(np.where(ignore, NAN, y))
    for completed in out:
        assert np.allclose(completed.loc[[0, 3, 6, 9], "y"], expected, rtol=1e-12)
    # continue_chains does not check ignore again, as R's mice.mids
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        mice.continue_chains()


def test_mean_imputes_the_mean_of_the_rows_that_are_not_ignored():
    """The ignored missing cell (row 3) is imputed too; the decoys 50 and 60 do not count."""
    df = _xy_frame()
    mice, out, messages = _run(df, ignore=IGNORED, method="mean")
    assert messages == []
    y = df["y"].to_numpy()
    expected = y[~np.isnan(y) & ~IGNORED].mean()
    for completed in out:
        assert completed.loc[[0, 3, 6, 9], "y"].notna().all(), "the missing cell of an ignored row is not imputed"
        assert np.allclose(completed.loc[[0, 3, 6, 9], "y"], expected, rtol=1e-12), (
            "y is not imputed by the mean of the rows that are not ignored"
        )
        pd.testing.assert_series_equal(completed["x"], df["x"])
        assert completed.loc[[12, 15], "y"].tolist() == [50.0, 60.0]
    np.testing.assert_allclose(mice.chain_mean["y"], expected, rtol=1e-12)


def _where_in_ignored_rows():
    """The where-cells of y: its missing cells (0, 3, 6, 9) and the observed cells in rows 1 and 15; 3 and 15
    are ignored. x is imputed by mean in its where-cells, rows 3 (ignored) and 5."""
    df = _xy_frame()
    where = df.isna()
    where.loc[[1, 15], "y"] = True
    where.loc[[3, 5], "x"] = True
    return df, where


def test_ignored_rows_with_where_cells_are_imputed_from_the_fit_on_the_other_rows():
    """norm.predict fitted on the observed rows that are not ignored (not 12, 15) and on the observed x (the
    observed where-cells of x keep their values in the data the chains work on), predicted at every where-cell
    of y, the observed one of the ignored row 15 too. x gets the mean of its rows that are not ignored."""
    df, where = _where_in_ignored_rows()
    mice, out, messages = _run(df, where=where, ignore=IGNORED, method={"x": "mean", "y": "norm.predict"})
    x, y = df["x"].to_numpy(), df["y"].to_numpy()
    fit = ~np.isnan(y) & ~IGNORED
    rows = where["y"].to_numpy()
    expected_y = _fitted(y, x, fit, rows)
    for completed in out:
        assert completed.loc[rows, "y"].notna().all(), "the where-cells of ignored rows are not imputed"
        np.testing.assert_allclose(
            completed.loc[rows, "y"],
            expected_y,
            rtol=1e-10,
            err_msg="y is not fitted on the rows that are not ignored",
        )
        np.testing.assert_allclose(completed.loc[[3, 5], "x"], x[~IGNORED].mean(), rtol=1e-12)
        # cells that are not where-cells keep their values, the decoy of the ignored row 12 too
        assert completed.loc[12, "y"] == 50.0
        pd.testing.assert_frame_equal(completed[~where], df[~where])
    assert messages == []


def test_continue_chains_keeps_the_ignore():
    df, where = _where_in_ignored_rows()
    method = {"x": "mean", "y": "norm.predict"}
    mice, _, _ = _run(df, where=where, ignore=IGNORED, method=method, maxit=1)
    out = mice.continue_chains(2)
    _, longer, _ = _run(df, where=where, ignore=IGNORED, method=method, maxit=3)
    np.testing.assert_array_equal(mice.ignore, IGNORED)
    for a, b in zip(out, longer):
        pd.testing.assert_frame_equal(a, b, check_exact=True)


def test_start_values_come_from_rows_that_are_not_ignored():
    """pmm, maxit=0: every start value of y is an observed value of a row that is not ignored, the start of the
    ignored row 3 too; never a decoy (50, 60)."""
    df = _xy_frame()
    mice, out, _ = _run(df, ignore=IGNORED, method="pmm", maxit=0, n_imputations=20)
    y = df["y"].to_numpy()
    allowed = set(y[~np.isnan(y) & ~IGNORED].tolist())
    starts = [v for completed in out for v in completed.loc[[0, 3, 6, 9], "y"].tolist()]
    assert set(starts) <= allowed, "a start value comes from an ignored row"
    assert len(set(starts)) > 5


def test_meanobs_start_takes_the_most_frequent_label_of_the_rows_that_are_not_ignored():
    """g: "a" is most frequent in all rows (6 times, 4 of them ignored), "b" in the rows that are not ignored."""
    df = pd.DataFrame(
        {
            "g": ["a", "a", "a", "a", "a", "a", "b", "b", "b", "c", None, None] * 2,
            "x": np.arange(24, dtype=float),
        }
    )
    ignore = np.zeros(24, dtype=bool)
    ignore[[0, 1, 2, 3, 12, 13, 14, 15]] = True
    _, out, _ = _run(df, ignore=ignore, method={"g": "sample"}, initial="meanobs", maxit=0)
    for completed in out:
        assert completed.loc[[10, 11, 22, 23], "g"].tolist() == ["b"] * 4
    _, out, _ = _run(df, method={"g": "sample"}, initial="meanobs", maxit=0)
    assert out[0].loc[[10, 11, 22, 23], "g"].tolist() == ["a"] * 4


def _few_fit_rows_frame():
    """y observed in rows 0 to 4 only, rows 3 and 4 ignored (and 18, 19, where y is missing): 3 fit rows for an
    intercept and four complete predictors."""
    i = np.arange(1, 21)
    df = pd.DataFrame({f"q{k}": np.sin((k + 2) * i) for k in range(4)})
    df.insert(0, "y", np.cos(1.7 * i))
    df.loc[5:, "y"] = NAN
    return df, np.isin(np.arange(20), [3, 4, 18, 19])


def test_df_set_to_1_counts_the_fit_rows_without_the_ignored_ones():
    """R's check.df counts sum(ry): 3, not the 5 observed values of y."""
    df, ignore = _few_fit_rows_frame()
    mice, _, _ = _run(df, ignore=ignore, method="mean", maxit=2)
    events = mice.logged_events
    df_set = events.loc[events["out"].str.startswith("df set to 1"), "out"].tolist()
    assert df_set == ["df set to 1. # observed cases: 3  # predictors: 5"] * 2, (
        f"check.df does not count the fit rows: {df_set}"
    )


def _constant_in_fit_rows_frame():
    """y missing in rows 6, 9 and 12; q0, q1 and p complete. p is 1.0 in every row that is not ignored and 2.0,
    3.0 in the ignored rows 18 and 19, where y is observed."""
    i = np.arange(1, 21)
    df = pd.DataFrame({"y": np.cos(1.7 * i), "q0": np.sin(2 * i), "q1": np.sin(3 * i), "p": 1.0})
    df.loc[[6, 9, 12], "y"] = NAN
    df.loc[[18, 19], "p"] = [2.0, 3.0]
    return df, np.isin(np.arange(20), [18, 19])


def test_remove_lindep_works_on_the_fit_rows():
    """p is constant in the rows that are not ignored: dropped from the model of y in every iteration and chain,
    as R's remove.lindep. The setup step uses all rows, where p varies: it keeps p."""
    df, ignore = _constant_in_fit_rows_frame()
    mice, _, _ = _run(df, ignore=ignore, method="mean", maxit=2)
    assert (mice.logged_events["it"] > 0).all()  # nothing in the setup step
    dropped = mice.logged_events[mice.logged_events["out"] == "p"]
    assert list(zip(dropped["it"], dropped["im"], dropped["dep"])) == [
        (1, 1, "y"),
        (1, 2, "y"),
        (2, 1, "y"),
        (2, 2, "y"),
    ], "the predictor that is constant in the fit rows is not dropped"
    _, _, messages = _run(df, method="mean", maxit=2)
    assert messages == []  # without ignore nothing is dropped


# ---------------------------------------------------------------------------
# a column without observed value in the rows that are not ignored
# ---------------------------------------------------------------------------


def _all_observed_ignored_frame():
    """y observed in rows 0 to 5 only, z in rows 0 to 3 only (with two values), c in row 0 only; rows 0 to 5
    are ignored. x is complete."""
    i = np.arange(16)
    df = pd.DataFrame({"x": np.sin(i), "y": np.cos(i), "z": np.cos(2 * i), "c": NAN})
    df.loc[6:, "y"] = NAN
    df.loc[4:, "z"] = NAN
    df.loc[0, "c"] = 1.0
    return df, i < 6


def test_column_without_observed_value_in_the_rows_that_are_not_ignored_raises():
    df, ignore = _all_observed_ignored_frame()
    mice = MICE(df)
    with pytest.raises(
        ValueError,
        match=r"^Every observed value of the columns \['y', 'z'\] lies in a row that ignore leaves out: their "
        r"imputation model has no row to be fitted on and their start no value to draw \(R stops in the first "
        r"iteration with '0 \(non-NA\) cases'\)\. Leave out fewer rows, or give the method '' to these columns\.$",
    ):
        mice.impute(ignore=ignore, seed=1)
    with pytest.raises(
        ValueError,
        match=r"^Every observed value of the column 'y' lies in a row that ignore leaves "
        r"out: its imputation model has no row to be fitted on and its start",
    ):
        mice.impute(ignore=ignore, method={"z": ""}, seed=1)
    with pytest.raises(ValueError, match="no chains to continue"):
        mice.continue_chains()


def test_columns_without_method_need_no_rows_that_are_not_ignored():
    """y and z get method "" from the user, c (one observed value) loses its method in the setup step: none of
    them is checked, and all keep their missing cells."""
    df, ignore = _all_observed_ignored_frame()
    mice, out, messages = _run(df, ignore=ignore, method={"y": "", "z": ""}, maxit=1)
    assert mice.method == {"x": "", "y": "", "z": "", "c": ""}
    assert messages == [
        "Number of logged events: 1. Not imputed because constant or collinear, their missing values are kept: "
        "['c']. See MICE.logged_events."
    ]
    for completed in out:
        pd.testing.assert_frame_equal(completed, df)


# ---------------------------------------------------------------------------
# mutants
# ---------------------------------------------------------------------------
#
# Each applies one defect by patching one function for one test, checks that
# it was applied, and runs a named test of this file, which must fail with
# its own message:
#
# * ignore not applied to the fit rows (_fit_rows);
# * ignore applied to the rows to impute (_rows_to_impute), so that the
#   where-cells of ignored rows are not imputed;
# * the start values drawn from the ignored rows too
#   (MICE._initial_imputation);
# * check_df counting the ignored rows (MICE._check_df);
# * remove_lindep given the rows without ignore (_remove_lindep).


def test_ignore_not_applied_to_the_fit_rows_is_caught(monkeypatch):
    calls = []

    def without_ignore(cx, observed, ignore):
        calls.append(ignore.any())
        return cx & observed

    monkeypatch.setattr(MICE_MODULE, "_fit_rows", without_ignore)
    with pytest.raises(AssertionError, match="y is not imputed by the mean of the rows that are not ignored"):
        test_mean_imputes_the_mean_of_the_rows_that_are_not_ignored()
    with pytest.raises(AssertionError, match="y is not fitted on the rows that are not ignored"):
        test_ignored_rows_with_where_cells_are_imputed_from_the_fit_on_the_other_rows()
    assert any(calls), "mutant not applied: no row ignored"


def test_ignore_applied_to_the_rows_to_impute_is_caught(monkeypatch):
    calls = []
    original = MICE_MODULE._rows_to_impute

    def without_ignored_rows(cx, where):
        calls.append(1)
        return original(cx, where) & ~IGNORED

    monkeypatch.setattr(MICE_MODULE, "_rows_to_impute", without_ignored_rows)
    with pytest.raises(AssertionError, match="the missing cell of an ignored row is not imputed"):
        test_mean_imputes_the_mean_of_the_rows_that_are_not_ignored()
    with pytest.raises(AssertionError, match="the where-cells of ignored rows are not imputed"):
        test_ignored_rows_with_where_cells_are_imputed_from_the_fit_on_the_other_rows()
    assert calls, "mutant not applied"


def test_start_values_from_the_ignored_rows_are_caught(monkeypatch):
    calls = []
    original = MICE._initial_imputation

    def from_every_row(self, rng):
        calls.append(self.ignore.any())
        ignore, self.ignore = self.ignore, np.zeros_like(self.ignore)
        try:
            return original(self, rng)
        finally:
            self.ignore = ignore

    monkeypatch.setattr(MICE, "_initial_imputation", from_every_row)
    with pytest.raises(AssertionError, match="a start value comes from an ignored row"):
        test_start_values_come_from_rows_that_are_not_ignored()
    assert any(calls), "mutant not applied: no row ignored"


def test_check_df_counting_the_ignored_rows_is_caught(monkeypatch):
    calls = []
    original = MICE._check_df
    df, ignore = _few_fit_rows_frame()

    def with_ignored_rows(self, n_predictors, n_observed, col, method_name, chain_idx):
        # the predictors are complete: the ignored rows with y observed are those R's ry leaves out
        extra = int((ignore & df[col].notna().to_numpy()).sum())
        calls.append(extra)
        original(self, n_predictors, n_observed + extra, col, method_name, chain_idx)

    monkeypatch.setattr(MICE, "_check_df", with_ignored_rows)
    with pytest.raises(
        AssertionError, match=r"check.df does not count the fit rows: \['df set to 1\. # observed cases: 5"
    ):
        test_df_set_to_1_counts_the_fit_rows_without_the_ignored_ones()
    assert any(calls), "mutant not applied"


def test_remove_lindep_given_the_rows_without_ignore_is_caught(monkeypatch):
    calls = []
    original = MICE_MODULE._remove_lindep
    df, ignore = _constant_in_fit_rows_frame()

    def with_ignored_rows(x, y, ry):
        # the rows R's ry leaves out because they are ignored (y observed, predictors complete)
        left_out = ignore & df[y.name].notna().to_numpy() & x.notna().all(axis=1).to_numpy()
        calls.append(left_out.any())
        return original(x, y, ry | left_out)

    monkeypatch.setattr(MICE_MODULE, "_remove_lindep", with_ignored_rows)
    with pytest.raises(AssertionError, match="the predictor that is constant in the fit rows is not dropped"):
        test_remove_lindep_works_on_the_fit_rows()
    assert any(calls), "mutant not applied"
