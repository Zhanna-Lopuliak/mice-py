"""Default methods by column type, and the checks of a method against its column.

Without ``method``, ``MICE.impute`` gives each incomplete column the default
of its type, as R's ``make.method`` (``imputation.make_method``): pmm for
numeric columns, logreg for boolean columns and columns with two categories,
polyreg for unordered categorical and text columns with more, polr for
ordered ones with more. A ``method`` dict takes these defaults for the
columns it leaves out; ``default_method`` (R's ``defaultMethod``) changes
them. ``method="pmm"`` and ``default_method=("pmm",) * 4`` give back the
0.2 behaviour.

Before the chains start, a method that cannot impute the type of its column
raises ``ValueError`` with the column named; a categorical method on a
numeric column is allowed with R's warning. A ``ValueError`` raised by an
imputer inside a chain is raised again with the column and the method in
front of its message.
"""

import re
import sys
import warnings

import numpy as np
import pandas as pd
import pytest

from ._helpers import (
    MAXIT,
    N_IMP,
    NUMERIC_MODEL_METHODS,
    TWO_CATEGORY_METHODS,
    assert_all_invariants,
    full_predictor_matrix,
    impute,
    register_imputer,
    typed_frame,
)

TYPED_DEFAULTS = {"num": "pmm", "flag": "logreg", "two": "logreg", "three": "polyreg", "grade": "polr", "x": ""}


@pytest.fixture
def chains_started(monkeypatch):
    """Records each chain that ``impute`` starts (``MICE._impute_once``)."""
    mice_class = sys.modules["imputation.MICE"].MICE
    started = []
    original = mice_class._impute_once

    def recording(self, chain_idx, rng):
        started.append(chain_idx)
        return original(self, chain_idx, rng)

    monkeypatch.setattr(mice_class, "_impute_once", recording)
    return started


def _run(df, **kwargs):
    """``MICE(df).impute`` with two chains and two iterations, warnings recorded; returns (mice, out, warnings)."""
    from imputation import MICE

    mice = MICE(df)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        out = mice.impute(n_imputations=N_IMP, maxit=MAXIT, seed=1, **kwargs)
    return mice, out, [w for w in caught if issubclass(w.category, UserWarning)]


# ---------------------------------------------------------------------------
# make_method and the defaults of impute
# ---------------------------------------------------------------------------


def test_make_method_is_exported():
    import imputation
    import imputation.utils

    assert "make_method" in imputation.__all__
    assert imputation.make_method is imputation.utils.make_method


def test_defaults_on_typed_frame():
    from imputation import make_method

    df = typed_frame()
    before = df.copy(deep=True)
    assert make_method(df) == TYPED_DEFAULTS
    mice, out, _ = _run(df)
    assert mice.method == TYPED_DEFAULTS
    assert mice.visit_sequence == ["num", "flag", "two", "three", "grade"]
    assert_all_invariants(df, before, mice, out, N_IMP, MAXIT)


def _pandas_dtypes_frame(n=80, seed=12):
    """One incomplete column per dtype and number of categories, and two complete columns."""
    rng = np.random.default_rng(seed)
    x = rng.normal(size=n)
    latent = x + rng.normal(scale=0.8, size=n)

    def labels(values, cuts):
        return np.asarray(values, dtype=object)[np.digitize(latent + rng.normal(scale=0.5, size=n), cuts)]

    df = pd.DataFrame(
        {
            "float": 0.5 * latent + rng.normal(size=n),
            "int": pd.array(np.round(2 * latent).astype(int), dtype="Int64"),
            "object2": pd.Series(labels(["no", "yes"], [0.0]), dtype=object),
            "object3": pd.Series(labels(["lo", "mid", "hi"], [-0.5, 0.5]), dtype=object),
            "object_bool": pd.Series(latent + rng.normal(size=n) > 0, dtype=object),
            "string2": pd.Series(labels(["u", "v"], [0.2]), dtype="string"),
            "string3": pd.Series(labels(["a", "b", "c"], [-0.5, 0.5]), dtype="string"),
            "boolean": pd.array(latent + rng.normal(size=n) > 0.3, dtype="boolean"),
            "category2": pd.Categorical(labels(["s", "t"], [0.1])),
            "category3": pd.Categorical(labels(["p", "q", "r"], [-0.4, 0.4]), categories=["r", "q", "p"]),
            "category_unused": pd.Categorical(labels(["k", "m"], [0.0]), categories=["k", "l", "m"]),
            "category_numbers": pd.Categorical(labels([1, 2, 3], [-0.5, 0.5]), categories=[1, 2, 3]),
            "ordered2": pd.Categorical(labels(["lo", "hi"], [0.0]), categories=["lo", "hi"], ordered=True),
            "ordered3": pd.Categorical(labels(["s", "m", "l"], [-0.5, 0.5]), categories=["s", "m", "l"], ordered=True),
            "x": x,
            "bool_complete": x > 0,
        }
    )
    for j, col in enumerate(df.columns[:-2]):
        rows = rng.choice(n, size=8 + j % 3, replace=False)
        df.loc[rows, col] = None
    return df


PANDAS_DTYPES_DEFAULTS = {
    "float": "pmm",
    "int": "pmm",
    "object2": "logreg",
    "object3": "polyreg",
    "object_bool": "logreg",
    "string2": "logreg",
    "string3": "polyreg",
    "boolean": "logreg",
    "category2": "logreg",
    "category3": "polyreg",
    "category_unused": "polyreg",  # three declared categories, as R's nlevels counts
    "category_numbers": "polyreg",
    "ordered2": "logreg",
    "ordered3": "polr",
    "x": "",
    "bool_complete": "",
}


def test_defaults_on_pandas_dtypes():
    from imputation import make_method

    df = _pandas_dtypes_frame()
    assert df["int"].dtype == "Int64" and df["string2"].dtype == "string" and df["boolean"].dtype == "boolean"
    assert df.drop(columns=["x", "bool_complete"]).isna().any().all()
    assert make_method(df) == PANDAS_DTYPES_DEFAULTS
    before = df.copy(deep=True)
    mice, out, _ = _run(df, predictor_matrix=full_predictor_matrix(df))
    assert mice.method == PANDAS_DTYPES_DEFAULTS
    assert_all_invariants(df, before, mice, out, N_IMP, MAXIT)


def test_make_method_counts_declared_categories_and_ignores_complete_columns():
    from imputation import make_method

    df = pd.DataFrame(
        {
            "one_level": pd.Categorical(["a", None, "a"]),
            "two_declared": pd.Categorical(["a", None, "a"], categories=["a", "b"]),
            "four_declared": pd.Categorical(["a", None, "a"], categories=["a", "b", "c", "d"], ordered=True),
            "all_missing_text": pd.Series([None, None, None], dtype=object),
            "only_true": pd.array([True, None, True], dtype="boolean"),
            "complete_text": ["a", "b", "c"],
        }
    )
    assert make_method(df) == {
        "one_level": "pmm",
        "two_declared": "logreg",
        "four_declared": "polr",
        "all_missing_text": "pmm",
        "only_true": "logreg",
        "complete_text": "",
    }
    # anything pandas can turn into a DataFrame
    assert make_method({"a": [1.0, None], "b": ["u", None]}) == {"a": "pmm", "b": "pmm"}


def test_default_method_replaces_the_defaults():
    from imputation import make_method

    df = typed_frame()
    other = ("norm", "logreg.boot", "cart", "polyreg")
    expected = {
        "num": "norm",
        "flag": "logreg.boot",
        "two": "logreg.boot",
        "three": "cart",
        "grade": "polyreg",
        "x": "",
    }
    assert make_method(df, default_method=other) == expected
    assert make_method(df, default_method=list(other)) == expected
    mice, _, _ = _run(df, default_method=other)
    assert mice.method == expected
    # a dict takes them for the columns it leaves out
    mice, _, _ = _run(df, method={"three": "pmm"}, default_method=other)
    assert mice.method == {**expected, "three": "pmm"}


def test_version_0_2_behaviour_is_reachable():
    """``method="pmm"`` for a call without method, ``default_method=("pmm",) * 4`` for a partial dict."""
    df = typed_frame()
    pmm_everywhere = {col: ("pmm" if meth else "") for col, meth in TYPED_DEFAULTS.items()}
    mice, out_string, _ = _run(df, method="pmm")
    assert mice.method == pmm_everywhere
    mice, out_defaults, _ = _run(df, method={}, default_method=("pmm",) * 4)
    assert mice.method == pmm_everywhere
    for a, b in zip(out_string, out_defaults):
        pd.testing.assert_frame_equal(a, b)
    mice, _, _ = _run(df, method={"num": "norm"}, default_method=("pmm",) * 4)
    assert mice.method == {**pmm_everywhere, "num": "norm"}


@pytest.mark.parametrize(
    "default_method, message",
    [
        ("pmm", "list or tuple of four method names"),
        (("pmm", "logreg", "polyreg"), "list or tuple of four method names"),
        (None, "list or tuple of four method names"),
        (("pmm", "logreg", "polyreg", "lda"), r"unsupported methods: \['lda'\]"),
        (("pmm", "", "polyreg", "polr"), r"unsupported methods: \[''\]"),
    ],
)
def test_default_method_is_validated(default_method, message, chains_started):
    from imputation import MICE, make_method

    df = typed_frame()
    with pytest.raises(ValueError, match=message):
        make_method(df, default_method=default_method)
    # also when method is a string, which does not use the defaults
    for method in (None, "pmm"):
        with pytest.raises(ValueError, match=message):
            MICE(df).impute(n_imputations=1, maxit=1, method=method, default_method=default_method)
    assert chains_started == []


# ---------------------------------------------------------------------------
# checks before the chains start
# ---------------------------------------------------------------------------

KIND_OF = {
    "flag": "boolean",
    "two": "categorical or text",
    "three": "categorical or text",
    "grade": "ordered categorical",
}
SUITABLE = {
    "flag": "logreg, logreg.boot, or a donor method (pmm, cart, rf, sample)",
    "two": "logreg, logreg.boot, or a donor method (pmm, cart, rf, sample)",
    "three": "polyreg, or a donor method (pmm, cart, rf, sample)",
    "grade": "polr, polyreg, or a donor method (pmm, cart, rf, sample)",
}


@pytest.mark.parametrize("column", list(KIND_OF))
@pytest.mark.parametrize("method", NUMERIC_MODEL_METHODS)
def test_numeric_method_on_other_column_raises_before_the_chains(method, column, chains_started):
    from imputation import MICE

    df = typed_frame()
    pattern = (
        rf"^Column '{column}' is {KIND_OF[column]} \(dtype .*\), but {re.escape(method)} imputes numeric columns "
        rf"only .* Methods for this column: {re.escape(SUITABLE[column])}\.$"
    )
    with pytest.raises(ValueError, match=pattern):
        MICE(df).impute(n_imputations=1, maxit=1, method={column: method}, seed=1)
    assert chains_started == []


@pytest.mark.parametrize("method", TWO_CATEGORY_METHODS)
@pytest.mark.parametrize(
    "column, shown, use",
    [
        ("three", "'hi', 'lo', 'mid'", "polyreg for more than two categories, or a donor method"),
        ("grade", "'high', 'low', 'mid', 'top'", "polr or polyreg for more than two categories"),
        ("count", "0.0, 1.0, 2.0", "polyreg for more than two categories, or a method for numeric columns such as pmm"),
    ],
)
def test_two_category_method_on_more_categories_raises_before_the_chains(method, column, shown, use, chains_started):
    from imputation import MICE

    df = typed_frame()
    df["count"] = (df["x"] > -0.5).astype(int) + (df["x"] > 0.5)
    df.loc[df.index[:6], "count"] = np.nan
    n = 4 if column == "grade" else 3
    pattern = (
        rf"^Column '{column}': {re.escape(method)} imputes a target with two categories, but the column has {n} "
        rf"distinct observed values \({shown}\)\. Use {re.escape(use)}"
    )
    with pytest.raises(ValueError, match=pattern):
        MICE(df).impute(n_imputations=1, maxit=1, method={column: method}, seed=1)
    assert chains_started == []


@pytest.mark.parametrize("method", TWO_CATEGORY_METHODS)
def test_two_category_method_counts_observed_categories(method):
    """A third declared category that is never observed does not count."""
    df = typed_frame()
    df["two"] = df["two"].cat.add_categories(["maybe"])
    mice, out, _ = _run(df, method={"two": method})
    assert mice.method["two"] == method
    for completed in out:
        assert set(completed["two"]) == {"yes", "no"}


def _names_frame(n=80, n_names=60, seed=4):
    """A text column of names: 60 distinct observed values, ten missing."""
    rng = np.random.default_rng(seed)
    names = [f"person{i:02d}" for i in range(n_names)]
    # every name once, then repeats; ten of the repeats are missing
    name = np.array(names + list(rng.choice(names, size=n - n_names)), dtype=object)
    name[n_names : n_names + 10] = None
    order = rng.permutation(n)
    df = pd.DataFrame({"name": pd.Series(name[order], dtype=object), "x": rng.normal(size=n)})
    assert df["name"].nunique() == n_names and df["name"].isna().sum() == 10
    return df


@pytest.mark.parametrize("method", [None, "polyreg", "polr", "logreg", "logreg.boot"])
def test_more_than_50_categories_raises_with_the_remedy(method, chains_started):
    """A text column of names reaches polyreg with the defaults; the message says why and what to do."""
    from imputation import MICE

    df = _names_frame()
    expected = method or "polyreg"
    pattern = (
        rf"^Column 'name' has 60 distinct observed values, more categories than {re.escape(expected)} can model: "
        r"R's mice allows at most 50 \('Maximum number of categories \(50\) exceeded'\)\. .*"
        r"for example method=\{'name': 'pmm'\}"
    )
    with pytest.raises(ValueError, match=pattern):
        MICE(df).impute(n_imputations=1, maxit=1, method=None if method is None else {"name": method}, seed=1)
    assert chains_started == []


def test_more_than_50_categories_imputed_with_the_remedy():
    df = _names_frame()
    before = df.copy(deep=True)
    mice, out, _ = _run(df, method={"name": "pmm"})
    assert_all_invariants(df, before, mice, out, N_IMP, MAXIT)


@pytest.mark.parametrize("method", ["logreg", "logreg.boot", "polyreg", "polr"])
def test_categorical_method_on_numeric_column_warns_as_r(method):
    df = typed_frame()
    df["k"] = (df["x"] > 0).astype(float)
    df.loc[df.index[:7], "k"] = np.nan
    before = df.copy(deep=True)
    mice, out, caught = _run(df, method={"k": method})
    mismatch = [w for w in caught if "Type mismatch" in str(w.message)]
    assert [str(w.message) for w in mismatch] == [
        f"Type mismatch for variable(s): k\nImputation method {method} is for categorical data."
    ]
    assert mismatch[0].filename == __file__  # points at the call of impute
    assert mice.method["k"] == method
    assert_all_invariants(df, before, mice, out, N_IMP, MAXIT)


def test_type_mismatch_warning_comes_before_the_chains(chains_started):
    from imputation import MICE

    df = typed_frame()
    df["k"] = (df["x"] > 0).astype(float)
    df.loc[df.index[:7], "k"] = np.nan
    with warnings.catch_warnings():
        warnings.filterwarnings("error", message="Type mismatch", category=UserWarning)
        with pytest.raises(UserWarning, match=r"^Type mismatch for variable\(s\): k\n"):
            MICE(df).impute(n_imputations=1, maxit=1, method={"k": "polyreg"}, seed=1)
    assert chains_started == []


def test_categorical_method_on_boolean_column_does_not_warn():
    df = typed_frame()
    mice, _, caught = _run(df, method={"flag": "polyreg", "two": "polr"})
    assert not [w for w in caught if "Type mismatch" in str(w.message)]


def test_check_method_types_skips_donor_methods_and_empty_ones():
    from imputation.validators import check_method_types

    df = typed_frame()
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        check_method_types(df, {"num": "pmm", "flag": "cart", "two": "", "three": "midas", "grade": "sample"})


# ---------------------------------------------------------------------------
# errors from inside a chain name the column
# ---------------------------------------------------------------------------


def test_imputer_value_error_is_raised_again_with_column_and_method(monkeypatch):
    def failing(y, id_obs, x, id_mis=None, rng=None):
        raise ValueError("fake: nothing to fit")

    register_imputer(monkeypatch, "fake", failing)
    df = typed_frame()
    with pytest.raises(ValueError) as excinfo:
        impute(df, {"num": "fake"}, check_log_warning=False)
    message = str(excinfo.value)
    assert message == "Column 'num', method 'fake' (iteration 1, imputation 1): fake: nothing to fit"
    cause = excinfo.value.__cause__
    assert type(cause) is ValueError and str(cause) == "fake: nothing to fit"


def test_imputer_other_errors_pass_unchanged(monkeypatch):
    def failing(y, id_obs, x, id_mis=None, rng=None):
        raise TypeError("fake: wrong type")

    register_imputer(monkeypatch, "fake", failing)
    with pytest.raises(TypeError, match=r"^fake: wrong type$"):
        impute(typed_frame(), {"num": "fake"}, check_log_warning=False)


def test_norm_nob_with_too_few_observed_values_names_the_column():
    """The gap of phase 2: norm.nob cannot say for which column it has too few observed values."""
    rng = np.random.default_rng(8)
    n = 30
    df = pd.DataFrame({f"p{i}": rng.normal(size=n) for i in range(6)})
    df.insert(0, "y", df.sum(axis=1) + rng.normal(size=n))
    df.loc[5:, "y"] = np.nan
    with pytest.raises(ValueError) as excinfo:
        impute(df, {"y": "norm.nob"}, check_log_warning=False)
    message = str(excinfo.value)
    assert message.startswith("Column 'y', method 'norm.nob' (iteration 1, imputation 1): ")
    assert message.endswith(str(excinfo.value.__cause__))
    assert re.search(r"norm\.nob needs more than \d+ observed values", str(excinfo.value.__cause__))


def test_midas_on_categorical_column_names_the_column():
    """midas is checked by its imputer; the error now names the column."""
    with pytest.raises(
        ValueError, match=r"^Column 'three', method 'midas' \(iteration 1, imputation 1\): Target y must"
    ):
        impute(typed_frame(), {"three": "midas"}, check_log_warning=False)
