"""The setup step and the per-call design matrix of ``MICE.impute`` (C6, B2, B7 to B9, C4).

``impute`` edits the imputation model as R's ``mice`` does: before the chains
(``mice.edit.setup``: constant and collinear variables) and around each
univariate call (``sampler.univ``: dummy-coded design matrix, rows with an
incomplete design left out, ``remove.lindep``, ``check.df``). Every change is
a row of ``MICE.logged_events``.

The expected logged events below are what R mice 3.19.0 returns for the same
data (built from closed-form expressions, so R can rebuild it exactly), with
``mice(data, m = 2, maxit = 2, seed = 1)`` and the default (full) predictor
matrix.
"""

import warnings

import numpy as np
import pandas as pd
import pytest

from ._helpers import N_IMP, MAXIT, assert_all_invariants, full_predictor_matrix, impute

LOG_COLUMNS = ["it", "im", "dep", "meth", "out"]


def _run(df, method="pmm", **kwargs):
    """``impute`` with two chains and two iterations; returns (mice, out, warnings)."""
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        mice, out = impute(df, method, n_imputations=2, maxit=2, seed=1, **kwargs)
    messages = [str(w.message) for w in caught if issubclass(w.category, UserWarning)]
    return mice, out, messages


def _events(mice):
    return {col: list(mice.logged_events[col]) for col in LOG_COLUMNS}


# ---------------------------------------------------------------------------
# logged events, compared with R
# ---------------------------------------------------------------------------


def _redundant_columns_frame():
    """``y`` incomplete; ``x3`` is ``x1 + x2`` up to 1e-3 noise, so only the
    eigenvalue step of ``remove_lindep`` catches it; ``const`` is constant;
    ``dup`` copies ``x1`` and has two missing cells."""
    n = 40
    i = np.arange(1, n + 1)
    x1, x2 = np.sin(i), np.cos(1.3 * i)
    y = x1 - x2 + 0.5 * np.sin(3.7 * i)
    y[[1, 8, 16, 24, 32]] = np.nan
    df = pd.DataFrame(
        {"y": y, "x1": x1, "x2": x2, "x3": x1 + x2 + 1e-3 * np.sin(7.1 * i), "const": 3.0, "dup": x1.copy()}
    )
    df.loc[[2, 3], "dup"] = np.nan
    return df


def test_logged_events_constant_duplicated_and_lindep_columns():
    df = _redundant_columns_frame()
    before = df.copy(deep=True)
    mice, out, messages = _run(df)
    assert _events(mice) == {
        "it": [0, 0, 1, 1, 2, 2],
        "im": [0, 0, 1, 2, 1, 2],
        "dep": ["", "", "y", "y", "y", "y"],
        "meth": ["constant", "collinear", "pmm", "pmm", "pmm", "pmm"],
        "out": ["const", "dup", "x3", "x3", "x3", "x3"],
    }
    assert mice.logged_events["it"].dtype == np.int64 and mice.logged_events["im"].dtype == np.int64
    # one warning with the number of events, naming the variable left incomplete
    assert len(messages) == 1
    assert "Number of logged events: 6" in messages[0] and "['dup']" in messages[0]
    # dup is not imputed, keeps its missing cells and is not visited
    assert mice.method == {"y": "pmm", "x1": "", "x2": "", "x3": "", "const": "", "dup": ""}
    assert mice.visit_sequence == ["y"]
    assert (mice.predictor_matrix[["const", "dup"]] == 0).all().all()
    assert (mice.predictor_matrix.loc["dup"] == 0).all()
    for completed in out:
        assert completed["dup"].isna().sum() == 2
    assert_all_invariants(df, before, mice, out, 2, 2, allow_unimputed={"dup"})


def test_logged_events_check_df_and_rank_deficient_design():
    """Five observed values, six predictors: R logs "df set to 1" in the first
    iteration and remove.lindep drops p0 and p4 in every call."""
    n = 40
    i = np.arange(1, n + 1)
    p = np.column_stack([np.sin((k + 2) * i) for k in range(6)])
    df = pd.DataFrame(p, columns=[f"p{k}" for k in range(6)])
    df.insert(0, "y", p.sum(axis=1) + 0.3 * np.cos(5.3 * i))
    df.loc[5:, "y"] = np.nan
    mice, _, messages = _run(df)
    df_message = "df set to 1. # observed cases: 5  # predictors: 7"
    assert _events(mice) == {
        "it": [1, 1, 1, 1, 2, 2],
        "im": [1, 1, 2, 2, 1, 2],
        "dep": ["y"] * 6,
        "meth": ["pmm"] * 6,
        "out": [df_message, "p0, p4", df_message, "p0, p4", "p0, p4", "p0, p4"],
    }
    assert messages == ["Number of logged events: 6. See MICE.logged_events."]


@pytest.mark.parametrize("method", ["pmm", "cart", "rf", "midas", "sample"])
def test_no_events_no_warning(method):
    from ._helpers import numeric_frame

    mice, _, messages = _run(numeric_frame(n=40), method)
    assert list(mice.logged_events.columns) == LOG_COLUMNS
    assert mice.logged_events.empty
    assert not [m for m in messages if "logged events" in m]


def test_logged_events_are_reset_by_each_impute():
    from imputation import MICE

    from ._helpers import numeric_frame

    mice = MICE(_redundant_columns_frame())
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        mice.impute(n_imputations=1, maxit=1, method="pmm", seed=1)
        first = len(mice.logged_events)
        mice.impute(n_imputations=1, maxit=1, method="pmm", seed=1)
    assert len(mice.logged_events) == first
    assert MICE(numeric_frame(n=20)).logged_events.empty  # exists before impute


def test_entirely_missing_column_is_kept_not_imputed_and_logged():
    """As in R: an entirely missing column is constant (no variance), so it is
    neither imputed nor used as a predictor."""
    from ._helpers import regression_frame

    df = regression_frame()
    df["z"] = np.nan
    before = df.copy(deep=True)
    mice, out, messages = _run(df)
    assert list(mice.data.columns) == ["y", "x1", "x2", "z"]
    assert mice.method["z"] == ""
    assert "z" not in mice.visit_sequence
    assert _events(mice) == {"it": [0], "im": [0], "dep": [""], "meth": ["constant"], "out": ["z"]}
    assert "['z']" in messages[0]
    for completed in out:
        assert completed["z"].isna().all()
    assert_all_invariants(df, before, mice, out, 2, 2, allow_unimputed={"z"})


# ---------------------------------------------------------------------------
# a predictor that is not imputed
# ---------------------------------------------------------------------------


def _empty_method_predictor_frame():
    rng = np.random.default_rng(31)
    n = 60
    x = rng.normal(size=n)
    z = x + rng.normal(scale=0.5, size=n)
    y = x - z + rng.normal(scale=0.5, size=n)
    df = pd.DataFrame({"y": y, "x": x, "z": z})
    df.loc[[0, 5, 10, 15, 20, 25], "y"] = np.nan
    df.loc[[0, 5, 10, 30, 35, 40, 45], "z"] = np.nan  # rows 0, 5, 10: both missing
    return df


@pytest.mark.parametrize("method", ["pmm", "cart", "rf", "midas", "sample"])
def test_predictor_with_empty_method_leaves_its_missing_rows_untouched(method, monkeypatch):
    """``z`` has method "" but predicts ``y``. As in R's sampler.univ, a row
    whose design has a missing value is left out of the model (also as a
    donor) and its ``y`` is not imputed."""
    import imputation.utils as utils

    seen = []
    original = utils._IMPUTER_MAP[method]

    def recording(y, id_obs, x, id_mis=None, **kwargs):
        seen.append((len(y), int(np.sum(id_obs)), int(np.sum(id_mis)), np.isnan(np.asarray(x, float)).any()))
        return original(y=y, id_obs=id_obs, x=x, id_mis=id_mis, **kwargs)

    from imputation import MICE

    monkeypatch.setitem(utils._IMPUTER_MAP, method, recording)
    df = _empty_method_predictor_frame()
    before = df.copy(deep=True)
    mice = MICE(df)
    out = mice.impute(
        n_imputations=2,
        maxit=2,
        method={"y": method, "z": ""},
        predictor_matrix=full_predictor_matrix(df),
        seed=1,
    )
    pd.testing.assert_frame_equal(df, before)
    both = df["y"].isna() & df["z"].isna()
    assert both.sum() == 3
    assert mice.visit_sequence == ["y"]
    # the imputer saw only the 53 rows with z observed: 50 donors, 3 cells to impute
    assert seen and all(call == (53, 50, 3, False) for call in seen)
    for completed in out:
        assert completed.loc[both, "y"].isna().all()
        assert completed.loc[df["y"].isna() & ~both, "y"].notna().all()
        assert completed["z"].isna().sum() == df["z"].isna().sum()
        imputed = completed.loc[df["y"].isna() & ~both, "y"]
        assert imputed.isin(df["y"].dropna()).all()
        pd.testing.assert_frame_equal(completed[df.notna()], df[df.notna()])


# ---------------------------------------------------------------------------
# mixed frames: dummy-coded predictors, targets keep their type (B2, B8, B9)
# ---------------------------------------------------------------------------


def _mixed_types_frame(n=80, seed=17):
    """Numeric, object, ordered Categorical and boolean columns, all incomplete."""
    rng = np.random.default_rng(seed)
    a = rng.normal(size=n)
    g = np.where(a + rng.normal(scale=0.8, size=n) > 0.4, "hi", np.where(a < -0.4, "lo", "mid"))
    size = pd.Categorical(rng.choice(["S", "M", "L"], size=n), categories=["S", "M", "L"], ordered=True)
    flag = a + rng.normal(scale=0.8, size=n) > 0
    b = 0.5 * a + (g == "hi") - flag + rng.normal(scale=0.7, size=n)
    df = pd.DataFrame(
        {
            "a": a,
            "b": b,
            "g": pd.Series(g, dtype=object),
            "size": size,
            "flag": pd.Series(flag, dtype=object),
            "k": rng.integers(0, 5, size=n),
        }
    )
    for col, rows in {
        "a": [1, 9, 33],
        "b": [4, 12, 40, 61],
        "g": [2, 20, 50],
        "size": [7, 30, 70],
        "flag": [3, 15, 44, 77],
    }.items():
        df.loc[rows, col] = np.nan
    return df


@pytest.mark.parametrize("method", ["pmm", "cart", "rf"])
def test_mixed_frame_is_imputed(method):
    df = _mixed_types_frame()
    before = df.copy(deep=True)
    mice, out, messages = _run(df, method)
    assert_all_invariants(df, before, mice, out, 2, 2)
    assert mice.logged_events.empty and not [m for m in messages if "logged events" in m]
    for completed in out:
        assert isinstance(completed["size"].dtype, pd.CategoricalDtype) and completed["size"].cat.ordered
        assert all(isinstance(v, (bool, np.bool_)) for v in completed["flag"])
        assert set(completed["g"]) <= {"hi", "lo", "mid"}


@pytest.mark.parametrize("method", ["pmm", "cart", "rf", "sample"])
def test_categorical_target_reaches_the_imputer_as_categorical(method, monkeypatch):
    """A Categorical with numeric categories stays categorical on its way to
    the imputer, and every predictor arrives as a numeric design matrix."""
    import imputation.utils as utils

    seen = []
    original = utils._IMPUTER_MAP[method]

    def recording(y, id_obs, x, id_mis=None, **kwargs):
        seen.append((getattr(y, "dtype", None), np.asarray(x).dtype, np.asarray(x).shape[1]))
        return original(y=y, id_obs=id_obs, x=x, id_mis=id_mis, **kwargs)

    monkeypatch.setitem(utils._IMPUTER_MAP, method, recording)
    rng = np.random.default_rng(3)
    n = 60
    x = rng.normal(size=n)
    level = pd.Categorical(np.where(x > 0.5, 3, np.where(x < -0.5, 1, 2)), categories=[1, 2, 3])
    df = pd.DataFrame({"level": level, "x": x, "g": pd.Series(rng.choice(["u", "v", "w"], n), dtype="string")})
    df.loc[[0, 7, 19, 33, 48], "level"] = np.nan
    before = df.copy(deep=True)
    mice, out, _ = _run(df, method)
    assert_all_invariants(df, before, mice, out, 2, 2)
    assert len(seen) == 4
    for dtype, x_dtype, n_columns in seen:
        assert isinstance(dtype, pd.CategoricalDtype)
        assert x_dtype == np.float64 and n_columns == 3  # x, gv, gw


# ---------------------------------------------------------------------------
# direct calls of the imputers
# ---------------------------------------------------------------------------


def _direct_problem():
    rng = np.random.default_rng(9)
    n = 50
    x = pd.DataFrame(
        {
            "num": rng.normal(size=n),
            "g": pd.Series(rng.choice(["b", "a", "c"], n), dtype=object),
            "s": pd.Series(rng.choice(["u", "v"], n), dtype="string"),
            "size": pd.Categorical(rng.choice(["S", "M", "L"], n), categories=["S", "M", "L"], ordered=True),
            "flag": rng.random(n) > 0.5,
        }
    )
    y = x["num"] + (x["g"] == "c") + rng.normal(scale=0.5, size=n)
    y = y.to_numpy()
    ry = np.ones(n, dtype=bool)
    ry[::7] = False
    y[~ry] = np.nan
    return y, ry, x


@pytest.mark.parametrize("method", ["pmm", "midas", "cart", "rf"])
def test_direct_call_with_categorical_dataframe_equals_design_matrix(method):
    """A DataFrame with non-numeric predictors goes through obtain_design, so a
    direct call gives what a call with the design matrix gives."""
    from imputation.design import obtain_design
    from imputation.utils import get_imputer_func

    imputer = get_imputer_func(method)
    y, ry, x = _direct_problem()
    design = obtain_design(x, list(x.columns))
    assert list(design.columns) == ["num", "gb", "gc", "sv", "size.L", "size.Q", "flagTRUE"]
    from_frame = imputer(y, ry, x, rng=np.random.default_rng(4))
    from_design = imputer(y, ry, design.to_numpy(), rng=np.random.default_rng(4))
    np.testing.assert_array_equal(from_frame, from_design)
    assert np.isin(from_frame, y[ry]).all()


@pytest.mark.parametrize("method", ["cart", "rf"])
def test_tree_methods_without_predictors_draw_observed_values(method):
    """B7: rf without predictors behaves like cart: a draw from the observed values."""
    from imputation.utils import get_imputer_func

    imputer = get_imputer_func(method)
    rng = np.random.default_rng(0)
    y = rng.normal(size=40)
    ry = np.ones(40, dtype=bool)
    ry[::5] = False
    for x in (np.empty((40, 0)), pd.DataFrame(index=range(40))):
        out = imputer(y, ry, x, rng=np.random.default_rng(2))
        assert out.shape == (8,)
        assert np.isin(out, y[ry]).all()
    draws = np.concatenate([imputer(y, ry, np.empty((40, 0)), rng=np.random.default_rng(s)) for s in range(40)])
    assert len(np.unique(draws)) > 16  # not a single donor


@pytest.mark.parametrize("method", ["cart", "rf"])
@pytest.mark.parametrize("kind", ["object", "numpy"])
def test_tree_methods_impute_booleans_with_booleans(method, kind):
    """B8: an incomplete boolean target is classified and imputed with booleans."""
    from imputation.utils import get_imputer_func

    imputer = get_imputer_func(method)
    rng = np.random.default_rng(1)
    n = 40
    x = rng.normal(size=(n, 2))
    flag = x[:, 0] + rng.normal(scale=0.5, size=n) > 0
    ry = np.ones(n, dtype=bool)
    ry[[2, 6, 11, 30]] = False
    y = pd.Series(flag, dtype=object).where(ry, np.nan).to_numpy() if kind == "object" else flag
    out = imputer(y, ry, x, rng=np.random.default_rng(0))
    assert len(out) == 4
    assert all(isinstance(v, (bool, np.bool_)) for v in out), list(out)


@pytest.mark.parametrize("method", ["cart", "rf"])
def test_tree_methods_accept_string_dtype_predictors_directly(method):
    """B9: a pandas ``string`` predictor is dummy-coded, not cast to float."""
    from imputation.utils import get_imputer_func

    imputer = get_imputer_func(method)
    rng = np.random.default_rng(6)
    n = 40
    s = pd.Series(rng.choice(["u", "v", "w"], n), dtype="string")
    y = (s == "w").to_numpy() * 2.0 + rng.normal(size=n)
    ry = np.ones(n, dtype=bool)
    ry[::8] = False
    out = imputer(y, ry, pd.DataFrame({"s": s, "x": rng.normal(size=n)}), rng=np.random.default_rng(0))
    assert len(out) == 5 and np.isin(out, y[ry]).all()


# ---------------------------------------------------------------------------
# input handling (C4)
# ---------------------------------------------------------------------------


def test_unsupported_dtype_raises_type_error_naming_column_and_dtype():
    from imputation import MICE

    df = pd.DataFrame({"a": [1.0, np.nan, 3.0], "when": pd.date_range("2024-01-01", periods=3)})
    with pytest.raises(TypeError, match=r"'when' has dtype datetime64\[ns\]"):
        MICE(df)


def test_all_empty_row_warns_and_nothing_is_printed(capsys):
    from imputation import MICE

    df = _redundant_columns_frame()
    df.iloc[6, :] = np.nan
    with pytest.warns(UserWarning, match="1 row"):
        mice = MICE(df)
    assert mice.data.shape == df.shape
    assert capsys.readouterr().out == ""


def test_complete_rows_in_predictor_matrix_do_not_warn():
    """Rows of complete variables are allowed and unused, as in R."""
    from ._helpers import regression_frame

    df = regression_frame()
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        mice, out = impute(df, "pmm", n_imputations=N_IMP, maxit=MAXIT)
    assert mice.method == {"y": "pmm", "x1": "", "x2": ""}
    assert len(out) == N_IMP


# ---------------------------------------------------------------------------
# defaults and visit sequences (A2, C3, A7)
# ---------------------------------------------------------------------------


def test_defaults_follow_r():
    """Without arguments: pmm for every incomplete column (R uses pmm for
    numeric columns; the port has no logreg / polyreg, so categorical columns
    get pmm too), five iterations, data order and the full predictor matrix."""
    from imputation import MICE

    from ._helpers import mixed_frame

    df = mixed_frame()
    mice = MICE(df)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        out = mice.impute(n_imputations=1, seed=1)
    assert mice.method == {"a": "pmm", "b": "pmm", "g": "pmm", "h": "pmm", "d": ""}
    assert mice.maxit == 5
    assert mice.chain_mean["a"].shape == (5, 1)
    assert mice.visit_sequence == ["a", "b", "g", "h"]
    pd.testing.assert_frame_equal(mice.predictor_matrix, full_predictor_matrix(df), check_dtype=False)
    assert not out[0].isna().any().any()


def test_method_dict_fills_unlisted_columns_with_pmm():
    from ._helpers import numeric_frame

    df = numeric_frame(n=40)
    mice, _ = impute(df, {"a": "sample"}, n_imputations=1, maxit=1)
    assert mice.method == {"a": "sample", "b": "pmm", "c": "pmm", "d": "", "k": ""}


def _visit_frame():
    """Missing counts: a 3, b 0, c 1, d 3, e 2, f 2 (f gets method "")."""
    rng = np.random.default_rng(5)
    df = pd.DataFrame(rng.normal(size=(30, 6)), columns=list("abcdef"))
    for col, rows in {"a": [0, 7, 14], "c": [3], "d": [2, 9, 20], "e": [5, 25], "f": [1, 11]}.items():
        df.loc[rows, col] = np.nan
    return df


@pytest.mark.parametrize(
    "code, expected",
    [
        ("roman", ["a", "c", "d", "e"]),
        ("arabic", ["e", "d", "c", "a"]),
        # stable sort by number of missing values: the tie a, d keeps data order
        ("monotone", ["c", "e", "a", "d"]),
        ("revmonotone", ["d", "a", "e", "c"]),
    ],
)
def test_visit_sequence_keywords(code, expected):
    """R's keywords; complete columns and columns with method "" are not visited."""
    df = _visit_frame()
    method = {c: "pmm" for c in df.columns} | {"f": ""}
    mice, _ = impute(df, method, n_imputations=1, maxit=1, visit_sequence=code, predictor_matrix=None)
    assert mice.visit_sequence == expected


def test_default_visit_sequence_is_roman():
    from imputation import MICE

    df = _visit_frame()
    mice = MICE(df)
    mice.impute(n_imputations=1, maxit=1, method={"f": ""}, seed=2)
    assert mice.visit_sequence == ["a", "c", "d", "e"]


def test_random_visit_sequence_is_a_permutation_of_the_imputed_columns():
    df = _visit_frame()
    method = {c: "pmm" for c in df.columns} | {"f": ""}
    orders = set()
    for seed in range(8):
        mice, _ = impute(df, method, n_imputations=1, maxit=1, visit_sequence="random", seed=seed)
        assert sorted(mice.visit_sequence) == ["a", "c", "d", "e"]
        assert all(type(col) is str for col in mice.visit_sequence)
        orders.add(tuple(mice.visit_sequence))
    assert len(orders) > 1


def test_unknown_visit_sequence_keyword_lists_the_supported_ones():
    df = _visit_frame()
    with pytest.raises(ValueError, match="revmonotone"):
        impute(df, "pmm", n_imputations=1, maxit=1, visit_sequence="backwards")
