"""L1 parity: the setup step of mice() and the design helpers.

Reference: tests/reference/r/l1_setup.R (mice(data, maxit = 0),
mice:::obtain.design, mice:::remove.lindep, mice:::find.collinear).

R's visitSequence may list variables it never imputes (complete variables
for "monotone", "revmonotone" and the default; variables whose method
mice.edit.setup set to ""). The sampler skips those, so the fixture stores
the effective order: R's sequence restricted to variables with a non-empty
method. The tests compare the Python sequence restricted to the same
variables, and check separately that Python visits no complete variable.

Methods are compared for every column: ``MICE.impute`` without ``method``
must give R's ``imp$method`` (the default of each column's type, then the
setup step), and ``imputation.make_method`` R's ``make.method`` where the
setup step changed nothing (nhanes2: logreg for the factor ``hyp``; boys:
polr for the ordered ``gen`` and ``phb``, polyreg for ``reg``).

Under ``where`` (``where_cases``): ``make_method(data, where=)`` and the
methods, predictor matrix and visit sequences of ``impute(maxit=0, where=)``
against R's ``make.method(data, where = )`` and ``mice(data, maxit = 0,
where = )``, for R's keywords and a matrix whose where-counts differ from
the missing counts. ``monotone_cases``: the predictor matrix that R's
``mice.edit.predictorMatrix`` leaves for ``visitSequence = "monotone"`` with
``maxit = 1`` (each column predicted only by the columns before it), and
leaves alone otherwise; one such iteration uses no start value, so its
completed data are compared too. test_l1_chain_mutants runs these
comparisons against a defective library.

Passive columns (``passive_cases``): ``impute(maxit=0, method=, passive=,
where=)`` against R's ``mice(data, maxit = 0, method, where)`` with the
passive formulas in ``method`` (built from the same recipes, see
``_helpers.passive_function``): the methods after the setup step, where
MICE shows ``"passive"`` for R's formula and ``""`` for a passive copy of a
column (collinear) and a complete passive column; the predictor matrix; the
logged events; the visit sequence by default (the passive columns last),
for each keyword and for a list that names a column twice (R's sequence
restricted to the columns R imputes, as MICE lists only those).
"""

import warnings

import numpy as np
import pandas as pd
import pytest

from imputation import MICE, make_method

from .._logged_events import impute_checking_log_warning
from ._helpers import as_array, as_bool, decode_frame, load_fixture, passive_function, str_list

pytestmark = pytest.mark.parity

SETUP = load_fixture("l1_setup")
DESIGN = load_fixture("l1_design")
VISITS = ["roman", "arabic", "monotone", "revmonotone"]


def _impute(case, **kwargs):
    """One imputation, one iteration; seeded, so that the chain is the same in every run."""
    df = decode_frame(case["data"])
    mice = MICE(df)
    kwargs.setdefault("seed", 20261112)
    impute_checking_log_warning(mice, n_imputations=1, maxit=1, **kwargs)
    return df, mice


def _r_matrix(payload):
    return pd.DataFrame(
        np.array(payload["values"], dtype=int),
        index=str_list(payload["rows"]),
        columns=str_list(payload["cols"]),
    )


# ---------------------------------------------------------------------------
# method
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("case", SETUP["cases"], ids=[c["name"] for c in SETUP["cases"]])
def test_default_methods_match_r(case):
    """Every column, after the setup step (which removes the method of a
    constant or collinear column, as in R)."""
    _, mice = _impute(case)
    assert {str(col): meth for col, meth in mice.method.items()} == case["method"]


# R's imp$method is make.method's result where the setup step logged nothing
UNEDITED = [c for c in SETUP["cases"] if not c["logged_events"]]


@pytest.mark.parametrize("case", UNEDITED, ids=[c["name"] for c in UNEDITED])
def test_make_method_matches_r(case):
    df = decode_frame(case["data"])
    assert {str(col): meth for col, meth in make_method(df).items()} == case["method"]


def test_cases_cover_every_default_method():
    methods = {meth for case in UNEDITED for meth in case["method"].values()}
    assert methods == {"", "pmm", "logreg", "polyreg", "polr"}


def test_boys_is_imputed_with_the_default_methods():
    """Smoke run on R's boys data: polr for gen and phb, polyreg for reg.

    No numerical warning may reach the user: polr's line search tries steps
    at which a cut point overflows, and rejects them silently, as R does.
    """
    case = next(c for c in SETUP["cases"] if c["name"] == "boys")
    df = decode_frame(case["data"])
    mice = MICE(df)
    with warnings.catch_warnings():
        warnings.simplefilter("error", RuntimeWarning)
        out = impute_checking_log_warning(mice, n_imputations=1, maxit=2, seed=2)
    assert {str(col): meth for col, meth in mice.method.items()} == case["method"]
    completed = out[0]
    assert not completed.isna().any().any()
    for col in ("gen", "phb", "reg"):
        assert completed[col].dtype == df[col].dtype  # same categories, same order
        assert set(completed[col]) <= set(df[col].cat.categories)


# ---------------------------------------------------------------------------
# predictor matrix
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("case", SETUP["cases"], ids=[c["name"] for c in SETUP["cases"]])
def test_default_predictor_matrix_matches_r(case):
    _, mice = _impute(case)
    expected = _r_matrix(case["predictor_matrix"])
    rows = str_list(case["imputed_variables"])
    got = mice.predictor_matrix.loc[rows, expected.columns].astype(int)
    pd.testing.assert_frame_equal(got, expected.loc[rows], check_names=False)


# ---------------------------------------------------------------------------
# which variables are imputed, setup events (mice.edit.setup)
# ---------------------------------------------------------------------------

LOG_COLUMNS = ("it", "im", "dep", "meth", "out")


def _impute_full(case):
    """``_impute`` with R's default predictor matrix (all ones, zero diagonal),
    passed explicitly so that these checks of the setup step do not depend on
    the default (which is the same matrix)."""
    df = decode_frame(case["data"])
    full = pd.DataFrame(1, index=df.columns, columns=df.columns)
    np.fill_diagonal(full.values, 0)
    return _impute(case, predictor_matrix=full)


@pytest.mark.parametrize("case", SETUP["cases"], ids=[c["name"] for c in SETUP["cases"]])
def test_imputed_variables_match_r(case):
    """The variables with a method: incomplete ones, minus those the setup step
    found constant or collinear."""
    _, mice = _impute_full(case)
    imputed = [str(col) for col, meth in mice.method.items() if meth != ""]
    assert imputed == str_list(case["imputed_variables"])


@pytest.mark.parametrize("case", SETUP["cases"], ids=[c["name"] for c in SETUP["cases"]])
def test_setup_logged_events_match_r(case):
    """R ran maxit = 0, so its log holds the setup events only (it = 0)."""
    _, mice = _impute_full(case)
    events = mice.logged_events
    assert list(events.columns) == list(LOG_COLUMNS)
    events = events[events["it"] == 0]
    got = {
        "it": [int(v) for v in events["it"]],
        "im": [int(v) for v in events["im"]],
        "dep": [str(v) for v in events["dep"]],
        "meth": [str(v) for v in events["meth"]],
        "out": [str(v) for v in events["out"]],
    }
    expected = case["logged_events"] or {name: [] for name in LOG_COLUMNS}
    expected = {
        name: [int(v) for v in expected[name]] if name in ("it", "im") else str_list(expected[name])
        for name in LOG_COLUMNS
    }
    assert got == expected


@pytest.mark.parametrize("case", SETUP["cases"], ids=[c["name"] for c in SETUP["cases"]])
def test_incomplete_after_imputation_matches_r(case):
    """R leaves a variable whose method the setup step set to "" incomplete."""
    _, mice = _impute_full(case)
    completed = mice.imputed_datasets[0]
    still_missing = [str(col) for col in completed.columns if completed[col].isna().any()]
    assert still_missing == str_list(case["incomplete_after_imputation"])


# ---------------------------------------------------------------------------
# visit sequence
# ---------------------------------------------------------------------------


def _check_visit_sequence(case, df, visit_sequence, expected):
    visit_sequence = [str(v) for v in visit_sequence]
    incomplete = [col for col in df.columns if df[col].isna().any()]
    assert set(visit_sequence) <= set(incomplete), "visits a complete variable"
    imputed = set(str_list(case["imputed_variables"]))
    assert [v for v in visit_sequence if v in imputed] == str_list(expected)


@pytest.mark.parametrize("case", SETUP["cases"], ids=[c["name"] for c in SETUP["cases"]])
def test_default_visit_sequence_matches_r(case):
    df, mice = _impute(case)
    _check_visit_sequence(case, df, mice.visit_sequence, case["effective_visit_sequence_default"])


def _visit_params():
    for case in SETUP["cases"]:
        for code in VISITS:
            yield pytest.param(case, code, id=f"{case['name']}-{code}")


@pytest.mark.parametrize("case, code", list(_visit_params()))
def test_visit_sequence_matches_r(case, code):
    df, mice = _impute(case, visit_sequence=code)
    _check_visit_sequence(case, df, mice.visit_sequence, case["effective_visit_sequence"][code])


# ---------------------------------------------------------------------------
# where: make_method, methods, predictor matrix, visit sequence
# ---------------------------------------------------------------------------

WHERE = SETUP["where_cases"]
MONOTONE = SETUP["monotone_cases"]
RTOL = 1e-8
ATOL = 1e-10


def _where(case, df):
    """The ``where`` of the case for `impute`: R's keyword, or its matrix as a frame with the index of `df`."""
    where = case["where"]
    if where is None or isinstance(where, str):
        return where
    return pd.DataFrame(np.array(where["values"], dtype=bool), index=df.index, columns=str_list(where["cols"]))


def _impute_where(case, **kwargs):
    """``impute(maxit=0)`` with the ``where`` and ``method`` of the case, as R's ``mice(maxit = 0)``."""
    df = decode_frame(case["data"])
    mice = MICE(df)
    if case["method_arg"] is not None:
        kwargs["method"] = {col: str(meth) for col, meth in case["method_arg"].items()}
    impute_checking_log_warning(mice, n_imputations=1, maxit=0, seed=20261112, where=_where(case, df), **kwargs)
    return mice


def check_where_methods(case):
    """``make_method(data, where=)`` and the methods of ``impute(where=)``: R's ``make.method`` and ``imp$method``."""
    df = decode_frame(case["data"])
    made = {str(col): meth for col, meth in make_method(df, where=_where(case, df)).items()}
    assert made == case["make_method"], f"{case['name']}: make_method gives {made}, R's {case['make_method']}"
    method = {str(col): meth for col, meth in _impute_where(case).method.items()}
    assert method == case["method"], f"{case['name']}: methods {method}, R's {case['method']}"


def check_where_visit_sequences(case):
    """The visit sequence by default and for each keyword: R's, restricted to the variables R imputes."""
    for code in [None, *VISITS]:
        if code is None:
            mice, expected = _impute_where(case), case["effective_visit_sequence_default"]
        else:
            mice, expected = _impute_where(case, visit_sequence=code), case["effective_visit_sequence"][code]
        visits = [str(col) for col in mice.visit_sequence]
        assert visits == str_list(expected), (
            f"{case['name']}: visit sequence {code or 'default'} {visits}, R's {str_list(expected)}"
        )


def _check_predictor_matrix(case, mice):
    """The whole predictor matrix: R's."""
    expected = _r_matrix(case["predictor_matrix"])
    got = mice.predictor_matrix.loc[expected.index, expected.columns].astype(int)
    rows = [row for row in expected.index if not (got.loc[row] == expected.loc[row]).all()]
    assert rows == [], f"{case['name']}: the predictor matrix differs from R's in the rows {rows}"


@pytest.mark.parametrize("case", WHERE, ids=[c["name"] for c in WHERE])
def test_where_methods_match_r(case):
    check_where_methods(case)


@pytest.mark.parametrize("case", WHERE, ids=[c["name"] for c in WHERE])
def test_where_predictor_matrix_matches_r(case):
    _check_predictor_matrix(case, _impute_where(case))


@pytest.mark.parametrize("case", WHERE, ids=[c["name"] for c in WHERE])
def test_where_visit_sequences_match_r(case):
    check_where_visit_sequences(case)


def test_where_cases_cover_the_keywords_and_the_counts():
    keywords = {case["where"] for case in WHERE if isinstance(case["where"], str)}
    assert keywords == {"missing", "all", "observed", "none"}
    cases = [case for case in WHERE if not isinstance(case["where"], str)]
    for case in cases:
        nmis, nimp = case["nmis"], case["nimp"]
        assert any(nmis[col] == 0 and nimp[col] > 0 for col in nimp), "no complete column with where-cells"
        assert any(nmis[col] > 0 and nimp[col] == 0 for col in nimp), "no incomplete column without where-cells"
        # R's monotone order by where-cells, not by missing values
        imputed = str_list(case["imputed_variables"])
        by_missing = sorted(imputed, key=lambda col: nmis[col])
        assert str_list(case["effective_visit_sequence"]["monotone"]) != by_missing
    # the user names a method for the column without where-cells: R gives it ""
    assert any(
        case["method_arg"] is not None
        and any(case["method_arg"][col] and case["nimp"][col] == 0 for col in case["method_arg"])
        for case in cases
    )


# ---------------------------------------------------------------------------
# the monotone edit of the predictor matrix (R's mice.edit.predictorMatrix)
# ---------------------------------------------------------------------------


def check_monotone_case(case):
    """``impute(n_imputations=1, maxit, visit_sequence, method="norm.predict", where)``: R's predictor matrix and
    visit sequence; for one iteration of "monotone" R's completed data."""
    df = decode_frame(case["data"])
    mice = MICE(df)
    impute_checking_log_warning(
        mice,
        n_imputations=1,
        maxit=case["maxit"],
        seed=20261112,
        method="norm.predict",
        visit_sequence=case["visit_sequence_arg"],
        where=_where(case, df),
    )
    _check_predictor_matrix(case, mice)
    visits = [str(col) for col in mice.visit_sequence]
    r_visits = str_list(case["effective_visit_sequence"])
    assert visits == r_visits, f"{case['name']}: visit sequence {visits}, R's {r_visits}"
    if case["completed"] is None:
        return
    r_frame = decode_frame(case["completed"])
    for col in r_frame.columns:
        np.testing.assert_allclose(
            mice.imputed_datasets[0][col].to_numpy(dtype=float, na_value=np.nan),
            r_frame[col].to_numpy(dtype=float),
            rtol=RTOL,
            atol=ATOL,
            err_msg=f"{case['name']}: completed data, {col!r}",
        )


@pytest.mark.parametrize("case", MONOTONE, ids=[c["name"] for c in MONOTONE])
def test_monotone_edit_matches_r(case):
    check_monotone_case(case)


def test_monotone_cases_cover_the_edit_and_its_absence():
    runs = {(case["maxit"], case["visit_sequence_arg"], case["where"] is not None) for case in MONOTONE}
    assert {(1, "monotone", False), (1, "monotone", True), (2, "monotone", False), (1, "revmonotone", False)} <= runs
    for case in MONOTONE:
        # R's matrix is the default (all ones off the diagonal) unless edited
        matrix = np.array(case["predictor_matrix"]["values"], dtype=int)
        unedited = (matrix == 1 - np.eye(len(matrix), dtype=int)).all()
        assert unedited == (case["completed"] is None), case["name"]


# ---------------------------------------------------------------------------
# passive columns: methods, predictor matrix, logged events, visit sequence
# ---------------------------------------------------------------------------

PASSIVE = SETUP["passive_cases"]


def _mice_method(r_method):
    """MICE's method for R's: ``"passive"`` for a passive formula ("~ ...")."""
    return "passive" if r_method.startswith("~") else r_method


def _impute_passive(case, **kwargs):
    """``impute(maxit=0)`` with the methods, passive columns and ``where`` of the case, as R's ``mice(maxit = 0)``."""
    df = decode_frame(case["data"])
    mice = MICE(df)
    impute_checking_log_warning(
        mice,
        n_imputations=1,
        maxit=0,
        seed=20261112,
        method={col: str(meth) for col, meth in case["method_arg"].items()},
        passive={col: passive_function(recipe) for col, recipe in case["passive"].items()},
        where=_where(case, df),
        **kwargs,
    )
    return mice


def _events(rows):
    """Logged events as (it, im, dep, meth, out) tuples."""
    return [(int(it), int(im), str(dep), str(meth), str(out)) for it, im, dep, meth, out in rows]


def check_passive_setup(case):
    """The methods, the predictor matrix and the logged events of the setup step: R's."""
    mice = _impute_passive(case)
    method = {str(col): meth for col, meth in mice.method.items()}
    r_method = {col: _mice_method(meth) for col, meth in case["method"].items()}
    assert method == r_method, f"{case['name']}: methods {method}, R's {r_method}"
    _check_predictor_matrix(case, mice)
    r_log = case["logged_events"] or {name: [] for name in LOG_COLUMNS}
    r_events = _events(zip(*(r_log[name] for name in LOG_COLUMNS)))
    events = _events(mice.logged_events.itertuples(index=False))
    assert events == r_events, f"{case['name']}: logged events {events}, R's {r_events}"


def check_passive_visit_sequences(case):
    """The visit sequence by default, for each keyword and for the list of the case: R's, restricted to the
    variables R imputes (repeats kept)."""
    runs = [("default", None, case["effective_visit_sequence_default"])]
    runs += [(code, code, case["effective_visit_sequence"][code]) for code in VISITS]
    runs.append(("list", str_list(case["visit_sequence_arg"]), case["effective_visit_sequence_vector"]))
    for label, visit_sequence, expected in runs:
        visits = [str(col) for col in _impute_passive(case, visit_sequence=visit_sequence).visit_sequence]
        expected = str_list(expected)
        assert visits == expected, f"{case['name']}: visit sequence {label} {visits}, R's {expected}"


@pytest.mark.parametrize("case", PASSIVE, ids=[c["name"] for c in PASSIVE])
def test_passive_setup_matches_r(case):
    check_passive_setup(case)


@pytest.mark.parametrize("case", PASSIVE, ids=[c["name"] for c in PASSIVE])
def test_passive_visit_sequences_match_r(case):
    check_passive_visit_sequences(case)


def test_passive_cases_cover_the_order_and_the_setup_rules():
    by_name = {case["name"]: case for case in PASSIVE}
    case = by_name["nhanes_sum"]
    # the default moves s to the end; "roman" keeps it first, before its inputs; the list names it twice
    assert str_list(case["effective_visit_sequence_default"])[-1] == "s"
    assert str_list(case["effective_visit_sequence"]["roman"])[0] == "s"
    assert str_list(case["effective_visit_sequence_vector"]).count("s") == 2
    # a passive copy of a column: method "", logged as collinear; a complete passive column: method "", not logged
    copy, complete = by_name["nhanes_copy"], by_name["nhanes_complete"]
    assert copy["method"]["c2"] == "" and copy["logged_events"]["meth"] == ["collinear"]
    assert complete["method"]["a2"] == "" and complete["logged_events"] is None
    # an entirely missing passive column keeps its formula
    case = by_name["nhanes_all_missing"]
    assert decode_frame(case["data"])["s"].isna().all() and case["method"]["s"].startswith("~")
    # with where, the passive column has observed where-cells and comes first in "monotone"
    case = by_name["nhanes_sum_where"]
    where = _where(case, decode_frame(case["data"]))
    assert (where["s"] & decode_frame(case["data"])["s"].notna()).any()
    assert str_list(case["effective_visit_sequence"]["monotone"])[0] == "s"


# ---------------------------------------------------------------------------
# imputation.design: obtain_design, remove_lindep, find_collinear
# ---------------------------------------------------------------------------


# model.matrix() codes ordered factors with polynomial contrasts (contr.poly)
# and unordered ones with treatment dummies; boys_ordered_factor pins the first.
@pytest.mark.parametrize("case", DESIGN["obtain_design"], ids=[c["name"] for c in DESIGN["obtain_design"]])
def test_obtain_design_matches_r(case):
    from imputation.design import obtain_design

    df = decode_frame(case["data"])
    x = obtain_design(df, str_list(case["predictors"]))
    assert [str(c) for c in x.columns] == str_list(case["columns"])
    np.testing.assert_allclose(np.asarray(x, dtype=float), as_array(case["x"]).reshape(x.shape), rtol=1e-12, atol=1e-12)


LINDEP_SINGLE = [c for c in DESIGN["remove_lindep"] if not c["several_dependencies"]]
LINDEP_SEVERAL = [c for c in DESIGN["remove_lindep"] if c["several_dependencies"]]


def _remove_lindep(case):
    from imputation.design import remove_lindep

    keep = remove_lindep(
        as_array(case["x"]),
        as_array(case["y"]),
        as_bool(case["ry"]),
        eps=case["eps"],
        maxcor=case["maxcor"],
    )
    return np.asarray(keep, dtype=bool)


@pytest.mark.parametrize("case", LINDEP_SINGLE, ids=[c["name"] for c in LINDEP_SINGLE])
def test_remove_lindep_matches_r(case):
    np.testing.assert_array_equal(_remove_lindep(case), as_bool(case["keep"]))


@pytest.mark.parametrize("case", LINDEP_SEVERAL, ids=[c["name"] for c in LINDEP_SEVERAL])
def test_remove_lindep_several_dependencies_matches_r(case):
    """R's result when more than one column must go is order dependent and
    can keep a collinear column or drop an unrelated one (see l1_setup.R).
    A port that follows the R code reproduces it; one that removes every
    dependency instead must change this test on purpose."""
    np.testing.assert_array_equal(_remove_lindep(case), as_bool(case["keep"]))


@pytest.mark.parametrize("case", DESIGN["find_collinear"], ids=[c["name"] for c in DESIGN["find_collinear"]])
def test_find_collinear_matches_r(case):
    from imputation.design import find_collinear

    df = decode_frame(case["data"])
    got = find_collinear(df, threshold=case["threshold"])
    assert [str(v) for v in got] == str_list(case["collinear"])
