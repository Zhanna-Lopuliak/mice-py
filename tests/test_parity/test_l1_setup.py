"""L1 parity: the setup step of mice() and the design helpers.

Reference: tests/reference/r/l1_setup.R (mice(data, maxit = 0),
mice:::obtain.design, mice:::remove.lindep, mice:::find.collinear).

R's visitSequence may list variables it never imputes (complete variables
for "monotone", "revmonotone" and the default; variables whose method
mice.edit.setup set to ""). The sampler skips those, so the fixture stores
the effective order: R's sequence restricted to variables with a non-empty
method. The tests compare the Python sequence restricted to the same
variables, and check separately that Python visits no complete variable.

The port has no logreg / polyreg / polr, so methods are compared for
numeric columns only.
"""

import numpy as np
import pandas as pd
import pytest

from imputation import MICE

from ._helpers import as_array, as_bool, decode_frame, load_fixture, str_list

pytestmark = pytest.mark.parity

SETUP = load_fixture("l1_setup")
DESIGN = load_fixture("l1_design")
VISITS = ["roman", "arabic", "monotone", "revmonotone"]


def _impute(case, **kwargs):
    df = decode_frame(case["data"])
    mice = MICE(df)
    mice.impute(n_imputations=1, maxit=1, **kwargs)
    return df, mice


def _params(cases, known):
    for case in cases:
        reason = known.get(case["name"])
        marks = [pytest.mark.xfail(strict=True, reason=reason)] if reason else []
        yield pytest.param(case, id=case["name"], marks=marks)


def _r_matrix(payload):
    return pd.DataFrame(
        np.array(payload["values"], dtype=int),
        index=str_list(payload["rows"]),
        columns=str_list(payload["cols"]),
    )


# ---------------------------------------------------------------------------
# method
# ---------------------------------------------------------------------------

A2 = "A2: default method is 'sample', R uses 'pmm'"


@pytest.mark.xfail(strict=True, reason=A2)
@pytest.mark.parametrize("case", SETUP["cases"], ids=[c["name"] for c in SETUP["cases"]])
def test_default_method_is_pmm_for_numeric_columns(case):
    df, mice = _impute(case)
    numeric = [
        col
        for col, meth in case["method"].items()
        if meth == "pmm" and pd.api.types.is_numeric_dtype(df[col])
    ]
    assert numeric, "fixture has no numeric incomplete column"
    assert {col: mice.method[col] for col in numeric} == {col: "pmm" for col in numeric}


# ---------------------------------------------------------------------------
# predictor matrix
# ---------------------------------------------------------------------------

PRED_KNOWN = {
    "nhanes": "A7: default predictor matrix comes from _quickpred, R uses the full matrix",
}


@pytest.mark.parametrize("case", list(_params(SETUP["cases"], PRED_KNOWN)))
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
    the port's default (finding A7)."""
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


C3 = "C3: default visit order is 'monotone', R uses left to right ('roman')"


@pytest.mark.xfail(strict=True, reason=C3)
@pytest.mark.parametrize("case", SETUP["cases"], ids=[c["name"] for c in SETUP["cases"]])
def test_default_visit_sequence_matches_r(case):
    df, mice = _impute(case)
    _check_visit_sequence(case, df, mice.visit_sequence, case["effective_visit_sequence_default"])


def _visit_params():
    for case in SETUP["cases"]:
        for code in VISITS:
            marks = []
            if code != "monotone":
                reason = f"C3: visit_sequence='{code}' is not supported"
                marks = [pytest.mark.xfail(strict=True, reason=reason)]
            yield pytest.param(case, code, id=f"{case['name']}-{code}", marks=marks)


@pytest.mark.parametrize("case, code", list(_visit_params()))
def test_visit_sequence_matches_r(case, code):
    df, mice = _impute(case, visit_sequence=code)
    _check_visit_sequence(case, df, mice.visit_sequence, case["effective_visit_sequence"][code])


# ---------------------------------------------------------------------------
# imputation.design: obtain_design, remove_lindep, find_collinear
# ---------------------------------------------------------------------------

# model.matrix() codes ordered factors with polynomial contrasts (contr.poly)
# and unordered ones with treatment dummies; boys_ordered_factor pins the first.
@pytest.mark.parametrize(
    "case", DESIGN["obtain_design"], ids=[c["name"] for c in DESIGN["obtain_design"]]
)
def test_obtain_design_matches_r(case):
    from imputation.design import obtain_design

    df = decode_frame(case["data"])
    x = obtain_design(df, str_list(case["predictors"]))
    assert [str(c) for c in x.columns] == str_list(case["columns"])
    np.testing.assert_allclose(
        np.asarray(x, dtype=float), as_array(case["x"]).reshape(x.shape), rtol=1e-12, atol=1e-12
    )


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


@pytest.mark.parametrize(
    "case", DESIGN["find_collinear"], ids=[c["name"] for c in DESIGN["find_collinear"]]
)
def test_find_collinear_matches_r(case):
    from imputation.design import find_collinear

    df = decode_frame(case["data"])
    got = find_collinear(df, threshold=case["threshold"])
    assert [str(v) for v in got] == str_list(case["collinear"])
