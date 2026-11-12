"""L1 parity: the default method of each column type, R's ``make.method``.

Reference: tests/reference/r/l1_make_method.R. One column of each type that
R's ``assign.method`` tells apart: numeric, integer 0/1, logical, factors with
2 and 3 levels, ordered factors with 2 and 4 levels, a factor with 3
declared and 2 observed levels, a factor with one level, a character column
and a complete column; ``make.method`` with the default ``defaultMethod`` and
with another one. The frame is rebuilt with factors as Categoricals,
integers as ``Int64`` and logicals as ``boolean``.

The one documented difference is the character column: R gives it the first
default whatever its values (and then does not impute it, as its setup step
finds it constant); mice-py treats text as an unordered factor of its
distinct values (here three: polyreg).
"""

import pandas as pd
import pytest

from imputation import make_method

from ._helpers import as_array, load_fixture, str_list

pytestmark = pytest.mark.parity

FIXTURE = load_fixture("l1_make_method")
# Columns where mice-py differs from R on purpose: the method mice-py gives
# with the default and with the other default methods
DIFFERENCES = {"chr": ("polyreg", "cart")}


def _frame():
    payload = FIXTURE["data"]
    columns = {}
    for col in str_list(payload["columns"]):
        values = payload["data"][col]
        kind = FIXTURE["types"][col]
        if col in payload["levels"]:
            columns[col] = pd.Categorical(
                values, categories=str_list(payload["levels"][col]), ordered=bool(payload["ordered"][col])
            )
        elif kind == "integer":
            columns[col] = pd.array(values, dtype="Int64")
        elif kind == "logical":
            columns[col] = pd.array(values, dtype="boolean")
        elif kind == "character":
            columns[col] = pd.Series(values, dtype=object)
        else:
            columns[col] = as_array(values)
    return pd.DataFrame(columns)


def _expected(r_methods, position):
    return {col: DIFFERENCES.get(col, (None, None))[position] or meth for col, meth in r_methods.items()}


def test_frame_has_the_types_of_r():
    df = _frame()
    assert df["int01"].dtype == "Int64" and df["lgl"].dtype == "boolean" and df["chr"].dtype == object
    assert df["o4"].cat.ordered and not df["f3"].cat.ordered
    assert list(df["f3_unused"].cat.categories) == ["a", "b", "c"] and df["f3_unused"].nunique() == 2
    assert df.isna().sum().drop("complete").eq(1).all() and not df["complete"].isna().any()


def test_make_method_matches_r():
    got = make_method(_frame())
    assert got == _expected(FIXTURE["method"], 0)
    # R's own result on the text column
    assert FIXTURE["method"]["chr"] == "pmm"


def test_make_method_with_other_default_methods_matches_r():
    other = list(FIXTURE["other_default_method"])
    got = make_method(_frame(), default_method=other)
    assert got == _expected(FIXTURE["method_other"], 1)
    assert FIXTURE["method_other"]["chr"] == other[0]
