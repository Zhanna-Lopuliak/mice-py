"""L1 parity: missing-data pattern table.

Reference: tests/reference/r/l1_md_pattern.R (md.pattern(data, plot = FALSE)).
Python's table labels the margins "#rows", "#miss_row" and "#miss_col"; R
leaves those names empty. The tests compare names of variables, pattern
counts and all numbers.
"""

import numpy as np
import pytest

from plotting.utils import md_pattern_like

from ._helpers import as_array, decode_frame, load_fixture, str_list

pytestmark = pytest.mark.parity

MD = load_fixture("l1_md_pattern")


def _r_table(case):
    table = as_array(case["table"]).astype(int)
    counts = [int(c) for c in str_list(case["rownames"])[:-1]]
    columns = str_list(case["colnames"])[:-1]
    return table, counts, columns


def _py_table(case):
    out = md_pattern_like(decode_frame(case["data"]))
    table = np.asarray(out, dtype=float).astype(int)
    counts = [int(c) for c in list(out.index)[:-1]]
    columns = [str(c) for c in list(out.columns)[:-1]]
    return table, counts, columns


def _patterns(table, counts):
    """Pattern rows as a multiset: {(pattern, n_missing): count}."""
    return {(tuple(row[:-1]), row[-1]): count for row, count in zip(table[:-1].tolist(), counts)}


A10 = "A10: md_pattern_like orders patterns by number of missing cells, R by pattern"


def _params():
    for case in MD["cases"]:
        marks = []
        if case["name"] != "nhanes":  # nhanes: both orders coincide
            marks = [pytest.mark.xfail(strict=True, reason=A10)]
        yield pytest.param(case, id=case["name"], marks=marks)


@pytest.mark.parametrize("case", list(_params()))
def test_md_pattern_matches_r_including_row_order(case):
    r_table, r_counts, r_columns = _r_table(case)
    table, counts, columns = _py_table(case)
    assert columns == r_columns
    assert counts == r_counts
    np.testing.assert_array_equal(table, r_table)


@pytest.mark.parametrize("case", MD["cases"], ids=[c["name"] for c in MD["cases"]])
def test_md_pattern_patterns_and_counts_match_r(case):
    r_table, r_counts, r_columns = _r_table(case)
    table, counts, columns = _py_table(case)
    assert columns == r_columns
    assert _patterns(table, counts) == _patterns(r_table, r_counts)


@pytest.mark.parametrize("case", MD["cases"], ids=[c["name"] for c in MD["cases"]])
def test_md_pattern_column_order_and_margins_match_r(case):
    r_table, _, r_columns = _r_table(case)
    table, _, columns = _py_table(case)
    assert columns == r_columns
    np.testing.assert_array_equal(table[-1], r_table[-1])
    assert table.shape == r_table.shape
