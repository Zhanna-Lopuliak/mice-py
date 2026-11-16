"""L1 parity: quickpred.

Reference: tests/reference/r/l1_quickpred.R (quickpred on nhanes and boys).
Factors are rebuilt as pandas Categoricals with R's level order, because R
correlates their integer codes (data.matrix).
"""

import numpy as np
import pandas as pd
import pytest

from imputation import MICE

from ._helpers import decode_frame, load_fixture, str_list

pytestmark = pytest.mark.parity

QUICKPRED = load_fixture("l1_quickpred")
IDS = [c["name"] for c in QUICKPRED["cases"]]


def _data(case):
    return decode_frame(QUICKPRED["data"][case["data_name"]])


def _expected(case):
    return pd.DataFrame(
        np.array(case["predictor_matrix"], dtype=int),
        index=str_list(case["rows"]),
        columns=str_list(case["cols"]),
    )


def _quickpred(case):
    from imputation import quickpred

    return quickpred(
        _data(case),
        mincor=case["mincor"],
        minpuc=case["minpuc"],
        include=str_list(case["include"]) or None,
        exclude=str_list(case["exclude"]) or None,
        method=case["method"],
    )


@pytest.mark.parametrize("case", QUICKPRED["cases"], ids=IDS)
def test_quickpred_matches_r(case):
    got = _quickpred(case)
    expected = _expected(case)
    pd.testing.assert_frame_equal(got.loc[expected.index, expected.columns].astype(int), expected, check_names=False)


# The matrix of quickpred reaches impute() unchanged when passed as
# predictor_matrix, the replacement of the keyword arguments min_cor, min_puc,
# include, exclude and correlation_method removed in version 0.4.
@pytest.mark.parametrize("case", QUICKPRED["cases"], ids=IDS)
def test_impute_with_quickpred_matrix_matches_r(case):
    mice = MICE(_data(case))
    mice.impute(n_imputations=1, maxit=1, predictor_matrix=_quickpred(case))
    expected = _expected(case)
    pd.testing.assert_frame_equal(
        mice.predictor_matrix.loc[expected.index, expected.columns].astype(int),
        expected,
        check_names=False,
    )
