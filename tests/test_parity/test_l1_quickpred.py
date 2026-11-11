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


@pytest.mark.parametrize("case", QUICKPRED["cases"], ids=IDS)
def test_quickpred_matches_r(case):
    from imputation import quickpred

    got = quickpred(
        _data(case),
        mincor=case["mincor"],
        minpuc=case["minpuc"],
        include=str_list(case["include"]) or None,
        exclude=str_list(case["exclude"]) or None,
        method=case["method"],
    )
    expected = _expected(case)
    pd.testing.assert_frame_equal(
        got.loc[expected.index, expected.columns].astype(int), expected, check_names=False
    )


# The same rule is reached through the keyword arguments of impute()
# (min_cor, min_puc, include, exclude, correlation_method). These stay, with a
# deprecation warning, for one version.
@pytest.mark.parametrize("case", QUICKPRED["cases"], ids=IDS)
def test_impute_quickpred_arguments_match_r(case):
    mice = MICE(_data(case))
    with pytest.warns(DeprecationWarning, match="imputation.quickpred"):
        mice.impute(
            n_imputations=1,
            maxit=1,
            min_cor=case["mincor"],
            min_puc=case["minpuc"],
            include=str_list(case["include"]) or None,
            exclude=str_list(case["exclude"]) or None,
            correlation_method=case["method"],
        )
    expected = _expected(case)
    pd.testing.assert_frame_equal(
        mice.predictor_matrix.loc[expected.index, expected.columns].astype(int),
        expected,
        check_names=False,
    )
