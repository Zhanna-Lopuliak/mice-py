"""L1 parity: donor probabilities of midastouch.

Reference: tests/reference/r/l1_midastouch.R. R's bootstrap weights (omega)
are injected, so the probabilities are deterministic.
"""

import numpy as np
import pytest

from ._helpers import as_array, as_bool, load_fixture

pytestmark = pytest.mark.parity

MIDAS = load_fixture("l1_midastouch")
IDS = [c["name"] for c in MIDAS["cases"]]


@pytest.mark.xfail(
    strict=True,
    raises=ImportError,
    reason="B5: _donor_probabilities not implemented yet (planned with the midas restructure)",
)
@pytest.mark.parametrize("case", MIDAS["cases"], ids=IDS)
def test_midastouch_donor_probabilities_match_r(case):
    from imputation.midas import _donor_probabilities

    y = as_array(MIDAS["y"])
    ry = as_bool(MIDAS["ry"])
    x = as_array(MIDAS["x"])
    probs, kappa = _donor_probabilities(
        y,
        ry,
        x,
        ~ry,
        as_array(case["omega"]),
        ridge=MIDAS["ridge"],
        midas_kappa=case["midas_kappa"],
        outout=case["outout"],
    )
    expected = as_array(case["probs"])
    assert np.shape(probs) == expected.shape
    np.testing.assert_allclose(float(kappa), case["kappa"], rtol=1e-8)
    # probabilities span many orders of magnitude; the smallest are below
    # 1e-10, so an absolute floor of 1e-14 keeps the check relative.
    np.testing.assert_allclose(probs, expected, rtol=1e-8, atol=1e-14)
    np.testing.assert_allclose(np.sum(probs, axis=0), 1.0, rtol=1e-12)
