"""L1 parity: least squares (estimice) and the coefficient draw (norm_draw).

Reference: tests/reference/r/l1_sampler.R (mice:::estimice, mice:::.norm.draw).
"""

import numpy as np
import pytest

from imputation.sampler import estimice, norm_draw

from ._helpers import ScriptedGenerator, as_array, as_bool, load_fixture

pytestmark = pytest.mark.parity

RTOL = 1e-8
ATOL = 1e-10

ESTIMICE = load_fixture("l1_estimice")
NORM_DRAW = load_fixture("l1_norm_draw")


def _ids(cases):
    return [case["name"] for case in cases]


@pytest.mark.parametrize("case", ESTIMICE["cases"], ids=_ids(ESTIMICE["cases"]))
def test_estimice_matches_r(case):
    x = as_array(case["x"])
    if x.ndim == 1:
        x = x.reshape(-1, 1)
    y = as_array(case["y"])
    p = estimice(x, y, ls_meth=case["ls_meth"], ridge=case["ridge"])

    assert p["df"] == case["df"]
    np.testing.assert_allclose(np.ravel(p["c"]), as_array(case["c"]), rtol=RTOL, atol=ATOL)
    # residuals are close to zero relative to y, so compare on the scale of y
    np.testing.assert_allclose(np.ravel(p["r"]), as_array(case["r"]), rtol=0, atol=RTOL * np.abs(y).max())
    v = as_array(case["v"])
    v = v.reshape(np.shape(p["v"]))
    np.testing.assert_allclose(p["v"], v, rtol=RTOL, atol=ATOL)


def _draw_inputs():
    y = as_array(NORM_DRAW["y"])
    ry = as_bool(NORM_DRAW["ry"])
    x = as_array(NORM_DRAW["x"])
    return y, ry, x


def _draw(case):
    y, ry, x = _draw_inputs()
    rng = ScriptedGenerator(chisq=case["chisq"], normals=case["normals"])
    parm = norm_draw(y, ry, x, rng=rng, ls_meth=case["ls_meth"])
    assert rng.unused == (0, 0), "not every captured R draw was consumed"
    return parm


DRAW_IDS = _ids(NORM_DRAW["cases"])


@pytest.mark.parametrize("case", NORM_DRAW["cases"], ids=DRAW_IDS)
def test_norm_draw_coef_matches_r(case):
    parm = _draw(case)
    np.testing.assert_allclose(parm["coef"], as_array(case["coef"]), rtol=RTOL, atol=ATOL)


@pytest.mark.parametrize("case", NORM_DRAW["cases"], ids=DRAW_IDS)
def test_norm_draw_sigma_matches_r(case):
    parm = _draw(case)
    np.testing.assert_allclose(np.ravel(parm["sigma"]), [case["sigma"]], rtol=RTOL)


@pytest.mark.parametrize("case", NORM_DRAW["cases"], ids=DRAW_IDS)
def test_norm_draw_beta_matches_r(case):
    parm = _draw(case)
    np.testing.assert_allclose(parm["beta"], as_array(case["beta"]), rtol=RTOL, atol=ATOL)
