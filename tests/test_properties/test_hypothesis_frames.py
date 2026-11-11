"""Item 5: the invariants of item 1 on random small numeric DataFrames.

Frames have 8 to 40 rows and 2 to 5 columns of bounded finite floats, are
sometimes rounded so that ties occur, sometimes carry a non-default index, and
have random missingness that leaves at least two observed values per column
and at least one observed value per row (all-empty rows are C4, tested in
test_edge_cases). The full predictor matrix is used, as planned.
"""

import numpy as np
import pandas as pd
import pytest
from hypothesis import HealthCheck, given, settings
from hypothesis import strategies as st
from hypothesis.extra.numpy import arrays

from ._helpers import assert_all_invariants, impute

FLOATS = st.floats(min_value=-1e3, max_value=1e3, allow_nan=False, allow_infinity=False, allow_subnormal=False)


@st.composite
def numeric_frames(draw, min_observed=2, values="hypothesis"):
    n = draw(st.integers(8, 40))
    k = draw(st.integers(2, 5))
    if values == "hypothesis":
        # Hypothesis floats: favours degenerate cases (constants, duplicates, zeros).
        vals = draw(arrays(np.float64, (n, k), elements=FLOATS))
    else:
        # Well-conditioned: correlated normal columns from a drawn seed.
        rng = np.random.default_rng(draw(st.integers(0, 2**32 - 1)))
        latent = rng.normal(size=(n, 1))
        vals = latent + rng.normal(scale=draw(st.sampled_from([0.5, 1.0, 2.0])), size=(n, k))
        vals = vals * draw(st.sampled_from([1.0, 5.0, 100.0]))
    decimals = draw(st.sampled_from([None, None, 0, 1]))
    if decimals is not None:
        vals = np.round(vals, decimals)
    vals = np.array(vals, dtype=float)

    missing = draw(arrays(np.bool_, (n, k)))
    for j in range(k):  # at least `min_observed` observed values per column
        obs = np.flatnonzero(~missing[:, j])
        if len(obs) < min_observed:
            missing[:min_observed, j] = False
    for i in range(n):  # no all-empty rows
        if missing[i].all():
            missing[i, i % k] = False
    vals[missing] = np.nan

    df = pd.DataFrame(vals, columns=[f"v{j}" for j in range(k)])
    if draw(st.booleans()):
        order = np.random.default_rng(n * 31 + k).permutation(n)
        df.index = pd.Index(order * 3 + 1)
    return df


def _check_frame(df, method):
    before = df.copy(deep=True)
    mice, out = impute(df, method, n_imputations=2, maxit=2, seed=1)
    assert_all_invariants(df, before, mice, out, 2, 2)


def _settings(n):
    return settings(
        max_examples=n,
        deadline=None,
        derandomize=True,
        database=None,
        suppress_health_check=[HealthCheck.too_slow, HealthCheck.data_too_large],
    )


@pytest.mark.parametrize(
    "method,n_examples",
    [
        pytest.param("pmm", 40, id="pmm"),
        pytest.param("sample", 40, id="sample"),
        pytest.param("cart", 15, id="cart"),
        pytest.param("rf", 6, id="rf"),
    ],
)
def test_invariants_on_random_frames(method, n_examples):
    @_settings(n_examples)
    @given(numeric_frames())
    def check(df):
        _check_frame(df, method)

    check()


@pytest.mark.parametrize("method,n_examples", [("pmm", 40), ("sample", 20)])
def test_invariants_on_well_conditioned_random_frames(method, n_examples):
    """Same invariants on frames where today's pmm does not hit B3/B4:
    correlated normal columns and at least eight observed values per column."""

    @_settings(n_examples)
    @given(numeric_frames(min_observed=8, values="normal"))
    def check(df):
        _check_frame(df, method)

    check()
