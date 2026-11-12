"""Item 5: the invariants of item 1 on random small numeric DataFrames.

Frames have 8 to 40 rows and 2 to 5 columns of bounded finite floats, are
sometimes rounded so that ties occur, sometimes carry a non-default index, and
have random missingness that leaves at least two observed values per column
and at least one observed value per row (all-empty rows are C4, tested in
test_edge_cases). The full predictor matrix is used, as planned.

A few frames with categorical columns (text, Categorical, ordered
Categorical; two to four categories) check polyreg the same way, and a few
with ordered Categoricals check polr.
"""

import numpy as np
import pandas as pd
import pytest
from hypothesis import HealthCheck, given, settings
from hypothesis import strategies as st
from hypothesis.extra.numpy import arrays

from ._helpers import assert_all_invariants, impute, imputes_observed_values

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


@st.composite
def categorical_frames(draw, kinds=("text", "category", "ordered")):
    """A complete predictor ``x``, an incomplete numeric column ``v`` and one
    or two incomplete categorical columns related to them, each of one of
    `kinds`: text, an unordered or an ordered Categorical with 2 to 4
    categories, declared in reverse order (a category may not occur); 12 to
    30 rows, at least two observed values per column."""
    n = draw(st.integers(12, 30))
    rng = np.random.default_rng(draw(st.integers(0, 2**32 - 1)))
    latent = rng.normal(size=n)
    data = {"x": latent + rng.normal(scale=0.5, size=n), "v": latent + rng.normal(size=n)}
    for j in range(draw(st.integers(1, 2))):
        k = draw(st.integers(2, 4))
        labels = np.array([f"c{i}" for i in range(k)])
        noise = draw(st.sampled_from([0.3, 1.0, 3.0]))
        codes = np.digitize(latent + rng.normal(scale=noise, size=n), np.linspace(-1, 1, k + 1)[1:-1])
        kind = draw(st.sampled_from(kinds))
        if kind == "text":
            data[f"g{j}"] = pd.Series(labels[codes], dtype=object)
        else:
            data[f"g{j}"] = pd.Categorical(labels[codes], categories=labels[::-1], ordered=kind == "ordered")
    df = pd.DataFrame(data)
    for col in df.columns[1:]:
        missing = rng.random(n) < draw(st.sampled_from([0.1, 0.25, 0.4]))
        missing[rng.choice(n, size=2, replace=False)] = False  # at least two observed values
        df.loc[missing, col] = np.nan
    return df


def _check_frame(df, method):
    before = df.copy(deep=True)
    mice, out = impute(df, method, n_imputations=2, maxit=2, seed=1)
    assert_all_invariants(df, before, mice, out, 2, 2, donor=imputes_observed_values(method))


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
        pytest.param("norm", 15, id="norm"),
    ],
)
def test_invariants_on_random_frames(method, n_examples):
    @_settings(n_examples)
    @given(numeric_frames())
    def check(df):
        _check_frame(df, method)

    check()


def test_polyreg_invariants_on_random_categorical_frames():
    """polyreg imputes the categorical columns, pmm the numeric one; every
    imputed cell holds a label or value observed in its column."""

    @_settings(10)
    @given(categorical_frames())
    def check(df):
        _check_frame(df, "polyreg")

    check()


def test_polr_invariants_on_random_ordered_frames():
    """polr imputes the ordered columns (by its fallback where they have two
    levels), pmm the numeric one; every imputed cell holds a declared
    category of its column, or an observed value."""

    @_settings(8)
    @given(categorical_frames(kinds=("ordered",)))
    def check(df):
        _check_frame(df, "polr")

    check()


@pytest.mark.parametrize("method,n_examples", [("pmm", 40), ("sample", 20), ("norm", 10)])
def test_invariants_on_well_conditioned_random_frames(method, n_examples):
    """Same invariants on frames where today's pmm does not hit B3/B4:
    correlated normal columns and at least eight observed values per column."""

    @_settings(n_examples)
    @given(numeric_frames(min_observed=8, values="normal"))
    def check(df):
        _check_frame(df, method)

    check()
