"""Layer 2 at chain level: 200 full pmm chains, R against Python.

R ran ``mice(data, m = 200, maxit = 5, method = "pmm", ...)`` on the
embedded frame (100 rows, three incomplete numeric columns) and stored the
mean, variance and kurtosis across the 200 chains of the last-iteration
``chainMean`` and ``chainVar`` per variable. Python runs the same set-up
(full predictor matrix, columns visited in order) and the two samples of 200
values are compared with the z-scores of T2. ``chainVar`` in R and
``chain_var`` in the port both hold variances.

Before finding A1 was fixed this test passed only narrowly: the variance
across chains of the last chain mean of y2 was about twice R's (|z| = 5.4
against a threshold of 5.5); with the corrected Cholesky factor it is about 1.
"""

import functools

import numpy as np
import pandas as pd
import pytest

from ._helpers import as_array, load_fixture
from ._l2_stats import Z_MAX, moments, z_logvar, z_mean

pytestmark = pytest.mark.parity

SEED = 20261


@functools.cache
def chain_run():
    from imputation import MICE

    fx = load_fixture("l2_chain")
    cols = list(fx["columns"])
    data = pd.DataFrame({c: as_array(fx["data"][c]) for c in cols})
    full = pd.DataFrame(1, index=cols, columns=cols)
    np.fill_diagonal(full.values, 0)
    mice = MICE(data)
    with pytest.warns(UserWarning, match="200 imputations is a large number"):
        mice.impute(
            n_imputations=fx["m"],
            maxit=fx["maxit"],
            method="pmm",
            predictor_matrix=full,
            visit_sequence=cols,
            seed=SEED,
        )
    return fx, mice


@pytest.mark.parametrize("statistic", ["chain_mean", "chain_var"])
def test_last_iteration_chain_statistics_match_r(statistic):
    fx, mice = chain_run()
    store = getattr(mice, statistic)
    m = fx["m"]
    lines, worst = [], 0.0
    for col in fx["columns"]:
        values = np.asarray(store[col])[-1, :]
        assert values.shape == (m,) and np.isfinite(values).all()
        py, r = moments(values), fx[statistic][col]
        zm = float(z_mean(py["mean"], py["var"], m, r["mean"], r["var"], m))
        zv = float(z_logvar(py["var"], py["kurt"], m, r["var"], r["kurt"], m))
        worst = max(worst, abs(zm), abs(zv))
        lines.append(f"{col}: z mean {zm:+.2f}, z log-variance {zv:+.2f}")
    assert worst <= Z_MAX, f"{statistic}: " + "; ".join(lines) + f" (pass: |z| <= {Z_MAX:g})"
