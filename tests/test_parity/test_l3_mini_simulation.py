"""L3 in-suite mini simulation: scenario S4 against R mice.

Scenario S4 of the L3 study (``simulations/scenarios.py``): x1 and x2 standard
normal with correlation 0.5, y = 0.5 + x1 + 0.5 x2 + e, n = 50, x1 missing
completely at random with probability 0.5. Every replication draws a new
dataset and imputes x1 m = 5 times through ``imputation.PMM.pmm``,
``imputation.midas.midas`` and ``imputation.norm.norm`` directly (x1 is the
only incomplete column, so this is one MICE iteration). Analysis: OLS of y on
x1 and x2.

The reference ``l3_mini_s4`` holds the same quantities from 20,000 R
replications through ``mice.impute.pmm``, ``mice.impute.midastouch`` and
``mice.impute.norm`` (``tests/reference/r/l3_mini_s4.R``). Python runs 200
replications with fixed seeds, in about two seconds. Each method has its own
generator per replication, ``default_rng([SEED, rep, method_index])``, so a
method is added at the end of ``METHODS`` and the others keep their numbers.
Checks, per method:

- between-imputation variance B of the x1 coefficient:
  |log(mean B Python / mean B R)| <= 5 standard errors of the log ratio;
- pooled x1 coefficient: |mean Python - mean R| <= 5 Monte Carlo errors.

Monte Carlo errors are sd / sqrt(reps) on each side, combined in quadrature.

Power: B is heavy-tailed across replications (sd about 1.6 times the mean), so
with 200 replications the B check detects a ratio of about 1.8 or more, not
small shifts. Finding A1 lowers pmm's mean B by about 14% here (0.0138 against
R's 0.0160, z = -5.9 with 3000 Python replications; 0.946 of R's value once A1
is patched in memory). That is invisible at 200 replications; the L2 tests
are the ones that catch A1.
"""

import numpy as np
import pytest

from imputation.midas import midas
from imputation.norm import norm
from imputation.PMM import pmm

from ._helpers import load_fixture

pytestmark = pytest.mark.parity

N = 50
M = 5
REPS = 200
SEED = 20261111
TOLERANCE = 5.0  # standard errors

#: Python function and R method name per study method. Append new methods: the
#: position is the method_index of the generator.
METHODS = {
    "pmm": (pmm, "pmm"),
    "midas": (midas, "midastouch"),
    "norm": (norm, "norm"),
}


def _dataset(rep):
    rng = np.random.default_rng([SEED, rep])
    z1 = rng.standard_normal(N)
    z2 = rng.standard_normal(N)
    e = rng.standard_normal(N)
    x1 = z1
    x2 = 0.5 * z1 + np.sqrt(0.75) * z2
    y = 0.5 + x1 + 0.5 * x2 + e
    observed = rng.random(N) >= 0.5
    x1 = np.where(observed, x1, np.nan)
    return x1, x2, y, observed


def _x1_coefficient(x1, x2, y):
    design = np.column_stack([np.ones(len(y)), x1, x2])
    return np.linalg.lstsq(design, y, rcond=None)[0][1]


def _summary(values):
    values = np.asarray(values, dtype=float)
    return values.mean(), values.std(ddof=1) / np.sqrt(len(values))


def _simulate(method):
    func = METHODS[method][0]
    method_index = list(METHODS).index(method)
    qbar, b = [], []
    for rep in range(1, REPS + 1):
        x1, x2, y, observed = _dataset(rep)
        x = np.column_stack([x2, y])
        rng = np.random.default_rng([SEED, rep, method_index])
        coefs = []
        for _ in range(M):
            imputed = np.asarray(func(y=x1, id_obs=observed, x=x, id_mis=~observed, rng=rng), dtype=float)
            x1c = x1.copy()
            x1c[~observed] = imputed
            coefs.append(_x1_coefficient(x1c, x2, y))
        qbar.append(np.mean(coefs))
        b.append(np.var(coefs, ddof=1))
    mean_q, se_q = _summary(qbar)
    mean_b, se_b = _summary(b)
    return {"qbar": (mean_q, se_q), "b": (mean_b, se_b)}


@pytest.fixture(scope="module")
def reference():
    return load_fixture("l3_mini_s4")


@pytest.fixture(scope="module")
def python_results():
    return {method: _simulate(method) for method in METHODS}


def _reference(reference, method, quantity):
    entry = reference["methods"][METHODS[method][1]][quantity]
    return entry["mean"], entry["se"]


@pytest.mark.parametrize("method", list(METHODS))
def test_between_imputation_variance_matches_r(method, python_results, reference):
    mean_py, se_py = python_results[method]["b"]
    mean_r, se_r = _reference(reference, method, "b_x1")
    log_ratio = np.log(mean_py / mean_r)
    se = np.hypot(se_py / mean_py, se_r / mean_r)
    assert abs(log_ratio) <= TOLERANCE * se, (
        f"{method}: mean B of the x1 coefficient {mean_py:.5f} (se {se_py:.5f}) in Python, "
        f"{mean_r:.5f} (se {se_r:.5f}) in R; log ratio {log_ratio:+.3f}, "
        f"z = {log_ratio / se:+.2f}"
    )


@pytest.mark.parametrize("method", list(METHODS))
def test_pooled_coefficient_matches_r(method, python_results, reference):
    mean_py, se_py = python_results[method]["qbar"]
    mean_r, se_r = _reference(reference, method, "qbar_x1")
    diff = mean_py - mean_r
    se = np.hypot(se_py, se_r)
    assert abs(diff) <= TOLERANCE * se, (
        f"{method}: mean pooled x1 coefficient {mean_py:.4f} (se {se_py:.4f}) in Python, "
        f"{mean_r:.4f} (se {se_r:.4f}) in R; difference {diff:+.4f}, z = {diff / se:+.2f}"
    )
