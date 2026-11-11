"""Scenarios of the L3 simulation study: data, missingness, analysis and truth.

Every scenario draws one complete dataset per replication from
``numpy.random.default_rng([scenario.index, rep])`` and then deletes values.
The order of the draws is fixed and documented in each generator, so a dataset
is reproducible from ``(scenario, rep)`` alone. Python and R both read the
CSV files written by ``study.py generate``; R never regenerates the data.

Missingness
-----------
"MCAR with rate r": each cell is missing independently with probability r.

"MAR on v with rate r": each cell is missing independently with probability
``expit(a + 1.5 * z(v))``, where ``z(v)`` is ``v`` standardised within the
sample (mean 0, standard deviation with divisor n - 1) and ``a`` is solved per
dataset so that the mean of the probabilities is exactly r.

Analysis
--------
:func:`analyse` returns, for one completed dataset, the estimate, its squared
standard error and the complete-data degrees of freedom of every term:

- regression terms: OLS coefficients with the usual ``sigma^2 (X'X)^-1``
  variance and ``dfcom = n - p`` (what R's ``lm`` and ``pool`` use);
- a mean: ``s^2 / n`` with ``dfcom = n - 1``;
- a proportion: ``p (1 - p) / n`` with ``dfcom = inf`` (Wald interval).

``run_r.R`` implements the same analysis with ``lm``; the pseudo-method
``complete`` checks that both give identical numbers on identical data.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable

import numpy as np
import pandas as pd
from scipy.optimize import brentq

#: Slope of the MAR logit on the standardised driver variable.
MAR_SLOPE = 1.5

#: Levels of the categorical variable of S5, in R factor order.
G_LEVELS = ("a", "b", "c")


# ---------------------------------------------------------------------------
# Missingness
# ---------------------------------------------------------------------------


def expit(t):
    """Logistic function."""
    return 1.0 / (1.0 + np.exp(-t))


def mar_probabilities(v, rate, slope=MAR_SLOPE):
    """Missingness probabilities ``expit(a + slope * z(v))`` with mean ``rate``.

    ``z(v)`` is ``v`` standardised within the sample (divisor n - 1). The
    intercept ``a`` is found by root finding, so ``mean(p) == rate`` up to
    1e-12.
    """
    v = np.asarray(v, dtype=float)
    z = (v - v.mean()) / v.std(ddof=1)

    def gap(a):
        return expit(a + slope * z).mean() - rate

    a = brentq(gap, -50.0, 50.0, xtol=1e-14, rtol=1e-14, maxiter=500)
    return expit(a + slope * z)


def mcar_mask(rng, n, rate):
    """Boolean mask, True = missing, each cell independently with ``rate``."""
    return rng.random(n) < rate


def mar_mask(rng, v, rate, slope=MAR_SLOPE):
    """Boolean mask, True = missing, with MAR probabilities driven by ``v``."""
    p = mar_probabilities(v, rate, slope)
    return rng.random(len(p)) < p


# ---------------------------------------------------------------------------
# Data generators. Each returns (complete, incomplete) DataFrames.
# ---------------------------------------------------------------------------


def _model_m(rng, n):
    """Common model M, draws in this order: z1, z2, e (each n standard normals).

    x1 = z1, x2 = 0.5 z1 + sqrt(0.75) z2 (unit variances, correlation 0.5),
    y = 0.5 + 1.0 x1 + 0.5 x2 + e.
    """
    z1 = rng.standard_normal(n)
    z2 = rng.standard_normal(n)
    e = rng.standard_normal(n)
    x1 = z1
    x2 = 0.5 * z1 + np.sqrt(0.75) * z2
    y = 0.5 + 1.0 * x1 + 0.5 * x2 + e
    return pd.DataFrame({"x1": x1, "x2": x2, "y": y})


def _gen_s1(rng, n):
    full = _model_m(rng, n)
    data = full.copy()
    data.loc[mcar_mask(rng, n, 0.30), "x1"] = np.nan
    return full, data


def _gen_s2(rng, n):
    full = _model_m(rng, n)
    data = full.copy()
    data.loc[mar_mask(rng, full["y"], 0.30), "x1"] = np.nan
    return full, data


def _gen_s3(rng, n):
    # Two independent draws with the same probabilities: first y, then x1.
    full = _model_m(rng, n)
    data = full.copy()
    miss_y = mar_mask(rng, full["x2"], 0.25)
    miss_x1 = mar_mask(rng, full["x2"], 0.25)
    data.loc[miss_y, "y"] = np.nan
    data.loc[miss_x1, "x1"] = np.nan
    return full, data


def _gen_s4(rng, n):
    full = _model_m(rng, n)
    data = full.copy()
    data.loc[mcar_mask(rng, n, 0.50), "x1"] = np.nan
    return full, data


def _gen_s5(rng, n):
    """Draws: g codes (n uniforms through ``rng.choice``), x noise, e, then masks."""
    code = rng.choice(3, size=n, p=[0.5, 0.3, 0.2])
    x = 0.5 * code + rng.standard_normal(n)
    e = rng.standard_normal(n)
    y = 0.5 * (code == 1) + 1.0 * (code == 2) + 0.7 * x + e
    g = np.asarray(G_LEVELS, dtype=object)[code]
    full = pd.DataFrame({"g": g, "x": x, "y": y})
    data = full.copy()
    miss_g = mar_mask(rng, full["y"], 0.30)
    miss_x = mcar_mask(rng, n, 0.20)
    data.loc[miss_g, "g"] = np.nan
    data.loc[miss_x, "x"] = np.nan
    return full, data


def _gen_s6(rng, n):
    """Draws: x1, u (sd 0.5), e, then the mask."""
    x1 = rng.standard_normal(n)
    u = 0.5 * rng.standard_normal(n)
    e = rng.standard_normal(n)
    w = np.exp(0.5 * x1 + u)
    y = 1.0 + 0.5 * w + 0.5 * x1 + e
    full = pd.DataFrame({"x1": x1, "w": w, "y": y})
    data = full.copy()
    data.loc[mar_mask(rng, full["y"], 0.30), "w"] = np.nan
    return full, data


# ---------------------------------------------------------------------------
# Scenario definitions
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Scenario:
    """One row of the scenario table.

    ``regression`` is ``(response, predictors)`` with predictors in the order
    of the model terms; a categorical predictor expands to treatment dummies
    named ``<col><level>`` (R's ``model.matrix`` names). ``means`` lists the
    columns whose mean is a target, ``proportions`` maps a term name to
    ``(column, level)``.
    """

    id: str
    index: int
    n: int
    maxit: int
    columns: tuple
    description: str
    missingness: str
    generator: Callable = field(repr=False)
    regression: tuple = ()
    means: tuple = ()
    proportions: dict = field(default_factory=dict)
    categorical: tuple = ()
    truth: dict = field(default_factory=dict)

    def generate(self, rep):
        """Return ``(complete, incomplete)`` for replication ``rep`` (1-based)."""
        rng = np.random.default_rng([self.index, rep])
        return self.generator(rng, self.n)

    @property
    def terms(self):
        return list(self.truth)


_TRUTH_M = {"(Intercept)": 0.5, "x1": 1.0, "x2": 0.5, "mean(x1)": 0.0}

SCENARIOS = {
    s.id: s
    for s in [
        Scenario(
            id="S1", index=1, n=200, maxit=1, columns=("x1", "x2", "y"),
            description="model M", missingness="x1 MCAR 30%",
            generator=_gen_s1, regression=("y", ("x1", "x2")), means=("x1",),
            truth=dict(_TRUTH_M),
        ),
        Scenario(
            id="S2", index=2, n=200, maxit=1, columns=("x1", "x2", "y"),
            description="model M", missingness="x1 MAR on y 30%",
            generator=_gen_s2, regression=("y", ("x1", "x2")), means=("x1",),
            truth=dict(_TRUTH_M),
        ),
        Scenario(
            id="S3", index=3, n=200, maxit=5, columns=("x1", "x2", "y"),
            description="model M",
            missingness="y and x1 each MAR on x2 25%, independent draws",
            generator=_gen_s3, regression=("y", ("x1", "x2")), means=("x1", "y"),
            truth={**_TRUTH_M, "mean(y)": 0.5},
        ),
        Scenario(
            id="S4", index=4, n=50, maxit=1, columns=("x1", "x2", "y"),
            description="model M, small sample", missingness="x1 MCAR 50%",
            generator=_gen_s4, regression=("y", ("x1", "x2")), means=("x1",),
            truth=dict(_TRUTH_M),
        ),
        Scenario(
            id="S5", index=5, n=200, maxit=5, columns=("g", "x", "y"),
            description="g in {a,b,c} (0.5, 0.3, 0.2); x ~ N(0.5 code, 1); "
            "y = 0.5[g=b] + 1.0[g=c] + 0.7x + e",
            missingness="g MAR on y 30%; x MCAR 20%",
            generator=_gen_s5, regression=("y", ("g", "x")), means=("x",),
            proportions={"P(g=c)": ("g", "c")}, categorical=("g",),
            truth={"(Intercept)": 0.0, "gb": 0.5, "gc": 1.0, "x": 0.7,
                   "P(g=c)": 0.2, "mean(x)": 0.35},
        ),
        Scenario(
            id="S6", index=6, n=200, maxit=1, columns=("x1", "w", "y"),
            description="w = exp(0.5 x1 + u), u ~ N(0, 0.5^2); "
            "y = 1 + 0.5w + 0.5x1 + e",
            missingness="w MAR on y 30%",
            generator=_gen_s6, regression=("y", ("w", "x1")), means=("w",),
            truth={"(Intercept)": 1.0, "w": 0.5, "x1": 0.5,
                   "mean(w)": float(np.exp(0.25))},
        ),
    ]
}


# ---------------------------------------------------------------------------
# Analysis of one completed dataset
# ---------------------------------------------------------------------------


def _design(df, scenario):
    response, predictors = scenario.regression
    cols = [np.ones(len(df))]
    names = ["(Intercept)"]
    for p in predictors:
        if p in scenario.categorical:
            values = df[p].astype(object).to_numpy()
            unknown = set(pd.unique(values)) - set(G_LEVELS)
            if unknown:
                raise ValueError(f"column {p!r} has values outside {G_LEVELS}: {sorted(map(str, unknown))}")
            for level in G_LEVELS[1:]:
                cols.append((values == level).astype(float))
                names.append(f"{p}{level}")
        else:
            cols.append(df[p].to_numpy(dtype=float))
            names.append(p)
    return np.column_stack(cols), df[response].to_numpy(dtype=float), names


def ols(x, y):
    """OLS by QR: coefficients, squared standard errors, residual df."""
    n, p = x.shape
    q, r = np.linalg.qr(x)
    coef = np.linalg.solve(r, q.T @ y)
    resid = y - x @ coef
    df_resid = n - p
    sigma2 = resid @ resid / df_resid
    rinv = np.linalg.solve(r, np.eye(p))
    var = sigma2 * np.sum(rinv**2, axis=1)  # diag(R^-1 R^-T)
    return coef, var, df_resid


def analyse(df, scenario):
    """Analyse one completed dataset.

    Returns a DataFrame with columns ``term, estimate, variance, dfcom`` in
    the order of ``scenario.terms``. Raises ``ValueError`` if a used value is
    missing or a categorical value is unknown.
    """
    used = list(scenario.columns)
    if df[used].isna().any().any():
        bad = df[used].columns[df[used].isna().any()].tolist()
        raise ValueError(f"completed data still contain missing values in {bad}")
    n = len(df)
    rows = {}
    x, y, names = _design(df, scenario)
    coef, var, df_resid = ols(x, y)
    for name, c, v in zip(names, coef, var):
        rows[name] = (c, v, float(df_resid))
    for col in scenario.means:
        values = df[col].to_numpy(dtype=float)
        rows[f"mean({col})"] = (values.mean(), values.var(ddof=1) / n, float(n - 1))
    for term, (col, level) in scenario.proportions.items():
        p = float(np.mean(df[col].astype(object).to_numpy() == level))
        rows[term] = (p, p * (1.0 - p) / n, np.inf)
    missing = set(scenario.terms) - set(rows)
    if missing:
        raise RuntimeError(f"analysis did not produce terms {sorted(missing)}")
    return pd.DataFrame(
        [(t, *rows[t]) for t in scenario.terms],
        columns=["term", "estimate", "variance", "dfcom"],
    )


def read_dataset(path, scenario):
    """Read one CSV written by ``study.py generate`` with the right dtypes."""
    df = pd.read_csv(path, keep_default_na=False, na_values=[""])
    for col in scenario.columns:
        if col in scenario.categorical:
            df[col] = df[col].astype(object).where(df[col].notna(), np.nan)
        else:
            df[col] = df[col].astype(float)
    return df[list(scenario.columns)]
