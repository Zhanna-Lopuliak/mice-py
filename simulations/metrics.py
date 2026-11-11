"""Metrics of the L3 study: one routine for both engines.

The inputs are the per-replication pooling ingredients written by
``study.py run-python`` and ``run_r.R``: ``qbar, ubar, b, m, n, dfcom`` per
scenario, method, replication and term. Pooling is done here, the same way for
both engines, so that the study judges imputation and not pooling. It does not
call the library's pooling code.

Rubin's rules with the Barnard-Rubin degrees of freedom, per term::

    t      = ubar + (1 + 1/m) b
    lambda = (1 + 1/m) b / t
    df     = (m - 1) tmp / ((dfcom + 3)(m - 1) + lambda^2 tmp),
             tmp = (1 - lambda)(1 + dfcom) dfcom;   (m - 1) / lambda^2 if dfcom = inf
    interval qbar -/+ t_{0.975, df} sqrt(t)

This is ``mice:::barnard.rubin`` of mice 3.19.0 (checked against ``pool()``
on R's own output to 1e-11). With m = 1 (pseudo-method ``complete``) t = ubar
and df = dfcom.

Per scenario, method, term and engine: mean estimate, bias, percent bias,
coverage, mean interval width, RMSE, mean b and mean t, each with its Monte
Carlo standard error (MCSE). Then paired Python-minus-R differences over the
replications where both engines succeeded, with z = difference / MCSE, flagged
when |z| > 4.2 (bias, coverage, width, RMSE, B and T are tested; the mean
estimate and percent bias differences are the bias difference rescaled). The
z-scores rest on a normal approximation and mean little below about 100
replications.

Monte Carlo standard errors (R = number of replications):

- mean estimate, bias, width, b, t: sd / sqrt(R); percent bias: 100 MCSE / |truth|;
- coverage: sqrt(CR (1 - CR) / R);
- RMSE: sd((qbar - truth)^2) / sqrt(R) / (2 RMSE) (delta method);
- paired coverage difference: sqrt(n10 + n01 - (n10 - n01)^2 / R) / R, with
  n10, n01 the discordant counts;
- other paired differences: sd(difference) / sqrt(R); for RMSE the paired
  difference of squared errors divided by (RMSE_py + RMSE_r).
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

Z_FLAG = 4.2
LEVEL = 0.95
ENGINES = ("python", "r")
#: Metrics in report order; ``estimate`` is in summary.csv only.
METRICS = ("estimate", "bias", "pct_bias", "coverage", "width", "rmse", "b", "t")
#: Metrics whose paired difference is tested and can be flagged. The paired
#: differences of ``estimate`` and ``pct_bias`` are the bias difference
#: rescaled, so they carry the same z and are not counted again.
TESTED_METRICS = ("bias", "coverage", "width", "rmse", "b", "t")
METRIC_LABELS = {
    "estimate": "mean estimate",
    "bias": "bias",
    "pct_bias": "% bias",
    "coverage": "coverage",
    "width": "CI width",
    "rmse": "RMSE",
    "b": "mean B",
    "t": "mean T",
}


# ---------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------


def load_results(results_dir):
    """All per-replication result rows of both engines, with an ``engine`` column."""
    frames = []
    for engine in ENGINES:
        for path in sorted(Path(results_dir).glob(f"{engine}_S*_*.csv")):
            if path.name.endswith("_status.csv"):
                continue
            df = pd.read_csv(path, keep_default_na=False, na_values=["", "NA", "NaN", "nan"])
            if len(df):
                df.insert(0, "engine", engine)
                frames.append(df)
    if not frames:
        raise FileNotFoundError(f"no result files in {results_dir}")
    out = pd.concat(frames, ignore_index=True)
    for col in ("qbar", "ubar", "b", "dfcom"):
        out[col] = pd.to_numeric(out[col].replace({"Inf": "inf", "-Inf": "-inf"}), errors="raise")
    return out


def load_status(results_dir):
    """All per-replication status rows of both engines."""
    frames = []
    for engine in ENGINES:
        for path in sorted(Path(results_dir).glob(f"{engine}_S*_*_status.csv")):
            df = pd.read_csv(path, keep_default_na=False, na_values=[""])
            df.insert(0, "engine", engine)
            frames.append(df)
    if not frames:
        return pd.DataFrame(columns=["engine", "scenario", "method", "rep", "status"])
    return pd.concat(frames, ignore_index=True)


def load_truth(manifest_path):
    """``{(scenario, term): truth}`` from the data manifest."""
    manifest = json.loads(Path(manifest_path).read_text())
    return {
        (sid, term): float(value)
        for sid, spec in manifest["scenarios"].items()
        for term, value in spec["truth"].items()
    }


# ---------------------------------------------------------------------------
# Pooling
# ---------------------------------------------------------------------------


def barnard_rubin(m, b, t, dfcom):
    """Barnard-Rubin degrees of freedom (vectorised; ``dfcom`` may be inf)."""
    m = np.asarray(m, dtype=float)
    b = np.asarray(b, dtype=float)
    t = np.asarray(t, dtype=float)
    dfcom = np.asarray(dfcom, dtype=float)
    lam = (1.0 + 1.0 / m) * b / t
    with np.errstate(divide="ignore"):
        dfold = (m - 1.0) / lam**2  # inf when b = 0
    finite = np.isfinite(dfcom)
    dfc = np.where(finite, dfcom, 1.0)  # placeholder where infinite
    tmp = (1.0 - lam) * (1.0 + dfc) * dfc
    df_finite = (m - 1.0) * tmp / ((dfc + 3.0) * (m - 1.0) + lam**2 * tmp)
    return np.where(finite, df_finite, dfold)


def pool(frame, truth):
    """Add pooled quantities and per-replication indicators to ``frame``.

    Adds ``truth, t, df, se, lower, upper, width, covered, sqerr``.
    """
    f = frame.copy()
    f["truth"] = [truth[(s, term)] for s, term in zip(f["scenario"], f["term"])]
    m = f["m"].to_numpy(dtype=float)
    single = m <= 1
    t = f["ubar"].to_numpy() + np.where(single, 0.0, (1.0 + 1.0 / m) * f["b"].to_numpy())
    df = np.where(
        single,
        f["dfcom"].to_numpy(dtype=float),
        barnard_rubin(np.where(single, 2.0, m), f["b"], t, f["dfcom"]),
    )
    q = stats.t.ppf(0.5 + LEVEL / 2.0, df)
    se = np.sqrt(t)
    f["t"] = t
    f["df"] = df
    f["se"] = se
    f["lower"] = f["qbar"] - q * se
    f["upper"] = f["qbar"] + q * se
    f["width"] = f["upper"] - f["lower"]
    f["covered"] = (f["lower"] <= f["truth"]) & (f["truth"] <= f["upper"])
    f["sqerr"] = (f["qbar"] - f["truth"]) ** 2
    return f


# ---------------------------------------------------------------------------
# Per-engine metrics
# ---------------------------------------------------------------------------


def _mcse_mean(x):
    x = np.asarray(x, dtype=float)
    return np.std(x, ddof=1) / np.sqrt(len(x)) if len(x) > 1 else np.nan


def engine_metrics(pooled):
    """Long table: engine, scenario, method, term, truth, metric, value, mcse, reps."""
    rows = []
    keys = ["engine", "scenario", "method", "term"]
    for (engine, sid, method, term), g in pooled.groupby(keys, sort=False):
        r = len(g)
        truth = g["truth"].iloc[0]
        q = g["qbar"].to_numpy()
        mean_q = q.mean()
        se_q = _mcse_mean(q)
        cr = g["covered"].mean()
        mse = g["sqerr"].mean()
        rmse = np.sqrt(mse)
        se_rmse = (np.std(g["sqerr"], ddof=1) / np.sqrt(r) / (2 * rmse)) if r > 1 and rmse > 0 else np.nan
        pct = 100.0 * (mean_q - truth) / abs(truth) if truth != 0 else np.nan
        se_pct = 100.0 * se_q / abs(truth) if truth != 0 else np.nan
        values = {
            "estimate": (mean_q, se_q),
            "bias": (mean_q - truth, se_q),
            "pct_bias": (pct, se_pct),
            "coverage": (cr, np.sqrt(cr * (1 - cr) / r)),
            "width": (g["width"].mean(), _mcse_mean(g["width"])),
            "rmse": (rmse, se_rmse),
            "b": (g["b"].mean(), _mcse_mean(g["b"])),
            "t": (g["t"].mean(), _mcse_mean(g["t"])),
        }
        for metric in METRICS:
            v, s = values[metric]
            rows.append(dict(engine=engine, scenario=sid, method=method, term=term,
                             truth=truth, metric=metric, value=v, mcse=s, reps=r))
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Paired Python-minus-R differences
# ---------------------------------------------------------------------------


def _z(diff, se):
    if not np.isfinite(diff):
        return np.nan
    if se == 0 or not np.isfinite(se):
        return 0.0 if diff == 0 else np.copysign(np.inf, diff)
    return diff / se


def paired_differences(pooled):
    """Long table: scenario, method, term, metric, diff, diff_se, z, flagged, n_pairs."""
    cols = ["scenario", "method", "term", "rep", "truth", "qbar", "width", "covered", "sqerr", "b", "t"]
    py = pooled.loc[pooled["engine"] == "python", cols]
    r = pooled.loc[pooled["engine"] == "r", cols]
    both = py.merge(r, on=["scenario", "method", "term", "rep", "truth"], suffixes=("_py", "_r"))
    rows = []
    for (sid, method, term), g in both.groupby(["scenario", "method", "term"], sort=False):
        n = len(g)
        truth = g["truth"].iloc[0]
        out = {}
        d = g["qbar_py"] - g["qbar_r"]
        out["estimate"] = (d.mean(), _mcse_mean(d))
        out["bias"] = out["estimate"]
        if truth != 0:
            out["pct_bias"] = (100 * d.mean() / abs(truth), 100 * _mcse_mean(d) / abs(truth))
        else:
            out["pct_bias"] = (np.nan, np.nan)
        n10 = int((g["covered_py"] & ~g["covered_r"]).sum())
        n01 = int((~g["covered_py"] & g["covered_r"]).sum())
        cov_se = np.sqrt(max(n10 + n01 - (n10 - n01) ** 2 / n, 0.0)) / n
        out["coverage"] = ((n10 - n01) / n, cov_se)
        for metric in ("width", "b", "t"):
            d = g[f"{metric}_py"] - g[f"{metric}_r"]
            out[metric] = (d.mean(), _mcse_mean(d))
        rmse_py = np.sqrt(g["sqerr_py"].mean())
        rmse_r = np.sqrt(g["sqerr_r"].mean())
        d = g["sqerr_py"] - g["sqerr_r"]
        denom = rmse_py + rmse_r
        out["rmse"] = (rmse_py - rmse_r, _mcse_mean(d) / denom if denom > 0 else np.nan)
        for metric in METRICS:
            diff, se = out[metric]
            z = _z(diff, se)
            # complete: identical data on both sides, judged by harness_identity().
            flagged = (method != "complete" and metric in TESTED_METRICS
                       and bool(np.isinf(z) or (np.isfinite(z) and abs(z) > Z_FLAG)))
            rows.append(dict(scenario=sid, method=method, term=term, metric=metric,
                             diff=diff, diff_se=se, z=z, flagged=flagged, n_pairs=n))
    return pd.DataFrame(rows, columns=["scenario", "method", "term", "metric", "diff",
                                       "diff_se", "z", "flagged", "n_pairs"])


# ---------------------------------------------------------------------------
# Harness checks
# ---------------------------------------------------------------------------


def harness_coverage(metrics):
    """Coverage of the complete-data analysis against 0.95 -/+ 3 MCSE.

    The MCSE here is the nominal one, sqrt(0.95 * 0.05 / R), so that a run
    with observed coverage 1 does not get a zero-width band.
    """
    c = metrics[(metrics["method"] == "complete") & (metrics["metric"] == "coverage")].copy()
    c["band"] = 3 * np.sqrt(LEVEL * (1 - LEVEL) / c["reps"])
    c["within"] = (c["value"] - LEVEL).abs() <= c["band"]
    return c[["engine", "scenario", "term", "value", "band", "reps", "within"]]


def harness_identity(results):
    """Largest Python-minus-R difference of the complete-data analysis.

    Both engines analyse identical data with OLS, so qbar and ubar must agree
    to rounding error.
    """
    c = results[results["method"] == "complete"]
    py = c[c["engine"] == "python"]
    r = c[c["engine"] == "r"]
    both = py.merge(r, on=["scenario", "term", "rep"], suffixes=("_py", "_r"))
    if both.empty:
        return None
    rel = lambda a, b: (a - b).abs() / np.maximum(b.abs(), 1e-300)  # noqa: E731
    return {
        "pairs": len(both),
        "max_abs_qbar": float((both["qbar_py"] - both["qbar_r"]).abs().max()),
        "max_rel_ubar": float(rel(both["ubar_py"], both["ubar_r"]).max()),
        "dfcom_equal": bool(np.all(np.isclose(both["dfcom_py"], both["dfcom_r"]))),  # inf == inf
    }


# ---------------------------------------------------------------------------
# Everything at once
# ---------------------------------------------------------------------------


def summarise(results_dir, manifest_path):
    """Return ``(summary, pooled, metrics, paired)``.

    ``summary`` has one row per scenario, method, term and metric with the
    Python and R values, their MCSEs, the paired difference, its MCSE, z and
    the flag.
    """
    results = load_results(results_dir)
    truth = load_truth(manifest_path)
    pooled = pool(results, truth)
    metrics = engine_metrics(pooled)
    paired = paired_differences(pooled)

    keys = ["scenario", "method", "term", "metric"]
    wide = None
    for engine in ENGINES:
        e = metrics[metrics["engine"] == engine][keys + ["truth", "value", "mcse", "reps"]]
        e = e.rename(columns={"value": engine, "mcse": f"{engine}_mcse", "reps": f"{engine}_reps"})
        wide = e if wide is None else wide.merge(e, on=keys + ["truth"], how="outer")
    summary = wide.merge(paired, on=keys, how="left")
    return summary, pooled, metrics, paired
