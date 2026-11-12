"""Write the L3 summary table and report.

Run from the repository root, after ``study.py`` and ``run_r.R``::

    uv run python simulations/report.py [--work-dir simulations/work] [--out simulations/results]

Writes ``simulations/results/summary.csv`` (one row per scenario, method, term
and metric: Python and R values with Monte Carlo errors, the paired difference,
its error, z and the flag) and ``simulations/results/REPORT.md``.
"""

from __future__ import annotations

import argparse
import json
import re
import time
from pathlib import Path

import numpy as np
import pandas as pd

import metrics as mx

HERE = Path(__file__).resolve().parent

#: Methods that leave out the uncertainty of the imputation model (and, for
#: norm.predict and mean, the residual noise): their intervals cover below the
#: nominal level in both engines by design.
IMPROPER_METHODS = ("norm.nob", "norm.predict", "mean")
IMPROPER_NOTE = (
    "`norm.nob`, `norm.predict` and `mean` are improper imputation methods: they ignore the "
    "uncertainty of the imputation model (`norm.predict` and `mean` also the residual noise), so "
    "their intervals cover below 95% in both engines, by design. The pass rule is the same as for "
    "every cell: mice-py matches R."
)

# Failures whose cause is known. (engine, method or None, error class, regex
# on the message, label). The first match wins.
KNOWN_FAILURES = [
    ("python", "pmm", "AttributeError", r"has no attribute 'values'", "B1: pmm crashes on a categorical target"),
    ("python", None, "ValueError", r"Predictors must be numeric", "B2: categorical predictor rejected"),
    (
        "python",
        "midastouch",
        "ValueError",
        r"Target y must be numeric",
        "midas does not take a categorical target (R's midastouch fails too)",
    ),
    (
        "r",
        "midastouch",
        "simpleError",
        r"analysis: completed data still contain missing values",
        "R midastouch imputes factor codes; R sets them to NA ('invalid factor level')",
    ),
]


def known_cause(engine, method, error_class, message):
    for eng, mth, cls, pattern, label in KNOWN_FAILURES:
        if eng == engine and (mth is None or mth == method) and cls == error_class and re.search(pattern, str(message)):
            return label
    return "unknown: investigate (NEW-?)"


# ---------------------------------------------------------------------------
# Formatting
# ---------------------------------------------------------------------------


def _digits(metric, value, se):
    fixed = {"pct_bias": 1, "coverage": 3}
    if metric in fixed:
        return fixed[metric]
    # Four decimals, more when value and SE are both small (B and T), so
    # that the larger of the two shows three significant digits.
    scale = max(abs(value), se if se is not None and np.isfinite(se) else 0.0)
    if 0 < scale < 0.01:
        return min(8, 2 - int(np.floor(np.log10(scale))))
    return 4


def fmt(value, se, metric):
    if value is None or not np.isfinite(value):
        return "–"
    d = _digits(metric, value, se)
    text = f"{value:.{d}f}"
    if se is not None and np.isfinite(se):
        text += f" ({se:.{d}f})"
    return text


def fmt_z(z, flagged):
    if z is None or (isinstance(z, float) and np.isnan(z)):
        return "–"
    text = f"{z:+.2f}" if np.isfinite(z) else ("+inf" if z > 0 else "-inf")
    return f"**{text}**" if flagged else text


def md_table(header, rows):
    out = ["| " + " | ".join(header) + " |", "|" + "|".join("---" for _ in header) + "|"]
    out += ["| " + " | ".join(str(c) for c in row) + " |" for row in rows]
    return "\n".join(out)


# ---------------------------------------------------------------------------
# Sections
# ---------------------------------------------------------------------------


def failures_table(status):
    if status.empty:
        return pd.DataFrame()
    rows = []
    for (engine, sid, method), g in status.groupby(["engine", "scenario", "method"], sort=True):
        failed = g[g["status"] != "ok"]
        if failed.empty:
            continue
        for (cls, msg), h in failed.groupby(["error_class", "error_message"], sort=False):
            rows.append(
                dict(
                    scenario=sid,
                    method=method,
                    engine=engine,
                    failed=len(h),
                    attempted=len(g),
                    error_class=cls,
                    message=str(msg)[:120],
                    cause=known_cause(engine, method, cls, msg),
                )
            )
    return pd.DataFrame(rows)


def events_table(status):
    if status.empty:
        return pd.DataFrame()
    st = status.copy()
    st["n_events"] = pd.to_numeric(st["n_events"], errors="coerce")
    st["n_warnings"] = pd.to_numeric(st["n_warnings"], errors="coerce").fillna(0)
    rows = []
    for (engine, sid, method), g in st.groupby(["engine", "scenario", "method"], sort=True):
        ev = int((g["n_events"] > 0).sum())
        wa = int((g["n_warnings"] > 0).sum())
        if ev or wa:
            texts = sorted(set(t for t in g["events"].fillna("").tolist() + g["warnings"].fillna("").tolist() if t))
            rows.append(
                dict(
                    scenario=sid,
                    method=method,
                    engine=engine,
                    reps_with_logged_events=ev if engine == "r" else "n/a",
                    reps_with_warnings=wa,
                    text="; ".join(texts)[:200],
                )
            )
    return pd.DataFrame(rows)


def timing_table(status):
    st = status[status["status"] == "ok"].copy()
    st["seconds"] = pd.to_numeric(st["seconds"], errors="coerce")
    st = st[st["method"] != "complete"]
    if st.empty:
        return pd.DataFrame()
    t = st.groupby(["scenario", "method", "engine"])["seconds"].agg(["mean", "count"]).reset_index()
    return t.pivot_table(index=["scenario", "method"], columns="engine", values="mean").reset_index()


def extrapolate(timing, cells, workers):
    """CPU and wall time of the full study from the measured seconds per replication.

    ``cells`` are the manifest's cells with their full replication counts;
    ``workers`` maps an engine to its number of processes. Measured seconds
    come from runs with all processes busy, so they include contention; wall
    time is CPU time / processes. A cell that was not measured (or failed
    throughout) takes the slowest measured scenario of the same method (or
    the slowest cell of all) and is listed as a guess.
    """
    if timing.empty:
        return None
    out = {}
    for engine in ("python", "r"):
        if engine not in timing:
            continue
        per = timing.set_index(["scenario", "method"])[engine]
        total = 0.0
        guessed = []
        for cell in cells:
            key = (cell["scenario"], cell["method"])
            if key in per.index and np.isfinite(per[key]):
                sec = per[key]
            else:
                same = per[[k for k in per.index if k[1] == cell["method"]]].dropna()
                sec = same.max() if len(same) else per.dropna().max()
                guessed.append(f"{key[0]}/{key[1]}")
            total += cell["reps"] * sec
        n = max(int(workers.get(engine) or 1), 1)
        out[engine] = dict(cpu_seconds=total, wall_seconds=total / n, workers=n, guessed=guessed)
    return out


def _cell_label(method):
    return f"{method}¹" if method in IMPROPER_METHODS else method


def _columns_text(cell):
    if cell.get("columns") is None:
        return "no `method` argument (each engine's default by column type)"
    return ", ".join(f"{c}: {m}" for c, m in cell["columns"].items())


def _sci(x):
    return "–" if x is None or not np.isfinite(x) else f"{x:.2e}"


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def build_report(work_dir, out_dir):
    results_dir = work_dir / "results"
    manifest_path = work_dir / "data" / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    summary, pooled, metrics, paired = mx.summarise(results_dir, manifest_path)
    status = mx.load_status(results_dir)
    cells = mx.load_cells(manifest_path)
    identity = mx.identity_cells(cells)

    meta_py = _read_json(results_dir / "python_meta.json")
    meta_r = _read_json(results_dir / "r_meta.json")

    out_dir.mkdir(parents=True, exist_ok=True)
    order = {sid: i for i, sid in enumerate(manifest["scenarios"])}
    # Methods in the order of their first cell; the 0.2 methods come first
    methods = list(dict.fromkeys(method for _, method in cells))
    method_order = {m: i for i, m in enumerate([*methods, "complete"])}
    metric_order = {m: i for i, m in enumerate(mx.METRICS)}
    term_order = {(sid, t): i for sid, spec in manifest["scenarios"].items() for i, t in enumerate(spec["terms"])}
    summary = (
        summary.assign(
            _s=summary["scenario"].map(order),
            _m=summary["method"].map(method_order).fillna(len(method_order)),
            _t=[term_order.get((s, t), 99) for s, t in zip(summary["scenario"], summary["term"])],
            _x=summary["metric"].map(metric_order),
        )
        .sort_values(["_s", "_m", "_t", "_x"])
        .drop(columns=["_s", "_m", "_t", "_x"])
    )
    cols = [
        "scenario",
        "method",
        "term",
        "truth",
        "metric",
        "python",
        "python_mcse",
        "python_reps",
        "r",
        "r_mcse",
        "r_reps",
        "diff",
        "diff_se",
        "z",
        "flagged",
        "n_pairs",
    ]
    for c in cols:
        if c not in summary:
            summary[c] = np.nan
    summary = summary[cols]
    summary.to_csv(out_dir / "summary.csv", index=False, float_format="%.6g")

    lines = []
    w = lines.append
    w("# L3 simulation study: mice-py against R mice")
    w("")
    w(
        f"Generated {time.strftime('%Y-%m-%d %H:%M')} by `simulations/report.py` from "
        "`simulations/work/results/`. Method and formulas: `simulations/README.md` and "
        "`simulations/metrics.py`."
    )
    w("")
    if meta_py and not meta_py.get("seed_supported", True):
        w(
            "> **Not valid for inference (finding C1).** The library had no `seed` argument when "
            "the Python side ran: chain j is always seeded with 42 + j, so every replication "
            "reuses the same random streams. The replications are not independent and the Monte "
            "Carlo errors, z-scores and flags below do not mean what they say. This run checks "
            "the pipeline only."
        )
        w("")

    reps_info = _plan_summary(meta_py, meta_r)
    w("## Run")
    w("")
    w(
        md_table(
            ["", "Python", "R"],
            [
                [
                    "software",
                    f"mice-py {meta_py.get('library_version', '?')}, Python {meta_py.get('python', '?')}, "
                    f"NumPy {meta_py.get('numpy', '?')}"
                    if meta_py
                    else "–",
                    f"{meta_r.get('r', '?')}, mice {meta_r.get('mice', '?')}, ranger {meta_r.get('ranger', '?')}, "
                    f"rpart {meta_r.get('rpart', '?')}"
                    if meta_r
                    else "–",
                ],
                [
                    "seeds",
                    "per replication" if meta_py.get("seed_supported") else "hard-coded (C1)",
                    "per replication (`mice(seed = )`)",
                ],
                ["replications", reps_info.get("python", "–"), reps_info.get("r", "–")],
                ["processes", meta_py.get("jobs", "–"), meta_r.get("jobs", "–")],
            ],
        )
    )
    w("")
    w("Scenarios:")
    w("")
    w(
        md_table(
            ["ID", "data", "n", "missingness", "maxit", "terms (truth)"],
            [
                [
                    sid,
                    spec["description"],
                    spec["n"],
                    spec["missingness"],
                    spec["maxit"],
                    ", ".join(f"{t} = {v:.4g}" for t, v in spec["truth"].items()),
                ]
                for sid, spec in manifest["scenarios"].items()
            ],
        )
    )
    w("")
    categorical = [
        f"`{c}` ({' < '.join(f['levels']) if f['ordered'] else ', '.join(f['levels'])}, "
        f"{'ordered' if f['ordered'] else 'unordered'}) in {sid}"
        for sid, spec in manifest["scenarios"].items()
        for c, f in spec.get("categorical", {}).items()
    ]
    if categorical:
        w(
            "Categorical columns: "
            + "; ".join(categorical)
            + ". Python reads an unordered column as object dtype and an ordered one as an ordered "
            "`pd.Categorical`; R reads factors with these levels. The analysis uses treatment dummies "
            "for both (R analyses an unordered copy of an ordered factor)."
        )
        w("")

    # Cells
    w("## Cells")
    w("")
    ran = status[status["method"] != "complete"].groupby(["scenario", "method", "engine"]).size()
    w(
        "A study method on a scenario. Replications: planned for the full study, and run here by "
        "Python and R. Rule `z`: every paired difference |z| ≤ "
        f"{mx.Z_FLAG}; rule `identity`: see [Identity rule](#identity-rule)."
    )
    w("")
    w(
        md_table(
            ["method", "scenario", "planned", "Python", "R", "rule", "method of each incomplete column"],
            [
                [
                    _cell_label(method),
                    sid,
                    c["reps"],
                    int(ran.get((sid, method, "python"), 0)) or "–",
                    int(ran.get((sid, method, "r"), 0)) or "–",
                    c.get("rule", "z"),
                    _columns_text(c),
                ]
                for (sid, method), c in sorted(
                    cells.items(), key=lambda kv: (method_order[kv[0][1]], order.get(kv[0][0], 99))
                )
            ],
        )
    )
    w("")
    w(f"¹ {IMPROPER_NOTE}")
    w("")

    # Harness
    w("## Harness checks")
    w("")
    hc = mx.harness_coverage(metrics)
    if hc.empty:
        w("No complete-data rows.")
    else:
        bad = hc[~hc["within"]]
        w(
            "Coverage of the complete-data analysis (pseudo-method `complete`) must lie within "
            "0.95 ± 3 nominal Monte Carlo errors. "
            + ("All cells pass." if bad.empty else f"**{len(bad)} cell(s) outside the band.**")
        )
        w("")
        wide = hc.pivot_table(
            index=["scenario", "term"],
            columns="engine",
            values=["value", "band", "within", "reps"],
            aggfunc="first",
            sort=False,
        )
        # pivot_table does not keep the term order: restore the scenario's
        wide = wide.loc[sorted(wide.index, key=lambda k: (order.get(k[0], 99), term_order.get(k, 99)))]
        rows = []
        for (sid, term), r in wide.iterrows():
            entries = []
            for engine in mx.ENGINES:
                if ("value", engine) in r.index and pd.notna(r[("value", engine)]):
                    ok = "" if r[("within", engine)] else " **outside**"
                    entries.append(
                        f"{r[('value', engine)]:.3f} ± {r[('band', engine)]:.3f} ({int(r[('reps', engine)])} reps){ok}"
                    )
                else:
                    entries.append("–")
            rows.append([sid, term, *entries])
        w(md_table(["scenario", "term", "Python: coverage ± band", "R: coverage ± band"], rows))
    w("")
    ident = mx.harness_identity(pooled)
    if ident:
        w(
            f"Identical data, identical analysis: over {ident['pairs']} complete-data rows the largest "
            f"|qbar Python − qbar R| is {ident['max_abs_qbar']:.2e}, the largest relative ubar "
            f"difference {ident['max_rel_ubar']:.2e}; dfcom equal: {'yes' if ident['dfcom_equal'] else '**no**'}."
        )
        w("")

    # Methods used
    w("## Methods used")
    w("")
    mc = mx.methods_check(status, cells, {sid: spec["columns"] for sid, spec in manifest["scenarios"].items()})
    if mc.empty:
        w("No method records (status files without a `methods` column).")
    else:
        w(
            "Each engine records the method of every column as it ran it (Python `MICE.method` in R's "
            "names, R `imp$method`; empty for a column that is not imputed). Only replications that "
            "succeeded in both engines are compared."
        )
        w("")
        is_default = [
            cells.get((r.scenario, r.method), {}).get("expected_methods") is not None for r in mc.itertuples()
        ]
        default = mc[is_default]
        if not default.empty:
            w("Cells without a `method` argument, where each engine chooses by column type (R's `make.method`):")
            w("")
            w(
                md_table(
                    ["scenario", "method", "expected", "Python", "R", "Python = R", "verdict"],
                    [
                        [
                            r.scenario,
                            r.method,
                            f"`{r.expected}`",
                            r.python,
                            r.r,
                            f"{r.equal} / {r.pairs}",
                            "pass" if r.passed else "**fail**",
                        ]
                        for r in default.itertuples()
                    ],
                )
            )
            w("")
        others = mc[~mc.index.isin(default.index)]
        bad = others[~others["passed"]]
        if not others.empty:
            if bad.empty:
                w(
                    f"In the other {len(others)} cells Python and R recorded the same methods in every "
                    f"paired replication ({int(others['pairs'].sum())} pairs)."
                )
            else:
                w(f"**{len(bad)} cell(s) where Python's methods differ from R's:**")
                w("")
                w(
                    md_table(
                        ["scenario", "method", "Python", "R", "Python = R"],
                        [[r.scenario, r.method, r.python, r.r, f"{r.equal} / {r.pairs}"] for r in bad.itertuples()],
                    )
                )
            w("")

    # Identity rule
    w("## Identity rule")
    w("")
    ic = mx.identity_check(pooled, identity)
    if ic.empty:
        w("No cell is judged by the identity rule.")
    else:
        w(
            "Deterministic methods on S4 (one incomplete column, maxit = 1) draw nothing, so the paired "
            "z statistic is 0/0 and is not computed. Such a cell passes when the largest |qbar Python − "
            f"qbar R| over replications and terms is at most {mx.IDENTITY_TOL:g} and B = 0 in both "
            f"engines (to rounding: largest B at most {mx.IDENTITY_TOL**2:g})."
        )
        w("")
        w(
            md_table(
                ["scenario", "method", "pairs × terms", "max |qbar Py − qbar R|", "max B Python", "max B R", "verdict"],
                [
                    [
                        r.scenario,
                        _cell_label(r.method),
                        f"{r.pairs} × {r.terms}",
                        _sci(r.max_abs_qbar),
                        _sci(r.max_b_python),
                        _sci(r.max_b_r),
                        "pass" if r.passed else "**fail**",
                    ]
                    for r in ic.itertuples()
                ],
            )
        )
    w("")

    # Flags
    w(f"## Flagged comparisons (|z| > {mx.Z_FLAG})")
    w("")
    flagged = summary[(summary["flagged"] == True) & (summary["method"] != "complete")]  # noqa: E712
    tested = (summary["method"] != "complete") & summary["metric"].isin(mx.TESTED_METRICS)
    n_tests = int(summary.loc[tested, "z"].notna().sum())
    min_pairs = summary.loc[tested, "n_pairs"].min()
    if pd.notna(min_pairs) and min_pairs < 100:
        w(
            f"Some cells have only {int(min_pairs)} paired replications. The z-scores use a normal "
            "approximation and Monte Carlo errors estimated from the same replications; below about "
            "100 replications they are not meaningful and flags are expected by chance."
        )
        w("")
    if identity:
        w(
            "Cells judged by the identity rule have no z and are not counted: "
            + ", ".join(f"{sid} {method}" for sid, method in sorted(identity))
            + "."
        )
        w("")
    if flagged.empty:
        w(f"None of {n_tests} paired comparisons.")
    else:
        w(f"{len(flagged)} of {n_tests} paired comparisons:")
        w("")
        w(
            md_table(
                ["scenario", "method", "term", "metric", "Python", "R", "Py − R (SE)", "z"],
                [
                    [
                        r.scenario,
                        r.method,
                        r.term,
                        mx.METRIC_LABELS[r.metric],
                        fmt(r.python, None, r.metric),
                        fmt(r.r, None, r.metric),
                        fmt(r.diff, r.diff_se, r.metric),
                        fmt_z(r.z, True),
                    ]
                    for r in flagged.itertuples()
                ],
            )
        )
    w("")

    # Failures
    w("## Failures")
    w("")
    ft = failures_table(status)
    if ft.empty:
        w("No replication failed.")
    else:
        w(
            md_table(
                ["scenario", "method", "engine", "failed / attempted", "exception", "message", "cause"],
                [
                    [
                        r.scenario,
                        r.method,
                        r.engine,
                        f"{r.failed} / {r.attempted}",
                        r.error_class,
                        r.message.replace("|", "\\|"),
                        r.cause,
                    ]
                    for r in ft.itertuples()
                ],
            )
        )
    w("")
    et = events_table(status)
    if not et.empty:
        w("Replications with R `loggedEvents` or with warnings:")
        w("")
        w(
            md_table(
                ["scenario", "method", "engine", "logged events", "warnings", "text"],
                [
                    [
                        r.scenario,
                        r.method,
                        r.engine,
                        r.reps_with_logged_events,
                        r.reps_with_warnings,
                        r.text.replace("|", "\\|"),
                    ]
                    for r in et.itertuples()
                ],
            )
        )
        w("")

    # Timing
    timing = timing_table(status)
    if not timing.empty:
        w("## Timing")
        w("")
        w("Mean seconds per replication (m = 5 imputations plus analysis), measured with all processes busy:")
        w("")
        w(
            md_table(
                ["scenario", "method", "Python", "R"],
                [
                    [
                        r.scenario,
                        r.method,
                        f"{getattr(r, 'python', np.nan):.3f}" if np.isfinite(getattr(r, "python", np.nan)) else "–",
                        f"{getattr(r, 'r', np.nan):.3f}" if np.isfinite(getattr(r, "r", np.nan)) else "–",
                    ]
                    for r in timing.itertuples()
                ],
            )
        )
        w("")
        workers = {"python": meta_py.get("jobs"), "r": meta_r.get("jobs")}
        ext = extrapolate(timing, list(cells.values()), workers)
        if ext:
            n_reps = sorted({c["reps"] for c in cells.values()})
            w(
                f"Extrapolated to the full study (all {len(cells)} cells of {len(manifest['scenarios'])} "
                f"scenarios, {' or '.join(map(str, n_reps))} replications each), with as many processes "
                "as this run used:"
            )
            w("")
            for engine, e in ext.items():
                g = ""
                if e["guessed"]:
                    g = (
                        "; cells not measured, taken from the slowest measured scenario of the "
                        f"same method: {', '.join(e['guessed'])}"
                    )
                w(
                    f"- {engine}: about {e['wall_seconds'] / 60:.0f} min with {e['workers']} processes "
                    f"({e['cpu_seconds'] / 3600:.1f} CPU hours){g}."
                )
            w("")

    # Per scenario and method
    w("## Results per scenario and method")
    w("")
    w(
        "Cells: value (Monte Carlo SE). Py − R is the paired difference over replications where "
        "both engines succeeded, with its SE; z = difference / SE; bold z is flagged. Cells of the "
        "identity rule have no z."
    )
    w("")
    w(f"¹ {IMPROPER_NOTE}")
    w("")
    ident_by_cell = {(r.scenario, r.method): r for r in ic.itertuples()} if not ic.empty else {}
    for sid, spec in manifest["scenarios"].items():
        sub = summary[(summary["scenario"] == sid) & (summary["method"] != "complete")]
        if sub.empty:
            continue
        w(f"### {sid}: {spec['description']}; {spec['missingness']}; n = {spec['n']}, maxit = {spec['maxit']}")
        w("")
        for method in sorted(sub["method"].unique(), key=lambda m: method_order.get(m, len(method_order))):
            sm = sub[sub["method"] == method]
            np_ = _int_or_dash(sm["python_reps"].max())
            nr_ = _int_or_dash(sm["r_reps"].max())
            npair = _int_or_dash(sm["n_pairs"].max())
            w(f"#### {sid} {_cell_label(method)} (Python {np_} reps, R {nr_} reps, {npair} pairs)")
            w("")
            cell = cells.get((sid, method), {})
            if cell:
                w(f"Methods: {_columns_text(cell)}.")
                w("")
            if (sid, method) in ident_by_cell:
                r = ident_by_cell[(sid, method)]
                w(
                    f"Identity rule: max |qbar Python − qbar R| = {_sci(r.max_abs_qbar)}, max B Python "
                    f"{_sci(r.max_b_python)}, R {_sci(r.max_b_r)}: {'pass' if r.passed else '**fail**'}."
                )
                w("")
            rows = []
            for r in sm.itertuples():
                if r.metric == "estimate":
                    continue
                if r.metric == "pct_bias" and r.truth == 0:
                    continue
                rows.append(
                    [
                        f"{r.term} = {r.truth:.4g}" if r.metric == "bias" else r.term,
                        mx.METRIC_LABELS[r.metric],
                        fmt(r.python, r.python_mcse, r.metric),
                        fmt(r.r, r.r_mcse, r.metric),
                        fmt(r.diff, r.diff_se, r.metric),
                        fmt_z(r.z, bool(r.flagged) if r.flagged == r.flagged else False),
                    ]
                )
            w(md_table(["term", "metric", "Python", "R", "Py − R", "z"], rows))
            w("")
    (out_dir / "REPORT.md").write_text("\n".join(lines) + "\n")
    return summary


def _int_or_dash(x):
    return "–" if x is None or not np.isfinite(x) else str(int(x))


def _plan_summary(meta_py, meta_r):
    out = {}
    for engine, meta in (("python", meta_py), ("r", meta_r)):
        plan = (meta or {}).get("plan") or {}
        if plan:
            counts = sorted(set(int(v) for k, v in plan.items() if not k.endswith("/complete")))
            out[engine] = ", ".join(map(str, counts)) + " per cell" if counts else "–"
    return out


def _read_json(path):
    return json.loads(path.read_text()) if path.exists() else {}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--work-dir", type=Path, default=HERE / "work")
    parser.add_argument("--out", type=Path, default=HERE / "results")
    args = parser.parse_args(argv)
    summary = build_report(args.work_dir.resolve(), args.out.resolve())
    n_flag = int((summary["flagged"] == True).sum())  # noqa: E712
    print(f"wrote {args.out / 'summary.csv'} ({len(summary)} rows) and {args.out / 'REPORT.md'}; {n_flag} flagged")


if __name__ == "__main__":
    main()
