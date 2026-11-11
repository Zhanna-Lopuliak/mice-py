"""L3 simulation study: generate the datasets and run the Python side.

Run from the repository root::

    uv run python simulations/study.py generate   [--scenarios S1,S4] [--reps 1000]
    uv run python simulations/study.py run-python [--scenarios S1,S4] [--methods pmm,sample]
                                                   [--reps 1000] [--rf-reps 500] [--jobs 8]
    uv run python simulations/study.py all        (generate, then run-python; the default)

Then ``Rscript simulations/run_r.R`` (same options) and
``uv run python simulations/report.py``.

Files (all under ``simulations/work/``, git-ignored):

- ``data/<scenario>/rep_<k>.csv``: incomplete dataset k (1-based);
- ``data/<scenario>/complete/rep_<k>.csv``: the same dataset before deletion;
- ``data/manifest.json``: scenario specifications for ``run_r.R``;
- ``results/python_<scenario>_<method>.csv``: one row per replication and term,
  columns ``scenario, method, rep, term, qbar, ubar, b, m, n, dfcom``;
- ``results/python_<scenario>_<method>_status.csv``: one row per replication,
  ``status`` ("ok" or "failed"), exception class and message, warnings, seconds
  (``n_events``/``events`` are R's ``loggedEvents`` and stay empty here);
- ``results/python_meta.json``: library version, whether seeds were honoured.

``method`` is R's name (``midastouch``, not ``midas``). The pseudo-method
``complete`` analyses the data before deletion (m = 1, b = 0) and validates
the harness; it always runs.
"""

from __future__ import annotations

import os

# One thread per worker: the study parallelises over replications. Must be
# set before NumPy is imported (spawned workers re-import this module).
for _var in (
    "OMP_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "MKL_NUM_THREADS",
    "VECLIB_MAXIMUM_THREADS",
    "NUMEXPR_NUM_THREADS",
):
    os.environ.setdefault(_var, "1")

import argparse  # noqa: E402
import json  # noqa: E402
import multiprocessing as mp  # noqa: E402
import platform  # noqa: E402
import sys  # noqa: E402
import time  # noqa: E402
import warnings  # noqa: E402
from pathlib import Path  # noqa: E402

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from scenarios import SCENARIOS, analyse, read_dataset  # noqa: E402

HERE = Path(__file__).resolve().parent
DEFAULT_WORK = HERE / "work"

METHODS = ("pmm", "midastouch", "cart", "rf", "sample")
COMPLETE = "complete"
M_IMPUTATIONS = 5
DEFAULT_REPS = 1000
DEFAULT_RF_REPS = 500

RESULT_COLUMNS = ["scenario", "method", "rep", "term", "qbar", "ubar", "b", "m", "n", "dfcom"]
STATUS_COLUMNS = [
    "scenario",
    "method",
    "rep",
    "status",
    "error_class",
    "error_message",
    "n_events",
    "events",
    "n_warnings",
    "warnings",
    "seconds",
]


def replication_seed(scenario_index, rep):
    """Imputation seed of one replication; ``run_r.R`` uses the same formula."""
    return scenario_index * 100_000 + rep


# ---------------------------------------------------------------------------
# generate
# ---------------------------------------------------------------------------


def _generate_one(args):
    scenario_id, rep, data_dir = args
    sc = SCENARIOS[scenario_id]
    full, data = sc.generate(rep)
    out = Path(data_dir) / scenario_id
    data.to_csv(out / f"rep_{rep}.csv", index=False, na_rep="")
    full.to_csv(out / "complete" / f"rep_{rep}.csv", index=False, na_rep="")
    return scenario_id, rep, data.isna().mean().to_dict()


def scenario_spec(sc, reps):
    """What ``run_r.R`` needs to know about a scenario (written to the manifest)."""
    response, predictors = sc.regression
    return {
        "id": sc.id,
        "index": sc.index,
        "n": sc.n,
        "maxit": sc.maxit,
        "reps": reps,
        "columns": list(sc.columns),
        "categorical": {c: ["a", "b", "c"] for c in sc.categorical},
        "regression": {"response": response, "predictors": list(predictors)},
        "means": list(sc.means),
        "proportions": {t: {"column": c, "level": lv} for t, (c, lv) in sc.proportions.items()},
        "terms": sc.terms,
        "truth": sc.truth,
        "description": sc.description,
        "missingness": sc.missingness,
    }


def cmd_generate(args):
    data_dir = args.work_dir / "data"
    manifest_path = data_dir / "manifest.json"
    manifest = json.loads(manifest_path.read_text()) if manifest_path.exists() else {"scenarios": {}}
    tasks = []
    for sid in args.scenarios:
        (data_dir / sid / "complete").mkdir(parents=True, exist_ok=True)
        tasks += [(sid, rep, str(data_dir)) for rep in range(1, args.reps + 1)]
    start = time.perf_counter()
    rates = {}
    with _pool(args.jobs) as pool:
        for sid, _rep, rate in pool.imap_unordered(_generate_one, tasks, chunksize=16):
            rates.setdefault(sid, []).append(rate)
    for sid in args.scenarios:
        manifest["scenarios"][sid] = scenario_spec(SCENARIOS[sid], args.reps)
        mean_rate = pd.DataFrame(rates[sid]).mean().round(4).to_dict()
        print(f"{sid}: {args.reps} datasets, mean missing rate per column {mean_rate}")
    manifest["seed_formula"] = "scenario_index * 100000 + rep"
    manifest["generator"] = "numpy.random.default_rng([scenario_index, rep])"
    manifest["numpy"] = np.__version__
    manifest_path.write_text(json.dumps(manifest, indent=2, allow_nan=True) + "\n")
    print(f"wrote {len(tasks)} datasets to {data_dir} in {time.perf_counter() - start:.1f} s")


# ---------------------------------------------------------------------------
# run-python
# ---------------------------------------------------------------------------


def _pool_terms(per_imputation, m):
    """qbar, ubar, b per term from a list of analyse() frames."""
    est = np.array([a["estimate"].to_numpy() for a in per_imputation])
    var = np.array([a["variance"].to_numpy() for a in per_imputation])
    qbar = est.mean(axis=0)
    ubar = var.mean(axis=0)
    b = est.var(axis=0, ddof=1) if m > 1 else np.zeros_like(qbar)
    return qbar, ubar, b


def _run_one(args):
    """One (scenario, method, rep): impute, analyse, return rows and status."""
    scenario_id, method, rep, data_dir = args
    sc = SCENARIOS[scenario_id]
    status = dict(
        scenario=scenario_id,
        method=method,
        rep=rep,
        status="ok",
        error_class="",
        error_message="",
        n_events=np.nan,
        events="",
        n_warnings=0,
        warnings="",
        seconds=np.nan,
    )
    start = time.perf_counter()
    rows = []
    stage = "read"
    try:
        if method == COMPLETE:
            df = read_dataset(Path(data_dir) / scenario_id / "complete" / f"rep_{rep}.csv", sc)
            stage = "analysis"
            analyses, m = [analyse(df, sc)], 1
        else:
            import _adapter  # loaded by _init_worker; 'generate' does not need it

            df = read_dataset(Path(data_dir) / scenario_id / f"rep_{rep}.csv", sc)
            stage = "imputation"
            outcome = _adapter.impute(df, method, M_IMPUTATIONS, sc.maxit, replication_seed(sc.index, rep))
            status["n_warnings"] = len(outcome.events)
            status["warnings"] = " | ".join(sorted(set(outcome.events)))[:500]
            if not outcome.ok:
                status.update(
                    status="failed",
                    error_class=outcome.error_class,
                    error_message=outcome.error_message,
                )
                status["seconds"] = time.perf_counter() - start
                return rows, status
            stage = "analysis"
            analyses, m = [analyse(d, sc) for d in outcome.datasets], len(outcome.datasets)
        qbar, ubar, b = _pool_terms(analyses, m)
        first = analyses[0]
        for i, term in enumerate(first["term"]):
            rows.append(
                dict(
                    scenario=scenario_id,
                    method=method,
                    rep=rep,
                    term=term,
                    qbar=qbar[i],
                    ubar=ubar[i],
                    b=b[i],
                    m=m,
                    n=len(df),
                    dfcom=first["dfcom"].iloc[i],
                )
            )
    except Exception as exc:  # recorded, so one bad replication does not stop the study
        status.update(
            status="failed",
            error_class=type(exc).__name__,
            error_message=f"{stage}: " + " ".join(str(exc).split())[:300],
        )
        rows = []
    status["seconds"] = time.perf_counter() - start
    return rows, status


def cmd_run_python(args):
    import _adapter

    data_dir = args.work_dir / "data"
    results_dir = args.work_dir / "results"
    results_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = data_dir / "manifest.json"
    if not manifest_path.exists():
        sys.exit(f"{manifest_path} not found: run 'study.py generate' first")
    manifest = json.loads(manifest_path.read_text())

    plan = {}  # (scenario, method) -> reps
    for sid in args.scenarios:
        available = manifest["scenarios"].get(sid, {}).get("reps", 0)
        reps_by_method = {mth: _reps_for(mth, args) for mth in args.methods}
        need = max(reps_by_method.values())
        if available < need:
            sys.exit(f"{sid}: {available} datasets generated, {need} needed: run 'study.py generate --reps {need}'")
        reps_by_method[COMPLETE] = need
        for mth, reps in reps_by_method.items():
            plan[(sid, mth)] = reps

    if not _adapter.SEED_SUPPORTED:
        print(
            "WARNING (finding C1): MICE.impute has no 'seed' argument; every replication "
            "reuses the same random streams. Pipeline check only, not valid for inference.",
            file=sys.stderr,
        )

    tasks = [(sid, mth, rep, str(data_dir)) for (sid, mth), reps in plan.items() for rep in range(1, reps + 1)]
    remaining = {key: reps for key, reps in plan.items()}
    collected = {key: ([], []) for key in plan}
    start = time.perf_counter()
    print(f"running {len(tasks)} replications on {args.jobs} processes")
    with _pool(args.jobs, initializer=_init_worker) as pool:
        for rows, status in pool.imap_unordered(_run_one, tasks, chunksize=2):
            key = (status["scenario"], status["method"])
            collected[key][0].extend(rows)
            collected[key][1].append(status)
            remaining[key] -= 1
            if remaining[key] == 0:
                _write_results(results_dir, key, *collected[key])
                collected[key] = None
    wall = time.perf_counter() - start

    meta = _adapter.library_info()
    meta.update(
        python=platform.python_version(),
        jobs=args.jobs,
        cpu_count=os.cpu_count(),
        wall_seconds=round(wall, 1),
        plan={f"{s}/{m}": r for (s, m), r in plan.items()},
        finished=time.strftime("%Y-%m-%d %H:%M:%S"),
    )
    meta_path = results_dir / "python_meta.json"
    old = json.loads(meta_path.read_text()) if meta_path.exists() else {}
    old_plan = old.get("plan", {})
    old_plan.update(meta["plan"])
    meta["plan"] = old_plan
    meta_path.write_text(json.dumps(meta, indent=2) + "\n")
    print(f"done in {wall:.1f} s")


def _reps_for(method, args):
    return min(args.rf_reps, args.reps) if method == "rf" else args.reps


def _write_results(results_dir, key, rows, statuses):
    sid, mth = key
    res = pd.DataFrame(rows, columns=RESULT_COLUMNS)
    if len(res):
        order = {t: i for i, t in enumerate(SCENARIOS[sid].terms)}
        res = res.assign(_o=res["term"].map(order)).sort_values(["rep", "_o"]).drop(columns="_o")
    res.to_csv(results_dir / f"python_{sid}_{mth}.csv", index=False, float_format="%.17g")
    st = pd.DataFrame(statuses, columns=STATUS_COLUMNS).sort_values("rep")
    st.to_csv(results_dir / f"python_{sid}_{mth}_status.csv", index=False, float_format="%.6g")
    n_fail = int((st["status"] != "ok").sum())
    classes = ", ".join(sorted(set(st.loc[st["status"] != "ok", "error_class"])))
    print(
        f"  {sid} {mth:<10} reps={len(st):5d} failed={n_fail:5d}"
        f"{' (' + classes + ')' if classes else ''}"
        f"  {st['seconds'].mean():.3f} s/rep"
    )


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _pool(jobs, initializer=None):
    # spawn: fresh interpreter per worker on every platform. Under fork all
    # workers would inherit one global NumPy random state, and midas still
    # draws from it (finding A5).
    return mp.get_context("spawn").Pool(processes=jobs, initializer=initializer)


def _init_worker():
    # Import the library before the first task, so that the import time is
    # not counted in the per-replication timings. The parent process prints
    # the C1 warning once; workers stay quiet about it.
    import _adapter

    warnings.simplefilter("ignore", _adapter.SharedStreamsWarning)


def _csv_list(valid, label):
    def parse(text):
        items = [t.strip() for t in text.split(",") if t.strip()]
        bad = [t for t in items if t not in valid]
        if bad:
            raise argparse.ArgumentTypeError(f"unknown {label}: {bad}; choose from {list(valid)}")
        return items

    return parse


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("command", nargs="?", default="all", choices=["generate", "run-python", "all"])
    parser.add_argument("--scenarios", type=_csv_list(SCENARIOS, "scenario"), default=list(SCENARIOS))
    parser.add_argument("--methods", type=_csv_list(METHODS, "method"), default=list(METHODS))
    parser.add_argument("--reps", type=int, default=DEFAULT_REPS, help="replications (default 1000)")
    parser.add_argument(
        "--rf-reps", type=int, default=DEFAULT_RF_REPS, help="replications for rf, capped at --reps (default 500)"
    )
    parser.add_argument("--jobs", type=int, default=os.cpu_count() or 1, help="processes (default: all cores)")
    parser.add_argument("--work-dir", type=Path, default=DEFAULT_WORK, help="default simulations/work")
    args = parser.parse_args(argv)
    if args.reps < 1 or args.rf_reps < 1 or args.jobs < 1:
        parser.error("--reps, --rf-reps and --jobs must be positive")
    args.work_dir = args.work_dir.resolve()

    warnings.simplefilter("default")
    if args.command in ("generate", "all"):
        cmd_generate(args)
    if args.command in ("run-python", "all"):
        cmd_run_python(args)


if __name__ == "__main__":
    main()
