"""L3 simulation study: generate the datasets and run the Python side.

Run from the repository root::

    uv run python simulations/study.py generate   [--scenarios S1,S4] [--reps N]
    uv run python simulations/study.py run-python [--scenarios S1,S4] [--methods pmm,default]
                                                   [--reps N] [--jobs 8]
    uv run python simulations/study.py all        (generate, then run-python; the default)

Then ``Rscript simulations/run_r.R`` (same options) and
``uv run python simulations/report.py``.

The study is a table of cells (:data:`CELLS`): a study method on a scenario,
with its number of replications. A study method gives each incomplete column
a method (:data:`STUDY_METHODS`), or gives no method at all (``default``:
each engine chooses by column type). ``generate`` writes the cells, with the
method of every column, to the manifest; ``run-python`` and ``run_r.R`` run
the cells of the manifest, so both engines run exactly the same cells.
``--scenarios`` and ``--methods`` select cells; ``--reps N`` caps the
replications of every selected cell (and the datasets ``generate`` writes).

Files (all under ``simulations/work/``, git-ignored):

- ``data/<scenario>/rep_<k>.csv``: incomplete dataset k (1-based);
- ``data/<scenario>/complete/rep_<k>.csv``: the same dataset before deletion;
- ``data/manifest.json``: scenario specifications and cells for ``run_r.R``;
- ``results/python_<scenario>_<method>.csv``: one row per replication and term,
  columns ``scenario, method, rep, term, qbar, ubar, b, m, n, dfcom``;
- ``results/python_<scenario>_<method>_status.csv``: one row per replication,
  ``status`` ("ok" or "failed"), exception class and message, warnings, the
  methods the library used (``methods``, ``col=method;...`` in data order,
  R's names), seconds (``n_events``/``events`` are R's ``loggedEvents`` and
  stay empty here);
- ``results/python_meta.json``: library version, whether seeds were honoured.

``method`` is the study method, with R's names (``midastouch``, not
``midas``). The pseudo-method ``complete`` analyses the data before deletion
(m = 1, b = 0) and validates the harness; it always runs.
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
from dataclasses import dataclass  # noqa: E402
from pathlib import Path  # noqa: E402

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from scenarios import SCENARIOS, analyse, read_dataset  # noqa: E402

HERE = Path(__file__).resolve().parent
DEFAULT_WORK = HERE / "work"

COMPLETE = "complete"
M_IMPUTATIONS = 5
DEFAULT_REPS = 1000
RF_REPS = 500

#: Study method -> (method of a numeric column, method of a categorical
#: column), R's names. None: no ``method`` argument at all on either side,
#: so each engine chooses by column type (R's ``make.method``). A study
#: method without a method for a column type cannot run on a scenario with
#: an incomplete column of that type.
STUDY_METHODS = {
    "pmm": ("pmm", "pmm"),
    "midastouch": ("midastouch", "midastouch"),
    "cart": ("cart", "cart"),
    "rf": ("rf", "rf"),
    "sample": ("sample", "sample"),
    "norm": ("norm", None),
    "norm.boot": ("norm.boot", None),
    "norm.nob": ("norm.nob", None),
    "norm.predict": ("norm.predict", None),
    "mean": ("mean", None),
    "logreg.boot": ("pmm", "logreg.boot"),
    "default": None,
}

#: Pass rules: ``z``, every paired Python-minus-R difference has |z| <= 4.2;
#: ``identity``, a deterministic method (no draw at all), where Python and R
#: must agree to rounding (``metrics.identity_check``).
RULES = ("z", "identity")


@dataclass(frozen=True)
class Cell:
    """One cell of the study: a study method on a scenario.

    ``expected`` (``default`` cells only) is the method of each incomplete
    column that R's ``make.method`` chooses; the report checks that both
    engines chose it.
    """

    method: str
    scenario: str
    reps: int = DEFAULT_REPS
    rule: str = "z"
    expected: dict | None = None

    def columns(self):
        """Method of each incomplete column of the scenario (R's names), or None for ``default``."""
        kinds = STUDY_METHODS[self.method]
        if kinds is None:
            return None
        sc = SCENARIOS[self.scenario]
        out = {}
        for col in sc.incomplete:
            method = kinds[1] if col in sc.categorical else kinds[0]
            if method is None:
                raise ValueError(f"study method {self.method!r} has no method for column {col!r} of {sc.id}")
            out[col] = method
        return out

    def spec(self):
        """The cell as written to the manifest."""
        return {
            "method": self.method,
            "scenario": self.scenario,
            "reps": self.reps,
            "rule": self.rule,
            "columns": self.columns(),
            "expected_methods": self.expected,
        }


#: The cells of the study. The 0.2 cells (five donor methods on S1-S6) come
#: first, unchanged; then the model-based methods of 0.3 and the defaults.
CELLS = [
    *(
        Cell(method, sid, RF_REPS if method == "rf" else DEFAULT_REPS)
        for sid in ("S1", "S2", "S3", "S4", "S5", "S6")
        for method in ("pmm", "midastouch", "cart", "rf", "sample")
    ),
    Cell("norm", "S3"),
    Cell("norm", "S4"),
    Cell("norm.boot", "S3"),
    Cell("norm.boot", "S4"),
    Cell("norm.nob", "S4"),
    # One incomplete column and maxit = 1: nothing is drawn
    Cell("norm.predict", "S4", rule="identity"),
    Cell("mean", "S4", rule="identity"),
    Cell("default", "S5", expected={"g": "polyreg", "x": "pmm"}),
    Cell("default", "S7", expected={"b": "logreg", "x": "pmm"}),
    Cell("default", "S8", expected={"o": "polr", "x": "pmm"}),
    Cell("logreg.boot", "S7"),
]

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
    "methods",
    "seconds",
]


def replication_seed(scenario_index, rep):
    """Imputation seed of one replication; ``run_r.R`` uses the same formula."""
    return scenario_index * 100_000 + rep


def cell_specs():
    """All cells as written to the manifest."""
    return [cell.spec() for cell in CELLS]


def select_cells(cells, scenarios, methods, reps_cap):
    """The cells (manifest dicts) of the given scenarios and methods, with ``reps`` capped."""
    out = []
    for cell in cells:
        if cell["scenario"] in scenarios and cell["method"] in methods:
            reps = cell["reps"] if reps_cap is None else min(cell["reps"], reps_cap)
            out.append({**cell, "reps": reps})
    return out


def format_methods(methods):
    """``{col: method}`` -> ``"col=method;..."``, the form both engines record."""
    return ";".join(f"{col}={meth}" for col, meth in methods.items())


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
        "incomplete": list(sc.incomplete),
        "categorical": {c: {"levels": list(f.levels), "ordered": f.ordered} for c, f in sc.categorical.items()},
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
    # As many datasets as the scenario's largest cell needs, capped by --reps
    all_methods = list(STUDY_METHODS)
    n_datasets = {}
    for sid in args.scenarios:
        need = max((c["reps"] for c in select_cells(cell_specs(), [sid], all_methods, args.reps)), default=0)
        if need:
            n_datasets[sid] = need
    tasks = []
    for sid, need in n_datasets.items():
        (data_dir / sid / "complete").mkdir(parents=True, exist_ok=True)
        tasks += [(sid, rep, str(data_dir)) for rep in range(1, need + 1)]
    start = time.perf_counter()
    rates = {}
    with _pool(args.jobs) as pool:
        for sid, _rep, rate in pool.imap_unordered(_generate_one, tasks, chunksize=16):
            rates.setdefault(sid, []).append(rate)
    for sid, need in n_datasets.items():
        manifest["scenarios"][sid] = scenario_spec(SCENARIOS[sid], need)
        mean_rate = pd.DataFrame(rates[sid]).mean().round(4).to_dict()
        print(f"{sid}: {need} datasets, mean missing rate per column {mean_rate}")
    manifest["cells"] = cell_specs()
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
    """One (scenario, method, rep): impute, analyse, return rows and status.

    ``columns`` is the cell's method of each incomplete column, or None for
    ``default`` (no method argument).
    """
    scenario_id, method, columns, rep, data_dir = args
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
        methods="",
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
            outcome = _adapter.impute(df, columns, M_IMPUTATIONS, sc.maxit, replication_seed(sc.index, rep))
            status["n_warnings"] = len(outcome.events)
            status["warnings"] = " | ".join(sorted(set(outcome.events)))[:500]
            if outcome.methods is not None:
                status["methods"] = format_methods(outcome.methods)
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

    if manifest.get("cells") != cell_specs():
        sys.exit(f"the cells in {manifest_path} differ from study.CELLS: run 'study.py generate' again")
    cells = select_cells(manifest["cells"], args.scenarios, args.methods, args.reps)
    if not cells:
        sys.exit("no cell matches --scenarios and --methods")

    plan = {}  # (scenario, method) -> reps
    columns = {}  # (scenario, method) -> method of each incomplete column, or None
    for sid in args.scenarios:
        own = [c for c in cells if c["scenario"] == sid]
        if not own:
            continue
        available = manifest["scenarios"].get(sid, {}).get("reps", 0)
        need = max(c["reps"] for c in own)
        if available < need:
            sys.exit(f"{sid}: {available} datasets generated, {need} needed: run 'study.py generate --reps {need}'")
        for c in own:
            plan[(sid, c["method"])] = c["reps"]
            columns[(sid, c["method"])] = c["columns"]
        plan[(sid, COMPLETE)] = need
        columns[(sid, COMPLETE)] = None

    if not _adapter.SEED_SUPPORTED:
        print(
            "WARNING (finding C1): MICE.impute has no 'seed' argument; every replication "
            "reuses the same random streams. Pipeline check only, not valid for inference.",
            file=sys.stderr,
        )

    tasks = [
        (sid, mth, columns[(sid, mth)], rep, str(data_dir))
        for (sid, mth), reps in plan.items()
        for rep in range(1, reps + 1)
    ]
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
        f"  {sid} {mth:<12} reps={len(st):5d} failed={n_fail:5d}"
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
    parser.add_argument(
        "--methods",
        type=_csv_list(STUDY_METHODS, "study method"),
        default=list(STUDY_METHODS),
        help="study methods whose cells run (default: all)",
    )
    parser.add_argument(
        "--reps",
        type=int,
        default=None,
        help="cap on the replications of every cell (default: the cell's own, 1000; rf 500)",
    )
    parser.add_argument("--jobs", type=int, default=os.cpu_count() or 1, help="processes (default: all cores)")
    parser.add_argument("--work-dir", type=Path, default=DEFAULT_WORK, help="default simulations/work")
    args = parser.parse_args(argv)
    if (args.reps is not None and args.reps < 1) or args.jobs < 1:
        parser.error("--reps and --jobs must be positive")
    args.work_dir = args.work_dir.resolve()

    warnings.simplefilter("default")
    if args.command in ("generate", "all"):
        cmd_generate(args)
    if args.command in ("run-python", "all"):
        cmd_run_python(args)


if __name__ == "__main__":
    main()
