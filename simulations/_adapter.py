"""The only place where the L3 study calls the mice-py library.

Everything the study needs from the library goes through :func:`impute`, so
when the library's API or defaults change only this file has to follow.

What it fixes explicitly, so that the study does not depend on defaults:

- the full predictor matrix (all ones, zero diagonal), as R's
  ``make.predictorMatrix``;
- the visit sequence: incomplete columns in column order (R's ``"roman"``);
- the method of every incomplete column, given by the study cell; or no
  ``method`` argument at all (cell ``default``), so that the library chooses
  by column type as R's ``make.method`` does;
- ``n_imputations`` and ``maxit``.

It records the method the library used for every column (``MICE.method``),
translated to R's names, so that the report can compare it with R's
``imp$method``.

joblib runs sequentially inside the call: the study parallelises over
replications instead. rf grows its trees in one job by default anyway; the
setting only guards against a forest that asks for more.

Seeds: if ``MICE.impute`` has a ``seed`` parameter it receives the per
replication seed. Otherwise (finding C1: the library hard-codes seed 42 + chain
index) every replication reuses the same random streams; the study still runs,
but a ``SharedStreamsWarning`` is issued once per process and the report states
that the numbers are not valid for inference.
"""

from __future__ import annotations

import contextlib
import inspect
import io
import re
import time
import warnings
from dataclasses import dataclass, field

import joblib
import numpy as np
import pandas as pd

from imputation import MICE

#: Method name (R's) -> mice-py method name.
PYTHON_METHOD = {
    "pmm": "pmm",
    "midastouch": "midas",
    "cart": "cart",
    "rf": "rf",
    "sample": "sample",
    "norm": "norm",
    "norm.boot": "norm.boot",
    "norm.nob": "norm.nob",
    "norm.predict": "norm.predict",
    "mean": "mean",
    "logreg": "logreg",
    "logreg.boot": "logreg.boot",
    "polyreg": "polyreg",
    "polr": "polr",
}

#: mice-py method name -> R's name ("" stays "": the column is not imputed).
R_METHOD = {py: r for r, py in PYTHON_METHOD.items()}

#: True if ``MICE.impute`` accepts ``seed`` (it does not before the C1 fix).
SEED_SUPPORTED = "seed" in inspect.signature(MICE.impute).parameters

# Warnings that the adapter itself provokes and that say nothing about the
# imputation. The library warns when a square predictor matrix has rows for
# complete columns; R's own predictor matrix has such rows too.
_BENIGN_WARNINGS = (re.compile(r"Target variables without missing data are being imputed"),)


class SharedStreamsWarning(UserWarning):
    """Replications share random streams (finding C1); not valid for inference."""


_warned_shared_streams = False


def _warn_shared_streams_once():
    global _warned_shared_streams
    if not _warned_shared_streams:
        warnings.warn(
            "MICE.impute has no 'seed' argument (finding C1): the library seeds "
            "chain j with 42 + j, so every replication reuses the same random "
            "streams. The study runs, but its numbers are not valid for "
            "inference.",
            SharedStreamsWarning,
            stacklevel=3,
        )
        _warned_shared_streams = True


def full_predictor_matrix(columns):
    """All ones with a zero diagonal, as R's ``make.predictorMatrix``."""
    cols = list(columns)
    pm = pd.DataFrame(1, index=cols, columns=cols, dtype=int)
    np.fill_diagonal(pm.values, 0)
    return pm


@dataclass
class Outcome:
    """Result of one call of :func:`impute`.

    ``datasets`` is None when the library raised; ``error_class`` and
    ``error_message`` then describe the exception. ``events`` holds the text
    of every non-benign warning the library issued during the call.
    ``methods`` is the method of every column in data order (R's names,
    ``""`` for a column that is not imputed) as the library set it, or None
    if the library raised before setting it.
    """

    datasets: list | None
    error_class: str = ""
    error_message: str = ""
    events: list = field(default_factory=list)
    methods: dict | None = None
    seconds: float = 0.0

    @property
    def ok(self):
        return self.datasets is not None


def impute(df, methods, m, maxit, seed):
    """Impute ``df`` with the given method of every incomplete column.

    Parameters
    ----------
    df : DataFrame
        Incomplete data; unordered categorical columns as object dtype,
        ordered ones as ordered ``pd.Categorical``.
    methods : dict or None
        Method of each incomplete column, by R's name (``pmm``,
        ``midastouch``, ``norm``, ``logreg.boot``, ...). Every incomplete
        column must be in it. None: no ``method`` argument, so the library
        chooses each column's method by its type.
    m, maxit : int
        Number of imputations and iterations.
    seed : int
        Seed of this replication; passed only if the library accepts it.

    Returns
    -------
    Outcome
        Never raises for a library failure: the exception class and message
        are recorded instead, so one failing replication does not stop the
        study.

    Raises
    ------
    ValueError
        If ``methods`` misses an incomplete column (an error of the study).
    """
    incomplete = [c for c in df.columns if df[c].isna().any()]
    kwargs = dict(
        n_imputations=int(m),
        maxit=int(maxit),
        predictor_matrix=full_predictor_matrix(df.columns),
        visit_sequence=list(incomplete),
    )
    if methods is not None:
        unassigned = [c for c in incomplete if c not in methods]
        if unassigned:
            raise ValueError(f"no method for the incomplete column(s) {unassigned}")
        kwargs["method"] = {c: PYTHON_METHOD[methods[c]] for c in incomplete}
    if SEED_SUPPORTED:
        kwargs["seed"] = int(seed)
    else:
        _warn_shared_streams_once()

    start = time.perf_counter()
    mice = None
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        try:
            # The library does not print; the redirect keeps worker output
            # readable should that change. The study parallelises over
            # replications, so joblib inside the library runs sequentially
            # (rf uses one job by default); with a fixed random_state this
            # does not change any result.
            with contextlib.redirect_stdout(io.StringIO()), joblib.parallel_config(backend="sequential"):
                mice = MICE(df.copy())
                datasets = mice.impute(**kwargs)
        except Exception as exc:  # recorded, not raised: see docstring
            return Outcome(
                datasets=None,
                error_class=type(exc).__name__,
                error_message=_one_line(exc),
                events=_events(caught),
                methods=_methods(mice),
                seconds=time.perf_counter() - start,
            )
    return Outcome(
        datasets=[d.copy() for d in datasets],
        events=_events(caught),
        methods=_methods(mice),
        seconds=time.perf_counter() - start,
    )


def _methods(mice):
    """The method of every column as the library set it, in R's names; None if unset."""
    used = getattr(mice, "method", None)
    if not isinstance(used, dict):
        return None
    return {col: R_METHOD.get(meth, meth) for col, meth in used.items()}


def _events(caught):
    out = []
    for w in caught:
        if issubclass(w.category, SharedStreamsWarning):
            continue
        text = str(w.message)
        if any(p.search(text) for p in _BENIGN_WARNINGS):
            continue
        out.append(f"{w.category.__name__}: {text}")
    return out


def _one_line(exc):
    text = " ".join(str(exc).split())
    return text[:300]


def library_info():
    """Facts about the library run, for the report."""
    import imputation

    version = getattr(imputation, "__version__", None)
    if version is None:
        try:
            from importlib.metadata import version as _v

            version = _v("mice-py")
        except Exception:  # pragma: no cover - metadata missing
            version = "unknown"
    return {
        "library_version": version,
        "seed_supported": SEED_SUPPORTED,
        "numpy": np.__version__,
        "pandas": pd.__version__,
    }
