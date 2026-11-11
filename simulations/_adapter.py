"""The only place where the L3 study calls the mice-py library.

Everything the study needs from the library goes through :func:`impute`, so
when the library's API or defaults change only this file has to follow.

What it fixes explicitly, so that the study does not depend on defaults:

- the full predictor matrix (all ones, zero diagonal), as R's
  ``make.predictorMatrix``;
- the visit sequence: incomplete columns in column order (R's ``"roman"``);
- the method, forced for every incomplete column;
- ``n_imputations`` and ``maxit``.

joblib runs sequentially inside the call (the study parallelises over
replications instead; rf would otherwise start one thread per core per tree
fit, which is 2-3 times slower and oversubscribes the machine).

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

#: Study method name (R's) -> mice-py method name.
PYTHON_METHOD = {
    "pmm": "pmm",
    "midastouch": "midas",
    "cart": "cart",
    "rf": "rf",
    "sample": "sample",
}

#: True if ``MICE.impute`` accepts ``seed`` (it does not before the C1 fix).
SEED_SUPPORTED = "seed" in inspect.signature(MICE.impute).parameters

# Warnings that the adapter itself provokes and that say nothing about the
# imputation. The library warns when a square predictor matrix has rows for
# complete columns; R's own predictor matrix has such rows too.
_BENIGN_WARNINGS = (
    re.compile(r"Target variables without missing data are being imputed"),
)


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
    """

    datasets: list | None
    error_class: str = ""
    error_message: str = ""
    events: list = field(default_factory=list)
    seconds: float = 0.0

    @property
    def ok(self):
        return self.datasets is not None


def impute(df, method, m, maxit, seed):
    """Impute ``df`` with ``method`` forced for every incomplete column.

    Parameters
    ----------
    df : DataFrame
        Incomplete data; categorical columns as object dtype.
    method : str
        Study method name (``pmm``, ``midastouch``, ``cart``, ``rf``, ``sample``).
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
    """
    incomplete = [c for c in df.columns if df[c].isna().any()]
    kwargs = dict(
        n_imputations=int(m),
        maxit=int(maxit),
        predictor_matrix=full_predictor_matrix(df.columns),
        method={c: PYTHON_METHOD[method] for c in incomplete},
        visit_sequence=list(incomplete),
    )
    if SEED_SUPPORTED:
        kwargs["seed"] = int(seed)
    else:
        _warn_shared_streams_once()

    start = time.perf_counter()
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        try:
            # Validators print to stdout; keep worker output readable. The
            # study parallelises over replications, so joblib inside the
            # library (rf uses n_jobs=-1) runs sequentially; with fixed
            # random_state this does not change any result.
            with contextlib.redirect_stdout(io.StringIO()), joblib.parallel_config(
                backend="sequential"
            ):
                datasets = MICE(df.copy()).impute(**kwargs)
        except Exception as exc:  # recorded, not raised: see docstring
            return Outcome(
                datasets=None,
                error_class=type(exc).__name__,
                error_message=_one_line(exc),
                events=_events(caught),
                seconds=time.perf_counter() - start,
            )
    return Outcome(
        datasets=[d.copy() for d in datasets],
        events=_events(caught),
        seconds=time.perf_counter() - start,
    )


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
