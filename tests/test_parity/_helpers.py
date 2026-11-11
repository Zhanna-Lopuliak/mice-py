"""Helpers shared by the parity tests against R mice.

The reference values live in tests/reference/fixtures and are written by the
R scripts in tests/reference/r. They are committed, so these tests do not
need R.
"""

import json
from pathlib import Path

import numpy as np
import pandas as pd

FIXTURE_DIR = Path(__file__).resolve().parents[1] / "reference" / "fixtures"


def load_fixture(name):
    """Return the reference fixture ``name`` as a dict."""
    with (FIXTURE_DIR / f"{name}.json").open() as fh:
        return json.load(fh)


def as_array(values, dtype=float):
    """Convert a JSON list (or scalar) to an array.

    JSON null becomes NaN. The strings "Inf", "-Inf" and "NaN", which the R
    side writes for non-finite doubles, become the matching floats.
    """
    return np.atleast_1d(np.array(values, dtype=dtype))


def as_bool(values):
    """Convert a JSON list of booleans (or a scalar) to a boolean array."""
    return np.atleast_1d(np.array(values, dtype=bool))


def decode_frame(payload):
    """Rebuild a data frame written by ``encode_frame()`` in ``_common.R``.

    Numeric columns become float64 (null becomes NaN). R factors become
    pandas Categoricals with R's level order and ordering flag, so codes match
    ``as.integer()`` / ``data.matrix()`` in R minus one.
    """
    levels = payload.get("levels") or {}
    ordered = payload.get("ordered") or {}
    columns = str_list(payload["columns"])
    data = {}
    for col in columns:
        values = payload["data"][col]
        if not isinstance(values, list):
            values = [values]
        if col in levels:
            data[col] = pd.Categorical(
                values,
                categories=str_list(levels[col]),
                ordered=bool(ordered.get(col, False)),
            )
        elif any(isinstance(v, str) and v not in _NONFINITE for v in values):
            data[col] = pd.Series(values, dtype=object)
        else:
            data[col] = as_array(values)
    return pd.DataFrame(data, columns=columns)


_NONFINITE = {"Inf", "-Inf", "NaN"}


def str_list(values):
    """A JSON string or list of strings as a list of plain ``str``."""
    if values is None:
        return []
    if not isinstance(values, list):
        values = [values]
    return [str(v) for v in values]


class ScriptedGenerator(np.random.Generator):
    """A NumPy Generator that replays random numbers captured in R.

    ``chisquare`` returns the scripted chi-square draws and ``normal`` /
    ``standard_normal`` the scripted standard normal draws, in order. Every
    other method (``choice``, ``integers``, ``permutation`` ...) is inherited
    and draws from an ordinary PCG64 stream seeded with ``seed``.

    It serves both the code of today, which draws through
    ``scipy.stats.chi2.rvs(df, size=1, random_state=rng)`` (scipy calls
    ``rng.chisquare(df, size)``) and ``rng.normal(size=p)``, and the planned
    code, which calls ``rng.chisquare(df)`` and ``rng.standard_normal(p)``.
    """

    def __init__(self, chisq=(), normals=(), seed=0):
        super().__init__(np.random.PCG64(seed))
        self._chisq = [float(v) for v in np.atleast_1d(chisq)]
        self._normals = [float(v) for v in np.atleast_1d(normals)]

    @property
    def unused(self):
        """Number of scripted values not consumed yet: (chisq, normals)."""
        return len(self._chisq), len(self._normals)

    def _take(self, pool, size, what):
        n = 1 if size is None else int(np.prod(size))
        if n > len(pool):
            raise AssertionError(f"code asked for {n} {what} draws, only {len(pool)} scripted left")
        out = np.array([pool.pop(0) for _ in range(n)])
        return out[0] if size is None else out.reshape(size)

    def chisquare(self, df, size=None):
        return self._take(self._chisq, size, "chi-square")

    def standard_normal(self, size=None, dtype=np.float64, out=None):
        if out is not None:
            raise NotImplementedError("out= is not scripted")
        return self._take(self._normals, size, "standard normal")

    def normal(self, loc=0.0, scale=1.0, size=None):
        if size is None:
            size = np.broadcast(np.asarray(loc), np.asarray(scale)).shape or None
        z = self._take(self._normals, size, "normal")
        return loc + scale * z
