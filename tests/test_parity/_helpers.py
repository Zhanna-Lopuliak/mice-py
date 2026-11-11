"""Helpers shared by the parity tests against R mice.

The reference values live in tests/reference/fixtures and are written by the
R scripts in tests/reference/r. They are committed, so these tests do not
need R.
"""

import json
from pathlib import Path
from typing import NamedTuple

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


# The pool each scripted method of `ScriptedGenerator` draws from
_POOL_OF = {
    "chisquare": "chisq",
    "standard_normal": "normal",
    "normal": "normal",
    "random": "uniform",
    "uniform": "uniform",
    "integers": "integer",
}

# Public attributes of np.random.Generator that draw no random number
_NOT_DRAWING = frozenset({"bit_generator", "spawn"})


class ScriptedCall(NamedTuple):
    """One call of a `ScriptedGenerator` served from a scripted pool."""

    method: str  # "chisquare", "standard_normal", "normal", "random", "uniform" or "integers"
    n: int  # number of values drawn
    args: dict  # df; loc, scale; low, high; or low, high, endpoint ({} for standard_normal and random)


def _refuse(name):
    """A stand-in for method ``name`` of a strict `ScriptedGenerator`."""

    def refuse(*args, **kwargs):
        raise AssertionError(f"rng.{name}() is not scripted (strict ScriptedGenerator)")

    return refuse


class ScriptedGenerator(np.random.Generator):
    """A NumPy Generator that replays random numbers captured in R.

    Four pools of scripted values are served in order:

    - ``chisq``: ``chisquare(df, size)``;
    - ``normals``: ``standard_normal(size)`` and ``normal(loc, scale, size)``,
      which returns ``loc + scale * z``;
    - ``uniforms``: ``random(size)`` and ``uniform(low, high, size)``, which
      returns ``low + (high - low) * u``;
    - ``integers``: ``integers(low, high, size, endpoint=...)``. The values
      are 0-based indices (R's ``sample(n, n, TRUE)`` minus one) and must lie
      in the requested range.

    Asking a pool for more values than it has left raises ``AssertionError``.
    ``chisq`` and ``normals`` are always scripted (empty unless given);
    ``uniforms`` and ``integers`` only when given. Without ``strict``, every
    other method (``choice``, ``permutation`` ...), and the methods of a pool
    that was not given, draw from an ordinary PCG64 stream seeded with
    ``seed``. With ``strict=True`` they raise ``AssertionError`` instead, so
    that code under test cannot draw in any other way; the attributes that
    draw nothing (``bit_generator``, ``spawn``) are left alone.

    Every call served from a pool is logged in `calls`; `order` merges
    consecutive calls on the same pool, and `assert_exhausted` checks that
    every scripted value was used.
    """

    def __init__(self, chisq=(), normals=(), seed=0, *, uniforms=None, integers=None, strict=False):
        super().__init__(np.random.PCG64(seed))
        self._pools = {
            "chisq": [float(v) for v in np.atleast_1d(chisq)],
            "normal": [float(v) for v in np.atleast_1d(normals)],
            "uniform": None if uniforms is None else [float(v) for v in np.atleast_1d(uniforms)],
            "integer": None if integers is None else [int(v) for v in np.atleast_1d(integers)],
        }
        self.calls = []
        if strict:
            # an instance attribute hides the method of the class
            for name in dir(np.random.Generator):
                scripted = name in _POOL_OF and self._pools[_POOL_OF[name]] is not None
                if not (name.startswith("_") or name in _NOT_DRAWING or scripted):
                    setattr(self, name, _refuse(name))

    @property
    def unused(self):
        """Number of scripted values not consumed yet: (chisq, normals)."""
        return len(self._pools["chisq"]), len(self._pools["normal"])

    @property
    def order(self):
        """The logged calls as ``(pool, n)``, consecutive calls on one pool merged.

        The pools are named ``"chisq"``, ``"normal"``, ``"uniform"`` and
        ``"integer"``.
        """
        merged = []
        for call in self.calls:
            pool = _POOL_OF[call.method]
            if merged and merged[-1][0] == pool:
                merged[-1] = (pool, merged[-1][1] + call.n)
            else:
                merged.append((pool, call.n))
        return merged

    def assert_exhausted(self):
        """Raise ``AssertionError`` unless every scripted value was used."""
        left = {name: len(pool) for name, pool in self._pools.items() if pool}
        if left:
            raise AssertionError("scripted values left over: " + ", ".join(f"{n} {name}" for name, n in left.items()))

    def _take(self, method, size, args, dtype=float):
        pool = self._pools[_POOL_OF[method]]
        n = 1 if size is None else int(np.prod(size))
        if n > len(pool):
            raise AssertionError(f"code asked for {n} {method} draws, only {len(pool)} scripted left")
        out = np.array([pool.pop(0) for _ in range(n)], dtype=dtype)
        self.calls.append(ScriptedCall(method, n, args))
        return out[0] if size is None else out.reshape(size)

    def chisquare(self, df, size=None):
        return self._take("chisquare", size, {"df": df})

    def standard_normal(self, size=None, dtype=np.float64, out=None):
        if out is not None:
            raise NotImplementedError("out= is not scripted")
        return self._take("standard_normal", size, {})

    def normal(self, loc=0.0, scale=1.0, size=None):
        if size is None:
            size = np.broadcast(np.asarray(loc), np.asarray(scale)).shape or None
        z = self._take("normal", size, {"loc": loc, "scale": scale})
        return loc + scale * z

    def random(self, size=None, dtype=np.float64, out=None):
        if self._pools["uniform"] is None:
            return super().random(size, dtype, out)
        if out is not None:
            raise NotImplementedError("out= is not scripted")
        return self._take("random", size, {})

    def uniform(self, low=0.0, high=1.0, size=None):
        if self._pools["uniform"] is None:
            return super().uniform(low, high, size)
        if size is None:
            size = np.broadcast(np.asarray(low), np.asarray(high)).shape or None
        u = self._take("uniform", size, {"low": low, "high": high})
        return low + (high - low) * u

    def integers(self, low, high=None, size=None, dtype=np.int64, endpoint=False):
        if self._pools["integer"] is None:
            return super().integers(low, high, size, dtype, endpoint)
        if high is None:  # integers(n) draws from [0, n), logged as low=0, high=n
            low, high = 0, low
        out = self._take("integers", size, {"low": low, "high": high, "endpoint": endpoint}, dtype=dtype)
        inside = (out >= low) & ((out <= high) if endpoint else (out < high))
        if not np.all(inside):
            upper = "]" if endpoint else ")"
            raise AssertionError(f"scripted integers {out} outside the requested range [{low}, {high}{upper}")
        return out
