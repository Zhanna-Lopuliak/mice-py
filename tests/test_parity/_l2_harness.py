"""Shared machinery for the layer-2 distribution parity tests.

The R side (tests/reference/r/l2_<method>.R) called ``mice.impute.<method>``
many times on fixed inputs and stored sufficient statistics in
``l2_<method>.json`` (dots in the method name become underscores). This
module rebuilds the inputs from those fixtures, calls the Python imputer many
times with one seeded ``np.random.Generator``, summarises the draws the same
way and computes the T1/T2/T3 statistics of ``_l2_stats``.

How the draws are recorded and summarised depends on the ``kind`` of the
method, stored in its fixture (see ``tests/reference/r/_l2_common.R``):

* ``"donor"`` (a fixture without ``kind``): the imputations are observed
  values or levels. Each is mapped to its donor index; T1 compares the donor
  counts per cell.
* ``"continuous"``: the imputations are new numbers. They are kept as floats;
  T1 compares the counts per cell in the ten bins cut at the deciles of R's
  draws.
* ``"category"``: the imputations are labels drawn from a fitted model (a
  factor target). Each is mapped to its level index; T1 compares the level
  counts per cell.

For a numeric target the first two kinds give T2 (moments per cell) and T3
(moments of the OLS coefficients of the completed y). The category kind gives
T3 on the OLS coefficients of the 0/1 indicator of each level but the first
in the completed target; there is no T2 for labels. A kind is one
:class:`Kind` entry in ``KINDS``.
"""

import functools
from dataclasses import dataclass
from typing import Callable

import numpy as np

from ._helpers import as_array, load_fixture
from ._l2_stats import P_MIN, Z_MAX, chi2_homogeneity, moments, z_logvar, z_mean

# Python calls per method and input (R: 20,000; rf: 5,000)
N_CALLS = {
    "pmm": 3000,
    "midastouch": 1500,
    "cart": 1500,
    "rf": 300,
    "sample": 3000,
    "norm": 3000,
    "norm.nob": 3000,
    "norm.boot": 3000,
    "logreg": 3000,
    "logreg.boot": 3000,
    "polyreg": 3000,
}

# Offset between the R seed stored in a fixture and the Python seed
PY_SEED_OFFSET = 10_000


@dataclass(frozen=True)
class L2Input:
    """One fixed input ``(y, ry, x)`` as stored in a layer-2 fixture."""

    name: str
    type: str  # "numeric" or "factor"
    y: np.ndarray  # float with NaN, or object with NaN, at the missing cells
    ry: np.ndarray  # bool, True where y is observed
    x: np.ndarray  # float, n x p, no intercept
    x_names: tuple
    donor_values: np.ndarray  # sorted distinct observed values, or the levels

    @property
    def is_factor(self):
        return self.type == "factor"

    @property
    def n_mis(self):
        return int((~self.ry).sum())


def load_input(payload):
    """Build an :class:`L2Input` from the ``input`` entry of a fixture."""
    ry = np.array(payload["ry"], dtype=bool)
    if payload["type"] == "factor":
        y = np.array([np.nan if v is None else v for v in payload["y"]], dtype=object)
        donor_values = np.array(payload["donor_values"], dtype=object)
    else:
        y = as_array(payload["y"])
        donor_values = as_array(payload["donor_values"])
    x = np.array(payload["x"], dtype=float).reshape(len(ry), -1)
    return L2Input(
        name=payload["name"],
        type=payload["type"],
        y=y,
        ry=ry,
        x=x,
        x_names=tuple(payload["x_names"]),
        donor_values=donor_values,
    )


def fixture_name(method):
    """Name of the fixture of a method: ``l2_<method>``, dots as underscores."""
    return "l2_" + method.replace(".", "_")


@functools.cache
def r_fixture(method):
    """The fixture of a method, as a dict."""
    return load_fixture(fixture_name(method))


def r_results(method):
    """The ``results`` entry of the fixture of a method (dict keyed by input)."""
    return r_fixture(method)["results"]


def r_kind(method):
    """The kind of a method, from its fixture; ``"donor"`` if none is stored."""
    kind = r_fixture(method).get("kind", "donor")
    if kind not in KINDS:
        raise ValueError(f"fixture {fixture_name(method)} has unknown kind {kind!r}")
    return kind


def r_reference(method, input_name):
    """Return ``(L2Input, R summary dict)`` for one method and input."""
    res = r_results(method)[input_name]
    return load_input(res["input"]), res


def python_seed(method, input_name):
    return int(r_results(method)[input_name]["seed"]) + PY_SEED_OFFSET


def imputer(method):
    """The Python function behind a mice method name (imported lazily)."""
    if method == "pmm":
        from imputation.PMM import pmm

        return pmm
    if method == "midastouch":
        from imputation.midas import midas

        return midas
    if method == "cart":
        from imputation.cart import cart

        return cart
    if method == "rf":
        from imputation.rf import rf

        return rf
    if method == "sample":
        from imputation.sample import sample

        return sample
    if method == "norm":
        from imputation.norm import norm

        return norm
    if method == "norm.nob":
        from imputation.norm import norm_nob

        return norm_nob
    if method == "norm.boot":
        from imputation.norm import norm_boot

        return norm_boot
    if method == "logreg":
        from imputation.logreg import logreg

        return logreg
    if method == "logreg.boot":
        from imputation.logreg import logreg_boot

        return logreg_boot
    if method == "polyreg":
        from imputation.polyreg import polyreg

        return polyreg
    raise ValueError(method)


def donor_index(values, inp):
    """Map imputed values to indices into ``inp.donor_values``; -1 if not a donor."""
    values = np.asarray(values)
    if inp.is_factor:
        lookup = {str(v): i for i, v in enumerate(inp.donor_values)}
        return np.array([lookup.get(str(v), -1) for v in values], dtype=int)
    vals = values.astype(float)
    dv = inp.donor_values
    pos = np.clip(np.searchsorted(dv, vals), 0, len(dv) - 1)
    return np.where(dv[pos] == vals, pos, -1)


def draw(func, inp, n_calls, seed, kind="donor", **kwargs):
    """Call ``func(y, ry, x, rng=rng, **kwargs)`` ``n_calls`` times.

    One ``np.random.default_rng(seed)`` is passed to every call.

    Returns an ``n_calls x n_mis`` matrix with one row per call, recorded as
    ``kind`` says: for ``"donor"`` and ``"category"`` integer donor (level)
    indices (-1 where an imputation is not an observed value or level), for
    ``"continuous"`` the imputed values as floats.
    """
    handler = KINDS[kind]
    rng = np.random.default_rng(seed)
    rows = np.empty((n_calls, inp.n_mis), dtype=handler.dtype)
    for i in range(n_calls):
        out = func(inp.y.copy(), inp.ry.copy(), inp.x.copy(), rng=rng, **kwargs)
        out = np.asarray(out).ravel()
        if out.shape != (inp.n_mis,):
            raise AssertionError(f"imputer returned shape {out.shape}, expected ({inp.n_mis},)")
        rows[i] = handler.record(out, inp)
    return rows


def summarise(draws, inp, kind="donor", r_sum=None):
    """Sufficient statistics of the matrix returned by :func:`draw`, as in the
    R fixtures. ``r_sum`` is R's summary of the same input; the continuous kind
    needs it for R's decile edges."""
    return KINDS[kind].summarise(draws, inp, r_sum)


def _value_moments(imp, inp):
    """Moments per cell (``cell``) of an ``n_calls x n_mis`` matrix of imputed
    values, and of the OLS coefficients of the completed y on ``[1, x]``,
    one fit per call (``coef``)."""
    x1 = np.column_stack([np.ones(len(inp.ry)), inp.x])
    ycomp = np.repeat(inp.y[:, None], imp.shape[0], axis=1)
    ycomp[~inp.ry, :] = imp.T
    coef = np.linalg.solve(x1.T @ x1, x1.T @ ycomp)
    return {"cell": moments(imp), "coef": moments(coef.T)}


def _summarise_donors(idx, inp, r_sum=None):
    """Donor counts per cell; for a numeric target whose imputations are all
    observed values, also the moments (``_value_moments``)."""
    k = len(inp.donor_values)
    valid = idx >= 0
    counts = np.stack([np.bincount(col[col >= 0], minlength=k) for col in idx.T])
    out = {
        "kind": "donor",
        "n_calls": idx.shape[0],
        "counts": counts,
        "n_not_donor": int((~valid).sum()),
    }
    if not inp.is_factor and valid.all():
        out.update(_value_moments(inp.donor_values[idx], inp))
    return out


def decile_bin_counts(vals, edges):
    """Per cell, the number of values in each of the bins cut at ``edges``.

    ``vals`` is ``n_calls x n_mis``, ``edges`` is ``n_mis x 9`` (R's deciles).
    Bin j holds the values in ``[edge j-1, edge j)``, with -inf and inf as the
    outer edges, which is how R's ``findInterval`` counted R's own draws.
    """
    return np.stack(
        [np.bincount(np.searchsorted(e, col, side="right"), minlength=len(e) + 1) for e, col in zip(edges, vals.T)]
    )


def _summarise_values(vals, inp, r_sum):
    """Moments (``_value_moments``) and the counts in R's decile bins."""
    if r_sum is None:
        raise ValueError("the continuous kind needs R's summary for its decile edges")
    edges = np.array(r_sum["bins"]["edges"], dtype=float).reshape(inp.n_mis, -1)
    out = {"kind": "continuous", "n_calls": vals.shape[0], "bins": decile_bin_counts(vals, edges)}
    out.update(_value_moments(vals, inp))
    return out


def _indicator_coef_moments(idx, inp):
    """Moments of the OLS coefficients of the 0/1 indicator of each level but
    the first in the completed target on ``[1, x]``, one fit per call; level
    by level, terms in the order of ``[1, x]`` (R's ``coef$names``)."""
    x1 = np.column_stack([np.ones(len(inp.ry)), inp.x])
    codes = np.repeat(donor_index(inp.y, inp)[:, None], idx.shape[0], axis=1)
    codes[~inp.ry, :] = idx.T
    coef = [
        np.linalg.solve(x1.T @ x1, x1.T @ (codes == level).astype(float)) for level in range(1, len(inp.donor_values))
    ]
    return moments(np.vstack(coef).T)


def _summarise_labels(idx, inp, r_sum=None):
    """Level counts per cell (as ``_summarise_donors`` for a factor target)
    and, when every imputation is a level, the moments of the coefficients
    of the level indicators (``coef``)."""
    if not inp.is_factor:
        raise ValueError("the category kind needs a factor target")
    out = _summarise_donors(idx, inp)
    out["kind"] = "category"
    if out["n_not_donor"] == 0:
        out["coef"] = _indicator_coef_moments(idx, inp)
    return out


@dataclass(frozen=True)
class Kind:
    """How the draws of one kind of method are recorded, summarised and
    tested by T1.

    A new kind is one more entry in ``KINDS`` and the matching entry in
    ``L2_KINDS`` of ``tests/reference/r/_l2_common.R``.
    """

    dtype: type  # of the matrix returned by draw()
    record: Callable  # (imputations of one call, L2Input) -> one row of it
    summarise: Callable  # (that matrix, L2Input, R summary) -> Python summary
    t1_counts: Callable  # (R summary, Python summary) -> what T1 compares, per cell
    t1_label: str  # what T1 compares, for messages
    not_recorded: str = ""  # what an imputation that record() maps to -1 is not; "" if there is none


KINDS = {
    "donor": Kind(
        dtype=int,
        record=donor_index,
        summarise=_summarise_donors,
        t1_counts=lambda r_sum, py_sum: (r_sum["counts"], py_sum["counts"]),
        t1_label="donor counts",
        not_recorded="observed values",
    ),
    "continuous": Kind(
        dtype=float,
        record=lambda out, inp: out.astype(float),
        summarise=_summarise_values,
        t1_counts=lambda r_sum, py_sum: (r_sum["bins"]["counts"], py_sum["bins"]),
        t1_label="counts in R's decile bins",
    ),
    "category": Kind(
        dtype=int,
        record=donor_index,
        summarise=_summarise_labels,
        t1_counts=lambda r_sum, py_sum: (r_sum["counts"], py_sum["counts"]),
        t1_label="level counts",
        not_recorded="levels of the target",
    ),
}


@dataclass
class Comparison:
    """T1/T2/T3 results for one method and input."""

    t1_p: np.ndarray  # per cell
    t2_z_mean: np.ndarray = None  # per cell
    t2_z_var: np.ndarray = None
    t3_z_mean: np.ndarray = None  # per coefficient
    t3_z_var: np.ndarray = None
    coef_names: tuple = ()
    n_not_donor: int = 0  # imputations that are not observed values (donor) or levels (category)
    kind: str = "donor"

    @staticmethod
    def _max_abs(z):
        return float(np.max(np.abs(z))) if z is not None and len(z) else float("nan")

    @property
    def t1_min_p(self):
        return float(np.min(self.t1_p))

    @property
    def t2_max_z(self):
        return max(self._max_abs(self.t2_z_mean), self._max_abs(self.t2_z_var))

    @property
    def t3_max_z(self):
        return max(self._max_abs(self.t3_z_mean), self._max_abs(self.t3_z_var))

    def t1_ok(self):
        return self.n_not_donor == 0 and self.t1_min_p >= P_MIN

    def t2_ok(self):
        return self.t2_max_z <= Z_MAX

    def t3_ok(self):
        return self.t3_max_z <= Z_MAX

    def t1_message(self):
        order = np.argsort(self.t1_p)[:3]
        worst = ", ".join(f"cell {j}: p={self.t1_p[j]:.3g}" for j in order)
        not_recorded = KINDS[self.kind].not_recorded
        not_donor = f"{self.n_not_donor} imputations are not {not_recorded}; " if not_recorded else ""
        return f"T1 ({KINDS[self.kind].t1_label}): {not_donor}smallest p-values {worst} (pass: p >= {P_MIN:g})"

    def t2_message(self):
        return _z_message("T2 cell", self.t2_z_mean, self.t2_z_var, None, self.n_not_donor)

    def t3_message(self):
        return _z_message("T3 coef", self.t3_z_mean, self.t3_z_var, self.coef_names, self.n_not_donor)


def _z_message(label, zm, zv, names, n_not_donor):
    if zm is None or zv is None:
        return (
            f"{label}: not computed ({n_not_donor} imputations are not observed values or levels, "
            "or the statistic does not apply to this kind and target)"
        )

    def worst(z):
        j = int(np.argmax(np.abs(z)))
        name = names[j] if names else j
        return f"{name} z={z[j]:+.2f}"

    return f"{label}: largest |z| mean {worst(zm)}, log-variance {worst(zv)} (pass: |z| <= {Z_MAX:g})"


def compare(r_sum, py_sum, inp):
    """Compute T1 (all inputs), T2 (numeric targets) and T3 (numeric targets
    and the category kind) for one method/input.

    T1 compares what the kind of ``py_sum`` counts per cell: donor counts,
    the counts in R's decile bins, or level counts. T2 and T3 are computed
    where ``py_sum`` has their moments (``cell``, ``coef``).
    """
    kind = py_sum["kind"]
    counts_r, counts_p = KINDS[kind].t1_counts(r_sum, py_sum)
    counts_r = np.array(counts_r, dtype=float).reshape(inp.n_mis, -1)
    t1_p = np.array([chi2_homogeneity(cr, cp)[2] for cr, cp in zip(counts_r, counts_p)])
    comp = Comparison(t1_p=t1_p, n_not_donor=py_sum.get("n_not_donor", 0), kind=kind)
    n_r, n_p = r_sum["n_calls"], py_sum["n_calls"]
    if "cell" in py_sum:
        rc, pc = r_sum["cell"], py_sum["cell"]
        comp.t2_z_mean = z_mean(pc["mean"], pc["var"], n_p, as_array(rc["mean"]), as_array(rc["var"]), n_r)
        comp.t2_z_var = z_logvar(pc["var"], pc["kurt"], n_p, as_array(rc["var"]), as_array(rc["kurt"]), n_r)
    if "coef" not in py_sum:
        return comp
    rb, pb = r_sum["coef"], py_sum["coef"]
    comp.t3_z_mean = z_mean(pb["mean"], pb["var"], n_p, as_array(rb["mean"]), as_array(rb["var"]), n_r)
    comp.t3_z_var = z_logvar(pb["var"], pb["kurt"], n_p, as_array(rb["var"]), as_array(rb["kurt"]), n_r)
    comp.coef_names = tuple(rb["names"])
    return comp


def run_python(method, input_name, func=None, n_calls=None, **kwargs):
    """Draw from the Python imputer for one fixture input and summarise.

    ``func`` defaults to the library function for ``method``; ``kwargs`` are
    passed to every call. The draws are recorded and summarised by the kind
    of R's fixture of ``method``, also when ``func`` is another function.
    Returns ``(L2Input, R summary, Python summary)``.
    """
    inp, r_sum = r_reference(method, input_name)
    kind = r_kind(method)
    func = imputer(method) if func is None else func
    draws = draw(
        func,
        inp,
        N_CALLS[method] if n_calls is None else n_calls,
        python_seed(method, input_name),
        kind=kind,
        **kwargs,
    )
    return inp, r_sum, summarise(draws, inp, kind, r_sum)


def reached_sets(counts):
    """Per cell, the set of donor indices drawn at least once."""
    return [frozenset(np.flatnonzero(np.asarray(row) > 0).tolist()) for row in counts]


def donor_set_mismatches(r_sum, py_sum):
    """Cells whose donor sets differ between R's first 2,000 calls (stored as
    ``reached_2000``) and the Python calls, described as text."""
    r_sets = reached_sets(r_sum["reached_2000"])
    p_sets = reached_sets(py_sum["counts"])
    return [
        f"cell {j}: python {len(p)} donors, R {len(r)}, common {len(p & r)}"
        for j, (p, r) in enumerate(zip(p_sets, r_sets))
        if p != r
    ]


def weak_numeric(r_sum, py_sum):
    """Weaker parity measures used where exact tree parity is out of reach.

    Returns ``(largest |mean_P - mean_R| / sd_R over cells, pooled variance
    ratio)`` where ``sd_R`` is R's standard deviation of the imputed value in
    that cell and the pooled ratio is ``mean(var_P) / mean(var_R)``.
    """
    rm, rv = as_array(r_sum["cell"]["mean"]), as_array(r_sum["cell"]["var"])
    pm, pv = py_sum["cell"]["mean"], py_sum["cell"]["var"]
    with np.errstate(divide="ignore", invalid="ignore"):
        dev = np.abs(pm - rm) / np.sqrt(rv)
    dev = np.where((rv == 0) & (pm == rm), 0.0, dev)
    return float(np.max(dev)), float(pv.mean() / rv.mean())


def weak_factor(r_sum, py_sum):
    """Largest absolute difference of the overall level proportions."""
    cr = np.asarray(r_sum["counts"], dtype=float).sum(axis=0)
    cp = np.asarray(py_sum["counts"], dtype=float).sum(axis=0)
    return float(np.max(np.abs(cr / cr.sum() - cp / cp.sum())))


@functools.cache
def cached_run(method, input_name):
    """``run_python`` for the unpatched library with default arguments (cached)."""
    return run_python(method, input_name)


@functools.cache
def cached_comparison(method, input_name):
    """T1/T2/T3 for the unpatched library with default arguments (cached)."""
    inp, r_sum, py_sum = cached_run(method, input_name)
    return compare(r_sum, py_sum, inp)
