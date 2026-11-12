"""Layer 2 at chain level with R's default methods: pmm, logreg, polyreg and polr.

R ran ``mice(data, m = 200, maxit = 5, seed = 7802)`` without ``method`` on
the embedded mixed frame of 100 rows (``x`` numeric and complete, ``y``
numeric, ``b`` a factor no / yes, ``g`` a factor a / b / c, ``o`` an ordered
factor p < q < r < s; see ``tests/reference/r/l2_chain_default.R``), so R's
``make.method`` chose pmm, logreg, polyreg and polr. Python runs
``MICE(frame).impute(n_imputations=N_CHAINS, maxit=5, seed=SEED)``, also
without ``method``, and must choose the same methods. Each statistic of the
last iteration is a sample over the chains on each side: the mean and the
variance of the imputed values of ``y`` (``chainMean`` and ``chainVar``),
and the share of each level but the first among the imputed values of
``b``, ``g`` and ``o``. Their means and log-variances are compared with the
z-scores of T2 (pass: ``|z| <= Z_MAX``).

120 Python chains take about 4.7 seconds. Measured largest |z| over the 16
z-scores: 1.67 with the committed seed, 0.92 to 2.22 with the seeds 7803 to
7810.

The 0.2 default as a mutant: ``make_method`` replaced by ``"pmm"`` for every
incomplete column. The methods then differ from R's, which
`test_default_methods_match_r` flags (asserted below). The distribution
statistics alone do not flag pmm clearly on this frame, so that is not
asserted: the same 120 chains with ``method="pmm"`` give a largest |z| of
3.2 to 5.6 with the seeds 7803 to 7808 (3.2 with the committed seed), 4.9
to 7.1 with 300 chains; the largest is always the mean share of ``yes`` in
``b`` (pmm imputes ``yes`` slightly more often), followed by the
log-variance of the share of ``s`` in ``o`` and of ``b`` in ``g``. Both
pmm and the model-based methods draw from close to the same conditional
distribution here, and R's 200 chains bound the precision of the comparison.
"""

import functools
import sys

import numpy as np
import pytest

from ._helpers import decode_frame, load_fixture
from ._l2_stats import Z_MAX, moments, z_logvar, z_mean

pytestmark = pytest.mark.parity

FIXTURE = load_fixture("l2_chain_default")
N_CHAINS = 120
SEED = 7803


def _frame():
    return decode_frame(FIXTURE["data"])


@functools.cache
def chain_run():
    from imputation import MICE

    mice = MICE(_frame())
    with pytest.warns(UserWarning, match=f"{N_CHAINS} imputations is a large number"):
        mice.impute(n_imputations=N_CHAINS, maxit=FIXTURE["maxit"], seed=SEED)
    return mice


def _method_mismatch(mice):
    """What differs from R's methods and visit sequence, or None.

    R's visitSequence also lists the complete column x, which it never
    imputes; the comparison uses the columns with a method.
    """
    got = {str(col): meth for col, meth in mice.method.items()}
    if got != FIXTURE["method"]:
        return f"methods {got} differ from R's {FIXTURE['method']}"
    expected = [col for col in FIXTURE["visit_sequence"] if FIXTURE["method"][col] != ""]
    if [str(col) for col in mice.visit_sequence] != expected:
        return f"visit sequence {mice.visit_sequence} differs from R's {expected}"
    return None


def test_default_methods_match_r():
    mice = chain_run()
    assert _method_mismatch(mice) is None, _method_mismatch(mice)
    assert mice.logged_events.empty  # R logged nothing either


def _python_values(mice, entry):
    """One value per Python chain of a statistic of the fixture."""
    column = entry["column"]
    if entry["statistic"] in ("chain_mean", "chain_var"):
        return np.asarray(getattr(mice, entry["statistic"])[column])[-1, :]
    missing = _frame()[column].isna().to_numpy()
    return np.array(
        [np.mean(completed[column][missing].to_numpy() == entry["level"]) for completed in mice.imputed_datasets]
    )


def _entry_id(entry):
    return ":".join(str(part) for part in (entry["column"], entry["statistic"], entry["level"]) if part is not None)


@pytest.mark.parametrize("entry", FIXTURE["statistics"], ids=[_entry_id(e) for e in FIXTURE["statistics"]])
def test_last_iteration_statistics_match_r(entry):
    mice = chain_run()
    values = _python_values(mice, entry)
    assert values.shape == (N_CHAINS,) and np.isfinite(values).all()
    py, m = moments(values), FIXTURE["m"]
    zm = float(z_mean(py["mean"], py["var"], N_CHAINS, entry["mean"], entry["var"], m))
    zv = float(z_logvar(py["var"], py["kurt"], N_CHAINS, entry["var"], entry["kurt"], m))
    assert max(abs(zm), abs(zv)) <= Z_MAX, (
        f"{_entry_id(entry)}: z mean {zm:+.2f}, z log-variance {zv:+.2f} (pass: |z| <= {Z_MAX:g})"
    )


def test_pmm_for_every_column_is_flagged(monkeypatch):
    """The 0.2 default (pmm for every column) in place of make_method: the methods differ from R's."""
    from imputation import MICE

    mice_module = sys.modules["imputation.MICE"]
    calls = []

    def pmm_everywhere(data, default_method):
        calls.append(1)
        return {col: ("pmm" if data[col].isna().any() else "") for col in data.columns}

    monkeypatch.setattr(mice_module, "make_method", pmm_everywhere)
    mice = MICE(_frame())
    mice.impute(n_imputations=1, maxit=1, seed=SEED)
    assert calls, "mutant not applied: impute no longer calls imputation.MICE.make_method"
    assert _method_mismatch(mice) is not None
