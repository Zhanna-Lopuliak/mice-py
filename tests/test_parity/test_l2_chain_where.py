"""Layer 2 at chain level under ``where="all"``: R's default methods with every cell a where-cell.

R ran ``mice(data, m = 200, maxit = 5, where = "all", seed = 7805)`` without
``method`` on the mixed frame of ``l2_chain_default`` (``x`` numeric and
complete, ``y`` numeric, ``b`` a factor no / yes, ``g`` a factor a / b / c,
``o`` an ordered factor p < q < r < s; see
``tests/reference/r/l2_chain_where.R``). Every cell is a where-cell, so R's
``make.method`` gives the complete ``x`` pmm too, and each column is fitted
on its observed values and imputes all 100 rows, while the chains predict
the other columns from the observed values. Python runs
``MICE(frame).impute(n_imputations=N_CHAINS, maxit=5, seed=SEED,
where="all")``, also without ``method``, and must choose the same methods.
Each statistic of the last iteration is taken over the where-cells and is a
sample over the chains on each side: the mean and the variance of the
imputed values of ``x`` and ``y`` (``chainMean`` and ``chainVar``), and the
share of each level but the first among the imputed values of ``b``, ``g``
and ``o``. Their means and log-variances are compared with the z-scores of
T2 (pass: ``|z| <= Z_MAX``).

120 Python chains take about 5.5 seconds. Measured largest |z| over the 20
z-scores: 2.40 with the committed seed (the log-variance of the share of
``r`` in ``o``), 1.96 to 2.60 with the seeds 7807 to 7814.

Mutant: the imputations of the observed where-cells written to the data the
chains work on (``_refresh_working_column`` writing every where-cell). Each
column is then fitted on the imputations of the last iteration instead of
its observed values, and the chains drift apart: the variance over the
chains of every statistic grows. With the committed seed all 10 statistics
are flagged by their log-variance (z +6.3 to +13.7), most clearly
``chainMean`` of ``x`` (+13.7) and of ``y`` (+13.6); the largest |z| is
13.3 to 15.5 with the seeds 7807 to 7811. The test asserts the first.
"""

import functools
import sys

import numpy as np
import pytest

from ._helpers import decode_frame, load_fixture
from ._l2_stats import Z_MAX, moments, z_logvar, z_mean

pytestmark = pytest.mark.parity

FIXTURE = load_fixture("l2_chain_where")
N_CHAINS = 120
SEED = 7806


def _frame():
    return decode_frame(FIXTURE["data"])


def _impute():
    from imputation import MICE

    mice = MICE(_frame())
    with pytest.warns(UserWarning, match=f"{N_CHAINS} imputations is a large number"):
        mice.impute(n_imputations=N_CHAINS, maxit=FIXTURE["maxit"], seed=SEED, where=FIXTURE["where"])
    return mice


@functools.cache
def chain_run():
    return _impute()


def test_frame_is_that_of_l2_chain_default():
    assert _frame().equals(decode_frame(load_fixture("l2_chain_default")["data"]))


def test_default_methods_match_r():
    """R's visitSequence lists every column; under where="all" each has a method."""
    mice = chain_run()
    assert {str(col): meth for col, meth in mice.method.items()} == FIXTURE["method"]
    assert [str(col) for col in mice.visit_sequence] == FIXTURE["visit_sequence"]
    assert mice.where.to_numpy().all()
    assert mice.logged_events.empty  # R logged nothing either


def _python_values(mice, entry):
    """One value per Python chain of a statistic of the fixture, over the where-cells."""
    column = entry["column"]
    if entry["statistic"] in ("chain_mean", "chain_var"):
        return np.asarray(getattr(mice, entry["statistic"])[column])[-1, :]
    where = mice.where[column].to_numpy()
    return np.array(
        [np.mean(completed[column][where].to_numpy() == entry["level"]) for completed in mice.imputed_datasets]
    )


def _entry_id(entry):
    return ":".join(str(part) for part in (entry["column"], entry["statistic"], entry["level"]) if part is not None)


def _z_scores(mice, entry):
    """T2 for one statistic of the last iteration: the z-scores of its mean and log-variance over the chains."""
    values = _python_values(mice, entry)
    assert values.shape == (N_CHAINS,) and np.isfinite(values).all()
    py, m = moments(values), FIXTURE["m"]
    zm = float(z_mean(py["mean"], py["var"], N_CHAINS, entry["mean"], entry["var"], m))
    zv = float(z_logvar(py["var"], py["kurt"], N_CHAINS, entry["var"], entry["kurt"], m))
    return zm, zv


def check_statistic(mice, entry):
    zm, zv = _z_scores(mice, entry)
    assert max(abs(zm), abs(zv)) <= Z_MAX, (
        f"{_entry_id(entry)}: z mean {zm:+.2f}, z log-variance {zv:+.2f} (pass: |z| <= {Z_MAX:g})"
    )


@pytest.mark.parametrize("entry", FIXTURE["statistics"], ids=[_entry_id(e) for e in FIXTURE["statistics"]])
def test_last_iteration_statistics_match_r(entry):
    check_statistic(chain_run(), entry)


def test_observed_where_cells_written_to_the_working_data_are_flagged(monkeypatch):
    mice_module = sys.modules["imputation.MICE"]
    calls = []

    def every_where_cell(working, column, imp, where, observed):
        calls.append((where & observed).any())
        working.loc[where, column] = mice_module._values(imp)

    monkeypatch.setattr(mice_module, "_refresh_working_column", every_where_cell)
    mice = _impute()
    assert any(calls), "mutant not applied: no observed where-cell refreshed"
    entry = next(e for e in FIXTURE["statistics"] if _entry_id(e) == "x:chain_mean")
    with pytest.raises(AssertionError, match=r"x:chain_mean: z mean"):
        check_statistic(mice, entry)
    # the chains drift apart: the chain means of x vary far more than R's
    _, zv = _z_scores(mice, entry)
    assert zv > 2 * Z_MAX, zv
