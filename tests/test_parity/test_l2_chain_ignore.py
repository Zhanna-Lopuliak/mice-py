"""Layer 2 at chain level with ``ignore``: R's default methods, with 40 rows from a shifted distribution ignored.

R ran ``mice(data, m = 200, maxit = 5, ignore = ignore, seed = 7822)``
without ``method`` on the mixed frame of ``l2_chain_default`` (``x`` numeric
and complete, ``y`` numeric, ``b`` a factor no / yes, ``g`` a factor a / b /
c, ``o`` an ordered factor p < q < r < s; 100 rows) with 40 added rows that
are ignored (see ``tests/reference/r/l2_chain_ignore.R``). The added rows
come from a clearly shifted distribution (``y`` higher given the latent
variable, other frequencies of the categories) and have missing cells too,
which are imputed from the models of the first 100 rows. Python runs
``MICE(frame).impute(n_imputations=N_CHAINS, maxit=5, seed=SEED,
ignore=ignore)``, also without ``method``, and must choose the same methods.
Each statistic of the last iteration is taken separately over the imputed
cells of the rows that are not ignored ("kept") and of the ignored rows, and
is a sample over the chains on each side: the mean and the variance of the
imputed values of ``y``, and the share of each level but the first among the
imputed values of ``b``, ``g`` and ``o``. Their means and log-variances are
compared with the z-scores of T2 (pass: ``|z| <= Z_MAX``).

120 Python chains take about 5 seconds. Measured largest |z| over the 32
z-scores: 1.56 with the committed seed (the mean share of ``yes`` in ``b``,
kept rows), 1.60 to 3.08 with the seeds 7824 to 7831.

Mutant: ``ignore`` not applied to the fit rows (``_fit_rows`` without it).
The models are then fitted on the shifted rows too, and the imputations of
both groups move. With the committed seed 8 of the 16 statistics are
flagged, most clearly the mean of the imputed ``y`` (z of the mean +26.4 in
the kept rows, +27.5 in the ignored rows), its variance (+26.2, +20.5) and
the shares of ``s`` in ``o`` and of ``yes`` in ``b`` in the ignored rows
(-20.8, -13.8); 8 and 9 statistics, the mean of ``y`` in the kept rows at
+28.2 and +28.0, with the seeds 7824 and 7825. The test asserts that the
mean of the imputed ``y`` of the kept rows is flagged by more than twice
the threshold.
"""

import functools
import sys

import numpy as np
import pytest

from ._helpers import as_bool, decode_frame, load_fixture
from ._l2_stats import Z_MAX, moments, z_logvar, z_mean

pytestmark = pytest.mark.parity

FIXTURE = load_fixture("l2_chain_ignore")
N_CHAINS = 120
SEED = 7823
IGNORE = as_bool(FIXTURE["ignore"])


def _frame():
    return decode_frame(FIXTURE["data"])


def _impute():
    from imputation import MICE

    mice = MICE(_frame())
    with pytest.warns(UserWarning, match=f"{N_CHAINS} imputations is a large number"):
        mice.impute(n_imputations=N_CHAINS, maxit=FIXTURE["maxit"], seed=SEED, ignore=IGNORE)
    return mice


@functools.cache
def chain_run():
    return _impute()


def test_frame_is_that_of_l2_chain_default_with_40_ignored_rows():
    frame = _frame()
    assert len(frame) == 140 and IGNORE.tolist() == [False] * 100 + [True] * 40
    assert frame.iloc[:100].equals(decode_frame(load_fixture("l2_chain_default")["data"]))
    # missing cells in the ignored rows too
    assert (frame.loc[IGNORE, ["y", "b", "g", "o"]].isna().sum() >= 8).all()


def test_default_methods_match_r():
    """R's visitSequence also lists the complete column x, which it never imputes."""
    mice = chain_run()
    assert {str(col): meth for col, meth in mice.method.items()} == FIXTURE["method"]
    expected = [col for col in FIXTURE["visit_sequence"] if FIXTURE["method"][col] != ""]
    assert [str(col) for col in mice.visit_sequence] == expected
    np.testing.assert_array_equal(mice.ignore, IGNORE)
    assert mice.logged_events.empty  # R logged nothing either


def _python_values(mice, entry):
    """One value per Python chain of a statistic of the fixture, over the imputed cells of its group of rows."""
    column = entry["column"]
    cells = mice.where[column].to_numpy() & (IGNORE == (entry["rows"] == "ignored"))
    values = [completed[column].to_numpy()[cells] for completed in mice.imputed_datasets]
    if entry["statistic"] == "mean":
        return np.array([np.mean(v.astype(float)) for v in values])
    if entry["statistic"] == "var":
        return np.array([np.var(v.astype(float), ddof=1) for v in values])
    return np.array([np.mean(v == entry["level"]) for v in values])


def _entry_id(entry):
    parts = (entry["column"], entry["statistic"], entry["level"], entry["rows"])
    return ":".join(str(part) for part in parts if part is not None)


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


def test_ignore_not_applied_to_the_fit_rows_is_flagged(monkeypatch):
    mice_module = sys.modules["imputation.MICE"]
    calls = []

    def without_ignore(cx, observed, ignore):
        calls.append(bool((cx & observed & ignore).any()))
        return cx & observed

    monkeypatch.setattr(mice_module, "_fit_rows", without_ignore)
    mice = _impute()
    assert any(calls), "mutant not applied: no ignored fit row"
    entry = next(e for e in FIXTURE["statistics"] if _entry_id(e) == "y:mean:kept")
    with pytest.raises(AssertionError, match=r"y:mean:kept: z mean"):
        check_statistic(mice, entry)
    zm, _ = _z_scores(mice, entry)
    assert zm > 2 * Z_MAX, zm
