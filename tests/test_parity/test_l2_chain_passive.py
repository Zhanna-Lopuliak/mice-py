"""Layer 2 at chain level with a passive column and ``post``: pmm, a passive product and bounds by ``post``.

R ran ``mice(data, m = 200, maxit = 5, seed = 7842)`` on a numeric frame of
100 rows (see ``tests/reference/r/l2_chain_passive.R``): ``x`` complete,
``y1`` and ``y2`` incomplete and imputed by pmm, ``p = y1 * y2`` passive
(R's method ``"~ I(y1 * y2)"``, missing where an input is missing; it does
not predict ``y1`` and ``y2``), ``y3`` incomplete, depending on the product,
imputed by pmm from ``x``, ``y1``, ``y2`` and ``p`` and bounded to [2.5, 8]
by ``post`` (R's ``squeeze``). Python runs ``MICE(frame).impute(
n_imputations=N_CHAINS, maxit=5, seed=SEED, predictor_matrix=..., passive=
{"p": ...}, post={"y3": ...})`` with the same recipes (``_helpers``), and
must choose the same methods and visit sequence (``p`` last). Each statistic
of the last iteration is a sample over the chains on each side: the chain
mean and the chain variance of ``y1``, ``y2``, ``p`` and ``y3`` (over the
where-cells, after ``post``), and the share of the imputations of ``y3`` at
the lower and at the upper bound (about 13% and 31% in R). Their means and
log-variances are compared with the z-scores of T2 (pass: ``|z| <=
Z_MAX``).

120 Python chains take about 2.5 seconds. Measured largest |z| over the 20
z-scores: 1.69 with the committed seed (the log-variance of the chain
variance of ``p``), 1.44 to 2.75 with the seeds 7844 to 7851.

Mutant: ``post`` not applied (``MICE._apply_post``). The imputations of
``y3`` are then the donors' values, also outside the bounds: no imputation
lies at a bound in any chain (z of the mean -30.8 at the lower and -84.7 at
the upper bound, log-variance -inf), the chain variance of ``y3`` grows (z
of the mean +45.1, of the log-variance +14.3) and its chain mean rises
(+28.0), as more donors lie above 8 than below 2.5; with the seeds 7844 to
7846 the chain variance at +42.8 to +47.3 and the chain mean at +25.0 to
+28.2. The test asserts that the chain mean of ``y3`` is flagged by more
than twice the threshold. Measured as well, not tested: a passive column
never computed (it keeps its start values, sampled observed products) is
flagged by the chain variance of ``p`` (z of the mean -20.6 to -23.1 with
the seeds 7843 to 7846); one computed in the first iteration only and then
kept is not (largest |z| 3.7 to 4.7).
"""

import functools

import numpy as np
import pandas as pd
import pytest

from imputation import MICE

from ._helpers import decode_frame, load_fixture, passive_function, post_function, str_list
from ._l2_stats import Z_MAX, moments, z_logvar, z_mean

pytestmark = pytest.mark.parity

FIXTURE = load_fixture("l2_chain_passive")
N_CHAINS = 120
SEED = 7843
BOUNDS = {"share_lower": FIXTURE["post"]["y3"]["lower"], "share_upper": FIXTURE["post"]["y3"]["upper"]}


def _frame():
    return decode_frame(FIXTURE["data"])


def _predictor_matrix():
    payload = FIXTURE["predictor_matrix"]
    return pd.DataFrame(
        np.array(payload["values"], dtype=int), index=str_list(payload["rows"]), columns=str_list(payload["cols"])
    )


def _impute():
    mice = MICE(_frame())
    with pytest.warns(UserWarning, match=f"{N_CHAINS} imputations is a large number"):
        mice.impute(
            n_imputations=N_CHAINS,
            maxit=FIXTURE["maxit"],
            seed=SEED,
            predictor_matrix=_predictor_matrix(),
            passive={col: passive_function(recipe) for col, recipe in FIXTURE["passive"].items()},
            post={col: post_function(recipe) for col, recipe in FIXTURE["post"].items()},
        )
    return mice


@functools.cache
def chain_run():
    return _impute()


def test_methods_and_visit_sequence_match_r():
    """R's visitSequence also lists the complete x, which it never imputes; the passive p comes last."""
    mice = chain_run()
    method = {str(col): ("passive" if meth.startswith("~") else meth) for col, meth in FIXTURE["method"].items()}
    assert {str(col): meth for col, meth in mice.method.items()} == method
    expected = [col for col in FIXTURE["visit_sequence"] if FIXTURE["method"][col] != ""]
    assert [str(col) for col in mice.visit_sequence] == expected == ["y1", "y2", "y3", "p"]
    assert mice.logged_events.empty  # R logged nothing either


def _python_values(mice, entry):
    """One value per Python chain of a statistic of the fixture, over the where-cells of its column."""
    column = entry["column"]
    if entry["statistic"] in ("chain_mean", "chain_var"):
        return np.asarray(getattr(mice, entry["statistic"])[column])[-1, :]
    where = mice.where[column].to_numpy()
    bound = BOUNDS[entry["statistic"]]
    return np.array([np.mean(completed[column].to_numpy()[where] == bound) for completed in mice.imputed_datasets])


def _entry_id(entry):
    return f"{entry['column']}:{entry['statistic']}"


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


def test_the_passive_column_is_the_product_in_every_completed_dataset():
    """p is visited after y1 and y2, so its imputations are the product of theirs, row by row."""
    mice = chain_run()
    where = mice.where["p"].to_numpy()
    for completed in mice.imputed_datasets:
        product = (completed["y1"] * completed["y2"]).to_numpy()[where]
        np.testing.assert_allclose(completed["p"].to_numpy()[where], product, rtol=1e-12)


def test_post_not_applied_is_flagged(monkeypatch):
    calls = []
    monkeypatch.setattr(MICE, "_apply_post", lambda self, working, imp, col, *args: calls.append(col))
    mice = _impute()
    assert "y3" in calls, "mutant not applied"
    entry = next(e for e in FIXTURE["statistics"] if _entry_id(e) == "y3:chain_mean")
    with pytest.raises(AssertionError, match=r"y3:chain_mean: z mean"):
        check_statistic(mice, entry)
    zm, _ = _z_scores(mice, entry)
    assert zm > 2 * Z_MAX, zm
    # no imputation of y3 lies at a bound
    for statistic in BOUNDS:
        assert not _python_values(mice, {"column": "y3", "statistic": statistic}).any()
