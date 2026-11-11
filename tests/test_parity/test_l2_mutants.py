"""Tests of the layer-2 tests: re-introduce an old defect, check it is caught.

Each mutant is written so that it is valid once the library is fixed:

* A1: ``np.linalg.cholesky`` patched to return the transpose. With the
  draw ``beta = coef + L @ z * sigma`` this reproduces the old draw, and T1,
  T2 and T3 on F1 must each flag it.
* A9: ``imputation.PMM.matchindex`` replaced by a matcher with
  deterministic tie-breaking (lowest index first, as the KD-tree did); T1 on
  F2 must flag it.
* A6: ``cart(..., bootstrap=True)``, the old bootstrapped tree; the donor-set
  test on F1 must flag it.
* ``norm`` without the parameter draw: ``imputation.norm.norm_nob`` (least
  squares coefficients, residual sd from the fit) compared with R's ``norm``
  on F1. Ignoring the parameter uncertainty narrows the imputations in the
  cells where the coefficients are poorly determined: T1 (decile bins), T2
  (cell variance) and T3 (coefficient variance) must each flag it. Measured
  with 3,000 calls: smallest T1 p about 1e-206, largest |z| about 43 (T2) and
  100 (T3). On F4, where every missing row lies in the bulk of the observed
  ones, only T3 flags it clearly (|z| about 17; T2 about 5.5, T1 p 0.002).
"""

import functools

import numpy as np
import pytest

from ._l2_harness import compare, donor_set_mismatches, run_python

pytestmark = pytest.mark.parity


@functools.cache
def transposed_cholesky_comparison():
    """pmm on F1 with np.linalg.cholesky returning the transposed factor."""
    calls = []
    original = np.linalg.cholesky

    def transposed(a, *args, **kwargs):
        calls.append(1)
        return np.swapaxes(original(a, *args, **kwargs), -1, -2)

    with pytest.MonkeyPatch.context() as mp:
        mp.setattr(np.linalg, "cholesky", transposed)
        inp, r_sum, py_sum = run_python("pmm", "F1")
    return compare(r_sum, py_sum, inp), len(calls)


@pytest.mark.parametrize("statistic", ["T1", "T2", "T3"])
def test_transposed_cholesky_is_flagged(statistic):
    comp, n_calls = transposed_cholesky_comparison()
    assert n_calls > 0, "mutant not applied: pmm no longer calls np.linalg.cholesky"
    passed = {"T1": comp.t1_ok, "T2": comp.t2_ok, "T3": comp.t3_ok}[statistic]()
    message = {"T1": comp.t1_message, "T2": comp.t2_message, "T3": comp.t3_message}
    assert not passed, f"{statistic} does not flag the mutant: " + message[statistic]()


def deterministic_matchindex(d, t, k=5, rng=None):
    """k nearest donors with ties broken by position (KD-tree behaviour)."""
    d = np.asarray(d, dtype=float).ravel()
    t = np.asarray(t, dtype=float).ravel()
    rng = np.random.default_rng() if rng is None else rng
    k = int(min(max(k, 1), len(d)))
    nearest = np.argsort(np.abs(d[None, :] - t[:, None]), axis=1, kind="stable")[:, :k]
    return nearest[np.arange(len(t)), rng.integers(k, size=len(t))]


def test_deterministic_ties_are_flagged_by_t1(monkeypatch):
    import imputation.PMM as pmm_module

    calls = []

    def counting(*args, **kwargs):
        calls.append(1)
        return deterministic_matchindex(*args, **kwargs)

    monkeypatch.setattr(pmm_module, "matchindex", counting)
    inp, r_sum, py_sum = run_python("pmm", "F2")
    comp = compare(r_sum, py_sum, inp)
    assert calls, "mutant not applied: pmm does not call imputation.PMM.matchindex"
    assert not comp.t1_ok(), "T1 does not flag deterministic ties: " + comp.t1_message()


@functools.cache
def norm_without_parameter_draw_comparison():
    """Python norm_nob in place of norm, against R's norm on F1."""
    from imputation.norm import norm_nob

    calls = []

    def counting(*args, **kwargs):
        calls.append(1)
        return norm_nob(*args, **kwargs)

    inp, r_sum, py_sum = run_python("norm", "F1", func=counting)
    return compare(r_sum, py_sum, inp), len(calls)


@pytest.mark.parametrize("statistic", ["T1", "T2", "T3"])
def test_norm_without_parameter_draw_is_flagged(statistic):
    comp, n_calls = norm_without_parameter_draw_comparison()
    assert n_calls > 0, "mutant not applied: norm_nob was not called"
    assert comp.kind == "continuous"
    passed = {"T1": comp.t1_ok, "T2": comp.t2_ok, "T3": comp.t3_ok}[statistic]()
    message = {"T1": comp.t1_message, "T2": comp.t2_message, "T3": comp.t3_message}
    assert not passed, f"{statistic} does not flag the mutant: " + message[statistic]()


def test_bootstrapped_cart_is_flagged_by_donor_sets():
    inp, r_sum, py_sum = run_python("cart", "F1", bootstrap=True, cp=1e-4, min_samples_leaf=5)
    assert donor_set_mismatches(r_sum, py_sum), "the donor-set test does not flag a bootstrapped tree"
