"""Unit tests of the layer-2 statistics on synthetic data with known answers.

These check that T1 (chi-square homogeneity with pooling) and the z-scores of
T2/T3 are calibrated under the null (same distribution on both sides, sample
sizes as in the parity tests) and that they flag differences of the size the
parity tests are meant to catch.
"""

import numpy as np
import pytest
from scipy import stats

from ._l2_stats import (
    MIN_EXPECTED,
    P_MIN,
    Z_MAX,
    chi2_homogeneity,
    kurtosis,
    moments,
    pool_small_categories,
    z_logvar,
    z_mean,
)

pytestmark = pytest.mark.parity

N_R, N_P = 20_000, 3_000


def test_kurtosis_of_known_distributions():
    rng = np.random.default_rng(1)
    assert kurtosis(rng.normal(size=400_000)) == pytest.approx(3.0, abs=0.05)
    assert kurtosis(rng.uniform(size=400_000)) == pytest.approx(1.8, abs=0.02)
    assert kurtosis(rng.exponential(size=400_000)) == pytest.approx(9.0, abs=0.5)
    assert kurtosis([1.0, -1.0, 1.0, -1.0]) == pytest.approx(1.0)
    assert np.isnan(kurtosis([2.0, 2.0, 2.0]))


def test_moments_match_numpy_along_axis():
    x = np.random.default_rng(2).gamma(2.0, size=(500, 3))
    m = moments(x)
    np.testing.assert_allclose(m["mean"], x.mean(axis=0))
    np.testing.assert_allclose(m["var"], x.var(axis=0, ddof=1))
    np.testing.assert_allclose(m["kurt"], stats.kurtosis(x, axis=0, fisher=False))


def test_pooling_merges_small_categories_into_one_bin():
    r = [10_000, 9_000, 1_000, 5, 0]
    p = [1_500, 1_350, 150, 0, 0]
    # threshold on the total count: 10 * 23_005 / 3_000 = 76.7; only the
    # category with 5 draws is small, and the pooled bin then absorbs the
    # smallest remaining category (1_150 draws)
    np.testing.assert_array_equal(pool_small_categories(r, p), [[10_000, 9_000, 1_005], [1_500, 1_350, 150]])


def test_pooling_leaves_every_expected_count_at_least_min_expected():
    rng = np.random.default_rng(3)
    for _ in range(200):
        k = rng.integers(2, 40)
        prob = rng.dirichlet(np.full(k, 0.3))
        table = pool_small_categories(rng.multinomial(N_R, prob), rng.multinomial(N_P, prob))
        assert table.sum() == N_R + N_P
        if table.shape[1] > 1:
            expected = table.sum(1, keepdims=True) * table.sum(0) / table.sum()
            assert expected.min() >= MIN_EXPECTED


def test_chi2_matches_scipy_when_nothing_is_pooled():
    r = np.array([5000, 7000, 8000])
    p = np.array([800, 1000, 1200])
    stat, df, pval = chi2_homogeneity(r, p)
    ref = stats.chi2_contingency(np.array([r, p]), correction=False)
    assert stat == pytest.approx(ref[0])
    assert df == ref[2]
    assert pval == pytest.approx(ref[1])


def test_chi2_with_a_single_category_is_not_a_failure():
    assert chi2_homogeneity([0, 20_000, 0], [0, 3_000, 0]) == (0.0, 0, 1.0)


def test_chi2_is_calibrated_under_the_null():
    # donor-like distribution: a few likely donors, a long tail of rare ones
    rng = np.random.default_rng(4)
    prob = np.r_[0.2, 0.2, 0.2, 0.15, 0.15, np.full(20, 0.1 / 20)]
    pvals = np.array([chi2_homogeneity(rng.multinomial(N_R, prob), rng.multinomial(N_P, prob))[2] for _ in range(400)])
    assert 0.02 <= np.mean(pvals < 0.05) <= 0.09
    assert pvals.min() > 1e-4


def test_chi2_flags_deterministic_tie_breaking():
    # R spreads a tied donor group of 20 evenly; a deterministic matcher
    # always returns the same 5 of them
    rng = np.random.default_rng(5)
    r = rng.multinomial(N_R, np.full(20, 1 / 20))
    p = rng.multinomial(N_P, np.r_[np.full(5, 1 / 5), np.zeros(15)])
    assert chi2_homogeneity(r, p)[2] < P_MIN


def test_chi2_flags_a_modest_shift_in_donor_probabilities():
    rng = np.random.default_rng(6)
    prob_r = np.array([0.3, 0.25, 0.2, 0.15, 0.1])
    prob_p = np.array([0.25, 0.25, 0.2, 0.15, 0.15])
    assert chi2_homogeneity(rng.multinomial(N_R, prob_r), rng.multinomial(N_P, prob_p))[2] < P_MIN


@pytest.mark.parametrize("dist", ["normal", "exponential", "discrete"])
def test_z_scores_are_calibrated_under_the_null(dist):
    rng = np.random.default_rng(7)
    values = rng.normal(size=8)

    def draw(n):
        if dist == "normal":
            return rng.normal(1.0, 2.0, size=n)
        if dist == "exponential":
            return rng.exponential(size=n)
        return rng.choice(values, size=n, p=np.r_[0.4, 0.2, 0.1, 0.1, 0.1, 0.05, 0.03, 0.02])

    zm, zv = [], []
    for _ in range(500):
        a, b = moments(draw(N_P)), moments(draw(N_R))
        zm.append(z_mean(a["mean"], a["var"], N_P, b["mean"], b["var"], N_R))
        zv.append(z_logvar(a["var"], a["kurt"], N_P, b["var"], b["kurt"], N_R))
    zm, zv = np.array(zm), np.array(zv)
    assert abs(zm.mean()) < 0.15 and 0.85 < zm.std() < 1.15
    assert abs(zv.mean()) < 0.15 and 0.85 < zv.std() < 1.2
    assert np.abs(np.r_[zm, zv]).max() < Z_MAX


def test_z_scores_flag_a_shift_and_a_variance_ratio():
    rng = np.random.default_rng(8)
    a = moments(rng.normal(0.0, 1.0, size=N_P))
    b = moments(rng.normal(0.0, 1.0, size=N_R))
    c = moments(rng.normal(0.15, 1.0, size=N_R))  # shift of 0.15 sd
    d = moments(rng.normal(0.0, np.sqrt(1.3), size=N_R))  # variance ratio 1.3
    assert abs(z_mean(a["mean"], a["var"], N_P, c["mean"], c["var"], N_R)) > Z_MAX
    assert abs(z_logvar(a["var"], a["kurt"], N_P, d["var"], d["kurt"], N_R)) > Z_MAX
    assert abs(z_mean(a["mean"], a["var"], N_P, b["mean"], b["var"], N_R)) < Z_MAX


def test_z_scores_with_zero_variances():
    # both constant and equal: no evidence of a difference
    assert z_mean(2.0, 0.0, N_P, 2.0, 0.0, N_R) == 0.0
    assert z_logvar(0.0, np.nan, N_P, 0.0, np.nan, N_R) == 0.0
    # both constant but different, or only one constant: flagged
    assert z_mean(2.0, 0.0, N_P, 3.0, 0.0, N_R) == -np.inf
    assert z_logvar(0.0, np.nan, N_P, 1.0, 3.0, N_R) == -np.inf
    assert z_logvar(1.0, 3.0, N_P, 0.0, np.nan, N_R) == np.inf
    # vectorised
    np.testing.assert_array_equal(
        z_logvar([0.0, 0.0], [np.nan, np.nan], N_P, [0.0, 2.0], [np.nan, 3.0], N_R),
        [0.0, -np.inf],
    )
