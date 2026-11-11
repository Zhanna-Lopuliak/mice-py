"""Statistics for the layer-2 distribution parity tests.

R and NumPy use different random number generators, so imputed values cannot
be compared one for one. Layer 2 instead calls an imputation method many times
on a fixed input in R and in Python and compares the two samples of imputed
values per missing cell:

* T1: chi-square homogeneity test of the donor counts (R against Python),
  pooling categories with small expected counts; pass if p >= ``P_MIN``.
* T2: z-scores for the mean and for the log-variance of the imputed value;
  pass if ``|z| <= Z_MAX``.
* T3: the same two z-scores for each OLS coefficient of the completed y.

The z-score for the variance uses the delta method: for a sample of size N
with kurtosis k (k = 3 for a normal distribution), ``Var(log s^2)`` is about
``(k - 1) / N``. Kurtosis here is ``m4 / m2**2``, not the excess kurtosis.
"""

import numpy as np
from scipy import stats

P_MIN = 1e-7
Z_MAX = 5.5
MIN_EXPECTED = 10.0


def kurtosis(x, axis=0):
    """Return ``m4 / m2**2`` along ``axis``; NaN where the sample is constant."""
    x = np.asarray(x, dtype=float)
    d = x - x.mean(axis=axis, keepdims=True)
    m2 = (d**2).mean(axis=axis)
    m4 = (d**4).mean(axis=axis)
    with np.errstate(invalid="ignore", divide="ignore"):
        k = m4 / m2**2
    return np.where(m2 > 0, k, np.nan)


def moments(x, axis=0):
    """Return a dict with ``mean``, ``var`` (ddof=1) and ``kurt`` along ``axis``."""
    x = np.asarray(x, dtype=float)
    return {
        "mean": x.mean(axis=axis),
        "var": x.var(axis=axis, ddof=1),
        "kurt": kurtosis(x, axis=axis),
    }


def pool_small_categories(counts_r, counts_p, min_expected=MIN_EXPECTED):
    """Return the 2 x K' table of (R, Python) counts after pooling.

    Categories never observed in either sample are dropped. A category is
    small when its expected count under homogeneity is below
    ``min_expected`` in either sample; all small categories are merged into one
    bin, and while that bin is still small it absorbs the smallest remaining
    category. The result always has at least one column.
    """
    r = np.asarray(counts_r, dtype=float)
    p = np.asarray(counts_p, dtype=float)
    if r.shape != p.shape or r.ndim != 1:
        raise ValueError("counts must be 1-D arrays of the same length")
    n_r, n_p = r.sum(), p.sum()
    if n_r <= 0 or n_p <= 0:
        raise ValueError("both samples must contain at least one draw")
    total = r + p
    keep = total > 0
    r, p, total = r[keep], p[keep], total[keep]
    # min over the two samples of the expected count of each category
    threshold = min_expected * (n_r + n_p) / min(n_r, n_p)
    small = total < threshold
    big_r, big_p = list(r[~small]), list(p[~small])
    pool_r, pool_p = r[small].sum(), p[small].sum()
    while pool_r + pool_p > 0 and pool_r + pool_p < threshold and big_r:
        j = int(np.argmin(np.add(big_r, big_p)))
        pool_r += big_r.pop(j)
        pool_p += big_p.pop(j)
    if pool_r + pool_p > 0:
        big_r.append(pool_r)
        big_p.append(pool_p)
    return np.array([big_r, big_p])


def chi2_homogeneity(counts_r, counts_p, min_expected=MIN_EXPECTED):
    """Pearson chi-square test that two count vectors share one distribution.

    Returns ``(statistic, df, p_value)``. With a single category after pooling
    there is nothing to test and the p-value is 1.
    """
    table = pool_small_categories(counts_r, counts_p, min_expected)
    k = table.shape[1]
    if k < 2:
        return 0.0, 0, 1.0
    row = table.sum(axis=1, keepdims=True)
    col = table.sum(axis=0, keepdims=True)
    expected = row * col / table.sum()
    statistic = float(((table - expected) ** 2 / expected).sum())
    df = k - 1
    return statistic, df, float(stats.chi2.sf(statistic, df))


def _ratio_or_inf(num, den):
    """num / den, with 0/0 -> 0 and x/0 -> +-inf (elementwise)."""
    num = np.asarray(num, dtype=float)
    den = np.asarray(den, dtype=float)
    with np.errstate(divide="ignore", invalid="ignore"):
        out = num / den
        signed_inf = np.sign(num) * np.inf  # NaN where num is 0; not selected below
    out = np.where((den == 0) & (num == 0), 0.0, out)
    out = np.where((den == 0) & (num != 0), signed_inf, out)
    return out


def z_mean(mean_p, var_p, n_p, mean_r, var_r, n_r):
    """``(mean_P - mean_R) / sqrt(s2_P/N_P + s2_R/N_R)``, elementwise.

    Where both variances are zero the score is 0 if the means agree and
    infinite otherwise.
    """
    se = np.sqrt(np.asarray(var_p, float) / n_p + np.asarray(var_r, float) / n_r)
    return _ratio_or_inf(np.asarray(mean_p, float) - np.asarray(mean_r, float), se)


def z_logvar(var_p, kurt_p, n_p, var_r, kurt_r, n_r):
    """``log(s2_P/s2_R) / sqrt((k_P - 1)/N_P + (k_R - 1)/N_R)``, elementwise.

    Both variances zero gives 0; exactly one of them zero gives +-inf. The
    kurtosis of a constant sample (NaN) does not enter in those cases.
    """
    var_p = np.asarray(var_p, dtype=float)
    var_r = np.asarray(var_r, dtype=float)
    k_p = np.nan_to_num(np.asarray(kurt_p, dtype=float), nan=1.0)
    k_r = np.nan_to_num(np.asarray(kurt_r, dtype=float), nan=1.0)
    both_zero = (var_p == 0) & (var_r == 0)
    one_zero = (var_p == 0) ^ (var_r == 0)
    with np.errstate(divide="ignore", invalid="ignore"):
        log_ratio = np.log(var_p) - np.log(var_r)
    se = np.sqrt(np.maximum(k_p - 1, 0) / n_p + np.maximum(k_r - 1, 0) / n_r)
    z = _ratio_or_inf(np.where(both_zero | one_zero, 0.0, log_ratio), se)
    z = np.where(one_zero, np.where(var_p > var_r, np.inf, -np.inf), z)
    return np.where(both_zero, 0.0, z)
