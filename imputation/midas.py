"""
Imputation by predictive mean matching with distance aided donor selection (``midas``).

The port of R mice's ``mice.impute.midastouch``; the title is that of R's help page.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from numpy.typing import ArrayLike

from .design import obtain_design


def bootfunc_plain(n: int, rng: np.random.Generator | None = None) -> np.ndarray:
    """
    Bootstrap weights of `n` observations, as R mice's ``bootfunc.plain``.

    This function simulates a nonparametric bootstrap by randomly drawing `n` integers
    from the range 1 to n (inclusive), with replacement. It returns the count of how many times
    each index (1-based) is selected, producing a frequency table that can be used
    as weights in e.g. MIDAS imputation.

    Parameters
    ----------
    n : int
        The number of observations to sample and also the length of the resulting weight vector.
    rng : np.random.Generator, optional
        Random number generator for reproducibility. If None, a fresh generator is used.

    Returns
    -------
    weights : ndarray of shape (n,)
        An array of integers indicating how often each index (1-based) was selected in the bootstrap sample.
    """
    if rng is None:
        rng = np.random.default_rng()
    # random n int with size n from 1:n
    random = rng.choice(n, size=n, replace=True) + 1
    # returns histogram of drawn ints
    table, _ = np.histogram(random, bins=np.arange(1, n + 2))
    return table


def minmax(x: ArrayLike, domin: bool = True, domax: bool = True) -> np.ndarray:
    """
    Keep values between ``sqrt(eps)`` and ``sqrt(max float)``, as R mice's ``minmax``.

    Parameters
    ----------
    x : array-like
        Values.
    domin : bool, default True
        Cap the values at ``sqrt(np.finfo(float).max)`` from above. (The
        names are R's; ``domin`` applies ``min``.)
    domax : bool, default True
        Raise the values to at least ``sqrt(np.finfo(float).eps)``.

    Returns
    -------
    np.ndarray
        The bounded values.
    """
    maxx = np.sqrt(np.finfo(float).max)
    minx = np.sqrt(np.finfo(float).eps)
    if domin:
        x = np.minimum(x, maxx)
    if domax:
        x = np.maximum(x, minx)
    return x


def midas(
    y: ArrayLike,
    id_obs: ArrayLike,
    x: ArrayLike | pd.DataFrame,
    id_mis: ArrayLike | None = None,
    ridge: float = 1e-5,
    kappa: float | None = None,
    outout: bool = True,
    rng: np.random.Generator | None = None,
) -> np.ndarray:
    """
    Imputation by predictive mean matching with distance aided donor selection.

    The port of R mice's ``mice.impute.midastouch``, whose help page has this
    title. As R's help page says, the imputation is based on Rubin (1987,
    p. 168, formulas a and b) and Siddique and Belin (2008); the R function
    is by Gaffert, Meinfelder and Bosch.

    It operates by weighting observed donors based on the similarity between predicted values,
    with optional leave-one-out model estimation for increased fidelity.

    Parameters
    ----------
    y : array-like of shape (n_samples,)
        The target variable with missing values to be imputed. Must be numeric or boolean;
        a boolean target is modelled as 0/1. Categorical and text targets raise ``ValueError``
        (R's midastouch does not impute factors either).

    id_obs : array-like of bool of shape (n_samples,)
        Logical array indicating observed values in `y`. True where `y` is observed, False where missing.

    x : array-like or pd.DataFrame of shape (n_samples, n_features)
        Predictor variables. Must be fully observed. A DataFrame with categorical,
        object, string or boolean columns is dummy-coded first by
        :func:`imputation.design.obtain_design`, as MICE does.

    id_mis : np.ndarray, optional
        Boolean mask of missing values to impute. If None, uses ~id_obs.

    ridge : float, default=1e-5
        Ridge penalty used in regularized regression to stabilize the solution in the presence of multicollinearity.

        - Set lower (e.g. 1e-6) to reduce bias in noisy data.
        - Set higher (e.g. 1e-4) if collinearity is suspected.

    kappa : float or None, default=None
        Controls the sharpness of donor weighting (R's ``midas.kappa``). If None, it is
        estimated from the R² of the bootstrap fit as described by Siddique and Belin (2008):
        100 for a perfect fit, 3 when R² is undefined. Siddique and Belin find 3 sensible.
        Through :meth:`imputation.MICE.impute` it is passed as ``midas_kappa``.

    outout : bool, default=True
        If True, uses leave-one-out regression for each donor (MI-proper).
        If False, a single model is estimated for all donors and recipients.
        WARNING: Setting `outout=False` gives in-sample predictions for the donors.

    rng : np.random.Generator, optional
        Random number generator for the bootstrap weights and the donor draw.
        If None, a fresh generator is used.

    Returns
    -------
    y_imp : np.ndarray
        Imputed values for missing positions only (matching R implementation): the
        observed values of the drawn donors, ``y[id_obs][index]``, so they keep the
        type of `y` (integers stay integers, booleans stay booleans).

    Raises
    ------
    ValueError
        If `y` has no observed value and some value is to be imputed, if `y`
        is not numeric or boolean, or if `x` is not numeric or has missing values.

    Notes
    -----
    The donor probabilities for given bootstrap weights are computed by
    :func:`_donor_probabilities`.

    References
    ----------
    The references of R's help page of ``mice.impute.midastouch``:

    - Gaffert, P., Meinfelder, F., Bosch, V. (2018). Towards an MI-proper
      Predictive Mean Matching. JSM 2018, discussion paper.
    - Little, R. J. A. (1988). Missing data adjustments in large surveys
      (with discussion). Journal of Business Economics and Statistics, 6,
      287-301.
    - Parzen, M., Lipsitz, S. R., Fitzmaurice, G. M. (2005). A note on
      reducing the bias of the approximate Bayesian bootstrap imputation
      variance estimator. Biometrika, 92(4), 971-974.
    - Rubin, D. B. (1987). Multiple Imputation for Nonresponse in Surveys.
      New York: Wiley.
    - Siddique, J., Belin, T. R. (2008). Multiple imputation using an
      iterative hot-deck with distance-based donor selection. Statistics in
      Medicine, 27(1), 83-102.
    - Van Buuren, S., Brand, J. P. L., Groothuis-Oudshoorn, C. G. M.,
      Rubin, D. B. (2006). Fully conditional specification in multivariate
      imputation. Journal of Statistical Computation and Simulation, 76(12),
      1049-1064.
    - Van Buuren, S., Groothuis-Oudshoorn, K. (2011). mice: Multivariate
      Imputation by Chained Equations in R. Journal of Statistical Software,
      45(3), 1-67.

    Examples
    --------
    >>> y = np.array([2.1, 3.9, np.nan, 8.2, 9.8, 12.3, np.nan, 15.7, 18.4, 19.9])
    >>> id_obs = ~np.isnan(y)
    >>> x = np.arange(10.0).reshape(-1, 1)  # one predictor
    >>> y_imp = midas(y, id_obs, x, rng=np.random.default_rng(1))
    >>> y_imp.shape  # one value per missing cell
    (2,)
    >>> bool(np.isin(y_imp, y[id_obs]).all())  # each one an observed value
    True
    """
    id_obs = np.asarray(id_obs, dtype=bool)
    id_mis = ~id_obs if id_mis is None else np.asarray(id_mis, dtype=bool)
    if not id_obs.any():
        if not id_mis.any():
            return np.asarray(y)[id_mis]
        raise ValueError(
            "midas: the target has no observed value in the rows used for the model, so there is no donor to draw from."
        )
    if rng is None:
        rng = np.random.default_rng()

    # P-step: bootstrap weights of the donors
    omega = bootfunc_plain(int(id_obs.sum()), rng)
    probs, _ = _donor_probabilities(y, id_obs, x, id_mis, omega, ridge=ridge, kappa=kappa, outout=outout)

    # One donor per missing cell; its observed value is the imputation
    index = _draw_donors(probs, rng)
    return np.asarray(y)[id_obs][index]


def _donor_probabilities(
    y: ArrayLike,
    id_obs: ArrayLike,
    x: ArrayLike | pd.DataFrame,
    id_mis: ArrayLike,
    omega: ArrayLike,
    ridge: float = 1e-5,
    kappa: float | None = None,
    outout: bool = True,
) -> tuple[np.ndarray, float]:
    """
    Donor probabilities of midas for given bootstrap weights.

    Everything in :func:`midas` that is deterministic once the bootstrap
    weights are drawn: the weighted ridge fit, kappa, the leave-one-out
    predictions (``outout=True``) and the conversion of distances into
    probabilities.

    Parameters
    ----------
    y, id_obs, x, id_mis, ridge, kappa, outout
        As in :func:`midas`; `id_mis` is required.
    omega : array-like of shape (n_obs,)
        Bootstrap counts of the observed rows, as drawn by :func:`bootfunc_plain`.

    Returns
    -------
    probs : np.ndarray of shape (n_obs, n_mis)
        Column j holds the probability of each observed row to be the donor of
        missing cell j, and sums to 1. These are R's ``probs`` just before its
        final draw. When several donors are at distance 0 (tied predictor rows),
        R's capped column sums leave columns that sum to more than 1, which R's
        ``sample()`` rescales; the rescaled probabilities are returned.
    kappa : float
        The kappa used: the argument if given, else the estimate.
    """
    id_obs = np.asarray(id_obs, dtype=bool)
    id_mis = np.asarray(id_mis, dtype=bool)
    x = _predictor_matrix(x)
    yobs = _numeric_target(y, id_obs)
    omega = np.asarray(omega, dtype=float)

    sminx = np.finfo(float).eps ** (1 / 4)
    x = np.c_[np.ones(x.shape[0]), x]
    nobs = len(yobs)
    m = x.shape[1]
    xobs = x[id_obs, :]
    xmis = x[id_mis, :]
    diag = np.arange(m)

    # Weighted least squares on the bootstrap sample; the ridge penalty
    # scales the diagonal of the slopes, not of the intercept
    cx = omega[:, np.newaxis] * xobs
    xcx = xobs.T @ cx
    if ridge > 0:
        xcx[diag[1:], diag[1:]] *= 1 + ridge
    zero = xcx[diag, diag] == 0
    xcx[diag[zero], diag[zero]] = max(sminx, ridge)
    xy = cx.T @ yobs
    beta = np.linalg.solve(xcx, xy)
    yhat_obs = xobs @ beta

    if kappa is None:
        mean_y = (yobs @ omega) / nobs
        eps = yobs - yhat_obs
        with np.errstate(divide="ignore", invalid="ignore"):
            r2 = 1 - (omega @ eps**2) / (omega @ (yobs - mean_y) ** 2)
        kappa = _kappa(float(r2))

    if outout:
        # One fit per donor without the donor itself: subtract its own
        # weighted term from the cross-products, then solve all systems at once
        own = xobs[:, :, np.newaxis] * xobs[:, np.newaxis, :] * omega[:, np.newaxis, np.newaxis]
        if ridge > 0:
            own[:, diag[1:], diag[1:]] *= 1 + ridge
        xx_loo = xcx - own
        # R replaces exact zeros on the slope diagonal of each donor's matrix.
        # The intercept entry is nobs - omega_i; it is zero only when one donor
        # carries the whole bootstrap sample, where R's solve() fails. Replacing
        # it too keeps that donor's prediction finite; as the only donor with
        # positive weight it is drawn anyway.
        d = xx_loo[:, diag, diag]
        xx_loo[:, diag, diag] = np.where(d == 0, max(sminx, ridge), d)
        xy_loo = xy - xobs * yobs[:, np.newaxis] * omega[:, np.newaxis]
        beta_loo = np.linalg.solve(xx_loo, xy_loo[:, :, np.newaxis])[:, :, 0]

        # Leave-one-out predictions of each donor and, with the same fit,
        # of each recipient; dist[i, j] compares donor i with recipient j
        yhat_don = np.sum(xobs * beta_loo, axis=1)
        yhat_rec = xmis @ beta_loo.T
        dist = yhat_don[:, np.newaxis] - yhat_rec.T
    else:
        yhat_mis = xmis @ beta
        dist = yhat_obs[:, np.newaxis] - yhat_mis[np.newaxis, :]

    # Distances to probabilities, kept inside minmax's bounds as in R
    with np.errstate(divide="ignore", over="ignore"):
        delta = minmax(1 / np.abs(dist) ** kappa)
    probs = delta * omega[:, np.newaxis]
    probs = probs / minmax(probs.sum(axis=0))
    # Donors at distance 0 hit the upper bound, and so can the column sums;
    # such columns then sum to more than 1. R's sample() rescales them.
    probs /= probs.sum(axis=0)
    return probs, kappa


def _kappa(r2: float) -> float:
    """kappa from the weighted R² of the bootstrap fit, as R computes it.

    ``min((50 r2 / (1 - r2))^(3/8), 100)``: 100 for a perfect fit, and 3 (the
    value of Siddique and Belin) where R gets NaN, that is when R² is
    undefined or negative.
    """
    if r2 >= 1:
        return 100.0
    ratio = 50 * r2 / (1 - r2)
    if not ratio >= 0:
        return 3.0
    return min(ratio ** (3 / 8), 100.0)


def _draw_donors(probs: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """One donor index per column of `probs`, drawn with those probabilities.

    Each column is normalised by its sum and compared with one uniform number,
    as ``Generator.choice`` does for a single draw.
    """
    if not np.isfinite(probs).all():
        raise ValueError("midas: the donor probabilities are not finite (R's sample() fails here too).")
    cdf = np.cumsum(probs, axis=0)
    cdf /= cdf[-1]
    u = rng.random(probs.shape[1])
    return np.sum(cdf <= u, axis=0)


def _predictor_matrix(x: Any) -> np.ndarray:
    """`x` as a float array of shape (n_samples, n_features); it must be complete.

    A DataFrame with non-numeric columns is dummy-coded by
    :func:`imputation.design.obtain_design`, as MICE does.
    """
    if isinstance(x, pd.DataFrame):
        if not x.select_dtypes(exclude=[np.number]).columns.empty:
            x = obtain_design(x, list(x.columns))
        missing_cols = x.columns[x.isna().any()].tolist()
        if missing_cols:
            raise ValueError(
                f"Predictors must not contain missing values for midas. Columns with missing values: {missing_cols}"
            )
        x = x.to_numpy(dtype=float)
    else:
        try:
            x = np.asarray(x, dtype=float)
        except (TypeError, ValueError):
            raise ValueError("Predictors must be numeric for midas. Could not convert 'x' to numeric array.") from None
        if np.isnan(x).any():
            raise ValueError("Predictors must not contain missing values for midas.")
    if x.ndim == 1:
        x = x.reshape(-1, 1)
    return x


def _numeric_target(y: Any, id_obs: np.ndarray) -> np.ndarray:
    """The observed values of `y` as floats; a boolean target becomes 0/1.

    midas predicts the target by linear regression, so like R's midastouch it
    needs a numeric target. Categorical and text targets raise ``ValueError``.
    """
    observed = np.asarray(y)[id_obs]
    if isinstance(getattr(y, "dtype", None), pd.CategoricalDtype):
        kind = "categorical"
    else:
        kind = pd.api.types.infer_dtype(observed, skipna=False)
    if kind not in ("floating", "integer", "mixed-integer-float", "boolean"):
        raise ValueError(
            f"Target y must be numeric for midas, got {kind} values. midas (R's midastouch) "
            "predicts the target by linear regression; impute categorical or text columns "
            "with pmm, cart, rf or sample."
        )
    return observed.astype(float)
