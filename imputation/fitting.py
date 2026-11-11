"""Model fits behind the categorical imputation methods, as R's ``glm.fit``.

Numerics only: arrays in, arrays out, no pandas. :func:`logit_fit` fits a
logistic regression step for step as R's ``stats::glm.fit`` does for the
binomial and quasibinomial families with the logit link, so that the
imputations of ``logreg`` and ``logreg.boot`` can match R's to many digits
when they are given R's random numbers.
"""

from __future__ import annotations

from typing import NamedTuple

import numpy as np
from numpy.typing import ArrayLike
from scipy.linalg import solve_triangular

from .sampler import _aliased_columns

# R's DBL_EPSILON and the bounds of the linear predictor beyond which the
# logit link of R's family.c clamps the mean (THRESH and MTHRESH)
_EPS = np.finfo(float).eps
_ETA_BOUND = 30.0


class LogitFit(NamedTuple):
    """The result of :func:`logit_fit`.

    Attributes
    ----------
    coef : np.ndarray of shape (n_features,)
        The coefficients; NaN for aliased columns, as R's ``coef(fit)``.
    cov_unscaled : np.ndarray of shape (rank, rank)
        ``(X' W X)^-1`` of the columns that are not aliased, in their order,
        with the working weights ``W`` of the last weighted least squares:
        R's ``summary.glm(fit)$cov.unscaled``. Not scaled by a dispersion.
    iterations : int
        Number of iterations run, R's ``fit$iter``.
    converged : bool
        Whether the deviance converged within the iteration limit, R's
        ``fit$converged``.
    aliased : np.ndarray of bool of shape (n_features,)
        True for the columns that are linear combinations of the columns
        before them in the last weighted least squares.
    deviance : float
        The deviance at `coef`.
    """

    coef: np.ndarray
    cov_unscaled: np.ndarray
    iterations: int
    converged: bool
    aliased: np.ndarray
    deviance: float


def logit_fit(
    x: ArrayLike,
    y: ArrayLike,
    weights: ArrayLike | None = None,
    maxit: int = 25,
    epsilon: float = 1e-8,
) -> LogitFit:
    """
    Logistic regression by iteratively reweighted least squares, as R's ``glm.fit``.

    Reproduces ``stats::glm.fit(x, y, weights, family = binomial())`` (and
    ``quasibinomial()``, which fits the same coefficients) step for step:

    1. start from ``mu = (w y + 0.5) / (w + 1)`` and ``eta = logit(mu)``;
    2. in each iteration, regress the working response
       ``z = eta + (y - mu) / mu'(eta)`` on `x` by least squares with the
       working weights ``w mu'(eta)^2 / (mu (1 - mu))``, using a QR
       decomposition that leaves out every column whose part orthogonal to
       the columns before it is smaller than ``min(1e-7, epsilon / 1000)``
       times its norm (R's ``dqrdc2``); their coefficients are 0 meanwhile;
    3. stop when ``|dev - dev_old| / (|dev| + 0.1) < epsilon`` (converged)
       or after `maxit` iterations (not converged; no error, R warns).

    The logit link is R's: the mean is ``exp(eta) / (1 + exp(eta))`` with
    ``exp(eta)`` replaced by machine epsilon below ``eta = -30`` and by its
    inverse above 30, and its derivative is machine epsilon where
    ``|eta| > 30``. So the fitted means never reach 0 or 1, and a fit of
    separated data runs into the iteration limit or ends with large
    coefficients, as in R.

    Parameters
    ----------
    x : array-like of shape (n_samples, n_features)
        Design matrix, including the intercept column if there is one (R's
        ``glm.fit`` adds none).
    y : array-like of shape (n_samples,)
        The response, with values between 0 and 1 (a two-category target as
        0 for its first category and 1 for its second).
    weights : array-like of shape (n_samples,), optional
        Prior weights, non-negative; all 1 if None. Rows with weight 0 do
        not enter the fit (their response counts as 0, as in R).
    maxit : int, default=25
        Maximal number of iterations (R's ``glm.control(maxit)``).
    epsilon : float, default=1e-8
        Convergence tolerance on the relative change of the deviance (R's
        ``glm.control(epsilon)``).

    Returns
    -------
    LogitFit
        Coefficients (NaN for aliased columns), ``cov_unscaled`` of the
        other columns, the number of iterations, whether the fit converged,
        the aliased columns and the deviance.

    Raises
    ------
    ValueError
        If the shapes do not match, if `y` has values outside ``[0, 1]``
        (R: "y values must be 0 <= y <= 1"), if a weight is negative, if no
        weight is positive, or if `maxit` is below 1.

    Notes
    -----
    ``cov_unscaled`` comes from the last weighted least squares, whose
    weights were computed before the last update of the coefficients; that
    is what R's ``summary.glm`` returns. Recomputing it at the final
    coefficients differs in about the fifth digit. Where R's ``glm.fit``
    halves the step after a non-finite deviance, or stops at non-finite
    coefficients, nothing is done here: with the clamped logit link the
    deviance is always finite, and with finite data and the rank tolerance
    so are the coefficients.

    Examples
    --------
    >>> x = np.c_[np.ones(6), [0.0, 1.0, 2.0, 3.0, 4.0, 5.0]]
    >>> fit = logit_fit(x, [0, 0, 1, 0, 1, 1])
    >>> fit.converged, fit.iterations
    (True, 5)
    >>> [round(float(c), 4) for c in fit.coef]  # R: -3.0351, 1.2140
    [-3.0351, 1.214]
    """
    x = np.asarray(x, dtype=float)
    if x.ndim == 1:
        x = x.reshape(-1, 1)
    y = np.asarray(y, dtype=float).ravel()
    n_rows = x.shape[0]
    w = np.ones(n_rows) if weights is None else np.asarray(weights, dtype=float).ravel()
    if y.shape[0] != n_rows or w.shape[0] != n_rows:
        raise ValueError(f"logit_fit: x has {n_rows} rows, y {y.shape[0]} and the weights {w.shape[0]} entries.")
    if np.any((y < 0) | (y > 1)) or np.isnan(y).any():
        raise ValueError("logit_fit: y values must be 0 <= y <= 1.")
    if np.any(w < 0) or not np.any(w > 0):
        raise ValueError("logit_fit: the weights must be non-negative, and at least one must be positive.")
    if maxit < 1:
        raise ValueError(f"logit_fit: maxit must be at least 1, got {maxit}.")

    # R's binomial()$initialize
    y = np.where(w == 0, 0.0, y)
    mu = (w * y + 0.5) / (w + 1)
    eta = np.log(mu / (1 - mu))
    mu = _logit_linkinv(eta)
    dev_old = _deviance(y, mu, w)
    tol = min(1e-7, epsilon / 1000)
    # R drops the rows with weight 0, and those where mu'(eta) is 0, which
    # the clamped logit link never gives
    good = w > 0
    converged = False
    iterations = 0
    while iterations < maxit:
        iterations += 1
        mu_eta = _logit_mu_eta(eta[good])
        z = eta[good] + (y[good] - mu[good]) / mu_eta
        working = np.sqrt(w[good] * mu_eta**2 / (mu[good] * (1 - mu[good])))
        coef, aliased, r = _least_squares(x[good] * working[:, None], z * working, tol)
        eta = x @ coef
        mu = _logit_linkinv(eta)
        dev = _deviance(y, mu, w)
        if abs(dev - dev_old) / (0.1 + abs(dev)) < epsilon:
            converged = True
            break
        dev_old = dev

    # chol2inv of the triangular factor of the last least squares
    r_inv = solve_triangular(r, np.eye(r.shape[0]))
    coef[aliased] = np.nan
    return LogitFit(
        coef=coef,
        cov_unscaled=r_inv @ r_inv.T,
        iterations=iterations,
        converged=converged,
        aliased=aliased,
        deviance=dev,
    )


def _least_squares(x: np.ndarray, z: np.ndarray, tol: float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Least squares of `z` on `x` without the aliased columns, as R's ``Cdqrls``.

    Returns the coefficients (0 for aliased columns), the aliased columns
    (see :func:`imputation.sampler._aliased_columns`) and the upper
    triangular factor of the QR decomposition of the other columns.
    """
    aliased = _aliased_columns(x, tol)
    q, r = np.linalg.qr(x[:, ~aliased])
    coef = np.zeros(x.shape[1])
    coef[~aliased] = solve_triangular(r, q.T @ z)
    return coef, aliased, r


def _logit_linkinv(eta: np.ndarray) -> np.ndarray:
    """``exp(eta) / (1 + exp(eta))`` with R's clamp of ``exp(eta)`` beyond ``|eta| = 30``."""
    e = np.exp(np.clip(eta, -_ETA_BOUND, _ETA_BOUND))
    e = np.where(eta < -_ETA_BOUND, _EPS, np.where(eta > _ETA_BOUND, 1 / _EPS, e))
    return e / (1 + e)


def _logit_mu_eta(eta: np.ndarray) -> np.ndarray:
    """Derivative of the mean, ``exp(eta) / (1 + exp(eta))^2``; machine epsilon where ``|eta| > 30`` (R)."""
    e = np.exp(np.clip(eta, -_ETA_BOUND, _ETA_BOUND))
    return np.where(np.abs(eta) > _ETA_BOUND, _EPS, e / ((1 + e) * (1 + e)))


def _deviance(y: np.ndarray, mu: np.ndarray, w: np.ndarray) -> float:
    """Binomial deviance ``2 sum w (y log(y / mu) + (1 - y) log((1 - y) / (1 - mu)))``, with ``0 log 0 = 0``."""
    return float(np.sum(2 * w * (_y_log_y(y, mu) + _y_log_y(1 - y, 1 - mu))))


def _y_log_y(y: np.ndarray, mu: np.ndarray) -> np.ndarray:
    """``y log(y / mu)``, 0 where ``y`` is 0."""
    nonzero = y != 0
    return np.where(nonzero, y * np.log(np.where(nonzero, y, 1.0) / mu), 0.0)
