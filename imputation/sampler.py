import logging

import numpy as np
from scipy.linalg import lapack

logger = logging.getLogger(__name__)

#: Relative tolerance below which a column counts as a linear combination of
#: the columns before it (the tolerance of R's ``lm.fit``).
ALIAS_TOL = 1e-7

_COLLINEAR_MESSAGE = (
    "estimice: the predictors are (nearly) multi-collinear; a ridge penalty was "
    "added to invert the cross-product matrix, so the coefficient draws can be "
    "unstable. Does the data contain duplicated columns, linear transformations "
    "of other columns, or constant columns?"
)


def sym(x):
    """
        Ensures the input square matrix is symmetric by averaging it with its transpose.

        Parameters
        ----------
        x : np.ndarray
            A square numpy matrix.

        Returns
        -------
        np.ndarray
            A symmetric matrix computed as (x + x.T) / 2.

        """
    return (x + x.T) / 2
def norm_draw(y, ry, x, rank_adjust=True, rng=None, ls_meth="qr", ridge=1e-5):
    """
        Bayesian linear regression draw of regression coefficients and residual variance,
        based on the least squares parameters from `estimice()`.

        This function replicates the `mice.impute.norm.draw()` algorithm from the R mice package,
        as described in Rubin (1987, p. 167).

        Parameters
        ----------
        y : np.ndarray
            Numeric vector of length n, containing the variable to be imputed.
        ry : np.ndarray of bool
            Boolean mask vector of length n, where True indicates observed values of `y` and False indicates missing values.
        x : np.ndarray
            Numeric design matrix of shape (n, p) with predictors for `y`. Must have no missing values.
        rank_adjust : bool, optional
            If True, replaces any NaN coefficients with zeros. NaN coefficients
            belong to columns of `x` that are linear combinations of earlier
            columns (only with `ls_meth="qr"`). Default is True.
        rng : np.random.Generator, optional
            Random number generator for reproducibility. If None, a fresh generator is used.
        ls_meth : str, optional
            Least squares method passed to `estimice()`: "qr" (default), "ridge" or "svd".
        ridge : float, optional
            Ridge penalty passed to `estimice()`: the penalty of `ls_meth="ridge"`
            and of the fallback used when the cross-product matrix cannot be
            inverted. Default is 1e-5.

        Returns
        -------
        dict
            Dictionary containing:
            - 'coef': Least squares coefficient estimates (numpy array).
            - 'beta': Bayesian drawn regression coefficients (numpy array).
            - 'sigma': Drawn residual standard deviation (float).
            - 'estimation': Least squares method used (string).

        Notes
        -----
        The random numbers are drawn in the order R draws them: first one
        chi-square variate with ``df`` degrees of freedom (``rng.chisquare``),
        then ``p`` standard normal variates (``rng.standard_normal``). With
        ``L`` the lower Cholesky factor of the covariance factor ``v`` from
        `estimice()`::

            sigma = sqrt(sum(r ** 2) / chisq)
            beta = coef + (L @ z) * sigma

        so ``beta`` has covariance ``sigma ** 2 * v``.

        References
        ----------
        Rubin, D. B. (1987). Multiple Imputation for Nonresponse in Surveys.
        Wiley, p. 167.

        Examples
        --------
        >>> import numpy as np
        >>> y = np.array([1.0, 2.0, np.nan, 4.0])
        >>> ry = ~np.isnan(y)
        >>> x = np.array([[1, 0], [1, 1], [1, 2], [1, 3]])
        >>> result = norm_draw(y, ry, x, ls_meth='qr')
        >>> print(result['beta'])
        """
    if rng is None:
        rng = np.random.default_rng()
    p = estimice(x[ry, :], y[ry], ls_meth=ls_meth, ridge=ridge)
    # one chi-square variate, then one standard normal per coefficient (R's order)
    sigma_star = float(np.sqrt(np.sum(p["r"] ** 2) / rng.chisquare(p["df"])))
    z = rng.standard_normal(x.shape[1])
    # np.linalg.cholesky returns the lower factor L with L @ L.T = v
    chol = np.linalg.cholesky(sym(p["v"]))
    beta_star = p["c"] + (chol @ z) * sigma_star
    parm = {
        "coef": p["c"],
        "beta": beta_star,
        "sigma": sigma_star,
        "estimation": p["ls_meth"]
    }
    #Replaces NaN with 0 if rank_adjust = True
    if np.any(np.isnan(parm["coef"])) and rank_adjust:
        parm["coef"] = np.nan_to_num(parm["coef"], nan=0.0)
        parm["beta"] = np.nan_to_num(parm["beta"], nan=0.0)
    return parm
def estimice(x, y, ls_meth="qr", ridge=1e-5):
    """
        Computes least squares estimates, residuals, variance-covariance matrix,
        and degrees of freedom using different methods: ridge regression, QR decomposition, or Singular Value Decomposition.

        Parameters
        ----------
        x : np.ndarray
            Numeric design matrix with shape (n_samples, n_predictors). Must not contain missing values.
        y : np.ndarray
            Numeric vector of responses to be imputed, with possible missing values.
        ls_meth : str, optional
            Least squares method to use. Options are:
            - "qr": QR decomposition (default)
            - "ridge": Ridge regression
            - "svd": Singular Value Decomposition
        ridge : float, optional
            Ridge penalty size for ridge regression and for the fallback
            described below. Default is 1e-5, balancing stability and bias.

        Returns
        -------
        dict
            Dictionary containing:
            - 'c': Least squares coefficient estimates (numpy array).
            - 'r': Residuals (numpy array). For "svd" these are fitted minus
              observed values, as in R.
            - 'v': Variance-covariance matrix of coefficients (numpy array),
              up to the factor sigma^2.
            - 'df': Degrees of freedom (int).
            - 'ls_meth': Method used (str).

        Notes
        -----
        Follows ``estimice()`` of R mice 3.19.0.

        Rank deficiency. With "qr", a column whose part orthogonal to the
        columns before it is smaller than ``1e-7`` times its norm (R's
        ``lm.fit`` rule) is aliased: it is left out of the fit and its
        coefficient is NaN, which `norm_draw()` sets to 0. With "svd", singular
        values below ``1e-7`` times the largest one are treated as 0 (the
        minimum-norm solution). The fitted values do not depend on the aliased
        columns.

        Covariance. With "qr" and "svd", ``v`` is the inverse of ``X'X`` for
        all columns. When that matrix is singular to working precision (R's
        ``solve()`` rule: exactly singular, or reciprocal condition number
        below machine epsilon), the ridge penalty ``ridge * diag(X'X)`` is
        added before inverting and a warning is logged, as R does. A column of
        zeros, which R cannot handle, gets the penalty ``ridge``. ``v`` is
        ordered like the columns of `x`; R orders it like its pivoted QR,
        which differs from the column order when a column is aliased.

        Examples
        --------
        >>> import numpy as np
        >>> x = np.array([[1, 2], [3, 4], [5, 6]])
        >>> y = np.array([7, np.nan, 9])
        >>> # Assuming you handle missing y externally, e.g. ry = ~np.isnan(y)
        >>> result = estimice(x[~np.isnan(y)], y[~np.isnan(y)], ls_meth="qr")
        >>> print(result['c'])
        [-6.   6.5]
        """
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    if x.ndim == 1:
        x = x.reshape(-1, 1)
    #degrees of freedom = length of y - number of columns of x, mininmum df = 1
    #c = coefficients, f = fitted values, r = residuals
    df = max(len(y) - x.shape[1], 1)
    #QR Decomposition
    if ls_meth == "qr":
        q, rmat = np.linalg.qr(x)
        aliased = _aliased_columns(x)
        if aliased.any():
            # least squares on the columns that are not aliased; NaN for the rest
            c = np.full(x.shape[1], np.nan)
            if not aliased.all():
                q_kept, r_kept = np.linalg.qr(x[:, ~aliased])
                c[~aliased] = np.linalg.solve(r_kept, q_kept.T @ y)
            f = x[:, ~aliased] @ c[~aliased]
        else:
            c = np.linalg.solve(rmat, q.T @ y)
            f = x @ c
        r = y - f
        v = _inverse_or_ridge(rmat.T @ rmat, ridge)

        return {
            "c": c.flatten(),  # transpose to match shape
            "r": r.flatten(),
            "v": v,
            "df": df,
            "ls_meth": ls_meth
        }
    #Ridge Regression
    elif ls_meth == "ridge":
        xx = x.T @ x
        v = np.linalg.solve(xx + np.diag(_ridge_penalty(xx, ridge)), np.eye(xx.shape[1]))
        c = y.T @ x @ v
        r = y - x @ c.T
        return {
            "c": c.flatten(),
            "r": r.flatten(),
            "v": v,
            "df": df,
            "ls_meth": ls_meth
        }
    #Singular Value Decomposition
    elif ls_meth == "svd":
        svd = np.linalg.svd(x, full_matrices=False)
        # x = U diag(S) V' with V = Vh.T; singular values near 0 contribute nothing
        kept = svd.S > ALIAS_TOL * svd.S.max(initial=0.0)
        uty = (svd.U).T @ y
        c = svd.Vh.T @ np.divide(uty, svd.S, out=np.zeros_like(uty), where=kept)
        f = x @ c
        r = f - y
        v = _inverse_or_ridge((svd.Vh.T * svd.S ** 2) @ svd.Vh, ridge)
        return {
            "c": c.flatten(),
            "r": r.flatten(),
            "v": v,
            "df": df,
            "ls_meth": ls_meth
        }
    raise ValueError(f"unknown ls_meth {ls_meth!r}; use 'qr', 'ridge' or 'svd'")


def _aliased_columns(x, tol=ALIAS_TOL):
    """Mark the columns of `x` that are linear combinations of earlier columns.

    The columns are processed from left to right with Householder
    reflections. A column is aliased when the norm of its part orthogonal to
    the columns kept so far is below ``tol`` times its own norm (a column of
    zeros is always aliased). Aliased columns are skipped, so a later column
    can still be kept, and at most ``n_rows`` columns are kept. This is the
    rule of the limited column pivoting in R's ``lm.fit``.

    Returns
    -------
    np.ndarray of bool
        True for every aliased column.
    """
    n, p = x.shape
    work = np.array(x, dtype=float)
    norms = np.linalg.norm(work, axis=0)
    reference = np.where(norms > 0, norms, 1.0)
    aliased = np.zeros(p, dtype=bool)
    kept = 0
    for j in range(p):
        if kept == n:
            aliased[j] = True
            continue
        col = work[kept:, j]
        size = np.linalg.norm(col)
        if size < tol * reference[j]:
            aliased[j] = True
            continue
        # reflect col onto the first axis; apply the reflection to later columns
        u = col.copy()
        u[0] += np.copysign(size, col[0])
        u /= np.linalg.norm(u)
        rest = work[kept:, j + 1:]
        rest -= 2.0 * np.outer(u, u @ rest)
        kept += 1
    return aliased


def _inverse(a):
    """Inverse of `a`, or None where R's ``solve(a)`` would stop.

    R refuses a matrix that is exactly singular or whose reciprocal condition
    number (1-norm, LAPACK ``dgecon``) is below machine epsilon.
    """
    lu, piv, info = lapack.dgetrf(a)
    if info != 0:
        return None
    rcond, info = lapack.dgecon(lu, np.abs(a).sum(axis=0).max(), norm="1")
    if info != 0 or not rcond >= np.finfo(float).eps:
        return None
    inv, info = lapack.dgetrs(lu, piv, np.eye(a.shape[0]))
    return inv if info == 0 else None


def _ridge_penalty(xtx, ridge):
    """Diagonal ridge penalty ``ridge * diag(xtx)``; a zero diagonal entry gets ``ridge``."""
    d = np.diag(xtx)
    return ridge * np.where(d > 0, d, 1.0)


def _inverse_or_ridge(xtx, ridge):
    """Inverse of `xtx`; if it is singular to working precision, the inverse
    of ``xtx + diag(ridge * diag(xtx))`` (R's fallback), with a warning.

    The fallback is computed on the equilibrated matrix ``D^-1 xtx D^-1``
    with ``D = sqrt(diag(xtx))``, which has a unit diagonal, so the penalty
    becomes ``ridge * I``: the same inverse, but it cannot lose the penalty
    to underflow when a column is tiny.
    """
    v = _inverse(xtx)
    if v is not None:
        return v
    logger.warning(_COLLINEAR_MESSAGE)
    scale = np.sqrt(_ridge_penalty(xtx, 1.0))
    outer = np.outer(scale, scale)
    unit = xtx / outer + ridge * np.eye(xtx.shape[0])
    return np.linalg.solve(unit, np.eye(xtx.shape[0])) / outer
