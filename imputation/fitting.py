"""Model fits behind the categorical imputation methods, as R's ``glm.fit``, ``optim`` and ``nnet::multinom``.

Numerics only: arrays in, arrays out, no pandas. Each function repeats the
steps of its R counterpart, so that the imputations of the methods built on
it can match R's to many digits when they are given R's random numbers:

- :func:`logit_fit`: a logistic regression as R's ``stats::glm.fit`` fits it
  for the binomial and quasibinomial families with the logit link
  (``logreg``, ``logreg.boot``);
- :func:`vmmin`: the variable-metric (BFGS) minimiser of R's
  ``optim(method = "BFGS")``, which ``nnet`` calls too;
- :func:`multinom_fit`: a multinomial logistic regression as
  ``nnet::multinom`` fits it with :func:`vmmin` (``polyreg``).
"""

from __future__ import annotations

from typing import Callable, NamedTuple

import numpy as np
from numpy.typing import ArrayLike
from scipy.linalg import solve_triangular

from .sampler import _aliased_columns

# R's DBL_EPSILON and the bounds of the linear predictor beyond which the
# logit link of R's family.c clamps the mean (THRESH and MTHRESH)
_EPS = np.finfo(float).eps
_ETA_BOUND = 30.0

# Constants of the line search of R's vmmin (src/appl/optim.c): the factor
# that shortens a rejected step, the fraction of the predicted decrease a
# step must achieve, and the offset of the test whether a step still moves
# the parameters (stepredn, acctol and reltest)
_STEP_REDUCTION = 0.2
_ACCEPTANCE = 1e-4
_REL_TEST = 10.0

# nnet's logistic unit returns exactly 0 below this activation and exactly 1
# above its negative; the entropy of a two-class fit adds _NNET_EPS inside
# its logarithms, and a softmax probability of exactly 0 costs
# _NNET_ZERO_PROB_COST times the case weight (sigmoid, E and fpass in
# nnet/src/nnet.c)
_SIGMOID_CUT = 15.0
_NNET_EPS = 1e-80
_NNET_ZERO_PROB_COST = 1000.0


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


class Minimum(NamedTuple):
    """The result of :func:`vmmin`, as R's ``optim(method = "BFGS")`` reports it.

    Attributes
    ----------
    par : np.ndarray of shape (n_parameters,)
        The parameters at the end of the search, R's ``par``.
    value : float
        The function value R reports, ``value``: that of the last accepted
        point (see the Notes of :func:`vmmin` for a stop on a step that no
        longer moves the parameters).
    converged : bool
        False if the search stopped at the iteration limit, R's
        ``convergence == 0``.
    fn_count : int
        Number of evaluations of the function, R's ``counts["function"]``.
    gr_count : int
        Number of evaluations of the gradient, R's ``counts["gradient"]``;
        also the number of iterations.
    """

    par: np.ndarray
    value: float
    converged: bool
    fn_count: int
    gr_count: int


def vmmin(
    fn: Callable[[np.ndarray], float],
    gr: Callable[[np.ndarray], np.ndarray],
    par: ArrayLike,
    maxit: int = 100,
    abstol: float = -np.inf,
    reltol: float = float(np.sqrt(_EPS)),
) -> Minimum:
    """
    Minimise a function by the variable-metric (BFGS) method of R's ``optim``.

    Nash's variable-metric method (Nash 1990, algorithm 21) as R's C
    function ``vmmin`` runs it for ``optim(par, fn, gr, method = "BFGS")``
    and for ``nnet``, with the same steps, constants and counts:

    1. Start with the identity as the approximate inverse Hessian ``B`` and
       search along ``t = -B g`` (``g`` the last gradient evaluated).
    2. Line search: try the step lengths 1, 0.2, 0.04, ... until the point
       ``par + step t`` has a finite value that is at most
       ``f + 1e-4 step g't``, or until the step no longer moves any
       parameter (``10 + par_i == 10 + new_i`` for all ``i``).
    3. Progress is an accepted point whose value is above `abstol` and
       differs from the previous one by more than
       ``reltol (|f| + reltol)``. Then the gradient is evaluated there
       (one iteration) and ``B`` is updated by the BFGS formula if
       ``t'(g_new - g_old) > 0``, else reset to the identity.
    4. Without progress (also after an uphill direction, ``g't >= 0``) the
       minimisation ends if ``B`` was just reset; otherwise ``B`` is reset
       and the search repeated from the current point, along the gradient
       evaluated before. ``B`` is also reset after ``2 n`` iterations
       without a reset.
    5. Stop after `maxit` iterations, counting the first gradient as one.

    Parameters
    ----------
    fn : callable
        The function, ``fn(par) -> float``. It may be infinite or NaN at
        points the line search tries, which are then rejected, but not at
        `par`.
    gr : callable
        Its gradient, ``gr(par) -> array of shape (n_parameters,)``.
    par : array-like of shape (n_parameters,)
        Start values.
    maxit : int, default=100
        Maximal number of iterations (R's ``maxit``). If it is not positive,
        the function is evaluated at `par` and nothing else is done, as in
        R.
    abstol : float, default=-inf
        Stop when the value is at most this (R's ``abstol``).
    reltol : float, default=sqrt(machine epsilon)
        Relative tolerance on the decrease of the value (R's ``reltol``).

    Returns
    -------
    Minimum
        ``par``, ``value``, ``converged`` (False at the iteration limit) and
        the numbers of function and gradient evaluations, as R's ``optim``.

    Raises
    ------
    ValueError
        If the function is not finite at `par` (R: "initial value in
        'vmmin' is not finite").

    Notes
    -----
    The search can end on a step too short to move the parameters; then
    `par` is that point, a rounding error away from the last accepted one,
    and `value` is the last value computed there or before. R does the
    same. Reaching `maxit` is not an error; R's ``optim`` reports it as
    ``convergence = 1``.

    The parameters are those of ``nnet`` that are not masked; ``nnet``'s
    masked weights stay at their start values, which here means that
    `fn` and `gr` take only the free parameters.

    References
    ----------
    Nash, J. C. (1990). *Compact Numerical Methods for Computers: Linear
    Algebra and Function Minimisation*, 2nd edition. Adam Hilger.

    Examples
    --------
    >>> fr = lambda p: 100 * (p[1] - p[0] ** 2) ** 2 + (1 - p[0]) ** 2
    >>> grr = lambda p: np.array([-400 * p[0] * (p[1] - p[0] ** 2) - 2 * (1 - p[0]), 200 * (p[1] - p[0] ** 2)])
    >>> res = vmmin(fr, grr, [-1.2, 1.0])
    >>> np.round(res.par, 3), res.converged, res.fn_count, res.gr_count  # R: 1 1, 0, 110 43
    (array([1., 1.]), True, 110, 43)
    """
    b = np.array(par, dtype=float).ravel()
    n = b.size
    if maxit <= 0:
        return Minimum(par=b, value=float(fn(b)), converged=True, fn_count=0, gr_count=0)
    f = float(fn(b))
    if not np.isfinite(f):
        raise ValueError(
            "vmmin: the function is not finite at the start values (R: initial value in 'vmmin' is not finite)."
        )
    f_min = f
    fn_count = gr_count = 1
    g = np.asarray(gr(b), dtype=float)
    last_reset = gr_count  # R's ilast: the gradient count when B was last reset
    inverse_hessian = np.eye(n)
    lower = np.tril_indices(n, -1)
    while True:
        if last_reset == gr_count:
            inverse_hessian = np.eye(n)
        start, g_start = b, g
        t = -(inverse_hessian @ g)
        slope = float(t @ g)
        if slope < 0.0:  # downhill
            step = 1.0
            accepted = False
            while True:
                b = start + step * t
                # R's count: the parameters the step does not move; n also marks "no progress" below
                unmoved = int(np.count_nonzero(_REL_TEST + start == _REL_TEST + b))
                if unmoved < n:
                    f = float(fn(b))
                    fn_count += 1
                    accepted = bool(np.isfinite(f) and f <= f_min + slope * step * _ACCEPTANCE)
                    if not accepted:
                        step *= _STEP_REDUCTION
                if unmoved == n or accepted:
                    break
            # no progress if the value is small or barely changed (f may come from an earlier trial, as in R)
            if not (f > abstol and abs(f - f_min) > reltol * (abs(f_min) + reltol)):
                unmoved = n
                f_min = f
            if unmoved < n:  # progress
                f_min = f
                g = np.asarray(gr(b), dtype=float)
                gr_count += 1
                t = step * t
                c = g - g_start
                d1 = float(t @ c)
                if d1 > 0:
                    bc = inverse_hessian @ c
                    d2 = 1.0 + float(bc @ c) / d1
                    inverse_hessian = inverse_hessian + (np.outer(d2 * t, t) - np.outer(bc, t) - np.outer(t, bc)) / d1
                    # R updates and uses the lower triangle only
                    inverse_hessian.T[lower] = inverse_hessian[lower]
                else:
                    last_reset = gr_count
            elif last_reset < gr_count:  # no progress: reset, unless just reset
                unmoved = 0
                last_reset = gr_count
        else:  # uphill: reset, unless just reset
            if last_reset == gr_count:
                unmoved = n
            else:
                unmoved = 0
                last_reset = gr_count
        if gr_count >= maxit:
            break
        if gr_count - last_reset > 2 * n:  # periodic restart
            last_reset = gr_count
        if unmoved == n and last_reset == gr_count:
            break
    return Minimum(par=b, value=f_min, converged=gr_count < maxit, fn_count=fn_count, gr_count=gr_count)


class MultinomFit(NamedTuple):
    """The result of :func:`multinom_fit`.

    Attributes
    ----------
    coef : np.ndarray of shape (n_fitted - 1, n_features)
        Row ``k`` holds the coefficients of fitted class ``k + 1`` relative
        to the first fitted class (whose coefficients are 0): its log odds
        against the first class are ``x @ coef[k]``. R's ``coef(fit)``,
        which is a vector for two classes.
    classes : np.ndarray of int
        The classes fitted: those of ``0, ..., n_classes - 1`` that have at
        least one row, in increasing order (R's ``fit$lev``).
    n_classes : int
        The number of classes of the target, fitted or not.
    value : float
        The minimised objective, R's ``fit$value`` (half its deviance): the
        weighted negative log-likelihood with nnet's conventions (see
        :func:`multinom_fit`).
    converged : bool
        False if the minimiser stopped at the iteration limit, R's
        ``fit$convergence == 0``.
    fn_count : int
        Evaluations of the objective by :func:`vmmin` (R does not report it).
    gr_count : int
        Evaluations of its gradient, the number of iterations.
    logistic : bool
        True if two classes were fitted with nnet's logistic unit, False
        for the softmax of more classes.
    """

    coef: np.ndarray
    classes: np.ndarray
    n_classes: int
    value: float
    converged: bool
    fn_count: int
    gr_count: int
    logistic: bool

    def predict(self, x: ArrayLike) -> np.ndarray:
        """Probabilities of the classes for the rows of `x`, as R's ``predict(fit, newdata, type = "probs")``.

        Parameters
        ----------
        x : array-like of shape (n_rows, n_features)
            Rows of the design matrix, as passed to :func:`multinom_fit`.

        Returns
        -------
        np.ndarray of shape (n_rows, n_classes)
            One row of probabilities per row of `x`; 0 for the classes that
            were not fitted. With the logistic unit the fitted classes get
            ``1 - p`` and ``p``, where ``p`` is the output of the unit
            (exactly 0 or 1 beyond ``|x @ coef| = 15``), as mice builds them.
        """
        x = np.asarray(x, dtype=float)
        if x.ndim == 1:
            x = x.reshape(-1, 1)
        out = np.zeros((x.shape[0], self.n_classes))
        if self.logistic:
            p = _nnet_sigmoid(x @ self.coef[0])
            out[:, self.classes[0]] = 1 - p
            out[:, self.classes[1]] = p
        else:
            out[:, self.classes] = _softmax(x, self.coef)
        return out


def multinom_fit(
    x: ArrayLike,
    y: ArrayLike,
    weights: ArrayLike | None = None,
    n_classes: int | None = None,
    maxit: int = 100,
    max_weights: int = 1500,
    abstol: float = 1e-4,
    reltol: float = 1e-8,
) -> MultinomFit:
    """
    Multinomial logistic regression as ``nnet::multinom`` fits it.

    Reproduces ``nnet::multinom(y ~ x, weights = weights, maxit = maxit,
    MaxNWts = max_weights)`` as R's mice calls it, with `x` the model
    matrix:

    - Classes without a row are left out (nnet drops them with a warning);
      at least two must remain.
    - More than two classes: a softmax over the classes, the first class
      with all coefficients fixed at 0. The objective is
      ``sum_i w_i (-log p_i)`` with ``p_i`` the probability of the class of
      row ``i``; a probability of exactly 0 adds ``1000 w_i`` instead.
    - Two classes: nnet's single logistic unit for the second class, whose
      output is exactly 0 below an activation of -15 and exactly 1 above
      15. The objective is ``sum_i w_i (-log(p_i + 1e-80))`` with ``p_i``
      the output for the class of row ``i`` (``1 - output`` for the first).
    - All coefficients start at 0 (nnet's ``rang = 0``: no random numbers)
      and are fitted by :func:`vmmin` with `abstol`, `reltol` and `maxit`.
      Reaching `maxit` is not an error.

    The weights nnet counts are ``n_fitted (n_features + 1)`` for the
    softmax and ``n_features + 1`` for the logistic unit, its bias weights
    included (they stay at 0).

    Parameters
    ----------
    x : array-like of shape (n_samples, n_features)
        Design matrix, with the intercept column (R's model matrix).
    y : array-like of int of shape (n_samples,)
        The class of each row, from ``0, ..., n_classes - 1``.
    weights : array-like of shape (n_samples,), optional
        Case weights, non-negative; all 1 if None.
    n_classes : int, optional
        The number of classes; ``max(y) + 1`` if None. Classes without a
        row get probability 0 in :meth:`MultinomFit.predict`.
    maxit : int, default=100
        Maximal number of iterations of :func:`vmmin` (nnet's ``maxit``,
        mice's ``nnet.maxit``).
    max_weights : int, default=1500
        Maximal number of weights (nnet's ``MaxNWts``; mice passes 1500).
    abstol : float, default=1e-4
        Stop when the objective is at most this (nnet's ``abstol``).
    reltol : float, default=1e-8
        Relative tolerance on the decrease of the objective (nnet's
        ``reltol``).

    Returns
    -------
    MultinomFit
        The coefficients, the classes fitted, the objective, whether the
        fit stopped before `maxit`, the evaluation counts, and
        :meth:`MultinomFit.predict` for the probabilities of new rows.

    Raises
    ------
    ValueError
        If the shapes do not match, `x` has missing values, a weight is
        negative, a class is outside ``0, ..., n_classes - 1``, fewer than
        two classes have rows (R: "need two or more classes to fit a
        multinom model"), or the model has more than `max_weights` weights
        (R: "too many (N) weights").

    Notes
    -----
    The fitted probabilities stop where nnet's tolerances stop the
    minimiser, often 1e-5 to 1e-4 away from the exact maximum of the
    likelihood; that is the point mice's imputations come from.

    Examples
    --------
    >>> x = np.c_[np.ones(6), [0.0, 1.0, 2.0, 3.0, 4.0, 5.0]]
    >>> fit = multinom_fit(x, [0, 2, 1, 0, 2, 2])
    >>> fit.coef.shape, fit.converged
    ((2, 2), True)
    >>> fit.predict(x[:2]).sum(axis=1)
    array([1., 1.])
    """
    x = np.asarray(x, dtype=float)
    if x.ndim == 1:
        x = x.reshape(-1, 1)
    codes = np.asarray(y).ravel()
    n_rows, n_features = x.shape
    w = np.ones(n_rows) if weights is None else np.asarray(weights, dtype=float).ravel()
    if codes.shape[0] != n_rows or w.shape[0] != n_rows:
        raise ValueError(f"multinom_fit: x has {n_rows} rows, y {codes.shape[0]} and the weights {w.shape[0]} entries.")
    if np.isnan(x).any():
        raise ValueError("multinom_fit: missing values in 'x'.")
    if np.any(w < 0) or np.isnan(w).any():
        raise ValueError("multinom_fit: invalid weights vector; the weights must be non-negative.")
    if n_classes is None:
        n_classes = int(codes.max()) + 1 if codes.size else 0
    if codes.size and (codes.min() < 0 or codes.max() >= n_classes or np.any(codes != np.round(codes))):
        raise ValueError(f"multinom_fit: the classes in y must be integers from 0 to {n_classes - 1}.")
    codes = codes.astype(np.intp)
    classes = np.unique(codes)
    if classes.size < 2:
        raise ValueError("multinom_fit: need two or more classes to fit a multinom model.")
    logistic = _uses_logistic_unit(classes.size)
    n_weights = (n_features + 1) * (1 if logistic else classes.size)
    if n_weights > max_weights:
        raise ValueError(
            f"multinom_fit: too many ({n_weights}) weights: {classes.size} classes and {n_features} columns "
            f"need {n_weights}, more than max_weights={max_weights} (R: 'too many ({n_weights}) weights')."
        )
    fitted = np.asarray(np.searchsorted(classes, codes))  # the position of each row's class among the fitted ones
    if logistic:
        fn, gr = _logistic_objective(x, fitted, w)
    else:
        fn, gr = _softmax_objective(x, fitted, w, classes.size)
    n_free = (classes.size - 1) * n_features
    res = vmmin(fn, gr, np.zeros(n_free), maxit=maxit, abstol=abstol, reltol=reltol)
    return MultinomFit(
        coef=res.par.reshape(classes.size - 1, n_features),
        classes=classes,
        n_classes=int(n_classes),
        value=res.value,
        converged=res.converged,
        fn_count=res.fn_count,
        gr_count=res.gr_count,
        logistic=logistic,
    )


def _uses_logistic_unit(n_fitted: int) -> bool:
    """nnet's ``multinom`` fits two classes with one logistic unit and more with a softmax."""
    return n_fitted == 2


def _nnet_sigmoid(eta: np.ndarray) -> np.ndarray:
    """nnet's logistic unit: ``1 / (1 + exp(-eta))``, exactly 0 below ``eta = -15`` and 1 above 15."""
    with np.errstate(over="ignore"):
        p = 1.0 / (1.0 + np.exp(-eta))
    return np.where(eta < -_SIGMOID_CUT, 0.0, np.where(eta > _SIGMOID_CUT, 1.0, p))


def _softmax(x: np.ndarray, coef: np.ndarray) -> np.ndarray:
    """Probabilities of the classes, the first with coefficients 0, shifted by the row maximum as nnet does."""
    eta = x @ np.vstack([np.zeros(coef.shape[1]), coef]).T
    e = np.exp(eta - eta.max(axis=1, keepdims=True))
    return e / e.sum(axis=1, keepdims=True)


def _softmax_objective(
    x: np.ndarray, fitted: np.ndarray, w: np.ndarray, n_fitted: int
) -> tuple[Callable[[np.ndarray], float], Callable[[np.ndarray], np.ndarray]]:
    """nnet's objective and gradient of the softmax, as functions of the coefficients of classes 2, 3, ...

    The gradient of each class is ``sum_i w_i (p_ik - [y_i = k]) x_i``.
    """
    n_rows, n_features = x.shape
    rows = np.arange(n_rows)
    others = (fitted[:, None] == np.arange(1, n_fitted)).astype(float)

    def probabilities(beta: np.ndarray) -> np.ndarray:
        return _softmax(x, beta.reshape(n_fitted - 1, n_features))

    def fn(beta: np.ndarray) -> float:
        p = probabilities(beta)[rows, fitted]
        with np.errstate(divide="ignore"):
            loss = np.where(p > 0, -np.log(p), _NNET_ZERO_PROB_COST)
        return float(np.sum(w * loss))

    def gr(beta: np.ndarray) -> np.ndarray:
        residual = (probabilities(beta)[:, 1:] - others) * w[:, None]
        return (residual.T @ x).ravel()

    return fn, gr


def _logistic_objective(
    x: np.ndarray, fitted: np.ndarray, w: np.ndarray
) -> tuple[Callable[[np.ndarray], float], Callable[[np.ndarray], np.ndarray]]:
    """nnet's entropy objective and gradient of the logistic unit for the second class.

    The gradient is ``sum_i w_i (p_i - [y_i = 2]) x_i``, also where the unit
    is cut off at 0 or 1, as in nnet.
    """
    second = (fitted == 1).astype(float)

    def fn(beta: np.ndarray) -> float:
        p = _nnet_sigmoid(x @ beta)
        loss = np.where(second > 0, -np.log(p + _NNET_EPS), -np.log(1 - p + _NNET_EPS))
        return float(np.sum(w * loss))

    def gr(beta: np.ndarray) -> np.ndarray:
        return ((_nnet_sigmoid(x @ beta) - second) * w) @ x

    return fn, gr


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
