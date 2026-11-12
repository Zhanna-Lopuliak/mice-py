"""
Imputation of an ordered categorical target by the proportional-odds model (``polr``).

The port of R mice's ``mice.impute.polr``. The data are augmented with a
few weighted rows (:func:`imputation.logreg.augment`, White, Daniel and
Royston 2010), the proportional-odds logistic regression of the target on
``x`` is fitted to the observed and the added rows as R's ``MASS::polr``
fits it (:func:`imputation.fitting.polr_fit`), and each cell gets a
category drawn from the probabilities it predicts. Where ``MASS::polr``
stops (a target that is not categorical, fewer than three levels, start
values that cannot be found), the multinomial logistic regression of
``nnet::multinom`` is fitted instead (:func:`imputation.fitting.multinom_fit`),
as in R.

The order of the categories is the model's: an ordered Categorical in its
declared order, every declared level included; a text or unordered
categorical target in the sorted order of its labels, as R orders it after
the augmentation (:func:`imputation.logreg.target_categories`).
"""

from __future__ import annotations

from typing import Any, Callable, NamedTuple

import numpy as np
import pandas as pd
from numpy.typing import ArrayLike

from .design import _column_kind
from .fitting import MultinomFit, PolrError, PolrFit, multinom_fit, polr_fit
from .logreg import Categories, augment, target_categories
from .norm import _design_matrix, _masks
from .polyreg import _draw_categories

#: The event R's ``polr.to.loggedEvents = TRUE`` logs when ``polr`` falls back to ``multinom``.
FALLBACK_EVENT = "polr falls back to multinom"


class _Steps(NamedTuple):
    """What `polr` computed; the parity tests compare it with R."""

    labels: np.ndarray  # the imputations
    fit: PolrFit | MultinomFit | None  # None where no model was fitted (one category)
    prob: np.ndarray | None  # probabilities of the categories in each imputed cell, (n_imp, n_categories)
    fallback: str | None  # R's message of the condition that made polr fall back to multinom; None if it did not


def polr(
    y: ArrayLike,
    id_obs: ArrayLike,
    x: ArrayLike | pd.DataFrame,
    id_mis: ArrayLike | None = None,
    maxit: int = 100,
    max_weights: int = 1500,
    log_fallback: bool = False,
    rng: np.random.Generator | None = None,
    log_event: Callable[[str], None] | None = None,
) -> np.ndarray:
    """
    Imputation by proportional-odds logistic regression, as R mice's ``mice.impute.polr``.

    The data are augmented with weighted rows
    (:func:`imputation.logreg.augment`), the proportional-odds model
    ``P(y <= k) = plogis(zeta_k - x @ beta)`` is fitted to the observed and
    the added rows as R's ``MASS::polr`` fits it
    (:func:`imputation.fitting.polr_fit`), and each cell gets a category
    drawn with the probabilities the fit predicts for it. Where
    ``MASS::polr`` stops, the multinomial logistic regression of
    ``nnet::multinom`` is fitted instead
    (:func:`imputation.fitting.multinom_fit`), as R's ``polr`` does.

    Parameters
    ----------
    y : array-like of shape (n_samples,)
        Categorical target with at most 50 categories present: an ordered
        ``pd.Categorical`` (the case the method is for), an unordered one,
        or text, which are treated as ordered by their sorted labels;
        booleans and numbers are imputed by the fallback (see Notes).
        Inside MICE the missing entries hold the current imputations; they
        enter only through the categories they hold (see
        :func:`imputation.logreg.augment`), not the fit.
    id_obs : array-like of bool of shape (n_samples,)
        True where `y` is observed.
    x : array-like or pd.DataFrame of shape (n_samples, n_features)
        Predictors without the intercept, without missing values; zero
        columns (or None) for the model without predictors. A DataFrame
        with categorical, object, string or boolean columns is dummy-coded
        first by :func:`imputation.design.obtain_design`, as MICE does.
    id_mis : array-like of bool of shape (n_samples,), optional
        The cells to impute (R's ``wy``). If None, those where `id_obs` is
        False.
    maxit : int, default=100
        Maximal number of iterations of the fallback fit, R's
        ``nnet.maxit``; it does not apply to the proportional-odds fit,
        which stops after 100 iterations, as in R.
    max_weights : int, default=1500
        Maximal number of weights of the fallback fit, R's
        ``nnet.MaxNWts``: ``k (q + 1)`` for ``k`` categories and ``q``
        columns of ``[1, x]``, ``q + 1`` for two categories.
    log_fallback : bool, default=False
        Log the event "polr falls back to multinom" through `log_event`
        whenever the fallback is fitted (R's ``polr.to.loggedEvents``).
        Inside MICE the event becomes a row of ``MICE.logged_events``.
    rng : np.random.Generator, optional
        Random number generator. If None, a fresh generator is used.
    log_event : callable, optional
        Called with the text of an event to log; MICE supplies it.
        Without it nothing is logged.

    Returns
    -------
    np.ndarray
        The imputations of the cells in `id_mis`, in their order: labels of
        `y` (``target_categories(y, keep_unused=True).labels``), in the
        dtype of ``np.asarray(y)``.

    Raises
    ------
    ValueError
        If `y` has more than 50 categories present (R:
        "Maximum number of categories (50) exceeded"), if the fallback
        model has more than `max_weights` weights (R: "too many (N)
        weights"), if `y` has no observed value and some value is to be
        imputed, or if `x` is not numeric or has missing values.

    Notes
    -----
    Random numbers, in R's order: one uniform number per imputed cell
    (``rng.random``), drawn after the fit. A cell gets the category ``k``
    (counting from 0) for which its number exceeds ``k`` of the cumulative
    probabilities (R's ``un > cumsum``). The fits draw nothing.

    The categories, and so the order the model assumes:

    - an ordered Categorical: its declared levels in declared order, also
      those that do not occur. The model gives such a level a positive
      probability between its neighbours, so it can be imputed, as in R;
    - a text or unordered categorical target: its labels present, sorted,
      which is R's order after the augmentation (``MASS::polr`` takes the
      level order of any factor as ordinal);
    - a boolean or numeric target: its values present, sorted. R's
      ``MASS::polr`` refuses them (not a factor), so the fallback is fitted.

    The fallback is ``nnet::multinom`` on ``[1, x]`` with the weights of
    the augmentation (:func:`imputation.fitting.multinom_fit`), fitted
    where ``MASS::polr`` stops: the target is not categorical, it has fewer
    than three categories, the logistic regression behind the start values
    did not converge, or the objective is not finite at the start values
    (:class:`imputation.fitting.PolrError`). With `log_fallback`, each
    fallback logs "polr falls back to multinom", as R does with
    ``polr.to.loggedEvents = TRUE``.

    Where R fails or misbehaves, this port goes on:

    - a single category present in the observed rows (the added ones
      included): R stops (for an unordered target) or fits a degenerate
      model (for an ordered one, when its start values converge); here
      every cell gets that category, without a fit, after its uniform
      number is drawn;
    - two categories and exactly one cell to impute: R's fallback stops
      (``dim(X) must have a positive length``); here the cell is imputed;
    - a fallback on an ordered Categorical with an unused level: R stops
      (``dims ... do not match``); here the unused level gets probability 0;
    - no predictors: R's ``cbind.data.frame`` drops the zero-column ``x``,
      so both of R's fits see no response (``MASS::polr``: "response must
      be a factor", ``nnet::multinom``: "need two or more classes to fit a
      multinom model") and mice stops; here the model without predictors
      is fitted (only cut points), and where that is not possible the
      fallback on the intercept alone;
    - columns aliased in the start fit (a constant column, or collinear
      columns where the augmentation is skipped): R drops them and its
      ``predict`` stops (non-conformable arguments); here their
      coefficients are 0;
    - a uniform number above the last cumulative probability, which is 1
      up to rounding: R imputes NA; here the last category with a positive
      probability.

    Where R skips the augmentation (exactly one value missing), it keeps
    the declared order and the unused levels of an unordered factor; here
    the sorted labels present are used. For ``polr`` the order is part of
    the model, so the imputations of such a target differ from R's in
    distribution; declare the factor ordered to choose the order.

    References
    ----------
    White, I. R., Daniel, R. and Royston, P. (2010). Avoiding bias due to
    perfect prediction in multiple imputation of incomplete categorical
    variables. *Computational Statistics and Data Analysis*, 54, 2267-2275.

    Examples
    --------
    >>> grades = ["low", "mid", None, "high", "low", "mid", None, "high", "high", "low", "mid", "mid"]
    >>> y = pd.Categorical(grades, categories=["low", "mid", "high"], ordered=True)
    >>> id_obs = pd.notna(y)
    >>> x = np.array([-1.1, 0.2, 0.9, 1.4, -0.8, -0.1, -1.3, 1.6, 0.7, -0.4, 0.5, 0.1])
    >>> imputed = polr(y, id_obs, x, rng=np.random.default_rng(1))
    >>> imputed.shape  # one label per missing cell
    (2,)
    >>> bool(np.isin(imputed, ["low", "mid", "high"]).all())
    True
    """
    return _polr(y, id_obs, x, id_mis, maxit, max_weights, log_fallback, rng, log_event).labels


def _polr(
    y: Any,
    id_obs: ArrayLike,
    x: Any,
    id_mis: ArrayLike | None,
    maxit: int,
    max_weights: int,
    log_fallback: bool,
    rng: np.random.Generator | None,
    log_event: Callable[[str], None] | None,
) -> _Steps:
    """`polr`, returning its intermediate results too."""
    id_obs, id_mis = _masks(id_obs, id_mis, "polr")
    categories = _observed_categories(y, id_obs)
    design = _design_matrix(x, len(categories.codes), "polr")
    if not id_obs.any():  # and nothing to impute (_masks raises otherwise)
        return _Steps(labels=categories.labels[:0], fit=None, prob=None, fallback=None)
    if rng is None:
        rng = np.random.default_rng()
    aug = augment(categories.codes, id_obs, design[:, 1:], id_mis)
    observed = aug.y[aug.id_obs]
    present = np.unique(observed)
    n_imp = int(id_mis.sum())
    if present.size == 1:
        codes = np.full(n_imp, present[0], dtype=np.intp)
        _uniforms(rng, n_imp)  # drawn where R draws them, after the (here skipped) fit
        return _Steps(labels=categories.labels[codes], fit=None, prob=None, fallback=None)
    log = log_event if log_fallback else None
    fit, fallback = _fit_model(
        aug.x[aug.id_obs],
        observed,
        aug.weights[aug.id_obs],
        len(categories.levels),
        _is_factor(y),
        maxit,
        max_weights,
        log,
    )
    prob = _predict(fit, aug.x[aug.id_mis])
    u = _uniforms(rng, n_imp)
    codes = _draw_categories(u, prob)
    return _Steps(labels=categories.labels[codes], fit=fit, prob=prob, fallback=fallback)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _observed_categories(y: Any, id_obs: np.ndarray) -> Categories:
    """The categories of `y`, all declared levels of an ordered Categorical; every observed entry must have one."""
    categories = target_categories(y, keep_unused=True)
    if np.any(categories.codes[id_obs] < 0):
        raise ValueError("polr: y is missing in entries that id_obs marks as observed.")
    return categories


def _is_factor(y: Any) -> bool:
    """True if R's ``MASS::polr`` would see a factor: a Categorical or text, not booleans or numbers.

    Text counts as an unordered factor, as everywhere in this package
    (:func:`imputation.design._column_kind`).
    """
    if isinstance(getattr(y, "dtype", None), pd.CategoricalDtype):
        return True
    return _column_kind(pd.Series(np.asarray(y))) in ("factor", "ordered")


def _fit_model(
    x: np.ndarray,
    codes: np.ndarray,
    weights: np.ndarray,
    n_levels: int,
    factor: bool,
    maxit: int,
    max_weights: int,
    log: Callable[[str], None] | None,
) -> tuple[PolrFit | MultinomFit, str | None]:
    """R's ``try(MASS::polr(...))``, then ``nnet::multinom`` on ``[1, x]`` if it failed.

    Returns the fit and the condition (R's message) that made ``polr``
    fail, None if it did not. `log`, if given, receives the fallback event
    before the fallback is fitted, as R logs it.
    """
    try:
        return _fit_polr(x, codes, weights, n_levels, factor), None
    except PolrError as err:
        if log is not None:
            log(FALLBACK_EVENT)
        model = np.c_[np.ones(len(codes)), x]
        fit = multinom_fit(model, codes, weights=weights, n_classes=n_levels, maxit=maxit, max_weights=max_weights)
        return fit, err.condition


def _fit_polr(x: np.ndarray, codes: np.ndarray, weights: np.ndarray, n_levels: int, factor: bool) -> PolrFit:
    """``MASS::polr(y ~ x, weights)``; :class:`imputation.fitting.PolrError` where R's polr stops."""
    if not factor:
        raise PolrError(PolrError.NOT_A_FACTOR, "the target is boolean or numeric")
    return polr_fit(x, codes, weights=weights, n_levels=n_levels)


def _predict(fit: PolrFit | MultinomFit, x: np.ndarray) -> np.ndarray:
    """Probabilities of every category in the rows of `x` (predictors without intercept)."""
    if isinstance(fit, PolrFit):
        return fit.predict(x)
    return fit.predict(np.c_[np.ones(len(x)), x])


def _uniforms(rng: np.random.Generator, n: int) -> np.ndarray:
    """One uniform number per imputed cell, R's ``runif(sum(wy))``.

    R draws them after the fit (``polyreg`` draws them before).
    """
    return rng.random(n)
